"""
eval_component_miss.py -- component-level (object-level) miss analysis for change detection.

For each checkpoint the model is rebuilt from the config embedded in the checkpoint and run
single-pass (no TTA) on the test split at a fixed threshold. Reported per model:
  * pixel metrics (must reproduce the paper tables),
  * ground-truth change components (8-connected): complete misses (zero predicted pixels),
    component recall at a coverage threshold, share of changed area lying in completely
    missed components, all stratified by component area,
  * predicted components with no ground-truth overlap (false-alarm components).
Models are compared pairwise on identical ground-truth components: exact McNemar test on the
per-component complete-miss outcome, plus an image-level bootstrap CI for the difference in
complete-miss rate.

Usage (repo root):
    python eval_component_miss.py --gpu 0 \
        --ckpt diff=checkpoints/abl_whu_diff_s42_seed42/whu_cd/best.pth \
        --ckpt btot=checkpoints/v6btot_whu_s42_seed42/whu_cd/best.pth \
        --out_dir analysis_out/component_miss/whu
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

AREA_BINS = [(0, 64), (64, 256), (256, 1024), (1024, None)]
EIGHT_CONN = np.ones((3, 3), dtype=bool)


def load_model(ckpt_path, gpu, batch_size):
    from train_dinobcd import Trainer

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    cfg.setdefault('evaluation', {})
    cfg['evaluation']['use_tta'] = False
    cfg['evaluation']['visualize'] = False
    cfg.setdefault('device', {})['gpu_ids'] = [gpu]
    cfg['data']['batch_size'] = batch_size
    cfg.setdefault('experiment', {})['name'] = 'component_eval'
    cfg.setdefault('logging', {})
    cfg['logging']['log_dir'] = 'logs/component_eval'
    cfg['logging']['save_dir'] = 'checkpoints/component_eval'

    trainer = Trainer(cfg)
    missing, unexpected = trainer.model.load_state_dict(ck['model_state_dict'], strict=False)
    if missing:
        raise RuntimeError(f'{ckpt_path}: {len(missing)} missing keys, e.g. {missing[:5]}')
    if unexpected:
        print(f'[WARN] {ckpt_path}: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}')
    epoch = ck.get('epoch')
    del ck
    trainer.model.eval()
    return trainer, cfg, epoch


def build_test_loader(cfg, batch_size, workers):
    from dinobcd.datasets import ChangeDetectionDataset

    dc = cfg['data']
    ds = ChangeDetectionDataset(
        dataset_name=dc['dataset'],
        data_root=dc['data_root'],
        list_file=dc.get('test_list', 'test.txt'),
        image_size=dc['image_size'],
        is_train=False,
        augmentation=None,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return ds, loader


def analyse_image(label, pred):
    lab, n = ndimage.label(label, structure=EIGHT_CONN)
    flat = lab.ravel()
    if n:
        area = np.bincount(flat, minlength=n + 1)[1:]
        hit = np.bincount(flat, weights=pred.ravel().astype(np.float64), minlength=n + 1)[1:]
    else:
        area, hit = np.zeros(0, np.int64), np.zeros(0)

    plab, m = ndimage.label(pred, structure=EIGHT_CONN)
    pflat = plab.ravel()
    if m:
        parea = np.bincount(pflat, minlength=m + 1)[1:]
        pov = np.bincount(pflat, weights=label.ravel().astype(np.float64), minlength=m + 1)[1:]
    else:
        parea, pov = np.zeros(0, np.int64), np.zeros(0)
    return area, hit, parea, pov


def run_model(ckpt, args):
    trainer, cfg, epoch = load_model(ckpt, args.gpu, args.batch_size)
    ds, loader = build_test_loader(cfg, args.batch_size, args.workers)

    gt_img, gt_area, gt_hit, pr_img, pr_area, pr_ov = [], [], [], [], [], []
    tp = fp = fn = tn = 0
    probs = []
    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            img1 = batch['img1'].to(trainer.device, non_blocking=True)
            img2 = batch['img2'].to(trainer.device, non_blocking=True)
            pred_t, prob_t = trainer.model.predict(img1, img2, threshold=args.threshold)
            pred = pred_t.cpu().numpy().astype(np.uint8)
            label = (batch['label'].numpy() == 1).astype(np.uint8)
            if args.save_probs:
                probs.append(np.round(prob_t.float().cpu().numpy() * 255).astype(np.uint8))

            tp += int(((pred == 1) & (label == 1)).sum())
            fp += int(((pred == 1) & (label == 0)).sum())
            fn += int(((pred == 0) & (label == 1)).sum())
            tn += int(((pred == 0) & (label == 0)).sum())

            for b in range(pred.shape[0]):
                area, hit, parea, pov = analyse_image(label[b], pred[b])
                gt_img.append(np.full(area.size, idx, np.int32))
                gt_area.append(area)
                gt_hit.append(hit)
                pr_img.append(np.full(parea.size, idx, np.int32))
                pr_area.append(parea)
                pr_ov.append(pov)
                idx += 1
    print(f'  inference + components: {idx} images in {time.time() - t0:.0f}s')

    rec = {
        'gt_img': np.concatenate(gt_img), 'gt_area': np.concatenate(gt_area).astype(np.int64),
        'gt_hit': np.concatenate(gt_hit), 'pr_img': np.concatenate(pr_img),
        'pr_area': np.concatenate(pr_area).astype(np.int64), 'pr_ov': np.concatenate(pr_ov),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_images': idx,
        'dataset': cfg['data']['dataset'], 'ckpt': ckpt, 'epoch': -1 if epoch is None else epoch,
    }
    prob_arr = np.concatenate(probs) if args.save_probs else None
    del trainer
    torch.cuda.empty_cache()
    return rec, prob_arr


def _rate(mask):
    return float(mask.mean()) if mask.size else float('nan')


def summarize(rec, tau):
    area, hit = rec['gt_area'], rec['gt_hit']
    cov = np.divide(hit, area, out=np.zeros_like(hit, dtype=np.float64), where=area > 0)
    miss = hit == 0
    key = f'component_recall_cov{int(round(tau * 100))}'

    by_area = []
    for lo, hi in AREA_BINS:
        sel = area >= lo
        if hi is not None:
            sel &= area < hi
        by_area.append({
            'bin': f'[{lo},{hi if hi is not None else "inf"})',
            'n': int(sel.sum()),
            'complete_miss': int((miss & sel).sum()),
            'complete_miss_rate': _rate(miss[sel]),
            key: _rate(cov[sel] >= tau),
        })

    tp, fp, fn = rec['tp'], rec['fp'], rec['fn']
    fa = rec['pr_ov'] == 0
    return {
        'pixel': {
            'iou': tp / max(tp + fp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
        },
        'n_gt_components': int(area.size),
        'n_complete_miss': int(miss.sum()),
        'complete_miss_rate': _rate(miss),
        key: _rate(cov >= tau),
        'area_share_in_complete_misses': float(area[miss].sum() / max(area.sum(), 1)),
        'complete_miss_area_median': float(np.median(area[miss])) if miss.any() else float('nan'),
        'n_pred_components': int(rec['pr_area'].size),
        'n_false_alarm_components': int(fa.sum()),
        'false_alarm_component_rate': _rate(fa),
        'by_area': by_area,
    }


def mcnemar_exact(a_miss, b_miss):
    b = int(np.sum(a_miss & ~b_miss))
    c = int(np.sum(~a_miss & b_miss))
    if b + c == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), b + c, 0.5).pvalue
    except ImportError:
        from scipy.stats import binom_test
        p = binom_test(min(b, c), b + c, 0.5)
    return b, c, float(p)


def bootstrap_diff(img_ids, a_miss, b_miss, n_images, reps, seed):
    ncomp = np.bincount(img_ids, minlength=n_images).astype(np.float64)
    ma = np.bincount(img_ids, weights=a_miss.astype(np.float64), minlength=n_images)
    mb = np.bincount(img_ids, weights=b_miss.astype(np.float64), minlength=n_images)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        s = rng.integers(0, n_images, n_images)
        diffs[r] = (ma[s].sum() - mb[s].sum()) / max(ncomp[s].sum(), 1.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def compare(name_a, rec_a, name_b, rec_b, reps, seed):
    if not (np.array_equal(rec_a['gt_img'], rec_b['gt_img'])
            and np.array_equal(rec_a['gt_area'], rec_b['gt_area'])):
        raise RuntimeError(f'{name_a} vs {name_b}: ground-truth components differ (different test sets?)')
    a_miss, b_miss = rec_a['gt_hit'] == 0, rec_b['gt_hit'] == 0
    out = {}
    for label, sel in [('all', np.ones_like(a_miss))] + [
            (f'[{lo},{hi if hi is not None else "inf"})',
             (rec_a['gt_area'] >= lo) & ((rec_a['gt_area'] < hi) if hi is not None else True))
            for lo, hi in AREA_BINS]:
        sel = sel.astype(bool)
        b, c, p = mcnemar_exact(a_miss[sel], b_miss[sel])
        entry = {
            f'missed_by_{name_a}_found_by_{name_b}': b,
            f'found_by_{name_a}_missed_by_{name_b}': c,
            'mcnemar_exact_p': p,
            f'cmr_{name_a}': _rate(a_miss[sel]),
            f'cmr_{name_b}': _rate(b_miss[sel]),
        }
        if label == 'all':
            lo_ci, hi_ci = bootstrap_diff(rec_a['gt_img'], a_miss, b_miss, rec_a['n_images'], reps, seed)
            entry[f'cmr_diff_{name_a}_minus_{name_b}_ci95'] = [lo_ci, hi_ci]
        out[label] = entry
    return out


def save_rec(path, rec):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()})


def load_rec(path):
    z = np.load(path, allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    for k in ('tp', 'fp', 'fn', 'tn', 'n_images', 'epoch'):
        rec[k] = int(rec[k])
    for k in ('dataset', 'ckpt'):
        rec[k] = str(rec[k])
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True, help='name=path/to/best.pth (repeatable)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--tau', type=float, default=0.5, help='coverage needed to count a component as detected')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_probs', action='store_true')
    p.add_argument('--recompute', action='store_true', help='ignore cached per-model records')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = [tuple(s.split('=', 1)) for s in args.ckpt]

    records = {}
    for name, ckpt in models:
        cache = os.path.join(args.out_dir, f'{name}.npz')
        if os.path.isfile(cache) and not args.recompute:
            print(f'[{name}] cached -> {cache}')
            records[name] = load_rec(cache)
            continue
        print(f'[{name}] {ckpt}')
        rec, probs = run_model(ckpt, args)
        save_rec(cache, rec)
        if probs is not None:
            np.savez_compressed(os.path.join(args.out_dir, f'{name}_probs.npz'), probs=probs)
        records[name] = rec

    summary = {'threshold': args.threshold, 'tau': args.tau, 'area_bins_px': AREA_BINS, 'models': {}, 'pairs': {}}
    for name, rec in records.items():
        s = summarize(rec, args.tau)
        s.update({'dataset': rec['dataset'], 'ckpt': rec['ckpt'], 'epoch': rec['epoch'], 'n_images': rec['n_images']})
        summary['models'][name] = s
    names = list(records)
    for a, b in itertools.combinations(names, 2):
        summary['pairs'][f'{a}_vs_{b}'] = compare(a, records[a], b, records[b], args.bootstrap, args.seed)

    out_json = os.path.join(args.out_dir, 'summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    key = f'component_recall_cov{int(round(args.tau * 100))}'
    print('\n' + '=' * 118)
    print(f'{"model":10s} {"IoU":>6s} {"Pre":>6s} {"Rec":>6s} | {"GTcomp":>6s} {"CM":>5s} {"CMR":>6s} '
          f'{"rec@" + str(int(args.tau * 100)):>7s} {"CMarea":>7s} | {"PRcomp":>6s} {"FA":>5s} {"FArate":>6s} | CMR by area '
          + ' '.join(f'{lo}-{hi if hi else "inf"}' for lo, hi in AREA_BINS))
    for name in names:
        s = summary['models'][name]
        px = s['pixel']
        bins = ' '.join(f'{b["complete_miss_rate"] * 100:5.1f}%({b["n"]})' for b in s['by_area'])
        print(f'{name:10s} {px["iou"] * 100:6.2f} {px["precision"] * 100:6.2f} {px["recall"] * 100:6.2f} | '
              f'{s["n_gt_components"]:6d} {s["n_complete_miss"]:5d} {s["complete_miss_rate"] * 100:5.2f}% '
              f'{s[key] * 100:6.2f}% {s["area_share_in_complete_misses"] * 100:6.2f}% | '
              f'{s["n_pred_components"]:6d} {s["n_false_alarm_components"]:5d} {s["false_alarm_component_rate"] * 100:5.2f}% | {bins}')
    print('-' * 118)
    for pair, res in summary['pairs'].items():
        allr = res['all']
        vals = list(allr.values())
        ci = [v for k, v in allr.items() if k.endswith('_ci95')][0]
        print(f'{pair:22s} b={vals[0]:4d} c={vals[1]:4d} McNemar p={allr["mcnemar_exact_p"]:.2e} '
              f'CMR diff CI95=[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] pts')
    print('=' * 118)
    print(f'summary -> {out_json}')


if __name__ == '__main__':
    main()
