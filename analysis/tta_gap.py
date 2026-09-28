"""Where does the TTA gain come from? Single view vs flip-TTA on the WHU test set.

The trainer's TTA (DinoBCD.predict_tta) averages softmax probabilities over {identity, flip-W, flip-H, flip-HW}
and thresholds at 0.5. Here the four views are computed once per batch, so one pass gives:

- the no-TTA result (identity view) and the TTA result, bit-identical to predict / predict_tta;
- each flipped view on its own: flipped views about as good as identity -> ensemble variance reduction,
  one view clearly better -> orientation bias;
- the TTA confusion change against the identity view (TP gained, FP removed, FN added);
- view instability: 1 - |all views positive| / |any view positive|, and the mean pairwise Jaccard distance
  between the binary view masks.

Usage (repo root): python analysis/tta_gap.py OUT_DIR name=CKPT [name=CKPT ...]
Per-model results are cached as OUT_DIR/<name>.json plus per-image confusion counts OUT_DIR/<name>.npz
(counts[N, 5, 3]: views id/flipW/flipH/flipHW/tta x tp/fp/fn, for tta_gap_boot.py); summary.json collects every
cached model in OUT_DIR.
"""
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_component_miss as E  # noqa: E402

VIEWS = {'id': [], 'flipW': [3], 'flipH': [2], 'flipHW': [2, 3]}   # image dims; prob dims are one lower
THRESHOLD = 0.5


def counts(pred, label):
    return [(pred & label).sum().item(), (pred & ~label).sum().item(), (~pred & label).sum().item()]


def image_counts(pred, label):
    return torch.stack([(pred & label).sum((1, 2)), (pred & ~label).sum((1, 2)), (~pred & label).sum((1, 2))], dim=1)


def metrics(c):
    tp, fp, fn = c
    return {'iou': tp / max(tp + fp + fn, 1), 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'tp': tp, 'fp': fp, 'fn': fn}


def operator(model_cfg):
    for key in ('btot', 'hbca', 'ssm'):
        for k, v in model_cfg.items():
            if key in k.lower() and (v is True or (isinstance(v, dict) and v.get('enable') is True)):
                return key
    return 'diff'


def run(name, ckpt, out_dir):
    trainer, cfg, epoch = E.load_model(ckpt, 0, 4)
    _, loader = E.build_test_loader(cfg, 4, 4)
    model, dev = trainer.model, trainer.device
    names = list(VIEWS)
    pairs = list(itertools.combinations(names, 2))
    view_c = {v: [0, 0, 0] for v in names}
    tta_c = [0, 0, 0]
    any_pos = all_pos = 0
    inter = {p: 0 for p in pairs}
    union = {p: 0 for p in pairs}
    per_image = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            img1, img2 = batch['img1'].to(dev), batch['img2'].to(dev)
            label = batch['label'].to(dev) == 1
            probs = {}
            for v, dims in VIEWS.items():
                if dims:
                    p = F.softmax(model.forward(torch.flip(img1, dims), torch.flip(img2, dims)), dim=1)[:, 1]
                    probs[v] = torch.flip(p, [d - 1 for d in dims])
                else:
                    probs[v] = F.softmax(model.forward(img1, img2), dim=1)[:, 1]
            masks = {v: probs[v] >= THRESHOLD for v in names}
            for v in names:
                view_c[v] = [a + b for a, b in zip(view_c[v], counts(masks[v], label))]
            tta = torch.stack([probs[v] for v in names], dim=0).mean(dim=0) >= THRESHOLD
            tta_c = [a + b for a, b in zip(tta_c, counts(tta, label))]
            per_image.append(torch.stack([image_counts(masks[v], label) for v in names]
                                         + [image_counts(tta, label)], dim=1).cpu())
            stack = torch.stack([masks[v] for v in names])
            any_pos += stack.any(0).sum().item()
            all_pos += stack.all(0).sum().item()
            for a, b in pairs:
                inter[(a, b)] += (masks[a] & masks[b]).sum().item()
                union[(a, b)] += (masks[a] | masks[b]).sum().item()

    views = {v: metrics(view_c[v]) for v in names}
    tta_m = metrics(tta_c)
    jd = {f'{a}-{b}': 1 - inter[(a, b)] / max(union[(a, b)], 1) for a, b in pairs}
    res = {
        'name': name, 'ckpt': ckpt, 'epoch': epoch, 'operator': operator(cfg['model']),
        'views': views, 'tta': tta_m,
        'tta_minus_id': {k: tta_m[k] - views['id'][k] for k in tta_m},
        'instability': 1 - all_pos / max(any_pos, 1), 'any_pos': any_pos, 'all_pos': all_pos,
        'pair_jaccard_dist': jd, 'mean_pair_jaccard_dist': sum(jd.values()) / len(jd),
        'seconds': time.time() - t0,
    }
    with open(os.path.join(out_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f'{name}.npz'), counts=torch.cat(per_image).numpy(),
                        views=np.array(names + ['tta']), columns=np.array(['tp', 'fp', 'fn']))
    del trainer, model
    torch.cuda.empty_cache()
    return res


def line(r):
    v, t, d = r['views'], r['tta'], r['tta_minus_id']
    return (f"{r['name']:13s} {r['operator']:5s} ep{r['epoch']!s:>4s} | id {v['id']['iou'] * 100:5.2f} "
            f"P{v['id']['precision'] * 100:5.1f} R{v['id']['recall'] * 100:5.1f} | TTA {t['iou'] * 100:5.2f} "
            f"P{t['precision'] * 100:5.1f} R{t['recall'] * 100:5.1f} | dIoU {d['iou'] * 100:+5.2f} "
            f"dTP {d['tp']:+7d} dFP {d['fp']:+7d} | views W {v['flipW']['iou'] * 100:5.2f} "
            f"H {v['flipH']['iou'] * 100:5.2f} HW {v['flipHW']['iou'] * 100:5.2f} | "
            f"unstable {r['instability'] * 100:5.2f}% pairJD {r['mean_pair_jaccard_dist'] * 100:5.2f}%")


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for spec in sys.argv[2:]:
        name, ckpt = spec.split('=', 1)
        if all(os.path.isfile(os.path.join(out_dir, f'{name}.{ext}')) for ext in ('json', 'npz')):
            print(f'[{name}] cached', flush=True)
            continue
        print(line(run(name, ckpt, out_dir)), flush=True)

    rows = []
    for fp in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        if os.path.basename(fp) != 'summary.json':
            with open(fp, encoding='utf-8') as f:
                rows.append(json.load(f))
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({r['name']: r for r in rows}, f, indent=2)
    print('=' * 60)
    for r in rows:
        print(line(r))


if __name__ == '__main__':
    main()
