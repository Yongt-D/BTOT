"""Zero-shot transfer between LEVIR-CD and WHU-CD, from the logit margin without ties.

For each checkpoint the source is the dataset it was trained on and the target is the other one. On both test
splits the float64 logit margin l1 - l0 is histogrammed on a fine grid, separately for change and no-change pixels
(single pass). Reported per split: AUROC, AP, the IoU at the operating threshold (margin 0, probability 0.5), the
oracle IoU and its margin threshold, and for the target the IoU after positive-rate matching, i.e. at the threshold
that makes the predicted change share on the target equal to the true change share of the source test split.

Usage (repo root): python analysis/zeroshot_eval.py OUT_DIR name=CKPT [name=CKPT ...] [--max_images N]
Caches OUT_DIR/<name>.json and OUT_DIR/<name>_hists.npz.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_component_miss as E  # noqa: E402
from dinobcd.datasets import DATASET_CONFIGS  # noqa: E402

LO, HI, NB = -80.0, 40.0, 24000          # margin grid, bin width 5e-3; margin 0 falls on a bin edge
EDGES = np.linspace(LO, HI, NB + 1)
OTHER = {'LEVIR_CD': 'WHU_CD', 'WHU_CD': 'LEVIR_CD'}


def histogram(trainer, cfg, dataset, max_images):
    cfg = json.loads(json.dumps(cfg))
    cfg['data']['dataset'] = dataset
    cfg['data']['data_root'] = DATASET_CONFIGS[dataset]['data_root']
    cfg['data']['test_list'] = 'test.txt'
    _, loader = E.build_test_loader(cfg, 8, 4)
    h = np.zeros((2, NB))
    n = 0
    model = trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            img1, img2 = batch['img1'].to(trainer.device), batch['img2'].to(trainer.device)
            lab = batch['label'].to(trainer.device) == 1
            logits = model(img1, img2).double()
            margin = (logits[:, 1] - logits[:, 0]).clamp(LO, HI - 1e-9)
            for c, sel in ((0, ~lab), (1, lab)):
                h[c] += torch.histc(margin[sel], NB, LO, HI).cpu().numpy()
            n += img1.shape[0]
            if max_images and n >= max_images:
                break
    return h, n


def summarize(h, match_rate=None):
    """Metrics of predicting change for margin >= EDGES[k], for every k."""
    neg, pos = h
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    P, N = pos.sum(), neg.sum()
    iou = tp / np.maximum(tp + fp + (P - tp), 1)
    x = np.r_[fp / N, 0.0][::-1]
    y = np.r_[tp / P, 0.0][::-1]
    auroc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))
    rec = np.r_[tp / P, 0.0]
    ap = float(np.sum((rec[:-1] - rec[1:]) * (tp / np.maximum(tp + fp, 1))))
    k0 = int(np.searchsorted(EDGES, 0.0))
    kb = int(np.argmax(iou))
    r = {'auroc': auroc, 'ap': ap, 'iou_at_0.5': float(iou[k0]), 'oracle_iou': float(iou[kb]),
         'oracle_margin': float(EDGES[kb]), 'change_share': float(P / (P + N))}
    if match_rate is not None:
        km = int(np.argmin(np.abs((tp + fp) / (P + N) - match_rate)))
        r['rate_matched_iou'] = float(iou[km])
        r['rate_matched_margin'] = float(EDGES[km])
    return r


def main():
    args = sys.argv[1:]
    max_images = 0
    if '--max_images' in args:
        k = args.index('--max_images')
        max_images = int(args[k + 1])
        del args[k:k + 2]
    out_dir, specs = args[0], args[1:]
    os.makedirs(out_dir, exist_ok=True)
    for spec in specs:
        name, ckpt = spec.split('=', 1)
        path = os.path.join(out_dir, f'{name}.json')
        if os.path.isfile(path):
            print(f'{name}: cached', flush=True)
            continue
        trainer, cfg, epoch = E.load_model(ckpt, 0, 8)
        source = cfg['data']['dataset']
        target = OTHER[source]
        hs, ns = histogram(trainer, cfg, source, max_images)
        ht, nt = histogram(trainer, cfg, target, max_images)
        src = summarize(hs)
        tgt = summarize(ht, match_rate=src['change_share'])
        np.savez_compressed(os.path.join(out_dir, f'{name}_hists.npz'), source=hs, target=ht, edges=EDGES)
        res = {'name': name, 'ckpt': ckpt, 'epoch': epoch, 'source': source, 'target': target,
               'n_images': {'source': ns, 'target': nt}, 'source_metrics': src, 'target_metrics': tgt}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
        print(f"{name} {source}->{target}: source IoU@0.5 {src['iou_at_0.5'] * 100:.2f} | target AUROC "
              f"{tgt['auroc']:.3f} AP {tgt['ap']:.3f} IoU@0.5 {tgt['iou_at_0.5'] * 100:.2f} oracle "
              f"{tgt['oracle_iou'] * 100:.2f}@{tgt['oracle_margin']:.2f} rate-matched {tgt['rate_matched_iou'] * 100:.2f}",
              flush=True)
        del trainer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
