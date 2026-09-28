"""Paired image bootstrap on tta_gap.py per-image counts (WHU test).

For model A minus model B, resampling test images with replacement (same images for both models):
- id:        standard protocol, single pass on the test set as distributed;
- view_mean: single pass averaged over the four flipped copies of the test set (mean of per-view IoU, no ensembling);
- tta:       probability-averaged flip TTA.
Per model it also tests the identity view against the mean of the three flipped views, and lists the images that
contribute most to that gap (false-positive pixels in the identity view minus the flipped-view mean).

Usage (repo root): python analysis/tta_gap_boot.py OUT_DIR NAME_A NAME_B [N_BOOT] [TEST_LIST]
TEST_LIST (default: the WHU-CD test list) only names the listed images; it is ignored if its length does not match.
Writes OUT_DIR/boot_<A>_vs_<B>.json.
"""
import json
import os
import sys

import numpy as np

LIST_FILE = 'data_dir/WHU-CD/list/test.txt'


def iou(c):
    return c[..., 0] / np.maximum(c[..., 0] + c[..., 1] + c[..., 2], 1)


def summary(counts):
    v = iou(counts.sum(0))                       # [5]: id, flipW, flipH, flipHW, tta
    return {'id': v[0], 'view_mean': v[:4].mean(), 'tta': v[4], 'id_minus_flips': v[0] - v[1:4].mean()}


def ci(samples):
    s = np.asarray(samples)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))], float((s <= 0).mean())


def main():
    out_dir, a, b = sys.argv[1:4]
    n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    A = np.load(os.path.join(out_dir, f'{a}.npz'))['counts'].astype(np.float64)
    B = np.load(os.path.join(out_dir, f'{b}.npz'))['counts'].astype(np.float64)
    assert A.shape == B.shape, (A.shape, B.shape)
    n = A.shape[0]
    list_file = sys.argv[5] if len(sys.argv) > 5 else LIST_FILE
    names = [l.split()[0] for l in open(list_file, encoding='utf-8')] if os.path.isfile(list_file) else []
    if len(names) != n:
        names = []

    sa, sb = summary(A), summary(B)
    keys = ('id', 'view_mean', 'tta')
    boot = {k: [] for k in keys}
    own = {a: [], b: []}
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ba, bb = summary(A[idx]), summary(B[idx])
        for k in keys:
            boot[k].append(ba[k] - bb[k])
        own[a].append(ba['id_minus_flips'])
        own[b].append(bb['id_minus_flips'])

    res = {'n_images': n, 'n_boot': n_boot, 'models': {}, 'a_minus_b': {}}
    print(f'{a} vs {b}: {n} images, {n_boot} paired bootstrap resamples')
    for m, s, M in ((a, sa, A), (b, sb, B)):
        (lo, hi), p = ci(own[m])
        fp_gap = M[:, 0, 1] - M[:, 1:4, 1].mean(1)
        top = np.argsort(-fp_gap)[:5]
        res['models'][m] = {k: float(v) for k, v in s.items()}
        res['models'][m]['id_minus_flips_ci95'] = [lo, hi]
        res['models'][m]['top_identity_fp_excess'] = [
            {'index': int(i), 'image': names[i] if names else None, 'fp_excess_px': float(fp_gap[i])} for i in top]
        share = fp_gap[top].sum() / max(fp_gap.sum(), 1)
        print(f'  {m:13s} id {s["id"] * 100:6.2f}  view_mean {s["view_mean"] * 100:6.2f}  tta {s["tta"] * 100:6.2f} | '
              f'id - flips {s["id_minus_flips"] * 100:+5.2f} CI95 [{lo * 100:+5.2f}, {hi * 100:+5.2f}] | '
              f'top-5 images carry {share * 100:5.1f}% of the net identity FP excess '
              f'({", ".join(names[i] if names else str(i) for i in top)})')
    for k in keys:
        (lo, hi), p = ci(boot[k])
        d = sa[k] - sb[k]
        res['a_minus_b'][k] = {'diff': float(d), 'ci95': [lo, hi], 'p_le_0': p}
        print(f'  {a} - {b} {k:9s}: {d * 100:+5.2f} IoU  CI95 [{lo * 100:+5.2f}, {hi * 100:+5.2f}]  P(diff<=0)={p:.3f}')

    with open(os.path.join(out_dir, f'boot_{a}_vs_{b}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
