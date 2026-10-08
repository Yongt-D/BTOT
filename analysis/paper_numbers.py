"""Every number, table and statistics figure of the JSTARS operator-evaluation paper, from cached results.

Inputs (all produced by analysis/tta_gap.py, which caches per-image confusion counts):
- results/tta_gap_levir_img/<op>_levir_neutral_s<seed>.{json,npz}  matched recipe, LEVIR-CD
- results/tta_gap_whu_img/{diff_neutral,diff_neutral_s123,btot_neutral}.{json,npz}  matched recipe, WHU-CD
- results/tta_gap_whu/<name>.json  earlier training recipe, WHU-CD (orientation / recipe confounds)
Missing models (runs still training) are skipped and their table cells print as "--".

Outputs: paper/JSTARS_v2/gen/{numbers.tex, tab_levir.tex, tab_power.tex} and
paper/JSTARS_v2/figs/{forest.pdf, tail.pdf, similarity.pdf}.
Usage (repo root): python analysis/paper_numbers.py [N_BOOT]
"""
import itertools
import json
import os
import sys

import numpy as np

LEVIR_DIR = 'results/tta_gap_levir_img'
WHU_DIR = 'results/tta_gap_whu_img'
WHU_OLD_DIR = 'results/tta_gap_whu'
OUT = 'paper_outputs'
OPS = [('diff', 'Difference'), ('hbca', 'Cross-attention (HBCA)'), ('ssm', 'State-space (SSM)'),
       ('btot', 'Optimal transport (BTOT)')]
ABLATION = ('btot_nodustbin', 'BTOT without dustbin')
short_name = {'diff': 'Difference', 'hbca': 'HBCA', 'ssm': 'SSM', 'btot': 'BTOT', 'btot_nodustbin': 'BTOT without dustbin'}
SEEDS = (42, 123)
VIEW_ID, VIEW_TTA = 0, 4
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
RNG = np.random.default_rng(0)
for sub in ('gen', 'figs'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

numbers = {}


def put(key, value, fmt='{:.2f}'):
    numbers[key] = fmt.format(value)
    if fmt.startswith('{:+'):                      # unsigned copy for prose ("by 1.83 points")
        numbers[key + '.abs'] = fmt.replace('+', '').format(abs(value))


def load(d, name):
    p = os.path.join(d, name)
    if not (os.path.isfile(p + '.json') and os.path.isfile(p + '.npz')):
        return None
    return np.load(p + '.npz')['counts'].astype(np.float64)       # [N, 5 views, (tp, fp, fn)]


def iou(c):
    return c[..., 0] / np.maximum(c.sum(-1), 1)


def views(counts):
    """IoU of the identity view, the four-view mean, TTA and the view spread; P/R of the identity view."""
    tot = counts.sum(0)
    v = iou(tot)
    tp, fp, fn = tot[VIEW_ID]
    return {'id': v[0], 'vmean': v[:4].mean(), 'tta': v[4], 'spread': v[:4].max() - v[:4].min(),
            'p': tp / (tp + fp), 'r': tp / (tp + fn)}


def boot_weights(n, b):
    """Multinomial image-resampling weights, [b, n]."""
    return np.stack([np.bincount(RNG.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float64)


def boot_iou(counts, w, view):
    return iou(w @ counts[:, view, :])


def jk_loo(c):
    """Leave-one-image-out IoU of one view's counts [N, 3]."""
    tot = c.sum(0)
    return (tot[0] - c[:, 0]) / np.maximum(tot.sum() - c.sum(1), 1)


def jk_se(loo):
    n = len(loo)
    return np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())


def cell(x):
    return '--' if x is None else f'{x * 100:.2f}'


# ---------------------------------------------------------------- LEVIR-CD, matched recipe
levir = {}
for op, _ in OPS + [ABLATION]:
    for s in SEEDS:
        c = load(LEVIR_DIR, f'{op}_levir_neutral_s{s}')
        if c is not None:
            levir[(op, s)] = c
first = next(iter(levir.values()))
n_levir = first.shape[0]
gt_px = first[:, VIEW_ID, 0] + first[:, VIEW_ID, 2]
put('levir.n_images', n_levir, '{}')
put('levir.n_change_images', int((gt_px > 0).sum()), '{}')

W = boot_weights(n_levir, N_BOOT)
stats = {k: views(c) for k, c in levir.items()}
for (op, s), v in stats.items():
    for key, val in v.items():
        put(f'levir.{op}.s{s}.{key}', val * 100)


def seed_mean(op, view):
    """Seed-averaged IoU of an operator: point estimate, [b] image-bootstrap samples, number of seeds."""
    ks = [(op, s) for s in SEEDS if (op, s) in levir]
    if not ks:
        return None
    if view == 'vmean':
        smp = [np.mean([boot_iou(levir[k], W, v) for v in range(4)], axis=0) for k in ks]
    else:
        smp = [boot_iou(levir[k], W, VIEW_ID if view == 'id' else VIEW_TTA) for k in ks]
    return float(np.mean([stats[k][view] for k in ks])), np.mean(smp, axis=0), len(ks)


seedmean = {}
for op, _ in OPS + [ABLATION]:
    for view in ('id', 'vmean', 'tta'):
        r = seed_mean(op, view)
        if r is not None:
            seedmean[(op, view)] = r
            put(f'levir.{op}.mean.{view}', r[0] * 100)
            put(f'levir.{op}.nseeds', r[2], '{}')

delta, dsmp = {}, {}
for op, _ in OPS[1:] + [ABLATION]:
    ref = 'btot' if op == ABLATION[0] else 'diff'
    for view in ('id', 'vmean', 'tta'):
        if (op, view) in seedmean and (ref, view) in seedmean:
            pa, sa, _ = seedmean[(op, view)]
            pb, sb, _ = seedmean[(ref, view)]
            d, lo, hi = pa - pb, np.percentile(sa - sb, 2.5), np.percentile(sa - sb, 97.5)
            delta[(op, view)] = (d, lo, hi)
            dsmp[(op, view)] = sa - sb
            put(f'levir.{op}.delta.{view}', d * 100, '{:+.2f}')
            put(f'levir.{op}.delta.{view}.lo', lo * 100, '{:+.2f}')
            put(f'levir.{op}.delta.{view}.hi', hi * 100, '{:+.2f}')

# Training noise. The image bootstrap holds the trained models fixed. The seed-to-seed variance sigma^2 is pooled over
# the operators trained with both seeds (one df each, (x42 - x123)^2 / 2), added to the bootstrap variance of a
# seed-averaged difference as sigma^2 (1/n_A + 1/n_B), and the interval uses a t quantile with Satterthwaite df.
from scipy.stats import t as t_dist  # noqa: E402

for view in ('id', 'vmean', 'tta'):
    pairs = [(stats[(op, 42)][view] - stats[(op, 123)][view]) ** 2 / 2
             for op, _ in OPS if all((op, s) in stats for s in SEEDS)]
    if not pairs:
        continue
    s2, df_t = float(np.mean(pairs)), len(pairs)
    put(f'levir.train.sd.{view}', np.sqrt(s2) * 100)
    put('levir.train.df', df_t, '{}')
    for (op, v), smp in dsmp.items():
        if v != view:
            continue
        ref = 'btot' if op == ABLATION[0] else 'diff'
        vt = s2 * (1 / seedmean[(op, view)][2] + 1 / seedmean[(ref, view)][2])
        vb = float(smp.var(ddof=1))
        df = df_t * (1 + vb / vt) ** 2
        half = t_dist.ppf(0.975, df) * np.sqrt(vb + vt)
        d = delta[(op, view)][0]
        put(f'levir.{op}.delta.{view}.se.boot', np.sqrt(vb) * 100)
        put(f'levir.{op}.delta.{view}.lo2', (d - half) * 100, '{:+.2f}')
        put(f'levir.{op}.delta.{view}.hi2', (d + half) * 100, '{:+.2f}')
        put(f'levir.{op}.delta.{view}.df', df, '{:.1f}')

for op, _ in OPS:
    if all((op, s) in stats for s in SEEDS):
        put(f'levir.{op}.seedrange', abs(stats[(op, 42)]['id'] - stats[(op, 123)]['id']) * 100)
opd = [delta[(op, 'id')][0] for op, _ in OPS[1:] if (op, 'id') in delta]
if opd:
    put('levir.opdelta.min', min(opd) * 100, '{:+.2f}')
    put('levir.opdelta.max', max(opd) * 100, '{:+.2f}')

rows = []
for op, label in OPS + [ABLATION]:
    if op == ABLATION[0] and not any(k[0] == op for k in levir):
        continue
    ks = [(op, s) for s in SEEDS if (op, s) in stats]
    if op == 'diff':
        dcell = 'reference'
    elif (op, 'id') in delta:
        d, lo, hi = delta[(op, 'id')]
        dcell = f'${d * 100:+.2f}$ [${lo * 100:+.2f}$, ${hi * 100:+.2f}$]'
    else:
        dcell = '--'
    rows.append(' & '.join([
        label,
        cell(stats.get((op, 42), {}).get('id')),
        cell(stats.get((op, 123), {}).get('id')),
        cell(seedmean[(op, 'id')][0] if (op, 'id') in seedmean else None),
        dcell,
        cell(seedmean[(op, 'vmean')][0] if (op, 'vmean') in seedmean else None),
        cell(seedmean[(op, 'tta')][0] if (op, 'tta') in seedmean else None),
        cell(np.mean([stats[k]['p'] for k in ks]) if ks else None),
        cell(np.mean([stats[k]['r'] for k in ks]) if ks else None)]) + r' \\')
with open(os.path.join(OUT, 'gen', 'tab_levir.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(rows) + '\n')

# per-image error (FP + FN pixels) correlation between every pair of LEVIR models
order = [o for o, _ in OPS] + [ABLATION[0]]
keys = sorted(levir, key=lambda k: (order.index(k[0]), k[1]))
err = {k: levir[k][:, VIEW_ID, 1] + levir[k][:, VIEW_ID, 2] for k in keys}
R = np.array([[np.corrcoef(err[a], err[b])[0, 1] for b in keys] for a in keys])
groups = {'sameop': [], 'diffop.sameseed': [], 'diffop.diffseed': []}
for i, a in enumerate(keys):
    for j, b in enumerate(keys):
        if j <= i or ABLATION[0] in (a[0], b[0]):
            continue
        g = 'sameop' if a[0] == b[0] else 'diffop.sameseed' if a[1] == b[1] else 'diffop.diffseed'
        groups[g].append(R[i, j])
groups['diffop'] = groups['diffop.sameseed'] + groups['diffop.diffseed']
groups['btotdiff.sameseed'] = [R[keys.index(('diff', s)), keys.index(('btot', s))] for s in SEEDS
                               if ('diff', s) in keys and ('btot', s) in keys]
for g, v in groups.items():
    if v:
        put(f'levir.errcorr.{g}', float(np.mean(v)), '{:.3f}')
        put(f'levir.errcorr.{g}.n', len(v), '{}')
# how typical each model's errors are: mean correlation with every other operator run (ablation excluded)
main = [i for i, k in enumerate(keys) if k[0] != ABLATION[0]]
rowmean = {keys[i]: np.mean([R[i, j] for j in main if j != i]) for i in main}
for (op, s), v in rowmean.items():
    put(f'levir.errcorr.row.{op}.s{s}', v, '{:.3f}')
put('levir.errcorr.row.min', min(rowmean.values()), '{:.3f}')
put('levir.errcorr.row.max', max(rowmean.values()), '{:.3f}')
offdiag = sorted((R[i, j] for i in main for j in main if j > i), reverse=True)
put('levir.errcorr.third', offdiag[2], '{:.3f}')

# ---------------------------------------------------------------- WHU-CD, matched recipe
whu = {n: load(WHU_DIR, n) for n in ('diff_neutral', 'diff_neutral_s123', 'btot_neutral')}
whu = {k: v for k, v in whu.items() if v is not None}
A, B = whu['btot_neutral'], whu['diff_neutral']
n_whu = A.shape[0]
gw_px = A[:, VIEW_ID, 0] + A[:, VIEW_ID, 2]
put('whu.n_images', n_whu, '{}')
put('whu.n_change_images', int((gw_px > 0).sum()), '{}')
wstats = {k: views(v) for k, v in whu.items()}
for k, v in wstats.items():
    for key, val in v.items():
        put(f'whu.{k}.{key}', val * 100)
    put(f'whu.{k}.ttagain', (v['tta'] - v['id']) * 100, '{:+.2f}')
put('whu.diff.seedrange', abs(wstats['diff_neutral']['id'] - wstats['diff_neutral_s123']['id']) * 100)

Ww = boot_weights(n_whu, N_BOOT)
for view, vi in (('id', VIEW_ID), ('tta', VIEW_TTA)):
    s = boot_iou(A, Ww, vi) - boot_iou(B, Ww, vi)
    put(f'whu.btotdiff.{view}', (iou(A.sum(0))[vi] - iou(B.sum(0))[vi]) * 100, '{:+.2f}')
    put(f'whu.btotdiff.{view}.lo', np.percentile(s, 2.5) * 100, '{:+.2f}')
    put(f'whu.btotdiff.{view}.hi', np.percentile(s, 97.5) * 100, '{:+.2f}')
    ea, eb = A[:, vi, 1] + A[:, vi, 2], B[:, vi, 1] + B[:, vi, 2]
    gap = np.abs(ea - eb)
    keep = np.ones(n_whu, bool)
    keep[np.argsort(-gap)[:5]] = False
    put(f'whu.btotdiff.{view}.drop5', (iou(A[keep].sum(0))[vi] - iou(B[keep].sum(0))[vi]) * 100, '{:+.2f}')
    put(f'whu.btotdiff.{view}.top10share', np.sort(gap)[::-1][:10].sum() / gap.sum() * 100, '{:.1f}')

# ---------------------------------------------------------------- power of the two test sets
power = {}
for ds, ca, cb in (('whu', A, B), ('levir', levir[('btot', 42)], levir[('diff', 42)])):
    g = ca[:, VIEW_ID, 0] + ca[:, VIEW_ID, 2]
    la, lb = jk_loo(ca[:, VIEW_ID, :]), jk_loo(cb[:, VIEW_ID, :])
    se1, sed = (jk_se(la) + jk_se(lb)) / 2, jk_se(la - lb)
    power[ds] = dict(n=len(g), nchg=int((g > 0).sum()), top10=np.sort(g)[::-1][:10].sum() / g.sum(), se1=se1, sed=sed)
    put(f'{ds}.top10share', power[ds]['top10'] * 100, '{:.1f}')
    put(f'{ds}.se.single', se1 * 100)
    put(f'{ds}.se.diff', sed * 100)
    put(f'{ds}.mdd', 2.80 * sed * 100)          # 80% power, two-sided alpha 0.05
    put(f'{ds}.ci.half', 1.96 * sed * 100)
put('power.se.ratio', power['whu']['se1'] / power['levir']['se1'], '{:.1f}')
put('power.sed.ratio', power['whu']['sed'] / power['levir']['sed'], '{:.0f}')

# ---------------------------------------------------------------- spatial dependence of test patches, MDD over model pairs
# LEVIR-CD test patches are 256 px tiles of 128 source images (test_<img>_<r>_<c>.png); WHU-CD patches are tiles of
# one 127 x 60 tiling (img<k>.png, k zero-based, row-major). Resample source images / 8 x 8-tile blocks instead of patches.
BLOCK = 8


def read_names(ds):
    return [l.strip() for l in open(f'splits/{ds}/test.txt', encoding='utf-8') if l.strip()]


lev_names, whu_names = read_names('LEVIR-CD'), read_names('WHU-CD')
assert len(lev_names) == n_levir and len(whu_names) == n_whu
lev_grp = np.array([int(n.split('_')[1]) for n in lev_names])
whu_k = np.array([int(n[3:-4]) for n in whu_names])
assert whu_k.min() >= 0 and whu_k.max() < 127 * 60
whu_grp = (whu_k // 127 // BLOCK) * 1000 + (whu_k % 127 // BLOCK)


def cluster_weights(groups, b):
    """Resample whole clusters with replacement; per-image weights [b, N]."""
    u, inv = np.unique(groups, return_inverse=True)
    w = np.stack([np.bincount(RNG.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(b)]).astype(np.float64)
    return w[:, inv], len(u)


def jk_loo_cluster(c, groups):
    """Delete-one-cluster IoU of one view's counts [N, 3]."""
    u = np.unique(groups)
    tot = c.sum(0)
    rest = tot - np.stack([c[groups == x].sum(0) for x in u])
    return rest[:, 0] / np.maximum(rest.sum(1), 1)


Wc, n_lev_cl = cluster_weights(lev_grp, N_BOOT)
Wb, n_whu_bl = cluster_weights(whu_grp, N_BOOT)
put('levir.n_clusters', n_lev_cl, '{}')
put('whu.n_blocks', n_whu_bl, '{}')
put('whu.block', BLOCK, '{}')
for op, _ in OPS[1:] + [ABLATION]:                       # seed-averaged single-pass difference, cluster bootstrap
    ref = 'btot' if op == ABLATION[0] else 'diff'
    def smean(o):
        ks = [(o, s) for s in SEEDS if (o, s) in levir]
        return np.mean([boot_iou(levir[k], Wc, VIEW_ID) for k in ks], axis=0)
    s = smean(op) - smean(ref)
    put(f'levir.{op}.delta.id.clo', np.percentile(s, 2.5) * 100, '{:+.2f}')
    put(f'levir.{op}.delta.id.chi', np.percentile(s, 97.5) * 100, '{:+.2f}')
for view, vi in (('id', VIEW_ID), ('tta', VIEW_TTA)):      # WHU BTOT vs difference, block bootstrap
    s = boot_iou(A, Wb, vi) - boot_iou(B, Wb, vi)
    put(f'whu.btotdiff.{view}.blo', np.percentile(s, 2.5) * 100, '{:+.2f}')
    put(f'whu.btotdiff.{view}.bhi', np.percentile(s, 97.5) * 100, '{:+.2f}')

cpower = {}
for ds, ca, cb, grp in (('whu', A, B, whu_grp), ('levir', levir[('btot', 42)], levir[('diff', 42)], lev_grp)):
    g = ca[:, VIEW_ID, 0] + ca[:, VIEW_ID, 2]
    u = np.unique(grp)
    gc = np.array([g[grp == x].sum() for x in u])
    la, lb = jk_loo_cluster(ca[:, VIEW_ID, :], grp), jk_loo_cluster(cb[:, VIEW_ID, :], grp)
    se1, sed = (jk_se(la) + jk_se(lb)) / 2, jk_se(la - lb)
    cpower[ds] = dict(n=len(u), nchg=int((gc > 0).sum()), top10=np.sort(gc)[::-1][:10].sum() / gc.sum(), se1=se1, sed=sed)
    put(f'{ds}.se.single.cluster', se1 * 100)
    put(f'{ds}.se.diff.cluster', sed * 100)
    put(f'{ds}.mdd.cluster', 2.80 * sed * 100)

# MDD of every pair of matched-recipe operator models (not only BTOT vs difference, seed 42)
lev_models = {k: v for k, v in levir.items() if k[0] != ABLATION[0]}
whu_models = {'btot_neutral': A, 'diff_neutral': B, 'diff_neutral_s123': whu['diff_neutral_s123']}
for ds, models, grp, unit in (('levir', lev_models, lev_grp, 'cluster'), ('whu', whu_models, whu_grp, 'block')):
    for lab, g in (('', np.arange(len(grp))), (f'.{unit}', grp)):
        v = [2.80 * jk_se(jk_loo_cluster(ca[:, VIEW_ID, :], g) - jk_loo_cluster(cb[:, VIEW_ID, :], g)) * 100
             for ca, cb in itertools.combinations(models.values(), 2)]
        put(f'{ds}.mdd.pairs{lab}.min', min(v))
        put(f'{ds}.mdd.pairs{lab}.max', max(v))
        put(f'{ds}.mdd.pairs{lab}.median', float(np.median(v)))
    put(f'{ds}.mdd.pairs.n', len(v), '{}')
g0 = np.arange(n_whu)
put('whu.mdd.diffdiff', 2.80 * jk_se(jk_loo_cluster(B[:, VIEW_ID, :], g0) - jk_loo_cluster(whu['diff_neutral_s123'][:, VIEW_ID, :], g0)) * 100)
put('whu.mdd.btot_diff123', 2.80 * jk_se(jk_loo_cluster(A[:, VIEW_ID, :], g0) - jk_loo_cluster(whu['diff_neutral_s123'][:, VIEW_ID, :], g0)) * 100)
# overall range over pairs and resampling units, used for the audit comparison and the shaded band of the audit figure
mdd_range = {ds: (min(float(numbers[f'{ds}.mdd.pairs.min']), float(numbers[f'{ds}.mdd.pairs.{u}.min'])),
                  max(float(numbers[f'{ds}.mdd.pairs.max']), float(numbers[f'{ds}.mdd.pairs.{u}.max'])))
             for ds, u in (('levir', 'cluster'), ('whu', 'block'))}
for ds in mdd_range:
    put(f'{ds}.mdd.lo', mdd_range[ds][0])
    put(f'{ds}.mdd.hi', mdd_range[ds][1])

with open(os.path.join(OUT, 'gen', 'tab_power.tex'), 'w', encoding='utf-8') as f:
    for ds, name in (('whu', 'WHU-CD'), ('levir', 'LEVIR-CD')):
        for p, unit in ((power[ds], 'patch'), (cpower[ds], 'block' if ds == 'whu' else 'image')):
            f.write(f"{name}, {unit} & {p['n']} & {p['nchg']} & {p['top10'] * 100:.1f} & {p['se1'] * 100:.2f} & "
                    f"{p['sed'] * 100:.2f} & {2.80 * p['sed'] * 100:.2f} \\\\\n")

# ---------------------------------------------------------------- earlier recipe on WHU-CD (confounds)
old = {}
for n in ('diff_old', 'btot_s42', 'btot_s456', 'btot_iters1', 'hbca_s42', 'v6_s123', 'ssm_s42'):
    d = json.load(open(os.path.join(WHU_OLD_DIR, n + '.json'), encoding='utf-8'))
    v = [d['views'][k]['iou'] for k in ('id', 'flipW', 'flipH', 'flipHW')]
    old[n] = {'id': v[0], 'vmean': float(np.mean(v)), 'tta': d['tta']['iou'], 'spread': max(v) - min(v)}
    for key, val in old[n].items():
        put(f'whuold.{n}.{key}', val * 100)
put('whuold.btotdiff.id', (old['btot_s42']['id'] - old['diff_old']['id']) * 100, '{:+.2f}')
put('whu.recipe.diff.change', (wstats['diff_neutral']['id'] - old['diff_old']['id']) * 100, '{:+.2f}')
put('whu.recipe.btot.change', (wstats['btot_neutral']['id'] - old['btot_s42']['id']) * 100, '{:+.2f}')
put('whuold.iters1.minus.full.id', (old['btot_iters1']['id'] - old['btot_s42']['id']) * 100, '{:+.2f}')
put('whuold.iters1.minus.full.vmean', (old['btot_iters1']['vmean'] - old['btot_s42']['vmean']) * 100, '{:+.2f}')
spreads = [v['spread'] for v in old.values()] + [v['spread'] for v in wstats.values()]
put('whu.spread.max', max(spreads) * 100)
put('whu.spread.min', min(spreads) * 100)
put('levir.spread.max', max(v['spread'] for v in stats.values()) * 100)
put('levir.spread.min', min(v['spread'] for v in stats.values()) * 100)
lg = [v['tta'] - v['id'] for v in stats.values()]
put('levir.ttagain.min', min(lg) * 100, '{:+.2f}')
put('levir.ttagain.max', max(lg) * 100, '{:+.2f}')

# ---------------------------------------------------------------- Appendix table: every view of every model
def view_row(label, recipe, v4, tta):
    return (f'{label} & {recipe} & ' + ' & '.join(f'{x * 100:.2f}' for x in v4) +
            f' & {np.mean(v4) * 100:.2f} & {tta * 100:.2f} & {(max(v4) - min(v4)) * 100:.2f} \\\\')


VIEW_KEYS = ('id', 'flipW', 'flipH', 'flipHW')
old_labels = [('diff_old', 'Difference, s42'), ('hbca_s42', 'HBCA, s42'), ('v6_s123', 'HBCA, s123'),
              ('ssm_s42', 'SSM, s42'), ('btot_s42', 'BTOT, s42'), ('btot_s456', 'BTOT, s456'),
              ('btot_iters1', 'BTOT, one iteration, s42')]
lines = [r'\multicolumn{9}{l}{\textit{WHU-CD}} \\']
for n, label in old_labels:
    d = json.load(open(os.path.join(WHU_OLD_DIR, n + '.json'), encoding='utf-8'))
    lines.append(view_row(label, 'earlier', [d['views'][k]['iou'] for k in VIEW_KEYS], d['tta']['iou']))
for n, label in (('diff_neutral', 'Difference, s42'), ('diff_neutral_s123', 'Difference, s123'), ('btot_neutral', 'BTOT, s42')):
    v = iou(whu[n].sum(0))
    lines.append(view_row(label, 'matched', v[:4], v[4]))
lines.append(r'\midrule')
lines.append(r'\multicolumn{9}{l}{\textit{LEVIR-CD}} \\')
for k in keys:
    v = iou(levir[k].sum(0))
    lines.append(view_row(f'{short_name[k[0]]}, s{k[1]}', 'matched', v[:4], v[4]))
with open(os.path.join(OUT, 'gen', 'tab_views.tex'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- literature audit (paper/JSTARS_v2/lit_audit/claims.csv)
import csv  # noqa: E402

with open(os.path.join('results', 'lit_audit', 'claims.csv'), encoding='utf-8-sig') as f:
    claims = list(csv.DictReader(f))
papers = {r['paper_key'] for r in claims}
put('audit.n_papers', len(papers), '{}')
put('audit.n_rows', len(claims), '{}')


def paper_flag(pred):
    return len({r['paper_key'] for r in claims if pred(r)})


put('audit.var.any', paper_flag(lambda r: r['repeated_runs_or_variance'].startswith('yes')), '{}')
put('audit.var.pct', paper_flag(lambda r: r['repeated_runs_or_variance'].startswith('yes')) / len(papers) * 100, '{:.1f}')
put('audit.tta.yes', paper_flag(lambda r: r['tta'].startswith('yes')), '{}')


def margins(dataset, group_prefix):
    return np.array([float(r['margin_iou']) for r in claims
                     if r['dataset'] == dataset and r['metric_type'] == 'change-class' and r['margin_iou']
                     and r['split_group'].startswith(group_prefix)])


lev_m = margins('LEVIR-CD', 'LEVIR official')
whu_m = margins('WHU-CD', 'WHU random 256px split 6096/762/762')
whu_all = np.array([float(r['margin_iou']) for r in claims
                    if r['dataset'] == 'WHU-CD' and r['metric_type'] == 'change-class' and r['margin_iou']])
lev_mdd, whu_mdd = 2.80 * power['levir']['sed'] * 100, 2.80 * power['whu']['sed'] * 100
put('audit.levir.n', len(lev_m), '{}')
put('audit.levir.median', float(np.median(lev_m)))
put('audit.levir.q1', float(np.percentile(lev_m, 25)))
put('audit.levir.q3', float(np.percentile(lev_m, 75)))
put('audit.levir.below_mdd', int((lev_m < lev_mdd).sum()), '{}')
put('audit.levir.below_one', int((lev_m < 1.0).sum()), '{}')
put('audit.levir.below_one.pct', (lev_m < 1.0).mean() * 100, '{:.0f}')
put('audit.levir.nonpos', int((lev_m <= 0).sum()), '{}')
put('audit.whu.n', len(whu_m), '{}')
put('audit.whu.median', float(np.median(whu_m)))
put('audit.whu.max', float(whu_m.max()))
put('audit.whu.below_mdd', int((whu_m < whu_mdd).sum()), '{}')
put('audit.levir.below_mdd.lo', int((lev_m < mdd_range['levir'][0]).sum()), '{}')   # below the smallest MDD over pairs
put('audit.levir.below_mdd.hi', int((lev_m < mdd_range['levir'][1]).sum()), '{}')   # below the largest MDD over pairs
put('audit.whu.below_mdd.lo', int((whu_m < mdd_range['whu'][0]).sum()), '{}')       # below the smallest MDD over pairs
put('audit.whuall.n', len(whu_all), '{}')
put('audit.whuall.median', float(np.median(whu_all)))
lev_prop = np.array([float(r['proposed_iou']) for r in claims
                     if r['dataset'] == 'LEVIR-CD' and r['metric_type'] == 'change-class' and r['proposed_iou']
                     and r['split_group'].startswith('LEVIR official')])
put('audit.levir.prop.median', float(np.median(lev_prop)))
put('audit.levir.prop.max', float(lev_prop.max()))
put('audit.levir.prop.n', len(lev_prop), '{}')
put('audit.levir.prop.ge85', int((lev_prop >= 85.0).sum()), '{}')
ours_id = [v['id'] * 100 for (op, s), v in stats.items() if op != ABLATION[0]]
put('levir.id.min', min(ours_id), '{:.1f}')
put('levir.id.max', max(ours_id), '{:.1f}')
put('audit.levir.gap', lev_prop.max() - min(ours_id), '{:.1f}')      # every one of our models is within this of the best

# ---------------------------------------------------------------- object-level complete misses, WHU-CD matched recipe
cm = json.load(open('results/component_miss_whu_neutral/summary.json', encoding='utf-8'))
for name in ('diff_neutral', 'btot_neutral'):
    m = cm['models'][name]
    put(f'whu.cm.{name}.n', m['n_complete_miss'], '{}')
    put(f'whu.cm.{name}.rate', m['complete_miss_rate'] * 100, '{:.1f}')
    put(f'whu.cm.{name}.fa', m['n_false_alarm_components'], '{}')
put('whu.cm.ngt', cm['models']['diff_neutral']['n_gt_components'], '{}')
pair = cm['pairs']['diff_neutral_vs_btot_neutral']['all']
put('whu.cm.only_diff_missed', pair['missed_by_diff_neutral_found_by_btot_neutral'], '{}')
put('whu.cm.only_btot_missed', pair['found_by_diff_neutral_missed_by_btot_neutral'], '{}')
put('whu.cm.mcnemar_p', pair['mcnemar_exact_p'], '{:.2f}')

# ---------------------------------------------------------------- zero-shot WHU-CD -> LEVIR-CD, earlier recipe (to be rerun)
zs = json.load(open('results/margin_gap/margin_gap.json', encoding='utf-8'))
zs.update(json.load(open('results/margin_gap_ops/margin_gap.json', encoding='utf-8')))
for name in ('diff', 'ssm', 's42', 's456'):
    m = zs[name]['LEVIR_CD']['margin']
    for key in ('auroc', 'ap'):
        put(f'zs.{name}.{key}', m[key], '{:.3f}')
    put(f'zs.{name}.oracle', m['oracle_iou'] * 100)
    put(f'zs.{name}.srcthr', m['iou_at_whu_thr'] * 100)
    put(f'zs.{name}.thr.oracle', m['oracle_thr'], '{:.1f}')
    put(f'zs.{name}.thr.src', m['whu_thr'], '{:.1f}')

# ---------------------------------------------------------------- post-training sweep on LEVIR-CD, matched recipe
# (analysis/final_sweep.sh; models missing from a directory are skipped)
MODELS = [(op, s) for op, _ in OPS for s in SEEDS] + [(ABLATION[0], 42)]
mname = {k: f'{k[0]}_levir_neutral_s{k[1]}' for k in MODELS}


def seed_avg(op, get):
    v = [get(k) for k in MODELS if k[0] == op]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# object-level complete misses (IV-C)
p = 'results/component_miss_levir/summary.json'
cml = json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {'models': {}, 'pairs': {}}
cmm = {k: cml['models'][n] for k, n in mname.items() if n in cml['models']}
if cmm:
    put('levir.cm.ngt', next(iter(cmm.values()))['n_gt_components'], '{}')
    small = next(iter(cmm.values()))['by_area']
    put('levir.cm.small.n', small[0]['n'], '{}')
for (op, s), m in cmm.items():
    put(f'levir.cm.{op}.s{s}.n', m['n_complete_miss'], '{}')
    put(f'levir.cm.{op}.s{s}.rate', m['complete_miss_rate'] * 100, '{:.1f}')
    put(f'levir.cm.{op}.s{s}.fa', m['n_false_alarm_components'], '{}')
    put(f'levir.cm.{op}.s{s}.small', m['by_area'][0]['complete_miss_rate'] * 100, '{:.0f}')
    put(f'levir.cm.{op}.s{s}.large', m['by_area'][-1]['complete_miss_rate'] * 100, '{:.1f}')
for op, _ in OPS + [ABLATION]:
    r = seed_avg(op, lambda k: cmm[k]['complete_miss_rate'] * 100 if k in cmm else None)
    if r is not None:
        put(f'levir.cm.{op}.mean.rate', r, '{:.2f}')
cm_rates = [m['complete_miss_rate'] * 100 for k, m in cmm.items() if k[0] != ABLATION[0]]
if cm_rates:
    put('levir.cm.rate.min', min(cm_rates), '{:.1f}')
    put('levir.cm.rate.max', max(cm_rates), '{:.1f}')


def put_p(key, p):
    """p-value for math mode: two significant digits, or m x 10^e below 0.001."""
    if p >= 0.001:
        numbers[key] = f'{p:.2g}'
    else:
        m, e = f'{p:.0e}'.split('e')
        numbers[key] = f'{m}\\times10^{{{int(e)}}}'


small = [m['by_area'][0]['complete_miss_rate'] * 100 for k, m in cmm.items() if k[0] != ABLATION[0]]
if small:
    put('levir.cm.small.min', min(small), '{:.0f}')
    put('levir.cm.small.max', max(small), '{:.0f}')


def cm_pair(a, b):
    """Discordant complete misses (missed by a only, missed by b only) and McNemar p, whichever order was stored."""
    na, nb = mname[a], mname[b]
    for x, y, flip in ((na, nb, False), (nb, na, True)):
        q = cml['pairs'].get(f'{x}_vs_{y}')
        if q:
            q = q['all']
            ma, mb = q[f'missed_by_{x}_found_by_{y}'], q[f'found_by_{x}_missed_by_{y}']
            return (mb, ma, q['mcnemar_exact_p']) if flip else (ma, mb, q['mcnemar_exact_p'])
    return None


# discordant misses between two seeds of one operator vs. between operators with a shared seed
disc_seed, disc_op = [], []
for op, _ in OPS:
    r = cm_pair((op, 42), (op, 123))
    if r:
        put(f'levir.cm.seed.{op}.a', r[0], '{}')
        put(f'levir.cm.seed.{op}.b', r[1], '{}')
        put_p(f'levir.cm.seed.{op}.p', r[2])
        disc_seed.append(r[0] + r[1])
op_p = {s: [] for s in SEEDS}
for op, _ in OPS[1:]:
    for s in SEEDS:
        r = cm_pair(('diff', s), (op, s))
        if r:
            put(f'levir.cm.op.{op}.s{s}.a', r[0], '{}')
            put(f'levir.cm.op.{op}.s{s}.b', r[1], '{}')
            put_p(f'levir.cm.op.{op}.s{s}.p', r[2])
            disc_op.append(r[0] + r[1])
            op_p[s].append(r[2])
for s, ps in op_p.items():
    if ps:
        put_p(f'levir.cm.op.s{s}.pmin', min(ps))
        put_p(f'levir.cm.op.s{s}.pmax', max(ps))
if disc_seed:
    put('levir.cm.disc.seed', float(np.mean(disc_seed)), '{:.0f}')
if disc_op:
    put('levir.cm.disc.op', float(np.mean(disc_op)), '{:.0f}')
r = cm_pair((ABLATION[0], 42), ('btot', 42))
if r:
    put('levir.cm.nobin.a', r[0], '{}')
    put('levir.cm.nobin.b', r[1], '{}')
    put_p('levir.cm.nobin.p', r[2])

# gate, branch-to-difference norm ratio, IoU with the branch removed at test time (IV-D)
br = {}
for k, n in mname.items():
    p = f'results/branch_ratio_levir/{n}.json'
    if os.path.isfile(p):
        br[k] = json.load(open(p, encoding='utf-8'))
for (op, s), d in br.items():
    put(f'levir.br.{op}.s{s}.drop', d['iou_drop_gamma0'] * 100)
    put(f'levir.br.{op}.s{s}.change', -d['iou_drop_gamma0'] * 100, '{:+.2f}')   # IoU change when the branch is removed
    put(f'levir.br.{op}.s{s}.iou0', d['iou_gamma0'] * 100)
for op, _ in OPS[1:] + [ABLATION]:
    ks = [k for k in br if k[0] == op]
    if not ks:
        continue
    g = np.concatenate([np.abs(br[k]['gamma']) for k in ks])
    ratio = np.concatenate([br[k]['branch_to_difference_norm_ratio'] for k in ks])
    put(f'levir.br.{op}.absgamma.min', g.min())
    put(f'levir.br.{op}.absgamma.max', g.max())
    put(f'levir.br.{op}.ratio.min', ratio.min())
    put(f'levir.br.{op}.ratio.max', ratio.max())
    put(f'levir.br.{op}.ratio.mean', ratio.mean())
    put(f'levir.br.{op}.drop', np.mean([br[k]['iou_drop_gamma0'] for k in ks]) * 100)
    put(f'levir.br.{op}.drop.max', max(br[k]['iou_drop_gamma0'] for k in ks) * 100)
    put(f'levir.br.{op}.iou0', np.mean([br[k]['iou_gamma0'] for k in ks]) * 100)
    put(f'levir.br.{op}.nseeds', len(ks), '{}')

# zero-shot LEVIR-CD -> WHU-CD from the logit margin (VII-A)
zs2 = {}
for k, n in mname.items():
    p = f'results/zeroshot_levir2whu/{n}.json'
    if os.path.isfile(p):
        zs2[k] = json.load(open(p, encoding='utf-8'))['target_metrics']
ZKEYS = (('auroc', '{:.3f}', 1), ('ap', '{:.3f}', 1), ('iou_at_0.5', '{:.2f}', 100), ('oracle_iou', '{:.2f}', 100),
         ('rate_matched_iou', '{:.2f}', 100), ('oracle_margin', '{:.1f}', 1))
for (op, s), m in zs2.items():
    for key, fmt, sc in ZKEYS:
        put(f'zs2.{op}.s{s}.{key.replace("_at_0.5", "")}', m[key] * sc, fmt)
for op, _ in OPS + [ABLATION]:
    for key, fmt, sc in ZKEYS:
        v = seed_avg(op, lambda k: zs2[k][key] * sc if k in zs2 else None)
        if v is not None:
            put(f'zs2.{op}.mean.{key.replace("_at_0.5", "")}', v, fmt)
zmain = {k: m for k, m in zs2.items() if k[0] != ABLATION[0]}
if zmain:
    for key, fmt, sc in ZKEYS:
        v = [m[key] * sc for m in zmain.values()]
        put(f'zs2.{key.replace("_at_0.5", "")}.min', min(v), fmt)
        put(f'zs2.{key.replace("_at_0.5", "")}.max', max(v), fmt)
    gap = [(m['oracle_iou'] - m['iou_at_0.5']) * 100 for m in zmain.values()]
    put('zs2.thrgap.max', max(gap))
    for op, _ in OPS:
        sr = [(zmain[(op, s)]['iou_at_0.5']) * 100 for s in SEEDS if (op, s) in zmain]
        if len(sr) == 2:
            put(f'zs2.{op}.seedrange', abs(sr[0] - sr[1]))

# nuisance perturbations of the second image, real pairs (VII-B); one robustness.json per machine
nz = {}
for tag in ('dyt', '4090d'):
    p = f'results/nuisance_levir_{tag}/robustness.json'
    if os.path.isfile(p):
        for n, d in json.load(open(p, encoding='utf-8')).items():
            nz[n] = d
PERTS = ('clean', 'shift2', 'shift4', 'shift8', 'bright0.7', 'gamma1.6', 'colorcast', 'blur1.5')
PNAME = {'clean': 'Clean', 'shift2': 'Shift 2 px', 'shift4': 'Shift 4 px', 'shift8': 'Shift 8 px',
         'bright0.7': 'Brightness 0.7', 'gamma1.6': 'Gamma 1.6', 'colorcast': 'Color cast', 'blur1.5': 'Blur 1.5'}
nzm = {k: nz[n] for k, n in mname.items() if n in nz}
nzmean = {}
for op, _ in OPS + [ABLATION]:
    for pt in PERTS:
        v = seed_avg(op, lambda k: nzm[k][f'real/{pt}']['iou'] * 100 if k in nzm else None)
        if v is not None:
            nzmean[(op, pt)] = v
            put(f'nz.{op}.{pt}', v)
gaps = []
for op, _ in OPS[1:]:
    for pt in PERTS:
        if (op, pt) in nzmean and ('diff', pt) in nzmean:
            g = nzmean[(op, pt)] - nzmean[('diff', pt)]
            put(f'nz.gap.{op}.{pt}', g, '{:+.2f}')
            gaps.append(g)
if gaps:
    put('nz.gap.min', min(gaps), '{:+.2f}')
    put('nz.gap.max', max(gaps), '{:+.2f}')
    noblur = [abs(nzmean[(op, pt)] - nzmean[('diff', pt)]) for op, _ in OPS[1:] for pt in PERTS
              if pt != 'blur1.5' and (op, pt) in nzmean and ('diff', pt) in nzmean]
    put('nz.gap.noblur.maxabs', max(noblur))
    s8 = [nzmean[(op, 'shift8')] for op, _ in OPS if (op, 'shift8') in nzmean]
    put('nz.shift8.min', min(s8))
    put('nz.shift8.max', max(s8))
nzseed = [abs(nzm[(op, 42)][f'real/{pt}']['iou'] - nzm[(op, 123)][f'real/{pt}']['iou']) * 100
          for op, _ in OPS for pt in PERTS if (op, 42) in nzm and (op, 123) in nzm]
if nzseed:
    put('nz.seedrange.max', max(nzseed))
    if ('diff', 42) in nzm and ('diff', 123) in nzm:
        put('nz.diff.blur.s42', nzm[('diff', 42)]['real/blur1.5']['iou'] * 100)
        put('nz.diff.blur.s123', nzm[('diff', 123)]['real/blur1.5']['iou'] * 100)
with open(os.path.join(OUT, 'gen', 'tab_nuisance.tex'), 'w', encoding='utf-8') as f:
    for pt in PERTS:
        f.write(PNAME[pt] + ' & ' + ' & '.join(f'{nzmean[(op, pt)]:.2f}' if (op, pt) in nzmean else '--'
                                              for op, _ in OPS) + ' \\\\\n')

with open(os.path.join(OUT, 'gen', 'numbers.tex'), 'w', encoding='utf-8') as f:
    f.write('% generated by analysis/paper_numbers.py -- do not edit by hand\n')
    for k in sorted(numbers):
        f.write(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{numbers[k]}}}\n')
print(f'{len(numbers)} numbers; LEVIR models: {sorted(levir)}')

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
short = {'diff': 'Diff', 'hbca': 'HBCA', 'ssm': 'SSM', 'btot': 'BTOT', 'btot_nodustbin': 'BTOT w/o bin'}

# operator differences with 95% image-bootstrap CIs on LEVIR-CD, against the one WHU-CD comparison
fig, ax = plt.subplots(figsize=(3.45, 1.9))
rows = [(f'LEVIR, {short[op]} $-$ Diff', *delta[(op, 'id')], 'C0') for op, _ in OPS[1:] if (op, 'id') in delta]
for view, tag in (('id', 'single pass'), ('tta', 'TTA')):
    rows.append((f'WHU, BTOT $-$ Diff, {tag}', float(numbers[f'whu.btotdiff.{view}']) / 100,
                 float(numbers[f'whu.btotdiff.{view}.lo']) / 100, float(numbers[f'whu.btotdiff.{view}.hi']) / 100, 'C3'))
for i, (lab, d, lo, hi, c) in enumerate(rows[::-1]):
    ax.errorbar(d * 100, i, xerr=[[(d - lo) * 100], [(hi - d) * 100]], fmt='o', ms=3, color=c, capsize=2, lw=0.9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel('IoU difference (points)')
ax.grid(axis='x', lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figs', 'forest.pdf'))
plt.close(fig)

# concentration of change pixels in the two test sets
fig, ax = plt.subplots(figsize=(3.45, 1.9))
for ds, g, c in (('WHU-CD', gw_px, 'C3'), ('LEVIR-CD', gt_px, 'C0')):
    s = np.sort(g[g > 0])[::-1]
    ax.plot(np.arange(1, len(s) + 1), np.cumsum(s) / s.sum() * 100, color=c, lw=1.1)
    x_lab, y_lab, ha = (1.15, 84, 'left') if ds == 'WHU-CD' else (1150, 8, 'right')
    ax.text(x_lab, y_lab, f'{ds}\n{len(s)} images with change', color=c, fontsize=7, ha=ha, va='center')
ax.set_xscale('log')
ax.set_xlabel('number of test images, largest change first')
ax.set_ylabel('share of change pixels (%)')
ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figs', 'tail.pdf'))
plt.close(fig)

# per-image error correlation matrix
labels = [f'{short[o]} s{s}' for o, s in keys]
n = len(keys)
lo = np.floor(R[~np.eye(n, dtype=bool)].min() * 100) / 100
M = np.ma.masked_where(np.triu(np.ones((n, n), dtype=bool)), R)     # symmetric: lower triangle only
fig, ax = plt.subplots(figsize=(3.45, 3.1))
im = ax.imshow(M, vmin=lo, vmax=1.0, cmap='viridis')
ax.set_xticks(range(n - 1))
ax.set_yticks(range(1, n))
ax.set_xticklabels(labels[:-1], rotation=45, ha='right')
ax.set_yticklabels(labels[1:])
ax.set_xlim(-0.5, n - 1.5)
ax.set_ylim(n - 0.5, 0.5)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(n):
    for j in range(i):
        ax.text(j, i, f'{R[i, j]:.3f}'.lstrip('0'), ha='center', va='center', fontsize=5.5,
                color='w' if R[i, j] < lo + 0.6 * (1 - lo) else 'k')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figs', 'similarity.pdf'))
plt.close(fig)

# claimed margins in the literature against the minimum detectable difference of each test split
fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.3))
for ax, m, mdd, c, title in ((axes[0], lev_m, mdd_range['levir'], 'C0', f'LEVIR-CD, official split ({len(lev_m)} papers)'),
                             (axes[1], whu_m, mdd_range['whu'], 'C3', f'WHU-CD, 6096/762/762 split ({len(whu_m)} papers)')):
    ax.axvspan(-mdd[1], mdd[1], color=c, alpha=0.10, lw=0)     # largest MDD over model pairs and resampling units
    ax.axvspan(-mdd[0], mdd[0], color=c, alpha=0.18, lw=0)     # smallest
    mdd = mdd[1]
    ax.axvline(0, color='k', lw=0.6, ls='--')
    jit = np.random.default_rng(1).uniform(-0.3, 0.3, len(m))
    ax.scatter(m, jit, s=9, color=c, edgecolor='none', alpha=0.85)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(title, fontsize=7.5, loc='left', pad=2)
    lim = max(abs(m).max(), mdd) * 1.12
    ax.set_xlim(-lim, lim)
    ax.grid(axis='x', lw=0.3, alpha=0.5)
axes[1].set_xlabel('claimed IoU margin over the strongest competitor (points)')
fig.tight_layout(h_pad=0.6)
fig.savefig(os.path.join(OUT, 'figs', 'audit.pdf'))
plt.close(fig)
print('figures written')
