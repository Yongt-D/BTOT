"""How much does the operator-specific branch contribute at test time?

The residual operators (HBCA, SSM, BTOT) compute Z = B + gamma * Phi at every pyramid level, where B is the
difference term and Phi the operator-specific evidence after fusion. For each checkpoint this reports, per level,
the learned gamma and the ratio ||gamma * Phi|| / ||B|| (L2 norms over C, H, W per test image, averaged over the
test split), and the single-pass test IoU @0.5 of the model as trained and with every gamma set to 0, i.e. with
the operator branch removed at test time. For HBCA, B is |F1 - F2| and the top-down cross-scale path stays active.
The difference operator has no such branch and is skipped.

Usage (repo root): python analysis/branch_ratio.py OUT_DIR name=CKPT [name=CKPT ...] [--max_images N]
Caches OUT_DIR/<name>.json.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_component_miss as E  # noqa: E402

THRESHOLD = 0.5


def residual_levels(diff_module):
    """[(level module, module whose output is B or None, module whose output is Phi)] for residual operators."""
    if hasattr(diff_module, 'blocks'):                       # BTOT / SSM: Z = base + gamma * fuse(...)
        return [(b, b.base, b.fuse) for b in diff_module.blocks]
    if hasattr(diff_module, 'hbca_modules'):                 # HBCA: Z = |F1 - F2| + gamma * refine(...)
        return [(m, None, m.refine) for m in diff_module.hbca_modules]
    if hasattr(diff_module, 'shared_hbca'):
        return [(diff_module.shared_hbca, None, diff_module.shared_hbca.refine)]
    return []


def run_split(model, loader, device, levels, max_images, measure):
    captured = {}
    hooks = []
    if measure:
        for i, (lvl, base_mod, phi_mod) in enumerate(levels):
            if base_mod is None:
                hooks.append(lvl.register_forward_pre_hook(
                    lambda m, inp, i=i: captured.__setitem__(('B', i), (inp[0] - inp[1]).abs())))
            else:
                hooks.append(base_mod.register_forward_hook(lambda m, inp, out, i=i: captured.__setitem__(('B', i), out)))
            hooks.append(phi_mod.register_forward_hook(lambda m, inp, out, i=i: captured.__setitem__(('P', i), out)))
    ratio_sum = np.zeros(len(levels))
    n_ratio = 0
    tp = fp = fn = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            img1, img2 = batch['img1'].to(device), batch['img2'].to(device)
            label = batch['label'].to(device) == 1
            prob = F.softmax(model.forward(img1, img2), dim=1)[:, 1]
            pred = prob >= THRESHOLD
            tp += (pred & label).sum().item()
            fp += (pred & ~label).sum().item()
            fn += (~pred & label).sum().item()
            if measure:
                for i, (lvl, _, _) in enumerate(levels):
                    b = captured[('B', i)].flatten(1).norm(dim=1)
                    p = (lvl.gamma.abs() * captured[('P', i)]).flatten(1).norm(dim=1)
                    ratio_sum[i] += (p / b.clamp_min(1e-12)).sum().item()
                n_ratio += img1.shape[0]
            n += img1.shape[0]
            if max_images and n >= max_images:
                break
    for h in hooks:
        h.remove()
    iou = tp / max(tp + fp + fn, 1)
    return iou, (ratio_sum / max(n_ratio, 1)).tolist(), n


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
        model = trainer.model.eval()
        levels = residual_levels(model.diff_module)
        if not levels:
            print(f'{name}: no residual operator branch (difference operator), skipped', flush=True)
            del trainer
            continue
        _, loader = E.build_test_loader(cfg, 8, 4)
        iou_full, ratios, n = run_split(model, loader, trainer.device, levels, max_images, measure=True)
        gammas = [float(lvl.gamma.item()) for lvl, _, _ in levels]
        saved = [lvl.gamma.detach().clone() for lvl, _, _ in levels]
        with torch.no_grad():
            for lvl, _, _ in levels:
                lvl.gamma.zero_()
        iou_nobranch, _, _ = run_split(model, loader, trainer.device, levels, max_images, measure=False)
        with torch.no_grad():
            for (lvl, _, _), g in zip(levels, saved):
                lvl.gamma.copy_(g)
        res = {'name': name, 'ckpt': ckpt, 'epoch': epoch, 'dataset': cfg['data']['dataset'], 'n_images': n,
               'gamma': gammas, 'branch_to_difference_norm_ratio': ratios,
               'iou': iou_full, 'iou_gamma0': iou_nobranch, 'iou_drop_gamma0': iou_full - iou_nobranch}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1)
        print(f"{name}: gamma {' '.join(f'{g:+.3f}' for g in gammas)} | ||gPhi||/||B|| "
              f"{' '.join(f'{r:.3f}' for r in ratios)} | IoU {iou_full * 100:.2f} -> gamma=0 {iou_nobranch * 100:.2f} "
              f"({n} images)", flush=True)
        del trainer, model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
