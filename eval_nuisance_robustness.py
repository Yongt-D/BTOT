"""
eval_nuisance_robustness.py -- sensitivity of interaction operators to nuisance temporal variation.

Two protocols on the test split, single-pass, fixed threshold:
  real       genuine bi-temporal pairs with a controlled nuisance applied to the T2 image
             (ground truth unchanged): IoU/P/R, complete-miss rate, false-alarm components.
  unchanged  pseudo-unchanged pairs (T1, nuisance(T1)): the true change map is empty, so every
             predicted pixel is a false alarm produced by the nuisance alone.
Nuisances: translation (misregistration), brightness gain, gamma, per-channel colour cast, blur.

Usage (repo root):
    python eval_nuisance_robustness.py --gpu 0 \
        --ckpt diff=checkpoints/abl_whu_diff_s42_seed42/whu_cd/best.pth \
        --ckpt btot=checkpoints/v6btot_whu_s42_seed42/whu_cd/best.pth \
        --out_dir results/nuisance_whu
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from eval_component_miss import analyse_image, build_test_loader, load_model

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to01(x):
    return (x * STD.to(x) + MEAN.to(x)).clamp(0, 1)


def _from01(x):
    return (x - MEAN.to(x)) / STD.to(x)


def shift(x, s):
    if s == 0:
        return x
    return F.pad(x, (s, 0, s, 0), mode='reflect')[..., :x.shape[-2], :x.shape[-1]]


def gaussian_blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='reflect'), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def _gain(rgb):
    def f(x):
        g = torch.tensor(rgb, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return _from01((_to01(x) * g).clamp(0, 1))
    return f


NUISANCES = {
    'clean': lambda x: x,
    'shift2': lambda x: shift(x, 2),
    'shift4': lambda x: shift(x, 4),
    'shift8': lambda x: shift(x, 8),
    'bright0.7': _gain([0.7, 0.7, 0.7]),
    'bright1.3': _gain([1.3, 1.3, 1.3]),
    'gamma0.6': lambda x: _from01(_to01(x) ** 0.6),
    'gamma1.6': lambda x: _from01(_to01(x) ** 1.6),
    'colorcast': _gain([1.15, 1.0, 0.85]),
    'blur1.5': lambda x: gaussian_blur(x, 1.5),
}


def evaluate(trainer, loader, nuisance, protocol, threshold, max_images):
    fn_t = NUISANCES[nuisance]
    tp = fp = fn = 0
    n_comp = n_miss = n_pred_comp = n_fa_comp = 0
    fa_pixels = total_pixels = fa_images = n = 0
    with torch.no_grad():
        for batch in loader:
            img1 = batch['img1'].to(trainer.device, non_blocking=True)
            if protocol == 'real':
                img2 = fn_t(batch['img2'].to(trainer.device, non_blocking=True))
                label = (batch['label'].numpy() == 1).astype(np.uint8)
            else:
                img2 = fn_t(img1)
                label = np.zeros(batch['label'].shape, np.uint8)
            pred, _ = trainer.model.predict(img1, img2, threshold=threshold)
            pred = pred.cpu().numpy().astype(np.uint8)

            if protocol == 'real':
                tp += int(((pred == 1) & (label == 1)).sum())
                fp += int(((pred == 1) & (label == 0)).sum())
                fn += int(((pred == 0) & (label == 1)).sum())
                for b in range(pred.shape[0]):
                    area, hit, parea, pov = analyse_image(label[b], pred[b])
                    n_comp += area.size
                    n_miss += int((hit == 0).sum())
                    n_pred_comp += parea.size
                    n_fa_comp += int((pov == 0).sum())
            else:
                fa_pixels += int(pred.sum())
                total_pixels += pred.size
                fa_images += int((pred.reshape(pred.shape[0], -1).max(1) > 0).sum())
            n += pred.shape[0]
            if max_images and n >= max_images:
                break

    if protocol == 'real':
        return {
            'iou': tp / max(tp + fp + fn, 1), 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'complete_miss_rate': n_miss / max(n_comp, 1), 'n_gt_components': n_comp,
            'false_alarm_component_rate': n_fa_comp / max(n_pred_comp, 1), 'n_false_alarm_components': n_fa_comp,
            'n_images': n,
        }
    return {
        'false_alarm_pixel_rate': fa_pixels / max(total_pixels, 1),
        'images_with_false_alarm': fa_images / max(n, 1),
        'n_images': n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True, help='name=path/to/best.pth (repeatable)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--nuisance', action='append', default=None, help=f'subset of {list(NUISANCES)}')
    p.add_argument('--protocol', choices=['real', 'unchanged', 'both'], default='both')
    p.add_argument('--max_images', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nuisances = args.nuisance or list(NUISANCES)
    protocols = ['real', 'unchanged'] if args.protocol == 'both' else [args.protocol]
    out_json = os.path.join(args.out_dir, 'robustness.json')
    results = json.load(open(out_json, encoding='utf-8')) if os.path.isfile(out_json) else {}

    for spec in args.ckpt:
        name, ckpt = spec.split('=', 1)
        trainer, cfg, _ = load_model(ckpt, args.gpu, args.batch_size)
        _, loader = build_test_loader(cfg, args.batch_size, args.workers)
        results.setdefault(name, {})
        for proto in protocols:
            for nz in nuisances:
                key = f'{proto}/{nz}'
                if key in results[name]:
                    continue
                t0 = time.time()
                results[name][key] = evaluate(trainer, loader, nz, proto, args.threshold, args.max_images)
                print(f'[{name}] {key}: {results[name][key]} ({time.time() - t0:.0f}s)', flush=True)
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
        del trainer
        torch.cuda.empty_cache()

    print('\n' + '=' * 104)
    print(f'{"model":8s} {"nuisance":10s} | {"IoU":>6s} {"Rec":>6s} {"CMR":>6s} {"FAcomp%":>8s} | '
          f'{"unchanged: FA px (per-mille)":>28s} {"img w/ FA":>9s}')
    for name, res in results.items():
        for nz in nuisances:
            r, u = res.get(f'real/{nz}'), res.get(f'unchanged/{nz}')
            left = (f'{r["iou"] * 100:6.2f} {r["recall"] * 100:6.2f} {r["complete_miss_rate"] * 100:6.2f} '
                    f'{r["false_alarm_component_rate"] * 100:8.2f}') if r else ' ' * 30
            right = (f'{u["false_alarm_pixel_rate"] * 1000:28.3f} {u["images_with_false_alarm"] * 100:8.1f}%') if u else ''
            print(f'{name:8s} {nz:10s} | {left} | {right}')
    print('=' * 104)
    print(f'results -> {out_json}')


if __name__ == '__main__':
    main()
