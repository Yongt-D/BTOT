"""Change probabilities of selected test images for the qualitative figure.

For every model, saves the single-pass probability map and the flip-TTA probability map (mean of the
identity, flip-W, flip-H and flip-HW views, as in tta_gap.py) of each listed image as 8-bit PNGs:
OUT_DIR/<name>/<image stem>_id.png and _tta.png (value = probability x 255; the paper thresholds at 0.5).

Usage (repo root): python analysis/qual_preds.py OUT_DIR LIST_FILE name=CKPT [name=CKPT ...]
LIST_FILE holds one test image file name per line, as in data_dir/<dataset>/list/test.txt.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_component_miss as E  # noqa: E402

VIEWS = {'id': [], 'flipW': [3], 'flipH': [2], 'flipHW': [2, 3]}


def run(name, ckpt, list_file, names, out_dir):
    dst = os.path.join(out_dir, name)
    os.makedirs(dst, exist_ok=True)
    if all(os.path.isfile(os.path.join(dst, os.path.splitext(n)[0] + '_tta.png')) for n in names):
        print(f'[{name}] cached', flush=True)
        return
    trainer, cfg, epoch = E.load_model(ckpt, 0, 1)
    cfg['data']['test_list'] = os.path.abspath(list_file)
    ds, loader = E.build_test_loader(cfg, 1, 0)
    assert len(ds) == len(names), (len(ds), len(names))
    model, dev = trainer.model, trainer.device
    with torch.no_grad():
        for n, batch in zip(names, loader):
            img1, img2 = batch['img1'].to(dev), batch['img2'].to(dev)
            probs = []
            for v, dims in VIEWS.items():
                if dims:
                    p = F.softmax(model.forward(torch.flip(img1, dims), torch.flip(img2, dims)), dim=1)[:, 1]
                    probs.append(torch.flip(p, [d - 1 for d in dims]))
                else:
                    probs.append(F.softmax(model.forward(img1, img2), dim=1)[:, 1])
            stem = os.path.splitext(n)[0]
            for tag, p in (('id', probs[0]), ('tta', torch.stack(probs).mean(0))):
                arr = (p[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(dst, f'{stem}_{tag}.png'))
    print(f'[{name}] epoch {epoch}: {len(names)} images written', flush=True)
    del trainer, model
    torch.cuda.empty_cache()


def main():
    out_dir, list_file = sys.argv[1], sys.argv[2]
    names = [l.strip() for l in open(list_file, encoding='utf-8') if l.strip()]
    for spec in sys.argv[3:]:
        name, ckpt = spec.split('=', 1)
        run(name, ckpt, list_file, names, out_dir)


if __name__ == '__main__':
    main()
