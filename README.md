# On the Limited Role of Bitemporal Fusion Operators in Remote Sensing Change Detection with Vision Foundation Models

Code, configurations, data splits, and per-image results for the paper of the same title
(Deng, Lei, Zhang, Peng, and Li; submitted to IEEE JSTARS).

**About this repository.** It first described BTOT, a bitemporal optimal-transport operator that we proposed as an
improvement over feature differencing. When we compared the operator with a difference module, window cross-attention
(HBCA), and a selective state-space scan (SSM) under one backbone, one training recipe, and two seeds each, its
advantage disappeared. The paper reports that controlled comparison and what it implies for evaluating change detection
models. The earlier claims about BTOT (fewer complete misses, higher recall) do not hold under matched conditions and
are withdrawn.

## Main result (LEVIR-CD, single pass, IoU %)

| Operator | Seed 42 | Seed 123 | Mean | Difference to the difference operator [95% image-bootstrap CI] |
|---|---|---|---|---|
| Difference | 85.24 | 85.09 | 85.16 | reference |
| Cross-attention (HBCA) | 85.25 | 85.25 | 85.25 | +0.09 [-0.13, +0.29] |
| State-space (SSM) | 85.32 | 85.03 | 85.17 | +0.01 [-0.24, +0.24] |
| Optimal transport (BTOT) | 85.40 | 85.04 | 85.22 | +0.05 [-0.12, +0.21] |
| BTOT without dustbin | 85.01 | - | 85.01 | -0.21 [-0.50, +0.07] vs. BTOT |

Every interval includes zero, and the intervals that also include training variance are wider still. Replacing the
operator changes per-image errors and completely missed objects no more than replacing the seed, and no operator is
more robust to perturbations of the second image or transfers better to WHU-CD than the difference module.
See the paper for the power analysis of the WHU-CD test split and the audit of 62 recent papers.

## Contents

| Path | Content |
|---|---|
| `dinobcd/` | Model (shared architecture and the four operators), datasets, losses |
| `dinobcd/configs/rev_jstars/` | The matched training recipe for LEVIR-CD and WHU-CD |
| `train_dinobcd.py` | Training |
| `eval_component_miss.py` | Object-level complete misses and McNemar tests |
| `eval_nuisance_robustness.py` | Perturbations of the second image (shift, brightness, gamma, color cast, blur) |
| `analysis/tta_gap.py` | Single pass, flipped views, and TTA, with per-image confusion counts |
| `analysis/tta_gap_boot.py` | Paired image bootstrap between two models |
| `analysis/branch_ratio.py` | Learned gates, branch norm ratio, and IoU with the operator branch removed |
| `analysis/zeroshot_eval.py` | Zero-shot transfer from the logit margin |
| `analysis/final_sweep.sh` | Runs the four post-training evaluations above |
| `analysis/paper_numbers.py` | Recomputes every number, table, and statistics figure of the paper from `results/` |
| `splits/` | Train/validation/test lists for LEVIR-CD and WHU-CD (256x256 patches) |
| `results/` | All cached evaluation outputs used in the paper |
| `results/per_image_csv/` | Per-image confusion counts of every model as CSV |
| `results/lit_audit/` | The literature audit (62 papers) with the extraction rules, two independent extractions (`claims.csv`, `blind_reextraction.json`), and the check of all 50 formally published papers against their published tables (`published_version_check.json`) |
| `RUNS.md` | Hardware, software, seed, and checkpoint epoch of every trained model |

## Reproducing the numbers of the paper

No GPU or dataset is needed for this step:

```bash
pip install numpy scipy matplotlib
python analysis/paper_numbers.py 5000
```

It writes `paper_outputs/gen/numbers.tex`, the table bodies, and the statistics figures, and reproduces the values in
the paper exactly.

## Per-image results

`results/per_image_csv/<dataset>/<model>.csv` has one row per test image in the order of `splits/<dataset>/test.txt`.
Columns give true-positive, false-positive, and false-negative pixels at a threshold of 0.5 for the original image
(`original_*`), its horizontally, vertically, and doubly flipped copies (`flip_w_*`, `flip_h_*`, `flip_hw_*`), and
flip test-time augmentation (`tta_*`). The pooled IoU of a model is `sum(tp) / sum(tp + fp + fn)` over a column
group, and any paired comparison can be recomputed by resampling rows. Every row was checked against the reference
masks (TP + FN equals the number of change pixels in the image). The same counts are stored as NumPy arrays in
`results/tta_gap_levir_img/` and `results/tta_gap_whu_img/`.

## Training and evaluation

**Environment.** Python 3 with PyTorch 2.4 or 2.5.1 (see `requirements.txt`). Runs were split across an
RTX 3090 and an RTX 4090D; `RUNS.md` lists each one.

**Backbone.** Clone [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) into `./dinov3` and place
the ViT-L/16 SAT-493M weights at `dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`. The weights are
distributed by Meta under the DINOv3 license and are not included here.

**Data.** Obtain LEVIR-CD and WHU-CD from their providers in 256x256 patches and arrange each as
`data_dir/<LEVIR-CD|WHU-CD>/{A,B,label,list}`, copying the lists from `splits/`, which name the patches used in the
paper. The WHU-CD patches form a non-overlapping 127x60 tiling of the scene, and no test patch duplicates or overlaps a
training patch.

**Training** (seed 42 shown; the paper also uses seed 123):

```bash
R="--tversky_beta 0.5 --focal_alpha 0.5 --seed 42 --gpu 0"
python train_dinobcd.py --config dinobcd/configs/rev_jstars/diff_levir_recipe_slim.yaml $R --exp_name diff_levir
python train_dinobcd.py --config dinobcd/configs/rev_jstars/diff_levir_recipe_slim.yaml --hbca $R --exp_name hbca_levir
python train_dinobcd.py --config dinobcd/configs/rev_jstars/diff_levir_recipe_slim.yaml --ssm $R --exp_name ssm_levir
python train_dinobcd.py --config dinobcd/configs/rev_jstars/btot_levir_recipe_slim.yaml $R --exp_name btot_levir
python train_dinobcd.py --config dinobcd/configs/rev_jstars/btot_levir_recipe_slim.yaml --ot_no_dustbin $R --exp_name btot_nodustbin_levir
```

`--tversky_beta 0.5 --focal_alpha 0.5` gives the neutral loss used for every model in the paper.

**Evaluation** (from the repository root):

```bash
PYTHONPATH=. python analysis/tta_gap.py results/tta_gap_levir_img NAME=checkpoints/<run>/levir_cd/best.pth
python analysis/tta_gap_boot.py results/tta_gap_levir_img NAME_A NAME_B 5000 splits/LEVIR-CD/test.txt
bash analysis/final_sweep.sh python TAG NAME=checkpoints/<run>/levir_cd/best.pth [...]
```

Trained checkpoints are not released. The per-image results in `results/` are enough to verify every number in the
paper, and the configurations and seeds above retrain each model.

## Citation

```bibtex
@article{deng2026fusion,
  title   = {On the Limited Role of Bitemporal Fusion Operators in Remote Sensing Change Detection
             with Vision Foundation Models},
  author  = {Deng, Yongtao and Lei, Dajiang and Zhang, Liping and Peng, Yidong and Li, Weisheng},
  journal = {Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The DINOv3 backbone and its weights are subject to Meta's
DINOv3 license, and LEVIR-CD and WHU-CD to the terms of their providers.
