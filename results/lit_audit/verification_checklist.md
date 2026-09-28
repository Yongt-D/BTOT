# Audit rows for manual verification (2026-09-28)

All 101 rows were re-extracted blind from the original papers (blind_reextraction.json); 97 agree with claims.csv on every number and on the competitor (blind_comparison.json). The rows below are the ones that decide the paper's audit statements, plus the open decisions. Check each value against the table named in the "Where" column.

| Method | Dataset | Why check | Proposed IoU / F1 | Strongest competitor IoU / F1 | Margin IoU | Where (table, version) | Link |
|---|---|---|---|---|---|---|---|
| SemDINO (2026) | LEVIR-CD | highest proposed IoU on LEVIR (86.51 in the paper) | 86.51 / 92.97 | ChangeDINO 85.72 / 92.31 | 0.79 | Table VI, arXiv 2606.09772v3 PDF p.13 | https://arxiv.org/abs/2606.09772 |
| AFCF3D-Net (2023) | WHU-CD | largest WHU margins (claim: all 13 below MDD 7.86) | 87.93 / 93.58 | LightCDNet 84.30 / 91.50 | 3.63 | Table I, arXiv 2302.05109 PDF p.8 | https://arxiv.org/abs/2302.05109 |
| DDLNet (2024) | WHU-CD | largest WHU margins (claim: all 13 below MDD 7.86) | 82.75 / 90.56 | USSFC-Net 79.21 / 88.40 | 3.54 | Table I, arXiv 2406.13606v1 PDF p.4 | https://arxiv.org/abs/2406.13606 |
| CDMaskFormer (2024) | WHU-CD | largest WHU margins (claim: all 13 below MDD 7.86) | 84.44 / 91.56 | SARASNet 81.08 / 89.55 | 3.36 | Table 2, arXiv 2406.15320v1 PDF p.8 | https://arxiv.org/abs/2406.15320 |
| Dsfer-Net (2024) | LEVIR-CD | largest LEVIR margins | 83.09 / 90.76 | FrNet 79.87 / 88.81 | 3.22 | Table I, arXiv 2304.01101v2 PDF p.10 | https://arxiv.org/abs/2304.01101 |
| HCGMNet (2023) | LEVIR-CD | largest LEVIR margins | 84.79 / 91.77 | ChangeFormer 82.21 / 90.20 | 2.58 | Table I, arXiv 2302.10420v2 PDF p.3 (read from rendered page image) | https://arxiv.org/abs/2302.10420 |
| STNet (2023) | LEVIR-CD | below LEVIR MDD 0.29 | 82.09 / 90.52 | ChangeFormer 82.48 / 90.40 | -0.39 | Table I, arXiv 2304.11422v1 PDF p.4 | https://arxiv.org/abs/2304.11422 |
| TransY-Net (2023) | LEVIR-CD | below LEVIR MDD 0.29 | 83.64 / 91.90 | UVACD 83.98 / 91.30 | -0.34 | Table I, arXiv 2310.14214v1 PDF p.7 | https://arxiv.org/abs/2310.14214 |
| AdaDINO (2026) | LEVIR-CD | below LEVIR MDD 0.29 | 85.79 / 92.35 | FAEWNet 85.89 / 92.41 | -0.10 | Table 1, arXiv 2608.07982v1 PDF p.6 | https://arxiv.org/abs/2608.07982 |
| FCCDN (2022) | LEVIR-CD | counted as reporting repeated runs / significance (claim: 4 of 62) | 85.69 / 92.29 | CEECNetV2 84.89 / 91.83 | 0.80 | Table 3, arXiv 2105.10860v2 PDF p.25 | https://arxiv.org/abs/2105.10860 |
| MCTNet (2023) | LEVIR-CD | counted as reporting repeated runs / significance (claim: 4 of 62) | - / 90.91 | BIT - / 89.35 | - | Table II, arXiv 2210.07601v3 PDF p.4 | https://arxiv.org/abs/2210.07601 |
| SAM-CD (2024) | LEVIR-CD | counted as using TTA (claim: 1 of 62) | 91.68 / 95.50 | CGNet 91.58 / 95.44 | 0.10 | Table V, arXiv 2309.01429v4 PDF p.7 | https://arxiv.org/abs/2309.01429 |
| CDMaskFormer (2024) | LEVIR-CD | counted as reporting repeated runs / significance (claim: 4 of 62) | 82.92 / 90.66 | SARASNet 82.55 / 90.44 | 0.37 | Table 2, arXiv 2406.15320v1 PDF p.8 | https://arxiv.org/abs/2406.15320 |
| M-CD (2025) | LEVIR-CD | counted as reporting repeated runs / significance (claim: 4 of 62) | 85.0 / 92.1 | DDPM-CD 83.3 / 90.9 | 1.70 | Table 1, arXiv 2407.06839v1 PDF p.7 | https://arxiv.org/abs/2407.06839 |
| DDPM-CD (2025) | LEVIR-CD | DONE 2026-09-28: now uses WACV Table 1 (IN1k 89.82 / 94.39 vs ChangeFormer 82.48 / 90.40, margin 7.34); split unstated, so excluded from the official-split statistics | 89.82 / 94.39 | ChangeFormer 82.48 / 90.40 | 7.34 | Table 1, WACV 2025 CVF open-access PDF p.7 | https://openaccess.thecvf.com/content/WACV2025/html/Bandara_DDPM-CD_Denoising_Diffusion_Probabilistic_Models_as_Feature_Extractors_for_Remote_WACV_2025_paper.html |
| SRC-Net (2024) | LEVIR-CD | DONE 2026-09-28: published version describes only train/validation sets; moved to the unstated-split group | 85.60 / 92.24 | BIT 84.66 / 91.69 | 0.94 | Table I, IEEE JSTARS published PDF p.8 | https://doi.org/10.1109/JSTARS.2024.3411622 |
| SiamixFormer (2023) | LEVIR-CD | competitor row with impossible IoU > F1 skipped | 85.38 / 91.58 | UVACD 83.98 / 91.30 | 1.40 | Table 4, arXiv 2208.00657v2 PDF p.11 | https://arxiv.org/abs/2208.00657 |
