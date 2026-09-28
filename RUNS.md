# Trained models

All runs use the matched recipe of the paper (Section III-C). HBCA and SSM use the difference recipe file with the operator flag. "Best epoch" is the checkpoint selected by validation IoU.

| Result name | Dataset | Operator | Seed | Hardware | Best epoch | Config and flags |
|---|---|---|---|---|---|---|
| diff_levir_neutral_s42 | LEVIR-CD | difference | 42 | RTX 3090, PyTorch 2.5.1 | 131 | `diff_levir_recipe*.yaml`  `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| diff_levir_neutral_s123 | LEVIR-CD | difference | 123 | RTX 4090D, PyTorch 2.4.0 | 131 | `diff_levir_recipe*.yaml`  `--seed 123 --tversky_beta 0.5 --focal_alpha 0.5` |
| hbca_levir_neutral_s42 | LEVIR-CD | HBCA | 42 | RTX 3090, PyTorch 2.5.1 | 113 | `diff_levir_recipe*.yaml` --hbca `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| hbca_levir_neutral_s123 | LEVIR-CD | HBCA | 123 | RTX 3090, PyTorch 2.5.1 | 136 | `diff_levir_recipe*.yaml` --hbca `--seed 123 --tversky_beta 0.5 --focal_alpha 0.5` |
| ssm_levir_neutral_s42 | LEVIR-CD | SSM | 42 | RTX 4090D, PyTorch 2.4.0 | 138 | `diff_levir_recipe*.yaml` --ssm `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| ssm_levir_neutral_s123 | LEVIR-CD | SSM | 123 | RTX 4090D, PyTorch 2.4.0 | 106 | `diff_levir_recipe*.yaml` --ssm `--seed 123 --tversky_beta 0.5 --focal_alpha 0.5` |
| btot_levir_neutral_s42 | LEVIR-CD | BTOT | 42 | RTX 4090D, PyTorch 2.4.0 | 146 | `btot_levir_recipe*.yaml`  `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| btot_levir_neutral_s123 | LEVIR-CD | BTOT | 123 | RTX 4090D, PyTorch 2.4.0 | 135 | `btot_levir_recipe*.yaml`  `--seed 123 --tversky_beta 0.5 --focal_alpha 0.5` |
| btot_nodustbin_levir_neutral_s42 | LEVIR-CD | BTOT without dustbin | 42 | RTX 4090D, PyTorch 2.4.0 | 146 | `btot_levir_recipe*.yaml` --ot_no_dustbin `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| diff_neutral | WHU-CD | difference | 42 | RTX 4090D, PyTorch 2.4.0 | 150 | `diff_whu_recipe*.yaml`  `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
| diff_neutral_s123 | WHU-CD | difference | 123 | RTX 4090D, PyTorch 2.4.0 | 121 | `diff_whu_recipe*.yaml`  `--seed 123 --tversky_beta 0.5 --focal_alpha 0.5` |
| btot_neutral | WHU-CD | BTOT | 42 | L40S, PyTorch 2.4 | 141 | `btot_whu_recipe*.yaml`  `--seed 42 --tversky_beta 0.5 --focal_alpha 0.5` |
