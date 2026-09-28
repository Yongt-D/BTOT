# Literature audit: claimed margins on LEVIR-CD and WHU-CD (2022-2026)

Generated 2026-09-21. Companion files: `claims.csv` (one row per paper x dataset) and `refs.bib`.

## Key numbers

- 62 papers, 101 rows (LEVIR-CD 59, WHU-CD 42).
- LEVIR-CD, IoU margin over the strongest listed competitor: median **0.80** (IQR 0.25 to 1.25, n=50). 58% of margins are below 1.0, 42% below 0.5, 24% below our 0.2 bootstrap half-width, and 34% below the 0.4 seed spread. F1 margin: median 0.59 (IQR 0.22 to 0.88, n=58).
- WHU-CD, IoU margin: median **1.60** (IQR 0.74 to 2.66, n=39). 38% of margins are below 1.0, 13% below 0.5, and 92% below our 5-point bootstrap half-width. F1 margin: median 0.94 (IQR 0.43 to 1.59, n=41).
- Variance reporting: 4/62 papers report any repeated runs, std or significance test. Only 2 report a dispersion estimate or test, and none gives a CI.
- Test-time augmentation: 1/62 papers use it. 3 say explicitly that they do not, and 58 do not mention it.

## What was recorded

- Supervised binary change-detection papers (published 2022-2026, or arXiv preprints from that period) whose own main comparison table reports LEVIR-CD and/or WHU-CD. Every number was read from the paper itself: arXiv PDFs (text extraction plus a rendered image of the table page to confirm row alignment), open-access IEEE PDFs (HANet, USSFC-Net), or MDPI HTML tables (six Remote Sensing papers).
- **Proposed** = the best variant of the proposed method in that table (e.g. MambaBCD-Base, SGSLN/512). **Strongest competitor** = the competing method with the highest IoU in the same table. The authors' own ablation baselines and variants are excluded; earlier methods from the same group are kept. If the table has no IoU, the strongest competitor is chosen by F1. `margin_iou` = proposed IoU - competitor IoU; `margin_f1` = proposed F1 - F1 of that same competitor.
- Values are copied as printed. Tables printed as fractions (0.911) were converted to percent. `metric_type` flags tables whose IoU/F1 is not the change-class metric. `f1_iou_consistency` checks F1 against 2*IoU/(1+IoU), which must hold for dataset-level change-class metrics.

## Counts

| | papers | rows | rows with IoU margin (change-class) | rows with F1 margin (change-class) |
|---|---|---|---|---|
| LEVIR-CD | 59 | 59 | 50 | 58 |
| WHU-CD | 42 | 42 | 39 | 41 |
| **total** | **62** | **101** | | |

39 papers report both datasets. Venues: IEEE TGRS 21, IEEE JSTARS 6, Remote Sensing (MDPI) 6, arXiv preprint 4, IEEE GRSL 2, IGARSS 2024 2, Pattern Recognition 2, WACV 2025 2, AAAI 2023 1, AAAI 2026 1, ACCV 2022 (LNCS) 1, FUSION 2023 1, Geo-spatial Information Science 1, ICIP 2026 1, ICME 2023 1, ICME 2024 1, IEEE TPAMI 1, IGARSS 2022 1, IGARSS 2023 1, IJCAI 2023 1, ISPRS Annals 1, ISPRS P&RS 1, Int. J. Remote Sensing 1, Neural Computing and Applications 1, Scientific Reports 1.

## Headline statistics (change-class metrics only)

| dataset | metric | n | median | IQR (Q1 to Q3) | min to max | < 1.0 | < 0.5 | <= 0 |
|---|---|---|---|---|---|---|---|---|
| LEVIR-CD | margin_iou | 50 | 0.80 | 0.25 to 1.25 | -1.40 to 3.22 | 29/50 (58%) | 21/50 (42%) | 4/50 (8%) |
| LEVIR-CD | margin_f1 | 58 | 0.59 | 0.22 to 0.88 | -0.85 to 2.00 | 45/58 (78%) | 24/58 (41%) | 2/58 (3%) |
| WHU-CD | margin_iou | 39 | 1.60 | 0.74 to 2.66 | -1.52 to 12.35 | 15/39 (38%) | 5/39 (13%) | 3/39 (8%) |
| WHU-CD | margin_f1 | 41 | 0.94 | 0.43 to 1.59 | -1.43 to 7.76 | 22/41 (54%) | 14/41 (34%) | 5/41 (12%) |

Sensitivity, LEVIR-CD: adding the 3 row(s) whose IoU column is class-mean or inconsistent (SAM-CD, MaskCD, EHCTNet) gives margin_iou median 0.80, IQR 0.23 to 1.36 (n=53).
Sensitivity, WHU-CD: adding the 1 row(s) whose IoU column is class-mean or inconsistent (SAM-CD) gives margin_iou median 1.46, IQR 0.76 to 2.53 (n=40).

## Against our measured noise band

- **LEVIR-CD** (our paired image-bootstrap 95% CI half-width is about 0.2 IoU; seed-to-seed spread of one model is about 0.4 IoU): 12/50 (24%) of claimed IoU margins are below 0.2, 17/50 (34%) below 0.4, and 21/50 (42%) below 0.57. 0.57 is 0.4 x sqrt(2), the spread of a difference between two single runs if 0.4 is a per-model standard deviation.
- **WHU-CD** (our paired 95% CI half-width is about 5 IoU on the WHU-CD test split): 36/39 (92%) of claimed IoU margins are below 5, 29/39 (74%) below 2.5. Three margins exceed 5. Two of them (TinyCD +12.35, APD +6.38) evaluate the proposed model on one WHU split and list competitor values copied from papers that used another split (e.g. BIT's 6096/762/762 values). The third (HCGMNet +7.14) is against competitor rows identical to the same group's HANet table (official split). HCGMNet's own text states +1.33 IoU on WHU-CD, which does not match its table.
- Several papers' own tables are consistent with a flat frontier. Proposed methods lose on IoU to a listed competitor in AdaDINO (LEVIR-CD, -0.10), AdaDINO (WHU-CD, -1.03), ChangeTitans (WHU-CD, -1.52), STNet (LEVIR-CD, -0.39), SemDINO (WHU-CD, -0.11), TransY-Net (LEVIR-CD, -0.34), ViT adapter (DINO/DINOv2) (LEVIR-CD, -1.40). They lose on F1 in RFL-CDNet (WHU-CD, -0.82, F1-only) and ScratchFormer (WHU-CD, F1 -1.43 against a competitor with an inconsistent F1/IoU pair).

## By split protocol (change-class IoU margins)

| split group | n | median margin_iou | IQR |
|---|---|---|---|
| LEVIR official split (256px crops or full 1024px images) | 42 | 0.91 | 0.38 to 1.37 |
| LEVIR other, random or unstated split | 8 | 0.12 | 0.08 to 0.32 |
| WHU official area-based train/test split | 8 | 1.47 | 0.82 to 3.11 |
| WHU other or unstated split | 13 | 1.90 | 0.87 to 2.12 |
| WHU random 256px split 5947/743/744 | 5 | 2.90 | 1.04 to 4.54 |
| WHU random 256px split 6096/762/762 (BIT) | 13 | 1.32 | 0.72 to 2.13 |

LEVIR-CD, by publication year: 2022-2023 median 1.02 (n=20); 2024-2026 median 0.62 (n=30).
WHU-CD, by publication year: 2022-2023 median 1.94 (n=16); 2024-2026 median 1.04 (n=23).

## Reporting practices (per paper, n = 62)

- **Any repeated runs, std, CI or significance test: 4/62 (6.5%)**: FCCDN (ablation table averaged over 3 runs (no std); main comparison single values); MCTNet (mean of three runs reported for every method; no std/CI); CDMaskFormer (Fig. 1 plots median and standard deviation over 5 random seeds; table values are point estimates); M-CD (states p-value <= 1e-5 vs all other methods; test and sampling unit not described; no std).
- An actual dispersion estimate or test for the main comparison: 2/62 (3.2%): CDMaskFormer, M-CD. No paper reports a confidence interval, and none describes a paired or image-level test. Two papers (TinyCD, MaskCD) state that they fixed a single random seed.
- **Test-time augmentation: 1/62 (1.6%)** use it (SAM-CD: 8x flips). 3 papers state explicitly that they do not (FCCDN, Changen2, AdaDINO). The other 58 do not mention it (recorded as "no (not mentioned)"). PeftCD uses sliding-window inference and ScratchFormer upsamples inputs at inference; neither counts as TTA here.

- Internal consistency: in 11 of 89 change-class rows, the proposed method's printed F1 differs from 2*IoU/(1+IoU) by more than 0.15 (e.g. TransY-Net LEVIR-CD, F1 91.90 against IoU 83.64, where its own P/R imply 91.09). They point to transcription errors or non-standard averaging, and they are comparable in size to the margins being claimed.

## Caveats

1. **Split heterogeneity, WHU-CD in particular.** WHU rows use at least four protocols: random 256 px split 6096/762/762 (BIT), random 5947/743/744, the official area-based train/test split (4536/504/2760 or 1260/690 tiles), and various own random splits (7:1:2, 6:1:3, 6690/744/744, 5141/2293, 1024 px tiles) or unstated splits. Absolute IoUs are not comparable across rows (BIT is listed anywhere from 68.0 to 88.2 IoU on "WHU-CD"). Most LEVIR-CD rows use the official 445/64/128 split (7120/1024/2048 256 px crops). Twelve do not, or do not say which split they use, including random re-splits of overlapping patches (SGSLN), 10000/1024/2048 (USSFC-Net, LSAT) and random splits (MaskCD, SMDNet).
2. **Copied versus re-run competitors, and protocol mixing.** Many tables copy competitor numbers from the original papers, often obtained under a different split or crop size, and place them next to the authors' own run. The two largest WHU margins come from such mixing: TinyCD WHU +12.35 (own 5947/743/744 split against BIT numbers from 6096/762/762) and APD WHU +6.38 (official 512 px split against copied rows such as BiT 72.39 from the 6096/762/762 random split). Re-run baselines are sometimes far below their published values (Dsfer-Net re-runs ChangeFormer at 73.1 IoU on LEVIR-CD, against 82.5 in its paper). The CSV `notes` column flags each case found; provenance was not audited for every competitor row.
3. **F1-only papers.** 8 rows have no IoU (Changer, USSFC-Net, MCTNet, LSAT, T-UNet x2, RFL-CDNet, Changen2). They count toward the F1 statistics only.
4. **Non-standard metrics.** SAM-CD reports class-mean mIoU/mF1 on both datasets, and MaskCD reports mIoU (its F1 is change-class). Class-mean margins are roughly half the size of change-class margins, because the unchanged-class IoU is near 99 and barely moves. EHCTNet's IoU column is internally inconsistent: several IoUs exceed the corresponding F1, which is impossible for change-class IoU. These IoU margins are excluded from the headline IoU statistics; see the sensitivity line above.
5. **Choices made here.** The best proposed variant is used when several are reported. The strongest competitor is taken by IoU, so the F1 margin is against that same row. SiamixFormer's table lists FDOR-Net with IoU 91.13 above its F1 90.85, which is impossible for change-class IoU, so UVACD is used instead. For Changen2, a pre-training paper, the competitor is the same architecture with SA-1B pre-training; the best external method, BAN-BiT, would give +0.6 F1.
6. **Versions.** For most published papers the numbers come from the latest arXiv version (recorded in `source`), which can differ from the version of record. Examples are ScratchFormer (arXiv v1), Changer (arXiv v1) and SAM-CD ("manuscript under review" version). Values printed with one decimal (TTP, Changen2) or as 3-decimal fractions (M-CD, NeXt2Former-CD) carry rounding of up to 0.05. SRC-Net, EGCTNet and MaskCD print 4-decimal fractions.
7. **Sample selection.** The sample is biased toward papers with arXiv or open-access versions and toward more-cited papers. Paywalled papers without a preprint were excluded (next section), which removes several heavily cited IEEE TGRS models. SemDINO's binary-CD table is a secondary experiment in a semantic-CD paper.
8. **What the comparison can and cannot show.** Our noise band is specific to our models and the WHU-CD/LEVIR-CD splits we use. Published margins are single-run point estimates from mixed protocols, so the comparison shows only that typical claimed gains are the same size as, or smaller than, the evaluation noise. It does not show that any particular claim is wrong. Independent evidence points the same way. Corley et al. (arXiv:2402.06994, Table 2) retrain ChangeFormer, TinyCD and BIT on the original WHU-CD train/test split over 10 seeds and report IoU standard deviations of 1.72 to 3.24 points; a plain U-Net baseline outperforms all three on that split. Rolih et al. (IEEE TGRS 2025) find that backbone, pre-training and training choices can outweigh architectural additions.

## Papers attempted but excluded

- **No open-access full text** (IEEE Xplore/Elsevier paywall and no arXiv version; Semantic Scholar lists them as closed): ChangeCLIP (ISPRS P&RS 2024), DMINet (TGRS 2023), ICIF-Net (TGRS 2022), A2Net (TGRS 2023), SEIFNet (TGRS 2024), AERNet (TGRS 2023), BiFA (TGRS 2024), WNet (TGRS 2023), DARNet (TGRS 2022), CSTSUNet (TGRS 2023), P2V-CD (TIP 2023), AMTNet (ISPRS P&RS 2023), SNUNet-CD (IEEE GRSL). SNUNet-CD's dataset coverage could not be checked without the full text.
- **Could not identify:** "MSCCA-Net". No paper with this name reporting LEVIR-CD/WHU-CD was found. The closest match was Mamba-MSCCA-Net (Displays, 2025), which was not accessed.
- **Out of scope after reading:** LCD-Net (JSTARS 2025) reports LEVIR-CD+, not LEVIR-CD. ChangeMamba's LEVIR-CD+ table is not used; only its WHU-CD row is kept. ChangeStar's IJCV 2024 extension (single-temporal supervision) uses LEVIR-CD/WHU-CD only for zero-shot or generalization tests. Changen2's WHU-CD results are zero-shot only. BAN's WHU-CD results are semi-supervised only. SiamixFormer uses WHU only for building extraction. L-UNet (arXiv 2026) is a 2020 paper. "A Change Detection Reality Check" and "Be the Change" are method-audit papers, cited above but not counted.

## Update 2026-09-28: blind re-extraction and one correction

All 101 rows were re-extracted from the original papers by a second, independent pass that did not see this table
(`blind_reextraction.json`; row-by-row comparison in `blind_comparison.json`). 97 rows agree on all four numbers and on
the strongest competitor. Of the other four, SiamixFormer (LEVIR-CD) agrees once the impossible FDOR-Net row
(IoU above F1) is skipped as recorded here, and Changen2 (LEVIR-CD, F1 only) differs only in the choice of competitor.

Correction: both DDPM-CD rows now use the published WACV 2025 version (Table 1, IN1k variant) instead of arXiv
2206.11892v3, which is a differently titled version with a different table. The WACV paper does not state its split, so
the LEVIR-CD row moved to "LEVIR other, random or unstated split". The statistics above were computed before this
change and are kept for the record; the paper uses the values recomputed by `analysis/paper_numbers.py` from
`claims.csv`.

Rows that decide the paper's statements are listed for manual verification in `verification_checklist.md`.

Further corrections on 2026-09-28, after checking the 12 open-access papers against their published versions
(`published_version_check.json`; 20 of 21 rows identical): SMDNet LEVIR-CD F1 corrected to the published 89.17, and
SRC-Net LEVIR-CD moved to the unstated-split group because its published version describes only training and validation
sets. After checking 29 papers against their published versions, ScratchFormer was also corrected to its published TGRS numbers
(LEVIR-CD 84.63/91.68, WHU-CD 84.97/91.87). Current LEVIR-CD official-split figures: 40 papers, median margin 0.90 IoU
points, 21 below one point.

## Published-version check completed 2026-09-28

All 50 formally published papers in the audit were checked against their published versions (82 rows,
`published_version_check.json`), each table read blind from a rendered page image. The remaining 12 papers are arXiv
preprints and were read from arXiv. Corrections made to `claims.csv`: DDPM-CD (published table, split not stated),
ScratchFormer (IoU and F1), SMDNet (F1), MaskCD (published table reports change-class IoU), and the competitor of three
F1-only rows (Changer, LSAT, Changen2). SRC-Net was moved out of the official split because it reports validation
results. Final LEVIR-CD official-split figures: 40 papers, median margin 0.90, 21 below one point, 7 below the MDD;
highest proposed IoU 86.51. WHU-CD 6096/762/762 split: all 13 margins below the MDD, largest 3.63.
