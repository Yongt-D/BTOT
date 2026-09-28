#!/bin/bash
# Post-training evaluation of matched-recipe LEVIR-CD models (single pass @0.5), run from the repo root:
#   bash analysis/final_sweep.sh PYTHON TAG name=ckpt [name=ckpt ...]
# 1. object-level complete misses on the LEVIR-CD test split  -> results/component_miss_levir/<name>.npz
# 2. gamma, branch/difference norm ratio, IoU with gamma = 0   -> results/branch_ratio_levir/<name>.json
# 3. zero-shot LEVIR-CD -> WHU-CD from the logit margin         -> results/zeroshot_levir2whu/<name>.json
# 4. nuisance perturbations of the second image (real pairs)   -> results/nuisance_levir_<TAG>/
# Steps 1-3 cache per model, so a rerun only evaluates what is missing.
PY=$1
TAG=$2
shift 2
export PYTHONPATH=$(pwd)
export PYTHONIOENCODING=utf-8
O=results
CK=()
for s in "$@"; do CK+=(--ckpt "$s"); done

echo "$(date '+%F %T') [1/4] component misses"
$PY -u eval_component_miss.py --gpu 0 "${CK[@]}" --out_dir $O/component_miss_levir || echo "step 1 failed"
echo "$(date '+%F %T') [2/4] branch ratio"
$PY -u analysis/branch_ratio.py $O/branch_ratio_levir "$@" || echo "step 2 failed"
echo "$(date '+%F %T') [3/4] zero-shot LEVIR-CD -> WHU-CD"
$PY -u analysis/zeroshot_eval.py $O/zeroshot_levir2whu "$@" || echo "step 3 failed"
echo "$(date '+%F %T') [4/4] nuisance perturbations"
$PY -u eval_nuisance_robustness.py --gpu 0 "${CK[@]}" --out_dir $O/nuisance_levir_$TAG --protocol real \
  --nuisance clean --nuisance shift2 --nuisance shift4 --nuisance shift8 --nuisance bright0.7 \
  --nuisance gamma1.6 --nuisance colorcast --nuisance blur1.5 || echo "step 4 failed"
echo "$(date '+%F %T') sweep finished"
