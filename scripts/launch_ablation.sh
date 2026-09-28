#!/usr/bin/env bash
# Run the first experiment round in parallel, one run per GPU (4x RTX 2080 Super):
#   E1 degrade=none, E2 degrade=context_only, E3 degrade=shared, E5 ImageNet init.
# For a single bigger run use: torchrun --standalone --nproc_per_node=4 scripts/train.py experiment=...
set -euo pipefail
EXP=${EXP:-ijepa_tiny_224}
WORKERS=${WORKERS:-24}            # data-loader workers per run (112 cores / 4 runs)
mkdir -p logs
run() { gpu=$1; name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/train.py "$@" train.num_workers=$WORKERS > "logs/$name.log" 2>&1 &
  echo "GPU $gpu -> $name (logs/$name.log)"; }
run 0 e1_none         experiment=$EXP degrade=none
run 1 e2_context_only experiment=$EXP degrade=context_only
run 2 e3_shared       experiment=$EXP degrade=shared
run 3 e5_imagenet     experiment=ijepa_tiny_imagenet_init degrade=none
wait
