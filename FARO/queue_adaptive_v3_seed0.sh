#!/usr/bin/env bash
# Adaptive risk appetite v3: relative calibration + wait for 3 independent probe
# calibrations before explore->freeze.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-adaptive-v3
PHYSICAL_GPU=7
REQUIRED_MIB=53000
SEED=0
cd "$REPO_ROOT"
wait_for_gpu_memory() {
  local required_mib=$1
  while true; do
    local free_mib
    free_mib=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if (( free_mib >= required_mib )); then return; fi
    echo "[$(date +%F_%T)] GPU $PHYSICAL_GPU free=${free_mib}MiB; waiting"
    sleep 60
  done
}
run_one() {
  local kind=$1
  local run_id="faro-adaptive-v3-${kind}-seed${SEED}"
  local run_dir="$RESULTS_ROOT/runs/faro/seed_${SEED}/$run_id"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date +%F_%T)] skipping completed kind=$kind"
    return
  fi
  mkdir -p "$run_dir"
  local extra=""
  [[ "$kind" == "noprior" ]] && extra="FARO_COORDINATE_START=1"
  wait_for_gpu_memory "$REQUIRED_MIB"
  echo "[$(date +%F_%T)] starting kind=$kind"
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
    LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
    FARO_VARIANT=faro FARO_SEED="$SEED" FARO_RUN_ID="$run_id" FARO_MAX_STEPS=1285 \
    FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 FARO_RESULTS_ROOT="$RESULTS_ROOT" \
    FARO_ADAPTIVE_RISK_APPETITE=1 FARO_PROBE_TOKENS=1048576 $extra \
    .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$run_dir/stdout.txt"
  echo "[$(date +%F_%T)] finished kind=$kind exit=${PIPESTATUS[0]}"
}
run_one prior
run_one noprior
echo "[$(date +%F_%T)] adaptive v3 finished"
