#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-kelly-smoke-v2
PHYSICAL_GPU=6
REQUIRED_MIB=54000
SEED=0
MAX_STEPS=96
cd "$REPO_ROOT"
wait_for_gpu_memory() {
  local free_mib
  while true; do
    free_mib=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    (( free_mib >= REQUIRED_MIB )) && return
    printf '[%s] GPU %s free=%s MiB; waiting\n' "$(date +%F_%T)" "$PHYSICAL_GPU" "$free_mib"
    sleep 60
  done
}
run_one() {
  local kind=$1
  local run_id="faro-kelly-smoke-v2-${kind}-seed${SEED}"
  local run_dir="$RESULTS_ROOT/runs/faro/seed_${SEED}/$run_id"
  [[ -f "$run_dir/summary.json" ]] && { echo "skip $run_id"; return; }
  mkdir -p "$run_dir"
  wait_for_gpu_memory
  local cf=0 kelly=0
  [[ "$kind" == counterfactual || "$kind" == kelly ]] && cf=1
  [[ "$kind" == kelly ]] && kelly=1
  echo "[$(date +%F_%T)] start $run_id"
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
    TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
    FARO_VARIANT=faro FARO_SEED="$SEED" FARO_RUN_ID="$run_id" FARO_MAX_STEPS="$MAX_STEPS" \
    FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 FARO_RESULTS_ROOT="$RESULTS_ROOT" \
    FARO_COUNTERFACTUAL_PROBE_BUDGET="$cf" FARO_KELLY_RISK_CONTROLLER="$kelly" \
    .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$run_dir/stdout.txt"
  echo "[$(date +%F_%T)] finish $run_id exit=${PIPESTATUS[0]}"
}
run_one control
run_one counterfactual
run_one kelly
echo "[$(date +%F_%T)] smoke v2 finished"
