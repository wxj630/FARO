#!/usr/bin/env bash
# A/B: does normalizing candidate UWR onto the coordinate scale change the
# no-prior auto-switch outcome? Same seed0 coordinate-start config, fix off/on.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-coordstart
PHYSICAL_GPU=6
cd "$REPO_ROOT"
run_one() {
    local tag=$1
    local run_id="faro-coordinate-start-uwrnorm-${tag}-seed0"
    local run_dir="$RESULTS_ROOT/runs/faro/seed_0/$run_id"
    [[ -f "$run_dir/summary.json" ]] && { echo "skip $run_dir"; return; }
    mkdir -p "$run_dir"
    local extra=""
    [[ "$tag" == "on" ]] && extra="FARO_NORMALIZE_CANDIDATE_UWR=1"
    echo "[$(date +%F_%T)] Starting uwrnorm=$tag"
    env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
        LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
        FARO_VARIANT=faro FARO_SEED=0 FARO_RUN_ID="$run_id" FARO_MAX_STEPS=1285 \
        FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 FARO_RESULTS_ROOT="$RESULTS_ROOT" \
        FARO_COORDINATE_START=1 $extra \
        .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished uwrnorm=$tag (exit=${PIPESTATUS[0]})"
}
for tag in off on; do run_one "$tag"; done
echo "[$(date +%F_%T)] uwrnorm A/B finished"
