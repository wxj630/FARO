#!/usr/bin/env bash
# Opt-in check: force every FARO block to coordinate geometry and see whether the
# run still converges at 1285 steps. Isolated results root; does not touch the
# formal sparse results. Explicit FARO_COORDINATE_ONLY=1 (default stays off).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-coordonly
PHYSICAL_GPU=6
REQUIRED_MIB=54000
RUN_ID=faro-coordinate-only-seed0
SEED=0

cd "$REPO_ROOT"

wait_for_gpu_memory() {
    local required_mib=$1
    while true; do
        local free_mib
        free_mib=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if (( free_mib >= required_mib )); then return; fi
        echo "[$(date +%F_%T)] GPU $PHYSICAL_GPU has ${free_mib} MiB free; waiting for ${required_mib} MiB"
        sleep 60
    done
}

run_dir="$RESULTS_ROOT/runs/faro/seed_${SEED}/$RUN_ID"
if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$(date +%F_%T)] Already complete: $run_dir"
    exit 0
fi
mkdir -p "$run_dir"
wait_for_gpu_memory "$REQUIRED_MIB"
echo "[$(date +%F_%T)] Starting coordinate-only seed=$SEED on GPU $PHYSICAL_GPU"
env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
    FARO_VARIANT=faro \
    FARO_SEED="$SEED" \
    FARO_RUN_ID="$RUN_ID" \
    FARO_MAX_STEPS=1285 \
    FARO_MATRIX_LR=0.0015 \
    FARO_MUON_LR=0.023 \
    FARO_RESULTS_ROOT="$RESULTS_ROOT" \
    TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
    FARO_COORDINATE_ONLY=1 \
    .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
    2>&1 | tee "$run_dir/stdout.txt"
echo "[$(date +%F_%T)] Finished coordinate-only (exit=${PIPESTATUS[0]})"
