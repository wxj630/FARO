#!/usr/bin/env bash
# Complete the hard-locked all-coordinate control for the remaining formal seeds.
# FARO_COORDINATE_ONLY=1 pins every block to coordinate; it is the control that the
# no-prior auto-switch (FARO_COORDINATE_START) result is compared against.
# seed0 already exists in this results root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-coordonly
PHYSICAL_GPU=6
REQUIRED_MIB=54000
SEEDS=(42 1024)

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

run_one() {
    local seed=$1
    local run_id="faro-coordinate-only-seed${seed}"
    local run_dir="$RESULTS_ROOT/runs/faro/seed_${seed}/$run_id"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[$(date +%F_%T)] Skipping completed run: $run_dir"
        return
    fi
    mkdir -p "$run_dir"
    wait_for_gpu_memory "$REQUIRED_MIB"
    echo "[$(date +%F_%T)] Starting coordinate-only seed=$seed on GPU $PHYSICAL_GPU"
    env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
        LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
        FARO_VARIANT=faro \
        FARO_SEED="$seed" \
        FARO_RUN_ID="$run_id" \
        FARO_MAX_STEPS=1285 \
        FARO_MATRIX_LR=0.0015 \
        FARO_MUON_LR=0.023 \
        FARO_RESULTS_ROOT="$RESULTS_ROOT" \
        FARO_COORDINATE_ONLY=1 \
        .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished seed=$seed (exit=${PIPESTATUS[0]})"
}

for seed in "${SEEDS[@]}"; do
    run_one "$seed"
done
echo "[$(date +%F_%T)] coordinate-only seed42/1024 queue finished"
