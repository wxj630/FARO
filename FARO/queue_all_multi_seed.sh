#!/usr/bin/env bash
# Queue all multi-seed formal runs (seed 42 and 1024 for faro/adam/muon/normuon),
# one after another on the same physical GPU, waiting for enough free memory.
# Completed runs (summary.json present) are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results
PHYSICAL_GPU=1
SEEDS=(42 1024)
FARO_REQUIRED_MIB=61500
OTHER_REQUIRED_MIB=50000

cd "$REPO_ROOT"

wait_for_gpu_memory() {
    local required_mib=$1
    while true; do
        local free_mib
        free_mib=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if (( free_mib >= required_mib )); then
            return
        fi
        echo "[$(date +%F_%T)] GPU $PHYSICAL_GPU has ${free_mib} MiB free; waiting for ${required_mib} MiB"
        sleep 30
    done
}

run_one() {
    local variant=$1
    local seed=$2
    local required_mib
    if [[ "$variant" == faro ]]; then
        required_mib=$FARO_REQUIRED_MIB
    else
        required_mib=$OTHER_REQUIRED_MIB
    fi
    local run_id
    if [[ "$variant" == faro ]]; then
        run_id="faro-dynamic-explore-freeze-seed${seed}"
    else
        run_id="${variant}-seed${seed}"
    fi
    local run_dir="$RESULTS_ROOT/runs/$variant/seed_${seed}/$run_id"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[$(date +%F_%T)] Skipping completed run: $run_dir"
        return
    fi
    mkdir -p "$run_dir"
    wait_for_gpu_memory "$required_mib"
    echo "[$(date +%F_%T)] Starting variant=${variant} seed=${seed} max_steps=1285 run_id=${run_id} on GPU ${PHYSICAL_GPU}"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
    FARO_VARIANT="$variant" \
    FARO_SEED="$seed" \
    FARO_RUN_ID="$run_id" \
    FARO_MAX_STEPS=1285 \
    FARO_MATRIX_LR=0.0015 \
    FARO_MUON_LR=0.023 \
    FARO_RESULTS_ROOT="$RESULTS_ROOT" \
    .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished ${variant} seed=${seed} (exit=${PIPESTATUS[0]})"
}

for variant in faro adam muon normuon; do
    for seed in "${SEEDS[@]}"; do
        run_one "$variant" "$seed"
    done
done
echo "[$(date +%F_%T)] All multi-seed runs finished"
