#!/usr/bin/env bash
# Adaptive risk appetite v2 (relative-calibration appetite) seed-0 check.
#
# v1 pinned appetite to its floor because realized/predicted is structurally << 1.
# v2 makes appetite track deviation from its own running baseline, so constant
# bias leaves appetite at 1.0 and only a change in conversion moves it. Appetite > 1
# can now push budget above the baseline's 1.0 ceiling (toward budget_max).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-adaptive-v2
PHYSICAL_GPU=4
REQUIRED_MIB=55000
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

run_one() {
    local kind=$1
    local run_id="faro-adaptive-v2-${kind}-seed${SEED}"
    local run_dir="$RESULTS_ROOT/runs/faro/seed_${SEED}/$run_id"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[$(date +%F_%T)] Skipping completed run: $run_dir"
        return
    fi
    mkdir -p "$run_dir"
    local extra=""
    if [[ "$kind" == "noprior" ]]; then
        extra="FARO_COORDINATE_START=1"
    fi
    wait_for_gpu_memory "$REQUIRED_MIB"
    echo "[$(date +%F_%T)] Starting adaptive v2 kind=$kind seed=$SEED on GPU $PHYSICAL_GPU"
    env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken \
        LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
        FARO_VARIANT=faro \
        FARO_SEED="$SEED" \
        FARO_RUN_ID="$run_id" \
        FARO_MAX_STEPS=1285 \
        FARO_MATRIX_LR=0.0015 \
        FARO_MUON_LR=0.023 \
        FARO_RESULTS_ROOT="$RESULTS_ROOT" \
        FARO_ADAPTIVE_RISK_APPETITE=1 \
        FARO_PROBE_TOKENS=1048576 \
        $extra \
        .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished kind=$kind (exit=${PIPESTATUS[0]})"
}

for kind in prior noprior; do
    run_one "$kind"
done
echo "[$(date +%F_%T)] adaptive v2 seed0 queue finished"
