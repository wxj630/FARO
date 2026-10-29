#!/usr/bin/env bash
# Verify the sparse micro-gradient collector on GPU6:
#   dense (FARO_DENSE_MICRO=1) must reproduce the old baseline bit-for-bit,
#   sparse (default) is the candidate change.
# Both write to an isolated results root so the formal summary is untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-sparse
PHYSICAL_GPU=6

cd "$REPO_ROOT"

run_one() {
    local mode=$1
    local run_id="faro-${mode}-seed0"
    local run_dir="$RESULTS_ROOT/runs/faro/seed_0/$run_id"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[$(date +%F_%T)] Skipping completed run: $run_dir"
        return
    fi
    mkdir -p "$run_dir"
    local extra=""
    if [[ "$mode" == "dense" ]]; then
        extra="FARO_DENSE_MICRO=1"
    fi
    echo "[$(date +%F_%T)] Starting mode=$mode run_id=$run_id on GPU $PHYSICAL_GPU"
    env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
        FARO_VARIANT=faro \
        FARO_SEED=0 \
        FARO_RUN_ID="$run_id" \
        FARO_MAX_STEPS=1285 \
        FARO_MATRIX_LR=0.0015 \
        FARO_MUON_LR=0.023 \
        FARO_RESULTS_ROOT="$RESULTS_ROOT" \
        $extra \
        .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished mode=$mode (exit=${PIPESTATUS[0]})"
}

for mode in dense sparse; do
    run_one "$mode"
done
echo "[$(date +%F_%T)] Sparse-verification queue finished"
