#!/usr/bin/env bash
# Second pass: measure the run-to-run noise band for the sparse-vs-dense A/B.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-sparse
PHYSICAL_GPU=6
cd "$REPO_ROOT"
run_one() {
    local mode=$1
    local run_id="faro-${mode}-repeat-seed0"
    local run_dir="$RESULTS_ROOT/runs/faro/seed_0/$run_id"
    [[ -f "$run_dir/summary.json" ]] && { echo "skip $run_dir"; return; }
    mkdir -p "$run_dir"
    local extra=""
    [[ "$mode" == "dense" ]] && extra="FARO_DENSE_MICRO=1"
    echo "[$(date +%F_%T)] Starting repeat mode=$mode"
    env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
        FARO_VARIANT=faro FARO_SEED=0 FARO_RUN_ID="$run_id" FARO_MAX_STEPS=1285 \
        FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 FARO_RESULTS_ROOT="$RESULTS_ROOT" $extra \
        .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
    echo "[$(date +%F_%T)] Finished repeat mode=$mode (exit=${PIPESTATUS[0]})"
}
for mode in dense sparse; do run_one "$mode"; done
echo "[$(date +%F_%T)] repeat queue finished"
