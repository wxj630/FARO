#!/usr/bin/env bash
# Adaptive v4: global held-out outcome + robust selected-risk pressure. No
# per-structure realized attribution and no predicted-return level in appetite.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results-adaptive-v4
PHYSICAL_GPU=7
REQUIRED_MIB=55000
SEED=0
cd "$REPO_ROOT"
wait_for_gpu_memory(){ local need=$1; while true; do local free; free=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits|tr -d ' '); (( free >= need )) && return; echo "waiting GPU $PHYSICAL_GPU free=$free"; sleep 60; done; }
run_one(){ local kind=$1; local id="faro-adaptive-v4-${kind}-seed${SEED}"; local dir="$RESULTS_ROOT/runs/faro/seed_${SEED}/$id"; [[ -f "$dir/summary.json" ]] && { echo skip; return; }; mkdir -p "$dir"; local extra=""; [[ "$kind" == noprior ]] && extra="FARO_COORDINATE_START=1"; wait_for_gpu_memory "$REQUIRED_MIB"; env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 FARO_VARIANT=faro FARO_SEED="$SEED" FARO_RUN_ID="$id" FARO_MAX_STEPS=1285 FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 FARO_RESULTS_ROOT="$RESULTS_ROOT" FARO_ADAPTIVE_RISK_APPETITE=1 FARO_PROBE_TOKENS=1048576 $extra .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$dir/stdout.txt"; echo finished $kind exit=${PIPESTATUS[0]}; }
run_one prior; run_one noprior
