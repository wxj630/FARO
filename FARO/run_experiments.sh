#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=FARO/results
PHYSICAL_GPU=1
SMOKE_SESSION=faro-smoke-gpu1
PILOT_SESSION=faro-pilot-gpu1
DIAGNOSTIC_SESSION=faro-diagnostic-gpu1
FIXED_DIAGNOSTIC_SESSION=faro-fixed-gpu1
FORMAL_SESSION=faro-formal-gpu1

cd "$REPO_ROOT"

selected_adam_lr() {
    echo "${FARO_ADAM_LR:-0.0015}"
}

wait_for_gpu_memory() {
    local required_mib=$1
    while true; do
        local free_mib
        free_mib=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if (( free_mib >= required_mib )); then
            return
        fi
        echo "GPU $PHYSICAL_GPU has ${free_mib} MiB free; waiting for ${required_mib} MiB"
        sleep 30
    done
}

run_one() {
    local variant=$1
    local seed=$2
    local max_steps=$3
    local run_id=$4
    local matrix_lr=$5
    local required_mib=50000
    if [[ "$variant" == faro || "$variant" == faro_fixed ]]; then
        required_mib=65000
    fi
    local run_dir="$RESULTS_ROOT/runs/$variant/seed_${seed}/$run_id"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "Skipping completed run: $run_dir"
        return
    fi
    if [[ -d "$run_dir" ]]; then
        run_id="${run_id}-retry-$(date +%Y%m%d-%H%M%S)"
        run_dir="$RESULTS_ROOT/runs/$variant/seed_${seed}/$run_id"
    fi
    mkdir -p "$run_dir"
    wait_for_gpu_memory "$required_mib"
    echo "Starting variant=$variant seed=$seed max_steps=$max_steps matrix_lr=$matrix_lr"
    CUDA_VISIBLE_DEVICES=1 \
    LOCAL_KERNELS=kernels-community/flash-attn3=$HOME/.cache/modelscope-kernels/flash-attn3 \
    FARO_VARIANT="$variant" \
    FARO_SEED="$seed" \
    FARO_RUN_ID="$run_id" \
    FARO_MAX_STEPS="$max_steps" \
    FARO_MATRIX_LR="$matrix_lr" \
    FARO_MUON_LR=0.023 \
    FARO_RESULTS_ROOT="$RESULTS_ROOT" \
    .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "$run_dir/stdout.txt"
}

start_session() {
    local session=$1
    local worker=$2
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session already exists: $session"
        return
    fi
    tmux new-session -d -s "$session" "cd $REPO_ROOT && bash FARO/run_experiments.sh $worker"
    echo "Started tmux session: $session"
}

smoke_worker() {
    local matrix_lr=${FARO_ADAM_LR:-0.0015}
    for variant in adam muon normuon faro; do
        run_one "$variant" 0 2 "smoke-${variant}-seed0" "$matrix_lr"
    done
}

pilot_worker() {
    local lr
    local -a directories=()
    for lr in 0.00075 0.0015 0.003; do
        local tag=${lr//./p}
        local run_id="pilot-adam-lr${tag}-seed0"
        run_one adam 0 250 "$run_id" "$lr"
        directories+=("$RESULTS_ROOT/runs/adam/seed_0/$run_id")
    done
    mkdir -p "$RESULTS_ROOT/adam_pilot"
    .venv/bin/python - "${directories[@]}" <<'PY'
import json
import sys
from pathlib import Path
from FARO.experiment import select_adam_pilot

selection = select_adam_pilot([Path(value) for value in sys.argv[1:]])
destination = Path("FARO/results/adam_pilot/selected.json")
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
print(json.dumps(selection, indent=2, sort_keys=True))
PY
}

diagnostic_worker() {
    local max_steps=${FARO_DIAGNOSTIC_STEPS:-250}
    local run_id=${FARO_DIAGNOSTIC_RUN_ID:-faro-v2-diagnostic${max_steps}-seed0}
    run_one faro 0 "$max_steps" "$run_id" "$(selected_adam_lr)"
}

fixed_diagnostic_worker() {
    local max_steps=${FARO_FIXED_STEPS:-250}
    local run_id=${FARO_FIXED_RUN_ID:-faro-fixed-diagnostic250-seed0}
    run_one faro_fixed 0 "$max_steps" "$run_id" "$(selected_adam_lr)"
}

formal_worker() {
    local matrix_lr
    matrix_lr=$(selected_adam_lr)
    local seed variant
    for seed in 0 1 2; do
        for variant in adam muon normuon faro; do
            run_one "$variant" "$seed" 1285 "${variant}-seed${seed}" "$matrix_lr"
        done
    done
}

manifest() {
    local matrix_lr
    matrix_lr=$(selected_adam_lr)
    local seed variant
    for seed in 0 1 2; do
        for variant in adam muon normuon faro; do
            printf '{"variant":"%s","seed":%d,"physical_gpu":1,"world_size":1,"max_steps":1285,"matrix_lr":%s,"muon_lr":0.023}\n' \
                "$variant" "$seed" "$matrix_lr"
        done
    done
}

status() {
    tmux list-sessions 2>/dev/null | rg '^faro-' || true
    find "$RESULTS_ROOT/runs" -name summary.json -print 2>/dev/null | sort || true
}

case "${1:-}" in
    start-smoke) start_session "$SMOKE_SESSION" smoke-worker ;;
    smoke-worker) smoke_worker ;;
    start-pilot) start_session "$PILOT_SESSION" pilot-worker ;;
    pilot-worker) pilot_worker ;;
    start-diagnostic) start_session "$DIAGNOSTIC_SESSION" diagnostic-worker ;;
    diagnostic-worker) diagnostic_worker ;;
    start-fixed-diagnostic) start_session "$FIXED_DIAGNOSTIC_SESSION" fixed-diagnostic-worker ;;
    fixed-diagnostic-worker) fixed_diagnostic_worker ;;
    start-formal) start_session "$FORMAL_SESSION" formal-worker ;;
    formal-worker) formal_worker ;;
    manifest) manifest ;;
    status) status ;;
    *)
        echo "Usage: $0 {start-smoke|smoke-worker|start-pilot|pilot-worker|start-diagnostic|diagnostic-worker|start-fixed-diagnostic|fixed-diagnostic-worker|start-formal|formal-worker|manifest|status}" >&2
        exit 2
        ;;
esac
