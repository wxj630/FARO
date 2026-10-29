import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "FARO" / "run_experiments.sh"


def test_runner_is_tmux_backed_and_never_targets_gpu3() -> None:
    source = SCRIPT.read_text()

    for action in (
        "start-smoke",
        "smoke-worker",
        "start-pilot",
        "pilot-worker",
        "start-diagnostic",
        "diagnostic-worker",
        "start-fixed-diagnostic",
        "fixed-diagnostic-worker",
        "start-formal",
        "formal-worker",
        "manifest",
        "status",
    ):
        assert action in source
    assert "tmux new-session" in source
    assert "CUDA_VISIBLE_DEVICES=1" in source
    assert "LOCAL_KERNELS=kernels-community/flash-attn3=" in source
    assert "CUDA_VISIBLE_DEVICES=3" not in source
    assert "kill" not in source
    assert "nohup" not in source


def test_manifest_contains_exact_formal_matrix() -> None:
    environment = dict(os.environ, FARO_ADAM_LR="0.0015")
    completed = subprocess.run(
        ["bash", str(SCRIPT), "manifest"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]

    assert len(rows) == 12
    assert {(row["variant"], row["seed"]) for row in rows} == {
        (variant, seed)
        for variant in ("adam", "muon", "normuon", "faro")
        for seed in (0, 1, 2)
    }
    assert all(row["physical_gpu"] == 1 for row in rows)
    assert all(row["world_size"] == 1 for row in rows)
    assert all(row["max_steps"] == 1285 for row in rows)
    assert all(row["muon_lr"] == 0.023 for row in rows)
    assert all(row["matrix_lr"] == 0.0015 for row in rows)


def test_formal_runner_defaults_adam_matrix_lr_without_pilot_dependency() -> None:
    source = SCRIPT.read_text()
    selector = source.split("selected_adam_lr() {", 1)[1].split("\n}\n", 1)[0]

    assert 'echo "${FARO_ADAM_LR:-0.0015}"' in selector
    assert "selected.json" not in selector


def test_fixed_diagnostic_is_isolated_and_uses_faro_memory_gate() -> None:
    source = SCRIPT.read_text()

    assert 'run_one faro_fixed 0 "$max_steps" "$run_id"' in source
    assert '"$variant" == faro_fixed' in source
    assert "FARO_FIXED_STEPS:-250" in source
    assert "faro-fixed-diagnostic250-seed0" in source
