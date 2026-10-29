import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
RUNS_ROOT = REPO_ROOT / "FARO" / "results" / "runs"


def _run_dir(variant: str) -> Path:
    seed_dir = RUNS_ROOT / variant / "seed_0"
    matches = sorted(seed_dir.glob(f"smoke-{variant}-seed0*"))
    assert matches, f"missing {variant} smoke run under {seed_dir}"
    completed = [path for path in matches if (path / "summary.json").is_file()]
    assert completed, f"no completed {variant} smoke run"
    return completed[-1]


def _events(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.cuda
def test_four_smoke_runs_are_complete_finite_and_parseable() -> None:
    for variant in ("adam", "muon", "normuon", "faro"):
        directory = _run_dir(variant)
        events = _events(directory)
        event_types = {event["event"] for event in events}
        summary = json.loads((directory / "summary.json").read_text())

        assert {"run", "train", "validation", "summary"} <= event_types
        assert summary["status"] == "complete"
        assert summary["step"] == 2
        assert math.isfinite(summary["final_validation_loss"])
        assert summary["peak_reserved_mib"] < 81559
        assert not [
            event
            for event in events
            if event["event"] == "event"
            and event["kind"] in {"nan", "inf", "oom", "hard_risk_violation"}
        ]


@pytest.mark.cuda
def test_smoke_initialization_and_data_hashes_match_across_variants() -> None:
    metadata = {}
    for variant in ("adam", "muon", "normuon", "faro"):
        run_event = next(
            event
            for event in _events(_run_dir(variant))
            if event["event"] == "run"
        )
        metadata[variant] = run_event["metadata"]

    for field in (
        "initialization_hash",
        "bigram_sign_table_hash",
        "data_manifest_hash",
    ):
        assert len({metadata[variant][field] for variant in metadata}) == 1


@pytest.mark.cuda
def test_faro_smoke_emits_routes_and_probe_after_reset() -> None:
    events = _events(_run_dir("faro"))
    routes = [event for event in events if event["event"] == "route"]
    probes = [event for event in events if event["event"] == "probe"]

    assert routes
    assert probes
    assert all(event["step"] >= 0 for event in routes)
    assert all(math.isfinite(event["realized_return"]) for event in probes)
