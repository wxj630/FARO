import json
from pathlib import Path

import matplotlib
import pytest

import FARO.plot_results as plot_results
from FARO.plot_results import aggregate_runs, audit_runs, load_runs, plot_all


matplotlib.use("Agg")


FORMAL_SEEDS = (0, 42, 1024)
FORMAL_RUN_IDS = {
    "adam": {
        "0": "adam-seed0",
        "42": "adam-seed42",
        "1024": "adam-seed1024",
    },
    "muon": {
        "0": "muon-seed0",
        "42": "muon-seed42",
        "1024": "muon-seed1024",
    },
    "normuon": {
        "0": "normuon-seed0",
        "42": "normuon-seed42",
        "1024": "normuon-seed1024",
    },
    "faro": {
        "0": "faro-coordinate-start-fixed-seed0",
        "42": "faro-coordinate-start-fixed-seed42",
        "1024": "faro-coordinate-start-fixed-seed1024",
    },
    "faro_prior": {
        "0": "faro-sparse-seed0",
        "42": "faro-sparse-seed42",
        "1024": "faro-sparse-seed1024",
    },
}

FIGURES = {
    "01-val-loss-vs-tokens.png",
    "02-val-loss-vs-time.png",
    "03-train-loss-ema.png",
    "04-geometry-mode-stack.png",
    "05-routing-heatmap.png",
    "06-risk-drift-calibration.png",
    "07-budget-and-switches.png",
    "08-throughput-overhead-memory.png",
    "09-final-val-seeds.png",
}


def _event(
    run_id: str, variant: str, seed: int, event: str, step: int, **payload: object
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "event": event,
        "step": step,
        **payload,
    }


def _write_run(
    root: Path,
    *,
    variant: str,
    seed: int,
    run_id: str,
    final_loss: float,
    final_step: int = 1285,
    metadata_variant: str | None = None,
) -> None:
    metadata_variant = metadata_variant or variant
    directory = root / variant / f"seed_{seed}" / run_id
    directory.mkdir(parents=True)
    events = [
        _event(
            run_id,
            metadata_variant,
            seed,
            "run",
            0,
            metadata={
                "run_id": run_id,
                "initialization_hash": f"init-{seed}",
                "bigram_sign_table_hash": f"bigram-{seed}",
                "data_manifest_hash": "data",
                "code_hash": f"code-{metadata_variant}",
                "world_size": 1,
                "grad_accum_steps": 8,
                "config": {
                    "matrix_lr": 0.0015,
                    "muon_lr": 0.023,
                    "max_steps": final_step,
                },
            },
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "validation",
            0,
            loss=4.0,
            tokens=0,
            training_time_ms=0.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "train",
            1,
            loss=4.0,
            tokens=100,
            training_time_ms=100.0,
            wall_time_ms=110.0,
            tokens_per_second=1000.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "train",
            48,
            loss=3.9,
            tokens=400,
            training_time_ms=200.0,
            wall_time_ms=210.0,
            tokens_per_second=2000.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "train",
            49,
            loss=3.8,
            tokens=500,
            training_time_ms=250.0,
            wall_time_ms=260.0,
            tokens_per_second=2000.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "train",
            50,
            loss=3.7,
            tokens=700,
            training_time_ms=300.0,
            wall_time_ms=310.0,
            tokens_per_second=2333.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "train",
            final_step,
            loss=final_loss + 0.1,
            tokens=1000,
            training_time_ms=500.0,
            wall_time_ms=550.0,
            tokens_per_second=2000.0,
        ),
        _event(
            run_id,
            metadata_variant,
            seed,
            "validation",
            final_step,
            loss=final_loss,
            tokens=1000,
            training_time_ms=500.0,
        ),
    ]
    if variant.startswith("faro"):
        events.extend(
            (
                _event(
                    run_id,
                    metadata_variant,
                    seed,
                    "event",
                    48,
                    kind="faro_freeze",
                    message="FARO routes frozen",
                    newly_frozen=102,
                    total_frozen=102,
                    mode_counts={"spectral": 102},
                    all_routes_frozen=False,
                    current_allocated_mib=200.0,
                    current_reserved_mib=300.0,
                ),
                _event(
                    run_id,
                    metadata_variant,
                    seed,
                    "event",
                    49,
                    kind="faro_freeze",
                    message="FARO routes frozen",
                    newly_frozen=16,
                    total_frozen=118,
                    mode_counts={"coordinate": 16, "spectral": 102},
                    all_routes_frozen=True,
                    current_allocated_mib=100.0,
                    current_reserved_mib=120.0,
                ),
            )
        )
        events.append(
            _event(
                run_id,
                metadata_variant,
                seed,
                "route",
                40,
                block="mlp_bank.layer0.up",
                structure_class="mlp",
                mode="spectral",
                budget_scale=0.8,
                scores={"coordinate": 0.0, "spectral": 1.0},
                switched=True,
                reason="challenger_won_twice",
                hard_violations=[],
                cvar=0.1,
                drift=0.01,
                calibration_error=-0.02,
                gradient_sv_concentration=0.8,
                momentum_sv_concentration=0.7,
                qk_joint_spectral_risk=0.0,
            )
        )
        events.append(
            _event(
                run_id,
                metadata_variant,
                seed,
                "probe",
                48,
                realized_return=0.1,
                predicted_return=0.2,
                drift=0.02,
            )
        )
    summary = _event(
        run_id,
        metadata_variant,
        seed,
        "summary",
        final_step,
        status="complete",
        final_validation_loss=final_loss,
        peak_allocated_mib=100.0,
        peak_reserved_mib=120.0,
    )
    events.append(summary)
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    (directory / "summary.json").write_text(json.dumps(summary))


def _write_fixture(root: Path) -> None:
    losses = {
        "adam": 3.5,
        "muon": 3.4,
        "normuon": 3.3,
        "faro": 3.39,
        "faro_prior": 3.28,
    }
    for variant, seeds in FORMAL_RUN_IDS.items():
        for seed, run_id in seeds.items():
            _write_run(
                root,
                variant=variant,
                seed=seed,
                run_id=run_id,
                final_loss=losses[variant] + int(seed) / 10000,
                # Both FARO experiments carry the metadata variant "faro".
                metadata_variant="faro" if variant.startswith("faro") else None,
            )
    _write_run(
        root,
        variant="faro",
        seed=0,
        run_id="faro-diagnostic250-seed0",
        final_loss=4.8,
        final_step=250,
    )
    incomplete = root / "adam" / "seed_0" / "interrupted-smoke-seed0"
    incomplete.mkdir(parents=True)
    interrupted_run = _event(
        "interrupted-smoke-seed0",
        "adam",
        0,
        "run",
        0,
        metadata={},
    )
    (incomplete / "metrics.jsonl").write_text(json.dumps(interrupted_run) + "\n")


def _formal_run_ids() -> list[str]:
    return [
        run_id for seeds in FORMAL_RUN_IDS.values() for run_id in seeds.values()
    ]


def test_select_audit_and_aggregate_twelve_formal_runs(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="must contain one run and one summary"):
        load_runs(tmp_path)

    included_ids = (*_formal_run_ids(), "faro-diagnostic250-seed0")
    all_runs = load_runs(tmp_path, run_ids=included_ids)
    assert len(all_runs) == 16
    assert plot_results.FORMAL_RUN_IDS == FORMAL_RUN_IDS
    assert plot_results.FORMAL_SEEDS == FORMAL_SEEDS
    assert plot_results.VARIANTS == tuple(FORMAL_RUN_IDS)
    assert "faro_fixed" not in plot_results.DISPLAY_NAMES
    assert plot_results.DISPLAY_NAMES["faro"] == "FARO no-prior auto-switch"

    runs = plot_results.select_formal_runs(all_runs)
    audit = audit_runs(runs, require_formal=True)
    summary = aggregate_runs(runs)

    assert [run.run_id for run in runs] == _formal_run_ids()
    assert audit["complete_runs"] == 15
    assert audit["fairness_hashes_match"]
    assert audit["fairness_hashes"] == {
        "0": {
            "initialization_hash": "init-0",
            "bigram_sign_table_hash": "bigram-0",
            "data_manifest_hash": "data",
        },
        "42": {
            "initialization_hash": "init-42",
            "bigram_sign_table_hash": "bigram-42",
            "data_manifest_hash": "data",
        },
        "1024": {
            "initialization_hash": "init-1024",
            "bigram_sign_table_hash": "bigram-1024",
            "data_manifest_hash": "data",
        },
    }
    assert not audit["code_hashes_match"]
    assert audit["code_hashes"] == {
        variant: {
            str(seed): (
                "code-faro"
                if variant.startswith("faro")
                else f"code-{variant}"
            )
            for seed in FORMAL_SEEDS
        }
        for variant in FORMAL_RUN_IDS
    }
    assert audit["settings"]["seeds"] == (0, 42, 1024)
    assert audit["settings"]["world_size"] == 1
    assert audit["settings"]["max_steps"] == 1285
    assert audit["mlp_spectral_ratio"] == pytest.approx(1.0)
    assert summary["variants"]["faro"]["seed_values"] == pytest.approx(
        {
            "0": 3.39,
            "42": 3.39 + 42 / 10000,
            "1024": 3.39 + 1024 / 10000,
        }
    )
    assert summary["variants"]["faro"]["final_validation_loss_mean"] == pytest.approx(
        (3.39 + (3.39 + 42 / 10000) + (3.39 + 1024 / 10000)) / 3
    )
    assert "faro_fixed" not in summary["variants"]
    assert "faro_fixed" not in summary["validation"]
    assert "faro_fixed" not in summary["train"]
    assert "faro_fixed" not in summary["system"]
    assert summary["routes"] == {"mlp": {"spectral": 1}}
    assert summary["route_audit"] == {
        "event_count": 1,
        "mode_counts": {"spectral": 1},
        "switch_count": 1,
        "budget_min": 0.8,
        "budget_mean": 0.8,
        "budget_max": 0.8,
    }
    assert summary["probe_audit"] == {
        "event_count": 1,
        "predicted_return_mean": 0.2,
        "realized_return_mean": 0.1,
        "realized_to_predicted_median": 0.5,
    }
    assert summary["freeze_audit"] == {
        "event_count": 2,
        "freeze_steps": [48, 49],
        "all_frozen_step": 49,
        "total_frozen": 118,
        "mode_counts": {"coordinate": 16, "spectral": 102},
        "post_freeze_route_events": 0,
        "post_freeze_probe_events": 0,
        "current_allocated_mib": 100.0,
        "current_reserved_mib": 120.0,
        "pre_freeze_tokens_per_second_median": 2500.0,
        "post_freeze_tokens_per_second_median": 2750.0,
    }
    assert len(summary["source_runs"]) == 15
    assert all("faro_fixed" not in source for source in summary["source_runs"])
    assert all("diagnostic250" not in source for source in summary["source_runs"])


def test_all_nine_figures_are_generated_from_twelve_formal_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    output = tmp_path / "output"
    _write_fixture(runs_root)
    runs = plot_results.select_formal_runs(
        load_runs(runs_root, run_ids=_formal_run_ids())
    )

    paths = plot_all(runs, output)

    assert {path.name for path in paths} == FIGURES
    assert all(path.stat().st_size > 0 for path in paths)


def test_selected_candidate_stat_series_reads_nested_route_statistics() -> None:
    routes = [
        {
            "step": 8,
            "mode": "spectral",
            "candidate_statistics": {
                "spectral": {"uwr": 0.04, "spectral_risk": 0.03}
            },
        },
        {
            "step": 8,
            "mode": "coordinate",
            "candidate_statistics": {
                "coordinate": {"uwr": 0.02, "spectral_risk": 0.01}
            },
        },
        {
            "step": 16,
            "mode": "spectral",
            "candidate_statistics": {
                "spectral": {"uwr": 0.03, "spectral_risk": 0.02}
            },
        },
    ]

    series = plot_results._selected_candidate_stat_series(routes)

    assert series["uwr"] == [(8, 0.03), (16, 0.03)]
    assert series["spectral_risk"] == [(8, 0.02), (16, 0.02)]
