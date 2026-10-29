from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .metrics import validate_event


VARIANTS = ("adam", "muon", "normuon", "faro", "faro_prior")
FORMAL_SEEDS = (0, 42, 1024)
# Each formal run is identified by the results root it lives in plus its run id.
# The main FARO experiment is the no-prior auto-switch run (geometry discovered
# from a coordinate start); the human-prior routing is kept as faro_prior control.
RESULTS_ROOTS = {
    "adam": "results/runs",
    "muon": "results/runs",
    "normuon": "results/runs",
    "faro": "results-coordstart/runs",
    "faro_prior": "results-sparse/runs",
}
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
# The main and control FARO experiments both run with metadata variant "faro".
METADATA_VARIANT = {
    "adam": "adam",
    "muon": "muon",
    "normuon": "normuon",
    "faro": "faro",
    "faro_prior": "faro",
}
# The main FARO run discovers geometry on its own, so it is not expected to route
# 100% of MLP to spectral; the human-prior control is expected to stay near 1.0.
MLP_SPECTRAL_RATIO_FLOOR = {"faro": 0.50, "faro_prior": 0.80}
DISPLAY_NAMES = {
    "adam": "Adam",
    "muon": "Muon",
    "normuon": "NorMuonAndAdam",
    "faro": "FARO no-prior auto-switch",
    "faro_prior": "FARO prior control",
}
MODES = ("coordinate", "row_column", "spectral")
FIGURE_NAMES = (
    "01-val-loss-vs-tokens.png",
    "02-val-loss-vs-time.png",
    "03-train-loss-ema.png",
    "04-geometry-mode-stack.png",
    "05-routing-heatmap.png",
    "06-risk-drift-calibration.png",
    "07-budget-and-switches.png",
    "08-throughput-overhead-memory.png",
    "09-final-val-seeds.png",
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    path: Path
    run_id: str
    variant: str
    seed: int
    events: tuple[dict[str, object], ...]
    metadata: dict[str, object]
    summary: dict[str, object]

    def events_of(self, event_type: str) -> list[dict[str, object]]:
        return [event for event in self.events if event["event"] == event_type]


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        event = json.loads(line)
        try:
            validate_event(event)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        events.append(event)
    return tuple(events)


def load_runs(
    results_root: Path | Sequence[Path], *, run_ids: Iterable[str] | None = None
) -> list[RunRecord]:
    roots = (
        (results_root,)
        if isinstance(results_root, (str, Path))
        else tuple(results_root)
    )
    included_run_ids = set(run_ids) if run_ids is not None else None
    runs: list[RunRecord] = []
    seen_paths: set[Path] = set()
    for root in roots:
        for metrics_path in sorted(Path(root).rglob("metrics.jsonl")):
            if metrics_path in seen_paths:
                continue
            seen_paths.add(metrics_path)
            if (
                included_run_ids is not None
                and metrics_path.parent.name not in included_run_ids
            ):
                continue
            events = _load_jsonl(metrics_path)
            run_events = [event for event in events if event["event"] == "run"]
            summaries = [event for event in events if event["event"] == "summary"]
            if len(run_events) != 1 or len(summaries) != 1:
                raise ValueError(f"{metrics_path} must contain one run and one summary")
            run_event = run_events[0]
            metadata = run_event["metadata"]
            if not isinstance(metadata, dict):
                raise TypeError(f"{metrics_path} run metadata must be an object")
            runs.append(
                RunRecord(
                    path=metrics_path.parent,
                    run_id=str(run_event["run_id"]),
                    variant=str(run_event["variant"]),
                    seed=int(run_event["seed"]),
                    events=events,
                    metadata=metadata,
                    summary=summaries[0],
                )
            )
    return runs


def select_formal_runs(runs: Sequence[RunRecord]) -> list[RunRecord]:
    selected: list[RunRecord] = []
    for variant, seeds in FORMAL_RUN_IDS.items():
        expected_metadata_variant = METADATA_VARIANT[variant]
        for seed, run_id in seeds.items():
            matches = [run for run in runs if run.run_id == run_id]
            if len(matches) != 1:
                raise ValueError(
                    f"formal run {run_id!r} must appear exactly once; found {len(matches)}"
                )
            run = matches[0]
            if run.variant != expected_metadata_variant:
                raise ValueError(
                    f"formal run {run_id!r} has variant {run.variant!r}, "
                    f"expected {expected_metadata_variant!r}"
                )
            if int(run.seed) != int(seed):
                raise ValueError(
                    f"formal run {run_id!r} has seed {run.seed!r}, expected {seed!r}"
                )
            # Re-stamp with the formal key: the main and control FARO experiments
            # share the metadata variant "faro", so downstream grouping must use
            # the formal key to keep them apart.
            selected.append(replace(run, variant=variant))
    return selected


def _structure_class(event: dict[str, object]) -> str:
    if "structure_class" in event:
        return str(event["structure_class"])
    block = str(event.get("block", ""))
    if block.startswith("mlp_bank"):
        return "mlp"
    if block.startswith(("qk_bank", "vo_bank")):
        return "attention"
    if block == "lm_head":
        return "lm_head"
    if block in {"embed", "bigram_embed", "value_embeds"}:
        return "embedding"
    return "coordinate"


def _freeze_events(run: RunRecord) -> list[dict[str, object]]:
    return [
        event
        for event in run.events_of("event")
        if event.get("kind") == "faro_freeze"
    ]


def audit_runs(
    runs: Sequence[RunRecord], *, require_formal: bool = True
) -> dict[str, object]:
    pairs = {(run.variant, run.seed) for run in runs}
    expected = {
        (variant, seed) for variant in VARIANTS for seed in FORMAL_SEEDS
    }
    if require_formal and pairs != expected:
        missing = sorted(expected - pairs)
        extra = sorted(pairs - expected)
        raise ValueError(f"formal run matrix mismatch; missing={missing}, extra={extra}")
    if len(pairs) != len(runs):
        raise ValueError("duplicate variant/seed runs are not allowed")
    incomplete = [
        str(run.path)
        for run in runs
        if run.summary.get("status") != "complete"
        or (require_formal and int(run.summary["step"]) != 1285)
    ]
    if incomplete:
        raise ValueError(f"incomplete runs: {incomplete}")

    fairness_fields = (
        "initialization_hash",
        "bigram_sign_table_hash",
        "data_manifest_hash",
    )
    fairness_match = True
    fairness_hashes: dict[str, dict[str, str]] = {}
    for seed in sorted({run.seed for run in runs}):
        seed_runs = [run for run in runs if run.seed == seed]
        seed_hashes: dict[str, str] = {}
        for field in fairness_fields:
            values = {str(run.metadata[field]) for run in seed_runs}
            fairness_match &= len(values) == 1
            if len(values) == 1:
                seed_hashes[field] = values.pop()
        fairness_hashes[str(seed)] = seed_hashes
    if not fairness_match:
        raise ValueError("cross-variant fairness hashes do not match")

    code_hashes = {
        variant: {
            str(run.seed): str(run.metadata["code_hash"])
            for run in sorted(
                (item for item in runs if item.variant == variant),
                key=lambda item: item.seed,
            )
        }
        for variant in VARIANTS
    }
    code_hashes_match = len(
        {
            code_hash
            for seed_hashes in code_hashes.values()
            for code_hash in seed_hashes.values()
        }
    ) == 1

    settings = {
        "seeds": tuple(sorted({run.seed for run in runs})),
        "world_size": int(runs[0].metadata["world_size"]),
        "grad_accum_steps": int(runs[0].metadata["grad_accum_steps"]),
        "max_steps": int(runs[0].metadata["config"]["max_steps"]),
        "matrix_lr": float(runs[0].metadata["config"]["matrix_lr"]),
        "muon_lr": float(runs[0].metadata["config"]["muon_lr"]),
    }
    for run in runs[1:]:
        candidate = {
            "world_size": int(run.metadata["world_size"]),
            "grad_accum_steps": int(run.metadata["grad_accum_steps"]),
            "max_steps": int(run.metadata["config"]["max_steps"]),
            "matrix_lr": float(run.metadata["config"]["matrix_lr"]),
            "muon_lr": float(run.metadata["config"]["muon_lr"]),
        }
        reference = {
            "world_size": settings["world_size"],
            "grad_accum_steps": settings["grad_accum_steps"],
            "max_steps": settings["max_steps"],
            "matrix_lr": settings["matrix_lr"],
            "muon_lr": settings["muon_lr"],
        }
        if candidate != reference:
            raise ValueError(
                f"formal run settings do not match: {run.run_id}: {candidate}"
            )

    mlp_routes = [
        event
        for run in runs
        if run.variant == "faro"
        for event in run.events_of("route")
        if int(event["step"]) >= 33 and _structure_class(event) == "mlp"
    ]
    spectral_routes = [
        event for event in mlp_routes if event["mode"] == "spectral"
    ]
    ratio = len(spectral_routes) / len(mlp_routes) if mlp_routes else 0.0
    if require_formal and ratio < MLP_SPECTRAL_RATIO_FLOOR["faro"]:
        raise ValueError(
            f"MLP spectral routing ratio {ratio:.3f} is below "
            f"{MLP_SPECTRAL_RATIO_FLOOR['faro']:.2f} for faro"
        )
    prior_mlp_routes = [
        event
        for run in runs
        if run.variant == "faro_prior"
        for event in run.events_of("route")
        if int(event["step"]) >= 33 and _structure_class(event) == "mlp"
    ]
    prior_spectral_routes = [
        event for event in prior_mlp_routes if event["mode"] == "spectral"
    ]
    prior_ratio = (
        len(prior_spectral_routes) / len(prior_mlp_routes)
        if prior_mlp_routes
        else 0.0
    )
    if require_formal and prior_ratio < MLP_SPECTRAL_RATIO_FLOOR["faro_prior"]:
        raise ValueError(
            f"MLP spectral routing ratio {prior_ratio:.3f} is below "
            f"{MLP_SPECTRAL_RATIO_FLOOR['faro_prior']:.2f} for faro_prior"
        )

    faro_runs = [run for run in runs if run.variant == "faro"]
    all_frozen_steps: list[int] = []
    post_freeze_route_events = 0
    post_freeze_probe_events = 0
    for run in faro_runs:
        completed_freezes = [
            event for event in _freeze_events(run) if event.get("all_routes_frozen")
        ]
        if require_formal and len(completed_freezes) != 1:
            raise ValueError(
                f"formal FARO run {run.run_id!r} must have one all-routes-frozen event"
            )
        if not completed_freezes:
            continue
        all_frozen_step = int(completed_freezes[0]["step"])
        all_frozen_steps.append(all_frozen_step)
        post_freeze_route_events += sum(
            int(event["step"]) > all_frozen_step for event in run.events_of("route")
        )
        post_freeze_probe_events += sum(
            int(event["step"]) > all_frozen_step for event in run.events_of("probe")
        )
    if require_formal and (post_freeze_route_events or post_freeze_probe_events):
        raise ValueError(
            "formal FARO run contains route or probe events after all routes froze"
        )
    return {
        "complete_runs": len(runs),
        "fairness_hashes_match": fairness_match,
        "fairness_hashes": fairness_hashes,
        "code_hashes_match": code_hashes_match,
        "code_hashes": code_hashes,
        "settings": settings,
        "mlp_routing_events": len(mlp_routes),
        "mlp_spectral_ratio": ratio,
        "prior_mlp_routing_events": len(prior_mlp_routes),
        "prior_mlp_spectral_ratio": prior_ratio,
        "all_routes_frozen": bool(all_frozen_steps),
        "all_frozen_step": max(all_frozen_steps) if all_frozen_steps else None,
        "post_freeze_route_events": post_freeze_route_events,
        "post_freeze_probe_events": post_freeze_probe_events,
    }


def _ema(values: Sequence[float], beta: float = 0.9) -> list[float]:
    result: list[float] = []
    current = 0.0
    for index, value in enumerate(values):
        current = value if index == 0 else beta * current + (1 - beta) * value
        result.append(current)
    return result


def aggregate_runs(runs: Sequence[RunRecord]) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    validation: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    train: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(dict)
    system: dict[str, dict[str, float]] = {}
    route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    route_events: list[dict[str, object]] = []
    probe_events: list[dict[str, object]] = []
    freeze_events: list[dict[str, object]] = []
    pre_freeze_throughputs: list[float] = []
    post_freeze_throughputs: list[float] = []
    post_freeze_route_events = 0
    post_freeze_probe_events = 0

    for variant in VARIANTS:
        variant_runs = sorted(
            (run for run in runs if run.variant == variant), key=lambda run: run.seed
        )
        if not variant_runs:
            continue
        seed_values = {
            str(run.seed): float(run.summary["final_validation_loss"])
            for run in variant_runs
        }
        final_values = list(seed_values.values())
        variants[variant] = {
            "final_validation_loss_mean": statistics.mean(final_values),
            "final_validation_loss_std": (
                statistics.stdev(final_values) if len(final_values) > 1 else 0.0
            ),
            "seed_values": seed_values,
        }
        throughput = []
        peak_allocated = []
        peak_reserved = []
        final_training_ms = []
        final_wall_ms = []
        for run in variant_runs:
            validation[variant][str(run.seed)] = run.events_of("validation")
            train_events = run.events_of("train")
            losses = [float(event["loss"]) for event in train_events]
            train[variant][str(run.seed)] = {
                "steps": [int(event["step"]) for event in train_events],
                "loss": losses,
                "loss_ema": _ema(losses),
            }
            if train_events:
                throughput.append(float(train_events[-1]["tokens_per_second"]))
                final_training_ms.append(float(train_events[-1]["training_time_ms"]))
                final_wall_ms.append(float(train_events[-1]["wall_time_ms"]))
            peak_allocated.append(float(run.summary["peak_allocated_mib"]))
            peak_reserved.append(float(run.summary["peak_reserved_mib"]))
            if run.variant == "faro" and run.seed == 0:
                run_freeze_events = _freeze_events(run)
                freeze_events.extend(run_freeze_events)
                completed_freezes = [
                    event
                    for event in run_freeze_events
                    if event.get("all_routes_frozen")
                ]
                all_frozen_step = (
                    int(completed_freezes[-1]["step"])
                    if completed_freezes
                    else None
                )
                if all_frozen_step is not None:
                    for previous, current in pairwise(train_events):
                        elapsed_ms = float(current["training_time_ms"]) - float(
                            previous["training_time_ms"]
                        )
                        if elapsed_ms <= 0.0:
                            continue
                        instant_throughput = (
                            (int(current["tokens"]) - int(previous["tokens"]))
                            * 1000.0
                            / elapsed_ms
                        )
                        target = (
                            pre_freeze_throughputs
                            if int(current["step"]) <= all_frozen_step
                            else post_freeze_throughputs
                        )
                        target.append(instant_throughput)
                    post_freeze_route_events += sum(
                        int(event["step"]) > all_frozen_step
                        for event in run.events_of("route")
                    )
                    post_freeze_probe_events += sum(
                        int(event["step"]) > all_frozen_step
                        for event in run.events_of("probe")
                    )
                for event in run.events_of("route"):
                    structure = _structure_class(event)
                    route_counts[structure][str(event["mode"])] += 1
                    route_events.append({**event, "structure_class": structure})
                probe_events.extend(run.events_of("probe"))
        system[variant] = {
            "tokens_per_second_mean": statistics.mean(throughput) if throughput else 0.0,
            "training_time_ms_mean": statistics.mean(final_training_ms) if final_training_ms else 0.0,
            "wall_time_ms_mean": statistics.mean(final_wall_ms) if final_wall_ms else 0.0,
            "peak_allocated_mib_mean": statistics.mean(peak_allocated),
            "peak_reserved_mib_mean": statistics.mean(peak_reserved),
        }

    mode_counts = Counter(str(event["mode"]) for event in route_events)
    budgets = [float(event["budget_scale"]) for event in route_events]
    predicted = [float(event["predicted_return"]) for event in probe_events]
    realized = [float(event["realized_return"]) for event in probe_events]
    realized_to_predicted = [
        actual / estimate
        for actual, estimate in zip(realized, predicted, strict=True)
        if estimate != 0.0
    ]
    completed_freezes = [
        event for event in freeze_events if event.get("all_routes_frozen")
    ]
    final_freeze = completed_freezes[-1] if completed_freezes else None
    return {
        "schema_version": 1,
        "variants": variants,
        "validation": dict(validation),
        "train": dict(train),
        "train_loss_ema_beta": 0.9,
        "routes": {
            structure: dict(counts) for structure, counts in route_counts.items()
        },
        "route_events": route_events,
        "probe_events": probe_events,
        "route_audit": {
            "event_count": len(route_events),
            "mode_counts": dict(mode_counts),
            "switch_count": sum(bool(event.get("switched")) for event in route_events),
            "budget_min": min(budgets) if budgets else 0.0,
            "budget_mean": statistics.mean(budgets) if budgets else 0.0,
            "budget_max": max(budgets) if budgets else 0.0,
        },
        "probe_audit": {
            "event_count": len(probe_events),
            "predicted_return_mean": statistics.mean(predicted) if predicted else 0.0,
            "realized_return_mean": statistics.mean(realized) if realized else 0.0,
            "realized_to_predicted_median": (
                statistics.median(realized_to_predicted)
                if realized_to_predicted
                else 0.0
            ),
        },
        "freeze_audit": {
            "event_count": len(freeze_events),
            "freeze_steps": sorted({int(event["step"]) for event in freeze_events}),
            "all_frozen_step": (
                int(final_freeze["step"]) if final_freeze is not None else None
            ),
            "total_frozen": (
                int(final_freeze["total_frozen"])
                if final_freeze is not None
                else 0
            ),
            "mode_counts": (
                dict(final_freeze["mode_counts"])
                if final_freeze is not None
                else {}
            ),
            "post_freeze_route_events": post_freeze_route_events,
            "post_freeze_probe_events": post_freeze_probe_events,
            "current_allocated_mib": (
                float(final_freeze["current_allocated_mib"])
                if final_freeze is not None
                else 0.0
            ),
            "current_reserved_mib": (
                float(final_freeze["current_reserved_mib"])
                if final_freeze is not None
                else 0.0
            ),
            "pre_freeze_tokens_per_second_median": (
                statistics.median(pre_freeze_throughputs)
                if pre_freeze_throughputs
                else 0.0
            ),
            "post_freeze_tokens_per_second_median": (
                statistics.median(post_freeze_throughputs)
                if post_freeze_throughputs
                else 0.0
            ),
        },
        "system": system,
        "source_runs": [str(run.path.resolve()) for run in runs],
    }


def _save(fig: plt.Figure, directory: Path, name: str) -> Path:
    path = directory / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# One colour per variant. The training-curve figures give each variant its own
# subplot with a mean line and a spread band, so five variants are legible where
# 15 overlaid seed curves were not.
VARIANT_COLORS = {
    "adam": "tab:blue",
    "muon": "tab:orange",
    "normuon": "tab:green",
    "faro": "tab:red",
    "faro_prior": "tab:purple",
}
DEFAULT_VARIANT_COLOR = "tab:gray"


def _mean_interval_by_step(
    seeds: Mapping[str, Sequence[Mapping[str, object]]],
    x_field: str,
    *,
    x_scale: float = 1.0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Seed mean and ±1 sample-std band, aligned on the shared event step."""

    loss_by_step: dict[int, list[float]] = defaultdict(list)
    x_by_step: dict[int, list[float]] = defaultdict(list)
    for events in seeds.values():
        for event in events:
            step = int(event["step"])
            loss_by_step[step].append(float(event["loss"]))
            x_by_step[step].append(float(event[x_field]) * x_scale)
    steps = sorted(loss_by_step)
    xs = [statistics.mean(x_by_step[step]) for step in steps]
    means = [statistics.mean(loss_by_step[step]) for step in steps]
    halfwidths = [
        statistics.stdev(loss_by_step[step]) if len(loss_by_step[step]) > 1 else 0.0
        for step in steps
    ]
    return (
        xs,
        means,
        [mean - half for mean, half in zip(means, halfwidths)],
        [mean + half for mean, half in zip(means, halfwidths)],
    )


def _train_mean_interval(
    seeds: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Seed mean and ±1 sample-std band for the per-seed train-loss EMA series."""

    series = list(seeds.values())
    steps = [int(step) for step in series[0]["steps"]]
    curves = [list(seed_series["loss_ema"]) for seed_series in series]
    means = [
        statistics.mean(curve[index] for curve in curves)
        for index in range(len(steps))
    ]
    halfwidths = [
        statistics.stdev(curve[index] for curve in curves)
        if len(curves) > 1
        else 0.0
        for index in range(len(steps))
    ]
    return (
        steps,
        means,
        [mean - half for mean, half in zip(means, halfwidths)],
        [mean + half for mean, half in zip(means, halfwidths)],
    )


def _draw_mean_interval(
    ax: plt.Axes,
    xs: Sequence[float],
    mean: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    color: str,
    seed_series: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
) -> None:
    for seed_x, seed_y in seed_series:
        ax.plot(seed_x, seed_y, color=color, alpha=0.30, linewidth=0.8)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.25, linewidth=0)
    ax.plot(xs, mean, color=color, linewidth=2.0)


def _variant_panels(
    summary: Mapping[str, object],
    *,
    section: str,
    x_field: str,
    x_scale: float,
    x_label: str,
    y_label: str,
    figure_name: str,
    figure_dir: Path,
    final_label: str,
    zoom_points: int | None = None,
    zoom_frac: float | None = None,
) -> Path:
    """Small multiples: one subplot per variant, mean line with a ±1 std band.

    ``section`` is "validation" (loss at checkpoints) or "train" (per-step loss
    EMA). Each variant panel also shows its individual seed curves faintly so the
    band's basis is visible. A second row repeats the panels over a trailing
    window (``zoom_points`` trailing checkpoints, or ``zoom_frac`` trailing
    x-range) so the seed band and cross-variant gaps stay legible instead of
    being flattened by the step-0 drop.
    """

    data = summary[section]
    variants = list(data)
    zooming = zoom_points is not None or zoom_frac is not None
    rows = 2 if zooming else 1
    fig, axes = plt.subplots(
        rows,
        len(variants),
        figsize=(3.2 * len(variants), 3.4 * rows),
        squeeze=False,
    )
    drawn: list[tuple[list[float], list[float], list[float], list[float], str]] = []
    for column, variant in enumerate(variants):
        color = VARIANT_COLORS.get(variant, DEFAULT_VARIANT_COLOR)
        seeds = data[variant]
        if section == "train":
            xs, mean, lo, hi = _train_mean_interval(seeds)
            seed_series = [
                (list(series["steps"]), list(series["loss_ema"]))
                for series in seeds.values()
            ]
        else:
            xs, mean, lo, hi = _mean_interval_by_step(
                seeds, x_field, x_scale=x_scale
            )
            seed_series = [
                ([event[x_field] * x_scale for event in events],
                 [event["loss"] for event in events])
                for events in seeds.values()
            ]
        drawn.append((xs, mean, lo, hi, color))
        for row in range(rows):
            ax = axes[row][column]
            _draw_mean_interval(ax, xs, mean, lo, hi, color, seed_series)
            ax.tick_params(labelsize=7)
            if row == rows - 1:
                ax.set_xlabel(x_label, fontsize=8)
        axes[0][column].set_title(
            f"{DISPLAY_NAMES[variant]}\n{final_label} {mean[-1]:.4f}",
            fontsize=8,
        )
    if zooming:
        # The top row keeps the full range (sharex is off so the zoom below
        # cannot clip it); the bottom row zooms both axes onto the final segment
        # so the seed band and cross-variant gaps are actually visible.
        for column, (xs, _mean, lo, hi, _color) in enumerate(drawn):
            if zoom_points is not None:
                left = xs[max(0, len(xs) - zoom_points)]
            else:
                left = xs[0] + (1.0 - zoom_frac) * (xs[-1] - xs[0])
            axes[1][column].set_xlim(left, xs[-1])
            visible = [
                (x, l, h)
                for x, l, h in zip(xs, lo, hi)
                if x >= left
            ]
            if visible:
                ymin = min(l for _, l, _ in visible)
                ymax = max(h for _, _, h in visible)
                pad = max(0.005, 0.15 * (ymax - ymin))
                axes[1][column].set_ylim(ymin - pad, ymax + pad)
        axes[1][0].set_ylabel(f"{y_label} (zoomed)", fontsize=8)
    axes[0][0].set_ylabel(y_label, fontsize=8)
    fig.suptitle(
        "one panel per optimizer; bold = 3-seed mean, band = +/-1 std, "
        f"thin = individual seeds{'; bottom row zooms the final segment' if zooming else ''}",
        fontsize=8,
    )
    return _save(fig, figure_dir, figure_name)


def _selected_candidate_stat_series(
    routes: Sequence[dict[str, object]],
) -> dict[str, list[tuple[int, float]]]:
    by_field: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in routes:
        candidates = event.get("candidate_statistics")
        if not isinstance(candidates, dict):
            continue
        selected = candidates.get(str(event.get("mode")))
        if not isinstance(selected, dict):
            continue
        step = int(event["step"])
        for field in (
            "cvar",
            "drift",
            "uwr",
            "spectral_risk",
            "gradient_sv_concentration",
            "momentum_sv_concentration",
        ):
            if field in selected:
                by_field[field][step].append(float(selected[field]))
    return {
        field: [
            (step, statistics.median(values))
            for step, values in sorted(step_values.items())
        ]
        for field, step_values in by_field.items()
    }


def plot_all(runs: Sequence[RunRecord], output_root: Path) -> list[Path]:
    summary = aggregate_runs(runs)
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        _variant_panels(
            summary,
            section="validation",
            x_field="tokens",
            x_scale=1.0,
            x_label="training tokens",
            y_label="validation loss",
            final_label="final val",
            zoom_points=3,
            figure_name=FIGURE_NAMES[0],
            figure_dir=figure_dir,
        )
    )
    paths.append(
        _variant_panels(
            summary,
            section="validation",
            x_field="training_time_ms",
            x_scale=1.0 / 1000.0,
            x_label="training time (s)",
            y_label="validation loss",
            final_label="final val",
            zoom_points=3,
            figure_name=FIGURE_NAMES[1],
            figure_dir=figure_dir,
        )
    )
    paths.append(
        _variant_panels(
            summary,
            section="train",
            x_field="step",
            x_scale=1.0,
            x_label="step",
            y_label="train loss EMA",
            final_label="last EMA",
            zoom_frac=0.2,
            figure_name=FIGURE_NAMES[2],
            figure_dir=figure_dir,
        )
    )

    # Routing-analysis figures use the seed-0 FARO run as the representative;
    # mixing all seeds would alias routing events across runs on the same axes.
    seed0_faro = next(
        run for run in runs if run.variant == "faro" and run.seed == 0
    )
    routes0 = seed0_faro.events_of("route")

    fig, ax = plt.subplots()
    structures = sorted(summary["routes"])
    bottom = np.zeros(len(structures))
    for mode in MODES:
        values = np.array(
            [summary["routes"][structure].get(mode, 0) for structure in structures]
        )
        ax.bar(structures, values, bottom=bottom, label=mode)
        bottom += values
    ax.set(ylabel="routing event count")
    ax.legend()
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[3]))

    fig, ax = plt.subplots()
    routes = routes0
    blocks = sorted({str(event["block"]) for event in routes})
    steps = sorted({int(event["step"]) for event in routes})
    heat = np.full((len(blocks), len(steps)), np.nan)
    mode_code = {mode: index for index, mode in enumerate(MODES)}
    for event in routes:
        heat[blocks.index(str(event["block"])), steps.index(int(event["step"]))] = mode_code[str(event["mode"])]
    if heat.size:
        image = ax.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=len(MODES) - 1,
        )
        colorbar = fig.colorbar(image, ax=ax, ticks=range(len(MODES)))
        colorbar.ax.set_yticklabels(MODES)
        ax.set_xticks(range(len(steps)), [str(step) for step in steps], rotation=45)
    ax.set(xlabel="routing step", ylabel="functional block")
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[4]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    route_groups = (
        (
            "dense selected candidate",
            [
                event
                for event in routes0
                if _structure_class(event) in {"attention", "mlp"}
            ],
        ),
        (
            "coordinate/token selected candidate",
            [
                event
                for event in routes0
                if _structure_class(event) not in {"attention", "mlp"}
            ],
        ),
    )
    for ax, (title, events) in zip(axes[:2], route_groups, strict=True):
        for field, points in _selected_candidate_stat_series(events).items():
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                markersize=2,
                label=field,
            )
        ax.set(title=title, xlabel="step", ylabel="median statistic")
        if ax.lines:
            ax.legend(fontsize=5)
    probe_events = seed0_faro.events_of("probe")
    axes[2].plot(
        [event["step"] for event in probe_events],
        [event["predicted_return"] for event in probe_events],
        marker="o",
        label="predicted return",
    )
    axes[2].plot(
        [event["step"] for event in probe_events],
        [event["realized_return"] for event in probe_events],
        marker="o",
        label="realized return",
    )
    axes[2].set(title="probe calibration", xlabel="step", ylabel="return")
    drift_axis = axes[2].twinx()
    drift_axis.plot(
        [event["step"] for event in probe_events],
        [event["drift"] for event in probe_events],
        marker="x",
        linestyle="--",
        color="black",
        label="probe drift",
    )
    drift_axis.set_ylabel("drift")
    handles, labels = axes[2].get_legend_handles_labels()
    drift_handles, drift_labels = drift_axis.get_legend_handles_labels()
    axes[2].legend(handles + drift_handles, labels + drift_labels, fontsize=5)
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[5]))

    fig, ax = plt.subplots()
    for block in blocks:
        events = [event for event in routes0 if event["block"] == block]
        ax.plot(
            [event["step"] for event in events],
            [event["budget_scale"] for event in events],
            label=block,
        )
        switches = [event for event in events if event.get("switched")]
        ax.scatter(
            [event["step"] for event in switches],
            [event["budget_scale"] for event in switches],
            s=8,
        )
    ax.set(xlabel="step", ylabel="budget scale")
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[6]))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    variants = list(summary["system"])
    labels = [DISPLAY_NAMES[variant] for variant in variants]
    positions = np.arange(len(variants))
    axes[0].bar(positions, [summary["system"][variant]["tokens_per_second_mean"] for variant in variants])
    axes[0].set_ylabel("tokens/s")
    axes[1].bar(positions, [summary["system"][variant]["wall_time_ms_mean"] / 1000 for variant in variants])
    axes[1].set_ylabel("wall time (s)")
    axes[2].bar(positions, [summary["system"][variant]["peak_reserved_mib_mean"] for variant in variants])
    axes[2].set_ylabel("peak reserved MiB")
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=7)
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[7]))

    fig, ax = plt.subplots()
    variants = list(summary["variants"])
    positions = np.arange(len(variants))
    seed_colors = {seed: color for seed, color in zip(
        FORMAL_SEEDS, ("tab:blue", "tab:orange", "tab:green")
    )}
    for seed in FORMAL_SEEDS:
        values = [
            summary["variants"][variant]["seed_values"].get(str(seed))
            for variant in variants
        ]
        ax.scatter(
            positions,
            values,
            s=45,
            color=seed_colors[seed],
            label=f"seed {seed}",
            zorder=3,
        )
    means = [
        summary["variants"][variant]["final_validation_loss_mean"]
        for variant in variants
    ]
    ax.scatter(positions, means, s=90, marker="x", color="black", label="mean", zorder=4)
    ax.set_xticks(
        positions,
        [DISPLAY_NAMES[variant] for variant in variants],
        rotation=25,
        ha="right",
    )
    all_values = [
        value
        for variant in variants
        for value in summary["variants"][variant]["seed_values"].values()
    ]
    margin = max(0.02, (max(all_values) - min(all_values)) * 0.15)
    ax.set_ylim(min(all_values) - margin, max(all_values) + margin)
    ax.set_ylabel("final validation loss")
    ax.legend(fontsize=6)
    paths.append(_save(fig, figure_dir, FIGURE_NAMES[8]))
    return paths


def _write_summary(summary: dict[str, object], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "summary.json"
    temporary = output_root / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def resolve_results_roots(base: Path) -> list[Path]:
    """Map the per-variant results roots onto a single base directory.

    ``base`` is the FARO directory that holds ``results/``, ``results-sparse/``
    and ``results-coordstart/``. The main and control FARO experiments live in
    separate roots, so one root is not enough to select the formal matrix.
    """

    return [base / relative for relative in dict.fromkeys(RESULTS_ROOTS.values())]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--results", type=Path, required=True, nargs="+")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--results", type=Path, required=True, nargs="+")
    build_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = select_formal_runs(
        load_runs(
            tuple(args.results),
            run_ids=[
                run_id
                for seeds in FORMAL_RUN_IDS.values()
                for run_id in seeds.values()
            ],
        )
    )
    audit = audit_runs(runs, require_formal=True)
    if args.command == "audit":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    summary = aggregate_runs(runs)
    summary["audit"] = audit
    _write_summary(summary, args.output)
    plot_all(runs, args.output)


if __name__ == "__main__":
    main()
