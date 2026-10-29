from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


SCHEMA_VERSION = 1
_BASE_FIELDS = {"schema_version", "event", "run_id", "variant", "seed", "step"}
_EVENT_FIELDS = {
    "run": {"metadata"},
    "train": {
        "loss",
        "tokens",
        "training_time_ms",
        "wall_time_ms",
        "tokens_per_second",
    },
    "validation": {"loss", "tokens", "training_time_ms"},
    "route": {"block", "mode", "budget_scale", "scores"},
    "frozen_update": {
        "block",
        "mode",
        "budget_scale",
        "projected",
        "nominal_budget",
        "effective_budget",
        "reference_uwr",
        "update_uwr",
        "effective_uwr",
    },
    "probe": {"realized_return", "predicted_return", "drift"},
    "event": {"kind", "message"},
    "summary": {
        "status",
        "final_validation_loss",
        "peak_allocated_mib",
        "peak_reserved_mib",
    },
}


def validate_event(event: Mapping[str, object]) -> None:
    missing_base = sorted(_BASE_FIELDS - event.keys())
    if missing_base:
        raise ValueError(f"missing event fields: {', '.join(missing_base)}")
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    event_type = event["event"]
    if event_type not in _EVENT_FIELDS:
        raise ValueError(f"unknown event type: {event_type!r}")
    missing = sorted(_EVENT_FIELDS[event_type] - event.keys())
    if missing:
        raise ValueError(
            f"missing {event_type} fields: {', '.join(missing)}"
        )


class ActivationObserver(nn.Module):
    def __init__(self, num_blocks: int, device: torch.device):
        super().__init__()
        self.enabled = True
        self.register_buffer(
            "input_rms", torch.zeros(num_blocks, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "output_rms", torch.zeros(num_blocks, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "write_count", torch.zeros(num_blocks, dtype=torch.int64, device=device)
        )

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @torch.no_grad()
    def record(self, block_indices: Tensor, inputs: Tensor, outputs: Tensor) -> None:
        if not self.enabled:
            return
        indices = block_indices.to(device=self.input_rms.device, dtype=torch.int64)
        count = indices.numel()
        if inputs.shape[0] != count or outputs.shape[0] != count:
            raise ValueError("one input and output summary is required per block index")
        input_rms = inputs.float().reshape(count, -1).square().mean(1).sqrt()
        output_rms = outputs.float().reshape(count, -1).square().mean(1).sqrt()
        self.input_rms.index_copy_(0, indices, input_rms)
        self.output_rms.index_copy_(0, indices, output_rms)
        self.write_count.index_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=self.write_count.dtype),
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.input_rms.zero_()
        self.output_rms.zero_()
        self.write_count.zero_()

    def snapshot(self) -> dict[str, Tensor]:
        return {
            "input_rms": self.input_rms.detach().clone(),
            "output_rms": self.output_rms.detach().clone(),
            "write_count": self.write_count.detach().clone(),
        }


def activation_output_drift(
    before: Mapping[str, Tensor], after: Mapping[str, Tensor]
) -> Tensor:
    before_output = before["output_rms"]
    after_output = after["output_rms"]
    before_count = before["write_count"]
    after_count = after["write_count"]
    if (
        before_output.shape != after_output.shape
        or before_count.shape != before_output.shape
        or after_count.shape != before_output.shape
    ):
        raise ValueError("activation snapshots must have matching block shapes")
    observed = after_count > before_count
    return torch.where(observed, (after_output - before_output).abs(), 0.0)


class DeviceMetricBuffer:
    def __init__(self, capacity: int, device: torch.device):
        if capacity < 1:
            raise ValueError("metric buffer capacity must be positive")
        self.capacity = capacity
        self.device = device
        self.length = 0
        self.steps = torch.empty(capacity, dtype=torch.int64, device=device)
        self.losses = torch.empty(capacity, dtype=torch.float32, device=device)
        self.tokens = torch.empty(capacity, dtype=torch.int64, device=device)
        self.training_time_ms = torch.empty(
            capacity, dtype=torch.float64, device=device
        )
        self.wall_time_ms = torch.empty(capacity, dtype=torch.float64, device=device)
        self.tokens_per_second = torch.empty(
            capacity, dtype=torch.float64, device=device
        )

    def append_train(
        self,
        *,
        step: int | Tensor,
        loss: Tensor,
        tokens: int | Tensor,
        training_time_ms: float | Tensor,
        wall_time_ms: float | Tensor,
        tokens_per_second: float | Tensor,
    ) -> None:
        if self.length >= self.capacity:
            raise BufferError("device metric buffer is full")
        index = self.length
        self.steps[index].copy_(torch.as_tensor(step, device=self.device))
        self.losses[index].copy_(loss.detach().to(self.device, dtype=torch.float32))
        self.tokens[index].copy_(torch.as_tensor(tokens, device=self.device))
        self.training_time_ms[index].copy_(
            torch.as_tensor(training_time_ms, device=self.device)
        )
        self.wall_time_ms[index].copy_(
            torch.as_tensor(wall_time_ms, device=self.device)
        )
        self.tokens_per_second[index].copy_(
            torch.as_tensor(tokens_per_second, device=self.device)
        )
        self.length += 1

    def drain(self) -> list[dict[str, int | float]]:
        length = self.length
        if length == 0:
            return []
        columns = {
            "step": self.steps[:length].detach().cpu().tolist(),
            "loss": self.losses[:length].detach().cpu().tolist(),
            "tokens": self.tokens[:length].detach().cpu().tolist(),
            "training_time_ms": self.training_time_ms[:length]
            .detach()
            .cpu()
            .tolist(),
            "wall_time_ms": self.wall_time_ms[:length].detach().cpu().tolist(),
            "tokens_per_second": self.tokens_per_second[:length]
            .detach()
            .cpu()
            .tolist(),
        }
        self.length = 0
        return [
            {name: values[index] for name, values in columns.items()}
            for index in range(length)
        ]


class JsonlMetricWriter:
    def __init__(self, path: Path, run_metadata: Mapping[str, object]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_metadata["run_id"])
        self.variant = str(run_metadata["variant"])
        self.seed = int(run_metadata["seed"])
        self._file = self.path.open("w", encoding="utf-8")
        self._closed = False
        metadata = run_metadata.get("metadata", dict(run_metadata))
        self.write_event("run", step=0, metadata=metadata)

    def _base_event(self, event: str, step: int) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "run_id": self.run_id,
            "variant": self.variant,
            "seed": self.seed,
            "step": int(step),
        }

    def write_event(self, event: str, *, step: int, **payload: object) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("metric writer is closed")
        record = self._base_event(event, step)
        record.update(payload)
        validate_event(record)
        self._file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if event in {"validation", "route", "frozen_update", "probe", "event", "summary"}:
            self._file.flush()
        return record

    def write_events(
        self, records: Iterable[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if self._closed:
            raise RuntimeError("metric writer is closed")
        written: list[dict[str, object]] = []
        for source in records:
            event = str(source["event"])
            step = int(source["step"])
            record = self._base_event(event, step)
            record.update(
                {
                    key: value
                    for key, value in source.items()
                    if key not in {"event", "step"}
                }
            )
            validate_event(record)
            self._file.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )
            written.append(record)
        self._file.flush()
        return written

    def flush_device_buffer(self, buffer: DeviceMetricBuffer) -> None:
        for record in buffer.drain():
            step = int(record.pop("step"))
            self.write_event("train", step=step, **record)
        self._file.flush()

    def close(self, *, step: int, **summary: object) -> dict[str, object]:
        record = self.write_event("summary", step=step, **summary)
        self._file.flush()
        self._file.close()
        self._closed = True
        temporary = self.path.with_name("summary.json.tmp")
        destination = self.path.with_name("summary.json")
        temporary.write_text(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return record


def _read_events(paths: Iterable[Path]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            validate_event(event)
            events.append(event)
    return events


def summarize_runs(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    summaries = [
        event for event in _read_events(paths) if event["event"] == "summary"
    ]
    by_variant: dict[str, list[dict[str, object]]] = {}
    for summary in summaries:
        by_variant.setdefault(str(summary["variant"]), []).append(summary)
    result: dict[str, dict[str, object]] = {}
    for variant, variant_summaries in by_variant.items():
        seed_values = {
            int(summary["seed"]): float(summary["final_validation_loss"])
            for summary in variant_summaries
        }
        values = list(seed_values.values())
        result[variant] = {
            "final_validation_loss_mean": statistics.mean(values),
            "final_validation_loss_std": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
            "seed_values": seed_values,
        }
    return result


def summarize_run(path: Path) -> dict[str, object]:
    events = _read_events([path])
    run_ids = {str(event["run_id"]) for event in events}
    if len(run_ids) != 1:
        raise ValueError("summarize_run requires exactly one run id")
    summaries = [event for event in events if event["event"] == "summary"]
    if len(summaries) != 1:
        raise ValueError("run must contain exactly one summary event")
    return summaries[0]
