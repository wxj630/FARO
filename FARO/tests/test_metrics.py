import json
from pathlib import Path

import pytest
import torch
import FARO.metrics as faro_metrics

from FARO.metrics import (
    ActivationObserver,
    DeviceMetricBuffer,
    JsonlMetricWriter,
    summarize_runs,
    validate_event,
)


def _base(event: str, **payload):
    return {
        "schema_version": 1,
        "event": event,
        "run_id": "run",
        "variant": "faro",
        "seed": 0,
        "step": 1,
        **payload,
    }


@pytest.mark.parametrize(
    "event",
    [
        _base("run", metadata={}),
        _base(
            "train",
            loss=4.0,
            tokens=100,
            training_time_ms=10.0,
            wall_time_ms=11.0,
            tokens_per_second=1000.0,
        ),
        _base("validation", loss=3.5, tokens=1000, training_time_ms=20.0),
        _base(
            "route",
            block="b",
            mode="spectral",
            budget_scale=1.0,
            scores={"spectral": 1.0},
        ),
        _base(
            "probe",
            realized_return=0.1,
            predicted_return=0.2,
            drift=0.01,
            counterfactual_scales=(0.9, 1.0, 1.1),
            counterfactual_losses=(3.21, 3.20, 3.22),
            counterfactual_returns=(0.01, 0.02, 0.0),
            counterfactual_selected_scale=1.0,
            counterfactual_applied_scale=1.0,
            counterfactual_selection_used_next_update=True,
            counterfactual_restore_verified=True,
        ),
        _base(
            "frozen_update",
            block="b",
            mode="spectral",
            budget_scale=0.4,
            projected=True,
            nominal_budget=1.0,
            effective_budget=0.4,
            reference_uwr=1e-5,
            update_uwr=0.1,
            effective_uwr=4e-5,
        ),
        _base("event", kind="nan", message="non-finite"),
        _base(
            "summary",
            status="complete",
            final_validation_loss=3.2,
            peak_allocated_mib=10.0,
            peak_reserved_mib=12.0,
        ),
    ],
)
def test_schema_accepts_every_event_type(event: dict) -> None:
    validate_event(event)


def test_schema_rejects_missing_base_and_event_fields() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        validate_event({"event": "train"})
    with pytest.raises(ValueError, match="tokens_per_second"):
        validate_event(
            _base(
                "train",
                loss=4.0,
                tokens=100,
                training_time_ms=10.0,
                wall_time_ms=11.0,
            )
        )


def test_jsonl_writer_accepts_frozen_update_audit(tmp_path: Path) -> None:
    writer = JsonlMetricWriter(
        tmp_path / "metrics.jsonl",
        {"run_id": "run", "variant": "faro", "seed": 0, "metadata": {}},
    )
    record = writer.write_event(
        "frozen_update",
        step=48,
        block="b",
        mode="spectral",
        budget_scale=0.4,
        projected=True,
        nominal_budget=1.0,
        effective_budget=0.4,
        reference_uwr=1e-5,
        update_uwr=0.1,
        effective_uwr=4e-5,
        reward_engine=True,
        reward_edge=1.0,
        reward_variance=0.1,
        reward_target=0.9,
        risk_capacity=0.9,
        counterfactual_scales=(0.9, 1.0, 1.1),
        counterfactual_losses=(3.21, 3.20, 3.22),
        counterfactual_returns=(0.01, 0.02, 0.0),
        counterfactual_selected_scale=1.0,
        counterfactual_applied_scale=1.0,
        counterfactual_selection_used_next_update=True,
        counterfactual_restore_verified=True,
        kelly_state="defensive",
        kelly_scale=1.0,
        kelly_steps_remaining=0,
        kelly_projection_multiple=3.0,
    )
    writer.close(
        step=48,
        status="complete",
        final_validation_loss=3.2,
        peak_allocated_mib=10.0,
        peak_reserved_mib=12.0,
    )
    assert record["event"] == "frozen_update"


def test_jsonl_writer_preserves_generic_faro_freeze_audit_payload(
    tmp_path: Path,
) -> None:
    writer = JsonlMetricWriter(
        tmp_path / "metrics.jsonl",
        {"run_id": "run", "variant": "faro", "seed": 0, "metadata": {}},
    )
    event = writer.write_event(
        "event",
        step=48,
        kind="faro_freeze",
        message="FARO routes frozen",
        newly_frozen=3,
        total_frozen=8,
        mode_counts={"coordinate": 2, "row_column": 3, "spectral": 3},
        all_routes_frozen=True,
        current_allocated_mib=123.5,
        current_reserved_mib=256.0,
    )

    assert event["schema_version"] == 1
    assert event["event"] == "event"
    assert event["kind"] == "faro_freeze"
    assert event["newly_frozen"] == 3
    assert event["total_frozen"] == 8
    assert event["mode_counts"] == {"coordinate": 2, "row_column": 3, "spectral": 3}
    assert event["all_routes_frozen"] is True
    assert event["current_allocated_mib"] == 123.5
    assert event["current_reserved_mib"] == 256.0


def test_activation_observer_records_vectorized_rms() -> None:
    observer = ActivationObserver(num_blocks=3, device=torch.device("cpu"))
    inputs = torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])
    outputs = torch.tensor([[[0.0, 5.0]], [[6.0, 8.0]]])

    observer.record(torch.tensor([0, 2]), inputs, outputs)
    snapshot = observer.snapshot()

    torch.testing.assert_close(
        snapshot["input_rms"][[0, 2]],
        torch.tensor([(12.5) ** 0.5, 2.0**0.5]),
    )
    torch.testing.assert_close(
        snapshot["output_rms"][[0, 2]],
        torch.tensor([(12.5) ** 0.5, 50.0**0.5]),
    )
    assert snapshot["write_count"].tolist() == [1, 0, 1]


def test_activation_observer_can_skip_non_probe_forwards() -> None:
    observer = ActivationObserver(num_blocks=2, device=torch.device("cpu"))
    indices = torch.tensor([0, 1])
    values = torch.ones(2, 1, 2)

    observer.disable()
    observer.record(indices, values, values)
    assert observer.snapshot()["write_count"].tolist() == [0, 0]

    observer.enable()
    observer.record(indices, values, values)
    assert observer.snapshot()["write_count"].tolist() == [1, 1]


def test_activation_output_drift_is_kept_per_observed_block() -> None:
    before = {
        "output_rms": torch.tensor([2.0, 4.0, 8.0]),
        "write_count": torch.tensor([1, 0, 2]),
    }
    after = {
        "output_rms": torch.tensor([3.0, 9.0, 6.0]),
        "write_count": torch.tensor([2, 0, 3]),
    }

    drift = faro_metrics.activation_output_drift(before, after)

    torch.testing.assert_close(drift, torch.tensor([1.0, 0.0, 2.0]))


def test_device_metric_buffer_append_does_not_call_tensor_item(monkeypatch) -> None:
    buffer = DeviceMetricBuffer(capacity=2, device=torch.device("cpu"))

    def forbidden_item(self):
        raise AssertionError("append must not call Tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    buffer.append_train(
        step=1,
        loss=torch.tensor(4.0),
        tokens=100,
        training_time_ms=torch.tensor(10.0),
        wall_time_ms=torch.tensor(11.0),
        tokens_per_second=torch.tensor(1000.0),
    )
    assert buffer.length == 1


def test_device_metric_buffer_drains_in_order() -> None:
    buffer = DeviceMetricBuffer(capacity=3, device=torch.device("cpu"))
    for step in (1, 2):
        buffer.append_train(
            step=step,
            loss=torch.tensor(4.0 - step),
            tokens=step * 100,
            training_time_ms=torch.tensor(step * 10.0),
            wall_time_ms=torch.tensor(step * 11.0),
            tokens_per_second=torch.tensor(1000.0),
        )

    records = buffer.drain()

    assert [record["step"] for record in records] == [1, 2]
    assert [record["tokens"] for record in records] == [100, 200]
    assert buffer.length == 0


def test_jsonl_writer_round_trips_and_writes_atomic_summary(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    writer = JsonlMetricWriter(
        metrics_path,
        {"run_id": "run", "variant": "faro", "seed": 0, "metadata": {}},
    )
    writer.write_event(
        "validation", step=2, loss=3.5, tokens=1000, training_time_ms=20.0
    )
    writer.close(
        step=2,
        status="complete",
        final_validation_loss=3.5,
        peak_allocated_mib=10.0,
        peak_reserved_mib=12.0,
    )

    events = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["run", "validation", "summary"]
    assert json.loads((tmp_path / "summary.json").read_text()) == events[-1]
    assert not (tmp_path / "summary.json.tmp").exists()


def test_jsonl_writer_batches_route_records_with_one_flush(
    tmp_path: Path,
) -> None:
    writer = JsonlMetricWriter(
        tmp_path / "metrics.jsonl",
        {"run_id": "run", "variant": "faro", "seed": 0, "metadata": {}},
    )
    original_file = writer._file

    class CountingFile:
        def __init__(self) -> None:
            self.flushes = 0

        def write(self, value: str) -> int:
            return original_file.write(value)

        def flush(self) -> None:
            self.flushes += 1
            original_file.flush()

    counting_file = CountingFile()
    writer._file = counting_file  # type: ignore[assignment]
    records = [
        {
            "event": "route",
            "step": 8,
            "block": f"dense.{index}",
            "mode": "spectral",
            "budget_scale": 1.0,
            "scores": {"spectral": 1.0},
        }
        for index in range(3)
    ]

    writer.write_events(records)

    assert counting_file.flushes == 1


def test_fixed_two_seed_fixture_has_known_mean_and_sample_std() -> None:
    fixture = Path(__file__).parent / "fixtures" / "metrics_two_seed.jsonl"

    summary = summarize_runs([fixture])

    assert summary["adam"]["final_validation_loss_mean"] == pytest.approx(3.3)
    assert summary["adam"]["final_validation_loss_std"] == pytest.approx(
        2**0.5 / 10
    )
    assert summary["adam"]["seed_values"] == {0: 3.2, 1: 3.4}
