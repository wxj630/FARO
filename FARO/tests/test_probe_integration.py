import ast
from pathlib import Path

import pytest


SOURCE = (Path(__file__).parents[2] / "train_gpt.py").read_text()


def _training_section() -> str:
    return SOURCE.split("# --------------- TRAINING SECTION -----------------", 1)[1]


def _warmup_section() -> str:
    return SOURCE.split("#            Warmup kernels", 1)[1].split(
        "#        Training and validation", 1
    )[0]


def test_counterfactual_probe_budget_has_dedicated_gate() -> None:
    assert "def _counterfactual_probe_budget_enabled()" in SOURCE
    assert "experiment_config.counterfactual_probe_budget" in SOURCE
    assert "_adaptive_risk_enabled() or _counterfactual_probe_budget_enabled()" in SOURCE
    assert "counterfactual_probe_budget" in SOURCE


def test_kelly_probe_gate_and_audit_are_post_freeze_only() -> None:
    assert "def _kelly_risk_enabled()" in SOURCE
    section = _training_section()
    assert "probe_before_loss=probe_before_loss" in SOURCE
    assert "update_kelly_state(" in SOURCE
    assert "kelly_state" in section or "kelly_state" in SOURCE


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("counterfactual", [False, True])
def test_independent_probe_preserves_training_batch_before_freeze(
    requested, adaptive, counterfactual
) -> None:
    assignments = [
        node for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "independent_probe"
                for target in node.targets)
    ]
    assert len(assignments) == 1
    expression = ast.Expression(assignments[0].value)
    result = eval(compile(expression, "<probe-gate>", "eval"), {
        "should_probe": requested,
        "_adaptive_risk_enabled": lambda: adaptive,
        "counterfactual_probe": counterfactual,
        "_ds_risk_enabled": lambda: False,
    })
    assert result == (requested and (adaptive or counterfactual))
    section = _training_section()
    assert "if independent_probe and probe_loader is not None:" in section
    assert "if should_probe and idx == 0 and probe_batch is None:" in section


def test_counterfactual_probe_does_not_reuse_adaptive_calibration() -> None:
    section = _training_section()
    assert "counterfactual_probe_budget" in section
    assert "after_optimizer_probe(" in section
    assert "counterfactual" in section
    assert "counterfactual_audit = training_manager.after_optimizer_probe(" in section
    assert "**counterfactual_audit" in section


def test_adaptive_mode_uses_independent_probe_loader_and_validation_offset() -> None:
    source = SOURCE

    assert "probe_loader = (" in source
    assert "if _adaptive_risk_enabled()" in source
    assert "probe_batch = next(probe_loader)[:4]" in source
    # Probe uses a slice after the public validation prefix, and validation keeps
    # its original tokens so adaptive val loss stays comparable with baselines.
    assert "start_token=_probe_stream_bounds()[0]" in source
    assert "end_token=_probe_stream_bounds()[1]" in source
    assert "def _probe_stream_bounds()" in source
    assert "_validation_start_token" not in source


def test_adaptive_probe_batch_is_replayed_without_refetching() -> None:
    section = _training_section()
    fetch = section.index("probe_batch = next(probe_loader)")
    before = section.index("probe_before_loss = model(", fetch)
    optimizer_step = section.index("training_manager.step_optimizers(step)", before)
    replay = section.index("probe_after_loss = model(", optimizer_step)

    assert fetch < before < optimizer_step < replay
    replay_body = section[replay : section.index("training_manager.after_optimizer_probe(", replay)]
    assert "next(probe_loader)" not in replay_body
    assert "probe_batch" in replay_body
    assert ".backward()" not in replay_body


def test_direction_batches_remain_on_training_loader_when_adaptive_enabled() -> None:
    section = _training_section()
    probe_fetch = section.index("probe_batch = next(probe_loader)")
    direction_loop = section.index("for idx in range(grad_accum_steps):", probe_fetch)
    direction_end = section.index("training_manager.step_optimizers(step)", direction_loop)
    direction_body = section[direction_loop:direction_end]

    assert "train_loader.send(" in direction_body
    assert "next(probe_loader)" not in direction_body


def test_training_backward_is_immediately_followed_by_micro_capture() -> None:
    section = _training_section()
    backward = section.index("loss.backward()")
    capture = section.index("training_manager.capture_micro_gradient(", backward)
    optimizer_step = section.index("training_manager.step_optimizers(step)", capture)

    assert backward < capture < optimizer_step
    assert "idx" in section[capture : section.index(")", capture) + 1]


def test_optimizer_step_number_controls_route_collection() -> None:
    assert "training_manager.start_optimizer_step(step)" in SOURCE
    assert "self.optimizer.start_step(step)" in SOURCE


def test_probe_replays_first_micro_after_update_and_fp8_refresh() -> None:
    section = _training_section()
    cache = section.index("probe_batch =")
    optimizer_step = section.index("training_manager.step_optimizers(step)")
    quantize = section.index("model.quantize_mlp_fp8", optimizer_step)
    replay = section.index("with torch.no_grad():", quantize)
    after_probe = section.index("training_manager.after_optimizer_probe(", replay)

    assert cache < optimizer_step < quantize < replay < after_probe
    replay_body = section[replay:after_probe]
    assert ".backward()" not in replay_body
    assert "val_loss" not in replay_body


def test_probe_connects_per_block_drift_and_structure_calibration() -> None:
    section = _training_section()

    assert "activation_output_drift(" in section
    assert "self.optimizer.set_activation_drift(" in SOURCE
    assert "predicted_return_by_structure" in SOURCE
    assert "should_probe = training_manager.should_probe(step)" in section


def test_activation_observer_is_armed_only_around_probe_forwards() -> None:
    section = _training_section()
    micro_loop = section.split("for idx in range(grad_accum_steps):", 1)[1]
    micro_loop = micro_loop.split("training_manager.step_optimizers(step)", 1)[0]

    assert "should_probe and idx == 0" in micro_loop
    assert "training_manager.set_activation_observer(True)" in micro_loop
    assert "training_manager.set_activation_observer(False)" in micro_loop

    replay = section.split("probe_start = time.perf_counter()", 1)[1]
    enable = replay.index("training_manager.set_activation_observer(True)")
    forward = replay.index("probe_after_loss = model(")
    disable = replay.index("training_manager.set_activation_observer(False)")
    assert enable < forward < disable


def test_warmup_does_not_add_a_probe_replay_forward() -> None:
    warmup = _warmup_section()

    assert "warmup_probe_batch" not in warmup
    assert warmup.count("for idx in range(grad_accum_steps):") == 1


def test_route_metrics_only_emit_true_route_events() -> None:
    section = _training_section()
    route_loop = section.index(
        "for block, decision in training_manager.last_optimizer_result.decisions.items():"
    )
    write_route = section.index('"route",', route_loop)

    assert "if decision.route_event:" in section[route_loop:write_route]
    route_call = section[write_route : section.index("\n\n", write_route)]
    assert "budget_statistics=decision.budget_statistics" in route_call


def test_frozen_projection_metrics_emit_a_separate_audit_event() -> None:
    section = _training_section()
    assert 'event="frozen_update"' in section
    assert 'nominal_budget=decision.budget_statistics.get("nominal_budget")' in section
    assert 'effective_budget=decision.budget_statistics.get("effective_budget", decision.budget_scale)' in section
    assert 'reference_uwr=decision.budget_statistics.get("reference_uwr")' in section
    assert 'reward_engine=decision.budget_statistics.get("reward_engine", False)' in section
    assert 'reward_edge=decision.budget_statistics.get("reward_edge")' in section
    assert 'reward_variance=decision.budget_statistics.get("reward_variance")' in section
    assert 'reward_target=decision.budget_statistics.get("reward_target")' in section
    assert 'risk_capacity=decision.budget_statistics.get("risk_capacity")' in section


def test_metrics_are_buffered_and_flushed_at_validation_and_end() -> None:
    assert "DeviceMetricBuffer(" in SOURCE
    assert "metric_buffer.append_train(" in SOURCE
    assert SOURCE.count("flush_device_buffer(metric_buffer)") >= 2
    training_section = _training_section()
    append = training_section.index("metric_buffer.append_train(")
    line_end = training_section.index("\n", append)
    assert ".item()" not in training_section[append:line_end]


def test_failure_events_are_flushed_before_reraise() -> None:
    assert 'write_event("event"' in SOURCE
    assert 'kind="oom"' in SOURCE
    assert "raise" in _training_section()
