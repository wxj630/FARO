from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
import torch

from FARO.descriptors import BlockDescriptor, legal_geometries
from FARO.router import CandidateObservation, FARORouter, RouterConfig


def _descriptor(
    access_pattern: str = "dense",
    *,
    label: str = "block",
    structure_class: str = "mlp",
    update_frequency: str = "every_step",
) -> BlockDescriptor:
    matrix = access_pattern != "coordinate"
    token_axis = 0 if access_pattern == "token_lookup" else None
    return BlockDescriptor(
        label=label,
        parameter_label="p",
        shape=(4, 3) if matrix else (4,),
        matrix_shape=(4, 3) if matrix else None,
        tensor_index=(),
        input_axis=1 if matrix else None,
        output_axis=0 if matrix else None,
        access_pattern=access_pattern,
        token_axis=token_axis,
        head_partition=None,
        shared_weight=None,
        update_frequency=update_frequency,
        structure_class=structure_class,
    )


def _observation(
    returns: list[float],
    *,
    cvar_uwr: float = 0.01,
    max_axis_uwr: float | None = None,
    spectral_risk: float = 0.01,
    drift: float = 0.01,
    gradient_concentration: float = 0.5,
    momentum_concentration: float = 0.5,
) -> CandidateObservation:
    return CandidateObservation.from_returns(
        torch.tensor(returns),
        uwr=torch.tensor(cvar_uwr),
        max_axis_uwr=torch.tensor(
            cvar_uwr if max_axis_uwr is None else max_axis_uwr
        ),
        spectral_risk=torch.tensor(spectral_risk),
        drift=torch.tensor(drift),
        gradient_sv_concentration=torch.tensor(gradient_concentration),
        momentum_sv_concentration=torch.tensor(momentum_concentration),
    )


def test_kelly_controller_enters_aggressive_and_defensive_states() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_min_observations=1,
            kelly_difficulty_ratio=1.01,
            kelly_candidate_margin=0.001,
            kelly_stable_variance=0.01,
        )
    )
    router.update_kelly_state(
        probe_before_loss=10.0,
        realized_return=0.0,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (10.0, 10.0, 10.0)},
    )
    aggressive = router.update_kelly_state(
        probe_before_loss=12.0,
        realized_return=1.0,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (12.0, 11.0, 8.0)},
    )
    assert aggressive["kelly_state"] == "aggressive"
    assert router.kelly_scale == pytest.approx(1.1)
    assert router.kelly_projection_multiple == pytest.approx(4.0)

    router.kelly_steps_remaining = 0
    defensive = router.update_kelly_state(
        probe_before_loss=10.0,
        realized_return=0.1,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (9.9, 10.0, 10.1)},
    )
    assert defensive["kelly_state"] == "defensive"
    assert router.kelly_scale == pytest.approx(1.0)
    assert router.kelly_projection_multiple == pytest.approx(3.0)


def test_continuous_kelly_maps_edge_to_non_discrete_budget_scale() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_continuous=True,
            kelly_min_observations=1,
            kelly_continuous_max_log_scale=0.2,
            kelly_continuous_prior_strength=1.0,
        )
    )
    audit = {"scales": (0.9, 1.0, 1.1), "losses": (10.1, 10.0, 9.8)}
    router.update_kelly_state(probe_before_loss=10.0, realized_return=0.1, audit=audit)
    result = router.update_kelly_state(
        probe_before_loss=10.0, realized_return=0.1, audit=audit
    )
    assert result["kelly_state"] == "aggressive"
    assert 1.0 < router.kelly_scale < math.exp(0.2)
    assert router.kelly_scale != pytest.approx(1.1)
    assert result["kelly_continuous"] is True


def test_continuous_kelly_reduces_scale_for_negative_local_edge() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_continuous=True,
            kelly_min_observations=1,
            kelly_continuous_prior_strength=1.0,
        )
    )
    audit = {"scales": (0.9, 1.0, 1.1), "losses": (9.8, 10.0, 10.1)}
    router.update_kelly_state(probe_before_loss=10.0, realized_return=0.1, audit=audit)
    result = router.update_kelly_state(
        probe_before_loss=10.0, realized_return=0.1, audit=audit
    )
    assert result["kelly_state"] == "defensive"
    assert router.kelly_scale < 1.0
    assert router.kelly_scale > 0.0


def test_ds_controller_shrinks_budget_when_probe_return_dips() -> None:
    router = FARORouter(
        RouterConfig(
            ds_risk_controller=True,
            ds_max_log_scale=0.5,
            ds_window=4,
            ds_z=0.5,
            ds_floor=0.01,
        )
    )
    # Warm up on a stable return level.
    for _ in range(6):
        router.update_ds_state(realized_return=0.5)
    stable_scale = router.kelly_scale

    # A drop below the trailing baseline activates a bounded shrink.
    audit = router.update_ds_state(realized_return=0.2)

    assert audit["ds_activation"] > 0.0
    assert router.kelly_state == "defensive"
    assert router.kelly_scale < stable_scale
    assert router.kelly_scale >= math.exp(-0.5)  # bounded by ds_max_log_scale
    assert router.effective_frozen_budget("block", 1.0) == pytest.approx(
        router.kelly_scale
    )


def test_ds_controller_stays_neutral_on_flat_returns() -> None:
    router = FARORouter(RouterConfig(ds_risk_controller=True, ds_window=4))
    for i in range(20):
        audit = router.update_ds_state(realized_return=0.3 + 0.001 * (i % 2))
    assert router.kelly_state == "neutral"
    assert router.kelly_scale == pytest.approx(1.0, abs=1e-6)
    assert audit["ds_activation"] == pytest.approx(0.0, abs=1e-6)


def test_ds_controller_state_round_trip() -> None:
    router = FARORouter(RouterConfig(ds_risk_controller=True, ds_window=4))
    for value in (0.5, 0.5, 0.5, 0.5, 0.2, 0.1):
        router.update_ds_state(realized_return=value)
    restored = FARORouter(router.config)
    restored.load_state_dict(router.state_dict())
    assert restored.ds_recent == router.ds_recent
    assert restored.ds_baseline == pytest.approx(router.ds_baseline)


def test_continuous_kelly_audit_and_state_round_trip() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_continuous=True,
            kelly_min_observations=1,
            kelly_continuous_prior_strength=1.0,
        )
    )
    audit = {"scales": (0.9, 1.0, 1.1), "losses": (10.1, 10.0, 9.8)}
    router.update_kelly_state(probe_before_loss=10.0, realized_return=0.1, audit=audit)
    result = router.update_kelly_state(
        probe_before_loss=10.0, realized_return=0.1, audit=audit
    )
    restored = FARORouter(router.config)
    restored.load_state_dict(router.state_dict())

    assert restored.kelly_scale == pytest.approx(router.kelly_scale)
    assert restored.kelly_edge == pytest.approx(result["kelly_edge"])
    assert restored.kelly_confidence == pytest.approx(result["kelly_confidence"])
    assert restored.kelly_log_scale == pytest.approx(result["kelly_log_scale"])
    assert restored.state_dict() == router.state_dict()


def test_continuous_kelly_evidence_shrinks_scale_toward_prior() -> None:
    config = RouterConfig(
        kelly_risk_controller=True,
        counterfactual_probe_budget=True,
        kelly_continuous=True,
        kelly_min_observations=1,
        kelly_continuous_max_log_scale=0.2,
        kelly_continuous_prior_strength=100.0,
    )
    audit = {"scales": (0.9, 1.0, 1.1), "losses": (10.1, 10.0, 9.8)}
    router = FARORouter(config)
    first = router.update_kelly_state(
        probe_before_loss=10.0, realized_return=0.1, audit=audit
    )

    assert first["kelly_confidence"] == pytest.approx(1.0 / 101.0)
    assert router.kelly_scale < 1.002


def test_continuous_kelly_flat_probe_stays_neutral() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_continuous=True,
            kelly_min_observations=1,
        )
    )
    audit = {"scales": (0.9, 1.0, 1.1), "losses": (10.0, 10.0, 10.0)}
    router.update_kelly_state(probe_before_loss=10.0, realized_return=0.0, audit=audit)
    result = router.update_kelly_state(
        probe_before_loss=10.0, realized_return=0.0, audit=audit
    )

    assert result["kelly_state"] == "neutral"
    assert router.kelly_scale == pytest.approx(1.0)
    assert router.kelly_log_scale == pytest.approx(0.0)


def test_kelly_defensive_scale_shrinks_frozen_budget() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_min_observations=1,
            kelly_stable_variance=0.01,
            kelly_defensive_scale=0.8,
        )
    )
    router.update_kelly_state(
        probe_before_loss=10.0,
        realized_return=0.1,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (9.9, 10.0, 10.1)},
    )
    defensive = router.update_kelly_state(
        probe_before_loss=10.0,
        realized_return=0.1,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (9.9, 10.0, 10.1)},
    )
    assert defensive["kelly_state"] == "defensive"
    assert router.kelly_scale == pytest.approx(0.8)
    assert router.effective_frozen_budget("block", 1.0) == pytest.approx(0.8)
    neutral_router = FARORouter(
        RouterConfig(kelly_risk_controller=True, counterfactual_probe_budget=True)
    )
    assert neutral_router.config.kelly_defensive_scale == pytest.approx(1.0)
    assert neutral_router.effective_frozen_budget("block", 1.0) == pytest.approx(1.0)


def test_kelly_difficulty_ema_uses_neutral_candidate_loss() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_min_observations=1,
        )
    )
    audit = router.update_kelly_state(
        probe_before_loss=10.0,
        realized_return=0.1,
        audit={"scales": (0.9, 1.0, 1.1), "losses": (8.0, 10.0, 9.0)},
    )
    assert audit["kelly_probe_loss_ema"] == pytest.approx(10.0)


def test_kelly_aggressive_steps_are_consumed_by_frozen_updates() -> None:
    router = FARORouter(
        RouterConfig(
            kelly_risk_controller=True,
            counterfactual_probe_budget=True,
            kelly_aggressive_steps=2,
        )
    )
    router.kelly_state = "aggressive"
    router.kelly_scale = 1.1
    router.kelly_steps_remaining = 2
    assert router.kelly_audit()["kelly_steps_remaining"] == 2
    router.advance_kelly_step()
    assert router.kelly_state == "aggressive"
    assert router.kelly_scale == pytest.approx(1.1)
    assert router.kelly_steps_remaining == 1
    router.advance_kelly_step()
    assert router.kelly_state == "neutral"
    assert router.kelly_scale == pytest.approx(1.0)
    assert router.kelly_steps_remaining == 0


def test_kelly_controller_state_round_trip() -> None:
    router = FARORouter(
        RouterConfig(kelly_risk_controller=True, counterfactual_probe_budget=True)
    )
    router.kelly_state = "aggressive"
    router.kelly_scale = 1.1
    router.kelly_steps_remaining = 2
    payload = router.state_dict()
    restored = FARORouter(router.config)
    restored.load_state_dict(payload)
    assert restored.kelly_state == "aggressive"
    assert restored.kelly_scale == pytest.approx(1.1)
    assert restored.kelly_steps_remaining == 2


@pytest.mark.parametrize(
    ("returns", "target", "edge", "variance"),
    [
        ([1.0] * 8, 1.1, 1.0, 0.0),
        ([0.0] * 8, 1.0, 0.0, 0.0),
        ([-1.0] * 8, 0.5, -1.0, 0.0),
        ([-1.0] * 4 + [3.0] * 4, 0.5, 0.0, 1.0),
    ],
)
def test_local_reward_target_matches_risk_adjusted_formula(
    returns, target, edge, variance
) -> None:
    router = FARORouter(RouterConfig(local_reward_engine=True))
    actual, statistics = router.local_reward_target(returns)

    assert actual == pytest.approx(target)
    assert statistics["reward_edge"] == pytest.approx(edge)
    assert statistics["reward_variance"] == pytest.approx(variance)
    scaled, _ = router.local_reward_target([10 * value for value in returns])
    assert scaled == pytest.approx(actual)


@pytest.mark.parametrize(("step", "unit_budget"), [(0, False), (40, True)])
def test_local_reward_engine_preserves_collection_and_unit_budget(step, unit_budget) -> None:
    descriptor = _descriptor("coordinate")
    router = FARORouter(RouterConfig(local_reward_engine=True, unit_budget=unit_budget))
    decision = router.route(
        step, (descriptor,), {descriptor.label: {"coordinate": _observation([1.0] * 8)}}
    )[descriptor.label]

    assert decision.budget_scale == 1.0
    assert decision.budget_statistics["target"] == 1.0


def test_local_reward_engine_waits_until_geometry_is_frozen() -> None:
    descriptor = _descriptor()
    observations = {
        descriptor.label: {mode: _observation([1.0] * 8) for mode in legal_geometries(descriptor)}
    }
    control = FARORouter(RouterConfig(collect_steps=0))
    engine = FARORouter(RouterConfig(collect_steps=0, local_reward_engine=True))
    baseline = control.route(0, (descriptor,), observations)[descriptor.label]
    decision = engine.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.mode == baseline.mode
    assert decision.scores == baseline.scores
    assert decision.hard_violations == baseline.hard_violations
    assert decision.budget_scale == pytest.approx(baseline.budget_scale)
    assert decision.budget_statistics == baseline.budget_statistics


def test_local_reward_budget_is_limited_by_risk_capacity() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(local_reward_engine=True))
    state = router.initialize(descriptor)
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 0.01
    observation = _observation([1.0] * 8, cvar_uwr=0.10, drift=0.10)

    budget, statistics = router.local_reward_budget(
        descriptor.label, 1.0, observation
    )

    assert statistics["reward_engine"] is True
    assert statistics["reward_raw_target"] == pytest.approx(1.1)
    assert statistics["reward_target"] < 1.0
    assert statistics["reward_target"] == pytest.approx(
        statistics["risk_capacity"]
    )
    assert budget == pytest.approx(statistics["reward_target"])


def test_local_reward_rejects_nonfinite_returns_without_mutating_budget() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(local_reward_engine=True))
    state = router.initialize(descriptor)
    state.budget_scale = 0.8
    with pytest.raises(ValueError, match="finite"):
        router.local_reward_target([float("nan")] * 8)
    assert state.budget_scale == 0.8


def test_budget_only_appetite_does_not_delay_freeze() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
            freeze_stable_routes=True,
            collect_steps=0,
            min_residence_steps=0,
        )
    )
    observations = _stable_dense_observations(descriptor)

    router.route(0, (descriptor,), observations)

    assert router.is_frozen(descriptor.label)


def test_adaptive_appetite_waits_for_three_probe_calibrations_before_freeze() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            freeze_stable_routes=True,
            collect_steps=0,
            min_residence_steps=0,
        )
    )
    observations = _stable_dense_observations(descriptor)
    for step in (0, 8, 16, 24):
        decision = router.route(step, (descriptor,), observations)[descriptor.label]
        assert not router.is_frozen(descriptor.label)
        assert decision.route_event
    router.global_calibration_count = 3
    router.route(32, (descriptor,), observations)
    assert router.is_frozen(descriptor.label)


def test_adaptive_appetite_defaults_to_one_and_round_trips() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    state = router.initialize(descriptor)

    assert state.risk_appetite == pytest.approx(1.0)
    restored = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    restored.load_state_dict(router.state_dict())
    assert restored.states[descriptor.label].risk_appetite == pytest.approx(1.0)


def test_trust_region_projection_contracts_only_risky_budget() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(risk_budget_projection=True, hard_risk_multiple=4.0)
    )
    state = router.initialize(descriptor)
    state.reference_uwr = 0.01

    safe = router.project_budget(descriptor.label, 0.8, 0.02, 0.02)
    risky = router.project_budget(descriptor.label, 0.8, 0.10, 0.10)

    assert safe == pytest.approx(0.8)
    assert risky == pytest.approx(0.4)


def test_trust_region_projection_does_not_raise_budget_floor() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(risk_budget_projection=True, hard_risk_multiple=4.0)
    )
    state = router.initialize(descriptor)
    state.reference_uwr = 0.001

    projected = router.project_budget(descriptor.label, 0.8, 0.10, 0.10)

    assert projected == pytest.approx(0.04)


def test_projection_defaults_to_one_sided_contraction() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(risk_budget_projection=True, hard_risk_multiple=1.0)
    )
    state = router.initialize(descriptor)
    state.reference_uwr = 0.01

    # Low-risk update (r=0.5 -> cap 2.0) has headroom far above nominal, but the
    # default expand ceiling of 1.0 keeps the historical one-sided behaviour.
    safe = router.project_budget(descriptor.label, 0.5, 0.005, 0.005)

    assert router.config.risk_expand_max == pytest.approx(1.0)
    assert safe == pytest.approx(0.5)


def test_projection_expands_safe_budget_when_enabled() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            risk_budget_projection=True,
            hard_risk_multiple=1.0,
            risk_expand_max=1.5,
        )
    )
    state = router.initialize(descriptor)
    state.reference_uwr = 0.01

    # Safe step: cap/nominal = 2.0/0.5 = 4.0 headroom, capped at expand 1.5.
    expanded = router.project_budget(descriptor.label, 0.5, 0.005, 0.005)
    # Risky step: cap/nominal = 0.5/0.8 -> still contracts.
    contracted = router.project_budget(descriptor.label, 0.8, 0.02, 0.02)

    assert expanded == pytest.approx(0.75)
    assert contracted == pytest.approx(0.5)


def test_projection_expansion_respects_budget_max() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            risk_budget_projection=True,
            hard_risk_multiple=1.0,
            risk_expand_max=3.0,
        )
    )
    state = router.initialize(descriptor)
    state.reference_uwr = 0.01

    expanded = router.project_budget(descriptor.label, 0.8, 0.002, 0.002)

    assert expanded == pytest.approx(router.config.budget_max)


def test_unit_budget_overrides_trust_region_projection() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(risk_budget_projection=True, unit_budget=True)
    )
    router.initialize(descriptor).reference_uwr = 1e-5

    assert router.project_budget(descriptor.label, 1.0, 0.1, 0.1) == 1.0


def test_more_conservative_appetite_does_not_expand_budget() -> None:
    descriptor = _descriptor()
    observations = _stable_dense_observations(descriptor)
    conservative = FARORouter(
        RouterConfig(adaptive_risk_appetite=True, collect_steps=0)
    )
    aggressive = FARORouter(
        RouterConfig(adaptive_risk_appetite=True, collect_steps=0)
    )
    conservative.initialize(descriptor).risk_appetite = 0.75
    aggressive.initialize(descriptor).risk_appetite = 1.25

    conservative_decision = conservative.route(
        32, (descriptor,), observations
    )[descriptor.label]
    aggressive_decision = aggressive.route(
        32, (descriptor,), observations
    )[descriptor.label]

    assert conservative_decision.budget_scale <= aggressive_decision.budget_scale
    assert conservative_decision.budget_statistics["risk_appetite"] == pytest.approx(0.75)
    assert aggressive_decision.budget_statistics["risk_appetite"] == pytest.approx(1.25)
    assert 0.5 <= conservative_decision.budget_scale <= 1.1
    assert 0.5 <= aggressive_decision.budget_scale <= 1.1


def test_constant_structural_bias_does_not_pin_appetite_to_floor() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    router.initialize(descriptor)

    # realized/predicted is structurally far below 1 (in-sample vs held-out), but
    # constant. Appetite must not drift to its floor just because the level is low.
    for _ in range(10):
        router.update_risk_appetite({"mlp": (0.05, 1.0, 0.0)})

    assert router.states[descriptor.label].risk_appetite == pytest.approx(1.0)


def test_small_probe_outcomes_are_neutral_under_appetite_deadband() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    router.initialize(descriptor)
    router.update_global_risk_appetite(1.0, 0.0)
    for _ in range(10):
        router.update_global_risk_appetite(0.05, 0.0)
    assert router.states[descriptor.label].risk_appetite == pytest.approx(1.0)


def test_better_than_baseline_conversion_raises_appetite() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    router.initialize(descriptor)

    for _ in range(6):
        router.update_risk_appetite({"mlp": (0.05, 1.0, 0.0)})
    baseline = router.states[descriptor.label].risk_appetite
    assert baseline == pytest.approx(1.0)

    # Conversion improves well above the running baseline -> appetite > 1.
    for _ in range(10):
        router.update_risk_appetite({"mlp": (0.5, 1.0, 0.0)})

    assert router.states[descriptor.label].risk_appetite > 1.0


def test_adaptive_appetite_update_reacts_to_overprediction_and_clamps() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(adaptive_risk_appetite=True))
    router.initialize(descriptor)

    router.calibrate(
        realized_return=0.1,
        predicted_return=1.0,
        structure_returns={"mlp": (0.1, 1.0)},
    )
    assert router.states[descriptor.label].risk_appetite == pytest.approx(1.0)
    for _ in range(4):
        router.update_risk_appetite({"mlp": (0.1, 1.0, 2.0)})
    assert 0.75 <= router.states[descriptor.label].risk_appetite < 1.0


def test_candidate_observation_packs_device_values_before_one_cpu_transfer(
    monkeypatch,
) -> None:
    cpu_calls = 0
    original_cpu = torch.Tensor.cpu

    def counted_cpu(self, *args, **kwargs):
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    observation = CandidateObservation.from_device_values(
        torch.arange(8, dtype=torch.float32),
        uwr=torch.tensor(0.1),
        max_axis_uwr=torch.tensor(0.2),
        spectral_risk=torch.tensor(0.3),
        drift=torch.tensor(0.4),
        gradient_sv_concentration=torch.tensor(0.5),
        momentum_sv_concentration=torch.tensor(0.6),
    )

    assert cpu_calls == 1
    assert observation.returns == tuple(float(value) for value in range(8))


def _prime_reference(
    router: FARORouter,
    descriptor: BlockDescriptor,
    *,
    cvar: float = 0.01,
    drift: float = 0.01,
    uwr: float = 0.01,
    spectral: float = 0.01,
    ready: bool = True,
) -> None:
    state = router.initialize(descriptor)
    state.reference_initialized = True
    state.reference_ready = ready
    state.reference_cvar = cvar
    state.reference_drift = drift
    state.reference_uwr = uwr
    state.reference_spectral = spectral


def _stable_dense_observations(
    descriptor: BlockDescriptor,
) -> dict[str, dict[str, CandidateObservation]]:
    return {
        descriptor.label: {
            "coordinate": _observation([0.0] * 8),
            "row_column": _observation([0.0] * 8),
            "spectral": _observation([1.0] * 8),
        }
    }


def test_stability_freeze_never_applies_during_collection() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    observations = _stable_dense_observations(descriptor)

    for step in range(32):
        router.route(step, (descriptor,), observations)
        assert not router.is_frozen(descriptor.label)


def test_stable_dense_block_freezes_at_step_48() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    observations = _stable_dense_observations(descriptor)

    for step in range(49):
        router.route(step, (descriptor,), observations)

    assert router.is_frozen(descriptor.label)
    assert router.states[descriptor.label].frozen_step == 48
    assert not router.is_route_event(56, descriptor)
    assert not router.route(56, (descriptor,), observations)[
        descriptor.label
    ].route_event


def test_frozen_block_ignores_nonfinite_incumbent_observation() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    stable_observations = _stable_dense_observations(descriptor)
    for step in range(49):
        router.route(step, (descriptor,), stable_observations)
    before = deepcopy(router.state_dict())
    observations = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 8),
            "row_column": _observation([0.0] * 8),
            "spectral": _observation([math.nan] * 8),
        }
    }

    decision = router.route(56, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "spectral"
    assert decision.budget_scale == before["states"][descriptor.label][
        "budget_scale"
    ]
    assert not decision.switched
    assert not decision.route_event
    assert router.states[descriptor.label].frozen_step == 48
    assert router.state_dict() == before


def test_stable_odd_step_block_freezes_at_step_49() -> None:
    descriptor = _descriptor(update_frequency="odd_step")
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    observations = _stable_dense_observations(descriptor)

    for step in range(50):
        router.route(step, (descriptor,), observations)

    assert router.is_frozen(descriptor.label)
    assert router.states[descriptor.label].frozen_step == 49


def test_mode_switch_resets_stability_freeze_clock() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            freeze_stable_routes=True,
            challenger_margin=0.0,
            challenger_events=2,
        )
    )
    router.initialize(descriptor).current_mode = "coordinate"
    observations = _stable_dense_observations(descriptor)

    router.route(32, (descriptor,), observations)
    switched = router.route(40, (descriptor,), observations)[descriptor.label]
    router.route(48, (descriptor,), observations)

    assert switched.switched
    assert router.states[descriptor.label].stable_since_step == 40
    assert not router.is_frozen(descriptor.label)
    assert router.route(56, (descriptor,), observations)[descriptor.label].route_event
    assert router.is_frozen(descriptor.label)
    assert router.states[descriptor.label].frozen_step == 56


def test_frozen_route_state_survives_state_dict_round_trip() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    observations = _stable_dense_observations(descriptor)
    for step in range(49):
        router.route(step, (descriptor,), observations)
    restored = FARORouter(RouterConfig(freeze_stable_routes=True))

    restored.load_state_dict(router.state_dict())

    assert restored.is_frozen(descriptor.label)
    assert restored.states[descriptor.label].frozen_step == 48
    assert restored.state_dict() == router.state_dict()


def test_observation_uses_worst_two_returns_for_cvar() -> None:
    observation = _observation([2.0, 2.0, 1.0, 1.0, 0.5, 0.5, -3.0, -5.0])

    assert observation.mean_return == pytest.approx(-0.125)
    assert observation.cvar == pytest.approx(4.0)


def test_dense_block_switches_to_spectral_after_two_challenger_events() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.05,
            challenger_events=2,
        )
    )
    router.initialize(descriptor)
    router.states[descriptor.label].current_mode = "coordinate"
    observations = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 6 + [-1.0, -1.0]),
            "row_column": _observation([0.5] * 6 + [-1.0, -1.0]),
            "spectral": _observation([1.0] * 8),
        }
    }

    first = router.route(32, (descriptor,), observations)[descriptor.label]
    second = router.route(40, (descriptor,), observations)[descriptor.label]

    assert first.mode == "coordinate"
    assert not first.switched
    assert second.mode == "spectral"
    assert second.switched
    assert second.reason == "challenger_won_twice"


def test_pending_challenger_blocks_stability_freeze() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            route_interval=8,
            collect_steps=32,
            min_residence_steps=16,
            challenger_margin=0.05,
            challenger_events=2,
            freeze_stable_routes=True,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "coordinate"
    _prime_reference(router, descriptor, cvar=10.0, drift=100.0, uwr=100.0, spectral=100.0)
    settled = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 8),
            "row_column": _observation([1.0] * 8),
            "spectral": _observation([-1.0] * 8),
        }
    }
    challenging = {
        descriptor.label: {
            "coordinate": _observation([-1.0] * 8),
            "row_column": _observation([-1.0] * 8),
            "spectral": _observation([1.0] * 8),
        }
    }

    for step in (0, 8, 16, 24, 32, 40):
        router.route(step, (descriptor,), settled)
    assert not router.is_frozen(descriptor.label)

    # Step 48 is freeze-eligible, but a challenger just won its first vote.
    pending = router.route(48, (descriptor,), challenging)[descriptor.label]
    assert pending.reason == "challenger_pending"
    assert not pending.switched
    assert not router.is_frozen(descriptor.label)

    # The next decision point lets the challenger finish, then it can freeze.
    won = router.route(56, (descriptor,), challenging)[descriptor.label]
    assert won.mode == "spectral"
    assert won.switched
    assert won.reason == "challenger_won_twice"


def test_dense_singular_value_concentration_rewards_spectral_geometry() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.05,
            challenger_events=2,
        )
    )
    router.initialize(descriptor).current_mode = "coordinate"
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                drift=0.01,
                gradient_concentration=0.9,
                momentum_concentration=0.9,
            ),
            "row_column": _observation(
                [0.5] * 8,
                drift=0.02,
                gradient_concentration=0.9,
                momentum_concentration=0.9,
            ),
            "spectral": _observation(
                [1.0] * 8,
                drift=0.012,
                gradient_concentration=0.9,
                momentum_concentration=0.9,
            ),
        }
    }

    first = router.route(0, (descriptor,), observations)[descriptor.label]
    second = router.route(8, (descriptor,), observations)[descriptor.label]

    assert first.reason == "challenger_pending"
    assert second.mode == "spectral"
    assert second.switched


def test_token_candidates_are_shadow_only_without_label_lookup() -> None:
    descriptor = _descriptor(
        "token_lookup",
        label="unrelated-token-table",
        structure_class="embedding",
        update_frequency="odd_step",
    )
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.0,
            challenger_events=1,
        )
    )
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 6 + [-2.0, -2.0]),
            "row_column": _observation([0.8] * 8),
            "spectral": _observation(
                [0.8] * 8, spectral_risk=0.2, drift=0.1
            ),
        }
    }

    decision = router.route(1, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "coordinate"
    assert set(decision.candidate_statistics) == {
        "coordinate",
        "row_column",
        "spectral",
    }
    assert router.states[descriptor.label].challenger_mode is None
    assert router.states[descriptor.label].challenger_wins == 0
    renamed = replace(descriptor, label="still-unrelated")
    assert router.initial_mode(renamed) == "coordinate"


def test_dynamic_token_starts_coordinate_with_row_column_legal() -> None:
    descriptor = _descriptor("token_lookup")
    output = _descriptor("token_output")
    dense = _descriptor("dense")
    dynamic = FARORouter(RouterConfig())
    fixed = FARORouter(RouterConfig(fixed_geometry=True))

    assert dynamic.initial_mode(descriptor) == "coordinate"
    assert dynamic.initial_mode(output) == "coordinate"
    assert dynamic.initial_mode(dense) == "spectral"
    assert fixed.initial_mode(descriptor) == "row_column"
    assert fixed.initial_mode(output) == "row_column"
    assert "row_column" in legal_geometries(descriptor)


def test_opt_in_dynamic_token_starts_row_column_and_selects_legal_geometries() -> None:
    descriptor = _descriptor("token_lookup")
    output = _descriptor("token_output")
    router = FARORouter(RouterConfig(dynamic_token_geometry=True))

    assert router.initial_mode(descriptor) == "row_column"
    assert router.initial_mode(output) == "row_column"
    assert router.selectable_geometries(descriptor) == legal_geometries(descriptor)
    assert router.selectable_geometries(output) == legal_geometries(output)


def test_coordinate_only_forces_every_block_to_coordinate() -> None:
    dense = _descriptor("dense")
    descriptor = _descriptor("token_lookup")
    output = _descriptor("token_output")
    router = FARORouter(RouterConfig(coordinate_only=True))

    assert router.initial_mode(dense) == "coordinate"
    assert router.initial_mode(descriptor) == "coordinate"
    assert router.initial_mode(output) == "coordinate"
    assert router.selectable_geometries(dense) == ("coordinate",)
    assert router.selectable_geometries(descriptor) == ("coordinate",)
    assert router.selectable_geometries(output) == ("coordinate",)


def test_default_config_keeps_dense_spectral_without_coordinate_only() -> None:
    dense = _descriptor("dense")
    default = FARORouter(RouterConfig())

    assert default.initial_mode(dense) == "spectral"
    assert "spectral" in default.selectable_geometries(dense)


def test_coordinate_start_begins_at_coordinate_but_keeps_all_candidates() -> None:
    dense = _descriptor("dense")
    descriptor = _descriptor("token_lookup")
    router = FARORouter(RouterConfig(coordinate_start=True))

    assert router.initial_mode(dense) == "coordinate"
    assert router.initial_mode(descriptor) == "coordinate"
    assert router.selectable_geometries(dense) == legal_geometries(dense)
    assert router.selectable_geometries(descriptor) == legal_geometries(descriptor)


def test_coordinate_start_lets_dense_block_leave_coordinate_on_hard_risk() -> None:
    dense = _descriptor("dense")
    router = FARORouter(
        RouterConfig(
            coordinate_start=True,
            collect_steps=0,
            route_interval=1,
            min_residence_steps=0,
        )
    )
    _prime_reference(router, dense, cvar=0.01)
    # coordinate is hard-risk (huge cvar ratio); spectral is safe -> evacuation.
    safe = _observation([1.0] * 8, cvar_uwr=0.0)
    hard = _observation([1.0] * 8, cvar_uwr=1e6)
    observations = {"coordinate": hard, "row_column": hard, "spectral": safe}

    decision = router.route(1, (dense,), {dense.label: observations})[dense.label]

    assert decision.switched
    assert decision.mode == "spectral"


def test_router_compares_returns_on_shared_block_scale() -> None:
    descriptor = _descriptor("dense")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.0,
            challenger_events=1,
        )
    )
    router.initialize(descriptor).current_mode = "coordinate"
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.1] * 8),
            "row_column": _observation([0.2] * 8),
            "spectral": _observation([1.0] * 8),
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "spectral"
    assert decision.switched


def test_token_nonfinite_coordinate_raises_when_shadows_are_finite() -> None:
    descriptor = _descriptor(
        "token_lookup", structure_class="embedding", update_frequency="odd_step"
    )
    router = FARORouter(RouterConfig(collect_steps=0))
    observations = {
        descriptor.label: {
            "coordinate": _observation([math.nan] * 8),
            "row_column": _observation([2.0] * 8),
            "spectral": _observation([3.0] * 8),
        }
    }

    with pytest.raises(FloatingPointError, match="selectable FARO candidate"):
        router.route(1, (descriptor,), observations)


@pytest.mark.parametrize("failure", ["hard", "nonfinite"])
def test_dense_prefers_spectral_fallback(failure: str) -> None:
    descriptor = _descriptor("dense")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=999,
            challenger_margin=100.0,
            challenger_events=2,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "row_column"
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 0.01
    state.reference_spectral = 0.01
    failed = (
        _observation([math.nan] * 8)
        if failure == "nonfinite"
        else _observation([1.0] * 8, cvar_uwr=1e6)
    )
    observations = {
        descriptor.label: {
            "coordinate": _observation([10.0] * 8),
            "row_column": failed,
            "spectral": _observation([0.1] * 8),
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "spectral"
    assert decision.switched
    assert decision.reason in {
        "hard_risk_evacuation",
        "nonfinite_evacuation",
    }


def test_block_level_drift_does_not_gate_geometry_switch() -> None:
    descriptor = _descriptor("dense")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.0,
            challenger_events=1,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "coordinate"
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 1.0
    state.reference_spectral = 1.0
    # Every candidate shares the same inflated drift; only spectral is safe on
    # the mode-discriminating cvar ratio.
    observations = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 8, cvar_uwr=1e6, drift=1.0),
            "row_column": _observation([1.0] * 8, cvar_uwr=1e6, drift=1.0),
            "spectral": _observation([1.0] * 8, cvar_uwr=0.0, drift=1.0),
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    # drift is still reported, but it must not force all candidates hard.
    assert "drift" in decision.hard_risk_triggers["coordinate"]
    assert decision.mode == "spectral"
    assert decision.switched
    assert decision.reason == "hard_risk_evacuation"


def test_normalize_candidate_uwr_puts_spectral_on_coordinate_scale() -> None:
    descriptor = _descriptor("dense")

    def decide(normalize: bool) -> object:
        router = FARORouter(
            RouterConfig(
                collect_steps=8,
                min_residence_steps=0,
                challenger_margin=0.0,
                challenger_events=1,
                normalize_candidate_uwr=normalize,
            )
        )
        router.initialize(descriptor).current_mode = "coordinate"
        observations = {
            descriptor.label: {
                # spectral's raw UWR is 5x coordinate's, as dense geometry does.
                "coordinate": _observation([1.0] * 8, cvar_uwr=1.0),
                "row_column": _observation([1.0] * 8, cvar_uwr=1.0),
                "spectral": _observation([1.0] * 8, cvar_uwr=5.0),
            }
        }
        router.route(0, (descriptor,), observations)  # collection sample
        return router.route(8, (descriptor,), observations)[descriptor.label]

    normalized = decide(True)
    raw = decide(False)

    # Raw UWR is 5x the reference (hard at multiple 4): flagged without the fix.
    assert raw.risk_ratios["spectral"]["uwr"] == pytest.approx(5.0)
    assert "spectral" in raw.hard_violations
    # Once placed on the coordinate scale it is ~1x the reference: not flagged.
    assert normalized.risk_ratios["spectral"]["uwr"] == pytest.approx(1.0)
    assert "spectral" not in normalized.hard_violations


def test_axis_amplification_rejects_bad_tail_candidate() -> None:
    descriptor = _descriptor("token_lookup", update_frequency="odd_step")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.0,
            challenger_events=1,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "coordinate"
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 1.0
    state.reference_spectral = 1.0
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8, max_axis_uwr=0.01),
            "row_column": _observation([1.0] * 8, max_axis_uwr=1.0),
            "spectral": _observation([0.01] * 8, max_axis_uwr=0.01),
        }
    }

    decision = router.route(1, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "coordinate"
    assert "row_column" in decision.hard_violations
    assert decision.hard_risk_triggers["row_column"] == (
        "axis_amplification",
    )
    assert decision.risk_ratios["row_column"]["axis_amplification"] == pytest.approx(
        100.0
    )
    assert "axis_amplification" in decision.hard_risk_triggers["row_column"]


def test_hard_token_shadow_cannot_displace_coordinate() -> None:
    descriptor = _descriptor("token_lookup", update_frequency="odd_step")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=999,
            challenger_margin=100.0,
            challenger_events=2,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "row_column"
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 1.0
    state.reference_spectral = 1.0
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8, max_axis_uwr=0.01),
            "row_column": _observation([1.0] * 8, max_axis_uwr=0.05),
            "spectral": _observation([0.1] * 8, max_axis_uwr=0.01),
        }
    }

    decision = router.route(1, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "coordinate"
    assert not decision.switched
    assert decision.reason == "incumbent_best"
    assert "row_column" in decision.hard_violations


def test_nonfinite_token_shadow_does_not_poison_audit_state() -> None:
    descriptor = _descriptor("token_lookup", update_frequency="odd_step")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=999,
            challenger_margin=100.0,
            challenger_events=2,
        )
    )
    router.initialize(descriptor).current_mode = "row_column"
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8),
            "row_column": _observation(
                [math.nan] * 8,
                cvar_uwr=0.9,
                max_axis_uwr=math.nan,
                spectral_risk=0.7,
                drift=0.8,
            ),
            "spectral": _observation([0.1] * 8),
        }
    }

    decision = router.route(2, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "coordinate"
    assert not decision.switched
    assert not decision.route_event
    assert decision.reason == "not_route_event"
    assert router.states[descriptor.label].ema_abs_return["row_column"] == 0.0
    assert router.states[descriptor.label].reference_uwr == pytest.approx(0.01)
    assert router.states[descriptor.label].reference_drift == pytest.approx(0.01)
    assert all(math.isfinite(score) for score in decision.scores.values())
    assert decision.candidate_statistics["row_column"]["mean_return"] is None
    assert decision.candidate_statistics["row_column"]["max_axis_uwr"] is None
    assert decision.risk_ratios["row_column"]["axis_amplification"] is None
    json.dumps(
        {
            "scores": decision.scores,
            "ratios": decision.risk_ratios,
            "statistics": decision.candidate_statistics,
        },
        allow_nan=False,
    )


def test_nonfinite_challenger_does_not_poison_future_return_scale() -> None:
    descriptor = _descriptor("token_lookup", update_frequency="odd_step")
    router = FARORouter(RouterConfig(collect_steps=0))
    invalid = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8),
            "row_column": _observation([math.inf] * 8),
            "spectral": _observation([0.1] * 8),
        }
    }

    first = router.route(1, (descriptor,), invalid)[descriptor.label]

    assert "row_column" in first.hard_violations
    assert router.states[descriptor.label].ema_abs_return["row_column"] == 0.0
    assert all(math.isfinite(score) for score in first.scores.values())
    router.states[descriptor.label].ema_abs_return["row_column"] = math.inf

    finite = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8),
            "row_column": _observation([0.4] * 8),
            "spectral": _observation([0.1] * 8),
        }
    }
    router.route(9, (descriptor,), finite)

    assert math.isfinite(
        router.states[descriptor.label].ema_abs_return["row_column"]
    )
    assert router.states[descriptor.label].ema_abs_return["row_column"] > 0.0


def test_router_rejects_block_without_any_finite_candidate() -> None:
    descriptor = _descriptor("token_lookup", update_frequency="odd_step")
    router = FARORouter(RouterConfig(collect_steps=0))
    observations = {
        descriptor.label: {
            mode: _observation([math.nan] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    with pytest.raises(FloatingPointError, match="no finite FARO candidate"):
        router.route(1, (descriptor,), observations)


def test_axis_amplification_does_not_reject_dense_spectral_candidate() -> None:
    descriptor = _descriptor("dense")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.0,
            challenger_events=1,
        )
    )
    state = router.initialize(descriptor)
    state.current_mode = "coordinate"
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 1.0
    state.reference_spectral = 1.0
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.5] * 8, max_axis_uwr=0.01),
            "row_column": _observation([0.1] * 8, max_axis_uwr=0.01),
            "spectral": _observation([1.0] * 8, max_axis_uwr=0.1),
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "spectral"
    assert "spectral" not in decision.hard_violations
    assert decision.risk_ratios["spectral"]["axis_amplification"] == pytest.approx(
        10.0
    )


def test_coordinate_block_has_only_coordinate_route() -> None:
    descriptor = _descriptor("coordinate", structure_class="coordinate")
    router = FARORouter(RouterConfig(collect_steps=0))
    decision = router.route(
        32,
        (descriptor,),
        {descriptor.label: {"coordinate": _observation([0.1] * 8)}},
    )[descriptor.label]

    assert decision.mode == "coordinate"
    assert set(decision.scores) == {"coordinate"}


def test_equal_mean_candidate_with_worse_tail_gets_lower_score() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(collect_steps=0))
    observations = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 8),
            "row_column": _observation([2.0] * 4 + [1.0] * 2 + [-1.0] * 2),
            "spectral": _observation([0.5] * 8),
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert observations[descriptor.label]["coordinate"].mean_return == pytest.approx(
        observations[descriptor.label]["row_column"].mean_return
    )
    assert decision.scores["coordinate"] > decision.scores["row_column"]


def test_budget_only_appetite_does_not_change_geometry_score() -> None:
    descriptor = _descriptor()
    observation = _observation([1.0] * 8)
    conservative = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
            collect_steps=0,
        )
    )
    neutral = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
            collect_steps=0,
        )
    )
    conservative_state = conservative.initialize(descriptor)
    neutral_state = neutral.initialize(descriptor)
    conservative_state.risk_appetite = 0.75

    assert conservative._score(
        descriptor, "coordinate", observation, conservative_state, 1.0, observation
    ) == pytest.approx(
        neutral._score(
            descriptor, "coordinate", observation, neutral_state, 1.0, observation
        )
    )


def test_budget_only_appetite_keeps_unfrozen_budget_neutral() -> None:
    descriptor = _descriptor()
    observations = _stable_dense_observations(descriptor)
    router = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
            collect_steps=0,
        )
    )
    router.initialize(descriptor).risk_appetite = 0.75

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert decision.budget_statistics["risk_appetite"] == pytest.approx(1.0)


def test_unit_budget_blocks_frozen_appetite_scaling() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
            unit_budget=True,
        )
    )
    router.initialize(descriptor).risk_appetite = 0.75

    assert router.effective_frozen_budget(descriptor.label, 1.0) == pytest.approx(1.0)


def test_frozen_budget_scale_is_applied_only_by_explicit_controller() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(frozen_budget_scale=0.8)
    )
    assert router.effective_frozen_budget(descriptor.label, 1.0) == pytest.approx(1.0)


def test_frozen_budget_only_appetite_scales_update_budget() -> None:
    descriptor = _descriptor()
    conservative = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
        )
    )
    aggressive = FARORouter(
        RouterConfig(
            adaptive_risk_appetite=True,
            risk_appetite_budget_only=True,
        )
    )
    conservative.initialize(descriptor).risk_appetite = 0.75
    aggressive.initialize(descriptor).risk_appetite = 1.25

    assert conservative.effective_frozen_budget(descriptor.label, 1.0) == pytest.approx(0.75)
    assert aggressive.effective_frozen_budget(descriptor.label, 1.0) == pytest.approx(1.1)


def test_risk_appetite_scales_tail_risk_penalty() -> None:
    descriptor = _descriptor()
    low_tail = _observation([1.0] * 8)
    high_tail = _observation([2.0] * 4 + [1.0] * 2 + [-1.0] * 2)

    conservative = FARORouter(
        RouterConfig(adaptive_risk_appetite=True, collect_steps=0)
    )
    aggressive = FARORouter(
        RouterConfig(adaptive_risk_appetite=True, collect_steps=0)
    )
    conservative_state = conservative.initialize(descriptor)
    aggressive_state = aggressive.initialize(descriptor)
    conservative_state.risk_appetite = 0.75
    aggressive_state.risk_appetite = 1.25

    conservative_gap = conservative._score(
        descriptor,
        "coordinate",
        low_tail,
        conservative_state,
        1.0,
        low_tail,
    ) - conservative._score(
        descriptor,
        "coordinate",
        high_tail,
        conservative_state,
        1.0,
        low_tail,
    )
    aggressive_gap = aggressive._score(
        descriptor,
        "coordinate",
        low_tail,
        aggressive_state,
        1.0,
        low_tail,
    ) - aggressive._score(
        descriptor,
        "coordinate",
        high_tail,
        aggressive_state,
        1.0,
        low_tail,
    )

    assert conservative_gap > aggressive_gap


def test_single_challenger_win_does_not_switch() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.05,
            challenger_events=2,
        )
    )
    router.initialize(descriptor)
    router.states[descriptor.label].current_mode = "coordinate"
    observations = {
        descriptor.label: {
            "coordinate": _observation([1.0] * 6 + [-1.0, -1.0]),
            "row_column": _observation([0.0] * 8),
            "spectral": _observation([1.0] * 8),
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "coordinate"
    assert not decision.switched
    assert router.states[descriptor.label].challenger_wins == 1


def test_all_hard_violations_keep_mode_and_shrink_budget() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(collect_steps=8))
    low = {
        descriptor.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    router.route(0, (descriptor,), low)
    high = {
        descriptor.label: {
            mode: _observation(
                [1.0] * 8,
                cvar_uwr=1.0,
                spectral_risk=1.0,
                drift=1.0,
            )
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    before = router.states[descriptor.label].current_mode

    decision = router.route(8, (descriptor,), high)[descriptor.label]

    assert decision.mode == before
    assert not decision.switched
    assert decision.reason == "all_candidates_hard_risk"
    assert decision.budget_scale == pytest.approx(0.95)


def test_odd_step_block_routes_on_phase_one() -> None:
    descriptor = _descriptor(update_frequency="odd_step")
    router = FARORouter(
        RouterConfig(
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.05,
            challenger_events=2,
        )
    )
    router.initialize(descriptor).current_mode = "coordinate"
    observations = {
        descriptor.label: {
            "coordinate": _observation([0.0] * 8),
            "row_column": _observation([0.0] * 8),
            "spectral": _observation([1.0] * 8),
        }
    }

    first = router.route(1, (descriptor,), observations)[descriptor.label]
    second = router.route(9, (descriptor,), observations)[descriptor.label]

    assert first.route_event
    assert first.reason == "challenger_pending"
    assert second.route_event
    assert second.mode == "spectral"
    assert second.switched


def test_non_route_zero_spectral_sample_does_not_decay_reference() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(collect_steps=0))
    route_observations = {
        descriptor.label: {
            mode: _observation([1.0] * 8, spectral_risk=0.2)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    router.route(0, (descriptor,), route_observations)
    reference = router.states[descriptor.label].reference_spectral
    non_route_observations = {
        descriptor.label: {
            mode: _observation([1.0] * 8, spectral_risk=0.0)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    decision = router.route(1, (descriptor,), non_route_observations)[
        descriptor.label
    ]

    assert not decision.route_event
    assert router.states[descriptor.label].reference_spectral == pytest.approx(
        reference
    )


def test_collection_budget_is_one_while_references_still_update() -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, ready=False)
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [0.1] * 6 + [-1.0, -1.0],
                cvar_uwr=1.0,
                drift=1.0,
            )
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.budget_scale == 1.0
    assert decision.budget_statistics["collection_locked"] is True
    assert decision.budget_statistics["target"] == 1.0
    assert router.states[descriptor.label].reference_drift > 0.01


def test_coordinate_budget_contracts_after_collection_and_recovers_toward_one() -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, ready=False)
    poor = {
        descriptor.label: {
            "coordinate": _observation(
                [0.1] * 6 + [-1.0, -1.0],
                cvar_uwr=0.04,
                drift=0.04,
            )
        }
    }

    contracted = router.route(32, (descriptor,), poor)[descriptor.label]
    restored = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                cvar_uwr=0.0025,
                drift=0.0025,
            )
        }
    }
    recovered = router.route(40, (descriptor,), restored)[descriptor.label]

    assert contracted.budget_scale == pytest.approx(0.95)
    assert contracted.budget_statistics == {
        "quality": 0.0,
        "risk_capacity": 0.5,
        "return_cap": 0.5,
        "target": 0.5,
        "expansion_eligible": False,
        "collection_locked": False,
    }
    assert contracted.budget_scale < recovered.budget_scale <= 1.0


def test_two_positive_probes_do_not_allow_budget_expansion() -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, cvar=0.04, drift=0.04, uwr=0.04)
    for _ in range(2):
        router.calibrate(2.0, 1.0, {"norm": (2.0, 1.0)})
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                cvar_uwr=0.01,
                drift=0.01,
            )
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert decision.budget_scale == 1.0
    assert decision.budget_statistics["expansion_eligible"] is False


def test_three_positive_global_and_structure_probes_allow_expansion() -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, cvar=0.04, drift=0.04, uwr=0.04)
    for _ in range(3):
        router.calibrate(2.0, 1.0, {"norm": (2.0, 1.0)})
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                cvar_uwr=0.01,
                drift=0.01,
            )
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert router.global_calibration_count == 3
    assert router.structure_calibration_count == {"norm": 3}
    assert decision.budget_statistics["quality"] >= 0.9
    assert decision.budget_statistics["expansion_eligible"] is True
    assert 1.0 < decision.budget_scale <= 1.1


@pytest.mark.parametrize(
    "calibrations",
    [
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        [(2.0, 1.0), (0.0, 1.0), (2.0, 1.0)],
    ],
)
def test_negative_or_ambiguous_calibration_cannot_expand_budget(
    calibrations: list[tuple[float, float]],
) -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, cvar=0.04, drift=0.04, uwr=0.04)
    for realized, predicted in calibrations:
        router.calibrate(
            realized,
            predicted,
            {"norm": (realized, predicted)},
        )
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                cvar_uwr=0.01,
                drift=0.01,
            )
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert decision.budget_scale <= 1.0
    assert decision.budget_statistics["expansion_eligible"] is False


def test_global_probe_count_cannot_replace_structure_probe_count() -> None:
    descriptor = _descriptor("coordinate", structure_class="norm")
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor, cvar=0.04, drift=0.04, uwr=0.04)
    router.calibrate(2.0, 1.0, {"norm": (2.0, 1.0)})
    router.calibrate(2.0, 1.0, {"norm": (2.0, 1.0)})
    router.calibrate(2.0, 1.0, {})
    observations = {
        descriptor.label: {
            "coordinate": _observation(
                [1.0] * 8,
                cvar_uwr=0.01,
                drift=0.01,
            )
        }
    }

    decision = router.route(32, (descriptor,), observations)[descriptor.label]

    assert router.global_calibration_count == 3
    assert router.structure_calibration_count == {"norm": 2}
    assert decision.budget_statistics["expansion_eligible"] is False
    assert decision.budget_scale == 1.0


@pytest.mark.parametrize(
    ("access_pattern", "structure_class", "update_frequency", "step"),
    [
        ("dense", "mlp", "every_step", 32),
        ("token_lookup", "embedding", "odd_step", 33),
        ("token_output", "lm_head", "odd_step", 33),
        ("coordinate", "norm", "every_step", 32),
        ("coordinate", "bias", "every_step", 32),
        ("coordinate", "scalar", "every_step", 32),
    ],
)
def test_every_descriptor_reports_complete_budget_statistics(
    access_pattern: str,
    structure_class: str,
    update_frequency: str,
    step: int,
) -> None:
    descriptor = _descriptor(
        access_pattern,
        label=f"{structure_class}-block",
        structure_class=structure_class,
        update_frequency=update_frequency,
    )
    router = FARORouter(RouterConfig(collect_steps=32))
    _prime_reference(router, descriptor)
    observations = {
        descriptor.label: {
            mode: _observation([1.0] * 8)
            for mode in legal_geometries(descriptor)
        }
    }

    decision = router.route(step, (descriptor,), observations)[descriptor.label]

    assert decision.route_event
    assert set(decision.budget_statistics) == {
        "quality",
        "risk_capacity",
        "return_cap",
        "target",
        "expansion_eligible",
        "collection_locked",
    }
    assert 0.5 <= decision.budget_scale <= 1.1


def test_router_default_budget_bounds_match_fixed_anchor_design() -> None:
    config = RouterConfig()

    assert config.budget_min == 0.5
    assert config.budget_max == 1.1


def test_nonfinite_probe_is_atomic() -> None:
    router = FARORouter(RouterConfig())
    router.calibrate(2.0, 1.0, {"mlp": (2.0, 1.0)})
    before = deepcopy(router.state_dict())

    with pytest.raises(FloatingPointError, match="nonfinite FARO probe calibration"):
        router.calibrate(math.nan, 1.0, {"mlp": (2.0, 1.0)})

    assert router.state_dict() == before


def test_hard_risk_decision_reports_normalized_ratios_and_triggers() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(collect_steps=0))
    baseline = {
        descriptor.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    router.route(0, (descriptor,), baseline)
    high_drift = {
        descriptor.label: {
            mode: _observation([1.0] * 8, drift=1.0)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    decision = router.route(8, (descriptor,), high_drift)[descriptor.label]

    assert decision.risk_ratios["spectral"]["drift"] == pytest.approx(100.0)
    assert decision.hard_risk_triggers["spectral"] == ("drift",)
    assert decision.candidate_statistics["spectral"] == {
        "mean_return": 1.0,
        "cvar": 0.0,
        "uwr": pytest.approx(0.01),
        "max_axis_uwr": pytest.approx(0.01),
        "spectral_risk": pytest.approx(0.01),
        "drift": pytest.approx(1.0),
        "gradient_sv_concentration": pytest.approx(0.5),
        "momentum_sv_concentration": pytest.approx(0.5),
    }


def test_structure_budget_cap_uses_all_initialized_peer_blocks() -> None:
    descriptors = tuple(
        _descriptor(label=f"mlp-{index}", structure_class="mlp")
        for index in range(3)
    )
    router = FARORouter(RouterConfig(collect_steps=0))
    for descriptor, budget in zip(
        descriptors, (0.25, 0.25, 1.5), strict=True
    ):
        router.initialize(descriptor).budget_scale = budget
    target = descriptors[-1]
    observations = {
        target.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    decision = router.route(1, (target,), observations)[target.label]

    assert decision.budget_scale == pytest.approx(0.5)
    assert router.states[target.label].budget_scale == pytest.approx(0.5)


def test_fixed_geometry_uses_functional_access_pattern() -> None:
    router = FARORouter(RouterConfig(fixed_geometry=True))

    assert router.initial_mode(_descriptor("dense")) == "spectral"
    assert router.initial_mode(_descriptor("token_lookup")) == "row_column"
    assert router.initial_mode(_descriptor("coordinate")) == "coordinate"
    assert router.initial_mode(_descriptor("token_output")) == "row_column"


def test_fixed_token_coordinate_matches_muon_geometry() -> None:
    router = FARORouter(
        RouterConfig(fixed_geometry=True, fixed_token_coordinate=True)
    )

    assert router.initial_mode(_descriptor("dense")) == "spectral"
    assert router.initial_mode(_descriptor("token_lookup")) == "coordinate"
    assert router.initial_mode(_descriptor("token_output")) == "coordinate"
    assert router.initial_mode(_descriptor("coordinate")) == "coordinate"


def test_fixed_geometry_never_switches() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(
            fixed_geometry=True,
            collect_steps=0,
            min_residence_steps=0,
            challenger_margin=0.01,
            challenger_events=1,
        )
    )
    state = router.initialize(descriptor)
    state.reference_initialized = True
    state.reference_ready = True
    state.reference_cvar = 0.01
    state.reference_drift = 0.01
    state.reference_uwr = 0.01
    state.reference_spectral = 0.01
    observations = {
        descriptor.label: {
            "coordinate": _observation([2.0] * 8),
            "row_column": _observation([1.0] * 8),
            "spectral": _observation(
                [0.1] * 8, drift=1.0, spectral_risk=1.0
            ),
        }
    }

    decision = router.route(0, (descriptor,), observations)[descriptor.label]

    assert decision.mode == "spectral"
    assert decision.reason == "fixed_geometry"
    assert not decision.switched
    assert "spectral" in decision.hard_violations


def test_unit_budget_stays_one_during_high_risk_route_event() -> None:
    descriptor = _descriptor()
    router = FARORouter(
        RouterConfig(fixed_geometry=True, unit_budget=True, collect_steps=0)
    )
    router.initialize(descriptor).budget_scale = 0.25
    baseline = {
        descriptor.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    router.route(0, (descriptor,), baseline)
    mode_count = router.states[descriptor.label].mode_counts["spectral"]
    high_risk = {
        descriptor.label: {
            mode: _observation(
                [1.0] * 8,
                cvar_uwr=1.0,
                spectral_risk=1.0,
                drift=1.0,
            )
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    decision = router.route(8, (descriptor,), high_risk)[descriptor.label]
    state = router.states[descriptor.label]

    assert decision.route_event
    assert decision.budget_scale == 1.0
    assert state.budget_scale == 1.0
    assert state.mode_counts["spectral"] == mode_count + 1
    assert state.reference_drift > 0.01


def test_unit_budget_ignores_peer_budget_cap_on_non_route_step() -> None:
    descriptors = tuple(
        _descriptor(label=f"mlp-{index}", structure_class="mlp")
        for index in range(3)
    )
    router = FARORouter(RouterConfig(unit_budget=True, collect_steps=0))
    for descriptor in descriptors:
        router.initialize(descriptor).budget_scale = 0.25
    target = descriptors[-1]
    observations = {
        target.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }

    decision = router.route(1, (target,), observations)[target.label]

    assert not decision.route_event
    assert decision.budget_scale == 1.0
    assert router.states[target.label].budget_scale == 1.0


def test_router_state_dict_round_trip() -> None:
    descriptor = _descriptor()
    router = FARORouter(RouterConfig(collect_steps=0))
    observations = {
        descriptor.label: {
            mode: _observation([1.0] * 8)
            for mode in ("coordinate", "row_column", "spectral")
        }
    }
    router.route(32, (descriptor,), observations)
    router.calibrate(2.0, 1.0, {"mlp": (2.0, 1.0)})
    restored = FARORouter(RouterConfig(collect_steps=0))

    restored.load_state_dict(router.state_dict())

    assert restored.state_dict() == router.state_dict()


def test_old_router_checkpoint_migrates_missing_calibration_counts() -> None:
    router = FARORouter(RouterConfig())
    payload = router.state_dict()
    payload.pop("global_calibration_count", None)
    payload.pop("structure_calibration_count", None)
    restored = FARORouter(RouterConfig())

    restored.load_state_dict(payload)

    assert restored.global_calibration_count == 0
    assert restored.structure_calibration_count == {}


def test_erc_redistributes_budget_neutrally_and_inversely_to_risk() -> None:
    router = FARORouter(
        RouterConfig(
            erc_budget=True,
            erc_risk_beta=0.5,
            erc_winsor=1e6,
            erc_scale_cap=1e6,
        )
    )
    # Low-risk blocks (small uwr) should get multipliers > 1, high-risk < 1.
    for label, uwr in {"a": 0.01, "b": 0.01, "c": 0.04}.items():
        router.record_erc_risk(label, uwr)
    base = {"a": 0.6, "b": 0.6, "c": 0.6}
    multipliers = router.compute_erc_multipliers(base)

    # Budget neutrality: total redistributed budget equals total base budget.
    total = sum(base.values())
    assert sum(base[l] * multipliers[l] for l in base) == pytest.approx(total)
    # Equal risk => equal low-risk weights; the risky block is shrunk.
    assert multipliers["a"] == pytest.approx(multipliers["b"])
    assert multipliers["a"] > 1.0 > multipliers["c"]
    # Inverse-risk ratio a/c is 4x before normalisation, preserved after.
    assert multipliers["a"] / multipliers["c"] == pytest.approx(4.0)


def test_erc_scale_is_identity_when_disabled_and_neutral_without_risk() -> None:
    disabled = FARORouter(RouterConfig())
    assert disabled.erc_scale("block") == pytest.approx(1.0)
    assert disabled.compute_erc_multipliers({"a": 0.6, "b": 0.6}) == {}

    enabled = FARORouter(RouterConfig(erc_budget=True))
    # No observed risk yet => no reallocation => identity multipliers.
    assert enabled.compute_erc_multipliers({"a": 0.6, "b": 0.6}) == {}
    assert enabled.erc_scale("a") == pytest.approx(1.0)


def test_erc_winsor_and_cap_bound_extreme_risk() -> None:
    router = FARORouter(
        RouterConfig(erc_budget=True, erc_risk_beta=0.0, erc_winsor=3.0, erc_scale_cap=1.4)
    )
    # A degenerate tiny-risk block must not blow up the allocation.
    for label, uwr in {"a": 0.02, "b": 0.02, "c": 0.02, "scalar": 1e-9}.items():
        router.record_erc_risk(label, uwr)
    multipliers = router.compute_erc_multipliers(
        {"a": 0.6, "b": 0.6, "c": 0.6, "scalar": 0.6}
    )
    for value in multipliers.values():
        assert 1.0 / 1.4 - 1e-9 <= value <= 1.4 + 1e-9
    # Still budget-neutral after winsorising and capping.
    base = {"a": 0.6, "b": 0.6, "c": 0.6, "scalar": 0.6}
    assert sum(base[l] * multipliers[l] for l in base) == pytest.approx(sum(base.values()))


def test_erc_state_round_trips_and_effective_budget_applies_it() -> None:
    router = FARORouter(RouterConfig(erc_budget=True, erc_winsor=1e6, erc_scale_cap=1e6))
    router.record_erc_risk("a", 0.01)
    router.record_erc_risk("b", 0.04)
    router.compute_erc_multipliers({"a": 0.6, "b": 0.6})
    restored = FARORouter(RouterConfig(erc_budget=True, erc_winsor=1e6, erc_scale_cap=1e6))

    restored.load_state_dict(router.state_dict())

    assert restored.state_dict()["erc_multiplier"] == router.state_dict()["erc_multiplier"]
    # effective_frozen_budget multiplies in the ERC scale for the low-risk block.
    assert restored.effective_frozen_budget("a", 0.6) > 0.6
    assert restored.effective_frozen_budget("b", 0.6) < 0.6


def test_erc_budget_clamp_decouples_from_global_budget_max() -> None:
    # With the global clamp, the reallocation can never step a block above 1.1.
    global_clamp = FARORouter(
        RouterConfig(erc_budget=True, erc_winsor=1e6, erc_scale_cap=1e6)
    )
    global_clamp.record_erc_risk("low", 0.001)   # very low risk -> wants to expand
    global_clamp.record_erc_risk("high", 1.0)
    global_clamp.compute_erc_multipliers({"low": 1.0, "high": 1.0})
    assert global_clamp.effective_frozen_budget("low", 1.0) <= 1.1

    # With an ERC-specific ceiling the same block may step past budget_max,
    # while the geometry-facing budget_max is left untouched.
    decoupled = FARORouter(
        RouterConfig(
            erc_budget=True, erc_winsor=1e6, erc_scale_cap=1e6,
            erc_budget_max=2.0,
        )
    )
    decoupled.record_erc_risk("low", 0.001)
    decoupled.record_erc_risk("high", 1.0)
    decoupled.compute_erc_multipliers({"low": 1.0, "high": 1.0})
    assert decoupled.config.budget_max == 1.1  # routing ceiling unchanged
    assert decoupled.effective_frozen_budget("low", 1.0) > 1.1
    assert decoupled.effective_frozen_budget("low", 1.0) <= 2.0


def test_erc_budget_floor_decouples_from_global_budget_min() -> None:
    decoupled = FARORouter(
        RouterConfig(erc_budget=True, erc_winsor=1e6, erc_scale_cap=1e6,
                     erc_budget_min=0.2)
    )
    decoupled.record_erc_risk("low", 0.001)
    decoupled.record_erc_risk("high", 1.0)
    decoupled.compute_erc_multipliers({"low": 1.0, "high": 1.0})
    # The risky block may now shrink below the global 0.5 floor.
    assert decoupled.effective_frozen_budget("high", 1.0) < 0.5
