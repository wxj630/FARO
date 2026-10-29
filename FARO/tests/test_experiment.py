from pathlib import Path

import numpy as np
import pytest
import torch

import FARO.experiment as experiment
from FARO.experiment import (
    ExperimentConfig,
    build_run_metadata,
    hash_code_files,
    hash_data_manifest,
    hash_named_tensors,
    seed_everything,
)
from FARO.router import RouterConfig


def test_config_parses_formal_faro_environment() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_SEED": "2",
            "FARO_RUN_ID": "faro-seed2",
            "FARO_MAX_STEPS": "1285",
            "FARO_RESULTS_ROOT": "FARO/results",
        }
    )

    assert config.variant == "faro"
    assert config.seed == 2
    assert config.max_steps == 1285
    assert config.run_dir(Path("/repo")) == Path(
        "/repo/FARO/results/runs/faro/seed_2/faro-seed2"
    )
    assert config.is_formal_run


def test_config_parses_faro_fixed_diagnostic() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro_fixed",
            "FARO_SEED": "0",
            "FARO_RUN_ID": "faro-fixed-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
        }
    )

    assert config.variant == "faro_fixed"
    assert config.run_dir(Path("/repo")) == Path(
        "/repo/FARO/results/runs/faro_fixed/seed_0/faro-fixed-diagnostic250-seed0"
    )
    assert not config.is_formal_run


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("adam", False),
        ("muon", False),
        ("normuon", False),
        ("faro", True),
        ("faro_fixed", False),
    ],
)
def test_router_config_enables_stability_freeze_only_for_dynamic_faro(
    variant: str, expected: bool
) -> None:
    router_config = RouterConfig.from_experiment(
        ExperimentConfig(variant=variant)  # type: ignore[arg-type]
    )

    assert router_config.freeze_stable_routes is expected


def test_config_parses_unit_budget_diagnostic() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro_fixed",
            "FARO_RUN_ID": "faro-fixed-unit-budget-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_UNIT_BUDGET": "1",
        }
    )

    assert config.unit_budget
    assert RouterConfig.from_experiment(config).unit_budget


def test_config_parses_muon_control_diagnostic() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro_fixed",
            "FARO_RUN_ID": "faro-muon-control-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_UNIT_BUDGET": "1",
            "FARO_FIXED_TOKEN_COORDINATE": "1",
        }
    )

    router_config = RouterConfig.from_experiment(config)
    assert config.fixed_token_coordinate
    assert router_config.fixed_token_coordinate
    assert router_config.unit_budget


def test_local_reward_engine_defaults_off_and_parses_strict_flag() -> None:
    default = ExperimentConfig.from_env({})
    enabled = ExperimentConfig.from_env({"FARO_LOCAL_REWARD_ENGINE": "1"})

    assert not default.local_reward_engine
    assert not RouterConfig.from_experiment(default).local_reward_engine
    assert enabled.local_reward_engine
    assert RouterConfig.from_experiment(enabled).local_reward_engine
    with pytest.raises(ValueError, match="FARO_LOCAL_REWARD_ENGINE"):
        ExperimentConfig.from_env({"FARO_LOCAL_REWARD_ENGINE": "true"})


def test_kelly_risk_controller_defaults_off_and_requires_counterfactual_probe() -> None:
    default = ExperimentConfig.from_env({})
    assert not default.kelly_risk_controller
    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-kelly-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
            "FARO_RISK_BUDGET_PROJECTION": "1",
            "FARO_KELLY_RISK_CONTROLLER": "1",
        }
    )
    assert enabled.kelly_risk_controller
    assert RouterConfig.from_experiment(enabled).kelly_risk_controller
    with pytest.raises(ValueError, match="FARO_KELLY_RISK_CONTROLLER"):
        ExperimentConfig.from_env(
            {
                "FARO_VARIANT": "faro",
                "FARO_RUN_ID": "faro-kelly-diagnostic250-seed0",
                "FARO_MAX_STEPS": "250",
                "FARO_KELLY_RISK_CONTROLLER": "1",
            }
        )
    with pytest.raises(ValueError, match="cannot exceed 4.0"):
        ExperimentConfig.from_env(
            {
                "FARO_VARIANT": "faro",
                "FARO_RUN_ID": "faro-kelly-diagnostic250-seed0",
                "FARO_MAX_STEPS": "250",
                "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
                "FARO_KELLY_RISK_CONTROLLER": "1",
                "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE": "4.1",
                "FARO_RISK_BUDGET_PROJECTION": "1",
            }
        )


def test_kelly_requires_projection_and_finite_settings() -> None:
    env = {
        "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
        "FARO_KELLY_RISK_CONTROLLER": "1",
    }
    with pytest.raises(ValueError, match="FARO_RISK_BUDGET_PROJECTION"):
        ExperimentConfig.from_env(env)
    env["FARO_RISK_BUDGET_PROJECTION"] = "1"
    for key in (
        "FARO_KELLY_FRACTION", "FARO_KELLY_DIFFICULTY_RATIO",
        "FARO_KELLY_CANDIDATE_MARGIN", "FARO_KELLY_STABLE_VARIANCE",
        "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE",
    ):
        for value in ("nan", "inf", "-inf"):
            with pytest.raises(ValueError, match=key):
                ExperimentConfig.from_env({**env, key: value})


def test_frozen_budget_scale_defaults_to_one_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert default.frozen_budget_scale == pytest.approx(1.0)
    assert RouterConfig.from_experiment(default).frozen_budget_scale == pytest.approx(1.0)

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-understep-diagnostic480-seed0",
            "FARO_MAX_STEPS": "480",
            "FARO_FROZEN_BUDGET_SCALE": "0.8",
        }
    )
    assert enabled.frozen_budget_scale == pytest.approx(0.8)
    assert RouterConfig.from_experiment(enabled).frozen_budget_scale == pytest.approx(0.8)


def test_hard_risk_multiple_defaults_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert default.hard_risk_multiple == pytest.approx(4.0)
    assert RouterConfig.from_experiment(default).hard_risk_multiple == pytest.approx(4.0)

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-riskproj-diagnostic640-seed0",
            "FARO_MAX_STEPS": "640",
            "FARO_RISK_BUDGET_PROJECTION": "1",
            "FARO_HARD_RISK_MULTIPLE": "1.5",
        }
    )
    assert enabled.hard_risk_multiple == pytest.approx(1.5)
    assert RouterConfig.from_experiment(enabled).hard_risk_multiple == pytest.approx(1.5)


@pytest.mark.parametrize("value", ["0", "-0.1", "4.1", "10", "nan", "inf", "-inf"])
def test_hard_risk_multiple_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_HARD_RISK_MULTIPLE"):
        ExperimentConfig.from_env({"FARO_HARD_RISK_MULTIPLE": value})


def test_risk_expand_max_defaults_to_one_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert default.risk_expand_max == pytest.approx(1.0)
    assert RouterConfig.from_experiment(default).risk_expand_max == pytest.approx(1.0)

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-riskexpand-diagnostic640-seed0",
            "FARO_MAX_STEPS": "640",
            "FARO_RISK_BUDGET_PROJECTION": "1",
            "FARO_RISK_EXPAND_MAX": "1.4",
        }
    )
    assert enabled.risk_expand_max == pytest.approx(1.4)
    assert RouterConfig.from_experiment(enabled).risk_expand_max == pytest.approx(1.4)


@pytest.mark.parametrize("value", ["0", "0.9", "4.1", "nan", "inf", "-inf"])
def test_risk_expand_max_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_RISK_EXPAND_MAX"):
        ExperimentConfig.from_env({"FARO_RISK_EXPAND_MAX": value})


def test_val_loss_every_defaults_and_parses_environment() -> None:
    assert ExperimentConfig.from_env({}).val_loss_every == 250
    enabled = ExperimentConfig.from_env({"FARO_VAL_LOSS_EVERY": "50"})
    assert enabled.val_loss_every == 50
    with pytest.raises(ValueError, match="FARO_VAL_LOSS_EVERY"):
        ExperimentConfig.from_env({"FARO_VAL_LOSS_EVERY": "-1"})


def test_kelly_defensive_scale_defaults_to_one_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert default.kelly_defensive_scale == pytest.approx(1.0)
    assert RouterConfig.from_experiment(default).kelly_defensive_scale == pytest.approx(1.0)

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-kelly-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
            "FARO_RISK_BUDGET_PROJECTION": "1",
            "FARO_KELLY_RISK_CONTROLLER": "1",
            "FARO_KELLY_DEFENSIVE_SCALE": "0.8",
        }
    )
    assert enabled.kelly_defensive_scale == pytest.approx(0.8)
    assert RouterConfig.from_experiment(enabled).kelly_defensive_scale == pytest.approx(0.8)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf", "-inf"])
def test_kelly_defensive_scale_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_KELLY_DEFENSIVE_SCALE"):
        ExperimentConfig.from_env({"FARO_KELLY_DEFENSIVE_SCALE": value})


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf", "-inf"])
def test_frozen_budget_scale_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_FROZEN_BUDGET_SCALE"):
        ExperimentConfig.from_env({"FARO_FROZEN_BUDGET_SCALE": value})


def test_ds_controller_defaults_off_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert not default.ds_risk_controller
    assert default.ds_max_log_scale == pytest.approx(0.5)
    assert default.ds_window == 10
    assert default.ds_scale_beta == pytest.approx(0.9)
    assert default.ds_z == pytest.approx(1.0)
    assert default.ds_floor == pytest.approx(0.1)
    assert not RouterConfig.from_experiment(default).ds_risk_controller

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-ds-diagnostic640-seed0",
            "FARO_MAX_STEPS": "640",
            "FARO_DS_RISK_CONTROLLER": "1",
            "FARO_DS_MAX_LOG_SCALE": "0.4",
            "FARO_DS_WINDOW": "6",
            "FARO_DS_SCALE_BETA": "0.95",
            "FARO_DS_Z": "1.5",
            "FARO_DS_FLOOR": "0.2",
        }
    )
    rc = RouterConfig.from_experiment(enabled)
    assert enabled.ds_risk_controller
    assert rc.ds_max_log_scale == pytest.approx(0.4)
    assert rc.ds_window == 6
    assert rc.ds_scale_beta == pytest.approx(0.95)
    assert rc.ds_z == pytest.approx(1.5)
    assert rc.ds_floor == pytest.approx(0.2)


def test_ds_controller_rejects_combining_with_kelly() -> None:
    env = {
        "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
        "FARO_RISK_BUDGET_PROJECTION": "1",
        "FARO_KELLY_RISK_CONTROLLER": "1",
        "FARO_DS_RISK_CONTROLLER": "1",
    }
    with pytest.raises(ValueError, match="FARO_DS_RISK_CONTROLLER"):
        ExperimentConfig.from_env(env)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("FARO_DS_MAX_LOG_SCALE", "0"),
        ("FARO_DS_WINDOW", "1"),
        ("FARO_DS_Z", "0"),
        ("FARO_DS_FLOOR", "0"),
    ],
)
def test_ds_controller_rejects_invalid_values(key: str, value: str) -> None:
    with pytest.raises(ValueError, match=key):
        ExperimentConfig.from_env({key: value})


def test_continuous_kelly_defaults_off_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert not default.kelly_continuous
    assert default.kelly_continuous_max_log_scale == pytest.approx(0.1)
    assert default.kelly_continuous_prior_strength == pytest.approx(3.0)
    assert not RouterConfig.from_experiment(default).kelly_continuous

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-kelly-continuous-diagnostic640-seed0",
            "FARO_MAX_STEPS": "640",
            "FARO_COUNTERFACTUAL_PROBE_BUDGET": "1",
            "FARO_RISK_BUDGET_PROJECTION": "1",
            "FARO_KELLY_RISK_CONTROLLER": "1",
            "FARO_KELLY_CONTINUOUS": "1",
            "FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE": "0.15",
            "FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH": "5",
        }
    )
    assert enabled.kelly_continuous
    router_config = RouterConfig.from_experiment(enabled)
    assert router_config.kelly_continuous
    assert router_config.kelly_continuous_max_log_scale == pytest.approx(0.15)
    assert router_config.kelly_continuous_prior_strength == pytest.approx(5.0)


def test_continuous_kelly_requires_risk_controller() -> None:
    with pytest.raises(ValueError, match="FARO_KELLY_CONTINUOUS"):
        ExperimentConfig.from_env({"FARO_KELLY_CONTINUOUS": "1"})


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "-inf"])
def test_continuous_kelly_max_log_scale_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE"):
        ExperimentConfig.from_env(
            {"FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE": value}
        )


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "-inf"])
def test_continuous_kelly_prior_strength_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH"):
        ExperimentConfig.from_env(
            {"FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH": value}
        )


def test_regime_switching_defaults_off_and_parses_environment() -> None:
    default = ExperimentConfig.from_env({})
    assert not default.regime_switching
    assert default.regime_period_steps == 80
    assert default.regime_corruption_rate == pytest.approx(0.15)
    assert not default.regime_dirty_first
    assert default.regime_corrupt_targets
    assert not default.regime_fixed_input_mask
    assert not default.regime_probe_matching
    assert default.regime_short_context_length == 0

    enabled = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-regime-diagnostic480-seed0",
            "FARO_MAX_STEPS": "480",
            "FARO_REGIME_SWITCHING": "1",
            "FARO_REGIME_PERIOD": "64",
            "FARO_REGIME_CORRUPTION": "0.2",
            "FARO_REGIME_DIRTY_FIRST": "1",
            "FARO_REGIME_CORRUPT_TARGETS": "0",
            "FARO_REGIME_FIXED_INPUT_MASK": "1",
            "FARO_REGIME_PROBE_MATCHING": "1",
            "FARO_REGIME_SHORT_CONTEXT": "128",
        }
    )
    assert enabled.regime_switching
    assert enabled.regime_period_steps == 64
    assert enabled.regime_corruption_rate == pytest.approx(0.2)
    assert enabled.regime_dirty_first
    assert not enabled.regime_corrupt_targets
    assert enabled.regime_fixed_input_mask
    assert enabled.regime_probe_matching
    assert enabled.regime_short_context_length == 128


def test_regime_switching_rejects_invalid_settings() -> None:
    env = {
        "FARO_VARIANT": "faro",
        "FARO_RUN_ID": "faro-regime-diagnostic480-seed0",
        "FARO_MAX_STEPS": "480",
        "FARO_REGIME_SWITCHING": "1",
    }
    with pytest.raises(ValueError, match="FARO_REGIME_PERIOD"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_PERIOD": "0"})
    with pytest.raises(ValueError, match="FARO_REGIME_SHORT_CONTEXT"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_SHORT_CONTEXT": "-1"})
    with pytest.raises(ValueError, match="FARO_REGIME_CORRUPTION"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_CORRUPTION": "1.0"})
    with pytest.raises(ValueError, match="FARO_REGIME_CORRUPTION"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_CORRUPTION": "-0.1"})
    with pytest.raises(ValueError, match="FARO_REGIME_CORRUPTION"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_CORRUPTION": "nan"})
    with pytest.raises(ValueError, match="FARO_REGIME_SWITCHING"):
        ExperimentConfig.from_env({**env, "FARO_REGIME_SWITCHING": "true"})


def test_counterfactual_probe_budget_defaults_off_and_parses_strict_flag() -> None:
    default = ExperimentConfig.from_env({})
    enabled = ExperimentConfig.from_env(
        {"FARO_COUNTERFACTUAL_PROBE_BUDGET": "1"}
    )

    assert not default.counterfactual_probe_budget
    assert not RouterConfig.from_experiment(default).counterfactual_probe_budget
    assert enabled.counterfactual_probe_budget
    assert RouterConfig.from_experiment(enabled).counterfactual_probe_budget
    with pytest.raises(ValueError, match="FARO_COUNTERFACTUAL_PROBE_BUDGET"):
        ExperimentConfig.from_env({"FARO_COUNTERFACTUAL_PROBE_BUDGET": "true"})


def test_config_parses_risk_budget_projection_flag() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-risk-projection-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_RISK_BUDGET_PROJECTION": "1",
        }
    )

    assert config.risk_budget_projection
    assert RouterConfig.from_experiment(config).risk_budget_projection


def test_config_parses_dynamic_token_geometry_flag() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-token-dynamic-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_DYNAMIC_TOKEN_GEOMETRY": "1",
        }
    )

    assert config.dynamic_token_geometry
    assert RouterConfig.from_experiment(config).dynamic_token_geometry


def test_config_parses_coordinate_only_flag() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-coordinate-only-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_COORDINATE_ONLY": "1",
        }
    )

    assert config.coordinate_only
    assert RouterConfig.from_experiment(config).coordinate_only


def test_config_defaults_coordinate_only_off() -> None:
    config = ExperimentConfig.from_env({})

    assert not config.coordinate_only
    assert not RouterConfig.from_experiment(config).coordinate_only


def test_config_parses_coordinate_start_flag() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-coordinate-start-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_COORDINATE_START": "1",
        }
    )

    assert config.coordinate_start
    assert RouterConfig.from_experiment(config).coordinate_start


def test_config_defaults_coordinate_start_off() -> None:
    config = ExperimentConfig.from_env({})

    assert not config.coordinate_start
    assert not RouterConfig.from_experiment(config).coordinate_start


def test_config_parses_normalize_candidate_uwr_flag() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-uwrnorm-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_NORMALIZE_CANDIDATE_UWR": "1",
        }
    )

    assert config.normalize_candidate_uwr
    assert RouterConfig.from_experiment(config).normalize_candidate_uwr


def test_config_defaults_normalize_candidate_uwr_off() -> None:
    config = ExperimentConfig.from_env({})

    assert not config.normalize_candidate_uwr
    assert not RouterConfig.from_experiment(config).normalize_candidate_uwr


def test_config_defaults_adaptive_risk_appetite_off() -> None:
    config = ExperimentConfig.from_env({})

    assert not config.adaptive_risk_appetite
    assert config.probe_tokens == 1_048_576
    assert config.risk_appetite_min == pytest.approx(0.75)
    assert config.risk_appetite_max == pytest.approx(1.25)
    assert config.risk_appetite_beta == pytest.approx(0.9)
    assert not RouterConfig.from_experiment(config).adaptive_risk_appetite


def test_config_parses_adaptive_risk_appetite_settings() -> None:
    config = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-adaptive-risk-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
            "FARO_ADAPTIVE_RISK_APPETITE": "1",
            "FARO_PROBE_TOKENS": "2097152",
        }
    )

    assert config.adaptive_risk_appetite
    assert config.probe_tokens == 2_097_152
    router_config = RouterConfig.from_experiment(config)
    assert router_config.adaptive_risk_appetite
    assert router_config.probe_tokens == 2_097_152


def test_config_rejects_invalid_risk_appetite_bounds() -> None:
    with pytest.raises(ValueError, match="risk appetite"):
        ExperimentConfig.from_env(
            {
                "FARO_VARIANT": "faro",
                "FARO_RUN_ID": "faro-adaptive-risk-diagnostic250-seed0",
                "FARO_MAX_STEPS": "250",
                "FARO_RISK_APPETITE_MIN": "1.3",
                "FARO_RISK_APPETITE_MAX": "1.2",
            }
        )


def test_config_derives_run_id_after_variant_and_seed() -> None:
    config = ExperimentConfig.from_env(
        {"FARO_VARIANT": "muon", "FARO_SEED": "1"}
    )

    assert config.run_id == "muon-seed1"


@pytest.mark.parametrize("variant", ["sgd", "adamw", ""])
def test_config_rejects_unknown_variant(variant: str) -> None:
    with pytest.raises(ValueError, match="FARO_VARIANT"):
        ExperimentConfig.from_env({"FARO_VARIANT": variant})


def test_formal_run_rejects_seed_outside_declared_set() -> None:
    with pytest.raises(ValueError, match="formal seed"):
        ExperimentConfig.from_env(
            {
                "FARO_VARIANT": "adam",
                "FARO_SEED": "4",
                "FARO_RUN_ID": "adam-seed4",
                "FARO_MAX_STEPS": "1285",
            }
        )


def test_short_run_requires_declared_short_run_id() -> None:
    with pytest.raises(ValueError, match="smoke, pilot, or diagnostic"):
        ExperimentConfig.from_env(
            {
                "FARO_VARIANT": "faro",
                "FARO_RUN_ID": "faro-seed0",
                "FARO_MAX_STEPS": "2",
            }
        )

    smoke = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "smoke-faro-seed0",
            "FARO_MAX_STEPS": "2",
        }
    )
    assert not smoke.is_formal_run

    diagnostic = ExperimentConfig.from_env(
        {
            "FARO_VARIANT": "faro",
            "FARO_RUN_ID": "faro-v2-diagnostic250-seed0",
            "FARO_MAX_STEPS": "250",
        }
    )
    assert not diagnostic.is_formal_run


def test_seed_everything_repeats_numpy_and_torch_draws() -> None:
    seed_everything(7)
    first_numpy = np.random.standard_normal(4)
    first_torch = torch.randn(4)

    seed_everything(7)

    np.testing.assert_array_equal(np.random.standard_normal(4), first_numpy)
    torch.testing.assert_close(torch.randn(4), first_torch, rtol=0, atol=0)


def test_named_tensor_hash_supports_bfloat16_and_detects_changes() -> None:
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    first = hash_named_tensors([("weight", tensor)])
    second = hash_named_tensors([("weight", tensor.clone())])
    changed = tensor.clone()
    changed[0] = 3.0

    assert first == second
    assert first != hash_named_tensors([("weight", changed)])


def test_named_tensor_hash_supports_scalars_and_preserves_shape() -> None:
    scalar = torch.tensor(1.0)

    first = hash_named_tensors([("weight", scalar)])
    same = hash_named_tensors([("weight", scalar.clone())])
    changed = hash_named_tensors([("weight", torch.tensor(2.0))])
    vector = hash_named_tensors([("weight", torch.tensor([1.0]))])

    assert first == same
    assert first != changed
    assert first != vector


def test_data_manifest_hash_is_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"bb")

    forward = hash_data_manifest([str(first), str(second)])
    reverse = hash_data_manifest([str(second), str(first)])

    assert forward == reverse


def test_code_hash_includes_relative_path_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("x = 1\n")
    second.write_text("x = 1\n")

    assert hash_code_files([first], root=tmp_path) != hash_code_files(
        [second], root=tmp_path
    )


def test_run_metadata_uses_cuda_property_major_and_minor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCudaProperties:
        name = "NVIDIA H100"
        major = 9
        minor = 0
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: FakeCudaProperties()
    )
    monkeypatch.setattr(experiment, "hash_named_tensors", lambda tensors: "hash")
    monkeypatch.setattr(experiment, "hash_data_manifest", lambda patterns: "hash")
    monkeypatch.setattr(
        experiment, "hash_code_files", lambda paths, *, root: "hash"
    )

    metadata = build_run_metadata(
        ExperimentConfig(),
        Path("/repo"),
        [],
        torch.empty(0),
        data_patterns=[],
        code_paths=[],
        world_size=1,
        grad_accum_steps=1,
    )

    assert metadata["cuda"] == {
        "available": True,
        "cuda_version": torch.version.cuda,
        "device_index": 1,
        "name": "NVIDIA H100",
        "capability": [9, 0],
        "total_memory": 80 * 1024**3,
    }
    assert metadata["config"]["local_reward_engine"] is False

    enabled = build_run_metadata(
        ExperimentConfig(local_reward_engine=True),
        Path("/repo"),
        [],
        torch.empty(0),
        data_patterns=[],
        code_paths=[],
        world_size=1,
        grad_accum_steps=1,
    )
    assert enabled["config"]["local_reward_engine"] is True
    assert enabled["config"]["counterfactual_probe_budget"] is False

    counterfactual = build_run_metadata(
        ExperimentConfig(counterfactual_probe_budget=True),
        Path("/repo"),
        [],
        torch.empty(0),
        data_patterns=[],
        code_paths=[],
        world_size=1,
        grad_accum_steps=1,
    )
    assert counterfactual["config"]["counterfactual_probe_budget"] is True
