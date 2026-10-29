from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


Variant = Literal["adam", "muon", "normuon", "faro", "faro_fixed"]
_VARIANTS: tuple[Variant, ...] = (
    "adam",
    "muon",
    "normuon",
    "faro",
    "faro_fixed",
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    variant: Variant = "normuon"
    seed: int = 0
    run_id: str = "normuon-seed0"
    results_root: Path = Path("FARO/results")
    max_steps: int = 1285
    matrix_lr: float = 0.0015
    muon_lr: float = 0.023
    route_interval: int = 8
    probe_interval: int = 16
    collect_steps: int = 32
    min_residence_steps: int = 16
    challenger_margin: float = 0.10
    challenger_events: int = 2
    cvar_tail_count: int = 2
    unit_budget: bool = False
    fixed_token_coordinate: bool = False
    dynamic_token_geometry: bool = False
    coordinate_only: bool = False
    coordinate_start: bool = False
    normalize_candidate_uwr: bool = False
    adaptive_risk_appetite: bool = False
    risk_appetite_budget_only: bool = False
    risk_budget_projection: bool = False
    hard_risk_multiple: float = 4.0
    risk_expand_max: float = 1.0
    local_reward_engine: bool = False
    counterfactual_probe_budget: bool = False
    kelly_risk_controller: bool = False
    kelly_fraction: float = 0.25
    kelly_min_observations: int = 3
    kelly_difficulty_ratio: float = 1.05
    kelly_candidate_margin: float = 0.001
    kelly_stable_variance: float = 0.01
    kelly_aggressive_steps: int = 4
    kelly_defensive_hard_risk_multiple: float = 3.0
    kelly_defensive_scale: float = 1.0
    kelly_continuous: bool = False
    kelly_continuous_max_log_scale: float = 0.1
    kelly_continuous_prior_strength: float = 3.0
    ds_risk_controller: bool = False
    ds_max_log_scale: float = 0.5
    ds_window: int = 10
    ds_scale_beta: float = 0.9
    ds_z: float = 1.0
    ds_floor: float = 0.1
    # Equal-risk-contribution (risk parity) frozen-budget redistribution. See
    # RouterConfig for the mechanism. Requires risk_budget_projection so the
    # realized update-to-weight ratio is measured each frozen step.
    erc_budget: bool = False
    erc_risk_beta: float = 0.9
    erc_winsor: float = 3.0
    erc_scale_cap: float = 1.4
    # ERC-specific clamp, decoupled from budget_min/budget_max so the
    # risk-parity reallocation can exceed the routing ceiling. None = inherit
    # the global budget_min/budget_max.
    erc_budget_min: float | None = None
    erc_budget_max: float | None = None
    kelly_beta: float = 0.9
    frozen_budget_scale: float = 1.0
    regime_switching: bool = False
    regime_period_steps: int = 80
    regime_corruption_rate: float = 0.15
    regime_dirty_first: bool = False
    regime_corrupt_targets: bool = True
    regime_fixed_input_mask: bool = False
    regime_probe_matching: bool = False
    regime_scatter: bool = False
    regime_short_context_length: int = 0
    probe_tokens: int = 1_048_576
    val_loss_every: int = 250
    risk_appetite_min: float = 0.75
    risk_appetite_max: float = 1.25
    risk_appetite_beta: float = 0.9

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ExperimentConfig":
        defaults = cls()
        raw_variant = env.get("FARO_VARIANT", defaults.variant)
        if raw_variant not in _VARIANTS:
            raise ValueError(
                f"FARO_VARIANT must be one of {_VARIANTS}, got {raw_variant!r}"
            )
        variant: Variant = raw_variant  # type: ignore[assignment]
        seed = _parse_int(env, "FARO_SEED", defaults.seed)
        run_id = env.get("FARO_RUN_ID", f"{variant}-seed{seed}")
        max_steps = _parse_int(env, "FARO_MAX_STEPS", defaults.max_steps)
        config = cls(
            variant=variant,
            seed=seed,
            run_id=run_id,
            results_root=Path(
                env.get("FARO_RESULTS_ROOT", str(defaults.results_root))
            ),
            max_steps=max_steps,
            matrix_lr=_parse_float(env, "FARO_MATRIX_LR", defaults.matrix_lr),
            muon_lr=_parse_float(env, "FARO_MUON_LR", defaults.muon_lr),
            route_interval=_parse_int(
                env, "FARO_ROUTE_INTERVAL", defaults.route_interval
            ),
            probe_interval=_parse_int(
                env, "FARO_PROBE_INTERVAL", defaults.probe_interval
            ),
            collect_steps=_parse_int(
                env, "FARO_COLLECT_STEPS", defaults.collect_steps
            ),
            min_residence_steps=_parse_int(
                env, "FARO_MIN_RESIDENCE_STEPS", defaults.min_residence_steps
            ),
            challenger_margin=_parse_float(
                env, "FARO_CHALLENGER_MARGIN", defaults.challenger_margin
            ),
            challenger_events=_parse_int(
                env, "FARO_CHALLENGER_EVENTS", defaults.challenger_events
            ),
            cvar_tail_count=_parse_int(
                env, "FARO_CVAR_TAIL_COUNT", defaults.cvar_tail_count
            ),
            unit_budget=_parse_flag(
                env, "FARO_UNIT_BUDGET", defaults.unit_budget
            ),
            fixed_token_coordinate=_parse_flag(
                env,
                "FARO_FIXED_TOKEN_COORDINATE",
                defaults.fixed_token_coordinate,
            ),
            dynamic_token_geometry=_parse_flag(
                env,
                "FARO_DYNAMIC_TOKEN_GEOMETRY",
                defaults.dynamic_token_geometry,
            ),
            coordinate_only=_parse_flag(
                env,
                "FARO_COORDINATE_ONLY",
                defaults.coordinate_only,
            ),
            coordinate_start=_parse_flag(
                env,
                "FARO_COORDINATE_START",
                defaults.coordinate_start,
            ),
            normalize_candidate_uwr=_parse_flag(
                env,
                "FARO_NORMALIZE_CANDIDATE_UWR",
                defaults.normalize_candidate_uwr,
            ),
            adaptive_risk_appetite=_parse_flag(
                env,
                "FARO_ADAPTIVE_RISK_APPETITE",
                defaults.adaptive_risk_appetite,
            ),
            risk_appetite_budget_only=_parse_flag(
                env,
                "FARO_RISK_APPETITE_BUDGET_ONLY",
                defaults.risk_appetite_budget_only,
            ),
            risk_budget_projection=_parse_flag(
                env,
                "FARO_RISK_BUDGET_PROJECTION",
                defaults.risk_budget_projection,
            ),
            hard_risk_multiple=_parse_float(
                env, "FARO_HARD_RISK_MULTIPLE", defaults.hard_risk_multiple
            ),
            risk_expand_max=_parse_float(
                env, "FARO_RISK_EXPAND_MAX", defaults.risk_expand_max
            ),
            local_reward_engine=_parse_flag(
                env,
                "FARO_LOCAL_REWARD_ENGINE",
                defaults.local_reward_engine,
            ),
            counterfactual_probe_budget=_parse_flag(
                env,
                "FARO_COUNTERFACTUAL_PROBE_BUDGET",
                defaults.counterfactual_probe_budget,
            ),
            kelly_risk_controller=_parse_flag(
                env, "FARO_KELLY_RISK_CONTROLLER", defaults.kelly_risk_controller
            ),
            kelly_fraction=_parse_float(
                env, "FARO_KELLY_FRACTION", defaults.kelly_fraction
            ),
            kelly_min_observations=_parse_int(
                env,
                "FARO_KELLY_MIN_OBSERVATIONS",
                defaults.kelly_min_observations,
            ),
            kelly_difficulty_ratio=_parse_float(
                env, "FARO_KELLY_DIFFICULTY_RATIO", defaults.kelly_difficulty_ratio
            ),
            kelly_candidate_margin=_parse_float(
                env, "FARO_KELLY_CANDIDATE_MARGIN", defaults.kelly_candidate_margin
            ),
            kelly_stable_variance=_parse_float(
                env, "FARO_KELLY_STABLE_VARIANCE", defaults.kelly_stable_variance
            ),
            kelly_aggressive_steps=_parse_int(
                env, "FARO_KELLY_AGGRESSIVE_STEPS", defaults.kelly_aggressive_steps
            ),
            kelly_defensive_hard_risk_multiple=_parse_float(
                env,
                "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE",
                defaults.kelly_defensive_hard_risk_multiple,
            ),
            kelly_defensive_scale=_parse_float(
                env,
                "FARO_KELLY_DEFENSIVE_SCALE",
                defaults.kelly_defensive_scale,
            ),
            kelly_beta=_parse_float(env, "FARO_KELLY_BETA", defaults.kelly_beta),
            kelly_continuous=_parse_flag(
                env, "FARO_KELLY_CONTINUOUS", defaults.kelly_continuous
            ),
            kelly_continuous_max_log_scale=_parse_float(
                env,
                "FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE",
                defaults.kelly_continuous_max_log_scale,
            ),
            kelly_continuous_prior_strength=_parse_float(
                env,
                "FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH",
                defaults.kelly_continuous_prior_strength,
            ),
            ds_risk_controller=_parse_flag(
                env, "FARO_DS_RISK_CONTROLLER", defaults.ds_risk_controller
            ),
            ds_max_log_scale=_parse_float(
                env, "FARO_DS_MAX_LOG_SCALE", defaults.ds_max_log_scale
            ),
            ds_window=_parse_int(env, "FARO_DS_WINDOW", defaults.ds_window),
            ds_scale_beta=_parse_float(
                env, "FARO_DS_SCALE_BETA", defaults.ds_scale_beta
            ),
            ds_z=_parse_float(env, "FARO_DS_Z", defaults.ds_z),
            ds_floor=_parse_float(env, "FARO_DS_FLOOR", defaults.ds_floor),
            erc_budget=_parse_flag(env, "FARO_ERC_BUDGET", defaults.erc_budget),
            erc_risk_beta=_parse_float(
                env, "FARO_ERC_RISK_BETA", defaults.erc_risk_beta
            ),
            erc_winsor=_parse_float(env, "FARO_ERC_WINSOR", defaults.erc_winsor),
            erc_scale_cap=_parse_float(
                env, "FARO_ERC_SCALE_CAP", defaults.erc_scale_cap
            ),
            erc_budget_min=_parse_optional_float(
                env, "FARO_ERC_BUDGET_MIN", defaults.erc_budget_min
            ),
            erc_budget_max=_parse_optional_float(
                env, "FARO_ERC_BUDGET_MAX", defaults.erc_budget_max
            ),
            frozen_budget_scale=_parse_float(
                env, "FARO_FROZEN_BUDGET_SCALE", defaults.frozen_budget_scale
            ),
            regime_switching=_parse_flag(
                env, "FARO_REGIME_SWITCHING", defaults.regime_switching
            ),
            regime_period_steps=_parse_int(
                env, "FARO_REGIME_PERIOD", defaults.regime_period_steps
            ),
            regime_corruption_rate=_parse_float(
                env, "FARO_REGIME_CORRUPTION", defaults.regime_corruption_rate
            ),
            regime_dirty_first=_parse_flag(
                env, "FARO_REGIME_DIRTY_FIRST", defaults.regime_dirty_first
            ),
            regime_corrupt_targets=_parse_flag(
                env, "FARO_REGIME_CORRUPT_TARGETS", defaults.regime_corrupt_targets
            ),
            regime_fixed_input_mask=_parse_flag(
                env,
                "FARO_REGIME_FIXED_INPUT_MASK",
                defaults.regime_fixed_input_mask,
            ),
            regime_probe_matching=_parse_flag(
                env,
                "FARO_REGIME_PROBE_MATCHING",
                defaults.regime_probe_matching,
            ),
            regime_scatter=_parse_flag(
                env, "FARO_REGIME_SCATTER", defaults.regime_scatter
            ),
            regime_short_context_length=_parse_int(
                env,
                "FARO_REGIME_SHORT_CONTEXT",
                defaults.regime_short_context_length,
            ),
            probe_tokens=_parse_int(env, "FARO_PROBE_TOKENS", defaults.probe_tokens),
            val_loss_every=_parse_int(
                env, "FARO_VAL_LOSS_EVERY", defaults.val_loss_every
            ),
            risk_appetite_min=_parse_float(
                env, "FARO_RISK_APPETITE_MIN", defaults.risk_appetite_min
            ),
            risk_appetite_max=_parse_float(
                env, "FARO_RISK_APPETITE_MAX", defaults.risk_appetite_max
            ),
            risk_appetite_beta=_parse_float(
                env, "FARO_RISK_APPETITE_BETA", defaults.risk_appetite_beta
            ),
        )
        config._validate()
        return config

    @property
    def is_formal_run(self) -> bool:
        return self.max_steps == 1285

    def run_dir(self, repo_root: Path) -> Path:
        root = self.results_root
        if not root.is_absolute():
            root = repo_root / root
        return root / "runs" / self.variant / f"seed_{self.seed}" / self.run_id

    def _validate(self) -> None:
        if not self.run_id or "/" in self.run_id:
            raise ValueError("FARO_RUN_ID must be a non-empty path component")
        if self.max_steps < 1 or self.max_steps > 1285:
            raise ValueError("FARO_MAX_STEPS must be in [1, 1285]")
        if self.is_formal_run and self.seed not in {0, 1, 2, 42, 1024}:
            raise ValueError("formal seed must be one of 0, 1, 2, 42, or 1024")
        if not self.is_formal_run and not any(
            marker in self.run_id.lower()
            for marker in ("smoke", "pilot", "diagnostic")
        ):
            raise ValueError(
                "a short run id must identify a smoke, pilot, or diagnostic run"
            )
        positive = {
            "FARO_MATRIX_LR": self.matrix_lr,
            "FARO_MUON_LR": self.muon_lr,
            "FARO_ROUTE_INTERVAL": self.route_interval,
            "FARO_PROBE_INTERVAL": self.probe_interval,
            "FARO_COLLECT_STEPS": self.collect_steps,
            "FARO_MIN_RESIDENCE_STEPS": self.min_residence_steps,
            "FARO_CHALLENGER_MARGIN": self.challenger_margin,
            "FARO_CHALLENGER_EVENTS": self.challenger_events,
            "FARO_CVAR_TAIL_COUNT": self.cvar_tail_count,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive FARO values required: {', '.join(invalid)}")
        if self.cvar_tail_count > 8:
            raise ValueError("FARO_CVAR_TAIL_COUNT cannot exceed 8")
        if self.probe_tokens <= 0:
            raise ValueError("FARO_PROBE_TOKENS must be positive")
        if self.val_loss_every < 0:
            raise ValueError("FARO_VAL_LOSS_EVERY must be nonnegative")
        if not (0.0 < self.risk_appetite_min <= 1.0 <= self.risk_appetite_max):
            raise ValueError("risk appetite bounds must contain 1.0")
        if not (0.0 < self.risk_appetite_beta < 1.0):
            raise ValueError("risk appetite beta must be in (0, 1)")
        if self.kelly_risk_controller and not self.counterfactual_probe_budget:
            raise ValueError(
                "FARO_KELLY_RISK_CONTROLLER requires FARO_COUNTERFACTUAL_PROBE_BUDGET"
            )
        if self.kelly_risk_controller and not self.risk_budget_projection:
            raise ValueError(
                "FARO_KELLY_RISK_CONTROLLER requires FARO_RISK_BUDGET_PROJECTION"
            )
        kelly_values = {
            "FARO_KELLY_FRACTION": self.kelly_fraction,
            "FARO_KELLY_DIFFICULTY_RATIO": self.kelly_difficulty_ratio,
            "FARO_KELLY_CANDIDATE_MARGIN": self.kelly_candidate_margin,
            "FARO_KELLY_STABLE_VARIANCE": self.kelly_stable_variance,
            "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE": self.kelly_defensive_hard_risk_multiple,
            "FARO_KELLY_DEFENSIVE_SCALE": self.kelly_defensive_scale,
            "FARO_KELLY_BETA": self.kelly_beta,
            "FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE": self.kelly_continuous_max_log_scale,
            "FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH": self.kelly_continuous_prior_strength,
        }
        for name, value in kelly_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.kelly_beta < 1.0:
            raise ValueError("FARO_KELLY_BETA must be in (0, 1)")
        if self.kelly_risk_controller and (
            self.unit_budget
            or self.local_reward_engine
            or self.adaptive_risk_appetite
        ):
            raise ValueError(
                "FARO_KELLY_RISK_CONTROLLER cannot combine with other budget controllers"
            )
        if self.kelly_fraction <= 0.0:
            raise ValueError("FARO_KELLY_FRACTION must be positive")
        if self.kelly_min_observations <= 0:
            raise ValueError("FARO_KELLY_MIN_OBSERVATIONS must be positive")
        if self.kelly_difficulty_ratio <= 1.0:
            raise ValueError("FARO_KELLY_DIFFICULTY_RATIO must exceed 1")
        if self.kelly_candidate_margin < 0.0:
            raise ValueError("FARO_KELLY_CANDIDATE_MARGIN must be nonnegative")
        if self.kelly_stable_variance <= 0.0:
            raise ValueError("FARO_KELLY_STABLE_VARIANCE must be positive")
        if self.kelly_aggressive_steps <= 0:
            raise ValueError("FARO_KELLY_AGGRESSIVE_STEPS must be positive")
        if self.kelly_defensive_hard_risk_multiple <= 0.0:
            raise ValueError(
                "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE must be positive"
            )
        if self.kelly_defensive_hard_risk_multiple > 4.0:
            raise ValueError(
                "FARO_KELLY_DEFENSIVE_HARD_RISK_MULTIPLE cannot exceed 4.0"
            )
        if not 0.0 < self.kelly_defensive_scale <= 1.0:
            raise ValueError("FARO_KELLY_DEFENSIVE_SCALE must be in (0, 1]")
        if not math.isfinite(self.hard_risk_multiple):
            raise ValueError("FARO_HARD_RISK_MULTIPLE must be finite")
        if not 0.0 < self.hard_risk_multiple <= 4.0:
            raise ValueError("FARO_HARD_RISK_MULTIPLE must be in (0, 4]")
        if not math.isfinite(self.risk_expand_max):
            raise ValueError("FARO_RISK_EXPAND_MAX must be finite")
        if not 1.0 <= self.risk_expand_max <= 4.0:
            raise ValueError("FARO_RISK_EXPAND_MAX must be in [1, 4]")
        if self.kelly_continuous and not self.kelly_risk_controller:
            raise ValueError(
                "FARO_KELLY_CONTINUOUS requires FARO_KELLY_RISK_CONTROLLER"
            )
        if self.kelly_continuous_max_log_scale <= 0.0:
            raise ValueError("FARO_KELLY_CONTINUOUS_MAX_LOG_SCALE must be positive")
        if self.kelly_continuous_prior_strength <= 0.0:
            raise ValueError(
                "FARO_KELLY_CONTINUOUS_PRIOR_STRENGTH must be positive"
            )
        if self.ds_risk_controller and self.kelly_risk_controller:
            raise ValueError(
                "FARO_DS_RISK_CONTROLLER cannot combine with the Kelly controller"
            )
        for name, value in {
            "FARO_DS_MAX_LOG_SCALE": self.ds_max_log_scale,
            "FARO_DS_Z": self.ds_z,
            "FARO_DS_FLOOR": self.ds_floor,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.ds_max_log_scale <= 0.0:
            raise ValueError("FARO_DS_MAX_LOG_SCALE must be positive")
        if self.ds_window < 2:
            raise ValueError("FARO_DS_WINDOW must be at least 2")
        if not 0.0 < self.ds_scale_beta < 1.0:
            raise ValueError("FARO_DS_SCALE_BETA must be in (0, 1)")
        if self.ds_z <= 0.0:
            raise ValueError("FARO_DS_Z must be positive")
        if self.ds_floor <= 0.0:
            raise ValueError("FARO_DS_FLOOR must be positive")
        if self.erc_budget and (
            self.unit_budget or self.kelly_risk_controller or self.ds_risk_controller
        ):
            raise ValueError(
                "FARO_ERC_BUDGET cannot combine with unit_budget or other budget controllers"
            )
        if self.erc_budget and not self.risk_budget_projection:
            raise ValueError(
                "FARO_ERC_BUDGET requires FARO_RISK_BUDGET_PROJECTION=1 to measure block risk"
            )
        if not 0.0 < self.erc_risk_beta < 1.0:
            raise ValueError("FARO_ERC_RISK_BETA must be in (0, 1)")
        if self.erc_winsor <= 1.0:
            raise ValueError("FARO_ERC_WINSOR must exceed 1")
        if self.erc_scale_cap <= 1.0:
            raise ValueError("FARO_ERC_SCALE_CAP must exceed 1")
        for name, value in (
            ("FARO_ERC_BUDGET_MIN", self.erc_budget_min),
            ("FARO_ERC_BUDGET_MAX", self.erc_budget_max),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            self.erc_budget_min is not None
            and self.erc_budget_max is not None
            and self.erc_budget_min >= self.erc_budget_max
        ):
            raise ValueError("FARO_ERC_BUDGET_MIN must be below FARO_ERC_BUDGET_MAX")
        if not math.isfinite(self.frozen_budget_scale):
            raise ValueError("FARO_FROZEN_BUDGET_SCALE must be finite")
        if not 0.0 < self.frozen_budget_scale <= 1.0:
            raise ValueError("FARO_FROZEN_BUDGET_SCALE must be in (0, 1]")
        if self.regime_period_steps <= 0:
            raise ValueError("FARO_REGIME_PERIOD must be positive")
        if self.regime_short_context_length < 0:
            raise ValueError("FARO_REGIME_SHORT_CONTEXT must be nonnegative")
        if not math.isfinite(self.regime_corruption_rate):
            raise ValueError("FARO_REGIME_CORRUPTION must be finite")
        if not 0.0 <= self.regime_corruption_rate < 1.0:
            raise ValueError("FARO_REGIME_CORRUPTION must be in [0, 1)")


def _parse_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error


def _parse_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a float") from error


def _parse_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "1" if default else "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"{key} must be 0 or 1")
    return raw == "1"


def _parse_optional_float(
    env: Mapping[str, str], key: str, default: float | None
) -> float | None:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a float") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_field(digest: "hashlib._Hash", value: str | bytes) -> None:
    payload = value.encode() if isinstance(value, str) else value
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def hash_named_tensors(named_tensors: Iterable[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        detached = tensor.detach().contiguous()
        _hash_field(digest, name)
        _hash_field(digest, str(detached.dtype))
        _hash_field(digest, repr(tuple(detached.shape)))
        raw = detached.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        _hash_field(digest, raw)
    return digest.hexdigest()


def _expand_paths(patterns: Iterable[str | os.PathLike[str]]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        raw = os.fspath(pattern)
        matches = glob.glob(raw)
        if matches:
            paths.update(Path(match).resolve() for match in matches)
        elif Path(raw).exists():
            paths.add(Path(raw).resolve())
    return sorted(paths)


def hash_data_manifest(patterns: Iterable[str | os.PathLike[str]]) -> str:
    paths = _expand_paths(patterns)
    if not paths:
        raise FileNotFoundError("data manifest contains no files")
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        _hash_field(digest, str(path))
        _hash_field(digest, str(stat.st_size))
        _hash_field(digest, str(stat.st_mtime_ns))
    return digest.hexdigest()


def hash_code_files(
    paths: Iterable[str | os.PathLike[str]], *, root: Path
) -> str:
    resolved_root = root.resolve()
    files = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in files:
        try:
            label = path.relative_to(resolved_root)
        except ValueError:
            label = path
        _hash_field(digest, str(label))
        _hash_field(digest, path.read_bytes())
    return digest.hexdigest()


def build_run_metadata(
    config: ExperimentConfig,
    repo_root: Path,
    named_tensors: Iterable[tuple[str, Tensor]],
    bigram_sign_table: Tensor,
    *,
    data_patterns: Sequence[str | os.PathLike[str]],
    code_paths: Sequence[str | os.PathLike[str]],
    world_size: int,
    grad_accum_steps: int,
) -> dict[str, object]:
    config_payload = asdict(config)
    config_payload["results_root"] = str(config.results_root)
    cuda_metadata: dict[str, object] = {
        "available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_metadata.update(
            {
                "device_index": index,
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
            }
        )
    return {
        "config": config_payload,
        "run_id": config.run_id,
        "variant": config.variant,
        "seed": config.seed,
        "initialization_hash": hash_named_tensors(named_tensors),
        "bigram_sign_table_hash": hash_named_tensors(
            [("bigram_sign_table", bigram_sign_table)]
        ),
        "data_manifest_hash": hash_data_manifest(data_patterns),
        "code_hash": hash_code_files(code_paths, root=repo_root),
        "python": os.sys.version,
        "pytorch": torch.__version__,
        "cuda": cuda_metadata,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "hostname": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def select_adam_pilot(
    pilot_dirs: Sequence[Path],
    *,
    faro_summaries: Sequence[Path] = (),
) -> dict[str, object]:
    declared_lrs = {0.00075, 0.0015, 0.003}
    candidates: list[dict[str, object]] = []
    for directory in pilot_dirs:
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lr = float(summary["matrix_lr"])
        loss = float(summary["final_validation_loss"])
        stable = (
            summary.get("status") == "complete"
            and int(summary.get("step", -1)) == 250
            and not bool(summary.get("loss_spike", False))
            and math.isfinite(loss)
        )
        candidates.append(
            {
                "matrix_lr": lr,
                "validation_loss": loss,
                "stable": stable,
                "source": str(summary_path.resolve()),
                "created_at": str(summary["created_at"]),
            }
        )
    if {float(candidate["matrix_lr"]) for candidate in candidates} != declared_lrs:
        raise ValueError("pilot inputs must contain exactly the declared Adam LRs")
    stable_candidates = [candidate for candidate in candidates if candidate["stable"]]
    if not stable_candidates:
        raise ValueError("no stable Adam pilot is available")
    selected = min(
        stable_candidates,
        key=lambda candidate: (
            float(candidate["validation_loss"]),
            float(candidate["matrix_lr"]),
        ),
    )
    selected_at = datetime.now(timezone.utc)
    for path in faro_summaries:
        faro_summary = json.loads(path.read_text(encoding="utf-8"))
        faro_started = datetime.fromisoformat(str(faro_summary["created_at"]))
        if faro_started < selected_at:
            raise ValueError("FARO run started before Adam LR was frozen")
    return {
        "selected_lr": float(selected["matrix_lr"]),
        "selected_at": selected_at.isoformat(),
        "rule": "lowest stable step-250 validation loss; lower LR breaks ties",
        "candidates": sorted(
            candidates, key=lambda candidate: float(candidate["matrix_lr"])
        ),
    }
