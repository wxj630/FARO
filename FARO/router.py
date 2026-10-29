from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .descriptors import (
    BlockDescriptor,
    Geometry,
    legal_geometries,
    structure_prior,
)
from .experiment import ExperimentConfig


_INVALID_SCORE = -1e30

# Risks that are identical for every candidate of a block, so they cannot help
# distinguish geometries. They are reported and scored, but must not gate the
# geometry switch, otherwise one bad block-level reading freezes all candidates.
_NON_DISCRIMINATING_RISKS = frozenset({"drift"})


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    returns: tuple[float, ...]
    mean_return: float
    mean_abs_return: float
    cvar: float
    uwr: float
    max_axis_uwr: float
    spectral_risk: float
    drift: float
    gradient_sv_concentration: float
    momentum_sv_concentration: float

    @classmethod
    def from_returns(
        cls,
        returns: Tensor,
        *,
        uwr: Tensor,
        max_axis_uwr: Tensor,
        spectral_risk: Tensor,
        drift: Tensor,
        gradient_sv_concentration: Tensor,
        momentum_sv_concentration: Tensor,
    ) -> "CandidateObservation":
        return cls.from_device_values(
            returns,
            uwr=uwr,
            max_axis_uwr=max_axis_uwr,
            spectral_risk=spectral_risk,
            drift=drift,
            gradient_sv_concentration=gradient_sv_concentration,
            momentum_sv_concentration=momentum_sv_concentration,
        )

    @classmethod
    def from_device_values(
        cls,
        returns: Tensor,
        *,
        uwr: Tensor,
        max_axis_uwr: Tensor,
        spectral_risk: Tensor,
        drift: Tensor,
        gradient_sv_concentration: Tensor,
        momentum_sv_concentration: Tensor,
    ) -> "CandidateObservation":
        flat_returns = returns.detach().float().flatten()
        if flat_returns.numel() != 8:
            raise ValueError("FARO routing requires exactly eight micro returns")
        device = flat_returns.device
        scalars = torch.stack(
            [
                value.detach().to(device=device, dtype=torch.float32).reshape(())
                for value in (
                    uwr,
                    max_axis_uwr,
                    spectral_risk,
                    drift,
                    gradient_sv_concentration,
                    momentum_sv_concentration,
                )
            ]
        )
        packed = torch.cat((flat_returns, scalars)).cpu()
        flat = packed[:8]
        sorted_returns = flat.sort().values
        cvar = max(0.0, -float(sorted_returns[:2].mean()))
        values = tuple(float(value) for value in flat)
        return cls(
            returns=values,
            mean_return=float(flat.mean()),
            mean_abs_return=float(flat.abs().mean()),
            cvar=cvar,
            uwr=float(packed[8]),
            max_axis_uwr=float(packed[9]),
            spectral_risk=float(packed[10]),
            drift=float(packed[11]),
            gradient_sv_concentration=float(packed[12]),
            momentum_sv_concentration=float(packed[13]),
        )


@dataclass(frozen=True, slots=True)
class RouterConfig:
    route_interval: int = 8
    collect_steps: int = 32
    min_residence_steps: int = 16
    challenger_margin: float = 0.10
    challenger_events: int = 2
    ema_abs_return_beta: float = 0.9
    reference_beta: float = 0.99
    budget_beta: float = 0.9
    budget_min: float = 0.5
    budget_max: float = 1.1
    max_budget_event_change: float = 0.20
    hard_risk_multiple: float = 4.0
    # Two-sided risk projection. 1.0 keeps the historical one-sided behaviour
    # (only contract risky updates). Above 1.0 lets low-risk updates expand up
    # to this multiple of the nominal budget -- the risk-seeking side of the
    # risk-appetite spectrum -- still bounded by budget_max.
    risk_expand_max: float = 1.0
    spectral_concentration_weight: float = 0.4
    fixed_geometry: bool = False
    freeze_stable_routes: bool = False
    unit_budget: bool = False
    fixed_token_coordinate: bool = False
    # Opt-in: let token lookup/output blocks explore row/column instead of the
    # default coordinate-only incumbent.
    dynamic_token_geometry: bool = False
    # Opt-in: force every block to coordinate geometry (no spectral/row_column).
    coordinate_only: bool = False
    # Opt-in: start every block at coordinate but keep all legal candidates, so
    # the router must discover the better geometry on its own (no structure prior).
    coordinate_start: bool = False
    # Opt-in: express every candidate's update-to-weight ratio on the coordinate
    # scale before comparing it, so a geometry is not flagged as risky merely for
    # having a larger intrinsic update norm (spectral is ~3x coordinate).
    normalize_candidate_uwr: bool = False
    adaptive_risk_appetite: bool = False
    # Isolate appetite from geometry selection so it only controls update scale.
    risk_appetite_budget_only: bool = False
    risk_budget_projection: bool = False
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
    # Risk-averse frozen budget multiplier (<=1.0 shrinks the bet). The default
    # keeps the historical budget-neutral behaviour; smaller values make the
    # defensive state actually under-step, which is the fractional-Kelly reading
    # of reducing a position once returns are stable.
    kelly_defensive_scale: float = 1.0
    # Payoff-stream EWMA decay. Must be faster than the environment's change
    # timescale (e.g. regime phase length) or the controller cannot track it;
    # 0.9 keeps ~10 observations of memory at the 16-step probe cadence.
    kelly_beta: float = 0.9
    # Continuous Kelly-inspired budget controller. Replaces the discrete
    # 0.9/1.0/1.1 action set with a bounded log-scale map of the local
    # counterfactual edge, shrunk by how little evidence the stream holds.
    kelly_continuous: bool = False
    kelly_continuous_max_log_scale: float = 0.1
    kelly_continuous_prior_strength: float = 3.0
    # Difficulty-sensed controller: gate the frozen budget on the held-out probe
    # return trend. The local risk ratio is blind to data difficulty (Cohen's d
    # ~= 0.06); the held-out probe return, detrended by a trailing median,
    # separates clean/dirty stretches ~7x better (d ~= 0.40). A shortfall of the
    # current probe return below the trailing baseline (normalized by its own
    # dispersion) activates a bounded budget shrink.
    ds_risk_controller: bool = False
    ds_max_log_scale: float = 0.5
    ds_window: int = 10
    ds_scale_beta: float = 0.9
    ds_z: float = 1.0
    ds_floor: float = 0.1
    # Equal-risk-contribution (risk parity) frozen-budget controller. Instead of
    # scaling one global scalar, it reallocates the *total* frozen budget across
    # blocks so each block contributes an equal share of the update-to-weight
    # risk: w_b ~ 1/risk_b, renormalised to be budget-neutral (sum of block
    # budgets unchanged). Risk is an EMA of the realized update-to-weight ratio,
    # winsorised and the per-block multiplier capped, because raw uwr is
    # dominated by tiny-norm scalar/lambda blocks (spans ~14 orders of
    # magnitude). Geometry, routing and the freeze timeline are untouched.
    erc_budget: bool = False
    erc_risk_beta: float = 0.9
    erc_winsor: float = 3.0
    erc_scale_cap: float = 1.4
    # ERC-specific budget clamp, deliberately decoupled from the global
    # budget_min/budget_max so the risk-parity reallocation can let low-risk
    # blocks step past the routing ceiling without touching the geometry
    # controls (budget_max also bounds routing targets). ``None`` keeps the
    # historical behaviour: fall back to the global budget_min/budget_max.
    erc_budget_min: float | None = None
    erc_budget_max: float | None = None
    frozen_budget_scale: float = 1.0
    probe_tokens: int = 1_048_576
    risk_appetite_min: float = 0.75
    risk_appetite_max: float = 1.25
    risk_appetite_beta: float = 0.9
    eps: float = 1e-8

    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "RouterConfig":
        return cls(
            route_interval=config.route_interval,
            collect_steps=config.collect_steps,
            min_residence_steps=config.min_residence_steps,
            challenger_margin=config.challenger_margin,
            challenger_events=config.challenger_events,
            fixed_geometry=config.variant == "faro_fixed",
            freeze_stable_routes=config.variant == "faro",
            unit_budget=config.unit_budget,
            fixed_token_coordinate=config.fixed_token_coordinate,
            dynamic_token_geometry=config.dynamic_token_geometry,
            coordinate_only=config.coordinate_only,
            coordinate_start=config.coordinate_start,
            normalize_candidate_uwr=config.normalize_candidate_uwr,
            adaptive_risk_appetite=config.adaptive_risk_appetite,
            risk_appetite_budget_only=config.risk_appetite_budget_only,
            risk_budget_projection=config.risk_budget_projection,
            hard_risk_multiple=config.hard_risk_multiple,
            risk_expand_max=config.risk_expand_max,
            local_reward_engine=config.local_reward_engine,
            counterfactual_probe_budget=config.counterfactual_probe_budget,
            kelly_risk_controller=config.kelly_risk_controller,
            kelly_fraction=config.kelly_fraction,
            kelly_min_observations=config.kelly_min_observations,
            kelly_difficulty_ratio=config.kelly_difficulty_ratio,
            kelly_candidate_margin=config.kelly_candidate_margin,
            kelly_stable_variance=config.kelly_stable_variance,
            kelly_aggressive_steps=config.kelly_aggressive_steps,
            kelly_defensive_hard_risk_multiple=(
                config.kelly_defensive_hard_risk_multiple
            ),
            kelly_defensive_scale=config.kelly_defensive_scale,
            kelly_beta=config.kelly_beta,
            kelly_continuous=config.kelly_continuous,
            kelly_continuous_max_log_scale=config.kelly_continuous_max_log_scale,
            kelly_continuous_prior_strength=config.kelly_continuous_prior_strength,
            ds_risk_controller=config.ds_risk_controller,
            ds_max_log_scale=config.ds_max_log_scale,
            ds_window=config.ds_window,
            ds_scale_beta=config.ds_scale_beta,
            ds_z=config.ds_z,
            ds_floor=config.ds_floor,
            erc_budget=config.erc_budget,
            erc_risk_beta=config.erc_risk_beta,
            erc_winsor=config.erc_winsor,
            erc_scale_cap=config.erc_scale_cap,
            erc_budget_min=config.erc_budget_min,
            erc_budget_max=config.erc_budget_max,
            frozen_budget_scale=config.frozen_budget_scale,
            probe_tokens=config.probe_tokens,
            risk_appetite_min=config.risk_appetite_min,
            risk_appetite_max=config.risk_appetite_max,
            risk_appetite_beta=config.risk_appetite_beta,
        )


@dataclass(slots=True)
class BlockRouteState:
    current_mode: Geometry
    structure_class: str = "coordinate"
    budget_scale: float = 1.0
    last_switch_step: int = 0
    stable_since_step: int = 0
    frozen: bool = False
    frozen_step: int | None = None
    challenger_mode: Geometry | None = None
    challenger_wins: int = 0
    reference_initialized: bool = False
    reference_ready: bool = False
    reference_cvar: float = 0.0
    reference_drift: float = 0.0
    reference_uwr: float = 0.0
    reference_spectral: float = 0.0
    ema_abs_return: dict[Geometry, float] = field(default_factory=dict)
    mode_counts: dict[Geometry, int] = field(default_factory=dict)
    risk_appetite: float = 1.0
    risk_appetite_steps: int = 0
    uwr_scale: dict[Geometry, float] = field(default_factory=dict)
    uwr_scale_samples: dict[Geometry, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    label: str
    mode: Geometry
    budget_scale: float
    switched: bool
    route_event: bool
    reason: str
    scores: dict[Geometry, float]
    hard_violations: tuple[Geometry, ...]
    risk_ratios: dict[Geometry, dict[str, float | None]]
    hard_risk_triggers: dict[Geometry, tuple[str, ...]]
    candidate_statistics: dict[Geometry, dict[str, float | None]]
    budget_statistics: dict[str, object]


class FARORouter:
    def __init__(self, config: RouterConfig):
        # The defensive multiple only takes effect inside the Kelly controller, so
        # it constrains the neutral multiple only when Kelly is actually enabled.
        if (
            config.kelly_risk_controller
            and config.kelly_defensive_hard_risk_multiple > config.hard_risk_multiple
        ):
            raise ValueError(
                "Kelly defensive hard-risk multiple cannot exceed hard-risk multiple"
            )
        self.config = config
        self.states: dict[str, BlockRouteState] = {}
        self.global_calibration_error = 0.0
        self.global_calibration_abs = 0.0
        self.global_calibration_count = 0
        self.structure_calibration_error: dict[str, float] = {}
        self.structure_calibration_abs: dict[str, float] = {}
        self.structure_calibration_count: dict[str, int] = {}
        # Running baseline of realized/predicted per structure. Predicted return
        # is a first-order in-sample estimate while realized return is measured on
        # a disjoint probe slice, so this ratio is structurally far below 1 and
        # its *level* carries no signal. Appetite reacts to deviation from this
        # baseline, not to the level, so the constant bias cannot pin appetite.
        self.structure_calibration_ratio: dict[str, float] = {}
        self.appetite_return_scale: float = 0.0
        self.kelly_state = "neutral"
        self.kelly_scale = 1.0
        self.kelly_steps_remaining = 0
        self.kelly_observations = 0
        self.kelly_return_mean = 0.0
        self.kelly_return_variance = 0.0
        self.kelly_probe_loss_ema = 0.0
        self.kelly_projection_multiple = config.hard_risk_multiple
        # Audit-only diagnostics for the continuous map (also serialized so a
        # resumed run reports the same last decision).
        self.kelly_edge = 0.0
        self.kelly_confidence = 0.0
        self.kelly_log_scale = 0.0
        self.ds_recent: list[float] = []
        self.ds_baseline = 0.0
        self.ds_scale = 0.0
        # Equal-risk-contribution frozen-budget state. ``erc_risk`` is a
        # per-block EMA of the realized update-to-weight ratio; ``erc_multiplier``
        # is the budget-neutral reallocation applied to each frozen block.
        self.erc_risk: dict[str, float] = {}
        self.erc_multiplier: dict[str, float] = {}

    def record_erc_risk(self, label: str, update_uwr: float) -> None:
        if not self.config.erc_budget:
            return
        if not math.isfinite(update_uwr) or update_uwr <= 0.0:
            return
        previous = self.erc_risk.get(label)
        beta = self.config.erc_risk_beta
        self.erc_risk[label] = (
            update_uwr
            if previous is None
            else beta * previous + (1.0 - beta) * update_uwr
        )

    def compute_erc_multipliers(
        self, base_budgets: Mapping[str, float]
    ) -> dict[str, float]:
        """Reallocate the total frozen budget so each block carries equal risk.

        Weights are ``1/risk`` renormalised so ``sum(base * multiplier)`` equals
        ``sum(base)`` -- this is a *redistribution*, not an expansion, of the
        frozen budget. Raw risk is winsorised around the cross-block median and
        the multiplier is capped, because a handful of scalar/lambda blocks have
        near-zero weight norms and degenerate update-to-weight ratios.
        """
        if not self.config.erc_budget:
            return {}
        risks: dict[str, float] = {}
        for label, base_budget in base_budgets.items():
            value = self.erc_risk.get(label)
            if value is None:
                state = self.states.get(label)
                value = state.reference_uwr if state is not None else None
            if value is None or not math.isfinite(value) or value <= 0.0:
                continue
            if not math.isfinite(base_budget) or base_budget <= 0.0:
                continue
            risks[label] = value
        if len(risks) < 2:
            self.erc_multiplier = {}
            return {}
        median = statistics.median(risks.values())
        winsor = self.config.erc_winsor
        low, high = median / winsor, median * winsor
        risk_w = {label: min(max(value, low), high) for label, value in risks.items()}
        base = {label: float(base_budgets[label]) for label in risk_w}
        total = sum(base.values())
        raw = {label: 1.0 / risk_w[label] for label in risk_w}
        cap = self.config.erc_scale_cap
        lower, upper = 1.0 / cap, cap

        def allocated(scale: float) -> float:
            return sum(
                base[label] * min(max(scale * raw[label], lower), upper)
                for label in risk_w
            )

        # ``allocated`` is nondecreasing in ``scale``; bisect for the scale that
        # distributes exactly the original budget. Clipping is applied inside, so
        # the result is budget-neutral *and* within [1/cap, cap].
        lo, hi = 0.0, 1.0
        while allocated(hi) < total and hi < 1e12:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if allocated(mid) < total:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        self.erc_multiplier = {
            label: min(max(scale * raw[label], lower), upper)
            for label in risk_w
        }
        return dict(self.erc_multiplier)

    def erc_scale(self, label: str) -> float:
        return self.erc_multiplier.get(label, 1.0)

    def is_frozen(self, label: str) -> bool:
        state = self.states.get(label)
        return bool(state and state.frozen)

    def update_kelly_state(
        self,
        *,
        probe_before_loss: float,
        realized_return: float,
        audit: Mapping[str, object],
    ) -> dict[str, object]:
        if not self.config.kelly_risk_controller:
            return {}
        scales = tuple(float(value) for value in audit.get("scales", ()))
        losses = tuple(float(value) for value in audit.get("losses", ()))
        if len(scales) != 3 or len(losses) != 3:
            raise ValueError("Kelly probe audit must contain three scales and losses")
        if not all(math.isfinite(value) for value in (*scales, *losses, probe_before_loss, realized_return)):
            raise FloatingPointError("nonfinite Kelly probe signal")
        if probe_before_loss <= self.config.eps:
            raise ValueError("Kelly probe baseline loss must be positive")
        normalized_return = realized_return / max(abs(probe_before_loss), self.config.eps)
        neutral_index = min(
            range(len(scales)), key=lambda index: abs(scales[index] - 1.0)
        )
        neutral_loss = losses[neutral_index]
        previous_loss_ema = self.kelly_probe_loss_ema
        beta = self.config.kelly_beta
        if self.kelly_observations == 0:
            self.kelly_return_mean = normalized_return
            self.kelly_return_variance = 0.0
            self.kelly_probe_loss_ema = neutral_loss
        else:
            delta = normalized_return - self.kelly_return_mean
            self.kelly_return_mean = beta * self.kelly_return_mean + (1.0 - beta) * normalized_return
            self.kelly_return_variance = beta * self.kelly_return_variance + (1.0 - beta) * delta * delta
            self.kelly_probe_loss_ema = beta * self.kelly_probe_loss_ema + (1.0 - beta) * neutral_loss
        self.kelly_observations += 1
        full_kelly = self.kelly_return_mean / max(self.kelly_return_variance, self.config.eps)
        fractional_kelly = max(-1.0, min(1.0, self.config.kelly_fraction * full_kelly))
        if self.config.kelly_continuous:
            return self._update_continuous_kelly(
                scales=scales,
                losses=losses,
                neutral_loss=neutral_loss,
                fractional_kelly=fractional_kelly,
            )
        aggressive_edge = losses[2] < neutral_loss - (
            self.config.kelly_candidate_margin
            * max(abs(neutral_loss), self.config.eps)
        )
        difficult = (
            previous_loss_ema > 0.0
            and neutral_loss >= self.config.kelly_difficulty_ratio * previous_loss_ema
        )
        if self.kelly_state == "aggressive" and self.kelly_steps_remaining > 0:
            pass
        elif (
            self.kelly_observations >= self.config.kelly_min_observations
            and difficult
            and aggressive_edge
            and fractional_kelly > 0.0
        ):
            self.kelly_state = "aggressive"
            self.kelly_scale = 1.1
            self.kelly_steps_remaining = self.config.kelly_aggressive_steps
            self.kelly_projection_multiple = self.config.hard_risk_multiple
        elif (
            self.kelly_observations >= self.config.kelly_min_observations
            and self.kelly_return_mean > 0.0
            and self.kelly_return_variance <= self.config.kelly_stable_variance
        ):
            self.kelly_state = "defensive"
            self.kelly_scale = self.config.kelly_defensive_scale
            self.kelly_steps_remaining = 0
            self.kelly_projection_multiple = min(
                self.config.hard_risk_multiple,
                self.config.kelly_defensive_hard_risk_multiple,
            )
        else:
            self.kelly_state = "neutral"
            self.kelly_scale = 1.0
            self.kelly_steps_remaining = 0
            self.kelly_projection_multiple = self.config.hard_risk_multiple
        return {
            "kelly_state": self.kelly_state,
            "kelly_scale": self.kelly_scale,
            "kelly_fraction": fractional_kelly,
            "kelly_observations": self.kelly_observations,
            "kelly_return_mean": self.kelly_return_mean,
            "kelly_return_variance": self.kelly_return_variance,
            "kelly_probe_loss_ema": self.kelly_probe_loss_ema,
            "kelly_steps_remaining": self.kelly_steps_remaining,
            "kelly_projection_multiple": self.kelly_projection_multiple,
        }

    def _update_continuous_kelly(
        self,
        *,
        scales: tuple[float, ...],
        losses: tuple[float, ...],
        neutral_loss: float,
        fractional_kelly: float,
    ) -> dict[str, object]:
        """Bounded, evidence-shrunk log-scale map of the local probe edge.

        The continuous controller treats the same-state counterfactual as a
        local bet: stepping the frozen budget up by ``large_scale - 1`` is the
        wager and the relative loss improvement of that larger candidate is the
        edge. Risk is proxied by the spread of the three candidate losses on the
        same update direction -- the closest available estimate of how noisy a
        budget decision is right now. The edge/risk ratio is squashed into
        ``(-1, 1)`` and scaled by the EWMA evidence mass, so weak evidence or a
        flat probe collapses the action back onto the risk-neutral prior (1.0)
        instead of jumping between hard-coded multiples.
        """
        eps = self.config.eps
        baseline = max(abs(neutral_loss), eps)
        large_index = max(range(len(scales)), key=scales.__getitem__)
        large_loss = losses[large_index]
        edge = (neutral_loss - large_loss) / baseline
        loss_spread = (max(losses) - min(losses)) / baseline
        noise = max(loss_spread, self.config.kelly_candidate_margin, eps)
        raw_signal = edge / noise
        bounded_signal = raw_signal / (1.0 + abs(raw_signal))
        n_eff = float(self.kelly_observations)
        confidence = n_eff / (n_eff + self.config.kelly_continuous_prior_strength)
        max_log_scale = self.config.kelly_continuous_max_log_scale
        log_scale = max(
            -max_log_scale,
            min(max_log_scale, max_log_scale * confidence * bounded_signal),
        )
        self.kelly_scale = math.exp(log_scale)
        self.kelly_steps_remaining = 0
        self.kelly_projection_multiple = self.config.hard_risk_multiple
        if log_scale > eps:
            self.kelly_state = "aggressive"
        elif log_scale < -eps:
            self.kelly_state = "defensive"
        else:
            self.kelly_state = "neutral"
        self.kelly_edge = edge
        self.kelly_confidence = confidence
        self.kelly_log_scale = log_scale
        return {
            "kelly_state": self.kelly_state,
            "kelly_scale": self.kelly_scale,
            "kelly_fraction": fractional_kelly,
            "kelly_observations": self.kelly_observations,
            "kelly_return_mean": self.kelly_return_mean,
            "kelly_return_variance": self.kelly_return_variance,
            "kelly_probe_loss_ema": self.kelly_probe_loss_ema,
            "kelly_steps_remaining": self.kelly_steps_remaining,
            "kelly_projection_multiple": self.kelly_projection_multiple,
            "kelly_continuous": True,
            "kelly_edge": edge,
            "kelly_confidence": confidence,
            "kelly_log_scale": log_scale,
        }

    def kelly_audit(self) -> dict[str, object]:
        if not self.config.kelly_risk_controller:
            return {}
        audit: dict[str, object] = {
            "kelly_state": self.kelly_state,
            "kelly_scale": self.kelly_scale,
            "kelly_steps_remaining": self.kelly_steps_remaining,
            "kelly_projection_multiple": self.kelly_projection_multiple,
        }
        if self.config.kelly_continuous:
            audit.update(
                {
                    "kelly_continuous": True,
                    "kelly_edge": self.kelly_edge,
                    "kelly_confidence": self.kelly_confidence,
                    "kelly_log_scale": self.kelly_log_scale,
                }
            )
        return audit

    def advance_kelly_step(self) -> dict[str, object]:
        if not self.config.kelly_risk_controller:
            return {}
        if self.kelly_state == "aggressive" and self.kelly_steps_remaining > 0:
            self.kelly_steps_remaining -= 1
            if self.kelly_steps_remaining == 0:
                self.kelly_state = "neutral"
                self.kelly_scale = 1.0
                self.kelly_projection_multiple = self.config.hard_risk_multiple
        return self.kelly_audit()

    def update_ds_state(self, *, realized_return: float) -> dict[str, object]:
        """Shrink the frozen budget while the held-out probe return dips below
        its trailing baseline.

        The local update-magnitude ratio is blind to data difficulty (Cohen's d
        ~= 0.06); the held-out probe return, detrended by a trailing median,
        separates clean/dirty stretches ~7x better (d ~= 0.40). A shortfall of
        the current return below the median of the last ``ds_window`` probes,
        measured in units of its own dispersion, activates a bounded shrink.
        """
        if not self.config.ds_risk_controller:
            return {}
        level = float(realized_return)
        if not math.isfinite(level):
            raise FloatingPointError("nonfinite DS probe return")
        window = self.ds_recent
        activation = 0.0
        residual = 0.0
        if len(window) >= self.config.ds_window:
            baseline = statistics.median(window)
            residual = level - baseline
            self.ds_baseline = baseline
            # Long-run scale of the residual (not the short-window spread, which
            # co-moves with the dip and cancels it); this is what preserves the
            # clean/dirty separation (Cohen's d ~= 0.45).
            size = abs(residual)
            self.ds_scale = (
                size
                if self.ds_scale <= 0.0
                else self.config.ds_scale_beta * self.ds_scale
                + (1.0 - self.config.ds_scale_beta) * size
            )
            scale = max(self.ds_scale, self.config.ds_floor)
            shortfall = -residual / scale  # dip below baseline -> positive
            activation = min(1.0, max(0.0, shortfall / self.config.ds_z))
        window.append(level)
        if len(window) > self.config.ds_window:
            del window[0]
        log_scale = -self.config.ds_max_log_scale * activation
        self.kelly_log_scale = log_scale
        self.kelly_scale = math.exp(log_scale)
        self.kelly_state = "defensive" if activation > self.config.eps else "neutral"
        self.kelly_projection_multiple = self.config.hard_risk_multiple
        return {
            "ds_risk_controller": True,
            "ds_baseline": self.ds_baseline,
            "ds_scale": self.ds_scale,
            "ds_activation": activation,
            "kelly_state": self.kelly_state,
            "kelly_scale": self.kelly_scale,
            "kelly_projection_multiple": self.kelly_projection_multiple,
        }

    def local_reward_target(
        self, returns: Sequence[float]
    ) -> tuple[float, dict[str, float]]:
        values = [float(value) for value in returns]
        if len(values) != 8 or not all(math.isfinite(value) for value in values):
            raise ValueError("local reward returns must contain eight finite values")
        mean = statistics.fmean(values)
        mean_abs = statistics.fmean(abs(value) for value in values)
        sorted_values = sorted(values)
        cvar = max(0.0, -statistics.fmean(sorted_values[:2]))
        scale = max(mean_abs, self.config.eps)
        edge = min(1.0, max(-1.0, (mean - cvar) / scale))
        variance = min(
            4.0,
            max(
                0.0,
                statistics.pvariance(values) / (scale * scale),
            ),
        )
        target = min(
            self.config.budget_max,
            max(
                self.config.budget_min,
                (1.0 + edge) / (1.0 + variance),
            ),
        )
        return target, {
            "reward_edge": edge,
            "reward_variance": variance,
            "reward_target": target,
        }

    def local_reward_budget(
        self,
        label: str,
        base_budget: float,
        observation: CandidateObservation,
    ) -> tuple[float, dict[str, float | bool]]:
        if self.config.unit_budget or not self.config.local_reward_engine:
            return base_budget, {}
        state = self.states.get(label)
        if state is None:
            raise KeyError(f"unknown FARO route state: {label}")
        raw_target, reward_statistics = self.local_reward_target(
            observation.returns
        )
        base_statistics = self._budget_statistics(
            state,
            observation,
            state.structure_class,
            collection_locked=False,
            unit_budget=False,
        )
        risk_capacity = float(base_statistics["risk_capacity"])
        return_cap = float(base_statistics["return_cap"])
        target = min(raw_target, risk_capacity, return_cap)
        target = min(
            self.config.budget_max,
            max(self.config.budget_min, target),
        )
        budget = min(
            self.config.budget_max,
            max(self.config.budget_min, base_budget * target),
        )
        return budget, {
            **base_statistics,
            **reward_statistics,
            "reward_engine": True,
            "reward_raw_target": raw_target,
            "reward_target": target,
            "target": budget,
        }

    def risk_budget_cap(
        self, label: str, uwr: float, max_axis_uwr: float
    ) -> float | None:
        if self.config.unit_budget or not self.config.risk_budget_projection:
            return None
        state = self.states.get(label)
        if state is None or state.reference_uwr <= self.config.eps:
            return None
        update_uwr = max(uwr, max_axis_uwr)
        if not math.isfinite(update_uwr) or update_uwr <= self.config.eps:
            return None
        hard_multiple = (
            self.kelly_projection_multiple
            if self.config.kelly_risk_controller
            else self.config.hard_risk_multiple
        )
        cap = hard_multiple * state.reference_uwr / update_uwr
        return cap if math.isfinite(cap) else None

    def project_budget(
        self,
        label: str,
        nominal_budget: float,
        uwr: float,
        max_axis_uwr: float,
    ) -> float:
        """Project a nominal update into the block's risk region.

        Risky updates are always contracted toward the cap. When
        ``risk_expand_max > 1`` the unused risk headroom of a safe update is
        refunded as extra step size, up to that multiple of nominal and never
        above ``budget_max`` -- the risk-seeking half of the appetite spectrum.
        """
        cap = self.risk_budget_cap(label, uwr, max_axis_uwr)
        if cap is None:
            return nominal_budget
        if cap >= nominal_budget:
            if self.config.risk_expand_max <= 1.0:
                return nominal_budget
            expanded = min(
                self.config.risk_expand_max * nominal_budget,
                cap,
                self.config.budget_max,
            )
            return max(nominal_budget, expanded)
        return min(nominal_budget, max(0.0, cap))

    def effective_frozen_budget(self, label: str, base_budget: float) -> float:
        """Apply appetite to update scale without reopening geometry routing."""
        if (
            self.config.kelly_risk_controller or self.config.ds_risk_controller
        ) and not self.config.unit_budget:
            budget = min(
                self.config.budget_max,
                max(self.config.budget_min, base_budget * self.kelly_scale),
            )
        elif (
            self.config.unit_budget
            or not self.config.adaptive_risk_appetite
            or not self.config.risk_appetite_budget_only
        ):
            budget = base_budget
        else:
            state = self.states.get(label)
            appetite = state.risk_appetite if state is not None else 1.0
            budget = min(
                self.config.budget_max,
                max(self.config.budget_min, base_budget * appetite),
            )
        if self.config.erc_budget and not self.config.unit_budget:
            erc_min = (
                self.config.budget_min
                if self.config.erc_budget_min is None
                else self.config.erc_budget_min
            )
            erc_max = (
                self.config.budget_max
                if self.config.erc_budget_max is None
                else self.config.erc_budget_max
            )
            budget = min(erc_max, max(erc_min, budget * self.erc_scale(label)))
        return budget

    def initial_mode(self, descriptor: BlockDescriptor) -> Geometry:
        if self.config.fixed_geometry:
            return self.fixed_mode(descriptor)
        if self.config.coordinate_only or self.config.coordinate_start:
            return "coordinate"
        if descriptor.access_pattern in {"token_lookup", "token_output"}:
            return "row_column" if self.config.dynamic_token_geometry else "coordinate"
        modes = legal_geometries(descriptor)
        return max(modes, key=lambda mode: structure_prior(descriptor, mode))

    def fixed_mode(self, descriptor: BlockDescriptor) -> Geometry:
        if descriptor.access_pattern == "dense":
            return "spectral"
        if descriptor.access_pattern in {"token_lookup", "token_output"}:
            if self.config.fixed_token_coordinate:
                return "coordinate"
            return "row_column"
        return "coordinate"

    def selectable_geometries(
        self, descriptor: BlockDescriptor
    ) -> tuple[Geometry, ...]:
        legal = legal_geometries(descriptor)
        if self.config.fixed_geometry:
            return legal
        if self.config.coordinate_only:
            return ("coordinate",)
        if self.config.coordinate_start:
            return legal
        if descriptor.access_pattern in {"token_lookup", "token_output"}:
            if self.config.dynamic_token_geometry:
                return legal
            return ("coordinate",)
        return legal

    def initialize(self, descriptor: BlockDescriptor) -> BlockRouteState:
        state = self.states.get(descriptor.label)
        if state is None:
            modes = legal_geometries(descriptor)
            state = BlockRouteState(
                current_mode=self.initial_mode(descriptor),
                structure_class=descriptor.structure_class,
                ema_abs_return={mode: 0.0 for mode in modes},
                mode_counts={mode: 0 for mode in modes},
            )
            self.states[descriptor.label] = state
        return state

    @staticmethod
    def _observation_is_finite(observation: CandidateObservation) -> bool:
        values = (
            *observation.returns,
            observation.mean_return,
            observation.mean_abs_return,
            observation.cvar,
            observation.uwr,
            observation.max_axis_uwr,
            observation.spectral_risk,
            observation.drift,
            observation.gradient_sv_concentration,
            observation.momentum_sv_concentration,
        )
        return all(math.isfinite(value) for value in values)

    def _normalized_observations(
        self,
        state: BlockRouteState,
        observations: Mapping[Geometry, CandidateObservation],
        *,
        step: int,
    ) -> dict[Geometry, CandidateObservation]:
        """Rescale each candidate's UWR onto the coordinate scale.

        Spectral updates are intrinsically larger in Frobenius-norm-per-weight
        terms than coordinate updates (~3x in practice), and the risk threshold is
        a plain multiple of a reference anchored on the incumbent geometry. That
        makes the *same* candidate look hard-risk when the block happens to start
        from coordinate and safe when it starts from spectral. Dividing by the
        per-geometry collection-phase median ratio removes that frame dependence.
        """
        if not self.config.normalize_candidate_uwr:
            return dict(observations)
        coordinate = observations.get("coordinate")
        if coordinate is None:
            return dict(observations)
        if self._observation_is_finite(coordinate) and step < self.config.collect_steps:
            for mode, observation in observations.items():
                if mode == "coordinate" or not self._observation_is_finite(
                    observation
                ):
                    continue
                if observation.uwr <= 0 or coordinate.uwr <= 0:
                    continue
                state.uwr_scale_samples.setdefault(mode, []).append(
                    coordinate.uwr / observation.uwr
                )
        normalized: dict[Geometry, CandidateObservation] = {}
        for mode, observation in observations.items():
            scale = self._uwr_scale(state, mode)
            if scale == 1.0 or not self._observation_is_finite(observation):
                normalized[mode] = observation
                continue
            normalized[mode] = replace(
                observation,
                uwr=observation.uwr * scale,
                max_axis_uwr=observation.max_axis_uwr * scale,
            )
        return normalized

    @staticmethod
    def _uwr_scale(state: BlockRouteState, mode: Geometry) -> float:
        samples = state.uwr_scale_samples.get(mode)
        if samples:
            state.uwr_scale[mode] = statistics.median(samples)
        return state.uwr_scale.get(mode, 1.0)

    def _update_return_scales(
        self,
        state: BlockRouteState,
        observations: Mapping[Geometry, CandidateObservation],
    ) -> None:
        beta = self.config.ema_abs_return_beta
        for mode, observation in observations.items():
            if not self._observation_is_finite(observation):
                continue
            previous = state.ema_abs_return.get(mode, 0.0)
            value = max(observation.mean_abs_return, self.config.eps)
            state.ema_abs_return[mode] = (
                value
                if previous == 0.0 or not math.isfinite(previous)
                else beta * previous + (1 - beta) * value
            )

    @staticmethod
    def _spectral_signal(observation: CandidateObservation) -> float:
        concentration = max(
            observation.gradient_sv_concentration,
            observation.momentum_sv_concentration,
        )
        return observation.spectral_risk * (1.0 + concentration)

    def _set_reference(
        self, state: BlockRouteState, observation: CandidateObservation
    ) -> None:
        state.reference_cvar = max(observation.cvar, self.config.eps)
        state.reference_drift = max(observation.drift, self.config.eps)
        state.reference_uwr = max(
            observation.uwr, observation.max_axis_uwr, self.config.eps
        )
        state.reference_spectral = max(
            self._spectral_signal(observation), self.config.eps
        )
        state.reference_initialized = True

    def _update_references(
        self,
        step: int,
        state: BlockRouteState,
        observation: CandidateObservation,
        *,
        update_spectral: bool,
    ) -> None:
        if not state.reference_initialized:
            self._set_reference(state, observation)
        if step < self.config.collect_steps:
            beta = self.config.ema_abs_return_beta
            state.reference_cvar = beta * state.reference_cvar + (
                1 - beta
            ) * max(observation.cvar, self.config.eps)
            state.reference_drift = beta * state.reference_drift + (
                1 - beta
            ) * max(observation.drift, self.config.eps)
            state.reference_uwr = beta * state.reference_uwr + (
                1 - beta
            ) * max(observation.uwr, observation.max_axis_uwr, self.config.eps)
            if update_spectral:
                state.reference_spectral = beta * state.reference_spectral + (
                    1 - beta
                ) * max(self._spectral_signal(observation), self.config.eps)
            return
        state.reference_ready = True

    def _slow_reference_update(
        self,
        state: BlockRouteState,
        observation: CandidateObservation,
        *,
        update_spectral: bool,
    ) -> None:
        if not state.reference_ready:
            return
        beta = self.config.reference_beta
        state.reference_cvar = beta * state.reference_cvar + (
            1 - beta
        ) * max(observation.cvar, self.config.eps)
        state.reference_drift = beta * state.reference_drift + (
            1 - beta
        ) * max(observation.drift, self.config.eps)
        state.reference_uwr = beta * state.reference_uwr + (
            1 - beta
        ) * max(observation.uwr, observation.max_axis_uwr, self.config.eps)
        if update_spectral:
            state.reference_spectral = beta * state.reference_spectral + (
                1 - beta
            ) * max(self._spectral_signal(observation), self.config.eps)

    def _axis_amplification(
        self,
        mode: Geometry,
        observation: CandidateObservation,
        coordinate_observation: CandidateObservation,
    ) -> float:
        if mode == "coordinate":
            return 1.0
        return observation.max_axis_uwr / max(
            coordinate_observation.max_axis_uwr, self.config.eps
        )

    def _risk_ratios(
        self,
        state: BlockRouteState,
        descriptor: BlockDescriptor,
        mode: Geometry,
        observation: CandidateObservation,
        coordinate_observation: CandidateObservation,
    ) -> dict[str, float | None]:
        values = {
            "cvar": observation.cvar,
            "drift": observation.drift,
            "uwr": max(observation.uwr, observation.max_axis_uwr),
            "spectral": self._spectral_signal(observation),
        }
        references = {
            "cvar": state.reference_cvar,
            "drift": state.reference_drift,
            "uwr": state.reference_uwr,
            "spectral": state.reference_spectral,
        }
        ratios: dict[str, float | None] = {}
        for name, value in values.items():
            reference = references[name]
            if not math.isfinite(value) or not math.isfinite(reference):
                ratios[name] = None
                continue
            ratio = value / max(reference, self.config.eps)
            ratios[name] = ratio if math.isfinite(ratio) else None
        axis_amplification = self._axis_amplification(
            mode, observation, coordinate_observation
        )
        ratios["axis_amplification"] = (
            axis_amplification
            if math.isfinite(axis_amplification)
            else None
        )
        return ratios

    def _hard_risk_triggers(
        self,
        state: BlockRouteState,
        descriptor: BlockDescriptor,
        mode: Geometry,
        observation: CandidateObservation,
        coordinate_observation: CandidateObservation,
    ) -> tuple[str, ...]:
        if not self._observation_is_finite(observation):
            return ("nonfinite",)
        multiple = self.config.hard_risk_multiple
        ratios = self._risk_ratios(
            state,
            descriptor,
            mode,
            observation,
            coordinate_observation,
        )
        risk_names = ["cvar", "drift", "uwr", "spectral"]
        if descriptor.access_pattern in {"token_lookup", "token_output"}:
            risk_names.append("axis_amplification")
        return tuple(
            name
            for name in risk_names
            if ratios[name] is not None and ratios[name] > multiple
        )

    @staticmethod
    def _candidate_statistics(
        observation: CandidateObservation,
    ) -> dict[str, float | None]:
        values = {
            "mean_return": observation.mean_return,
            "cvar": observation.cvar,
            "uwr": observation.uwr,
            "max_axis_uwr": observation.max_axis_uwr,
            "spectral_risk": observation.spectral_risk,
            "drift": observation.drift,
            "gradient_sv_concentration": observation.gradient_sv_concentration,
            "momentum_sv_concentration": observation.momentum_sv_concentration,
        }
        return {
            name: value if math.isfinite(value) else None
            for name, value in values.items()
        }

    def _hard_violation(
        self,
        state: BlockRouteState,
        descriptor: BlockDescriptor,
        mode: Geometry,
        observation: CandidateObservation,
        coordinate_observation: CandidateObservation,
    ) -> bool:
        # Gate switching on mode-discriminating risks only. Block-level signals
        # such as drift are identical for every candidate, so treating them as a
        # hard violation would mark all candidates hard at once and trap the
        # block on its initial geometry. They still appear in the reported
        # triggers and still lower the candidate score.
        return bool(
            self._switch_blocking_triggers(
                state,
                descriptor,
                mode,
                observation,
                coordinate_observation,
            )
        )

    def _switch_blocking_triggers(
        self,
        state: BlockRouteState,
        descriptor: BlockDescriptor,
        mode: Geometry,
        observation: CandidateObservation,
        coordinate_observation: CandidateObservation,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._hard_risk_triggers(
                state,
                descriptor,
                mode,
                observation,
                coordinate_observation,
            )
            if name not in _NON_DISCRIMINATING_RISKS
        )

    def _score(
        self,
        descriptor: BlockDescriptor,
        mode: Geometry,
        observation: CandidateObservation,
        state: BlockRouteState,
        return_scale: float,
        coordinate_observation: CandidateObservation,
    ) -> float:
        scale = max(return_scale, self.config.eps)
        drift_excess = max(
            0.0,
            observation.drift / max(state.reference_drift, self.config.eps) - 1,
        )
        uwr_excess = max(
            0.0,
            max(observation.uwr, observation.max_axis_uwr)
            / max(state.reference_uwr, self.config.eps)
            - 1,
        )
        spectral_suitability = 0.0
        if descriptor.access_pattern == "dense" and mode == "spectral":
            spectral_suitability = (
                self.config.spectral_concentration_weight
                * max(
                    observation.gradient_sv_concentration,
                    observation.momentum_sv_concentration,
                )
            )
        axis_amplification = self._axis_amplification(
            mode, observation, coordinate_observation
        )
        axis_penalty = (
            0.25
            * max(
                0.0,
                math.log2(max(axis_amplification, self.config.eps)),
            )
            if descriptor.access_pattern in {"token_lookup", "token_output"}
            and math.isfinite(axis_amplification)
            else 0.0
        )
        appetite = (
            state.risk_appetite
            if self.config.adaptive_risk_appetite
            else 1.0
        )
        risk_weight = (
            1.0
            if self.config.risk_appetite_budget_only
            else 1.0 / max(appetite, self.config.eps)
        )
        return (
            observation.mean_return / scale
            - risk_weight * observation.cvar / scale
            - 0.5 * risk_weight * drift_excess
            - 0.25 * risk_weight * uwr_excess
            + structure_prior(descriptor, mode)
            + spectral_suitability
            - axis_penalty
        )

    def _risk_capacity(
        self, state: BlockRouteState, observation: CandidateObservation
    ) -> float:
        eps = self.config.eps
        ratios = (
            math.sqrt(state.reference_cvar / (observation.cvar + eps)),
            math.sqrt(state.reference_drift / (observation.drift + eps)),
            math.sqrt(
                state.reference_uwr
                / (max(observation.uwr, observation.max_axis_uwr) + eps)
            ),
        )
        return min(self.config.budget_max, max(self.config.budget_min, min(ratios)))

    def _expansion_eligible(self, structure_class: str) -> bool:
        structure_error = self.structure_calibration_error.get(structure_class, 0.0)
        structure_abs = self.structure_calibration_abs.get(structure_class, 0.0)
        return (
            self.global_calibration_count >= 3
            and self.structure_calibration_count.get(structure_class, 0) >= 3
            and self.global_calibration_abs > self.config.eps
            and structure_abs > self.config.eps
            and self.global_calibration_error
            >= 0.5 * self.global_calibration_abs
            and structure_error >= 0.5 * structure_abs
        )

    def _budget_statistics(
        self,
        state: BlockRouteState,
        observation: CandidateObservation,
        structure_class: str,
        *,
        collection_locked: bool,
        unit_budget: bool,
    ) -> dict[str, float | bool]:
        quality = min(
            1.0,
            max(
                0.0,
                (observation.mean_return - observation.cvar)
                / (observation.mean_abs_return + self.config.eps),
            ),
        )
        risk_capacity = self._risk_capacity(state, observation)
        expansion_eligible = self._expansion_eligible(structure_class)
        if quality < 0.9:
            return_cap = 0.5 + 0.5 * quality
        elif expansion_eligible:
            return_cap = self.config.budget_max
        else:
            return_cap = 1.0
        appetite = (
            state.risk_appetite
            if self.config.adaptive_risk_appetite
            else 1.0
        )
        if self.config.risk_appetite_budget_only:
            appetite = 1.0
        target = (
            1.0
            if collection_locked or unit_budget
            else min(
                self.config.budget_max,
                max(self.config.budget_min, min(risk_capacity, return_cap) * appetite),
            )
        )
        result = {
            "quality": quality,
            "risk_capacity": risk_capacity,
            "return_cap": return_cap,
            "target": target,
            "expansion_eligible": expansion_eligible,
            "collection_locked": collection_locked,
        }
        if self.config.adaptive_risk_appetite:
            result["risk_appetite"] = appetite
        return result

    def is_route_event(self, step: int, descriptor: BlockDescriptor) -> bool:
        if (
            self.config.freeze_stable_routes
            and self.is_frozen(descriptor.label)
        ):
            return False
        phase = 1 if descriptor.update_frequency == "odd_step" else 0
        return step % self.config.route_interval == phase

    def _update_budget(
        self,
        state: BlockRouteState,
        target: float,
    ) -> None:
        old = state.budget_scale
        smoothed = self.config.budget_beta * old + (
            1 - self.config.budget_beta
        ) * target
        change = self.config.max_budget_event_change
        changed = min(
            old * (1 + change), max(old * (1 - change), smoothed)
        )
        state.budget_scale = min(
            self.config.budget_max,
            max(self.config.budget_min, changed),
        )

    def route(
        self,
        step: int,
        descriptors: Sequence[BlockDescriptor],
        observations: Mapping[
            str, Mapping[Geometry, CandidateObservation]
        ],
    ) -> dict[str, RoutingDecision]:
        decisions: dict[str, RoutingDecision] = {}
        for descriptor in descriptors:
            state = self.initialize(descriptor)
            block_observations = observations[descriptor.label]
            legal = legal_geometries(descriptor)
            if set(block_observations) != set(legal):
                raise ValueError(
                    f"{descriptor.label} observations must match legal geometries {legal}"
                )
            if state.frozen:
                decisions[descriptor.label] = RoutingDecision(
                    label=descriptor.label,
                    mode=state.current_mode,
                    budget_scale=state.budget_scale,
                    switched=False,
                    route_event=False,
                    reason="frozen",
                    scores={},
                    hard_violations=(),
                    risk_ratios={},
                    hard_risk_triggers={},
                    candidate_statistics={},
                    budget_statistics={},
                )
                continue
            if self.config.unit_budget or step < self.config.collect_steps:
                state.budget_scale = 1.0
            if self.config.fixed_geometry:
                state.current_mode = self.fixed_mode(descriptor)
            elif self.config.coordinate_only:
                state.current_mode = "coordinate"
                state.challenger_mode = None
                state.challenger_wins = 0
            elif self.config.coordinate_start:
                pass
            elif (
                descriptor.access_pattern in {"token_lookup", "token_output"}
                and not self.config.dynamic_token_geometry
            ):
                state.current_mode = "coordinate"
                state.challenger_mode = None
                state.challenger_wins = 0
            selectable = self.selectable_geometries(descriptor)
            finite_modes = tuple(
                mode
                for mode in legal
                if self._observation_is_finite(block_observations[mode])
            )
            if not finite_modes:
                raise FloatingPointError(
                    f"{descriptor.label} has no finite FARO candidate"
                )
            finite_selectable = tuple(
                mode for mode in selectable if mode in finite_modes
            )
            if not finite_selectable:
                raise FloatingPointError(
                    f"{descriptor.label} has no finite selectable FARO candidate"
                )
            self._update_return_scales(state, block_observations)
            block_observations = self._normalized_observations(
                state, block_observations, step=step
            )
            current_observation = block_observations[state.current_mode]
            coordinate_observation = block_observations["coordinate"]
            reference_observation = (
                current_observation
                if state.current_mode in finite_modes
                else block_observations[
                    "coordinate"
                    if "coordinate" in finite_modes
                    else finite_modes[0]
                ]
            )
            was_ready = state.reference_ready
            route_event = self.is_route_event(step, descriptor)
            self._update_references(
                step,
                state,
                reference_observation,
                update_spectral=route_event,
            )
            finite_return_scales = [
                scale
                for mode in finite_modes
                if math.isfinite(
                    scale := state.ema_abs_return.get(mode, 0.0)
                )
            ]
            return_scale = max(
                finite_return_scales, default=self.config.eps
            )
            scores = {
                mode: (
                    self._score(
                        descriptor,
                        mode,
                        observation,
                        state,
                        return_scale,
                        coordinate_observation,
                    )
                    if mode in finite_modes
                    else _INVALID_SCORE
                )
                for mode, observation in block_observations.items()
            }
            hard = tuple(
                mode
                for mode, observation in block_observations.items()
                if self._hard_violation(
                    state,
                    descriptor,
                    mode,
                    observation,
                    coordinate_observation,
                )
            )
            selectable_hard = tuple(
                mode for mode in selectable if mode in hard
            )
            risk_ratios = {
                mode: self._risk_ratios(
                    state,
                    descriptor,
                    mode,
                    observation,
                    coordinate_observation,
                )
                for mode, observation in block_observations.items()
            }
            hard_risk_triggers = {
                mode: self._hard_risk_triggers(
                    state,
                    descriptor,
                    mode,
                    observation,
                    coordinate_observation,
                )
                for mode, observation in block_observations.items()
            }
            candidate_statistics = {
                mode: self._candidate_statistics(observation)
                for mode, observation in block_observations.items()
            }
            switched = False
            reason = "incumbent_best"

            if state.current_mode not in finite_selectable:
                allowed = [
                    mode for mode in finite_selectable if mode not in selectable_hard
                ]
                fallback = allowed or list(finite_selectable)
                best = (
                    "spectral"
                    if descriptor.access_pattern == "dense"
                    and "spectral" in allowed
                    else max(fallback, key=scores.__getitem__)
                )
                state.current_mode = best
                state.last_switch_step = step
                state.challenger_mode = None
                state.challenger_wins = 0
                switched = True
                reason = "nonfinite_evacuation"
            elif not route_event:
                reason = "not_route_event"
            elif self.config.fixed_geometry:
                state.challenger_mode = None
                state.challenger_wins = 0
                reason = "fixed_geometry"
            elif step < self.config.collect_steps:
                reason = "collecting_statistics"
            elif len(selectable_hard) == len(selectable):
                state.challenger_mode = None
                state.challenger_wins = 0
                reason = "all_candidates_hard_risk"
            elif state.current_mode in selectable_hard:
                allowed = [
                    mode for mode in selectable if mode not in selectable_hard
                ]
                best = (
                    "spectral"
                    if descriptor.access_pattern == "dense"
                    and "spectral" in allowed
                    else max(allowed, key=scores.__getitem__)
                )
                state.current_mode = best
                state.last_switch_step = step
                state.challenger_mode = None
                state.challenger_wins = 0
                switched = True
                reason = "hard_risk_evacuation"
            else:
                allowed = [
                    mode for mode in selectable if mode not in selectable_hard
                ]
                best = max(allowed, key=scores.__getitem__)
                incumbent_score = scores[state.current_mode]
                residence_met = (
                    step - state.last_switch_step
                    >= self.config.min_residence_steps
                )
                if (
                    best != state.current_mode
                    and residence_met
                    and scores[best]
                    >= incumbent_score + self.config.challenger_margin
                ):
                    if state.challenger_mode == best:
                        state.challenger_wins += 1
                    else:
                        state.challenger_mode = best
                        state.challenger_wins = 1
                    if state.challenger_wins >= self.config.challenger_events:
                        state.current_mode = best
                        state.last_switch_step = step
                        state.challenger_mode = None
                        state.challenger_wins = 0
                        switched = True
                        reason = "challenger_won_twice"
                    else:
                        reason = "challenger_pending"
                else:
                    state.challenger_mode = None
                    state.challenger_wins = 0
                    reason = (
                        "minimum_residence"
                        if best != state.current_mode and not residence_met
                        else "incumbent_best"
                    )

            selected_observation = block_observations[state.current_mode]
            collection_locked = step < self.config.collect_steps
            budget_statistics = self._budget_statistics(
                state,
                selected_observation,
                descriptor.structure_class,
                collection_locked=collection_locked,
                unit_budget=self.config.unit_budget,
            )
            # Local reward sizing is intentionally deferred until every route is
            # frozen; changing active budgets would change later geometry evidence.
            if route_event:
                if self.config.unit_budget or step < self.config.collect_steps:
                    state.budget_scale = 1.0
                else:
                    self._update_budget(
                        state, float(budget_statistics["target"])
                    )
                state.mode_counts[state.current_mode] += 1
            if (
                self.config.freeze_stable_routes
                and step >= self.config.collect_steps
                and not (
                    self.config.adaptive_risk_appetite
                    and not self.config.risk_appetite_budget_only
                    and self.global_calibration_count < 3
                )
            ):
                if state.stable_since_step < self.config.collect_steps:
                    state.stable_since_step = self.config.collect_steps
                # A block that switched, or is still weighing a challenger, is not
                # settled: restart the stability clock so a pending decision is
                # never frozen away before its confirmation vote can land.
                if switched or state.challenger_mode is not None:
                    state.stable_since_step = step
                if (
                    route_event
                    and state.challenger_mode is None
                    and step >= self.config.collect_steps + self.config.min_residence_steps
                    and step - state.stable_since_step
                    >= self.config.min_residence_steps
                ):
                    state.frozen = True
                    state.frozen_step = step
            if was_ready:
                self._slow_reference_update(
                    state,
                    selected_observation,
                    update_spectral=route_event,
                )
            decisions[descriptor.label] = RoutingDecision(
                label=descriptor.label,
                mode=state.current_mode,
                budget_scale=state.budget_scale,
                switched=switched,
                route_event=route_event,
                reason=reason,
                scores=scores,
                hard_violations=hard,
                risk_ratios=risk_ratios,
                hard_risk_triggers=hard_risk_triggers,
                candidate_statistics=candidate_statistics,
                budget_statistics=budget_statistics,
            )

        if not self.config.unit_budget and step >= self.config.collect_steps:
            for label, decision in decisions.items():
                if self.is_frozen(label):
                    continue
                structure_class = self.states[label].structure_class
                peer_budgets = [
                    state.budget_scale
                    for state in self.states.values()
                    if state.structure_class == structure_class
                ]
                cap = 2.0 * statistics.median(peer_budgets)
                if decision.budget_scale > cap:
                    capped = min(
                        self.config.budget_max,
                        max(self.config.budget_min, cap),
                    )
                    decision.budget_scale = capped
                    self.states[label].budget_scale = capped
        return decisions

    def calibrate(
        self,
        realized_return: float,
        predicted_return: float,
        structure_returns: Mapping[str, tuple[float, float]],
    ) -> None:
        values = [realized_return, predicted_return]
        values.extend(
            value
            for pair in structure_returns.values()
            for value in pair
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("nonfinite FARO probe calibration")
        beta = 0.9
        error = realized_return - predicted_return
        self.global_calibration_error = (
            beta * self.global_calibration_error + (1 - beta) * error
        )
        self.global_calibration_abs = (
            beta * self.global_calibration_abs + (1 - beta) * abs(error)
        )
        self.global_calibration_count += 1
        for structure_class, (realized, predicted) in structure_returns.items():
            structure_error = realized - predicted
            self.structure_calibration_error[structure_class] = beta * self.structure_calibration_error.get(
                structure_class, 0.0
            ) + (1 - beta) * structure_error
            self.structure_calibration_abs[structure_class] = beta * self.structure_calibration_abs.get(
                structure_class, 0.0
            ) + (1 - beta) * abs(structure_error)
            self.structure_calibration_count[structure_class] = (
                self.structure_calibration_count.get(structure_class, 0) + 1
            )

    def update_risk_appetite(
        self,
        structure_signals: Mapping[str, tuple[float, float, float]],
    ) -> None:
        """Update appetite from *relative* probe calibration and risk pressure.

        Predicted return is estimated in-sample while realized return comes from a
        disjoint probe slice, so realized/predicted sits far below 1 and that level
        is structural, not a signal. Appetite therefore tracks deviation of the
        current ratio from its own running baseline: better-than-usual conversion
        raises appetite, worse-than-usual lowers it.
        """
        if not self.config.adaptive_risk_appetite:
            return
        for state in self.states.values():
            structure = state.structure_class
            if structure not in structure_signals:
                continue
            realized, predicted, risk_pressure = structure_signals[structure]
            if not all(math.isfinite(x) for x in (realized, predicted, risk_pressure)):
                raise FloatingPointError("nonfinite FARO risk appetite signal")
            if predicted <= 0.0:
                state.risk_appetite_steps += 1
                continue
            ratio = max(0.0, realized) / predicted
            baseline = self.structure_calibration_ratio.get(structure)
            if baseline is None or baseline <= self.config.eps:
                # Establish the level first; it is not yet usable as a reference.
                self.structure_calibration_ratio[structure] = ratio
                state.risk_appetite_steps += 1
                continue
            self.structure_calibration_ratio[structure] = (
                self.config.reference_beta * baseline
                + (1.0 - self.config.reference_beta) * ratio
            )
            if state.risk_appetite_steps < 3:
                state.risk_appetite_steps += 1
                continue
            relative = ratio / max(baseline, self.config.eps) - 1.0
            calibration = max(-1.0, min(1.0, relative))
            target = 1.0 + 0.50 * calibration - 0.25 * max(0.0, risk_pressure)
            target = min(
                self.config.risk_appetite_max,
                max(self.config.risk_appetite_min, target),
            )
            beta = self.config.risk_appetite_beta
            previous = state.risk_appetite
            updated = beta * previous + (1.0 - beta) * target
            delta = max(-0.05, min(0.05, updated - previous))
            state.risk_appetite = min(
                self.config.risk_appetite_max,
                max(self.config.risk_appetite_min, previous + delta),
            )
            state.risk_appetite_steps += 1

    def update_global_risk_appetite(
        self, realized_return: float, risk_pressure: float
    ) -> None:
        """Adapt appetite from held-out outcome and robust risk pressure."""
        if not self.config.adaptive_risk_appetite:
            return
        if not all(math.isfinite(float(x)) for x in (realized_return, risk_pressure)):
            raise FloatingPointError("nonfinite global appetite signal")
        beta = self.config.reference_beta
        magnitude = abs(realized_return)
        if self.appetite_return_scale <= self.config.eps:
            self.appetite_return_scale = max(magnitude, self.config.eps)
        else:
            self.appetite_return_scale = beta * self.appetite_return_scale + (
                1.0 - beta
            ) * max(magnitude, self.config.eps)
        normalized_outcome = realized_return / max(
            self.appetite_return_scale, self.config.eps
        )
        # Small held-out returns are dominated by probe noise once training has
        # progressed. Treat them as neutral instead of steadily shrinking LR.
        if abs(normalized_outcome) < 0.25:
            normalized_outcome = 0.0
        normalized_outcome = max(-1.0, min(1.0, normalized_outcome))
        pressure_excess = max(
            0.0,
            (risk_pressure - self.config.hard_risk_multiple)
            / max(self.config.hard_risk_multiple, self.config.eps),
        )
        target = 1.0 + 0.25 * normalized_outcome - 0.10 * pressure_excess
        target = min(self.config.risk_appetite_max, max(self.config.risk_appetite_min, target))
        for state in self.states.values():
            if state.risk_appetite_steps < 3:
                state.risk_appetite_steps += 1
                continue
            previous = state.risk_appetite
            updated = self.config.risk_appetite_beta * previous + (1.0 - self.config.risk_appetite_beta) * target
            delta = max(-0.05, min(0.05, updated - previous))
            state.risk_appetite = min(self.config.risk_appetite_max, max(self.config.risk_appetite_min, previous + delta))
            state.risk_appetite_steps += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "states": {label: asdict(state) for label, state in self.states.items()},
            "global_calibration_error": self.global_calibration_error,
            "global_calibration_abs": self.global_calibration_abs,
            "global_calibration_count": self.global_calibration_count,
            "structure_calibration_error": dict(self.structure_calibration_error),
            "structure_calibration_abs": dict(self.structure_calibration_abs),
            "structure_calibration_count": dict(
                self.structure_calibration_count
            ),
            "structure_calibration_ratio": dict(
                self.structure_calibration_ratio
            ),
            "appetite_return_scale": self.appetite_return_scale,
            "kelly_state": self.kelly_state,
            "kelly_scale": self.kelly_scale,
            "kelly_steps_remaining": self.kelly_steps_remaining,
            "kelly_observations": self.kelly_observations,
            "kelly_return_mean": self.kelly_return_mean,
            "kelly_return_variance": self.kelly_return_variance,
            "kelly_probe_loss_ema": self.kelly_probe_loss_ema,
            "kelly_projection_multiple": self.kelly_projection_multiple,
            "kelly_edge": self.kelly_edge,
            "kelly_confidence": self.kelly_confidence,
            "kelly_log_scale": self.kelly_log_scale,
            "ds_recent": list(self.ds_recent),
            "ds_baseline": self.ds_baseline,
            "ds_scale": self.ds_scale,
            "erc_risk": dict(self.erc_risk),
            "erc_multiplier": dict(self.erc_multiplier),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        raw_states = payload["states"]
        if not isinstance(raw_states, Mapping):
            raise TypeError("router states must be a mapping")
        self.states = {
            str(label): BlockRouteState(**dict(raw_state))
            for label, raw_state in raw_states.items()
        }
        self.global_calibration_error = float(payload["global_calibration_error"])
        self.global_calibration_abs = float(payload["global_calibration_abs"])
        self.global_calibration_count = int(
            payload.get("global_calibration_count", 0)
        )
        self.structure_calibration_error = dict(
            payload["structure_calibration_error"]  # type: ignore[arg-type]
        )
        self.structure_calibration_abs = dict(
            payload["structure_calibration_abs"]  # type: ignore[arg-type]
        )
        self.structure_calibration_count = {
            str(key): int(value)
            for key, value in dict(
                payload.get("structure_calibration_count", {})  # type: ignore[arg-type]
            ).items()
        }
        self.structure_calibration_ratio = {
            str(key): float(value)
            for key, value in dict(
                payload.get("structure_calibration_ratio", {})  # type: ignore[arg-type]
            ).items()
        }
        self.appetite_return_scale = float(payload.get("appetite_return_scale", 0.0))
        self.kelly_state = str(payload.get("kelly_state", "neutral"))
        self.kelly_scale = float(payload.get("kelly_scale", 1.0))
        self.kelly_steps_remaining = int(payload.get("kelly_steps_remaining", 0))
        self.kelly_observations = int(payload.get("kelly_observations", 0))
        self.kelly_return_mean = float(payload.get("kelly_return_mean", 0.0))
        self.kelly_return_variance = float(payload.get("kelly_return_variance", 0.0))
        self.kelly_probe_loss_ema = float(payload.get("kelly_probe_loss_ema", 0.0))
        self.kelly_projection_multiple = float(
            payload.get("kelly_projection_multiple", self.config.hard_risk_multiple)
        )
        self.kelly_edge = float(payload.get("kelly_edge", 0.0))
        self.kelly_confidence = float(payload.get("kelly_confidence", 0.0))
        self.kelly_log_scale = float(payload.get("kelly_log_scale", 0.0))
        self.ds_recent = [float(v) for v in payload.get("ds_recent", [])]
        self.ds_baseline = float(payload.get("ds_baseline", 0.0))
        self.ds_scale = float(payload.get("ds_scale", 0.0))
        self.erc_risk = {
            str(label): float(value)
            for label, value in dict(payload.get("erc_risk", {})).items()  # type: ignore[arg-type]
        }
        self.erc_multiplier = {
            str(label): float(value)
            for label, value in dict(payload.get("erc_multiplier", {})).items()  # type: ignore[arg-type]
        }
