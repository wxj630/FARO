from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from .descriptors import (
    BlockDescriptor,
    Geometry,
    legal_geometries,
    select_block,
)
from .geometry import (
    CandidateState,
    CandidateUpdate,
    CoordinateState,
    GeometryConfig,
    RowColumnState,
    SpectralState,
    advance_row_column_state,
    coordinate_candidate,
    row_column_candidate,
    row_column_candidate_from_state,
    spectral_candidate,
    transpose_candidate_state,
    _candidate_metrics,
)
from .normuon_backend import (
    NativeAdamState,
    NorMuonState,
    advance_native_adam_state,
    apply_cautious_precise_,
    native_adam_delta,
    native_adam_delta_from_state,
    normuon_direction,
    normuon_lr_multiplier,
)
from .router import CandidateObservation, FARORouter, RoutingDecision


UpdateFrequency = Literal["every_step", "odd_step"]
FAROCandidateState = CandidateState | NativeAdamState | NorMuonState

# Giant per-token lookup tables whose gradient touches only the rows that the
# batch actually indexes. Storing their micro-batch deltas row-sparsely keeps the
# per-step return proxy exact while dropping the dense zero rows.
_SPARSE_MICRO_LABELS = frozenset({"bigram_embed", "value_embeds"})


@dataclass(slots=True)
class MicroBatchSet:
    gradients: tuple[Tensor, ...]
    effective_tokens: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class SparseMicroRows:
    """Changed rows of one micro-batch gradient for a row-sparse lookup table."""

    rows: Tensor
    values: Tensor


@dataclass(slots=True, frozen=True)
class SparseMicroSet:
    micros: tuple[SparseMicroRows, ...]
    effective_tokens: tuple[int, ...]


def densify_sparse_micro_rows(micro: SparseMicroRows, *, size: int) -> Tensor:
    dense = torch.zeros(
        (size, micro.values.shape[1]),
        dtype=micro.values.dtype,
        device=micro.values.device,
    )
    dense.index_copy_(0, micro.rows, micro.values)
    return dense


def sparse_micro_return(
    micro: SparseMicroRows, delta: Tensor, *, tokens: int, micro_count: int
) -> Tensor:
    return (
        -(micro.values.float() * delta.index_select(0, micro.rows).float()).sum()
        / tokens
        * micro_count
    )


def _merge_sparse_micro_rows(
    first: SparseMicroRows, second: SparseMicroRows
) -> SparseMicroRows:
    rows = torch.cat([first.rows, second.rows])
    values = torch.cat([first.values.float(), second.values.float()])
    unique, inverse = torch.unique(rows, return_inverse=True)
    merged = torch.zeros(
        (unique.numel(), values.shape[1]), dtype=torch.float32, device=values.device
    )
    merged.index_add_(0, inverse, values)
    return SparseMicroRows(rows=unique, values=merged.to(torch.bfloat16))


def _merge_sparse_micro_sets(
    first: SparseMicroSet, second: SparseMicroSet
) -> SparseMicroSet:
    return SparseMicroSet(
        micros=tuple(
            _merge_sparse_micro_rows(one, two)
            for one, two in zip(first.micros, second.micros, strict=True)
        ),
        effective_tokens=tuple(
            one + two
            for one, two in zip(
                first.effective_tokens, second.effective_tokens, strict=True
            )
        ),
    )


def _micro_blocks(
    micro_set: MicroBatchSet | SparseMicroSet | None,
    descriptor: BlockDescriptor,
    parameter: Tensor,
) -> tuple[Tensor | SparseMicroRows, ...]:
    if micro_set is None:
        return ()
    if isinstance(micro_set, SparseMicroSet):
        return micro_set.micros
    return tuple(
        select_block(micro, descriptor) if micro.shape == parameter.shape else micro
        for micro in micro_set.gradients
    )


def _micro_return(
    micro: Tensor | SparseMicroRows,
    delta: Tensor,
    *,
    tokens: int,
    micro_count: int,
) -> Tensor:
    if isinstance(micro, SparseMicroRows):
        return sparse_micro_return(
            micro, delta, tokens=tokens, micro_count=micro_count
        )
    return -(micro.float() * delta).sum() / tokens * micro_count


class MicroGradientCollector:
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        micro_count: int = 8,
        sparse_labels: Iterable[str] = (),
    ):
        if micro_count < 1:
            raise ValueError("micro_count must be positive")
        self.micro_count = micro_count
        self.sparse_labels = frozenset(sparse_labels)
        self.parameters: dict[str, nn.Parameter] = {}
        for name, parameter in named_parameters:
            label = getattr(parameter, "label", name.removesuffix(".weight"))
            if label in self.parameters:
                raise ValueError(f"duplicate parameter label: {label}")
            self.parameters[label] = parameter
        self._previous: dict[str, Tensor] = {}
        self._current: dict[str, list[Tensor | SparseMicroRows]] = {}
        self._tokens: list[int] = []
        self._pending: dict[str, MicroBatchSet | SparseMicroSet] = {}
        self._prepared: dict[str, MicroBatchSet | SparseMicroSet | None] = {}

    def start_step(self, active_labels: Iterable[str] | None = None) -> None:
        if self._current or self._prepared:
            raise RuntimeError("finish the preceding collector step first")
        active = (
            set(self.parameters)
            if active_labels is None
            else set(active_labels)
        )
        unknown = active - self.parameters.keys()
        if unknown:
            raise KeyError(f"unknown collector labels: {sorted(unknown)}")
        self._tokens = []
        self._previous = {
            label: (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for label, parameter in self.parameters.items()
            if label in active
        }
        self._current = {label: [] for label in active}

    @torch.no_grad()
    def capture(self, micro_index: int, *, effective_tokens: int) -> None:
        if micro_index != len(self._tokens) or micro_index >= self.micro_count:
            raise ValueError("micro indices must be captured once in ascending order")
        if effective_tokens < 1:
            raise ValueError("effective_tokens must be positive")
        for label in self._current:
            parameter = self.parameters[label]
            previous = self._previous[label]
            current = (
                parameter.grad.detach()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            if label in self.sparse_labels:
                delta = (current - previous).to(torch.bfloat16)
                rows = torch.nonzero(delta.any(dim=1)).squeeze(1)
                self._current[label].append(
                    SparseMicroRows(rows=rows, values=delta.index_select(0, rows))
                )
            else:
                self._current[label].append(
                    (current - previous).to(torch.bfloat16)
                )
            self._previous[label] = current.clone()
        self._tokens.append(int(effective_tokens))

    def prepare(
        self,
        update_frequencies: Mapping[str, UpdateFrequency],
        *,
        step: int,
    ) -> dict[str, MicroBatchSet | SparseMicroSet | None]:
        if len(self._tokens) != self.micro_count:
            raise RuntimeError(
                f"expected {self.micro_count} captures, got {len(self._tokens)}"
            )
        prepared: dict[str, MicroBatchSet | SparseMicroSet | None] = {}
        for label, frequency in update_frequencies.items():
            if label not in self._current:
                prepared[label] = None
                continue
            tokens = tuple(self._tokens)
            if label in self.sparse_labels:
                current: MicroBatchSet | SparseMicroSet = SparseMicroSet(
                    micros=tuple(self._current[label]),
                    effective_tokens=tokens,
                )
            else:
                current = MicroBatchSet(
                    gradients=tuple(self._current[label]),
                    effective_tokens=tokens,
                )
            if frequency == "odd_step" and step % 2 == 0:
                self._pending[label] = current
                prepared[label] = None
                continue
            if frequency == "odd_step" and label in self._pending:
                pending = self._pending.pop(label)
                if isinstance(current, SparseMicroSet):
                    if not isinstance(pending, SparseMicroSet):
                        raise TypeError("sparse micro set merged with dense pending set")
                    prepared[label] = _merge_sparse_micro_sets(pending, current)
                    continue
                if isinstance(pending, SparseMicroSet):
                    raise TypeError("dense micro set merged with sparse pending set")
                gradients = tuple(
                    (first.float() + second.float()).to(torch.bfloat16)
                    for first, second in zip(
                        pending.gradients, current.gradients, strict=True
                    )
                )
                tokens = tuple(
                    first + second
                    for first, second in zip(
                        pending.effective_tokens,
                        current.effective_tokens,
                        strict=True,
                    )
                )
                prepared[label] = MicroBatchSet(gradients, tokens)
            else:
                prepared[label] = current
        self._prepared = prepared
        return prepared

    def finish_step(self) -> None:
        self._previous.clear()
        self._current.clear()
        self._tokens.clear()
        self._prepared.clear()

    @property
    def is_empty(self) -> bool:
        return not (
            self._previous
            or self._current
            or self._tokens
            or self._prepared
            or self._pending
        )


@dataclass(slots=True)
class OptimizerStepResult:
    modes: dict[str, Geometry]
    decisions: dict[str, RoutingDecision]
    predicted_return: float
    predicted_return_by_structure: dict[str, float]


@dataclass(slots=True)
class _CounterfactualPlan:
    parameter_label: str
    descriptor: BlockDescriptor
    mode: Geometry
    pre_block: Tensor
    base_delta: Tensor | None
    pre_mantissa: Tensor | None
    direction: Tensor | None
    config: GeometryConfig
    base_budget: float
    update_uwr: float
    projection_cap: float | None
    applied_scale: float


def _named_parameter_map(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> tuple[list[tuple[str, nn.Parameter]], dict[str, nn.Parameter]]:
    materialized = list(named_parameters)
    by_label: dict[str, nn.Parameter] = {}
    for name, parameter in materialized:
        label = getattr(parameter, "label", name.removesuffix(".weight"))
        if label in by_label:
            raise ValueError(f"duplicate parameter label: {label}")
        by_label[label] = parameter
    return materialized, by_label


def _coordinate_config(
    defaults: Mapping[str, object],
    descriptor: BlockDescriptor,
    lr_scale: float,
    *,
    measure_spectral_risk: bool,
) -> GeometryConfig:
    raw_overrides = defaults.get("per_parameter", {})
    overrides = (
        raw_overrides.get(descriptor.parameter_label, {})
        if isinstance(raw_overrides, Mapping)
        else {}
    )
    if not isinstance(overrides, Mapping):
        raise TypeError("per-parameter coordinate defaults must be mappings")
    base_lr_key = "matrix_lr" if descriptor.access_pattern == "dense" else "lr"
    base_lr = float(overrides.get("lr", defaults.get(base_lr_key, defaults["lr"])))
    lr_multiplier = float(overrides.get("lr_multiplier", 1.0))
    weight_decay = float(
        overrides.get("weight_decay", defaults.get("weight_decay", 0.0))
    )
    return GeometryConfig(
        lr=base_lr * lr_multiplier * lr_scale,
        beta1=float(overrides.get("beta1", defaults.get("beta1", 0.9))),
        beta2=float(overrides.get("beta2", defaults.get("beta2", 0.999))),
        eps=float(overrides.get("eps", defaults.get("eps", 1e-8))),
        weight_decay=weight_decay,
        measure_spectral_risk=measure_spectral_risk,
    )


def _muon_config(
    defaults: Mapping[str, object],
    descriptor: BlockDescriptor,
    lr_scale: float,
    momentum: float,
    *,
    measure_spectral_risk: bool,
) -> GeometryConfig:
    projection_multiplier = (
        2.0
        if descriptor.structure_class == "mlp" and descriptor.storage_transposed
        else 1.0
    )
    return GeometryConfig(
        lr=float(defaults["lr"]) * projection_multiplier * lr_scale,
        momentum=momentum,
        weight_decay=float(defaults.get("weight_decay", 0.0)),
        measure_spectral_risk=measure_spectral_risk,
    )


def _normuon_config(
    defaults: Mapping[str, object],
    descriptor: BlockDescriptor,
    lr_scale: float,
    momentum: float,
    *,
    measure_spectral_risk: bool,
) -> GeometryConfig:
    base_lr = float(defaults["lr"]) * lr_scale
    multiplier = normuon_lr_multiplier(
        descriptor.shape,
        mlp_down=(
            descriptor.structure_class == "mlp"
            and descriptor.storage_transposed
        ),
    )
    return GeometryConfig(
        lr=base_lr * multiplier,
        beta2=float(defaults.get("beta2", 0.9)),
        momentum=momentum,
        weight_decay=float(defaults.get("weight_decay", 0.0)) * base_lr,
        measure_spectral_risk=measure_spectral_risk,
    )


def _row_axis(descriptor: BlockDescriptor) -> int:
    if descriptor.token_axis is not None:
        return descriptor.token_axis
    if descriptor.output_axis is None:
        raise ValueError("matrix descriptor has no functional output axis")
    return descriptor.output_axis


def _initialize_state(
    parameter: Tensor, descriptor: BlockDescriptor, mode: Geometry
) -> CandidateState:
    block = select_block(parameter, descriptor)
    if mode == "coordinate":
        return CoordinateState.zeros_like(block)
    if mode == "row_column":
        return RowColumnState.zeros_like(block, _row_axis(descriptor))
    return SpectralState.zeros_like(block)


def _initialize_faro_state(
    parameter: Tensor, descriptor: BlockDescriptor, mode: Geometry
) -> FAROCandidateState:
    block = select_block(parameter, descriptor)
    if mode == "coordinate":
        return NativeAdamState.zeros_like(block)
    if mode == "row_column":
        return RowColumnState.zeros_like(block, _row_axis(descriptor))
    return NorMuonState.zeros_like(block)


def _transpose_faro_state(state: FAROCandidateState) -> FAROCandidateState:
    if isinstance(state, NativeAdamState):
        return NativeAdamState(
            step=state.step,
            exp_avg=state.exp_avg.T.clone(),
            exp_avg_sq=state.exp_avg_sq.T.clone(),
        )
    if isinstance(state, NorMuonState):
        return NorMuonState(
            momentum=state.momentum.T.clone(),
            second_momentum=state.second_momentum.T.clone(),
            mantissa=state.mantissa.T.clone(),
        )
    return transpose_candidate_state(state)


def _clone_faro_state(state: FAROCandidateState) -> FAROCandidateState:
    if isinstance(state, NativeAdamState):
        return NativeAdamState(
            step=state.step,
            exp_avg=state.exp_avg.clone(),
            exp_avg_sq=state.exp_avg_sq.clone(),
        )
    if isinstance(state, NorMuonState):
        return NorMuonState(
            momentum=state.momentum.clone(),
            second_momentum=state.second_momentum.clone(),
            mantissa=state.mantissa.clone(),
        )
    return RowColumnState(
        step=state.step,
        momentum=state.momentum.clone(),
        axis_second=state.axis_second.clone(),
        access_count=state.access_count.clone(),
        global_second=state.global_second.clone(),
        token_axis=state.token_axis,
    )


def _candidate(
    mode: Geometry,
    parameter: Tensor,
    gradient: Tensor,
    state: CandidateState,
    config: GeometryConfig,
    descriptor: BlockDescriptor,
) -> CandidateUpdate:
    if mode == "coordinate":
        if not isinstance(state, CoordinateState):
            raise TypeError("coordinate candidate has the wrong state")
        return coordinate_candidate(parameter, gradient, state, config)
    if mode == "row_column":
        if not isinstance(state, RowColumnState):
            raise TypeError("row/column candidate has the wrong state")
        axis = _row_axis(descriptor)
        access_mask = gradient.float().ne(0).any(dim=1 - axis)
        return row_column_candidate(
            parameter,
            gradient,
            state,
            config,
            token_axis=axis,
            access_mask=access_mask,
        )
    if not isinstance(state, SpectralState):
        raise TypeError("spectral candidate has the wrong state")
    return spectral_candidate(parameter, gradient, state, config)


def _faro_candidate(
    mode: Geometry,
    parameter: Tensor,
    gradient: Tensor,
    state: FAROCandidateState,
    config: GeometryConfig,
    descriptor: BlockDescriptor,
    *,
    advanced_value: Tensor | None = None,
    state_advanced: bool = False,
) -> tuple[CandidateUpdate, Tensor | None, Tensor | None, Tensor | None]:
    if mode == "coordinate":
        if not isinstance(state, NativeAdamState):
            raise TypeError("FARO coordinate candidate has the wrong state")
        if state_advanced:
            delta = native_adam_delta_from_state(
                parameter,
                state,
                beta1=config.beta1,
                beta2=config.beta2,
                lr=config.lr,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        else:
            delta = native_adam_delta(
                parameter,
                gradient.float(),
                state,
                beta1=config.beta1,
                beta2=config.beta2,
                lr=config.lr,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        direction = -delta / max(config.lr, config.eps)
        update = _candidate_metrics(
            parameter,
            delta,
            direction,
            token_axis=None,
            measure_spectral_risk=config.measure_spectral_risk,
            eps=config.eps,
        )
        return update, None, None, None
    if mode == "row_column":
        if not isinstance(state, RowColumnState):
            raise TypeError("FARO row/column candidate has the wrong state")
        axis = _row_axis(descriptor)
        if state_advanced:
            if advanced_value is None:
                raise ValueError("advanced row/column state requires an access mask")
            update = row_column_candidate_from_state(
                parameter,
                state,
                config,
                token_axis=axis,
                access_mask=advanced_value,
            )
        else:
            update = _candidate(
                mode, parameter, gradient, state, config, descriptor
            )
        return update, None, None, None
    if not isinstance(state, NorMuonState):
        raise TypeError("FARO spectral candidate has the wrong state")
    direction = advanced_value
    if not state_advanced:
        direction = normuon_direction(
            gradient,
            state,
            momentum=config.momentum,
            beta2=config.beta2,
        )
    if direction is None:
        raise ValueError("advanced spectral state requires a direction")
    precise_parameter = None
    next_mantissa = None
    if parameter.dtype == torch.bfloat16:
        precise_parameter = parameter.detach().clone()
        next_mantissa = state.mantissa.clone()
        apply_cautious_precise_(
            precise_parameter,
            next_mantissa,
            direction,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        delta = precise_parameter.float() - parameter.float()
    else:
        parameter_f = parameter.float()
        mask = direction.float() * parameter_f >= 0
        delta = (
            -direction.float() * config.lr
            - parameter_f * mask * (config.weight_decay * config.lr)
        )
    update = _candidate_metrics(
        parameter,
        delta,
        direction,
        token_axis=None,
        measure_spectral_risk=config.measure_spectral_risk,
        eps=config.eps,
    )
    return update, precise_parameter, next_mantissa, direction


def _advance_faro_state(
    mode: Geometry,
    gradient: Tensor,
    state: FAROCandidateState,
    config: GeometryConfig,
    descriptor: BlockDescriptor,
) -> Tensor | None:
    if mode == "coordinate":
        if not isinstance(state, NativeAdamState):
            raise TypeError("FARO coordinate candidate has the wrong state")
        advance_native_adam_state(
            gradient.float(),
            state,
            beta1=config.beta1,
            beta2=config.beta2,
        )
        return None
    if mode == "row_column":
        if not isinstance(state, RowColumnState):
            raise TypeError("FARO row/column candidate has the wrong state")
        axis = _row_axis(descriptor)
        access_mask = gradient.float().ne(0).any(dim=1 - axis)
        return advance_row_column_state(
            gradient,
            state,
            config,
            token_axis=axis,
            access_mask=access_mask,
        )
    if not isinstance(state, NorMuonState):
        raise TypeError("FARO spectral candidate has the wrong state")
    return normuon_direction(
        gradient,
        state,
        momentum=config.momentum,
        beta2=config.beta2,
    )


class FixedGeometryOptimizer:
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        descriptors: Sequence[BlockDescriptor],
        coordinate_defaults: Mapping[str, object],
        muon_defaults: Mapping[str, object],
        variant: Literal["adam", "muon"],
        world_size: int,
    ):
        if world_size != 1:
            raise ValueError("the new Adam/Muon baselines are single-GPU only")
        if variant not in {"adam", "muon"}:
            raise ValueError("fixed geometry variant must be adam or muon")
        materialized, self.parameters = _named_parameter_map(named_parameters)
        del materialized
        self.descriptors = tuple(descriptors)
        self.coordinate_defaults = dict(coordinate_defaults)
        self.muon_defaults = dict(muon_defaults)
        self.variant = variant
        self.split_embed = False
        self.states: dict[str, dict[Geometry, CandidateState]] = {}
        for descriptor in self.descriptors:
            mode: Geometry = (
                "spectral"
                if variant == "muon" and descriptor.access_pattern == "dense"
                else "coordinate"
            )
            self.states[descriptor.label] = {
                mode: _initialize_state(
                    self.parameters[descriptor.parameter_label], descriptor, mode
                )
            }

    def _is_active(self, descriptor: BlockDescriptor, step: int) -> bool:
        if self.variant == "adam":
            return True
        return descriptor.update_frequency == "every_step" or step % 2 == 1

    def _effective_gradient(self, descriptor: BlockDescriptor) -> Tensor | None:
        parameter = self.parameters[descriptor.parameter_label]
        if descriptor.label == "embed" and not self.split_embed:
            return None
        if parameter.grad is None:
            return None
        gradient = select_block(parameter.grad, descriptor)
        if descriptor.label == "lm_head" and not self.split_embed:
            embed = self.parameters.get("embed")
            if embed is not None and embed.grad is not None:
                gradient = gradient + embed.grad.T
        return gradient

    @torch.no_grad()
    def step(
        self,
        *,
        step: int,
        lr_scale: float,
        muon_momentum: float,
        update_coordinate: bool,
    ) -> OptimizerStepResult:
        del update_coordinate
        modes: dict[str, Geometry] = {}
        updated_parameters: set[str] = set()
        for descriptor in self.descriptors:
            if not self._is_active(descriptor, step):
                continue
            gradient = self._effective_gradient(descriptor)
            if gradient is None:
                continue
            parameter = self.parameters[descriptor.parameter_label]
            block = select_block(parameter, descriptor)
            mode = next(iter(self.states[descriptor.label]))
            config = (
                _muon_config(
                    self.muon_defaults,
                    descriptor,
                    lr_scale,
                    muon_momentum,
                    measure_spectral_risk=False,
                )
                if mode == "spectral"
                else _coordinate_config(
                    self.coordinate_defaults,
                    descriptor,
                    lr_scale,
                    measure_spectral_risk=False,
                )
            )
            update = _candidate(
                mode,
                block,
                gradient,
                self.states[descriptor.label][mode],
                config,
                descriptor,
            )
            block.add_(update.delta.to(block.dtype))
            modes[descriptor.label] = mode
            updated_parameters.add(descriptor.parameter_label)

        if not self.split_embed and "lm_head" in updated_parameters:
            self.parameters["embed"].copy_(self.parameters["lm_head"].T)
            updated_parameters.add("embed")
            modes["embed"] = "coordinate"
        for label in updated_parameters:
            self.parameters[label].grad = None
        return OptimizerStepResult(
            modes=modes,
            decisions={},
            predicted_return=0.0,
            predicted_return_by_structure={},
        )

    def copy_lm_state_to_embed(self) -> None:
        source = self.states["lm_head"]
        self.states["embed"] = {
            mode: transpose_candidate_state(state) for mode, state in source.items()
        }
        self.split_embed = True

    def reset(self) -> None:
        for descriptor in self.descriptors:
            for mode in tuple(self.states[descriptor.label]):
                self.states[descriptor.label][mode] = _initialize_state(
                    self.parameters[descriptor.parameter_label], descriptor, mode
                )
        self.split_embed = False

    def state_dict(self) -> dict[str, object]:
        return {
            "states": copy.deepcopy(self.states),
            "split_embed": self.split_embed,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.states = copy.deepcopy(state_dict["states"])  # type: ignore[assignment]
        self.split_embed = bool(state_dict["split_embed"])


def _power_top_energy(matrix: Tensor, iterations: int = 2) -> Tensor:
    if matrix.ndim != 2 or not matrix.numel():
        return torch.zeros((), dtype=torch.float32, device=matrix.device)
    matrix_f = matrix.float()
    vector = torch.ones(
        matrix_f.shape[1], dtype=torch.float32, device=matrix.device
    ) / math.sqrt(matrix_f.shape[1])
    for _ in range(iterations):
        left = matrix_f @ vector
        left = left / left.norm().clamp_min(1e-12)
        vector = matrix_f.T @ left
        vector = vector / vector.norm().clamp_min(1e-12)
    top_squared = (matrix_f @ vector).square().sum()
    return top_squared / matrix_f.square().sum().clamp_min(1e-12)


def _state_momentum(state: FAROCandidateState) -> Tensor:
    if isinstance(state, (CoordinateState, NativeAdamState)):
        return state.exp_avg
    return state.momentum


class FAROOptimizer:
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        descriptors: Sequence[BlockDescriptor],
        coordinate_defaults: Mapping[str, object],
        muon_defaults: Mapping[str, object],
        router: FARORouter,
        world_size: int,
        probe_interval: int = 16,
        sparse_micro_labels: Iterable[str] = _SPARSE_MICRO_LABELS,
    ):
        if world_size != 1:
            raise ValueError("FARO is single-GPU only for this experiment")
        materialized, self.parameters = _named_parameter_map(named_parameters)
        self.descriptors = tuple(descriptors)
        self.coordinate_defaults = dict(coordinate_defaults)
        self.muon_defaults = dict(muon_defaults)
        self.router = router
        self.probe_interval = probe_interval
        self.collector = MicroGradientCollector(
            materialized, micro_count=8, sparse_labels=sparse_micro_labels
        )
        self.sparse_micro_labels = frozenset(sparse_micro_labels)
        self.states: dict[str, dict[Geometry, FAROCandidateState]] = {
            descriptor.label: {
                mode: _initialize_faro_state(
                    self.parameters[descriptor.parameter_label], descriptor, mode
                )
                for mode in legal_geometries(descriptor)
            }
            for descriptor in self.descriptors
        }
        self._dense_bank_groups: dict[str, tuple[BlockDescriptor, ...]] = {}
        for parameter_label in self.parameters:
            group = tuple(
                descriptor
                for descriptor in self.descriptors
                if descriptor.parameter_label == parameter_label
                and descriptor.access_pattern == "dense"
            )
            if len(group) > 1 and len({descriptor.shape for descriptor in group}) == 1:
                self._dense_bank_groups[parameter_label] = group
        self._all_dense_bank_groups = dict(self._dense_bank_groups)
        self.spectral_banks: dict[str, NorMuonState] = {}
        self._rebuild_spectral_banks(from_states=False)
        self.split_embed = False
        self.last_decisions: dict[str, RoutingDecision] = {}
        self.activation_drift: dict[str, Tensor] = {}
        self.frozen_budgets: dict[str, float] = {}
        self.all_routes_frozen = False
        self._counterfactual_scales = (0.9, 1.0, 1.1)
        self.counterfactual_selected_scale = 1.0
        self._counterfactual_plans: list[_CounterfactualPlan] | None = None
        self._counterfactual_capture_enabled = False
        self._freeze_compacted = False
        self._freeze_release_pending = False
        self._freeze_events: list[dict[str, object]] = []
        self.row_column_scale_samples: dict[str, list[Tensor]] = {
            descriptor.label: []
            for descriptor in self.descriptors
            if "row_column" in legal_geometries(descriptor)
        }
        self.row_column_scales: dict[str, Tensor] = {}
        for descriptor in self.descriptors:
            if descriptor.label not in self.row_column_scale_samples:
                continue
            block = select_block(
                self.parameters[descriptor.parameter_label], descriptor
            )
            self.row_column_scales[descriptor.label] = torch.ones(
                (), dtype=torch.float32, device=block.device
            )

    def _resolved_route(
        self, descriptor: BlockDescriptor
    ) -> tuple[Geometry, float] | None:
        state = self.router.states.get(descriptor.label)
        if state is not None and state.frozen:
            return state.current_mode, state.budget_scale
        if (
            not self.split_embed
            and descriptor.label == "embed"
            and descriptor.shared_weight is not None
        ):
            peer = self.router.states.get(descriptor.shared_weight)
            if peer is not None and peer.frozen:
                return peer.current_mode, peer.budget_scale
        return None

    def _compact_frozen_storage(self) -> None:
        for label, candidates in self.states.items():
            mode, state = next(iter(candidates.items()))
            self.states[label] = {mode: _clone_faro_state(state)}
        self._dense_bank_groups = {
            parameter_label: tuple(
                descriptor
                for descriptor in descriptors
                if next(iter(self.states[descriptor.label])) == "spectral"
            )
            for parameter_label, descriptors in self._all_dense_bank_groups.items()
            if any(
                next(iter(self.states[descriptor.label])) == "spectral"
                for descriptor in descriptors
            )
        }
        self._rebuild_spectral_banks(from_states=True)
        self.row_column_scale_samples = {}
        self.row_column_scales = {
            label: scale
            for label, scale in self.row_column_scales.items()
            if next(iter(self.states[label])) == "row_column"
        }
        self.activation_drift.clear()
        self._freeze_compacted = True
        self._freeze_release_pending = True

    def _sync_frozen_routes(self, step: int) -> None:
        newly_frozen: list[str] = []
        for descriptor in self.descriptors:
            if descriptor.label in self.frozen_budgets:
                continue
            resolved = self._resolved_route(descriptor)
            if resolved is None:
                continue
            mode, budget = resolved
            selected = self.states[descriptor.label][mode]
            self.states[descriptor.label] = {mode: selected}
            self.frozen_budgets[descriptor.label] = (
                budget * self.router.config.frozen_budget_scale
            )
            newly_frozen.append(descriptor.label)
        self.all_routes_frozen = len(self.frozen_budgets) == len(
            self.descriptors
        )
        if self.all_routes_frozen and not self._freeze_compacted:
            self._compact_frozen_storage()
        if not newly_frozen:
            return
        mode_counts: dict[Geometry, int] = {}
        for label in self.frozen_budgets:
            mode = next(iter(self.states[label]))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        self._freeze_events.append(
            {
                "step": step,
                "newly_frozen": len(newly_frozen),
                "total_frozen": len(self.frozen_budgets),
                "mode_counts": mode_counts,
                "all_routes_frozen": self.all_routes_frozen,
                "current_allocated_mib": None,
                "current_reserved_mib": None,
            }
        )

    def _cuda_device(self) -> torch.device | None:
        return next(
            (
                parameter.device
                for parameter in self.parameters.values()
                if parameter.is_cuda
            ),
            None,
        )

    def finalize_freeze_events(self) -> None:
        pending = [
            event
            for event in self._freeze_events
            if event["current_allocated_mib"] is None
        ]
        if not pending and not self._freeze_release_pending:
            return
        cuda_device = self._cuda_device()
        if self._freeze_release_pending and cuda_device is not None:
            torch.cuda.empty_cache()
        if cuda_device is None:
            allocated_mib = 0.0
            reserved_mib = 0.0
        else:
            allocated_mib = torch.cuda.memory_allocated(cuda_device) / (1024**2)
            reserved_mib = torch.cuda.memory_reserved(cuda_device) / (1024**2)
        for event in pending:
            event["current_allocated_mib"] = allocated_mib
            event["current_reserved_mib"] = reserved_mib
        self._freeze_release_pending = False

    def consume_freeze_events(self) -> list[dict[str, object]]:
        ready = [
            event
            for event in self._freeze_events
            if event["current_allocated_mib"] is not None
        ]
        self._freeze_events = [
            event
            for event in self._freeze_events
            if event["current_allocated_mib"] is None
        ]
        return ready

    def _rebuild_spectral_banks(self, *, from_states: bool) -> None:
        self.spectral_banks.clear()
        for parameter_label, descriptors in self._dense_bank_groups.items():
            if from_states:
                states = [
                    self.states[descriptor.label]["spectral"]
                    for descriptor in descriptors
                ]
                if not all(isinstance(state, NorMuonState) for state in states):
                    raise TypeError("dense spectral bank has non-NorMuon state")
                bank = NorMuonState(
                    momentum=torch.stack([state.momentum for state in states]),
                    second_momentum=torch.stack(
                        [state.second_momentum for state in states]
                    ),
                    mantissa=torch.stack([state.mantissa for state in states]),
                )
            else:
                parameter = self.parameters[parameter_label]
                shape = descriptors[0].shape
                blocks = parameter.view(-1, *shape)[: len(descriptors)]
                bank = NorMuonState.zeros_like(blocks)
            self.spectral_banks[parameter_label] = bank
            for index, descriptor in enumerate(descriptors):
                self.states[descriptor.label]["spectral"] = NorMuonState(
                    momentum=bank.momentum[index],
                    second_momentum=bank.second_momentum[index],
                    mantissa=bank.mantissa[index],
                )

    def capture_micro_gradient(self, micro_index: int, effective_tokens: int) -> None:
        if self.all_routes_frozen and not self.router.config.local_reward_engine:
            return
        self.collector.capture(micro_index, effective_tokens=effective_tokens)

    def start_step(self, step: int) -> None:
        if self.all_routes_frozen:
            self._counterfactual_capture_enabled = (
                self.router.config.counterfactual_probe_budget
                and not self.router.config.unit_budget
                and step % self.probe_interval == 1
            )
            if self.router.config.local_reward_engine:
                active_labels = {
                    descriptor.parameter_label for descriptor in self.descriptors
                }
                self.collector.start_step(active_labels)
            return
        probe_step = step % self.probe_interval == 1
        active_labels = {
            descriptor.parameter_label
            for descriptor in self.descriptors
            if probe_step
            or descriptor.label not in self.last_decisions
            or self.router.is_route_event(step, descriptor)
            or (
                descriptor.update_frequency == "odd_step"
                and self.router.is_route_event(step + 1, descriptor)
            )
        }
        self.collector.start_step(active_labels)

    def set_activation_drift(self, drift: Mapping[str, Tensor]) -> None:
        self.activation_drift = dict(drift)

    def _effective_full_gradient(self, descriptor: BlockDescriptor) -> Tensor | None:
        parameter = self.parameters[descriptor.parameter_label]
        if descriptor.label == "embed" and not self.split_embed:
            return None
        if parameter.grad is None:
            return None
        gradient = select_block(parameter.grad, descriptor)
        if descriptor.label == "lm_head" and not self.split_embed:
            embed = self.parameters.get("embed")
            if embed is not None and embed.grad is not None:
                gradient = gradient + embed.grad.T
        return gradient

    def _effective_micro_set(
        self,
        descriptor: BlockDescriptor,
        prepared: Mapping[str, MicroBatchSet | SparseMicroSet | None],
    ) -> MicroBatchSet | SparseMicroSet | None:
        if descriptor.label == "embed" and not self.split_embed:
            return None
        micro_set = prepared[descriptor.parameter_label]
        if micro_set is None:
            return None
        if descriptor.label != "lm_head" or self.split_embed:
            return micro_set
        embed_set = prepared.get("embed")
        if embed_set is None:
            return micro_set
        if isinstance(micro_set, SparseMicroSet) or isinstance(
            embed_set, SparseMicroSet
        ):
            raise TypeError("lm_head micro-set merge expects dense micro-batches")
        return MicroBatchSet(
            gradients=tuple(
                lm.float().add(embed.float().T).to(torch.bfloat16)
                for lm, embed in zip(
                    micro_set.gradients, embed_set.gradients, strict=True
                )
            ),
            effective_tokens=micro_set.effective_tokens,
        )

    def _config(
        self,
        mode: Geometry,
        descriptor: BlockDescriptor,
        lr_scale: float,
        muon_momentum: float,
        measure_spectral_risk: bool,
    ) -> GeometryConfig:
        if mode == "spectral":
            return _normuon_config(
                self.muon_defaults,
                descriptor,
                lr_scale,
                muon_momentum,
                measure_spectral_risk=measure_spectral_risk,
            )
        config = _coordinate_config(
            self.coordinate_defaults,
            descriptor,
            lr_scale,
            measure_spectral_risk=measure_spectral_risk,
        )
        if mode == "row_column":
            config.lr = float(config.lr)
            config.active_axis_second = (
                descriptor.access_pattern == "token_lookup"
            )
        return config

    def _spectral_projection_uwr(
        self,
        block: Tensor,
        state: NorMuonState,
        direction: Tensor,
        config: GeometryConfig,
    ) -> float:
        if block.dtype == torch.bfloat16:
            precise_raw = (
                block.view(torch.uint16).to(torch.int32) << 16
            ) | state.mantissa.to(torch.int32)
            parameter = precise_raw.view(torch.float32)
        else:
            parameter = block.float()
        direction = direction.float()
        mask = direction * parameter >= 0
        delta = -config.lr * (
            direction + parameter * mask * config.weight_decay
        )
        return float(delta.norm() / parameter.norm().clamp_min(config.eps))

    def _match_row_column_uwr(
        self,
        descriptor: BlockDescriptor,
        step: int,
        coordinate: CandidateUpdate,
        row_column: CandidateUpdate,
    ) -> None:
        eps = float(self.coordinate_defaults.get("eps", 1e-8))
        if descriptor.access_pattern in {"token_lookup", "token_output"}:
            coordinate_norm = coordinate.delta.float().norm()
            row_column_norm = row_column.delta.float().norm()
            scale = torch.where(
                row_column_norm > eps,
                coordinate_norm / row_column_norm.clamp_min(eps),
                torch.ones_like(row_column_norm),
            )
            self.row_column_scales[descriptor.label].copy_(scale)
        else:
            samples = self.row_column_scale_samples[descriptor.label]
            if step < self.router.config.collect_steps:
                samples.append(
                    (
                        coordinate.uwr / row_column.uwr.clamp_min(eps)
                    ).detach()
                )
                self.row_column_scales[descriptor.label] = torch.stack(
                    samples
                ).median()
            scale = self.row_column_scales[descriptor.label]
        row_column.delta.mul_(scale)
        row_column.uwr.mul_(scale)
        row_column.max_axis_uwr.mul_(scale)
        row_column.spectral_risk.mul_(scale)

    def _apply_saved_row_column_scale(
        self, descriptor: BlockDescriptor, update: CandidateUpdate
    ) -> None:
        scale = self.row_column_scales[descriptor.label]
        update.delta.mul_(scale)
        update.uwr.mul_(scale)
        update.max_axis_uwr.mul_(scale)
        update.spectral_risk.mul_(scale)

    def _nonroute_decision(
        self, descriptor: BlockDescriptor
    ) -> RoutingDecision:
        previous = self.last_decisions.get(descriptor.label)
        if previous is None:
            raise RuntimeError(
                f"{descriptor.label} has no preceding route decision"
            )
        state = self.router.initialize(descriptor)
        return replace(
            previous,
            mode=state.current_mode,
            budget_scale=state.budget_scale,
            switched=False,
            route_event=False,
            reason="not_route_event",
        )

    def _advance_spectral_banks(
        self,
        *,
        lr_scale: float,
        muon_momentum: float,
    ) -> dict[str, Tensor]:
        directions: dict[str, Tensor] = {}
        for parameter_label, descriptors in self._dense_bank_groups.items():
            parameter = self.parameters[parameter_label]
            if parameter.grad is None:
                continue
            if descriptors == self._all_dense_bank_groups[parameter_label]:
                shape = descriptors[0].shape
                gradients = parameter.grad.view(-1, *shape)[: len(descriptors)]
            else:
                gradients = torch.stack(
                    [
                        select_block(parameter.grad, descriptor)
                        for descriptor in descriptors
                    ]
                )
            config = self._config(
                "spectral",
                descriptors[0],
                lr_scale,
                muon_momentum,
                False,
            )
            bank_directions = normuon_direction(
                gradients,
                self.spectral_banks[parameter_label],
                momentum=config.momentum,
                beta2=config.beta2,
            )
            for index, descriptor in enumerate(descriptors):
                directions[descriptor.label] = bank_directions[index]
        return directions

    def _capture_counterfactual_update(self, plan: _CounterfactualPlan) -> None:
        if self._counterfactual_plans is not None:
            self._counterfactual_plans.append(plan)

    def _frozen_local_reward_statistics(
        self,
        descriptor: BlockDescriptor,
        update: CandidateUpdate,
        prepared: Mapping[str, MicroBatchSet | SparseMicroSet | None],
        *,
        base_budget: float,
    ) -> tuple[float, dict[str, float | bool]]:
        micro_set = self._effective_micro_set(descriptor, prepared)
        if micro_set is None:
            raise RuntimeError(
                f"missing frozen micro-gradients for {descriptor.label}"
            )
        micro_blocks = _micro_blocks(micro_set, descriptor, self.parameters[descriptor.parameter_label])
        returns = tuple(
            _micro_return(
                micro,
                update.delta,
                tokens=tokens,
                micro_count=self.collector.micro_count,
            )
            for micro, tokens in zip(
                micro_blocks, micro_set.effective_tokens, strict=True
            )
        )
        if len(returns) != self.collector.micro_count:
            raise RuntimeError("frozen local reward requires eight micro returns")
        observation = CandidateObservation.from_returns(
            torch.stack(returns),
            uwr=update.uwr,
            max_axis_uwr=update.max_axis_uwr,
            spectral_risk=update.spectral_risk,
            drift=torch.zeros((), device=update.delta.device),
            gradient_sv_concentration=torch.zeros((), device=update.delta.device),
            momentum_sv_concentration=torch.zeros((), device=update.delta.device),
        )
        budget, statistics = self.router.local_reward_budget(
            descriptor.label, base_budget, observation
        )
        return budget, statistics

    def _frozen_decision(
        self,
        descriptor: BlockDescriptor,
        mode: Geometry,
        budget: float,
        *,
        budget_statistics: dict[str, float | bool] | None = None,
    ) -> RoutingDecision:
        return RoutingDecision(
            label=descriptor.label,
            mode=mode,
            budget_scale=budget,
            switched=False,
            route_event=False,
            reason="frozen",
            scores={},
            hard_violations=(),
            risk_ratios={},
            hard_risk_triggers={},
            candidate_statistics={},
            budget_statistics=budget_statistics or {},
        )

    def _frozen_step(
        self,
        *,
        step: int,
        lr_scale: float,
        muon_momentum: float,
    ) -> OptimizerStepResult:
        directions = self._advance_spectral_banks(
            lr_scale=lr_scale,
            muon_momentum=muon_momentum,
        )
        capture_counterfactual = (
            self.router.config.counterfactual_probe_budget
            and not self.router.config.unit_budget
            and step % self.probe_interval == 1
        )
        self._counterfactual_plans = [] if capture_counterfactual else None
        if self.router.config.erc_budget:
            # Reallocate last step's observed risk before applying this step's
            # budgets. Uses only prior-step measurements, so the frozen update
            # direction is never affected by the current gradient.
            self.router.compute_erc_multipliers(self.frozen_budgets)
        prepared: dict[str, MicroBatchSet | SparseMicroSet | None] | None = None
        if self.router.config.local_reward_engine:
            frequencies = {
                label: next(
                    descriptor.update_frequency
                    for descriptor in self.descriptors
                    if descriptor.parameter_label == label
                )
                for label in self.parameters
                if any(
                    descriptor.parameter_label == label
                    for descriptor in self.descriptors
                )
            }
            prepared = self.collector.prepare(frequencies, step=step)
        decisions: dict[str, RoutingDecision] = {}
        modes: dict[str, Geometry] = {}
        updated_parameters: set[str] = set()
        for descriptor in self.descriptors:
            if descriptor.update_frequency == "odd_step" and step % 2 == 0:
                continue
            gradient = self._effective_full_gradient(descriptor)
            if gradient is None:
                continue
            parameter = self.parameters[descriptor.parameter_label]
            block = select_block(parameter, descriptor)
            gradient_block = (
                select_block(gradient, descriptor)
                if gradient.shape == parameter.shape
                else gradient
            )
            mode, state = next(iter(self.states[descriptor.label].items()))
            base_budget = self.frozen_budgets[descriptor.label]
            counterfactual_base_budget = base_budget
            budget = self.router.effective_frozen_budget(
                descriptor.label,
                base_budget,
            )
            appetite_budget = budget
            counterfactual_scale = (
                self.counterfactual_selected_scale
                if self.router.config.counterfactual_probe_budget
                and not self.router.config.kelly_risk_controller
                and not self.router.config.unit_budget
                else 1.0
            )
            pre_alpha_budget = budget
            reference_uwr = self.router.states[descriptor.label].reference_uwr
            config = self._config(
                mode,
                descriptor,
                lr_scale,
                muon_momentum,
                measure_spectral_risk=False,
            )
            budget_statistics: dict[str, float | bool] = {}
            update_uwr = 0.0
            reward_update: CandidateUpdate | None = None
            if mode == "coordinate":
                if not isinstance(state, NativeAdamState):
                    raise TypeError("frozen coordinate state has the wrong type")
                delta = native_adam_delta(
                    block,
                    gradient_block.float(),
                    state,
                    beta1=config.beta1,
                    beta2=config.beta2,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    eps=config.eps,
                )
                if prepared is not None:
                    reward_update = _candidate_metrics(
                        block,
                        delta,
                        -delta / max(config.lr, config.eps),
                        token_axis=None,
                        measure_spectral_risk=False,
                        eps=config.eps,
                    )
                    budget, budget_statistics = self._frozen_local_reward_statistics(
                        descriptor,
                        reward_update,
                        prepared,
                        base_budget=budget,
                    )
                    pre_alpha_budget = budget
                    budget_statistics["base_budget"] = base_budget
                    budget_statistics["appetite_budget"] = appetite_budget
                    budget_statistics["reward_budget"] = budget
                if self.router.config.counterfactual_probe_budget:
                    budget = min(
                        self.router.config.budget_max,
                        max(self.router.config.budget_min, pre_alpha_budget * counterfactual_scale),
                    )
                nominal_budget = budget
                if self.router.config.risk_budget_projection:
                    update_uwr = float(
                        delta.float().norm()
                        / block.float().norm().clamp_min(config.eps)
                    )
                    budget = self.router.project_budget(
                        descriptor.label, budget, update_uwr, update_uwr
                    )
                if self._counterfactual_plans is not None:
                    self._counterfactual_plans.append(_CounterfactualPlan(
                        descriptor.parameter_label, descriptor, mode, block.detach().clone(),
                        delta.detach().clone(), None, None, config,
                        counterfactual_base_budget
                        if self.router.config.kelly_risk_controller
                        else pre_alpha_budget,
                        update_uwr,
                        self.router.risk_budget_cap(
                            descriptor.label, update_uwr, update_uwr
                        ) if self.router.config.risk_budget_projection else None,
                        self.router.kelly_scale
                        if self.router.config.kelly_risk_controller
                        else counterfactual_scale,
                    ))
                block.add_(delta, alpha=budget)
            elif mode == "row_column":
                if not isinstance(state, RowColumnState):
                    raise TypeError("frozen row/column state has the wrong type")
                access_mask = _advance_faro_state(
                    mode, gradient_block, state, config, descriptor
                )
                if access_mask is None:
                    raise RuntimeError("row/column update requires an access mask")
                update = row_column_candidate_from_state(
                    block,
                    state,
                    config,
                    token_axis=_row_axis(descriptor),
                    access_mask=access_mask,
                )
                self._apply_saved_row_column_scale(descriptor, update)
                if prepared is not None:
                    reward_update = update
                    budget, budget_statistics = self._frozen_local_reward_statistics(
                        descriptor,
                        reward_update,
                        prepared,
                        base_budget=budget,
                    )
                    pre_alpha_budget = budget
                    budget_statistics["base_budget"] = base_budget
                    budget_statistics["appetite_budget"] = appetite_budget
                    budget_statistics["reward_budget"] = budget
                if self.router.config.counterfactual_probe_budget:
                    budget = min(
                        self.router.config.budget_max,
                        max(self.router.config.budget_min, pre_alpha_budget * counterfactual_scale),
                    )
                nominal_budget = budget
                if self.router.config.risk_budget_projection:
                    update_uwr = max(float(update.uwr), float(update.max_axis_uwr))
                    budget = self.router.project_budget(
                        descriptor.label, budget, update_uwr, update_uwr
                    )
                if self._counterfactual_plans is not None:
                    self._counterfactual_plans.append(_CounterfactualPlan(
                        descriptor.parameter_label, descriptor, mode, block.detach().clone(),
                        update.delta.detach().clone(), None, None, config,
                        counterfactual_base_budget
                        if self.router.config.kelly_risk_controller
                        else pre_alpha_budget,
                        update_uwr,
                        self.router.risk_budget_cap(
                            descriptor.label, update_uwr, update_uwr
                        ) if self.router.config.risk_budget_projection else None,
                        self.router.kelly_scale
                        if self.router.config.kelly_risk_controller
                        else counterfactual_scale,
                    ))
                block.add_(update.delta, alpha=budget)
            else:
                if not isinstance(state, NorMuonState):
                    raise TypeError("frozen spectral state has the wrong type")
                direction = directions.get(descriptor.label)
                if direction is None:
                    direction = normuon_direction(
                        gradient_block,
                        state,
                        momentum=config.momentum,
                        beta2=config.beta2,
                    )
                if block.dtype == torch.bfloat16:
                    precise_parameter = block.detach().clone()
                    next_mantissa = state.mantissa.clone()
                    apply_cautious_precise_(
                        precise_parameter,
                        next_mantissa,
                        direction,
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                    )
                if prepared is not None:
                    if block.dtype == torch.bfloat16:
                        precise_parameter = block.detach().clone()
                        next_mantissa = state.mantissa.clone()
                        apply_cautious_precise_(
                            precise_parameter,
                            next_mantissa,
                            direction,
                            lr=config.lr,
                            weight_decay=config.weight_decay,
                        )
                        reward_update = _candidate_metrics(
                            block,
                            precise_parameter.float() - block.float(),
                            direction,
                            token_axis=None,
                            measure_spectral_risk=False,
                            eps=config.eps,
                        )
                    else:
                        block_f = block.float()
                        mask = direction.float() * block_f >= 0
                        delta = (
                            -direction.float() * config.lr
                            - block_f
                            * mask
                            * (config.weight_decay * config.lr)
                        )
                        reward_update = _candidate_metrics(
                            block,
                            delta,
                            direction,
                            token_axis=None,
                            measure_spectral_risk=False,
                            eps=config.eps,
                        )
                    budget, budget_statistics = self._frozen_local_reward_statistics(
                        descriptor,
                        reward_update,
                        prepared,
                        base_budget=budget,
                    )
                    pre_alpha_budget = budget
                    budget_statistics["base_budget"] = base_budget
                    budget_statistics["appetite_budget"] = appetite_budget
                    budget_statistics["reward_budget"] = budget
                if self.router.config.counterfactual_probe_budget:
                    budget = min(
                        self.router.config.budget_max,
                        max(self.router.config.budget_min, pre_alpha_budget * counterfactual_scale),
                    )
                nominal_budget = budget
                if self.router.config.risk_budget_projection:
                    update_uwr = self._spectral_projection_uwr(
                        block, state, direction, config
                    )
                    budget = self.router.project_budget(
                        descriptor.label, budget, update_uwr, update_uwr
                    )
                if self._counterfactual_plans is not None:
                    self._counterfactual_plans.append(_CounterfactualPlan(
                        descriptor.parameter_label, descriptor, mode, block.detach().clone(),
                        None,
                        state.mantissa.detach().clone() if block.dtype == torch.bfloat16 else None,
                        direction.detach().clone(), config,
                        counterfactual_base_budget
                        if self.router.config.kelly_risk_controller
                        else pre_alpha_budget,
                        update_uwr,
                        self.router.risk_budget_cap(
                            descriptor.label, update_uwr, update_uwr
                        ) if self.router.config.risk_budget_projection else None,
                        self.router.kelly_scale
                        if self.router.config.kelly_risk_controller
                        else counterfactual_scale,
                    ))
                if block.dtype == torch.bfloat16:
                    apply_cautious_precise_(
                        block,
                        state.mantissa,
                        direction,
                        lr=config.lr * budget,
                        weight_decay=config.weight_decay,
                    )
                else:
                    block_f = block.float()
                    mask = direction.float() * block_f >= 0
                    block.add_(
                        -direction.float() * (config.lr * budget)
                        - block_f
                        * mask
                        * (config.weight_decay * config.lr * budget)
                    )
            if self.router.config.risk_budget_projection:
                budget_statistics = {
                    **budget_statistics,
                    "projected": budget < nominal_budget,
                    "nominal_budget": nominal_budget,
                    "pre_alpha_budget": pre_alpha_budget,
                    "counterfactual_scale": counterfactual_scale,
                    "effective_budget": budget,
                    "reference_uwr": reference_uwr,
                    "update_uwr": update_uwr,
                    "effective_uwr": update_uwr * budget,
                }
            elif self.router.config.counterfactual_probe_budget and not self.router.config.unit_budget:
                budget_statistics = {
                    **budget_statistics,
                    "pre_alpha_budget": pre_alpha_budget,
                    "counterfactual_scale": counterfactual_scale,
                    "nominal_budget": nominal_budget,
                    "effective_budget": budget,
                }
            elif budget_statistics:
                budget_statistics = {
                    **budget_statistics,
                    "nominal_budget": nominal_budget,
                    "effective_budget": budget,
                }
            if self.router.config.kelly_risk_controller:
                budget_statistics = {
                    **budget_statistics,
                    **self.router.kelly_audit(),
                }
            if self.router.config.erc_budget:
                self.router.record_erc_risk(descriptor.label, update_uwr)
                budget_statistics = {
                    **budget_statistics,
                    "erc_scale": self.router.erc_scale(descriptor.label),
                }
            decision = self._frozen_decision(
                descriptor, mode, budget, budget_statistics=budget_statistics
            )
            decisions[descriptor.label] = decision
            modes[descriptor.label] = mode
            updated_parameters.add(descriptor.parameter_label)
        if not self.split_embed and "lm_head" in updated_parameters:
            self.parameters["embed"].copy_(self.parameters["lm_head"].T)
            updated_parameters.add("embed")
        for label in updated_parameters:
            self.parameters[label].grad = None
        if prepared is not None:
            self.collector.finish_step()
        if self._counterfactual_plans is not None and not capture_counterfactual:
            self._counterfactual_plans = None
        if updated_parameters:
            self.router.advance_kelly_step()
        self.last_decisions.update(decisions)
        return OptimizerStepResult(
            modes=modes,
            decisions=decisions,
            predicted_return=0.0,
            predicted_return_by_structure={},
        )

    @torch.no_grad()
    def step(
        self,
        *,
        step: int,
        lr_scale: float,
        muon_momentum: float,
        update_coordinate: bool,
    ) -> OptimizerStepResult:
        del update_coordinate
        if self.all_routes_frozen:
            return self._frozen_step(
                step=step,
                lr_scale=lr_scale,
                muon_momentum=muon_momentum,
            )
        frequencies = {
            label: next(
                descriptor.update_frequency
                for descriptor in self.descriptors
                if descriptor.parameter_label == label
            )
            for label in self.parameters
            if any(
                descriptor.parameter_label == label
                for descriptor in self.descriptors
            )
        }
        prepared = self.collector.prepare(frequencies, step=step)
        decisions: dict[str, RoutingDecision] = {}
        modes: dict[str, Geometry] = {}
        predicted_return = 0.0
        predicted_return_by_structure: dict[str, float] = {}
        updated_parameters: set[str] = set()
        batched_spectral_directions = self._advance_spectral_banks(
            lr_scale=lr_scale,
            muon_momentum=muon_momentum,
        )

        for descriptor in self.descriptors:
            if (
                descriptor.update_frequency == "odd_step"
                and step % 2 == 0
            ):
                continue
            micro_set = self._effective_micro_set(descriptor, prepared)
            gradient = self._effective_full_gradient(descriptor)
            if gradient is None:
                continue
            parameter = self.parameters[descriptor.parameter_label]
            block = select_block(parameter, descriptor)
            gradient_block = select_block(gradient, descriptor) if gradient.shape == parameter.shape else gradient
            frozen_route = self.router.is_frozen(descriptor.label)
            route_event = not frozen_route and (
                self.router.is_route_event(step, descriptor)
                or descriptor.label not in self.last_decisions
            )
            route_state = self.router.initialize(descriptor)
            incumbent = route_state.current_mode
            candidate_updates: dict[Geometry, CandidateUpdate] = {}
            precise_parameters: dict[Geometry, Tensor] = {}
            next_mantissas: dict[Geometry, Tensor] = {}
            candidate_directions: dict[Geometry, Tensor] = {}
            candidate_configs: dict[Geometry, GeometryConfig] = {}
            candidate_returns: dict[Geometry, Tensor] = {}
            observations: dict[Geometry, CandidateObservation] = {}
            micro_blocks = _micro_blocks(micro_set, descriptor, parameter)
            gradient_concentration = (
                _power_top_energy(gradient_block)
                if route_event
                else None
            )
            spectral_state = self.states[descriptor.label].get("spectral")
            if (
                isinstance(spectral_state, NorMuonState)
                and incumbent != "spectral"
            ):
                spectral_state.mantissa.zero_()
            for mode, state in self.states[descriptor.label].items():
                config = self._config(
                    mode,
                    descriptor,
                    lr_scale,
                    muon_momentum,
                    measure_spectral_risk=route_event,
                )
                materialize = route_event or mode == incumbent
                advanced_value = (
                    batched_spectral_directions.get(descriptor.label)
                    if mode == "spectral"
                    else None
                )
                state_advanced = advanced_value is not None
                if not materialize:
                    if not state_advanced:
                        _advance_faro_state(
                            mode,
                            gradient_block,
                            state,
                            config,
                            descriptor,
                        )
                    continue
                (
                    update,
                    precise_parameter,
                    next_mantissa,
                    candidate_direction,
                ) = _faro_candidate(
                    mode,
                    block,
                    gradient_block,
                    state,
                    config,
                    descriptor,
                    advanced_value=advanced_value,
                    state_advanced=state_advanced,
                )
                candidate_configs[mode] = config
                candidate_updates[mode] = update
                if precise_parameter is not None:
                    precise_parameters[mode] = precise_parameter
                if next_mantissa is not None:
                    next_mantissas[mode] = next_mantissa
                if candidate_direction is not None:
                    candidate_directions[mode] = candidate_direction
                if mode == "row_column":
                    if route_event:
                        self._match_row_column_uwr(
                            descriptor,
                            step,
                            candidate_updates["coordinate"],
                            update,
                        )
                    else:
                        self._apply_saved_row_column_scale(
                            descriptor, update
                        )
                if micro_set is not None:
                    returns = torch.stack(
                        [
                            _micro_return(
                                micro,
                                update.delta,
                                tokens=tokens,
                                micro_count=self.collector.micro_count,
                            )
                            for micro, tokens in zip(
                                micro_blocks,
                                micro_set.effective_tokens,
                                strict=True,
                            )
                        ]
                    )
                    candidate_returns[mode] = returns
                    if route_event:
                        observations[mode] = CandidateObservation.from_returns(
                            returns,
                            uwr=update.uwr,
                            max_axis_uwr=update.max_axis_uwr,
                            spectral_risk=update.spectral_risk,
                            drift=self.activation_drift.get(
                                descriptor.label,
                                torch.zeros((), device=block.device),
                            ),
                            gradient_sv_concentration=gradient_concentration,
                            momentum_sv_concentration=_power_top_energy(
                                _state_momentum(state)
                            ),
                        )
            decision = (
                self._frozen_decision(
                    descriptor,
                    incumbent,
                    self.frozen_budgets[descriptor.label],
                )
                if frozen_route
                else self.router.route(
                    step,
                    (descriptor,),
                    {descriptor.label: observations},
                )[descriptor.label]
                if route_event
                else self._nonroute_decision(descriptor)
            )
            selected = candidate_updates[decision.mode]
            selected_state = self.states[descriptor.label][decision.mode]
            nominal_budget = decision.budget_scale
            projection_enabled = frozen_route and self.router.config.risk_budget_projection
            update_uwr = 0.0
            if projection_enabled:
                if decision.mode == "spectral":
                    update_uwr = self._spectral_projection_uwr(
                        block,
                        selected_state,
                        candidate_directions[decision.mode],
                        candidate_configs[decision.mode],
                    )
                else:
                    update_uwr = max(float(selected.uwr), float(selected.max_axis_uwr))
            projected_budget = (
                self.router.project_budget(
                    descriptor.label,
                    nominal_budget,
                    update_uwr,
                    update_uwr,
                )
                if projection_enabled
                else nominal_budget
            )
            if projection_enabled:
                decision.budget_scale = projected_budget
                decision.budget_statistics = {
                    **decision.budget_statistics,
                    "projected": projected_budget < nominal_budget,
                    "nominal_budget": nominal_budget,
                    "effective_budget": projected_budget,
                    "reference_uwr": self.router.states[descriptor.label].reference_uwr,
                    "update_uwr": update_uwr,
                    "effective_uwr": update_uwr * projected_budget,
                }
            if (
                decision.mode == "spectral"
                and isinstance(selected_state, NorMuonState)
                and block.dtype == torch.bfloat16
            ):
                if decision.budget_scale == 1.0:
                    block.copy_(precise_parameters[decision.mode])
                    selected_state.mantissa.copy_(
                        next_mantissas[decision.mode]
                    )
                else:
                    config = candidate_configs[decision.mode]
                    apply_cautious_precise_(
                        block,
                        selected_state.mantissa,
                        candidate_directions[decision.mode],
                        lr=config.lr * decision.budget_scale,
                        weight_decay=config.weight_decay,
                    )
            else:
                block.add_(selected.delta, alpha=decision.budget_scale)
                if isinstance(spectral_state, NorMuonState):
                    spectral_state.mantissa.zero_()
            selected_returns = candidate_returns.get(decision.mode)
            block_predicted_return = (
                float(selected_returns[0]) * decision.budget_scale
                if selected_returns is not None
                else 0.0
            )
            predicted_return += block_predicted_return
            predicted_return_by_structure[descriptor.structure_class] = (
                predicted_return_by_structure.get(descriptor.structure_class, 0.0)
                + block_predicted_return
            )
            decisions[descriptor.label] = decision
            modes[descriptor.label] = decision.mode
            updated_parameters.add(descriptor.parameter_label)

        self._sync_frozen_routes(step)

        if not self.split_embed and "lm_head" in updated_parameters:
            self.parameters["embed"].copy_(self.parameters["lm_head"].T)
            updated_parameters.add("embed")
        for label in updated_parameters:
            self.parameters[label].grad = None
        self.collector.finish_step()
        self.last_decisions.update(decisions)
        return OptimizerStepResult(
            modes=modes,
            decisions=decisions,
            predicted_return=predicted_return,
            predicted_return_by_structure=predicted_return_by_structure,
        )

    def update_kelly_state(
        self,
        *,
        probe_before_loss: float,
        realized_return: float,
        audit: Mapping[str, object],
    ) -> dict[str, object]:
        return self.router.update_kelly_state(
            probe_before_loss=probe_before_loss,
            realized_return=realized_return,
            audit=audit,
        )

    def update_ds_state(self, *, realized_return: float) -> dict[str, object]:
        return self.router.update_ds_state(realized_return=realized_return)

    def apply_probe(
        self,
        realized_return: float,
        predicted_return: float,
        structure_returns: Mapping[str, tuple[float, float]],
        optimizer_result: OptimizerStepResult | None = None,
    ) -> None:
        self.router.calibrate(
            realized_return, predicted_return, structure_returns
        )
        if not self.router.config.adaptive_risk_appetite:
            return
        risk_pressure: dict[str, float] = {}
        result = optimizer_result
        if result is None:
            return
        for label, decision in result.decisions.items():
            structure = next(
                descriptor.structure_class
                for descriptor in self.descriptors
                if descriptor.label == label
            )
            ratios = decision.risk_ratios.get(decision.mode, {})
            pressure = max(
                0.0,
                max(
                    float(value)
                    for value in ratios.values()
                    if value is not None and math.isfinite(float(value))
                )
                - 1.0,
            ) if ratios else 0.0
            risk_pressure[structure] = max(
                risk_pressure.get(structure, 0.0), pressure
            )
        pressure_value = max(risk_pressure.values(), default=0.0)
        self.router.update_global_risk_appetite(
            float(realized_return), pressure_value
        )

    @torch.no_grad()
    def evaluate_counterfactual_probe(
        self, evaluate_loss: callable
    ) -> dict[str, object]:
        plans = self._counterfactual_plans
        if not plans:
            return {}
        scales = self._counterfactual_scales
        parameter_snapshots = {
            label: parameter.detach().clone()
            for label, parameter in self.parameters.items()
        }
        losses: list[float] = []
        try:
            for scale in scales:
                try:
                    for plan in plans:
                        parameter = self.parameters[plan.parameter_label]
                        block = select_block(parameter, plan.descriptor)
                        block.copy_(plan.pre_block)
                        if plan.base_delta is not None:
                            budget = min(
                                self.router.config.budget_max,
                                max(
                                    self.router.config.budget_min,
                                    plan.base_budget * scale,
                                ),
                            )
                            if plan.projection_cap is not None:
                                budget = min(budget, max(0.0, plan.projection_cap))
                            block.add_(plan.base_delta, alpha=budget)
                        else:
                            if plan.direction is None:
                                raise RuntimeError("counterfactual spectral plan lacks direction")
                            budget = min(
                                self.router.config.budget_max,
                                max(
                                    self.router.config.budget_min,
                                    plan.base_budget * scale,
                                ),
                            )
                            if plan.projection_cap is not None:
                                budget = min(budget, max(0.0, plan.projection_cap))
                            if block.dtype == torch.bfloat16:
                                if plan.pre_mantissa is None:
                                    raise RuntimeError("BF16 spectral plan lacks mantissa")
                                scratch_mantissa = plan.pre_mantissa.clone()
                                apply_cautious_precise_(
                                    block,
                                    scratch_mantissa,
                                    plan.direction,
                                    lr=plan.config.lr * budget,
                                    weight_decay=plan.config.weight_decay,
                                )
                            else:
                                block_f = block.float()
                                mask = plan.direction.float() * block_f >= 0
                                block.add_(
                                    -plan.direction.float() * (plan.config.lr * budget)
                                    - block_f
                                    * mask
                                    * (plan.config.weight_decay * plan.config.lr * budget)
                                )
                    if not self.split_embed and any(
                        plan.descriptor.label == "lm_head" for plan in plans
                    ):
                        self.parameters["embed"].copy_(self.parameters["lm_head"].T)
                    loss = float(evaluate_loss())
                finally:
                    for label, snapshot in parameter_snapshots.items():
                        self.parameters[label].copy_(snapshot)
                    if not all(
                        torch.equal(self.parameters[label], snapshot)
                        for label, snapshot in parameter_snapshots.items()
                    ):
                        raise RuntimeError("counterfactual parameter restoration failed")
                if not math.isfinite(loss):
                    raise ValueError("counterfactual loss must be finite")
                losses.append(loss)
            best = min(
                range(len(scales)),
                key=lambda index: (losses[index], abs(scales[index] - 1.0)),
            )
            self.counterfactual_selected_scale = scales[best]
            return {
                "scales": scales,
                "losses": tuple(losses),
                "selected_scale": self.counterfactual_selected_scale,
                "applied_scale": plans[0].applied_scale,
                "current_scale": plans[0].applied_scale,
                "selection_used_next_update": True,
                "restore_verified": True,
            }
        finally:
            self._counterfactual_plans = None

    def copy_lm_state_to_embed(self) -> None:
        self.states["embed"] = {
            mode: _transpose_faro_state(state)
            for mode, state in self.states["lm_head"].items()
        }
        if "lm_head" in self.router.states:
            self.router.states["embed"] = copy.deepcopy(
                self.router.states["lm_head"]
            )
        if "lm_head" in self.frozen_budgets:
            self.frozen_budgets["embed"] = self.frozen_budgets["lm_head"]
        if "lm_head" in self.row_column_scales:
            self.row_column_scales["embed"] = self.row_column_scales[
                "lm_head"
            ].clone()
            if "lm_head" in self.row_column_scale_samples:
                self.row_column_scale_samples["embed"] = [
                    sample.clone()
                    for sample in self.row_column_scale_samples["lm_head"]
                ]
        self.split_embed = True
        self.all_routes_frozen = len(self.frozen_budgets) == len(
            self.descriptors
        )

    def reset(self) -> None:
        for descriptor in self.descriptors:
            self.states[descriptor.label] = {
                mode: _initialize_faro_state(
                    self.parameters[descriptor.parameter_label], descriptor, mode
                )
                for mode in legal_geometries(descriptor)
            }
        self._dense_bank_groups = dict(self._all_dense_bank_groups)
        self._rebuild_spectral_banks(from_states=False)
        self.router.states.clear()
        self.collector = MicroGradientCollector(
            [(label, parameter) for label, parameter in self.parameters.items()],
            micro_count=8,
            sparse_labels=self.sparse_micro_labels,
        )
        self.split_embed = False
        self.last_decisions.clear()
        self.activation_drift.clear()
        self.frozen_budgets.clear()
        self.counterfactual_selected_scale = 1.0
        self._counterfactual_plans = None
        self._counterfactual_capture_enabled = False
        self.all_routes_frozen = False
        self._freeze_compacted = False
        self._freeze_release_pending = False
        self._freeze_events.clear()
        self.row_column_scale_samples = {
            descriptor.label: []
            for descriptor in self.descriptors
            if "row_column" in legal_geometries(descriptor)
        }
        self.row_column_scales = {}
        for descriptor in self.descriptors:
            if descriptor.label not in self.row_column_scale_samples:
                continue
            block = select_block(
                self.parameters[descriptor.parameter_label], descriptor
            )
            self.row_column_scales[descriptor.label] = torch.ones(
                (), dtype=torch.float32, device=block.device
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "states": copy.deepcopy(self.states),
            "router": copy.deepcopy(self.router.state_dict()),
            "split_embed": self.split_embed,
            "frozen_budgets": dict(self.frozen_budgets),
            "all_routes_frozen": self.all_routes_frozen,
            "counterfactual_selected_scale": self.counterfactual_selected_scale,
            "freeze_compacted": self._freeze_compacted,
            "freeze_release_pending": self._freeze_release_pending,
            "freeze_events": copy.deepcopy(self._freeze_events),
            "row_column_scale_samples": copy.deepcopy(
                self.row_column_scale_samples
            ),
            "row_column_scales": copy.deepcopy(self.row_column_scales),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.states = copy.deepcopy(state_dict["states"])  # type: ignore[assignment]
        self.frozen_budgets = {
            str(label): float(budget)
            for label, budget in dict(
                state_dict.get("frozen_budgets", {})  # type: ignore[arg-type]
            ).items()
        }
        self.all_routes_frozen = bool(
            state_dict.get("all_routes_frozen", False)
        )
        self.counterfactual_selected_scale = float(
            state_dict.get("counterfactual_selected_scale", 1.0)
        )
        self._counterfactual_plans = None
        self._counterfactual_capture_enabled = False
        self._freeze_compacted = bool(
            state_dict.get("freeze_compacted", False)
        )
        self._freeze_release_pending = bool(
            state_dict.get("freeze_release_pending", False)
        )
        self._freeze_events = copy.deepcopy(
            state_dict.get("freeze_events", [])  # type: ignore[arg-type]
        )
        self._dense_bank_groups = {
            parameter_label: tuple(
                descriptor
                for descriptor in descriptors
                if "spectral" in self.states[descriptor.label]
            )
            for parameter_label, descriptors in self._all_dense_bank_groups.items()
            if any(
                "spectral" in self.states[descriptor.label]
                for descriptor in descriptors
            )
        }
        self._rebuild_spectral_banks(from_states=True)
        self.router.load_state_dict(state_dict["router"])  # type: ignore[arg-type]
        self.split_embed = bool(state_dict["split_embed"])
        if "row_column_scale_samples" in state_dict:
            self.row_column_scale_samples = copy.deepcopy(
                state_dict["row_column_scale_samples"]  # type: ignore[assignment]
            )
        if "row_column_scales" in state_dict:
            self.row_column_scales = copy.deepcopy(
                state_dict["row_column_scales"]  # type: ignore[assignment]
            )
