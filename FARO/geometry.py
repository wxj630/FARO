from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor


@dataclass(slots=True)
class GeometryConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.95
    nesterov: bool = True
    rare_token_threshold: int = 8
    axis_norm_cap: float = 2.0
    active_axis_second: bool = False
    shape_scale: float | None = None
    measure_spectral_risk: bool = True


@dataclass(slots=True)
class CoordinateState:
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor

    @classmethod
    def zeros_like(cls, tensor: Tensor) -> "CoordinateState":
        return cls(
            step=0,
            exp_avg=torch.zeros_like(tensor, dtype=torch.float32),
            exp_avg_sq=torch.zeros_like(tensor, dtype=torch.float32),
        )


@dataclass(slots=True)
class RowColumnState:
    step: int
    momentum: Tensor
    axis_second: Tensor
    access_count: Tensor
    global_second: Tensor
    token_axis: int

    @classmethod
    def zeros_like(cls, tensor: Tensor, token_axis: int) -> "RowColumnState":
        if tensor.ndim != 2 or token_axis not in (0, 1):
            raise ValueError("row/column state requires a rank-2 tensor and axis 0 or 1")
        return cls(
            step=0,
            momentum=torch.zeros_like(tensor, dtype=torch.float32),
            axis_second=torch.zeros(
                tensor.shape[token_axis], dtype=torch.float32, device=tensor.device
            ),
            access_count=torch.zeros(
                tensor.shape[token_axis], dtype=torch.int64, device=tensor.device
            ),
            global_second=torch.zeros((), dtype=torch.float32, device=tensor.device),
            token_axis=token_axis,
        )


@dataclass(slots=True)
class SpectralState:
    momentum: Tensor

    @classmethod
    def zeros_like(cls, tensor: Tensor) -> "SpectralState":
        if tensor.ndim != 2:
            raise ValueError("spectral state requires a rank-2 tensor")
        return cls(momentum=torch.zeros_like(tensor, dtype=torch.float32))


CandidateState = Union[CoordinateState, RowColumnState, SpectralState]


@dataclass(slots=True)
class CandidateUpdate:
    delta: Tensor
    direction_norm: Tensor
    uwr: Tensor
    max_axis_uwr: Tensor
    spectral_risk: Tensor


def _power_spectral_norm(matrix: Tensor, iterations: int = 2) -> Tensor:
    matrix = matrix.float()
    vector = torch.ones(
        matrix.shape[1], dtype=matrix.dtype, device=matrix.device
    ) / math.sqrt(matrix.shape[1])
    for _ in range(iterations):
        left = matrix @ vector
        left = left / left.norm().clamp_min(1e-12)
        vector = matrix.T @ left
        vector = vector / vector.norm().clamp_min(1e-12)
    return (matrix @ vector).norm()


def _candidate_metrics(
    parameter: Tensor,
    delta: Tensor,
    direction: Tensor,
    *,
    token_axis: int | None,
    measure_spectral_risk: bool,
    eps: float,
) -> CandidateUpdate:
    parameter_f = parameter.float()
    delta_f = delta.float()
    uwr = delta_f.norm() / parameter_f.norm().clamp_min(eps)
    max_axis_uwr = uwr
    if token_axis is not None:
        reduce_axis = 1 - token_axis
        local_delta = delta_f.norm(dim=reduce_axis)
        local_parameter = parameter_f.norm(dim=reduce_axis).clamp_min(eps)
        max_axis_uwr = (local_delta / local_parameter).max()
    if measure_spectral_risk and parameter.ndim == 2:
        spectral_risk = _power_spectral_norm(delta_f) / _power_spectral_norm(
            parameter_f
        ).clamp_min(eps)
    else:
        spectral_risk = torch.zeros((), dtype=torch.float32, device=parameter.device)
    return CandidateUpdate(
        delta=delta_f,
        direction_norm=direction.float().norm(),
        uwr=uwr,
        max_axis_uwr=max_axis_uwr,
        spectral_risk=spectral_risk,
    )


@torch.no_grad()
def coordinate_candidate(
    parameter: Tensor,
    gradient: Tensor,
    state: CoordinateState,
    config: GeometryConfig,
) -> CandidateUpdate:
    gradient_f = gradient.float()
    state.step += 1
    state.exp_avg.mul_(config.beta1).add_(gradient_f, alpha=1 - config.beta1)
    state.exp_avg_sq.mul_(config.beta2).addcmul_(
        gradient_f, gradient_f, value=1 - config.beta2
    )
    bias1 = 1 - config.beta1**state.step
    bias2 = 1 - config.beta2**state.step
    direction = (state.exp_avg / bias1) / (
        (state.exp_avg_sq / bias2).sqrt() + config.eps
    )
    delta = -config.lr * direction - config.lr * config.weight_decay * parameter.float()
    return _candidate_metrics(
        parameter,
        delta,
        direction,
        token_axis=None,
        measure_spectral_risk=config.measure_spectral_risk,
        eps=config.eps,
    )


def _axis_view(vector: Tensor, token_axis: int) -> Tensor:
    return vector[:, None] if token_axis == 0 else vector[None, :]


@torch.no_grad()
def row_column_candidate(
    parameter: Tensor,
    gradient: Tensor,
    state: RowColumnState,
    config: GeometryConfig,
    *,
    token_axis: int,
    access_mask: Tensor | None = None,
) -> CandidateUpdate:
    access_mask = advance_row_column_state(
        gradient,
        state,
        config,
        token_axis=token_axis,
        access_mask=access_mask,
    )
    return row_column_candidate_from_state(
        parameter,
        state,
        config,
        token_axis=token_axis,
        access_mask=access_mask,
    )


@torch.no_grad()
def advance_row_column_state(
    gradient: Tensor,
    state: RowColumnState,
    config: GeometryConfig,
    *,
    token_axis: int,
    access_mask: Tensor | None = None,
) -> Tensor:
    if token_axis != state.token_axis:
        raise ValueError("row/column token axis does not match its state")
    gradient_f = gradient.float()
    state.step += 1
    state.momentum.mul_(config.beta1).add_(gradient_f, alpha=1 - config.beta1)
    reduce_axis = 1 - token_axis
    local_second = gradient_f.square().mean(dim=reduce_axis)
    if access_mask is None:
        access_mask = gradient_f.ne(0).any(dim=reduce_axis)
    access_mask = access_mask.to(device=gradient.device, dtype=torch.bool)
    state.axis_second.mul_(config.beta2).add_(
        local_second, alpha=1 - config.beta2
    )
    if config.active_axis_second:
        mask = access_mask.to(local_second.dtype)
        global_observation = (
            (local_second * mask).sum() / mask.sum().clamp_min(1)
        )
    else:
        global_observation = gradient_f.square().mean()
    state.global_second.mul_(config.beta2).add_(
        global_observation, alpha=1 - config.beta2
    )
    state.access_count.add_(access_mask.to(torch.int64))
    return access_mask


@torch.no_grad()
def row_column_candidate_from_state(
    parameter: Tensor,
    state: RowColumnState,
    config: GeometryConfig,
    *,
    token_axis: int,
    access_mask: Tensor,
) -> CandidateUpdate:
    if token_axis != state.token_axis:
        raise ValueError("row/column token axis does not match its state")

    reduce_axis = 1 - token_axis
    rare = state.access_count < config.rare_token_threshold
    conservative_second = torch.where(
        rare,
        torch.maximum(state.axis_second, state.global_second),
        state.axis_second,
    )
    denominator = _axis_view(conservative_second.sqrt() + config.eps, token_axis)
    direction = state.momentum / denominator
    direction.mul_(_axis_view(access_mask.to(direction.dtype), token_axis))
    delta = -config.lr * direction - config.lr * config.weight_decay * parameter.float()

    axis_norms = delta.norm(dim=reduce_axis)
    active_norms = axis_norms[access_mask]
    if active_norms.numel():
        cap = active_norms.median() * config.axis_norm_cap
        scale = torch.minimum(
            torch.ones_like(axis_norms), cap / axis_norms.clamp_min(config.eps)
        )
        delta.mul_(_axis_view(scale, token_axis))

    return _candidate_metrics(
        parameter,
        delta,
        direction,
        token_axis=token_axis,
        measure_spectral_risk=config.measure_spectral_risk,
        eps=config.eps,
    )


@torch.no_grad()
def newton_schulz5(matrix: Tensor, steps: int = 5) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be a rank-2 matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    result = matrix.to(torch.bfloat16)
    result = result / (result.norm() + 1e-7)
    transposed = result.shape[0] > result.shape[1]
    if transposed:
        result = result.T
    for _ in range(steps):
        gram = result @ result.T
        polynomial = b * gram + c * (gram @ gram)
        result = a * result + polynomial @ result
    if transposed:
        result = result.T
    return result


@torch.no_grad()
def spectral_candidate(
    parameter: Tensor,
    gradient: Tensor,
    state: SpectralState,
    config: GeometryConfig,
) -> CandidateUpdate:
    if parameter.ndim != 2 or gradient.ndim != 2:
        raise ValueError("spectral candidate requires rank-2 tensors")
    gradient_f = gradient.float()
    state.momentum.lerp_(gradient_f, 1 - config.momentum)
    if config.nesterov:
        momentum_direction = gradient_f.lerp(state.momentum, config.momentum)
    else:
        momentum_direction = state.momentum
    direction = newton_schulz5(momentum_direction).float()
    scale = config.shape_scale
    if scale is None:
        scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
    direction.mul_(scale)
    delta = -config.lr * direction - config.lr * config.weight_decay * parameter.float()
    return _candidate_metrics(
        parameter,
        delta,
        direction,
        token_axis=None,
        measure_spectral_risk=config.measure_spectral_risk,
        eps=config.eps,
    )


def transpose_candidate_state(state: CandidateState) -> CandidateState:
    if isinstance(state, CoordinateState):
        return CoordinateState(
            step=state.step,
            exp_avg=state.exp_avg.T.contiguous(),
            exp_avg_sq=state.exp_avg_sq.T.contiguous(),
        )
    if isinstance(state, RowColumnState):
        return RowColumnState(
            step=state.step,
            momentum=state.momentum.T.contiguous(),
            axis_second=state.axis_second.clone(),
            access_count=state.access_count.clone(),
            global_second=state.global_second.clone(),
            token_axis=1 - state.token_axis,
        )
    if isinstance(state, SpectralState):
        return SpectralState(momentum=state.momentum.T.contiguous())
    raise TypeError(f"unsupported candidate state: {type(state).__name__}")
