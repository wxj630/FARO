from dataclasses import dataclass

import torch
from torch import Tensor


POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@dataclass
class NativeAdamState:
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor

    @classmethod
    def zeros_like(cls, parameter: Tensor) -> "NativeAdamState":
        state = torch.zeros_like(parameter, dtype=torch.float32)
        return cls(step=0, exp_avg=state, exp_avg_sq=torch.zeros_like(state))


@dataclass
class NorMuonState:
    momentum: Tensor
    second_momentum: Tensor
    mantissa: Tensor

    @classmethod
    def zeros_like(cls, matrix: Tensor) -> "NorMuonState":
        if matrix.shape[-2] >= matrix.shape[-1]:
            second_shape = (*matrix.shape[:-1], 1)
        else:
            second_shape = (*matrix.shape[:-2], 1, matrix.shape[-1])
        return cls(
            momentum=torch.zeros_like(matrix, dtype=torch.float32),
            second_momentum=torch.zeros(
                second_shape, dtype=torch.float32, device=matrix.device
            ),
            mantissa=torch.zeros_like(matrix, dtype=torch.uint16),
        )


def polar_express_from_nesterov(nesterov: Tensor) -> Tensor:
    result = nesterov.bfloat16()
    result = result / (
        result.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6
    )
    if result.is_cuda:
        from triton_kernels import XXT, XTX, ba_plus_cAA

        result = result.contiguous()
        tall = result.shape[-2] > result.shape[-1]
        split = result.shape[-2] > 1024
        if tall:
            gram = torch.empty(
                (*result.shape[:-2], result.shape[-1], result.shape[-1]),
                device=result.device,
                dtype=result.dtype,
            )
            polynomial = torch.empty_like(gram)
            output = torch.empty_like(result)
            multiply = torch.bmm if result.ndim > 2 else torch.mm
            fused = torch.baddbmm if result.ndim > 2 else torch.addmm
            for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
                XTX(result, out=gram)
                ba_plus_cAA(gram, alpha=c, beta=b, out=polynomial)
                if split:
                    multiply(result, polynomial, out=output)
                    output.add_(result, alpha=a)
                else:
                    fused(
                        result,
                        result,
                        polynomial,
                        beta=a,
                        out=output,
                    )
                result, output = output, result
        else:
            gram = torch.empty(
                (*result.shape[:-1], result.shape[-2]),
                device=result.device,
                dtype=result.dtype,
            )
            polynomial = torch.empty_like(gram)
            output = torch.empty_like(result)
            fused = torch.baddbmm if result.ndim > 2 else torch.addmm
            for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
                XXT(result, out=gram)
                ba_plus_cAA(gram, alpha=c, beta=b, out=polynomial)
                fused(
                    result,
                    polynomial,
                    result,
                    beta=a,
                    out=output,
                )
                result, output = output, result
        return result
    tall = result.shape[-2] > result.shape[-1]
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = result.mT @ result if tall else result @ result.mT
        polynomial = b * gram + c * (gram @ gram)
        result = a * result + (
            result @ polynomial if tall else polynomial @ result
        )
    return result


def normuon_direction(
    gradient: Tensor,
    state: NorMuonState,
    *,
    momentum: float,
    beta2: float,
) -> Tensor:
    gradient = gradient.float()
    state.momentum.lerp_(gradient, 1.0 - momentum)
    nesterov = gradient.lerp(state.momentum, momentum)
    direction = polar_express_from_nesterov(nesterov)

    reduce_dim = -1 if direction.shape[-2] >= direction.shape[-1] else -2
    mean_square = direction.float().square().mean(dim=reduce_dim, keepdim=True)
    reduced_size = direction.shape[reduce_dim]
    old_norm = (
        mean_square.sum(dim=(-2, -1), keepdim=True) * reduced_size
    ).sqrt()
    state.second_momentum.lerp_(mean_square, 1.0 - beta2)
    inv_root = state.second_momentum.clamp_min(1e-10).rsqrt()
    new_norm = (
        (mean_square * reduced_size * inv_root.square())
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
    )
    scale = inv_root * (old_norm / new_norm.clamp_min(1e-10))
    return direction * scale.to(direction.dtype)


def native_adam_delta(
    parameter: Tensor,
    gradient: Tensor,
    state: NativeAdamState,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> Tensor:
    advance_native_adam_state(
        gradient, state, beta1=beta1, beta2=beta2
    )
    return native_adam_delta_from_state(
        parameter,
        state,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
    )


def advance_native_adam_state(
    gradient: Tensor,
    state: NativeAdamState,
    *,
    beta1: float,
    beta2: float,
) -> None:
    state.step += 1
    state.exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    state.exp_avg_sq.mul_(beta2).addcmul_(
        gradient, gradient, value=1.0 - beta2
    )


def native_adam_delta_from_state(
    parameter: Tensor,
    state: NativeAdamState,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> Tensor:
    bias1 = 1.0 - beta1**state.step
    bias2 = 1.0 - beta2**state.step
    step_size = lr * (bias2**0.5 / bias1)
    update = state.exp_avg / (state.exp_avg_sq.sqrt() + eps) * step_size
    mask = update * parameter > 0
    update = update + parameter * mask * (lr * lr * weight_decay)
    return -update


def apply_cautious_precise_(
    parameter: Tensor,
    mantissa: Tensor,
    direction: Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> None:
    parameter_bits = parameter.view(torch.uint16)
    precise_raw = (
        parameter_bits.to(torch.int32) << 16
    ) | mantissa.to(torch.int32)
    precise = precise_raw.view(torch.float32)
    direction = direction.float()
    mask = direction * precise >= 0
    precise.copy_(
        precise
        - precise * mask * (weight_decay * lr)
        - direction * lr
    )
    parameter_bits.copy_((precise_raw >> 16).to(torch.uint16))
    mantissa.copy_(precise_raw.to(torch.uint16))


def normuon_lr_multiplier(shape: tuple[int, ...], *, mlp_down: bool) -> float:
    multiplier = max(1.0, shape[-2] / shape[-1]) ** 0.5
    return multiplier * (2.0 if mlp_down else 1.0)
