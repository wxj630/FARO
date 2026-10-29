import ast
import hashlib
from pathlib import Path

import pytest
import torch

from FARO.normuon_backend import (
    NativeAdamState,
    NorMuonState,
    apply_cautious_precise_,
    native_adam_delta,
    normuon_direction,
    normuon_lr_multiplier,
    polar_express_from_nesterov,
)


COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _polar_reference(nesterov: torch.Tensor) -> torch.Tensor:
    result = nesterov.bfloat16()
    result = result / (
        result.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6
    )
    tall = result.shape[-2] > result.shape[-1]
    for a, b, c in COEFFICIENTS:
        gram = result.mT @ result if tall else result @ result.mT
        polynomial = b * gram + c * (gram @ gram)
        result = a * result + (
            result @ polynomial if tall else polynomial @ result
        )
    return result


def _native_cuda_polar_reference(nesterov: torch.Tensor) -> torch.Tensor:
    from triton_kernels import XXT, XTX, ba_plus_cAA

    result = nesterov.bfloat16()
    result = result / (
        result.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6
    )
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
        for a, b, c in COEFFICIENTS:
            XTX(result, out=gram)
            ba_plus_cAA(gram, alpha=c, beta=b, out=polynomial)
            if split:
                torch.bmm(result, polynomial, out=output)
                output.add_(result, alpha=a)
            else:
                torch.baddbmm(
                    result, result, polynomial, beta=a, out=output
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
        for a, b, c in COEFFICIENTS:
            XXT(result, out=gram)
            ba_plus_cAA(gram, alpha=c, beta=b, out=polynomial)
            torch.baddbmm(
                result, polynomial, result, beta=a, out=output
            )
            result, output = output, result
    return result


def _variance_reference(
    direction: torch.Tensor, second: torch.Tensor, beta2: float
) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dim = -1 if direction.shape[-2] >= direction.shape[-1] else -2
    mean_square = direction.float().square().mean(dim=reduce_dim, keepdim=True)
    reduced_size = direction.shape[reduce_dim]
    old_norm = (
        mean_square.sum(dim=(-2, -1), keepdim=True) * reduced_size
    ).sqrt()
    expected_second = second.lerp(mean_square, 1 - beta2)
    inv_root = expected_second.clamp_min(1e-10).rsqrt()
    new_norm = (
        (mean_square * reduced_size * inv_root.square())
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
    )
    scale = inv_root * (old_norm / new_norm.clamp_min(1e-10))
    return direction * scale.to(direction.dtype), expected_second


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (2, 5, 3)])
def test_polar_express_matches_native_coefficient_sequence(shape: tuple[int, ...]) -> None:
    values = torch.linspace(-0.7, 0.9, steps=torch.tensor(shape).prod().item())
    nesterov = values.reshape(shape).float()

    actual = polar_express_from_nesterov(nesterov)
    expected = _polar_reference(nesterov)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(2, 256, 768), (2, 3072, 768)])
def test_cuda_polar_express_is_bitwise_native(shape: tuple[int, ...]) -> None:
    values = torch.randn(shape, device="cuda", dtype=torch.float32)

    actual = polar_express_from_nesterov(values)
    expected = _native_cuda_polar_reference(values)

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


def test_normuon_direction_updates_momentum_and_reduced_second_state() -> None:
    gradient = torch.tensor(
        [[0.2, -0.1, 0.4], [-0.3, 0.7, 0.1], [0.5, -0.2, -0.4], [0.1, 0.3, -0.6]],
        dtype=torch.float32,
    )
    state = NorMuonState.zeros_like(gradient)
    state.momentum.copy_(torch.linspace(-0.2, 0.3, gradient.numel()).reshape_as(gradient))
    old_momentum = state.momentum.clone()
    old_second = state.second_momentum.clone()
    momentum = 0.9
    beta2 = 0.8

    actual = normuon_direction(
        gradient, state, momentum=momentum, beta2=beta2
    )

    expected_momentum = old_momentum.lerp(gradient, 1 - momentum)
    nesterov = gradient.lerp(expected_momentum, momentum)
    polar = _polar_reference(nesterov)
    expected, expected_second = _variance_reference(polar, old_second, beta2)
    torch.testing.assert_close(state.momentum, expected_momentum)
    torch.testing.assert_close(state.second_momentum, expected_second)
    torch.testing.assert_close(actual, expected)
    assert state.mantissa.dtype == torch.uint16
    assert state.mantissa.shape == gradient.shape


def test_native_adam_delta_matches_bias_correction_and_cautious_weight_decay() -> None:
    parameter = torch.tensor([0.5, -0.25, 0.75], dtype=torch.float32)
    gradient = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float32)
    state = NativeAdamState.zeros_like(parameter)
    beta1, beta2, lr, weight_decay, eps = 0.5, 0.8, 0.02, 0.3, 1e-10

    delta = native_adam_delta(
        parameter,
        gradient,
        state,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
    )

    exp_avg = gradient * (1 - beta1)
    exp_avg_sq = gradient.square() * (1 - beta2)
    step_size = lr * ((1 - beta2) ** 0.5 / (1 - beta1))
    update = exp_avg / (exp_avg_sq.sqrt() + eps) * step_size
    mask = update * parameter > 0
    update = update + parameter * mask * (lr * lr * weight_decay)
    torch.testing.assert_close(delta, -update)
    torch.testing.assert_close(state.exp_avg, exp_avg)
    torch.testing.assert_close(state.exp_avg_sq, exp_avg_sq)
    assert state.step == 1


def test_cautious_precise_write_matches_native_uint16_reconstruction() -> None:
    parameter = torch.tensor([0.5, -0.25, 1.0], dtype=torch.bfloat16)
    mantissa = torch.tensor([0x1234, 0xABCD, 0x0001], dtype=torch.uint16)
    direction = torch.tensor([0.2, 0.4, -0.3], dtype=torch.float32)
    lr, weight_decay = 0.03, 0.4
    expected_parameter = parameter.clone()
    expected_mantissa = mantissa.clone()
    raw = (
        expected_parameter.view(torch.uint16).to(torch.int32) << 16
    ) | expected_mantissa.to(torch.int32)
    precise = raw.view(torch.float32)
    mask = direction * precise >= 0
    precise.copy_(
        precise
        - precise * mask * (weight_decay * lr)
        - direction * lr
    )
    expected_parameter.view(torch.uint16).copy_((raw >> 16).to(torch.uint16))
    expected_mantissa.copy_(raw.to(torch.uint16))

    apply_cautious_precise_(
        parameter,
        mantissa,
        direction,
        lr=lr,
        weight_decay=weight_decay,
    )

    assert torch.equal(parameter.view(torch.uint16), expected_parameter.view(torch.uint16))
    assert torch.equal(mantissa, expected_mantissa)


def test_normuon_lr_multiplier_matches_shape_and_mlp_down_rules() -> None:
    assert normuon_lr_multiplier((6, 3), mlp_down=False) == pytest.approx(2**0.5)
    assert normuon_lr_multiplier((3, 6), mlp_down=False) == pytest.approx(1.0)
    assert normuon_lr_multiplier((6, 3), mlp_down=True) == pytest.approx(2 * 2**0.5)


def test_native_normuon_source_digest_is_unchanged() -> None:
    source = (Path(__file__).parents[2] / "train_gpt.py").read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "NorMuonAndAdam"
    )
    class_source = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"

    assert hashlib.sha256(class_source.encode()).hexdigest() == (
        "f61a857d53eb41cec91bfa2621cc97010f6542175c5d8c177d3be16c35be8f4d"
    )
