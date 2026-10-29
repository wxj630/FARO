from copy import deepcopy

import torch

from FARO.geometry import (
    CoordinateState,
    GeometryConfig,
    RowColumnState,
    SpectralState,
    coordinate_candidate,
    newton_schulz5,
    row_column_candidate,
    spectral_candidate,
    transpose_candidate_state,
)


def test_coordinate_candidate_matches_torch_adamw() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    parameter = initial.clone()
    reference = torch.nn.Parameter(initial.clone())
    config = GeometryConfig(
        lr=0.01,
        beta1=0.8,
        beta2=0.9,
        eps=1e-8,
        weight_decay=0.05,
    )
    state = CoordinateState.zeros_like(parameter)
    optimizer = torch.optim.AdamW(
        [reference],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    for gradient in (
        torch.tensor([[0.5, -0.2], [0.1, 0.3]]),
        torch.tensor([[-0.1, 0.4], [0.2, -0.5]]),
        torch.tensor([[0.2, 0.2], [-0.3, 0.1]]),
    ):
        update = coordinate_candidate(parameter, gradient, state, config)
        parameter.add_(update.delta)
        reference.grad = gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(parameter, reference.detach(), rtol=1e-6, atol=1e-7)


def test_row_candidate_uses_token_axis_state_and_rare_guard() -> None:
    parameter = torch.ones(4, 3)
    gradient = torch.tensor(
        [[8.0, 8.0, 8.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1]]
    )
    config = GeometryConfig(
        lr=0.1,
        beta1=0.0,
        beta2=0.0,
        eps=1e-8,
        weight_decay=0.0,
        rare_token_threshold=8,
        axis_norm_cap=2.0,
    )
    state = RowColumnState.zeros_like(parameter, token_axis=0)

    update = row_column_candidate(
        parameter,
        gradient,
        state,
        config,
        token_axis=0,
        access_mask=torch.tensor([True, False, True, True]),
    )

    assert state.axis_second.shape == (4,)
    assert state.access_count.tolist() == [1, 0, 1, 1]
    assert torch.count_nonzero(update.delta[1]) == 0
    row_norms = update.delta.norm(dim=1)
    active = row_norms[row_norms > 0]
    assert active.max() <= 2.0 * active.median() + 1e-6
    assert torch.isfinite(update.delta).all()


def test_row_candidate_supports_token_columns() -> None:
    parameter = torch.ones(3, 5)
    gradient = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    state = RowColumnState.zeros_like(parameter, token_axis=1)

    update = row_column_candidate(
        parameter,
        gradient,
        state,
        GeometryConfig(lr=0.01),
        token_axis=1,
    )

    assert state.axis_second.shape == (5,)
    assert update.delta.shape == parameter.shape
    assert update.max_axis_uwr.ndim == 0


def _sparse_row_candidate(vocab_size: int, *, active_axis_second: bool):
    parameter = torch.ones(vocab_size, 3)
    gradient = torch.zeros_like(parameter)
    gradient[0].fill_(1.0)
    gradient[1].fill_(0.01)
    state = RowColumnState.zeros_like(parameter, token_axis=0)
    state.access_count[0] = 7
    update = row_column_candidate(
        parameter,
        gradient,
        state,
        GeometryConfig(
            lr=0.1,
            beta1=0.0,
            beta2=0.0,
            eps=1e-8,
            weight_decay=0.0,
            rare_token_threshold=8,
            axis_norm_cap=float("inf"),
            active_axis_second=active_axis_second,
        ),
        token_axis=0,
    )
    return state, update


def test_active_axis_second_is_invariant_to_inactive_vocab_padding() -> None:
    small_state, small_update = _sparse_row_candidate(
        2, active_axis_second=True
    )
    large_state, large_update = _sparse_row_candidate(
        10_000, active_axis_second=True
    )

    expected = torch.tensor((1.0 + 0.01**2) / 2)
    torch.testing.assert_close(small_state.global_second, expected)
    torch.testing.assert_close(large_state.global_second, expected)
    torch.testing.assert_close(
        small_update.delta[:2], large_update.delta[:2]
    )


def test_active_axis_second_decays_on_empty_active_mask() -> None:
    parameter = torch.ones(5, 3)
    state = RowColumnState.zeros_like(parameter, token_axis=0)
    state.global_second.fill_(4.0)

    update = row_column_candidate(
        parameter,
        torch.zeros_like(parameter),
        state,
        GeometryConfig(
            lr=0.1,
            beta1=0.0,
            beta2=0.75,
            weight_decay=0.0,
            active_axis_second=True,
        ),
        token_axis=0,
        access_mask=torch.zeros(5, dtype=torch.bool),
    )

    torch.testing.assert_close(state.global_second, torch.tensor(3.0))
    assert torch.count_nonzero(update.delta) == 0
    assert torch.isfinite(update.delta).all()


def test_default_row_global_second_keeps_full_matrix_mean() -> None:
    state, _ = _sparse_row_candidate(100, active_axis_second=False)

    torch.testing.assert_close(
        state.global_second, torch.tensor((1.0 + 0.01**2) / 100)
    )


def test_newton_schulz_preserves_shape_and_equalizes_singular_values() -> None:
    torch.manual_seed(3)
    matrix = torch.randn(7, 3)

    result = newton_schulz5(matrix)
    singular_values = torch.linalg.svdvals(result.float())

    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()
    output_condition = singular_values.max() / singular_values.min()
    input_condition = torch.linalg.cond(matrix)
    assert output_condition < 1.70
    assert output_condition < input_condition / 3


def test_muon_and_faro_share_bitwise_spectral_candidate() -> None:
    torch.manual_seed(4)
    weight = torch.randn(7, 3)
    gradient = torch.randn_like(weight)
    state = SpectralState.zeros_like(weight)
    config = GeometryConfig(lr=0.023, momentum=0.9, weight_decay=0.01)

    muon = spectral_candidate(weight, gradient, deepcopy(state), config)
    faro = spectral_candidate(weight, gradient, deepcopy(state), config)

    torch.testing.assert_close(muon.delta, faro.delta, rtol=0, atol=0)


def test_candidate_state_transpose_migrates_tied_layout() -> None:
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    coordinate = CoordinateState(
        step=3,
        exp_avg=matrix.clone(),
        exp_avg_sq=matrix.square(),
    )
    row = RowColumnState(
        step=2,
        momentum=matrix.clone(),
        axis_second=torch.arange(4, dtype=torch.float32),
        access_count=torch.arange(4, dtype=torch.int64),
        global_second=torch.tensor(2.0),
        token_axis=1,
    )
    spectral = SpectralState(momentum=matrix.clone())

    coordinate_t = transpose_candidate_state(coordinate)
    row_t = transpose_candidate_state(row)
    spectral_t = transpose_candidate_state(spectral)

    torch.testing.assert_close(coordinate_t.exp_avg, matrix.T)
    torch.testing.assert_close(coordinate_t.exp_avg_sq, matrix.square().T)
    torch.testing.assert_close(row_t.momentum, matrix.T)
    torch.testing.assert_close(row_t.axis_second, row.axis_second)
    torch.testing.assert_close(row_t.access_count, row.access_count)
    assert row_t.token_axis == 0
    torch.testing.assert_close(spectral_t.momentum, matrix.T)
