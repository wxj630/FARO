from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

import FARO.optimizer as optimizer_module
from FARO.descriptors import BlockDescriptor
from FARO.geometry import (
    CandidateUpdate,
    CoordinateState,
    GeometryConfig,
    RowColumnState,
    SpectralState,
    coordinate_candidate,
    row_column_candidate,
)
from FARO.normuon_backend import (
    NativeAdamState,
    NorMuonState,
    apply_cautious_precise_,
    native_adam_delta,
    normuon_direction,
    normuon_lr_multiplier,
)
from FARO.optimizer import (
    FAROOptimizer,
    FixedGeometryOptimizer,
    MicroBatchSet,
    MicroGradientCollector,
    SparseMicroRows,
    SparseMicroSet,
    densify_sparse_micro_rows,
    sparse_micro_return,
    _SPARSE_MICRO_LABELS,
)
from FARO.router import CandidateObservation, FARORouter, RouterConfig


def _label(parameter: nn.Parameter, label: str) -> nn.Parameter:
    parameter.label = label
    return parameter


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = _label(nn.Parameter(torch.randn(4, 3)), "dense")
        self.embed = _label(nn.Parameter(torch.randn(7, 3)), "embed")
        self.lm_head = _label(
            nn.Parameter(self.embed.detach().T.clone()), "lm_head"
        )
        self.vector = _label(nn.Parameter(torch.randn(3)), "vector")


class SparseLookupModel(nn.Module):
    """One row-sparse lookup table plus one dense block, for collector tests."""

    def __init__(self) -> None:
        super().__init__()
        self.table = _label(
            nn.Parameter(torch.zeros(8, 3, dtype=torch.bfloat16)), "bigram_embed"
        )
        self.dense = _label(nn.Parameter(torch.randn(4, 3)), "dense")


def _lookup_increments() -> list[dict[str, torch.Tensor]]:
    increments = []
    for index in range(8):
        increment = torch.zeros(8, 3, dtype=torch.bfloat16)
        increment[index] = float(index + 1)
        increment[(index + 3) % 8] = -0.5 * (index + 1)
        increments.append({"bigram_embed": increment})
    return increments


def _run_micro_gradients(
    model: nn.Module,
    collector: MicroGradientCollector,
    increments: list[dict[str, torch.Tensor]],
) -> None:
    collector.start_step()
    parameters = {parameter.label: parameter for parameter in model.parameters()}
    for index, micro in enumerate(increments):
        for label, increment in micro.items():
            parameter = parameters[label]
            if parameter.grad is None:
                parameter.grad = increment.clone()
            else:
                parameter.grad.add_(increment)
        collector.capture(index, effective_tokens=10)


class TinyBankModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bank = _label(nn.Parameter(torch.randn(2, 4, 3)), "bank")


def _bank_descriptors() -> tuple[BlockDescriptor, ...]:
    return tuple(
        BlockDescriptor(
            label=f"bank.{index}",
            parameter_label="bank",
            shape=(4, 3),
            matrix_shape=(4, 3),
            tensor_index=(index,),
            input_axis=1,
            output_axis=0,
            access_pattern="dense",
            token_axis=None,
            head_partition=None,
            shared_weight=None,
            update_frequency="every_step",
            storage_transposed=False,
            structure_class="attention",
        )
        for index in range(2)
    )


def _descriptor(
    label: str,
    shape: tuple[int, ...],
    *,
    access_pattern: str,
    token_axis: int | None = None,
    shared_weight: str | None = None,
    update_frequency: str = "every_step",
) -> BlockDescriptor:
    matrix = len(shape) == 2
    return BlockDescriptor(
        label=label,
        parameter_label=label,
        shape=shape,
        matrix_shape=shape if matrix else None,
        tensor_index=(),
        input_axis=1 if matrix else None,
        output_axis=0 if matrix else None,
        access_pattern=access_pattern,
        token_axis=token_axis,
        head_partition=None,
        shared_weight=shared_weight,
        update_frequency=update_frequency,
        storage_transposed=access_pattern == "token_output",
        structure_class=(
            "mlp"
            if access_pattern == "dense"
            else "embedding"
            if access_pattern == "token_lookup"
            else "lm_head"
            if access_pattern == "token_output"
            else "coordinate"
        ),
    )


def _descriptors() -> tuple[BlockDescriptor, ...]:
    return (
        _descriptor("dense", (4, 3), access_pattern="dense"),
        _descriptor(
            "embed",
            (7, 3),
            access_pattern="token_lookup",
            token_axis=0,
            shared_weight="lm_head",
            update_frequency="odd_step",
        ),
        _descriptor(
            "lm_head",
            (3, 7),
            access_pattern="token_output",
            token_axis=1,
            shared_weight="embed",
            update_frequency="odd_step",
        ),
        _descriptor("vector", (3,), access_pattern="coordinate"),
    )


def _coordinate_defaults() -> dict[str, float]:
    return {
        "lr": 0.01,
        "matrix_lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.0,
    }


def _muon_defaults() -> dict[str, float]:
    return {"lr": 0.023, "momentum": 0.9, "weight_decay": 0.0}


def _accumulate_micro_gradients(
    model: TinyModel,
    collector: MicroGradientCollector,
    increments: list[dict[str, torch.Tensor]],
) -> None:
    collector.start_step()
    parameters = {parameter.label: parameter for parameter in model.parameters()}
    for index, micro in enumerate(increments):
        for label, increment in micro.items():
            parameter = parameters[label]
            if parameter.grad is None:
                parameter.grad = increment.clone()
            else:
                parameter.grad.add_(increment)
        collector.capture(index, effective_tokens=10)


def _accumulate_optimizer_gradients(
    model: TinyModel,
    optimizer: FAROOptimizer,
    *,
    step: int,
    increments: list[dict[str, torch.Tensor]],
) -> None:
    optimizer.start_step(step)
    parameters = {parameter.label: parameter for parameter in model.parameters()}
    for index, micro in enumerate(increments):
        for label, increment in micro.items():
            parameter = parameters[label]
            if parameter.grad is None:
                parameter.grad = increment.clone()
            else:
                parameter.grad.add_(increment)
        optimizer.capture_micro_gradient(index, effective_tokens=10)


def test_micro_gradient_sum_matches_accumulated_gradient() -> None:
    model = TinyModel()
    collector = MicroGradientCollector(model.named_parameters(), micro_count=8)
    increments = [
        {"dense": torch.full_like(model.dense, float(index + 1))}
        for index in range(8)
    ]

    _accumulate_micro_gradients(model, collector, increments)
    prepared = collector.prepare({"dense": "every_step"}, step=0)["dense"]

    assert prepared is not None
    assert all(micro.dtype == torch.bfloat16 for micro in prepared.gradients)
    recovered = torch.stack([micro.float() for micro in prepared.gradients]).sum(0)
    torch.testing.assert_close(recovered, model.dense.grad.float(), rtol=0, atol=0.25)


def test_odd_step_pairs_equal_micro_indices_across_two_steps() -> None:
    model = TinyModel()
    collector = MicroGradientCollector(model.named_parameters(), micro_count=8)
    first = [
        {"vector": torch.full_like(model.vector, float(index + 1))}
        for index in range(8)
    ]
    second = [
        {"vector": torch.full_like(model.vector, float(10 + index))}
        for index in range(8)
    ]

    _accumulate_micro_gradients(model, collector, first)
    assert collector.prepare({"vector": "odd_step"}, step=0)["vector"] is None
    collector.finish_step()
    _accumulate_micro_gradients(model, collector, second)
    prepared = collector.prepare({"vector": "odd_step"}, step=1)["vector"]

    assert prepared is not None
    for index, gradient in enumerate(prepared.gradients):
        expected = first[index]["vector"] + second[index]["vector"]
        torch.testing.assert_close(gradient.float(), expected)
    recovered = torch.stack([micro.float() for micro in prepared.gradients]).sum(0)
    torch.testing.assert_close(recovered, model.vector.grad.float())


def test_sparse_micro_labels_target_giant_lookup_tables() -> None:
    assert _SPARSE_MICRO_LABELS == frozenset({"bigram_embed", "value_embeds"})


def test_sparse_micro_storage_drops_unaccessed_rows_and_matches_dense() -> None:
    dense_model = SparseLookupModel()
    sparse_model = SparseLookupModel()
    sparse_model.load_state_dict(dense_model.state_dict())
    increments = _lookup_increments()

    dense_collector = MicroGradientCollector(
        dense_model.named_parameters(), micro_count=8
    )
    sparse_collector = MicroGradientCollector(
        sparse_model.named_parameters(),
        micro_count=8,
        sparse_labels=frozenset({"bigram_embed"}),
    )
    _run_micro_gradients(dense_model, dense_collector, increments)
    _run_micro_gradients(sparse_model, sparse_collector, increments)

    dense_prepared = dense_collector.prepare(
        {"bigram_embed": "every_step"}, step=0
    )["bigram_embed"]
    sparse_prepared = sparse_collector.prepare(
        {"bigram_embed": "every_step"}, step=0
    )["bigram_embed"]

    assert isinstance(dense_prepared, MicroBatchSet)
    assert isinstance(sparse_prepared, SparseMicroSet)
    expected_rows = [
        torch.nonzero(increment["bigram_embed"].any(dim=1)).squeeze(1)
        for increment in increments
    ]
    for index, (dense_micro, sparse_micro) in enumerate(
        zip(dense_prepared.gradients, sparse_prepared.micros, strict=True)
    ):
        torch.testing.assert_close(
            densify_sparse_micro_rows(sparse_micro, size=8), dense_micro, rtol=0, atol=0
        )
        torch.testing.assert_close(sparse_micro.rows, expected_rows[index])
        assert sparse_micro.rows.numel() < 8


def test_sparse_odd_step_merge_matches_dense_merge() -> None:
    dense_model = SparseLookupModel()
    sparse_model = SparseLookupModel()
    sparse_model.load_state_dict(dense_model.state_dict())
    first = _lookup_increments()
    second = []
    for index in range(8):
        increment = torch.zeros(8, 3, dtype=torch.bfloat16)
        increment[index] = 0.25 * float(index + 1)
        increment[(index + 5) % 8] = 2.0
        second.append({"bigram_embed": increment})

    merged: dict[str, MicroBatchSet | SparseMicroSet] = {}
    for name, model, collector in (
        (
            "dense",
            dense_model,
            MicroGradientCollector(dense_model.named_parameters(), micro_count=8),
        ),
        (
            "sparse",
            sparse_model,
            MicroGradientCollector(
                sparse_model.named_parameters(),
                micro_count=8,
                sparse_labels=frozenset({"bigram_embed"}),
            ),
        ),
    ):
        _run_micro_gradients(model, collector, first)
        assert collector.prepare({"bigram_embed": "odd_step"}, step=0)["bigram_embed"] is None
        collector.finish_step()
        _run_micro_gradients(model, collector, second)
        result = collector.prepare({"bigram_embed": "odd_step"}, step=1)["bigram_embed"]
        assert result is not None
        merged[name] = result

    dense_merged = merged["dense"]
    sparse_merged = merged["sparse"]
    assert isinstance(dense_merged, MicroBatchSet)
    assert isinstance(sparse_merged, SparseMicroSet)
    for dense_micro, sparse_micro in zip(
        dense_merged.gradients, sparse_merged.micros, strict=True
    ):
        torch.testing.assert_close(
            densify_sparse_micro_rows(sparse_micro, size=8), dense_micro, rtol=0, atol=0
        )
    assert sparse_merged.effective_tokens == dense_merged.effective_tokens


def test_sparse_micro_return_matches_dense_formula() -> None:
    rows = torch.tensor([1, 3, 6])
    values = torch.randn(3, 4).to(torch.bfloat16)
    delta = torch.randn(8, 4)
    micro = SparseMicroRows(rows=rows, values=values)

    dense_micro = torch.zeros(8, 4, dtype=torch.bfloat16)
    dense_micro.index_copy_(0, rows, values)
    expected = -(dense_micro.float() * delta).sum() / 10 * 8

    torch.testing.assert_close(
        sparse_micro_return(micro, delta, tokens=10, micro_count=8),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_faro_optimizer_uses_sparse_collection_for_lookup_tables() -> None:
    model = TinyModel()
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )

    assert optimizer.collector.sparse_labels == _SPARSE_MICRO_LABELS


def test_sparse_micro_labels_survive_optimizer_reset() -> None:
    model = TinyModel()
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    labels = frozenset({"value_embeds"})
    optimizer.sparse_micro_labels = labels
    optimizer.reset()

    assert optimizer.collector.sparse_labels == labels


def test_fixed_adam_uses_coordinate_for_all_blocks() -> None:
    model = TinyModel()
    optimizer = FixedGeometryOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        variant="adam",
        world_size=1,
    )
    before = {parameter.label: parameter.detach().clone() for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    result = optimizer.step(
        step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert result.modes == {descriptor.label: "coordinate" for descriptor in _descriptors()}
    assert all(
        not torch.equal(parameter, before[parameter.label])
        for parameter in model.parameters()
        if parameter.label != "embed"
    )


def test_fixed_muon_uses_spectral_for_dense_blocks() -> None:
    model = TinyModel()
    optimizer = FixedGeometryOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        variant="muon",
        world_size=1,
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    result = optimizer.step(
        step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert result.modes["dense"] == "spectral"
    assert result.modes["vector"] == "coordinate"


def test_faro_fixed_dense_matches_direct_normuon_backend() -> None:
    torch.manual_seed(9)
    faro_model = TinyModel()
    faro_model.dense = _label(
        nn.Parameter(faro_model.dense.detach().bfloat16()), "dense"
    )
    descriptor = _descriptors()[0]
    faro = FAROOptimizer(
        [("dense", faro_model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(
            RouterConfig(
                fixed_geometry=True,
                unit_budget=True,
                fixed_token_coordinate=True,
            )
        ),
        world_size=1,
    )
    expected_parameter = faro_model.dense.detach().clone()
    expected_state = NorMuonState.zeros_like(expected_parameter)
    increments = [torch.randn_like(faro_model.dense) / 8 for _ in range(8)]
    _accumulate_micro_gradients(
        faro_model,
        faro.collector,
        [{"dense": increment} for increment in increments],
    )
    gradient = faro_model.dense.grad.detach().clone()
    direction = normuon_direction(
        gradient,
        expected_state,
        momentum=0.9,
        beta2=0.9,
    )
    lr = 0.023 * normuon_lr_multiplier(tuple(expected_parameter.shape), mlp_down=False)
    apply_cautious_precise_(
        expected_parameter,
        expected_state.mantissa,
        direction,
        lr=lr,
        weight_decay=0.0,
    )

    faro.step(step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert torch.equal(faro_model.dense.view(torch.uint16), expected_parameter.view(torch.uint16))
    actual_state = faro.states["dense"]["spectral"]
    assert isinstance(actual_state, NorMuonState)
    torch.testing.assert_close(actual_state.momentum, expected_state.momentum)
    torch.testing.assert_close(actual_state.second_momentum, expected_state.second_momentum)
    assert torch.equal(actual_state.mantissa, expected_state.mantissa)
    assert faro.last_decisions["dense"].mode == "spectral"


def test_faro_spectral_budget_scales_native_direction_before_precise_update() -> None:
    torch.manual_seed(10)
    model = TinyModel()
    model.dense = _label(
        nn.Parameter(model.dense.detach().bfloat16()), "dense"
    )
    descriptor = _descriptors()[0]
    router = FARORouter(RouterConfig(fixed_geometry=True))
    original_route = router.route

    def half_budget(step, descriptors, observations):
        decisions = original_route(step, descriptors, observations)
        decision = decisions[descriptor.label]
        decisions[descriptor.label] = replace(decision, budget_scale=0.5)
        return decisions

    router.route = half_budget
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    expected_parameter = model.dense.detach().clone()
    expected_state = NorMuonState.zeros_like(expected_parameter)
    increments = [torch.randn_like(model.dense) / 8 for _ in range(8)]
    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"dense": increment} for increment in increments],
    )
    direction = normuon_direction(
        model.dense.grad.detach(),
        expected_state,
        momentum=0.9,
        beta2=0.9,
    )
    lr = 0.5 * 0.023 * normuon_lr_multiplier(
        tuple(expected_parameter.shape), mlp_down=False
    )
    apply_cautious_precise_(
        expected_parameter,
        expected_state.mantissa,
        direction,
        lr=lr,
        weight_decay=0.0,
    )

    optimizer.step(step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert torch.equal(model.dense.view(torch.uint16), expected_parameter.view(torch.uint16))
    actual_state = optimizer.states["dense"]["spectral"]
    assert isinstance(actual_state, NorMuonState)
    assert torch.equal(actual_state.mantissa, expected_state.mantissa)


def test_faro_coordinate_matches_native_adam_and_skips_even_step() -> None:
    torch.manual_seed(11)
    model = TinyModel()
    descriptor = replace(_descriptors()[3], update_frequency="odd_step")
    optimizer = FAROOptimizer(
        [("vector", model.vector)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(fixed_geometry=True, unit_budget=True)),
        world_size=1,
    )
    expected_parameter = model.vector.detach().clone()
    expected_state = NativeAdamState.zeros_like(expected_parameter)
    first = [torch.randn_like(model.vector) / 8 for _ in range(8)]
    second = [torch.randn_like(model.vector) / 8 for _ in range(8)]

    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"vector": increment} for increment in first],
    )
    optimizer.step(step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=False)
    assert torch.equal(model.vector, expected_parameter)
    assert optimizer.states["vector"]["coordinate"].step == 0

    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"vector": increment} for increment in second],
    )
    gradient = torch.stack([*first, *second]).sum(0)
    expected_parameter.add_(
        native_adam_delta(
            expected_parameter,
            gradient,
            expected_state,
            beta1=0.9,
            beta2=0.99,
            lr=0.01,
            weight_decay=0.0,
            eps=1e-8,
        )
    )
    optimizer.step(step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    torch.testing.assert_close(model.vector, expected_parameter, rtol=0, atol=0)
    actual_state = optimizer.states["vector"]["coordinate"]
    assert isinstance(actual_state, NativeAdamState)
    assert actual_state.step == expected_state.step
    torch.testing.assert_close(actual_state.exp_avg, expected_state.exp_avg)
    torch.testing.assert_close(actual_state.exp_avg_sq, expected_state.exp_avg_sq)


def test_faro_uses_normuon_states_and_preserves_classic_muon_state() -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    faro = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    classic = FixedGeometryOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        variant="muon",
        world_size=1,
    )

    assert isinstance(faro.states["dense"]["coordinate"], NativeAdamState)
    assert isinstance(faro.states["dense"]["spectral"], NorMuonState)
    assert isinstance(classic.states["dense"]["spectral"], SpectralState)


def test_faro_normuon_mlp_down_receives_two_x_lr() -> None:
    model = TinyModel()
    up = _descriptors()[0]
    down = replace(up, label="dense_down", storage_transposed=True)
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(up,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )

    up_config = optimizer._config("spectral", up, 1.0, 0.9, False)
    down_config = optimizer._config("spectral", down, 1.0, 0.9, False)

    assert down_config.lr == pytest.approx(2 * up_config.lr)


def test_candidate_returns_restore_micro_accumulation_scale(monkeypatch) -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    router = FARORouter(RouterConfig())
    captured_observations = {}
    candidate_deltas = {}
    original_route = router.route
    original_candidate = optimizer_module._faro_candidate

    def capture_route(step, descriptors, observations):
        captured_observations.update(observations)
        return original_route(step, descriptors, observations)

    def capture_candidate(mode, *args, **kwargs):
        result = original_candidate(mode, *args, **kwargs)
        update = result[0]
        candidate_deltas[mode] = update.delta.detach().clone()
        return result

    router.route = capture_route
    monkeypatch.setattr(optimizer_module, "_faro_candidate", capture_candidate)
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    increments = [
        torch.full_like(model.dense, float(index + 1) / 8)
        for index in range(8)
    ]
    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"dense": increment} for increment in increments],
    )

    optimizer.step(
        step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    spectral_delta = candidate_deltas["spectral"].float()
    expected = tuple(
        float(
            -(
                increment.to(torch.bfloat16).float() * spectral_delta
            ).sum()
            / 10
            * optimizer.collector.micro_count
        )
        for increment in increments
    )
    assert captured_observations["dense"]["spectral"].returns == pytest.approx(
        expected
    )


def test_faro_advances_all_shadow_states_and_releases_candidates() -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    increments = [torch.ones_like(model.dense) / 8 for _ in range(8)]
    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"dense": increment} for increment in increments],
    )

    optimizer.step(step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert optimizer.states["dense"]["coordinate"].step == 1
    assert optimizer.states["dense"]["row_column"].step == 1
    assert torch.count_nonzero(
        optimizer.states["dense"]["spectral"].momentum
    )
    assert optimizer.collector.is_empty
    assert not hasattr(optimizer, "candidate_deltas")


def test_nonroute_step_advances_shadows_but_materializes_only_incumbent(
    monkeypatch,
) -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(route_interval=8)),
        world_size=1,
    )
    increments = [{"dense": torch.randn_like(model.dense) / 8} for _ in range(8)]
    _accumulate_optimizer_gradients(
        model, optimizer, step=0, increments=increments
    )
    optimizer.step(step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    before = deepcopy(optimizer.states["dense"])
    calls: list[str] = []
    original = optimizer_module._faro_candidate

    def counted(mode, *args, **kwargs):
        calls.append(mode)
        return original(mode, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_faro_candidate", counted)
    _accumulate_optimizer_gradients(
        model, optimizer, step=2, increments=increments
    )
    assert optimizer.collector._current == {}

    optimizer.step(step=2, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert calls == ["spectral"]
    assert optimizer.states["dense"]["coordinate"].step == before["coordinate"].step + 1
    assert optimizer.states["dense"]["row_column"].step == before["row_column"].step + 1
    assert not torch.equal(
        optimizer.states["dense"]["spectral"].momentum,
        before["spectral"].momentum,
    )


def test_route_step_materializes_every_legal_candidate(monkeypatch) -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(route_interval=8)),
        world_size=1,
    )
    calls: list[str] = []
    original = optimizer_module._faro_candidate

    def counted(mode, *args, **kwargs):
        calls.append(mode)
        return original(mode, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_faro_candidate", counted)
    increments = [{"dense": torch.randn_like(model.dense) / 8} for _ in range(8)]
    _accumulate_optimizer_gradients(
        model, optimizer, step=8, increments=increments
    )

    optimizer.step(step=8, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert calls == ["coordinate", "row_column", "spectral"]


def test_first_active_nonperiodic_block_collects_route_observations() -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(route_interval=8)),
        world_size=1,
    )

    optimizer.start_step(2)

    assert set(optimizer.collector._current) == {"dense"}


def test_dense_bank_advances_spectral_anchor_with_one_batched_call(
    monkeypatch,
) -> None:
    model = TinyBankModel()
    descriptors = _bank_descriptors()
    optimizer = FAROOptimizer(
        [("bank", model.bank)],
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(fixed_geometry=True, unit_budget=True)),
        world_size=1,
    )
    calls: list[tuple[int, ...]] = []
    gradient_pointers: list[int] = []
    original = optimizer_module.normuon_direction

    def counted(gradient, *args, **kwargs):
        calls.append(tuple(gradient.shape))
        gradient_pointers.append(gradient.data_ptr())
        return original(gradient, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "normuon_direction", counted)
    increments = [{"bank": torch.randn_like(model.bank) / 8} for _ in range(8)]
    optimizer.start_step(0)
    for index, micro in enumerate(increments):
        if model.bank.grad is None:
            model.bank.grad = micro["bank"].clone()
        else:
            model.bank.grad.add_(micro["bank"])
        optimizer.capture_micro_gradient(index, effective_tokens=10)
    full_gradient_pointer = model.bank.grad.data_ptr()

    optimizer.step(step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert calls == [(2, 4, 3)]
    assert gradient_pointers == [full_gradient_pointer]
    assert optimizer.spectral_banks["bank"].momentum.shape == (2, 4, 3)
    for index, descriptor in enumerate(descriptors):
        state = optimizer.states[descriptor.label]["spectral"]
        assert state.momentum.data_ptr() == optimizer.spectral_banks["bank"].momentum[index].data_ptr()


def test_probe_prediction_uses_same_first_micro_batch_as_realized_return() -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    router = FARORouter(RouterConfig())
    captured = {}
    original_route = router.route

    def capture_route(step, descriptors, observations):
        captured.update(observations)
        return original_route(step, descriptors, observations)

    router.route = capture_route
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    increments = [
        torch.full_like(model.dense, float(index + 1) / 8)
        for index in range(8)
    ]
    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"dense": increment} for increment in increments],
    )

    result = optimizer.step(
        step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    selected = result.decisions["dense"]
    expected = captured["dense"][selected.mode].returns[0] * selected.budget_scale
    assert result.predicted_return == expected
    assert result.predicted_return_by_structure == {"mlp": expected}


def test_row_column_candidate_matches_coordinate_uwr_during_collection() -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    router = FARORouter(RouterConfig(collect_steps=32))
    captured = {}
    original_route = router.route

    def capture_route(step, descriptors, observations):
        captured.update(observations)
        return original_route(step, descriptors, observations)

    router.route = capture_route
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    increments = [torch.randn_like(model.dense) / 8 for _ in range(8)]
    _accumulate_micro_gradients(
        model,
        optimizer.collector,
        [{"dense": increment} for increment in increments],
    )

    optimizer.step(
        step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert captured["dense"]["row_column"].uwr == pytest.approx(
        captured["dense"]["coordinate"].uwr
    )
    assert len(optimizer.row_column_scale_samples["dense"]) == 1


def test_active_axis_second_is_enabled_only_for_lookup_row_column() -> None:
    model = TinyModel()
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    dense, lookup, output, _ = _descriptors()

    assert optimizer._config(
        "row_column", lookup, 1.0, 0.9, False
    ).active_axis_second
    assert not optimizer._config(
        "row_column", output, 1.0, 0.9, False
    ).active_axis_second
    assert not optimizer._config(
        "row_column", dense, 1.0, 0.9, False
    ).active_axis_second
    assert not optimizer._config(
        "coordinate", lookup, 1.0, 0.9, False
    ).active_axis_second
    assert not optimizer._config(
        "spectral", lookup, 1.0, 0.9, False
    ).active_axis_second


def _counterfactual_optimizer(mode="coordinate", dtype=torch.float32, **options):
    descriptor = _descriptors()[0]
    parameter = _label(nn.Parameter(torch.ones(4, 3, dtype=dtype)), "dense")
    router = FARORouter(RouterConfig(counterfactual_probe_budget=True, **options))
    optimizer = FAROOptimizer(
        [("dense", parameter)], descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults={**_muon_defaults(), "weight_decay": 0.7},
        router=router, world_size=1,
    )
    route = router.initialize(descriptor)
    route.current_mode = mode
    route.budget_scale = 0.8
    route.frozen = True
    route.reference_uwr = 0.01
    optimizer._sync_frozen_routes(48)
    return parameter, optimizer


def test_frozen_budget_scale_changes_only_synced_frozen_budget() -> None:
    parameter, optimizer = _counterfactual_optimizer(
        frozen_budget_scale=0.8,
    )
    assert optimizer.frozen_budgets["dense"] == pytest.approx(0.64)
    assert optimizer.router.states["dense"].current_mode == "coordinate"
    assert optimizer.router.states["dense"].frozen is True


def _assert_counterfactual_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        actual_bytes = actual.detach().contiguous().reshape(1) if actual.ndim == 0 else actual.detach().contiguous().reshape(-1)
        expected_bytes = expected.detach().contiguous().reshape(1) if expected.ndim == 0 else expected.detach().contiguous().reshape(-1)
        assert torch.equal(actual_bytes.view(torch.uint8), expected_bytes.view(torch.uint8))
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_counterfactual_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for one, two in zip(actual, expected, strict=True):
            _assert_counterfactual_state_equal(one, two)
    elif hasattr(actual, "__dataclass_fields__"):
        for field in actual.__dataclass_fields__:
            _assert_counterfactual_state_equal(getattr(actual, field), getattr(expected, field))
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("best_scale", [0.9, 1.0, 1.1])
def test_counterfactual_same_preweights_and_exact_state_restore(mode, dtype, best_scale) -> None:
    parameter, optimizer = _counterfactual_optimizer(mode, dtype)
    descriptor = optimizer.descriptors[0]
    state = optimizer.states["dense"][mode]
    if isinstance(state, NorMuonState):
        state.mantissa.fill_(12345)
    before = parameter.detach().clone()
    expected_state = deepcopy(state)
    gradient = torch.arange(1, 13, dtype=dtype).reshape_as(parameter)
    config = optimizer._config(mode, descriptor, 1.0, 0.9, False)
    update, _, _, direction = optimizer_module._faro_candidate(
        mode, before, gradient, expected_state, config, descriptor
    )
    expected = []
    for scale in (0.9, 1.0, 1.1):
        candidate = before.clone()
        budget = 0.8 * scale
        if mode == "spectral" and dtype == torch.bfloat16:
            mantissa = state.mantissa.clone()
            apply_cautious_precise_(
                candidate, mantissa, direction,
                lr=config.lr * budget, weight_decay=config.weight_decay,
            )
        elif mode == "spectral":
            block_f = candidate.float()
            mask = direction.float() * block_f >= 0
            candidate.add_(
                -direction.float() * (config.lr * budget)
                - block_f * mask * (config.weight_decay * config.lr * budget)
            )
        else:
            candidate.add_(update.delta, alpha=budget)
        expected.append(candidate)
    parameter.grad = gradient
    result = optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert torch.equal(parameter, expected[1])
    assert result.decisions["dense"].budget_scale == pytest.approx(0.8)
    post = parameter.detach().clone()
    state_snapshot = optimizer.state_dict()
    state_ids = {field: id(getattr(state, field)) for field in state.__dataclass_fields__ if isinstance(getattr(state, field), torch.Tensor)}
    seen = []

    def evaluate_loss():
        index = len(seen)
        _assert_counterfactual_state_equal(parameter, expected[index])
        _assert_counterfactual_state_equal(optimizer.state_dict(), state_snapshot)
        seen.append(parameter.detach().clone())
        return float(index != (0.9, 1.0, 1.1).index(best_scale))

    audit = optimizer.evaluate_counterfactual_probe(evaluate_loss)
    assert len(seen) == 3
    assert audit["scales"] == (0.9, 1.0, 1.1)
    assert len(audit["losses"]) == 3
    assert audit["selected_scale"] == best_scale
    assert audit["applied_scale"] == audit["current_scale"] == 1.0
    assert audit["selection_used_next_update"] is True
    assert audit["restore_verified"] is True
    _assert_counterfactual_state_equal(parameter, post)
    state_snapshot["counterfactual_selected_scale"] = best_scale
    _assert_counterfactual_state_equal(optimizer.state_dict(), state_snapshot)
    assert state_ids == {field: id(getattr(state, field)) for field in state.__dataclass_fields__ if isinstance(getattr(state, field), torch.Tensor)}
    if mode == "coordinate":
        assert state.step == expected_state.step == 1
    else:
        _assert_counterfactual_state_equal(state.momentum, expected_state.momentum)
    assert optimizer._counterfactual_plans is None
    restored_parameter, restored = _counterfactual_optimizer(mode, dtype)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.counterfactual_selected_scale == best_scale
    assert restored._counterfactual_plans is None
    assert not any("arm" in key for key in restored.state_dict())
    restored.reset()
    assert restored.counterfactual_selected_scale == 1.0
    assert not hasattr(optimizer, "record_counterfactual_probe")
    assert not hasattr(optimizer, "counterfactual_budget_scale")


@pytest.mark.parametrize("failure", ["callback", "nan", "inf"])
def test_counterfactual_callback_failure_restores_and_releases(failure) -> None:
    parameter, optimizer = _counterfactual_optimizer("spectral", torch.bfloat16)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    post = parameter.detach().clone()
    snapshot = optimizer.state_dict()
    calls = []

    def evaluate_loss():
        calls.append(1)
        if len(calls) == 2:
            if failure == "callback":
                raise RuntimeError("probe callback failed")
            return float(failure)
        return 0.0

    with pytest.raises(RuntimeError if failure == "callback" else ValueError):
        optimizer.evaluate_counterfactual_probe(evaluate_loss)
    _assert_counterfactual_state_equal(parameter, post)
    _assert_counterfactual_state_equal(optimizer.state_dict(), snapshot)
    assert optimizer._counterfactual_plans is None
    assert optimizer.evaluate_counterfactual_probe(lambda: pytest.fail("no pending probe")) == {}


def test_counterfactual_neutral_tie_and_next_absolute_budget() -> None:
    parameter, optimizer = _counterfactual_optimizer()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert optimizer.evaluate_counterfactual_probe(lambda: 1.0)["selected_scale"] == 1.0
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=65, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    losses = iter((1.0, 0.5, 0.0))
    optimizer.evaluate_counterfactual_probe(lambda: next(losses))
    assert optimizer.counterfactual_selected_scale == 1.1
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(step=66, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert result.decisions["dense"].budget_scale == pytest.approx(0.8 * 1.1)
    assert optimizer._counterfactual_plans is None
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=81, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    audit = optimizer.evaluate_counterfactual_probe(lambda: 1.0)
    assert audit["applied_scale"] == 1.1
    assert audit["selected_scale"] == 1.0


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kelly_shadow_candidates_keep_neutral_base(mode, dtype) -> None:
    parameter, optimizer = _counterfactual_optimizer(
        mode, dtype, kelly_risk_controller=True
    )
    optimizer.router.kelly_state = "aggressive"
    optimizer.router.kelly_scale = 1.1
    optimizer.router.kelly_steps_remaining = 4
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert optimizer._counterfactual_plans[0].base_budget == pytest.approx(0.8)
    observed = []
    optimizer.evaluate_counterfactual_probe(
        lambda: observed.append(parameter.detach().clone()) or 1.0
    )
    reference_parameter, reference = _counterfactual_optimizer(mode, dtype)
    reference_parameter.grad = torch.ones_like(reference_parameter)
    reference.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    expected = []
    reference.evaluate_counterfactual_probe(
        lambda: expected.append(reference_parameter.detach().clone()) or 1.0
    )
    _assert_counterfactual_state_equal(observed, expected)


def test_counterfactual_multiple_probe_steps_do_not_open_micro_collector() -> None:
    parameter, optimizer = _counterfactual_optimizer()
    for step in (49, 65, 81):
        optimizer.start_step(step)
        for micro in range(8):
            parameter.grad = torch.ones_like(parameter) * (micro + 1)
            optimizer.capture_micro_gradient(micro, 10)
        optimizer.step(step=step, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
        assert optimizer.evaluate_counterfactual_probe(lambda: 1.0)["restore_verified"]
        assert not optimizer.collector._current
        assert not optimizer.collector._prepared


def test_kelly_does_not_consume_aggressive_duration_without_gradient() -> None:
    _, optimizer = _counterfactual_optimizer(kelly_risk_controller=True)
    optimizer.router.kelly_state = "aggressive"
    optimizer.router.kelly_scale = 1.1
    optimizer.router.kelly_steps_remaining = 2
    optimizer.step(step=50, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert optimizer.router.kelly_steps_remaining == 2


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
def test_counterfactual_uses_captured_projection_cap(mode) -> None:
    parameter, optimizer = _counterfactual_optimizer(mode, risk_budget_projection=True)
    optimizer.router.states["dense"].reference_uwr = 1e-5
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    post = parameter.detach().clone()
    assert optimizer._counterfactual_plans[0].projection_cap is not None
    optimizer.router.states["dense"].reference_uwr = 1.0
    def evaluate_loss():
        _assert_counterfactual_state_equal(parameter, post)
        return 1.0
    optimizer.evaluate_counterfactual_probe(evaluate_loss)


def test_kelly_budget_uses_router_scale_without_counterfactual_double_application() -> None:
    parameter, optimizer = _counterfactual_optimizer()
    optimizer.router.config = replace(
        optimizer.router.config,
        kelly_risk_controller=True,
    )
    optimizer.router.kelly_state = "aggressive"
    optimizer.router.kelly_scale = 1.1
    optimizer.router.kelly_steps_remaining = 1
    optimizer.counterfactual_selected_scale = 1.1
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(
        step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )
    assert result.decisions["dense"].budget_scale == pytest.approx(0.8 * 1.1)
    assert result.decisions["dense"].budget_scale < 0.8 * 1.1 * 1.1
    assert result.decisions["dense"].budget_statistics["kelly_state"] == "aggressive"
    assert optimizer.router.kelly_steps_remaining == 0
    assert optimizer.router.kelly_state == "neutral"
    parameter.grad = torch.ones_like(parameter)
    next_result = optimizer.step(
        step=50, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )
    assert next_result.decisions["dense"].budget_scale == pytest.approx(0.8)


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
def test_counterfactual_projection_last_uses_same_cap_and_clamp(mode) -> None:
    parameter, optimizer = _counterfactual_optimizer(mode, risk_budget_projection=True)
    optimizer.frozen_budgets["dense"] = 1.08
    optimizer.counterfactual_selected_scale = 1.1
    optimizer.router.states["dense"].reference_uwr = 1e-5
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    post = parameter.detach().clone()
    statistics = result.decisions["dense"].budget_statistics
    assert statistics["nominal_budget"] == pytest.approx(1.1)
    assert statistics["pre_alpha_budget"] == pytest.approx(1.08)
    assert statistics["counterfactual_scale"] == 1.1
    assert (post - before).norm() > 0

    def evaluate_loss():
        _assert_counterfactual_state_equal(parameter, post)
        return 0.0

    assert optimizer.evaluate_counterfactual_probe(evaluate_loss)["selected_scale"] == 1.0


@pytest.mark.parametrize("enabled,unit_budget", [(False, False), (True, True)])
def test_counterfactual_disabled_path_unchanged(enabled, unit_budget, monkeypatch) -> None:
    parameter, optimizer = _counterfactual_optimizer(unit_budget=unit_budget)
    optimizer.router.config = replace(optimizer.router.config, counterfactual_probe_budget=enabled)
    optimizer.counterfactual_selected_scale = 1.1
    before = parameter.detach().clone()
    expected_state = deepcopy(optimizer.states["dense"]["coordinate"])
    config = optimizer._config("coordinate", optimizer.descriptors[0], 1.0, 0.9, False)
    delta = native_adam_delta(
        before, torch.ones_like(before), expected_state,
        beta1=config.beta1, beta2=config.beta2, lr=config.lr,
        weight_decay=config.weight_decay, eps=config.eps,
    )
    monkeypatch.setattr(optimizer, "_capture_counterfactual_update", lambda *a, **kw: pytest.fail("disabled capture"))
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    _assert_counterfactual_state_equal(
        parameter,
        before.add(delta, alpha=0.8),
    )
    assert result.decisions["dense"].budget_statistics == {}
    assert optimizer._counterfactual_plans is None
    assert optimizer.evaluate_counterfactual_probe(lambda: pytest.fail("disabled callback")) == {}


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_counterfactual_tied_parameters_and_later_split(split, dtype) -> None:
    model = TinyModel().to(dtype=dtype)
    descriptors = _descriptors()[1:3]
    optimizer = FAROOptimizer(
        model.named_parameters(), descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(), muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(counterfactual_probe_budget=True)), world_size=1,
    )
    route = optimizer.router.initialize(descriptors[1])
    route.current_mode = "coordinate"
    route.frozen = True
    optimizer._sync_frozen_routes(48)
    if split:
        optimizer.copy_lm_state_to_embed()
    model.embed.grad = torch.ones_like(model.embed)
    model.lm_head.grad = -torch.ones_like(model.lm_head) * 0.5
    optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    post = {label: p.detach().clone() for label, p in optimizer.parameters.items()}
    snapshot = optimizer.state_dict()
    calls = []

    def evaluate_loss():
        calls.append(1)
        if split:
            assert not torch.equal(model.embed, model.lm_head.T)
        else:
            _assert_counterfactual_state_equal(model.embed, model.lm_head.T)
        return 0.0

    optimizer.evaluate_counterfactual_probe(evaluate_loss)
    assert len(calls) == 3
    for label, p in optimizer.parameters.items():
        _assert_counterfactual_state_equal(p, post[label])
    _assert_counterfactual_state_equal(optimizer.state_dict(), snapshot)


def test_counterfactual_only_captures_all_frozen_probe_steps() -> None:
    model = TinyBankModel()
    descriptors = _bank_descriptors()
    optimizer = FAROOptimizer(
        model.named_parameters(), descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(), muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(counterfactual_probe_budget=True)), world_size=1,
    )
    for descriptor in descriptors:
        optimizer.router.initialize(descriptor).current_mode = "coordinate"
    optimizer.router.states["bank.0"].frozen = True
    optimizer._sync_frozen_routes(48)
    optimizer.counterfactual_selected_scale = 1.1
    optimizer.start_step(49)
    for index in range(8):
        if model.bank.grad is None:
            model.bank.grad = torch.ones_like(model.bank) / 8
        else:
            model.bank.grad.add_(torch.ones_like(model.bank) / 8)
        optimizer.capture_micro_gradient(index, effective_tokens=10)
    result = optimizer.step(step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert result.decisions["bank.0"].budget_scale == 1.0
    assert optimizer._counterfactual_plans is None
    assert optimizer.evaluate_counterfactual_probe(lambda: pytest.fail("partial freeze")) == {}
    optimizer.router.states["bank.1"].frozen = True
    optimizer._sync_frozen_routes(50)
    model.bank.grad = torch.ones_like(model.bank)
    optimizer.step(step=50, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert optimizer._counterfactual_plans is None
    model.bank.grad = torch.ones_like(model.bank)
    optimizer.step(step=65, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)
    assert len(optimizer._counterfactual_plans) == 2
    optimizer.evaluate_counterfactual_probe(lambda: 0.0)


def _candidate_update(delta: torch.Tensor) -> CandidateUpdate:
    norm = delta.float().norm()
    return CandidateUpdate(
        delta=delta.float().clone(),
        direction_norm=norm.clone(),
        uwr=norm.clone(),
        max_axis_uwr=norm.clone(),
        spectral_risk=norm.clone(),
    )


def test_token_row_column_uses_current_update_norm_and_preserves_direction() -> None:
    model = TinyModel()
    descriptor = _descriptors()[2]
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(collect_steps=32)),
        world_size=1,
    )

    first_coordinate = _candidate_update(torch.tensor([[3.0, 4.0]]))
    first_row_column = _candidate_update(torch.tensor([[6.0, 8.0]]))
    optimizer._match_row_column_uwr(
        descriptor, 0, first_coordinate, first_row_column
    )

    second_coordinate = _candidate_update(torch.tensor([[3.0, 4.0]]))
    second_row_column = _candidate_update(torch.tensor([[0.6, 0.8]]))
    original_direction = second_row_column.delta / second_row_column.delta.norm()
    optimizer._match_row_column_uwr(
        descriptor, 1, second_coordinate, second_row_column
    )

    assert second_row_column.uwr == pytest.approx(second_coordinate.uwr)
    torch.testing.assert_close(
        second_row_column.delta / second_row_column.delta.norm(),
        original_direction,
    )
    assert optimizer.row_column_scale_samples[descriptor.label] == []


def test_token_row_column_exact_matching_handles_zero_initialized_table() -> None:
    model = TinyModel()
    descriptor = _descriptors()[2]
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    parameter = torch.zeros(3, 7)
    gradient = torch.arange(1, 22, dtype=torch.float32).reshape_as(parameter)
    config = GeometryConfig(
        lr=0.01,
        beta1=0.5,
        beta2=0.95,
        eps=1e-10,
        weight_decay=0.0,
    )
    coordinate = coordinate_candidate(
        parameter,
        gradient,
        CoordinateState.zeros_like(parameter),
        config,
    )
    row_column = row_column_candidate(
        parameter,
        gradient,
        RowColumnState.zeros_like(parameter, token_axis=1),
        config,
        token_axis=1,
    )

    optimizer._match_row_column_uwr(descriptor, 0, coordinate, row_column)

    assert torch.isfinite(row_column.delta).all()
    assert row_column.uwr == pytest.approx(coordinate.uwr)
    assert torch.isfinite(optimizer.row_column_scales[descriptor.label])

    zero_coordinate = _candidate_update(torch.zeros(1, 2))
    zero_row_column = _candidate_update(torch.zeros(1, 2))
    optimizer._match_row_column_uwr(
        descriptor, 1, zero_coordinate, zero_row_column
    )
    assert torch.isfinite(zero_row_column.delta).all()
    assert zero_row_column.uwr == 0
    assert optimizer.row_column_scales[descriptor.label] == 1


def test_copy_lm_state_to_embed_transposes_every_candidate() -> None:
    model = TinyModel()
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=_descriptors(),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig()),
        world_size=1,
    )
    lm_coordinate = optimizer.states["lm_head"]["coordinate"]
    assert isinstance(lm_coordinate, NativeAdamState)
    lm_coordinate.exp_avg.copy_(torch.arange(21).reshape(3, 7))

    optimizer.copy_lm_state_to_embed()

    embed_coordinate = optimizer.states["embed"]["coordinate"]
    torch.testing.assert_close(embed_coordinate.exp_avg, lm_coordinate.exp_avg.T)
    lm_spectral = optimizer.states["lm_head"]["spectral"]
    embed_spectral = optimizer.states["embed"]["spectral"]
    assert isinstance(lm_spectral, NorMuonState)
    assert isinstance(embed_spectral, NorMuonState)
    torch.testing.assert_close(embed_spectral.momentum, lm_spectral.momentum.T)
    torch.testing.assert_close(
        embed_spectral.second_momentum, lm_spectral.second_momentum.T
    )
    assert torch.equal(embed_spectral.mantissa, lm_spectral.mantissa.T)
    for mode in ("coordinate", "row_column", "spectral"):
        assert optimizer.states["embed"][mode] is not optimizer.states["lm_head"][mode]
    assert optimizer.split_embed


def test_freeze_prunes_after_selected_update_and_uses_fixed_fast_path(
    monkeypatch,
) -> None:
    model = TinyModel()
    model.dense = _label(
        nn.Parameter(model.dense.detach().bfloat16()), "dense"
    )
    descriptor = _descriptors()[0]
    router = FARORouter(
        RouterConfig(
            route_interval=1,
            collect_steps=0,
            min_residence_steps=0,
            freeze_stable_routes=True,
        )
    )
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    increments = [{"dense": torch.ones_like(model.dense) / 8} for _ in range(8)]
    _accumulate_optimizer_gradients(
        model, optimizer, step=0, increments=increments
    )

    optimizer.step(
        step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert router.is_frozen("dense")
    assert set(optimizer.states["dense"]) == {"spectral"}
    selected = optimizer.states["dense"]["spectral"]
    assert isinstance(selected, NorMuonState)
    assert torch.count_nonzero(selected.momentum)
    frozen_budget = router.states["dense"].budget_scale
    assert optimizer.frozen_budgets == {"dense": frozen_budget}
    assert optimizer.all_routes_frozen
    assert optimizer.consume_freeze_events() == []
    assert optimizer._freeze_events[0]["current_allocated_mib"] is None
    assert optimizer._freeze_events[0]["current_reserved_mib"] is None

    optimizer.finalize_freeze_events()

    events = optimizer.consume_freeze_events()
    assert events == [
        {
            "step": 0,
            "newly_frozen": 1,
            "total_frozen": 1,
            "mode_counts": {"spectral": 1},
            "all_routes_frozen": True,
            "current_allocated_mib": 0.0,
            "current_reserved_mib": 0.0,
        }
    ]
    assert optimizer.consume_freeze_events() == []
    before = selected.momentum.clone()
    expected_parameter = model.dense.detach().clone()
    expected_state = deepcopy(selected)
    config = optimizer._config("spectral", descriptor, 1.0, 0.9, False)
    expected_direction = normuon_direction(
        torch.ones_like(model.dense),
        expected_state,
        momentum=config.momentum,
        beta2=config.beta2,
    )
    apply_cautious_precise_(
        expected_parameter,
        expected_state.mantissa,
        expected_direction,
        lr=config.lr * frozen_budget,
        weight_decay=config.weight_decay,
    )
    candidate_calls: list[str] = []

    def fail_route(*args, **kwargs):
        raise AssertionError("frozen fast path must not route")

    def fail_observation(*args, **kwargs):
        raise AssertionError("frozen fast path must not build observations")

    def count_candidate(mode, *args, **kwargs):
        candidate_calls.append(mode)
        return original_candidate(mode, *args, **kwargs)

    metric_calls = 0
    original_candidate = optimizer_module._faro_candidate
    original_metrics = optimizer_module._candidate_metrics

    def count_metrics(*args, **kwargs):
        nonlocal metric_calls
        metric_calls += 1
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(router, "route", fail_route)
    monkeypatch.setattr(
        CandidateObservation, "from_returns", fail_observation
    )
    monkeypatch.setattr(optimizer_module, "_faro_candidate", count_candidate)
    monkeypatch.setattr(optimizer_module, "_candidate_metrics", count_metrics)
    _accumulate_optimizer_gradients(
        model, optimizer, step=1, increments=increments
    )
    assert optimizer.collector.is_empty

    optimizer.step(
        step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert candidate_calls == []
    assert metric_calls == 0
    assert not torch.equal(selected.momentum, before)
    assert torch.equal(
        model.dense.view(torch.uint16),
        expected_parameter.view(torch.uint16),
    )
    torch.testing.assert_close(selected.momentum, expected_state.momentum)
    torch.testing.assert_close(
        selected.second_momentum, expected_state.second_momentum
    )
    assert torch.equal(selected.mantissa, expected_state.mantissa)
    assert router.states["dense"].budget_scale == frozen_budget
    assert optimizer.collector.is_empty


def test_local_reward_runs_only_after_all_routes_freeze_without_reopening_geometry(
    monkeypatch,
) -> None:
    model = TinyModel()
    model.dense = _label(
        nn.Parameter(model.dense.detach().bfloat16()), "dense"
    )
    descriptor = _descriptors()[0]
    router = FARORouter(
        RouterConfig(
            route_interval=1,
            collect_steps=0,
            min_residence_steps=0,
            freeze_stable_routes=True,
            local_reward_engine=True,
        )
    )
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    increments = [{"dense": torch.ones_like(model.dense) / 8} for _ in range(8)]
    _accumulate_optimizer_gradients(model, optimizer, step=0, increments=increments)
    optimizer.step(step=0, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True)

    assert optimizer.all_routes_frozen
    mode = router.states["dense"].current_mode
    calls: list[tuple[float, ...]] = []

    def reward_target(returns):
        values = tuple(float(value) for value in returns)
        calls.append(values)
        assert len(values) == 8
        return 0.5, {
            "reward_edge": -1.0,
            "reward_variance": 0.0,
            "reward_target": 0.5,
        }

    monkeypatch.setattr(router, "local_reward_target", reward_target)
    _accumulate_optimizer_gradients(model, optimizer, step=1, increments=increments)
    result = optimizer.step(
        step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    decision = result.decisions["dense"]
    assert calls and all(value > 0.0 for value in calls[0])
    assert decision.mode == mode
    assert not decision.route_event
    assert decision.reason == "frozen"
    assert decision.budget_statistics["reward_engine"] is True
    assert decision.budget_statistics["reward_target"] == pytest.approx(0.5)
    assert decision.budget_scale < optimizer.frozen_budgets["dense"]
    assert optimizer.collector.is_empty


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_decay", [0.0, 1.2])
def test_projection_limits_frozen_update_without_changing_moments(mode, dtype, weight_decay) -> None:
    descriptor = _descriptors()[0]
    parameter = _label(nn.Parameter(torch.ones(4, 3, dtype=dtype)), "dense")
    router = FARORouter(
        RouterConfig(freeze_stable_routes=True, risk_budget_projection=True)
    )
    optimizer = FAROOptimizer(
        [("dense", parameter)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults={**_muon_defaults(), "weight_decay": weight_decay},
        router=router,
        world_size=1,
    )
    route = router.initialize(descriptor)
    route.current_mode = mode
    route.frozen = True
    route.reference_uwr = 1e-5
    optimizer._sync_frozen_routes(48)
    expected_state = deepcopy(optimizer.states["dense"][mode])
    before = parameter.detach().clone()
    gradient = torch.ones_like(parameter)
    config = optimizer._config(mode, descriptor, 1.0, 0.9, False)
    optimizer_module._faro_candidate(
        mode, before, gradient, expected_state, config, descriptor
    )
    parameter.grad = gradient

    result = optimizer.step(
        step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )
    decision = result.decisions["dense"]

    assert decision.mode == mode
    assert not decision.route_event
    assert decision.budget_scale < 0.5
    assert optimizer.frozen_budgets["dense"] == 1.0
    assert decision.budget_statistics["projected"] is True
    assert decision.budget_statistics["nominal_budget"] == pytest.approx(1.0)
    assert decision.budget_statistics["effective_budget"] == pytest.approx(
        decision.budget_scale
    )
    assert decision.budget_statistics["reference_uwr"] == pytest.approx(1e-5)
    assert decision.budget_statistics["effective_uwr"] <= 4e-5 * (1 + 1e-5)
    state = optimizer.states["dense"][mode]
    if mode == "coordinate":
        torch.testing.assert_close(state.exp_avg, expected_state.exp_avg)
        torch.testing.assert_close(state.exp_avg_sq, expected_state.exp_avg_sq)
    else:
        torch.testing.assert_close(state.momentum, expected_state.momentum)
    if mode == "spectral" and dtype == torch.bfloat16:
        expected = before.clone()
        mantissa = torch.zeros_like(state.mantissa)
        direction_state = NorMuonState.zeros_like(before)
        direction = normuon_direction(
            gradient, direction_state, momentum=config.momentum, beta2=config.beta2
        )
        apply_cautious_precise_(
            expected, mantissa, direction,
            lr=config.lr * decision.budget_scale, weight_decay=config.weight_decay,
        )
        assert torch.equal(parameter.view(torch.uint16), expected.view(torch.uint16))
        assert torch.equal(state.mantissa, mantissa)
    if dtype == torch.float32:
        realized_uwr = (parameter.detach() - before).norm() / before.norm()
        assert float(realized_uwr) <= 4e-5 * 1.01


@pytest.mark.parametrize("mode", ["coordinate", "row_column", "spectral"])
def test_disabled_projection_keeps_frozen_fast_path_uninstrumented(mode, monkeypatch) -> None:
    descriptor = _descriptors()[0]
    parameter = _label(nn.Parameter(torch.ones(4, 3)), "dense")
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        [("dense", parameter)], descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(), muon_defaults=_muon_defaults(),
        router=router, world_size=1,
    )
    route = router.initialize(descriptor)
    route.current_mode = mode
    route.frozen = True
    optimizer._sync_frozen_routes(48)

    def unexpected_projection(*args):
        pytest.fail("disabled projection must not enter the projection path")

    monkeypatch.setattr(router, "project_budget", unexpected_projection)
    parameter.grad = torch.ones_like(parameter)
    result = optimizer.step(
        step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert result.decisions["dense"].budget_statistics == {}


@pytest.mark.parametrize("projection", [False, True])
def test_projection_limits_individually_frozen_block_before_all_freeze(projection) -> None:
    model = TinyBankModel()
    model.bank.data.fill_(1.0)
    descriptors = _bank_descriptors()
    router = FARORouter(
        RouterConfig(freeze_stable_routes=True, risk_budget_projection=projection)
    )
    optimizer = FAROOptimizer(
        [("bank", model.bank)], descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(), muon_defaults=_muon_defaults(),
        router=router, world_size=1,
    )
    route = router.initialize(descriptors[0])
    route.current_mode = "coordinate"
    route.frozen = True
    route.reference_uwr = 1e-5
    router.initialize(descriptors[1])
    optimizer._sync_frozen_routes(48)
    assert not optimizer.all_routes_frozen
    optimizer.start_step(49)
    for index in range(8):
        if model.bank.grad is None:
            model.bank.grad = torch.ones_like(model.bank) / 8
        else:
            model.bank.grad.add_(torch.ones_like(model.bank) / 8)
        optimizer.capture_micro_gradient(index, effective_tokens=10)

    result = optimizer.step(
        step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    decision = result.decisions["bank.0"]
    if projection:
        assert decision.budget_scale < 0.5
        assert decision.budget_statistics["nominal_budget"] == 1.0
        assert decision.budget_statistics["projected"] is True
    else:
        assert decision.budget_scale == 1.0
        assert decision.budget_statistics == {}
    assert decision.mode == "coordinate"
    assert optimizer.frozen_budgets["bank.0"] == 1.0


def test_freeze_memory_finalization_defers_cuda_release_until_after_step(
    monkeypatch,
) -> None:
    model = TinyModel()
    descriptor = _descriptors()[0]
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        [("dense", model.dense)],
        descriptors=(descriptor,),
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    route = router.initialize(descriptor)
    route.current_mode = "spectral"
    route.frozen = True
    route.frozen_step = 48
    calls: list[object] = []
    cuda_device = torch.device("cuda", 6)
    monkeypatch.setattr(
        optimizer, "_cuda_device", lambda: cuda_device, raising=False
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: calls.append(("allocated", device)) or 3 * 1024**2,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: calls.append(("reserved", device)) or 5 * 1024**2,
    )

    optimizer._sync_frozen_routes(48)

    assert calls == []
    assert optimizer.consume_freeze_events() == []

    optimizer.finalize_freeze_events()

    assert calls == [
        "empty_cache",
        ("allocated", cuda_device),
        ("reserved", cuda_device),
    ]
    event = optimizer.consume_freeze_events()
    assert event[0]["current_allocated_mib"] == 3.0
    assert event[0]["current_reserved_mib"] == 5.0
    optimizer.finalize_freeze_events()
    assert calls == [
        "empty_cache",
        ("allocated", cuda_device),
        ("reserved", cuda_device),
    ]


def test_all_frozen_compacts_mixed_dense_bank_and_severs_old_storage(
    monkeypatch,
) -> None:
    model = TinyBankModel()
    descriptors = _bank_descriptors()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        [("bank", model.bank)],
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    old_bank = optimizer.spectral_banks["bank"]
    old_coordinate = optimizer.states["bank.1"]["coordinate"]
    assert isinstance(old_coordinate, NativeAdamState)
    old_coordinate_ptr = old_coordinate.exp_avg.data_ptr()
    first = router.initialize(descriptors[0])
    first.current_mode = "spectral"
    first.budget_scale = 0.75
    first.frozen = True
    first.frozen_step = 48
    second = router.initialize(descriptors[1])
    second.current_mode = "coordinate"
    second.budget_scale = 0.625
    second.frozen = True
    second.frozen_step = 48
    optimizer.row_column_scale_samples["bank.0"].append(torch.ones(()))
    optimizer.row_column_scale_samples["bank.1"].append(torch.ones(()))
    optimizer.activation_drift = {
        "bank.0": torch.ones(()),
        "bank.1": torch.ones(()),
    }

    optimizer._sync_frozen_routes(48)

    assert set(optimizer.states["bank.0"]) == {"spectral"}
    assert set(optimizer.states["bank.1"]) == {"coordinate"}
    compact = optimizer.spectral_banks["bank"]
    assert compact.momentum.shape == (1, 4, 3)
    spectral = optimizer.states["bank.0"]["spectral"]
    coordinate = optimizer.states["bank.1"]["coordinate"]
    assert isinstance(spectral, NorMuonState)
    assert isinstance(coordinate, NativeAdamState)
    assert spectral.momentum.data_ptr() == compact.momentum[0].data_ptr()
    assert spectral.momentum.untyped_storage().data_ptr() != old_bank.momentum.untyped_storage().data_ptr()
    assert coordinate.exp_avg.data_ptr() != old_coordinate_ptr
    assert optimizer.row_column_scale_samples == {}
    assert optimizer.row_column_scales == {}
    assert optimizer.activation_drift == {}
    expected_parameter = model.bank.detach().clone()
    expected_spectral = deepcopy(spectral)
    spectral_config = optimizer._config(
        "spectral", descriptors[0], 1.0, 0.9, False
    )
    expected_direction = normuon_direction(
        torch.ones_like(expected_parameter[0]),
        expected_spectral,
        momentum=spectral_config.momentum,
        beta2=spectral_config.beta2,
    )
    expected_block = expected_parameter[0].float()
    decay_mask = expected_direction.float() * expected_block >= 0
    expected_parameter[0].add_(
        -expected_direction.float() * (spectral_config.lr * 0.75)
        - expected_block
        * decay_mask
        * (spectral_config.weight_decay * spectral_config.lr * 0.75)
    )
    expected_coordinate = deepcopy(coordinate)
    coordinate_config = optimizer._config(
        "coordinate", descriptors[1], 1.0, 0.9, False
    )
    expected_delta = native_adam_delta(
        expected_parameter[1],
        torch.ones_like(expected_parameter[1]),
        expected_coordinate,
        beta1=coordinate_config.beta1,
        beta2=coordinate_config.beta2,
        lr=coordinate_config.lr,
        weight_decay=coordinate_config.weight_decay,
        eps=coordinate_config.eps,
    )
    expected_parameter[1].add_(expected_delta, alpha=0.625)
    calls: list[tuple[int, ...]] = []
    gradient_pointers: list[int] = []
    original = optimizer_module.normuon_direction

    def counted(gradient, *args, **kwargs):
        calls.append(tuple(gradient.shape))
        gradient_pointers.append(gradient.data_ptr())
        return original(gradient, *args, **kwargs)

    monkeypatch.setattr(optimizer_module, "normuon_direction", counted)
    model.bank.grad = torch.ones_like(model.bank)
    full_gradient_pointer = model.bank.grad.data_ptr()
    optimizer.step(
        step=49, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert calls == [(1, 4, 3)]
    assert gradient_pointers[0] != full_gradient_pointer
    torch.testing.assert_close(model.bank, expected_parameter)
    torch.testing.assert_close(spectral.momentum, expected_spectral.momentum)
    torch.testing.assert_close(
        spectral.second_momentum, expected_spectral.second_momentum
    )
    assert coordinate.step == 1
    assert coordinate.step == expected_coordinate.step
    torch.testing.assert_close(coordinate.exp_avg, expected_coordinate.exp_avg)
    torch.testing.assert_close(
        coordinate.exp_avg_sq, expected_coordinate.exp_avg_sq
    )


def test_partial_spectral_freeze_keeps_bank_view_and_round_trips() -> None:
    model = TinyBankModel()
    descriptors = _bank_descriptors()
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        [("bank", model.bank)],
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    state = router.initialize(descriptors[0])
    state.current_mode = "spectral"
    state.frozen = True
    state.frozen_step = 48

    optimizer._sync_frozen_routes(48)

    selected = optimizer.states["bank.0"]["spectral"]
    assert isinstance(selected, NorMuonState)
    assert selected.momentum.data_ptr() == optimizer.spectral_banks[
        "bank"
    ].momentum[0].data_ptr()
    assert not optimizer.all_routes_frozen

    restored = FAROOptimizer(
        [("bank", model.bank)],
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(freeze_stable_routes=True)),
        world_size=1,
    )
    restored.load_state_dict(optimizer.state_dict())

    assert set(restored.states["bank.0"]) == {"spectral"}
    assert set(restored.states["bank.1"]) == {
        "coordinate",
        "row_column",
        "spectral",
    }
    assert restored.states["bank.0"]["spectral"].momentum.data_ptr() == restored.spectral_banks[
        "bank"
    ].momentum[0].data_ptr()
    assert not restored.all_routes_frozen
    routed: list[tuple[str, ...]] = []
    original_route = restored.router.route

    def capture_route(step, routed_descriptors, observations):
        routed.append(tuple(item.label for item in routed_descriptors))
        return original_route(step, routed_descriptors, observations)

    restored.router.route = capture_route
    restored.start_step(1)
    for index in range(8):
        if model.bank.grad is None:
            model.bank.grad = torch.ones_like(model.bank) / 8
        else:
            model.bank.grad.add_(torch.ones_like(model.bank) / 8)
        restored.capture_micro_gradient(index, effective_tokens=10)

    restored.step(
        step=1, lr_scale=1.0, muon_momentum=0.9, update_coordinate=True
    )

    assert routed == [("bank.1",)]


def test_tied_embed_resolves_before_split_and_freeze_state_round_trips() -> None:
    model = TinyModel()
    descriptors = _descriptors()[1:3]
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    lm_state = router.initialize(descriptors[1])
    lm_state.current_mode = "coordinate"
    lm_state.budget_scale = 0.7
    lm_state.frozen = True
    lm_state.frozen_step = 49
    source = optimizer.states["lm_head"]["coordinate"]
    assert isinstance(source, NativeAdamState)
    source.exp_avg.copy_(torch.arange(21).reshape(3, 7))

    optimizer._sync_frozen_routes(49)

    assert optimizer.all_routes_frozen
    assert set(optimizer.states["embed"]) == {"coordinate"}
    assert set(optimizer.states["lm_head"]) == {"coordinate"}
    optimizer.copy_lm_state_to_embed()
    embed = optimizer.states["embed"]["coordinate"]
    assert isinstance(embed, NativeAdamState)
    torch.testing.assert_close(embed.exp_avg, source.exp_avg.T)
    assert optimizer.router.is_frozen("embed")
    assert optimizer.router.states["embed"].budget_scale == 0.7

    payload = optimizer.state_dict()
    restored = FAROOptimizer(
        model.named_parameters(),
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=FARORouter(RouterConfig(freeze_stable_routes=True)),
        world_size=1,
    )
    restored.load_state_dict(payload)
    assert restored.all_routes_frozen
    assert restored.frozen_budgets == {"embed": 0.7, "lm_head": 0.7}
    assert set(restored.states["embed"]) == {"coordinate"}
    assert restored.router.is_frozen("embed")

    restored.reset()
    assert not restored.all_routes_frozen
    assert restored.frozen_budgets == {}
    assert restored.consume_freeze_events() == []
    assert set(restored.states["embed"]) == {
        "coordinate",
        "row_column",
        "spectral",
    }


def test_frozen_row_column_tied_split_accepts_compacted_empty_samples() -> None:
    model = TinyModel()
    descriptors = _descriptors()[1:3]
    router = FARORouter(RouterConfig(freeze_stable_routes=True))
    optimizer = FAROOptimizer(
        model.named_parameters(),
        descriptors=descriptors,
        coordinate_defaults=_coordinate_defaults(),
        muon_defaults=_muon_defaults(),
        router=router,
        world_size=1,
    )
    lm_route = router.initialize(descriptors[1])
    lm_route.current_mode = "row_column"
    lm_route.budget_scale = 0.65
    lm_route.frozen = True
    lm_route.frozen_step = 49
    lm_state = optimizer.states["lm_head"]["row_column"]
    assert isinstance(lm_state, RowColumnState)
    lm_state.momentum.copy_(torch.arange(21).reshape(3, 7))
    optimizer.row_column_scales["lm_head"].fill_(1.75)

    optimizer._sync_frozen_routes(49)
    assert optimizer.row_column_scale_samples == {}

    optimizer.copy_lm_state_to_embed()

    embed_state = optimizer.states["embed"]["row_column"]
    assert isinstance(embed_state, RowColumnState)
    torch.testing.assert_close(embed_state.momentum, lm_state.momentum.T)
    assert embed_state.token_axis == 1 - lm_state.token_axis
    assert optimizer.row_column_scales["embed"] == 1.75
    assert optimizer.row_column_scale_samples == {}
    assert optimizer.frozen_budgets == {"embed": 0.65, "lm_head": 0.65}
    assert optimizer.router.is_frozen("embed")
    assert optimizer.router.states["embed"].budget_scale == 0.65


def test_new_optimizers_reject_multi_gpu() -> None:
    model = TinyModel()
    with torch.no_grad():
        try:
            FixedGeometryOptimizer(
                model.named_parameters(),
                descriptors=_descriptors(),
                coordinate_defaults=_coordinate_defaults(),
                muon_defaults=_muon_defaults(),
                variant="adam",
                world_size=2,
            )
        except ValueError as error:
            assert "single-GPU" in str(error)
        else:
            raise AssertionError("multi-GPU construction must fail")
