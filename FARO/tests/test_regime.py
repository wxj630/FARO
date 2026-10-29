import numpy as np
import pytest
import torch

from FARO.regime import DataRegime, corrupt_batch_, short_context_boundaries


def test_short_context_boundaries_add_fixed_segments_without_changing_tokens() -> None:
    boundaries = short_context_boundaries(
        torch.arange(10),
        torch.tensor([0, 6, 10], dtype=torch.int32),
        context_length=4,
    )

    assert boundaries.tolist() == [0, 4, 6, 10]


def test_short_context_boundaries_keep_clean_document_boundaries() -> None:
    boundaries = short_context_boundaries(
        torch.arange(10),
        torch.tensor([0, 6, 10], dtype=torch.int32),
        context_length=32,
    )

    assert boundaries.tolist() == [0, 6, 10]


def test_regime_phase_alternates_clean_and_dirty() -> None:
    regime = DataRegime(
        period_micro_batches=8,
        corruption_rate=0.5,
        dirty_first=False,
        seed=0,
        vocab_limit=100,
    )

    dirty = [regime.is_dirty(index) for index in range(32)]

    assert dirty == [False] * 8 + [True] * 8 + [False] * 8 + [True] * 8


def test_scatter_regime_is_deterministic_and_near_half_dirty() -> None:
    regime = DataRegime(
        period_micro_batches=8,
        corruption_rate=0.5,
        dirty_first=False,
        seed=0,
        vocab_limit=100,
        scatter=True,
    )

    first = [regime.is_dirty(index) for index in range(2000)]
    repeat = [regime.is_dirty(index) for index in range(2000)]

    assert first == repeat  # deterministic per (seed, index)
    rate = sum(first) / len(first)
    assert 0.45 < rate < 0.55
    longest = run = 0
    for value in first:
        run = run + 1 if value else 0
        longest = max(longest, run)
    assert longest <= 12  # no long blocks


def test_scatter_regime_differs_from_block_regime() -> None:
    kwargs = dict(
        period_micro_batches=8,
        corruption_rate=0.5,
        dirty_first=False,
        seed=3,
        vocab_limit=100,
    )
    block = [DataRegime(**kwargs).is_dirty(i) for i in range(64)]
    scatter = [DataRegime(**kwargs, scatter=True).is_dirty(i) for i in range(64)]

    assert block != scatter
    assert block == [i % 16 >= 8 for i in range(64)]


def test_regime_dirty_first_flips_starting_phase() -> None:
    regime = DataRegime(
        period_micro_batches=4,
        corruption_rate=0.5,
        dirty_first=True,
        seed=0,
        vocab_limit=100,
    )

    assert [regime.is_dirty(index) for index in range(8)] == [True] * 4 + [False] * 4


def test_regime_rate_is_zero_in_clean_phase() -> None:
    regime = DataRegime(
        period_micro_batches=2,
        corruption_rate=0.25,
        dirty_first=False,
        seed=0,
        vocab_limit=100,
    )

    assert regime.rate_at(0) == 0.0
    assert regime.rate_at(1) == 0.0
    assert regime.rate_at(2) == pytest.approx(0.25)


def test_regime_phase_offset_aligns_a_late_probe_stream() -> None:
    regime = DataRegime(
        period_micro_batches=8,
        corruption_rate=0.5,
        dirty_first=False,
        seed=0,
        vocab_limit=100,
        phase_offset_micro_batches=4,
    )

    assert [regime.is_dirty(index) for index in range(8)] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    ("period", "rate"),
    [(0, 0.1), (-1, 0.1), (4, -0.1), (4, 1.0)],
)
def test_regime_rejects_invalid_configuration(period: int, rate: float) -> None:
    with pytest.raises(ValueError):
        DataRegime(
            period_micro_batches=period,
            corruption_rate=rate,
            dirty_first=False,
            seed=0,
            vocab_limit=100,
        )


def test_corrupt_batch_is_deterministic_bounded_and_dirty() -> None:
    base = torch.arange(0, 4096, dtype=torch.int32).reshape(64, 64)
    inputs, targets = base.clone(), base.clone()

    corrupt_batch_(
        inputs, targets, 0.25, torch.Generator().manual_seed(7), vocab_limit=50
    )

    changed = inputs != base
    assert bool(changed.any())
    assert int(inputs[changed].max()) < 50
    assert not torch.equal(targets, base)
    fraction = float(changed.float().mean())
    assert 0.15 < fraction < 0.35

    replay_inputs, replay_targets = base.clone(), base.clone()
    corrupt_batch_(
        replay_inputs,
        replay_targets,
        0.25,
        torch.Generator().manual_seed(7),
        vocab_limit=50,
    )
    assert torch.equal(inputs, replay_inputs)
    assert torch.equal(targets, replay_targets)


def test_corrupt_batch_can_keep_targets_clean() -> None:
    base = torch.arange(0, 4096, dtype=torch.int32).reshape(64, 64)
    inputs, targets = base.clone(), base.clone()

    corrupt_batch_(
        inputs,
        targets,
        0.5,
        torch.Generator().manual_seed(11),
        vocab_limit=50,
        corrupt_targets=False,
    )

    assert bool((inputs != base).any())
    assert torch.equal(targets, base)


def test_corrupt_batch_can_use_a_fixed_input_mask_token() -> None:
    base = torch.arange(0, 4096, dtype=torch.int32).reshape(64, 64)
    inputs, targets = base.clone(), base.clone()

    corrupt_batch_(
        inputs,
        targets,
        0.5,
        torch.Generator().manual_seed(11),
        vocab_limit=50,
        corrupt_targets=False,
        mask_inputs=True,
    )

    changed = inputs != base
    assert bool(changed.any())
    assert torch.equal(inputs[changed], torch.full_like(inputs[changed], 49))
    assert torch.equal(targets, base)

    replay_inputs, replay_targets = base.clone(), base.clone()
    corrupt_batch_(
        replay_inputs,
        replay_targets,
        0.5,
        torch.Generator().manual_seed(97),
        vocab_limit=50,
        corrupt_targets=False,
        mask_inputs=True,
    )
    assert torch.equal(inputs, replay_inputs)
    assert torch.equal(targets, replay_targets)


def test_corrupt_batch_handles_uint16_shard_tokens() -> None:
    base = torch.from_numpy(np.arange(0, 4096, dtype=np.uint16)).reshape(64, 64)
    inputs, targets = base.clone(), base.clone()

    corrupt_batch_(
        inputs, targets, 0.2, torch.Generator().manual_seed(3), vocab_limit=50256
    )

    changed_inputs = inputs != base
    changed_targets = targets != base
    assert bool(changed_inputs.any())
    assert bool(changed_targets.any())
    assert int(inputs[changed_inputs].to(torch.int32).max()) < 50256
    assert int(targets[changed_targets].to(torch.int32).max()) < 50256


def test_zero_rate_leaves_batch_untouched() -> None:
    base = torch.arange(0, 256, dtype=torch.int32)
    inputs, targets = base.clone(), base.clone()

    corrupt_batch_(
        inputs, targets, 0.0, torch.Generator().manual_seed(0), vocab_limit=50
    )

    assert torch.equal(inputs, base)
    assert torch.equal(targets, base)
