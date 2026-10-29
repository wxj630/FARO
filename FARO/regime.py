from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DataRegime:
    """Alternating clean/dirty training phases for regime-switching runs.

    Phase length is counted in micro-batches. A dirty phase may either replace
    input tokens or shorten attention context while leaving token targets intact.

    With ``scatter=True`` each micro-batch independently becomes dirty (~50%),
    keyed on a deterministic hash of ``(seed, index)``. This keeps the same
    total corruption as the block mode while removing the temporal structure --
    the control that isolates "structured difficulty" from the same difficulty
    spread at random.
    """

    period_micro_batches: int
    corruption_rate: float
    dirty_first: bool
    seed: int
    vocab_limit: int
    corrupt_targets: bool = True
    fixed_input_mask: bool = False
    short_context_length: int = 0
    phase_offset_micro_batches: int = 0
    scatter: bool = False

    def __post_init__(self) -> None:
        if self.period_micro_batches <= 0:
            raise ValueError("period_micro_batches must be positive")
        if not 0.0 <= self.corruption_rate < 1.0:
            raise ValueError("corruption_rate must be in [0, 1)")
        if self.vocab_limit <= 1:
            raise ValueError("vocab_limit must exceed 1")
        if self.short_context_length < 0:
            raise ValueError("short_context_length must be nonnegative")

    @staticmethod
    def _scatter_is_dirty(batch_index: int, seed: int) -> bool:
        """Deterministic ~50% Bernoulli draw keyed on (seed, batch_index)."""
        x = (batch_index * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
        x ^= x >> 12
        x = (x * 0x297A2D39) & 0xFFFFFFFF
        x ^= x >> 15
        return bool(x & 1)

    def is_dirty(self, batch_index: int) -> bool:
        if self.scatter:
            return self._scatter_is_dirty(
                batch_index + self.phase_offset_micro_batches, self.seed
            )
        phase = (
            (batch_index + self.phase_offset_micro_batches)
            // self.period_micro_batches
        ) % 2
        return bool(phase) != self.dirty_first

    def rate_at(self, batch_index: int) -> float:
        return self.corruption_rate if self.is_dirty(batch_index) else 0.0


def short_context_boundaries(
    inputs: Tensor,
    document_boundaries: Tensor,
    *,
    context_length: int,
) -> Tensor:
    """Add fixed-length attention segments without changing token streams."""
    if context_length <= 0:
        return document_boundaries
    if document_boundaries.ndim != 1 or document_boundaries.numel() < 2:
        raise ValueError("document_boundaries must contain at least two offsets")
    total_tokens = int(inputs.numel())
    if int(document_boundaries[-1]) > total_tokens:
        raise ValueError("document_boundaries cannot exceed the input length")
    chunks: list[int] = [int(document_boundaries[0])]
    for start, end in zip(document_boundaries[:-1], document_boundaries[1:]):
        start_value = int(start)
        end_value = int(end)
        if end_value <= start_value:
            continue
        chunks.extend(range(start_value + context_length, end_value, context_length))
        chunks.append(end_value)
    if chunks[-1] != total_tokens:
        chunks.extend(
            range(chunks[-1] + context_length, total_tokens, context_length)
        )
        chunks.append(total_tokens)
    return torch.tensor(chunks, dtype=document_boundaries.dtype, device=document_boundaries.device)


def corrupt_batch_(
    inputs: Tensor,
    targets: Tensor,
    rate: float,
    generator: torch.Generator,
    *,
    vocab_limit: int,
    corrupt_targets: bool = True,
    mask_inputs: bool = False,
) -> None:
    """Replace a `rate` fraction of selected token streams with random ids.

    Inputs and targets draw independent masks when both streams are enabled.
    Keeping targets clean models damaged context with an intact training
    objective. The replacement ids stay below `vocab_limit` (callers pass the
    BOS id so a stray BOS cannot be introduced). The masked write happens
    through an int32 copy because uint16 does not implement index_put.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    if rate == 0.0:
        return
    tensors = (inputs, targets) if corrupt_targets else (inputs,)
    for tensor in tensors:
        if mask_inputs and tensor is inputs:
            indices = torch.arange(tensor.numel(), dtype=torch.int64).reshape(
                tensor.shape
            )
            mask = (indices * 7919 % 10_000) < int(rate * 10_000)
        else:
            mask = torch.rand(tensor.shape, generator=generator) < rate
        if not bool(mask.any()):
            continue
        if mask_inputs and tensor is inputs:
            replacement = torch.full(
                tensor.shape,
                vocab_limit - 1,
                dtype=torch.int32,
            )
        else:
            replacement = torch.randint(
                0, vocab_limit, tensor.shape, generator=generator, dtype=torch.int32
            )
        replaced = torch.where(mask, replacement, tensor.to(torch.int32))
        tensor.copy_(replaced.to(tensor.dtype))
