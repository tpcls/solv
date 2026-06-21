"""Length-linear index-routing algorithms for sequence tasks.

The core idea is to separate *where each answer token comes from* from *what the
answer token is*.  For tasks whose output is a known routing of input positions
(such as reversal), the solver builds all source indices at once and gathers the
answer in a single vectorized pass.

This avoids autoregressive re-decoding for deterministic routing tasks:
producing N materialized tokens performs O(N) routing and copy work, not O(N^2)
repeated decoding. O(N) is still the lower bound for materializing N output
tokens, and these helpers do not solve open-ended text generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

PAD = 0
EOS = 2
RoutingTask = Literal["reverse", "copy"]


@dataclass(frozen=True)
class RoutingPlan:
    """A vectorized plan that maps each output position to one input position."""

    source_indices: Tensor
    output_mask: Tensor


def infer_lengths(sequences: Tensor, *, eos_token: int = EOS, pad_token: int = PAD) -> Tensor:
    """Infer sequence lengths up to and including EOS, ignoring trailing padding."""

    eos_hits = sequences.eq(eos_token)
    has_eos = eos_hits.any(dim=1)
    first_eos = eos_hits.float().argmax(dim=1) + 1
    non_pad_lengths = sequences.ne(pad_token).sum(dim=1)
    return torch.where(has_eos, first_eos, non_pad_lengths)


def build_routing_plan(
    sequences: Tensor,
    *,
    task: RoutingTask = "reverse",
    eos_token: int = EOS,
    pad_token: int = PAD,
) -> RoutingPlan:
    """Create all output-to-input indices for a routing task in one tensor."""

    if sequences.dim() != 2:
        raise ValueError("sequences must have shape [batch, time]")
    if task not in {"reverse", "copy"}:
        raise ValueError(f"unsupported routing task: {task}")

    batch, width = sequences.shape
    device = sequences.device
    lengths = infer_lengths(sequences, eos_token=eos_token, pad_token=pad_token)
    positions = torch.arange(width, device=device).expand(batch, width)
    output_mask = positions < lengths.unsqueeze(1)

    if task == "copy":
        source_indices = positions
    else:
        last_content = (lengths - 2).clamp_min(0).unsqueeze(1)
        reversed_content = last_content - positions
        eos_position = (lengths - 1).unsqueeze(1)
        source_indices = torch.where(positions == eos_position, eos_position, reversed_content)
        source_indices = source_indices.clamp_min(0)

    source_indices = torch.where(output_mask, source_indices, torch.zeros_like(source_indices))
    return RoutingPlan(source_indices=source_indices, output_mask=output_mask)


def route_sequence(
    sequences: Tensor,
    *,
    task: RoutingTask = "reverse",
    eos_token: int = EOS,
    pad_token: int = PAD,
) -> Tensor:
    """Solve a sequence routing task with one vectorized gather operation."""

    plan = build_routing_plan(sequences, task=task, eos_token=eos_token, pad_token=pad_token)
    routed = sequences.gather(dim=1, index=plan.source_indices)
    return torch.where(plan.output_mask, routed, torch.full_like(routed, pad_token))

@dataclass(frozen=True)
class LazyRoutedSequence:
    """O(1)-setup routed view over one sequence.

    This view does not materialize the full answer. Creating it is O(1), asking
    for its length is O(1), and reading one output token is O(1). This is useful
    for partial/random access patterns only. Converting the whole view to a list
    is necessarily O(N), because N tokens must be written.
    """

    sequence: tuple[int, ...]
    length: int
    task: RoutingTask = "reverse"
    eos_token: int = EOS
    pad_token: int = PAD

    def __len__(self) -> int:
        return self.length

    def source_index(self, output_index: int) -> int:
        """Return the source index for one output index in O(1)."""

        if output_index < 0:
            output_index += self.length
        if output_index < 0 or output_index >= self.length:
            raise IndexError(output_index)
        if self.task == "copy":
            return output_index
        if self.task != "reverse":
            raise ValueError(f"unsupported routing task: {self.task}")
        if output_index == self.length - 1:
            return self.length - 1
        return self.length - 2 - output_index

    def __getitem__(self, output_index: int) -> int:
        return self.sequence[self.source_index(output_index)]

    def materialize(self, *, padded_to: int | None = None) -> list[int]:
        """Materialize the routed answer; this is intentionally O(N)."""

        values = [self[index] for index in range(self.length)]
        if padded_to is not None:
            values.extend([self.pad_token] * max(0, padded_to - len(values)))
        return values


def lazy_route_sequence(
    sequence: list[int] | tuple[int, ...],
    *,
    task: RoutingTask = "reverse",
    eos_token: int = EOS,
    pad_token: int = PAD,
) -> LazyRoutedSequence:
    """Create an O(1)-setup routed view for a single sequence."""

    immutable = tuple(sequence)
    try:
        length = immutable.index(eos_token) + 1
    except ValueError:
        length = sum(token != pad_token for token in immutable)
    return LazyRoutedSequence(
        sequence=immutable,
        length=length,
        task=task,
        eos_token=eos_token,
        pad_token=pad_token,
    )
