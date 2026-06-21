"""Configurable decoder-only Transformer, including a ~1B parameter preset.

The default preset is intentionally created for architecture definition and
parameter accounting. Instantiate it on the ``meta`` device first when you only
need to inspect the model shape or parameter count without allocating weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Transformer1BConfig:
    """Architecture settings for the approximately one-billion-parameter model."""

    vocab_size: int = 50_000
    max_seq_len: int = 2_048
    d_model: int = 2_048
    n_layers: int = 18
    n_heads: int = 16
    d_ff: int = 8_192
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention block."""

    def __init__(self, config: Transformer1BConfig, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False, device=device, dtype=dtype)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False, device=device, dtype=dtype)
        self.dropout = config.dropout

    def forward(self, hidden: Tensor) -> Tensor:
        batch, seq_len, width = hidden.shape
        qkv = self.qkv(hidden)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.out(attended)


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, config: Transformer1BConfig, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(self.dropout(F.gelu(self.up(hidden), approximate="tanh")))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, config: Transformer1BConfig, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(config, device=device, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(config.d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(config, device=device, dtype=dtype)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class BillionParameterTransformer(nn.Module):
    """Decoder-only Transformer language model with a ~1B parameter preset."""

    def __init__(self, config: Transformer1BConfig = Transformer1BConfig(), *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, device=device, dtype=dtype)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([DecoderBlock(config, device=device, dtype=dtype) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=device, dtype=dtype)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters, respecting tied weights."""

        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            identifier = id(parameter)
            if identifier in seen:
                continue
            seen.add(identifier)
            total += parameter.numel()
        return total


def build_1b_model_on_meta() -> BillionParameterTransformer:
    """Build the 1B preset on meta tensors for zero-allocation inspection."""

    return BillionParameterTransformer(device="meta")


def build_1b_model(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> BillionParameterTransformer:
    """Materialize the 1B preset with real weights.

    The default dtype is bfloat16 to keep parameter memory around 2 GiB instead
    of around 4 GiB for float32. Optimizer state and training activations would
    require substantially more memory.
    """

    return BillionParameterTransformer(device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true", help="allocate the real 1B model instead of meta tensors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


if __name__ == "__main__":
    args = parse_args()
    model = (
        build_1b_model(device=args.device, dtype=dtype_from_name(args.dtype))
        if args.materialize
        else build_1b_model_on_meta()
    )
    print(f"parameters={model.parameter_count():,}")
    print(f"device={next(model.parameters()).device}")
    print(f"dtype={next(model.parameters()).dtype}")
