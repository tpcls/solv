"""Cloud-oriented training entry point for the 1B Transformer.

This script is intentionally usable in two modes:

* ``--model-size small`` for local smoke tests in seconds.
* ``--model-size 1b`` for cloud runs with real accelerators.

It uses synthetic token batches by default so infrastructure can be validated
without adding a tokenizer or dataset dependency. Replace ``sample_batch`` with a
real tokenized dataset loader for production training.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributed import destroy_process_group, init_process_group, is_initialized
from torch.nn.parallel import DistributedDataParallel

from billion_transformer import BillionParameterTransformer, Transformer1BConfig, dtype_from_name


@dataclass(frozen=True)
class TrainingConfig:
    model_size: str = "small"
    steps: int = 10
    batch_size: int = 2
    seq_len: int = 64
    lr: float = 3e-4
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints/one_b"
    seed: int = 7


def small_config() -> Transformer1BConfig:
    """Return a compact config for smoke tests and CI."""

    return Transformer1BConfig(
        vocab_size=512,
        max_seq_len=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
    )


def model_config(name: str) -> Transformer1BConfig:
    if name == "small":
        return small_config()
    if name == "1b":
        return Transformer1BConfig()
    raise ValueError(f"unknown model size: {name}")


def distributed_context() -> tuple[int, int, int]:
    """Initialize torch.distributed when launched with torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_process_group(backend=backend)
    return world_size, rank, local_rank


def sample_batch(config: TrainingConfig, model_cfg: Transformer1BConfig, device: torch.device) -> tuple[Tensor, Tensor]:
    """Generate a next-token-prediction batch for training-loop validation."""

    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(config.batch_size, config.seq_len + 1),
        device=device,
    )
    return tokens[:, :-1], tokens[:, 1:]


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: str) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "step": step,
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path / f"step_{step:06d}.pt",
    )


def train(config: TrainingConfig) -> None:
    world_size, rank, local_rank = distributed_context()
    if config.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.device)

    torch.manual_seed(config.seed + rank)
    model_cfg = model_config(config.model_size)
    model = BillionParameterTransformer(model_cfg, device=device, dtype=config.dtype)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for step in range(1, config.steps + 1):
        inputs, targets = sample_batch(config, model_cfg, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=config.dtype, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"step={step} loss={loss.item():.4f} parameters={model.module.parameter_count() if hasattr(model, 'module') else model.parameter_count():,}")

    if rank == 0:
        save_checkpoint(model, optimizer, config.steps, config.checkpoint_dir)
    if is_initialized():
        destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "1b"), default="small")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/one_b")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        TrainingConfig(
            model_size=args.model_size,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            dtype=dtype_from_name(args.dtype),
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
