"""Train a small Transformer to solve an algorithmic sequence problem.

The default task is sequence reversal.  Given an input such as::

    4 7 2 9

The model learns to emit all answer positions in parallel::

    9 2 7 4

Unlike autoregressive decoding, this non-autoregressive formulation does not run
a new Transformer pass for every output token. It encodes the source once, then
uses positional output queries to predict the whole answer in one decoder pass.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise SystemExit(
        "PyTorch is required. Install torch for your platform, then rerun this script."
    ) from exc

PAD = 0
BOS = 1
EOS = 2
FIRST_VALUE_TOKEN = 3


@dataclass(frozen=True)
class SequenceExample:
    """One synthetic sequence-to-sequence training example."""

    source: list[int]
    target: list[int]


class ReverseDataset(Dataset[SequenceExample]):
    """Synthetic dataset for the integer sequence reversal task."""

    def __init__(self, samples: int, *, max_len: int, vocab_values: int, seed: int) -> None:
        self.examples: list[SequenceExample] = []
        rng = random.Random(seed)
        for _ in range(samples):
            length = rng.randint(2, max_len)
            values = [rng.randrange(FIRST_VALUE_TOKEN, FIRST_VALUE_TOKEN + vocab_values) for _ in range(length)]
            self.examples.append(SequenceExample(source=values, target=list(reversed(values))))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SequenceExample:
        return self.examples[index]


def collate_batch(examples: list[SequenceExample]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad examples and build parallel output targets.

    The target length is the source length plus EOS.  There is no decoder-input
    token sequence because the model predicts every output position from learned
    positional queries instead of feeding previous answer tokens back into the
    model.
    """

    max_source = max(len(example.source) for example in examples) + 1

    sources: list[list[int]] = []
    targets: list[list[int]] = []
    lengths: list[int] = []
    for example in examples:
        source = example.source + [EOS]
        target = example.target + [EOS]
        length = len(target)

        sources.append(source + [PAD] * (max_source - len(source)))
        targets.append(target + [PAD] * (max_source - len(target)))
        lengths.append(length)

    return (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the original Transformer paper."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.encoding[:, : tokens.size(1)]


class TransformerSolver(nn.Module):
    """Non-autoregressive Transformer for symbolic sequence problems.

    The encoder reads the input sequence once. A bank of positional output
    queries is then decoded in parallel against that memory, so inference uses a
    constant number of Transformer passes regardless of answer length.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, source: Tensor, output_length: int) -> Tensor:
        source_padding = source.eq(PAD)
        encoded_source = self.embedding(source) * math.sqrt(self.d_model) + self.position(source)
        memory = self.encoder(encoded_source, src_key_padding_mask=source_padding)

        query_tokens = torch.zeros(source.size(0), output_length, dtype=torch.long, device=source.device)
        output_queries = self.position(query_tokens).expand(source.size(0), -1, -1)
        hidden = self.decoder(
            output_queries,
            memory,
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)

def train_epoch(model: TransformerSolver, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    for source, target_output, _ in loader:
        source = source.to(device)
        target_output = target_output.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(source, target_output.size(1))
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def exact_match_accuracy(model: TransformerSolver, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    matches = 0
    total = 0
    for source, target_output, _ in loader:
        source = source.to(device)
        target_output = target_output.to(device)
        generated = parallel_decode(model, source, output_length=target_output.size(1))
        for prediction, expected in zip(generated.cpu().tolist(), target_output.cpu().tolist(), strict=True):
            prediction_trimmed = prediction[: prediction.index(EOS) + 1] if EOS in prediction else prediction
            expected_trimmed = expected[: expected.index(EOS) + 1]
            matches += int(prediction_trimmed == expected_trimmed)
            total += 1
    return matches / max(total, 1)


@torch.no_grad()
def parallel_decode(model: TransformerSolver, source: Tensor, *, output_length: int) -> Tensor:
    model.eval()
    return model(source, output_length).argmax(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=1_024)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--vocab-values", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab_size = FIRST_VALUE_TOKEN + args.vocab_values

    train_data = ReverseDataset(args.train_samples, max_len=args.max_len, vocab_values=args.vocab_values, seed=args.seed)
    val_data = ReverseDataset(args.val_samples, max_len=args.max_len, vocab_values=args.vocab_values, seed=args.seed + 1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_batch)

    model = TransformerSolver(vocab_size, max_len=args.max_len + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        accuracy = exact_match_accuracy(model, val_loader, device)
        print(f"epoch={epoch} loss={loss:.4f} exact_match={accuracy:.3f}")


if __name__ == "__main__":
    main()
