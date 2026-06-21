import unittest

import torch

from transformer_solver import (
    EOS,
    FIRST_VALUE_TOKEN,
    ReverseDataset,
    TransformerSolver,
    collate_batch,
    parallel_decode,
)


class TransformerSolverTest(unittest.TestCase):
    def test_collate_builds_reversed_parallel_targets(self):
        dataset = ReverseDataset(samples=2, max_len=4, vocab_values=5, seed=123)
        source, target, lengths = collate_batch([dataset[0], dataset[1]])

        self.assertEqual(source.shape, target.shape)
        for index, example in enumerate([dataset[0], dataset[1]]):
            length = lengths[index].item()
            self.assertEqual(target[index, :length].tolist(), list(reversed(example.source)) + [EOS])

    def test_model_predicts_all_positions_in_one_forward_pass(self):
        torch.manual_seed(0)
        dataset = ReverseDataset(samples=3, max_len=4, vocab_values=5, seed=7)
        source, target, _ = collate_batch([dataset[0], dataset[1], dataset[2]])
        model = TransformerSolver(
            FIRST_VALUE_TOKEN + 5,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_len=target.size(1),
        )

        logits = model(source, target.size(1))
        decoded = parallel_decode(model, source, output_length=target.size(1))

        self.assertEqual(logits.shape, (source.size(0), target.size(1), FIRST_VALUE_TOKEN + 5))
        self.assertEqual(decoded.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
