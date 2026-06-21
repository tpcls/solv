import tempfile
import unittest
from pathlib import Path

import torch

from train_billion_transformer import TrainingConfig, model_config, train


class TrainBillionTransformerTest(unittest.TestCase):
    def test_small_training_smoke_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train(
                TrainingConfig(
                    model_size="small",
                    steps=1,
                    batch_size=1,
                    seq_len=8,
                    dtype=torch.float32,
                    device="cpu",
                    checkpoint_dir=tmpdir,
                )
            )

            self.assertTrue(Path(tmpdir, "step_000001.pt").exists())

    def test_model_config_selects_1b_preset(self):
        config = model_config("1b")

        self.assertEqual(config.d_model, 2048)
        self.assertEqual(config.n_layers, 18)


if __name__ == "__main__":
    unittest.main()
