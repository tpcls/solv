import unittest

import torch

from billion_transformer import BillionParameterTransformer, Transformer1BConfig, build_1b_model_on_meta


class BillionTransformerTest(unittest.TestCase):
    def test_1b_preset_parameter_count_uses_meta_device(self):
        model = build_1b_model_on_meta()
        parameters = model.parameter_count()

        self.assertGreater(parameters, 1_000_000_000)
        self.assertLess(parameters, 1_050_000_000)
        self.assertEqual(model.token_embedding.weight.device.type, "meta")

    def test_small_config_forward_shape(self):
        config = Transformer1BConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            d_ff=64,
        )
        model = BillionParameterTransformer(config)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        logits = model(input_ids)

        self.assertEqual(logits.shape, (1, 4, 32))

    def test_small_config_respects_requested_dtype(self):
        config = Transformer1BConfig(
            vocab_size=16,
            max_seq_len=4,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
        )
        model = BillionParameterTransformer(config, dtype=torch.bfloat16)

        self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
