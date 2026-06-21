import unittest

import torch

from index_routing_solver import EOS, PAD, build_routing_plan, infer_lengths, lazy_route_sequence, route_sequence


class IndexRoutingSolverTest(unittest.TestCase):
    def test_infer_lengths_stops_at_eos(self):
        sequences = torch.tensor([
            [7, 8, EOS, PAD, PAD],
            [4, 5, 6, PAD, PAD],
        ])

        self.assertEqual(infer_lengths(sequences).tolist(), [3, 3])

    def test_reverse_uses_single_vectorized_routing_plan(self):
        sequences = torch.tensor([
            [4, 7, 8, 9, EOS, PAD],
            [5, 6, EOS, PAD, PAD, PAD],
        ])

        plan = build_routing_plan(sequences, task="reverse")
        solved = route_sequence(sequences, task="reverse")

        self.assertEqual(plan.source_indices.tolist(), [
            [3, 2, 1, 0, 4, 0],
            [1, 0, 2, 0, 0, 0],
        ])
        self.assertEqual(solved.tolist(), [
            [9, 8, 7, 4, EOS, PAD],
            [6, 5, EOS, PAD, PAD, PAD],
        ])

    def test_copy_is_also_supported_by_same_routing_algorithm(self):
        sequences = torch.tensor([[4, 7, EOS, PAD]])
        self.assertEqual(route_sequence(sequences, task="copy").tolist(), [[4, 7, EOS, PAD]])

    def test_lazy_reverse_has_constant_time_index_mapping(self):
        view = lazy_route_sequence([4, 7, 8, 9, EOS, PAD], task="reverse")

        self.assertEqual(len(view), 5)
        self.assertEqual(view.source_index(0), 3)
        self.assertEqual(view.source_index(3), 0)
        self.assertEqual(view.source_index(4), 4)
        self.assertEqual(view[0], 9)
        self.assertEqual(view[-1], EOS)
        self.assertEqual(view.materialize(padded_to=6), [9, 8, 7, 4, EOS, PAD])


if __name__ == "__main__":
    unittest.main()
