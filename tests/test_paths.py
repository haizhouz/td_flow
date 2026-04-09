import unittest

import torch

from td_flow.paths import sample_linear_probability_path


class PathsTest(unittest.TestCase):
    def test_linear_probability_path_matches_closed_form(self) -> None:
        source = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[2.0, 5.0]])
        t = torch.tensor([0.25])
        xt, ut = sample_linear_probability_path(source, target, t)

        expected_xt = torch.tensor([[0.5, 2.0]])
        expected_ut = torch.tensor([[2.0, 4.0]])

        self.assertTrue(torch.allclose(xt, expected_xt))
        self.assertTrue(torch.allclose(ut, expected_ut))


if __name__ == "__main__":
    unittest.main()

