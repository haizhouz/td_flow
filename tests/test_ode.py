import unittest
from unittest.mock import patch

import torch

from td_flow.ode import midpoint_integrate


class ODETest(unittest.TestCase):
    def test_midpoint_integrate_constant_velocity(self) -> None:
        source = torch.zeros(2, 3)
        velocity = torch.tensor([[1.0, -1.0, 2.0], [0.5, 0.25, -0.5]])
        t_end = torch.tensor([1.0, 0.5])

        def vf(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del x_t, t
            return velocity

        x_t = midpoint_integrate(vf, source, t_end, steps=10)
        expected = velocity * t_end.unsqueeze(-1)
        self.assertTrue(torch.allclose(x_t, expected, atol=1e-5))

    def test_midpoint_integrate_manual_fallback_matches_constant_velocity(self) -> None:
        source = torch.zeros(2, 3)
        velocity = torch.tensor([[1.0, -1.0, 2.0], [0.5, 0.25, -0.5]])
        t_end = torch.tensor([1.0, 0.5])

        def vf(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del x_t, t
            return velocity

        with patch("td_flow.ode.torch.compiler.is_compiling", return_value=True):
            x_t = midpoint_integrate(vf, source, t_end, steps=10)

        expected = velocity * t_end.unsqueeze(-1)
        self.assertTrue(torch.allclose(x_t, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
