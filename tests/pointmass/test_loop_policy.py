from __future__ import annotations

import unittest

import numpy as np
import torch

from td_flow.pointmass.loop_policy import (
    PointMassLoopPolicyConfig,
    TorchPointMassLoopScriptedPolicy,
    desired_loop_velocity,
    loop_outward_normal,
    loop_tangent,
    scripted_pointmass_loop_action,
    torch_scripted_pointmass_loop_action,
)


class PointMassLoopPolicyTests(unittest.TestCase):
    def test_loop_tangent_matches_clockwise_loop_direction(self) -> None:
        np.testing.assert_allclose(loop_tangent(-0.2, 0.2), np.array([1.0, 1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_tangent(0.2, 0.2), np.array([1.0, -1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_tangent(0.2, -0.2), np.array([-1.0, -1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_tangent(-0.2, -0.2), np.array([-1.0, 1.0]) / np.sqrt(2.0))

    def test_outward_normal_points_away_from_diamond_center(self) -> None:
        np.testing.assert_allclose(loop_outward_normal(-0.2, 0.2), np.array([-1.0, 1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_outward_normal(0.2, 0.2), np.array([1.0, 1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_outward_normal(0.2, -0.2), np.array([1.0, -1.0]) / np.sqrt(2.0))
        np.testing.assert_allclose(loop_outward_normal(-0.2, -0.2), np.array([-1.0, -1.0]) / np.sqrt(2.0))

    def test_desired_velocity_is_tangent_on_the_nominal_loop(self) -> None:
        config = PointMassLoopPolicyConfig(diamond_radius=0.24, target_speed=0.10, radial_gain=1.0)
        observation = np.array([-0.12, 0.12, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            desired_loop_velocity(observation, config=config),
            np.array([0.10, 0.10], dtype=np.float32) / np.sqrt(2.0),
            atol=1e-6,
        )

    def test_action_is_clipped_and_two_dimensional(self) -> None:
        observation = np.array([-0.29, 0.29, -2.0, 2.0], dtype=np.float32)
        action = scripted_pointmass_loop_action(observation)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(action <= 1.0))
        self.assertTrue(np.all(action >= -1.0))

    def test_torch_policy_matches_numpy_policy(self) -> None:
        observations = np.array(
            [
                [-0.20, 0.20, 0.01, -0.02],
                [0.20, 0.20, 0.03, 0.01],
                [0.20, -0.20, -0.01, -0.04],
            ],
            dtype=np.float32,
        )
        expected = np.stack([scripted_pointmass_loop_action(obs) for obs in observations], axis=0)
        actual = torch_scripted_pointmass_loop_action(torch.from_numpy(observations)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_torch_policy_module_supports_batched_forward(self) -> None:
        policy = TorchPointMassLoopScriptedPolicy(PointMassLoopPolicyConfig())
        observations = torch.tensor(
            [[-0.2, 0.2, 0.0, 0.0], [0.2, -0.2, 0.0, 0.0]],
            dtype=torch.float32,
        )
        actions = policy(observations)
        self.assertEqual(tuple(actions.shape), (2, 2))
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))


if __name__ == "__main__":
    unittest.main()
