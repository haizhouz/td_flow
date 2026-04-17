from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from td_flow.toy.generate_circle_exploration_dataset import (
    GenerateToyCircleExplorationDatasetConfig,
    generate_toy_circle_exploration_dataset,
)


class GenerateToyCircleExplorationDatasetTests(unittest.TestCase):
    def test_generator_writes_expected_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toy_circle_exploration.h5"
            config = GenerateToyCircleExplorationDatasetConfig(
                output_hdf5_path=str(output_path),
                num_episodes=2,
                episode_length=5,
                policy_delta_theta=0.03,
                behavior_action_limit=0.2,
                seed=7,
            )
            generate_toy_circle_exploration_dataset(config)

            with h5py.File(output_path, "r") as handle:
                self.assertEqual(handle["observation"].shape, (10, 2))
                self.assertEqual(handle["action"].shape, (10, 1))
                self.assertEqual(handle["policy_action"].shape, (10, 1))
                np.testing.assert_array_equal(handle["ep_len"][:], np.array([5, 5], dtype=np.int64))
                np.testing.assert_array_equal(handle["ep_offset"][:], np.array([0, 5], dtype=np.int64))
                self.assertTrue(np.allclose(handle["policy_action"][:, 0], 0.03))
                self.assertTrue(np.all(np.abs(handle["action"][:, 0]) <= 0.2 + 1e-6))

                observations = np.asarray(handle["observation"][:], dtype=np.float64)
                actions = np.asarray(handle["action"][:, 0], dtype=np.float64)
                theta = np.arctan2(observations[:, 1], observations[:, 0])
                wrapped_dtheta = (theta[1:] - theta[:-1] + np.pi) % (2.0 * np.pi) - np.pi
                expected = actions[:-1]
                # Ignore the episode boundary transition between the two episodes.
                wrapped_dtheta = np.delete(wrapped_dtheta, 4)
                expected = np.delete(expected, 4)
                np.testing.assert_allclose(wrapped_dtheta, expected, atol=1e-5, rtol=1e-5)

    def test_disjoint_behavior_policy_excludes_target_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toy_circle_exploration_disjoint.h5"
            config = GenerateToyCircleExplorationDatasetConfig(
                output_hdf5_path=str(output_path),
                num_episodes=4,
                episode_length=16,
                policy_delta_theta=0.02,
                behavior_policy_kind="disjoint_uniform",
                behavior_action_limit=0.25,
                behavior_exclusion_radius=0.08,
                seed=11,
            )
            generate_toy_circle_exploration_dataset(config)

            with h5py.File(output_path, "r") as handle:
                actions = np.asarray(handle["action"][:, 0], dtype=np.float64)
                policy_actions = np.asarray(handle["policy_action"][:, 0], dtype=np.float64)
                self.assertTrue(np.allclose(policy_actions, 0.02))
                excluded_low = config.policy_delta_theta - config.behavior_exclusion_radius
                excluded_high = config.policy_delta_theta + config.behavior_exclusion_radius
                self.assertTrue(np.all((actions < excluded_low) | (actions > excluded_high)))

    def test_constant_behavior_policy_writes_fixed_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toy_circle_exploration_constant.h5"
            config = GenerateToyCircleExplorationDatasetConfig(
                output_hdf5_path=str(output_path),
                num_episodes=3,
                episode_length=8,
                policy_delta_theta=0.02,
                behavior_policy_kind="constant_delta_theta",
                behavior_delta_theta=-0.02,
                behavior_action_limit=0.25,
                seed=5,
            )
            generate_toy_circle_exploration_dataset(config)

            with h5py.File(output_path, "r") as handle:
                actions = np.asarray(handle["action"][:, 0], dtype=np.float64)
                policy_actions = np.asarray(handle["policy_action"][:, 0], dtype=np.float64)
                self.assertTrue(np.allclose(actions, -0.02))
                self.assertTrue(np.allclose(policy_actions, 0.02))


if __name__ == "__main__":
    unittest.main()
