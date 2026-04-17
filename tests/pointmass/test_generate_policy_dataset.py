from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from td_flow.pointmass.generate_policy_dataset import (
    GeneratePointMassPolicyDatasetConfig,
    generate_pointmass_policy_dataset,
)


class _FakeTimeStep:
    def __init__(self, observation: np.ndarray, reward: float | None = None, discount: float | None = 1.0) -> None:
        self.observation = {"observations": observation}
        self.reward = reward
        self.discount = discount


class _FakePhysics:
    def __init__(self, env: "_FakeEnv") -> None:
        self._env = env

    def get_state(self) -> np.ndarray:
        return self._env._obs.copy()


class _FakeEnv:
    def __init__(self) -> None:
        self._step = 0
        self._obs = np.array([-0.2, 0.2, 0.0, 0.0], dtype=np.float32)
        self.physics = _FakePhysics(self)

    def reset(self) -> _FakeTimeStep:
        self._step = 0
        self._obs = np.array([-0.2, 0.2, 0.0, 0.0], dtype=np.float32)
        return _FakeTimeStep(self._obs.copy(), reward=None, discount=None)

    def step(self, action: np.ndarray) -> _FakeTimeStep:
        self._step += 1
        self._obs = self._obs + np.array([action[0], action[1], 0.0, 0.0], dtype=np.float32) * 0.01
        return _FakeTimeStep(self._obs.copy(), reward=float(self._step), discount=1.0)


class GeneratePointMassPolicyDatasetTests(unittest.TestCase):
    def test_generator_writes_policy_rollout_hdf5(self) -> None:
        fake_pointmass = types.SimpleNamespace(loop=lambda **_: _FakeEnv())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "policy_dataset.h5"
            config = GeneratePointMassPolicyDatasetConfig(
                output_hdf5_path=str(output_path),
                num_episodes=2,
                episode_length=3,
                episode_modes=("straight", "circle"),
            )
            with mock.patch(
                "td_flow.pointmass.generate_policy_dataset._load_pointmass_module",
                return_value=fake_pointmass,
            ):
                generate_pointmass_policy_dataset(config)

            with h5py.File(output_path, "r") as handle:
                self.assertEqual(handle["observation"].shape, (6, 4))
                self.assertEqual(handle["action"].shape, (6, 2))
                self.assertEqual(handle["reward"].shape, (6, 1))
                self.assertEqual(handle["discount"].shape, (6, 1))
                self.assertEqual(handle["physics"].shape, (6, 4))
                self.assertEqual(handle["policy_mode_id"].shape, (6, 1))
                np.testing.assert_array_equal(handle["ep_len"][:], np.array([3, 3], dtype=np.int64))
                np.testing.assert_array_equal(handle["ep_offset"][:], np.array([0, 3], dtype=np.int64))
                np.testing.assert_array_equal(
                    handle["policy_mode_id"][:].reshape(-1),
                    np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
                )


if __name__ == "__main__":
    unittest.main()
