from __future__ import annotations

import contextlib
import tempfile
import unittest

import h5py
import numpy as np
import torch

from td_flow.pointmass.plot_policy_conditioned_occupancy import (
    _dataset_action_tensor,
    _dataset_future_positions,
    _resolve_baseline_mode,
    _sample_stochastic_rollout_positions,
    _set_env_state_from_observation,
)
from td_flow.pointmass.loop_policy import TorchPointMassLoopScriptedPolicy


class _FakePhysics:
    def __init__(self) -> None:
        self.data = type("Data", (), {})()
        self.data.qpos = np.zeros(2, dtype=np.float32)
        self.data.qvel = np.zeros(2, dtype=np.float32)
        self.state = None

    @contextlib.contextmanager
    def reset_context(self):
        yield

    def set_state(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()


class _FakeEnv:
    def __init__(self) -> None:
        self.physics = _FakePhysics()
        self.reset_calls = 0
        self.actions: list[np.ndarray] = []
        self.current_observation = np.zeros(4, dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return type(
            "TimeStep",
            (),
            {"observation": {"observations": self.current_observation.copy()}},
        )()


class PlotPointMassPolicyConditionedOccupancyTests(unittest.TestCase):
    def test_set_env_state_resets_before_restoring_observation_state(self) -> None:
        env = _FakeEnv()
        observation = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        _set_env_state_from_observation(env, observation)

        self.assertEqual(env.reset_calls, 1)
        np.testing.assert_allclose(env.physics.data.qpos, observation[:2])
        np.testing.assert_allclose(env.physics.data.qvel, observation[2:4])

    def test_set_env_state_prefers_full_physics_state_when_available(self) -> None:
        env = _FakeEnv()
        observation = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        physics_state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        _set_env_state_from_observation(env, observation, physics_state)

        self.assertEqual(env.reset_calls, 1)
        np.testing.assert_allclose(env.physics.state, physics_state)
        np.testing.assert_allclose(env.physics.data.qpos, np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(env.physics.data.qvel, np.zeros(2, dtype=np.float32))

    def test_stochastic_rollout_uses_shared_initial_conditioning_action(self) -> None:
        env = _FakeEnv()
        observation = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        env.current_observation = observation.copy()
        initial_action = np.array([0.7, -0.25], dtype=np.float32)

        _sample_stochastic_rollout_positions(
            env,
            observation,
            rollout_steps=1,
            rollout_count=3,
            policy=TorchPointMassLoopScriptedPolicy(),
            device=torch.device("cpu"),
            initial_action=initial_action,
            max_noise_fraction=0.1,
            gamma=0.99,
            sample_count=8,
            seed=0,
        )

        self.assertEqual(len(env.actions), 3)
        for action in env.actions:
            np.testing.assert_allclose(action, initial_action)

    def test_dataset_action_tensor_uses_logged_action_exactly(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as handle:
                handle.create_dataset(
                    "action",
                    data=np.array(
                        [
                            [0.1, -0.2],
                            [0.7, 0.3],
                        ],
                        dtype=np.float32,
                    ),
                )
                action_tensor = _dataset_action_tensor(handle["action"], 1, device=torch.device("cpu"))

        np.testing.assert_allclose(action_tensor.numpy(), np.array([[0.7, 0.3]], dtype=np.float32))

    def test_dataset_future_positions_excludes_current_state(self) -> None:
        observations = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        future_positions = _dataset_future_positions(
            observations,
            ep_offset=np.array([0], dtype=np.int64),
            ep_len=np.array([4], dtype=np.int64),
            start_index=1,
        )

        np.testing.assert_allclose(
            future_positions,
            np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        )

    def test_auto_baseline_treats_missing_next_action_key_as_dataset_episode(self) -> None:
        project_config = type(
            "ProjectConfig",
            (),
            {
                "data": type(
                    "DataConfig",
                    (),
                    {
                        "action_key": "action",
                        "next_action_key": None,
                    },
                )()
            },
        )()

        self.assertEqual(_resolve_baseline_mode(project_config, "auto"), "dataset_episode")


if __name__ == "__main__":
    unittest.main()
