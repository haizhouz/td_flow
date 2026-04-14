import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from td_flow.config import DataConfig
from td_flow.data import (
    OGBenchNPZDataset,
    build_td2_hdf5_dataset,
    compute_episode_lengths,
    summarize_episode_lengths,
)


class DataTest(unittest.TestCase):
    def test_ogbench_dataset_uses_next_action_until_terminal(self) -> None:
        dataset = OGBenchNPZDataset(
            {
                "observations": torch.randn(3, 4),
                "next_observations": torch.randn(3, 4),
                "actions": torch.tensor(
                    [
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                    ]
                ),
                "terminals": torch.tensor([0.0, 1.0, 0.0]),
            }
        )

        sample0 = dataset[0]
        sample1 = dataset[1]
        sample2 = dataset[2]

        self.assertTrue(torch.equal(sample0["next_action"], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.equal(sample1["next_action"], torch.zeros(2)))
        self.assertTrue(torch.equal(sample2["next_action"], torch.zeros(2)))

    def test_ogbench_dataset_supports_default_state_action_aliases(self) -> None:
        dataset = OGBenchNPZDataset(
            {
                "observations": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "next_observations": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "actions": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            },
            observation_key="state",
            action_key="action",
        )

        sample = dataset[0]
        self.assertTrue(torch.equal(sample["obs"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(sample["next_obs"], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(sample["action"], torch.tensor([0.1, 0.2])))

    def test_ogbench_dataset_honors_custom_keys(self) -> None:
        dataset = OGBenchNPZDataset(
            {
                "pixels": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "next_pixels": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "controls": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                "next_controls": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                "my_goal": torch.tensor([[9.0, 9.0], [8.0, 8.0]]),
                "policy_z": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
            },
            observation_key="pixels",
            action_key="controls",
            goal_key="my_goal",
            policy_embedding_key="policy_z",
        )

        sample = dataset[0]
        self.assertTrue(torch.equal(sample["obs"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(sample["next_obs"], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(sample["action"], torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.equal(sample["next_action"], torch.tensor([0.5, 0.6])))
        self.assertTrue(torch.equal(sample["goal"], torch.tensor([9.0, 9.0])))
        self.assertTrue(torch.equal(sample["policy_embedding"], torch.tensor([1.0, 1.5])))

    def test_ogbench_dataset_uses_explicit_next_action_key_when_provided(self) -> None:
        dataset = OGBenchNPZDataset(
            {
                "observations": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "next_observations": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "actions": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                "policy_actions": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
            },
            next_action_key="policy_actions",
        )

        sample = dataset[0]
        self.assertTrue(torch.equal(sample["next_action"], torch.tensor([0.9, 0.8])))

    def test_td2_dataset_returns_policy_embedding_when_requested(self) -> None:
        class _ToyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                del index
                return {
                    "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "action": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    "z": torch.tensor([0.5, 0.6, 0.7]),
                }

        from td_flow.data import TD2CFMDataset

        dataset = TD2CFMDataset(_ToyDataset(), policy_embedding_key="z")
        sample = dataset[0]
        self.assertIn("policy_embedding", sample)
        self.assertTrue(torch.equal(sample["policy_embedding"], torch.tensor([0.5, 0.6, 0.7])))

    def test_td2_dataset_uses_explicit_sequence_next_action_key(self) -> None:
        class _ToyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                del index
                return {
                    "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "action": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    "policy_action": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                }

        from td_flow.data import TD2CFMDataset

        dataset = TD2CFMDataset(_ToyDataset(), next_action_key="policy_action")
        sample = dataset[0]
        self.assertTrue(torch.equal(sample["next_action"], torch.tensor([0.7, 0.8])))

    def test_td2_hdf5_dataset_uses_next_action_key_on_second_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_name = "toy-hdf5"
            path = Path(tmpdir) / f"{dataset_name}.h5"
            observations = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                dtype=np.float32,
            )
            actions = np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                ],
                dtype=np.float32,
            )
            policy_actions = np.array(
                [
                    [0.9, 0.8],
                    [0.7, 0.6],
                    [0.5, 0.4],
                ],
                dtype=np.float32,
            )
            with h5py.File(path, "w") as handle:
                handle.create_dataset("observation", data=observations)
                handle.create_dataset("action", data=actions)
                handle.create_dataset("policy_action", data=policy_actions)
                handle.create_dataset("ep_len", data=np.array([3], dtype=np.int64))
                handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))

            dataset = build_td2_hdf5_dataset(
                DataConfig(
                    dataset_name=dataset_name,
                    backend="stablewm_hdf5",
                    dir=tmpdir,
                    observation_key="observation",
                    action_key="action",
                    next_action_key="policy_action",
                    num_steps=2,
                )
            )

            sample = dataset[0]

            self.assertTrue(torch.equal(sample["obs"], torch.tensor([1.0, 2.0, 3.0, 4.0])))
            self.assertTrue(torch.equal(sample["next_obs"], torch.tensor([5.0, 6.0, 7.0, 8.0])))
            self.assertTrue(torch.equal(sample["action"], torch.tensor([0.1, 0.2])))
            self.assertTrue(torch.equal(sample["next_action"], torch.tensor([0.7, 0.6])))

    def test_compute_episode_lengths_uses_terminal_boundaries(self) -> None:
        terminals = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(compute_episode_lengths(terminals), [2, 3, 1])

    def test_summarize_episode_lengths_reports_basic_stats(self) -> None:
        stats = summarize_episode_lengths(torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))
        self.assertEqual(stats["num_transitions"], 6)
        self.assertEqual(stats["num_episodes"], 3)
        self.assertEqual(stats["min_length"], 1)
        self.assertEqual(stats["max_length"], 3)
        self.assertAlmostEqual(stats["mean_length"], 2.0)


if __name__ == "__main__":
    unittest.main()
