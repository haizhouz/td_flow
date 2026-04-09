import unittest

import torch

from td_flow.data import OGBenchNPZDataset


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


if __name__ == "__main__":
    unittest.main()
