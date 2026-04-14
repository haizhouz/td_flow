from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from td_flow.toy.generate_circle_policy_dataset import (
    GenerateToyCirclePolicyDatasetConfig,
    generate_toy_circle_policy_dataset,
)


class GenerateToyCirclePolicyDatasetTests(unittest.TestCase):
    def test_generator_writes_expected_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toy_circle.h5"
            config = GenerateToyCirclePolicyDatasetConfig(
                output_hdf5_path=str(output_path),
                num_episodes=3,
                episode_length=4,
                delta_theta=0.25,
                seed=7,
            )
            generate_toy_circle_policy_dataset(config)

            with h5py.File(output_path, "r") as handle:
                self.assertEqual(handle["observation"].shape, (12, 2))
                self.assertEqual(handle["action"].shape, (12, 1))
                self.assertEqual(handle["reward"].shape, (12, 1))
                self.assertEqual(handle["discount"].shape, (12, 1))
                np.testing.assert_array_equal(handle["ep_len"][:], np.array([4, 4, 4], dtype=np.int64))
                np.testing.assert_array_equal(handle["ep_offset"][:], np.array([0, 4, 8], dtype=np.int64))
                self.assertTrue(np.allclose(handle["action"][:, 0], 0.25))


if __name__ == "__main__":
    unittest.main()
