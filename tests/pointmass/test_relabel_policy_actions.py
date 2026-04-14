from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from td_flow.pointmass.relabel_policy_actions import (
    RelabelPointMassPolicyActionsConfig,
    _sample_bounded_relative_noise,
    relabel_pointmass_policy_actions,
)


class RelabelPointMassPolicyActionsTests(unittest.TestCase):
    def test_relative_noise_is_bounded_by_action_norm_fraction(self) -> None:
        actions = torch.tensor(
            [
                [1.0, 0.0],
                [0.6, 0.8],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)

        noise = _sample_bounded_relative_noise(
            action_chunk=actions,
            max_noise_fraction=0.1,
            generator=generator,
        )

        noise_norm = torch.linalg.vector_norm(noise, dim=-1)
        action_norm = torch.linalg.vector_norm(actions, dim=-1)
        self.assertTrue(torch.all(noise_norm <= 0.1 * action_norm + 1e-6))
        self.assertEqual(float(noise_norm[-1]), 0.0)

    def test_relabel_writes_policy_actions_contiguously(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.h5"
            output_path = tmp_path / "output.h5"

            with h5py.File(input_path, "w") as handle:
                handle.create_dataset(
                    "observation",
                    data=np.array(
                        [
                            [0.0, 0.2, 0.0, 0.0],
                            [0.1, 0.1, 0.0, 0.0],
                            [0.2, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                )
                handle.create_dataset("action", data=np.zeros((3, 2), dtype=np.float32))

            config = RelabelPointMassPolicyActionsConfig(
                input_hdf5_path=str(input_path),
                output_hdf5_path=str(output_path),
                device="cpu",
                max_noise_fraction=0.0,
                chunk_size=2,
            )
            relabel_pointmass_policy_actions(config)

            with h5py.File(output_path, "r") as handle:
                self.assertIn("policy_action", handle)
                self.assertIsNone(handle["policy_action"].chunks)


if __name__ == "__main__":
    unittest.main()
