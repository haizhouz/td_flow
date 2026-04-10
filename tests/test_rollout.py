import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from td_flow.rollout import (
    decode_cube_single_state,
    find_valid_start,
    load_project_config_from_run_dir,
)


class RolloutTest(unittest.TestCase):
    def test_find_valid_start_rejects_terminal_crossing(self) -> None:
        terminals = np.array([0, 0, 1, 0, 0], dtype=bool)
        with self.assertRaises(ValueError):
            find_valid_start(terminals, horizon=3, start_index=1, seed=0)

    def test_find_valid_start_accepts_valid_index(self) -> None:
        terminals = np.array([0, 0, 0, 0, 0], dtype=bool)
        self.assertEqual(find_valid_start(terminals, horizon=3, start_index=1, seed=0), 1)

    def test_decode_cube_single_state_updates_render_relevant_fields(self) -> None:
        observation = np.zeros(28, dtype=np.float32)
        observation[:6] = np.arange(1, 7, dtype=np.float32)
        observation[6:12] = np.arange(11, 17, dtype=np.float32)
        observation[17] = 1.5
        observation[19:22] = np.array([1.0, -2.0, 0.2], dtype=np.float32)
        observation[22:26] = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_qpos = np.zeros(21, dtype=np.float32)
        base_qvel = np.ones(20, dtype=np.float32)

        qpos, qvel = decode_cube_single_state(observation, base_qpos, base_qvel)

        np.testing.assert_allclose(qpos[:6], observation[:6])
        np.testing.assert_allclose(qvel[:6], observation[6:12])
        self.assertAlmostEqual(qpos[6], 0.4)
        np.testing.assert_allclose(
            qpos[14:17],
            np.array([0.525, -0.2, 0.02], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(qpos[17:21], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(qvel[14:20], np.zeros(6, dtype=np.float32))

    def test_load_project_config_from_run_dir_supports_legacy_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            payload = {
                "data": {
                    "dataset_name": "cube-single-play-v0",
                    "backend": "ogbench_npz",
                    "split": "train",
                    "observation_key": "state",
                    "action_key": "action",
                    "goal_key": None,
                    "policy_embedding_key": None,
                    "batch_size": 64,
                    "num_workers": 0,
                    "frameskip": 1,
                    "num_steps": 2,
                    "cache_dir": "/tmp/ogbench",
                    "keys_to_load": [],
                },
                "model": {
                    "observation_shape": [28],
                    "action_dim": 5,
                    "backbone": {
                        "kind": "mlp",
                        "hidden_dims": [256, 256],
                        "torchvision_name": "resnet18",
                        "low_resolution": False,
                    },
                    "observation_encoder": "identity",
                    "network_variant": "paper",
                    "latent_dim": 28,
                    "policy_embedding_dim": 0,
                    "context_dim": 128,
                    "context_hidden_dims": [256, 256],
                    "vector_field_hidden_dims": [256, 256],
                    "time_embed_dim": 256,
                    "gamma": 0.99,
                    "polyak": None,
                    "ode_steps": 10,
                    "time_eps": 1e-4,
                    "policy_mode": "single_policy",
                },
                "train": {
                    "run_mode": "fit",
                    "compile": False,
                    "cache_root": ".cache/td_flow",
                    "matmul_precision": "high",
                    "compile_cache_name": None,
                    "train_semantics": "paper",
                    "lr": 1e-4,
                    "weight_decay": None,
                    "scheduler": None,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_eps": 1e-4,
                    "max_steps": 1,
                    "max_epochs": None,
                    "val_check_interval": None,
                    "accelerator": "auto",
                    "devices": "auto",
                    "precision": "32-true",
                    "log_every_n_steps": 10,
                    "enable_progress_bar": False,
                    "seed": 0,
                    "output_dir": "outputs",
                    "run_name": "demo",
                    "use_csv_logger": True,
                    "enable_checkpointing": True,
                    "checkpoint_monitor": "val_loss",
                    "checkpoint_mode": "min",
                    "checkpoint_save_top_k": 1,
                    "checkpoint_every_n_train_steps": 10000,
                    "checkpoint_save_last": True,
                    "resume": False,
                    "resume_ckpt_path": None,
                    "limit_train_batches": None,
                    "limit_val_batches": 0,
                    "use_wandb": False,
                    "wandb_project": "td_flow",
                    "wandb_name": None,
                    "wandb_entity": None,
                    "wandb_group": None,
                    "wandb_tags": [],
                    "wandb_notes": None,
                    "wandb_id": None,
                    "wandb_resume": None,
                    "wandb_save_dir": None,
                    "wandb_offline": False,
                    "wandb_log_model": False,
                },
                "planning": {},
            }
            (run_dir / "project_config.json").write_text(json.dumps(payload))
            project_config = load_project_config_from_run_dir(run_dir)
            self.assertEqual(project_config.data.dir, "/tmp/ogbench")


if __name__ == "__main__":
    unittest.main()
