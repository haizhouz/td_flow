import tempfile
import unittest
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from td_flow.config import DataConfig, ModelConfig, ProjectConfig, TrainConfig
from td_flow.train import build_callbacks, evaluate, resolve_run_dir


class TrainEntrypointTest(unittest.TestCase):
    def test_resolve_run_dir_uses_run_name(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(output_dir="outputs", run_name="paper-run"),
        )

        self.assertEqual(resolve_run_dir(project_config), Path("outputs") / "paper-run")

    def test_build_callbacks_monitors_validation_loss_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(output_dir=tmpdir),
            )

            callbacks = build_callbacks(project_config, Path(tmpdir), has_validation=True)
            checkpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
            self.assertEqual(checkpoint.monitor, "val_loss")

    def test_build_callbacks_disables_monitor_without_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(output_dir=tmpdir),
            )

            callbacks = build_callbacks(project_config, Path(tmpdir), has_validation=False)
            checkpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
            self.assertIsNone(checkpoint.monitor)

    def test_evaluate_requires_checkpoint_path(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(run_mode="validate", resume_ckpt_path=None),
        )

        with self.assertRaises(ValueError):
            evaluate(project_config)


if __name__ == "__main__":
    unittest.main()
