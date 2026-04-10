import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lightning.pytorch.callbacks import ModelCheckpoint

from td_flow.config import DataConfig, ModelConfig, ProjectConfig, TrainConfig
from td_flow.train import (
    apply_runtime_acceleration,
    build_callbacks,
    configure_compile_cache,
    evaluate,
    resolve_compile_cache_artifact_path,
    resolve_run_dir,
    save_compile_cache,
    train,
)


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

    def test_evaluate_compiles_module_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(
                    run_mode="validate",
                    compile=True,
                    output_dir=tmpdir,
                    resume_ckpt_path="dummy.ckpt",
                ),
            )
            module = MagicMock()
            trainer = MagicMock()
            trainer.validate.return_value = []

            with patch("td_flow.train.build_data_module", return_value=MagicMock()), patch(
                "td_flow.train.build_training_module", return_value=module
            ), patch("td_flow.train.build_loggers", return_value=False), patch(
                "td_flow.train.build_callbacks", return_value=[]
            ), patch(
                "td_flow.train.build_trainer", return_value=trainer
            ), patch(
                "td_flow.train.pl.seed_everything"
            ):
                evaluate(project_config)

            module.compile.assert_called_once_with()

    def test_train_forwards_compile_flag_to_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(output_dir=tmpdir, compile=True),
            )
            manager = MagicMock()

            with patch("td_flow.train.save_project_config"), patch(
                "td_flow.train.build_data_module", return_value=MagicMock(val_dataloader=MagicMock(return_value=None))
            ), patch(
                "td_flow.train.build_training_module", return_value=MagicMock()
            ), patch(
                "td_flow.train.build_loggers", return_value=False
            ), patch(
                "td_flow.train.build_callbacks", return_value=[]
            ), patch(
                "td_flow.train.build_trainer", return_value=MagicMock()
            ), patch(
                "td_flow.train.spt.Manager", return_value=manager
            ) as manager_cls:
                returned_manager = train(project_config)

            self.assertIs(returned_manager, manager)
            manager_cls.assert_called_once()
            self.assertTrue(manager_cls.call_args.kwargs["compile"])
            manager.assert_called_once_with()

    def test_apply_runtime_acceleration_sets_tf32_for_high_precision(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(matmul_precision="high"),
        )

        with patch("td_flow.train.torch.set_float32_matmul_precision") as set_precision, patch(
            "td_flow.train.torch.cuda.is_available", return_value=True
        ), patch("td_flow.train.torch.backends.cuda.matmul", create=True) as cuda_matmul, patch(
            "td_flow.train.torch.backends.cudnn", create=True
        ) as cudnn:
            apply_runtime_acceleration(project_config)

        set_precision.assert_called_once_with("high")
        self.assertTrue(cuda_matmul.allow_tf32)
        self.assertTrue(cudnn.allow_tf32)

    def test_configure_compile_cache_sets_env_and_loads_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "cube-single-play-v0" / "cache_artifacts.bin"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"cached")
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(compile=True, compile_cache_dir=tmpdir),
            )

            with patch("td_flow.train.torch.compiler.load_cache_artifacts") as load_cache:
                resolved = configure_compile_cache(project_config)

            self.assertEqual(resolved, artifact_path)
            self.assertEqual(os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
            self.assertEqual(os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"], "1")
            self.assertEqual(
                os.environ["TORCHINDUCTOR_CACHE_DIR"],
                str((Path(tmpdir) / "cube-single-play-v0").resolve()),
            )
            load_cache.assert_called_once_with(b"cached")

    def test_save_compile_cache_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "cache_artifacts.bin"
            with patch(
                "td_flow.train.torch.compiler.save_cache_artifacts",
                return_value=(b"payload", MagicMock()),
            ) as save_cache:
                save_compile_cache(artifact_path)

            save_cache.assert_called_once_with()
            self.assertEqual(artifact_path.read_bytes(), b"payload")

    def test_resolve_compile_cache_artifact_path_uses_override_name(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(
                compile=True,
                compile_cache_dir="/tmp/cache-root",
                compile_cache_name="manual-name",
            ),
        )

        self.assertEqual(
            resolve_compile_cache_artifact_path(project_config),
            Path("/tmp/cache-root") / "manual-name" / "cache_artifacts.bin",
        )


if __name__ == "__main__":
    unittest.main()
