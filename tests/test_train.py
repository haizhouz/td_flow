import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from td_flow.config import DataConfig, ModelConfig, ProjectConfig, TrainConfig
from td_flow.train import (
    ThroughputCallback,
    apply_runtime_acceleration,
    build_callbacks,
    build_loggers,
    configure_compile_cache,
    evaluate,
    resolve_cache_root,
    resolve_cache_run_dir,
    resolve_compile_cache_artifact_path,
    resolve_compile_cache_runtime_dir,
    resolve_compile_cache_save_artifact_path,
    resolve_checkpoint_global_step,
    resolve_run_dir,
    resolve_wandb_state_dir,
    resolve_wandb_resume_from,
    resolve_wandb_resume,
    resolve_wandb_run_id,
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

    def test_cache_root_defaults_cover_run_and_wandb_state_dirs(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(run_name="paper-run", cache_root="/tmp/td-flow-cache"),
        )

        self.assertEqual(resolve_cache_root(project_config), Path("/tmp/td-flow-cache"))
        self.assertEqual(
            resolve_cache_run_dir(project_config),
            Path("/tmp/td-flow-cache") / "paper-run",
        )
        self.assertEqual(
            resolve_wandb_state_dir(project_config),
            Path("/tmp/td-flow-cache") / "paper-run" / "wandb",
        )

    def test_build_callbacks_monitors_validation_loss_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(output_dir=tmpdir),
            )

            callbacks = build_callbacks(project_config, Path(tmpdir), has_validation=True)
            self.assertTrue(any(isinstance(cb, ThroughputCallback) for cb in callbacks))
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

    def test_throughput_callback_logs_train_fps(self) -> None:
        callback = ThroughputCallback(every_n_steps=2)
        trainer = MagicMock()
        trainer.is_global_zero = True
        trainer.world_size = 2
        trainer.loggers = [MagicMock()]
        module = MagicMock()
        batch = {"obs": torch.zeros(8, 4)}

        trainer.global_step = 0
        with patch("td_flow.train.time.perf_counter", side_effect=[10.0, 12.0, 12.0]):
            callback.on_train_start(trainer, module)
            trainer.global_step = 2
            callback.on_train_batch_end(trainer, module, None, batch, 0)

        trainer.loggers[0].log_metrics.assert_called_once()
        metrics, = trainer.loggers[0].log_metrics.call_args.args
        self.assertIn("train/fps", metrics)
        self.assertAlmostEqual(metrics["train/fps"], 16.0, places=5)
        self.assertEqual(trainer.loggers[0].log_metrics.call_args.kwargs["step"], 2)

    def test_build_loggers_persists_wandb_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "outputs" / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(
                    output_dir=str(Path(tmpdir) / "outputs"),
                    cache_root=str(Path(tmpdir) / "cache"),
                    use_wandb=True,
                    wandb_save_dir=None,
                    run_name="run",
                ),
            )

            with patch("td_flow.train.WandbLogger") as wandb_logger, patch(
                "td_flow.train.logger.info"
            ) as log_info:
                build_loggers(project_config, run_dir)

            id_path = Path(tmpdir) / "cache" / "run" / "wandb" / "wandb_run_id.txt"
            self.assertTrue(id_path.exists())
            run_id = id_path.read_text().strip()
            self.assertEqual(wandb_logger.call_args.kwargs["id"], run_id)
            self.assertEqual(
                wandb_logger.call_args.kwargs["save_dir"],
                str(Path(tmpdir) / "cache" / "run" / "wandb"),
            )
            self.assertEqual(wandb_logger.call_args.kwargs["job_type"], "fit")
            self.assertIsNone(wandb_logger.call_args.kwargs["resume_from"])
            self.assertEqual(log_info.call_args.args[1], "td_flow")
            self.assertEqual(log_info.call_args.args[2], "<default>")

    def test_build_loggers_returns_false_for_nonzero_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "outputs" / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(
                    output_dir=str(Path(tmpdir) / "outputs"),
                    cache_root=str(Path(tmpdir) / "cache"),
                    use_wandb=True,
                    run_name="run",
                ),
            )

            with patch.dict("os.environ", {"RANK": "1"}, clear=False), patch(
                "td_flow.train.WandbLogger"
            ) as wandb_logger:
                logger_obj = build_loggers(project_config, run_dir)

            self.assertFalse(logger_obj)
            wandb_logger.assert_not_called()

    def test_resolve_wandb_resume_uses_must_for_fit_resume(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(resume_ckpt_path="dummy.ckpt"),
        )

        self.assertEqual(resolve_wandb_resume(project_config), "must")

    def test_resolve_wandb_resume_ignores_resume_when_offline(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(resume_ckpt_path="dummy.ckpt", wandb_offline=True),
        )

        self.assertIsNone(resolve_wandb_resume(project_config))

    def test_resolve_checkpoint_global_step_reads_checkpoint_metadata(self) -> None:
        with patch("td_flow.train.torch.load", return_value={"global_step": 1234}) as load:
            self.assertEqual(resolve_checkpoint_global_step("dummy.ckpt"), 1234)

        load.assert_called_once_with("dummy.ckpt", map_location="cpu")

    def test_resolve_wandb_resume_from_uses_checkpoint_global_step(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(resume_ckpt_path="dummy.ckpt"),
        )

        with patch("td_flow.train.torch.load", return_value={"global_step": 42}):
            self.assertEqual(
                resolve_wandb_resume_from(project_config, "run-123"),
                "run-123?_step=42",
            )

    def test_build_loggers_uses_resume_from_for_resumed_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "outputs" / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(
                    output_dir=str(Path(tmpdir) / "outputs"),
                    cache_root=str(Path(tmpdir) / "cache"),
                    use_wandb=True,
                    run_name="run",
                    resume_ckpt_path="dummy.ckpt",
                    wandb_id="fixed-id",
                ),
            )

            with patch("td_flow.train.WandbLogger") as wandb_logger, patch(
                "td_flow.train.torch.load", return_value={"global_step": 77}
            ):
                build_loggers(project_config, run_dir)

            self.assertIsNone(wandb_logger.call_args.kwargs["id"])
            self.assertIsNone(wandb_logger.call_args.kwargs["resume"])
            self.assertEqual(
                wandb_logger.call_args.kwargs["resume_from"],
                "fixed-id?_step=77",
            )

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
            artifact_path = (
                Path(tmpdir) / "compile" / "cube-single-play-v0" / "cache_artifacts.bin"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"cached")
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(compile=True, cache_root=tmpdir),
            )

            with patch("td_flow.train.torch.compiler.load_cache_artifacts") as load_cache:
                resolved = configure_compile_cache(project_config)

            self.assertEqual(resolved, artifact_path)
            self.assertEqual(os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
            self.assertEqual(os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"], "1")
            self.assertEqual(
                os.environ["TORCHINDUCTOR_CACHE_DIR"],
                str((Path(tmpdir) / "compile" / "cube-single-play-v0").resolve()),
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
                cache_root="/tmp/cache-root",
                compile_cache_name="manual-name",
            ),
        )

        self.assertEqual(
            resolve_compile_cache_artifact_path(project_config),
            Path("/tmp/cache-root") / "compile" / "manual-name" / "cache_artifacts.bin",
        )

    def test_resolve_compile_cache_paths_default_to_cache_root(self) -> None:
        project_config = ProjectConfig(
            data=DataConfig(dataset_name="cube-single-play-v0"),
            model=ModelConfig(observation_shape=(4,), action_dim=2),
            train=TrainConfig(compile=True, cache_root="/tmp/td-flow-cache"),
        )

        self.assertEqual(
            resolve_compile_cache_artifact_path(project_config),
            Path("/tmp/td-flow-cache") / "compile" / "cube-single-play-v0" / "cache_artifacts.bin",
        )
        self.assertEqual(
            resolve_compile_cache_runtime_dir(project_config),
            Path("/tmp/td-flow-cache") / "compile" / "cube-single-play-v0",
        )
        self.assertEqual(
            resolve_compile_cache_save_artifact_path(project_config),
            Path("/tmp/td-flow-cache") / "compile" / "cube-single-play-v0" / "cache_artifacts.bin",
        )

    def test_resolve_wandb_run_id_uses_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfig(
                data=DataConfig(dataset_name="cube-single-play-v0"),
                model=ModelConfig(observation_shape=(4,), action_dim=2),
                train=TrainConfig(wandb_id="fixed-id"),
            )

            self.assertEqual(resolve_wandb_run_id(project_config, Path(tmpdir)), "fixed-id")


if __name__ == "__main__":
    unittest.main()
