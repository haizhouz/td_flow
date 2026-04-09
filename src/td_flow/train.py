from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import tyro
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .config import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrainEntryConfig,
    resolve_paper_max_steps,
)
from .data import build_td2_dataloader, infer_shapes
from .module import build_training_module


def resolve_run_dir(project_config: ProjectConfig) -> Path:
    run_name = project_config.train.run_name or project_config.data.dataset_name
    return Path(project_config.train.output_dir) / run_name


def save_project_config(project_config: ProjectConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "project_config.json"
    config_path.write_text(
        json.dumps(project_config.as_hparams(), indent=2, sort_keys=True) + "\n"
    )


def build_loggers(project_config: ProjectConfig, run_dir: Path):
    loggers: list = []
    if project_config.train.use_csv_logger:
        loggers.append(
            CSVLogger(
                save_dir=str(run_dir),
                name="csv",
            )
        )
    if project_config.train.use_wandb:
        loggers.append(
            WandbLogger(
                project=project_config.train.wandb_project,
                name=project_config.train.wandb_name or project_config.train.run_name,
                save_dir=project_config.train.wandb_save_dir,
                offline=project_config.train.wandb_offline,
                log_model=project_config.train.wandb_log_model,
            )
        )
    if not loggers:
        return False
    for logger in loggers:
        logger.log_hyperparams(project_config.as_hparams())
    return loggers[0] if len(loggers) == 1 else loggers


def build_callbacks(
    project_config: ProjectConfig,
    run_dir: Path,
    *,
    has_validation: bool,
) -> list:
    callbacks: list = []
    if project_config.train.use_csv_logger or project_config.train.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if not project_config.train.enable_checkpointing:
        return callbacks

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = project_config.train.checkpoint_monitor if has_validation else None
    filename = "step={step}"
    if monitor == "val_loss":
        filename = "step={step}-val_loss={val_loss:.6f}"
    checkpoint_kwargs = dict(
        dirpath=str(checkpoint_dir),
        filename=filename,
        save_top_k=project_config.train.checkpoint_save_top_k,
        save_last=project_config.train.checkpoint_save_last,
        every_n_train_steps=project_config.train.checkpoint_every_n_train_steps,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    if monitor is not None:
        checkpoint_kwargs["monitor"] = monitor
        checkpoint_kwargs["mode"] = project_config.train.checkpoint_mode
    callbacks.append(ModelCheckpoint(**checkpoint_kwargs))
    return callbacks


def _resolve_train_data_config(project_config: ProjectConfig) -> DataConfig:
    return replace(project_config.data, split="train")


def _resolve_val_data_config(project_config: ProjectConfig) -> DataConfig:
    split = "val" if project_config.data.backend == "ogbench_npz" else project_config.data.split
    return replace(project_config.data, split=split)


def build_data_module(project_config: ProjectConfig, *, mode: str) -> spt.data.DataModule:
    if mode not in {"fit", "validate"}:
        raise ValueError("mode must be one of: fit, validate")

    if mode == "fit":
        train_loader = build_td2_dataloader(_resolve_train_data_config(project_config), shuffle=True)
        val_loader = None
        if project_config.train.limit_val_batches not in (0, 0.0):
            val_loader = build_td2_dataloader(_resolve_val_data_config(project_config), shuffle=False)
        return spt.data.DataModule(train=train_loader, val=val_loader)

    val_loader = build_td2_dataloader(_resolve_val_data_config(project_config), shuffle=False)
    return spt.data.DataModule(val=val_loader)


def build_trainer(
    project_config: ProjectConfig,
    *,
    logger,
    callbacks: list,
    has_validation: bool,
) -> pl.Trainer:
    run_dir = resolve_run_dir(project_config)
    trainer_kwargs = dict(
        default_root_dir=str(run_dir),
        accelerator=project_config.train.accelerator,
        devices=project_config.train.devices,
        precision=project_config.train.precision,
        log_every_n_steps=project_config.train.log_every_n_steps,
        enable_checkpointing=project_config.train.enable_checkpointing,
        limit_train_batches=project_config.train.limit_train_batches,
        limit_val_batches=project_config.train.limit_val_batches,
        logger=logger,
        callbacks=callbacks,
    )
    if project_config.train.train_semantics == "paper":
        max_steps = (
            project_config.train.max_steps
            if project_config.train.max_steps is not None
            else resolve_paper_max_steps(project_config.model.policy_mode)
        )
        trainer_kwargs.update(
            max_steps=max_steps,
            max_epochs=-1 if project_config.train.max_epochs is None else project_config.train.max_epochs,
            num_sanity_val_steps=0,
        )
        if has_validation:
            trainer_kwargs["check_val_every_n_epoch"] = None
            trainer_kwargs["val_check_interval"] = (
                project_config.train.val_check_interval
                if project_config.train.val_check_interval is not None
                else max_steps
            )
    else:
        trainer_kwargs["max_epochs"] = (
            project_config.train.max_epochs
            if project_config.train.max_epochs is not None
            else 50
        )
    return pl.Trainer(**trainer_kwargs)


def build_project_config_from_sample(
    entry_config: TrainEntryConfig,
) -> ProjectConfig:
    sample_data_config = replace(entry_config.data, split="train")
    dataloader = build_td2_dataloader(sample_data_config, shuffle=True)
    sample = next(iter(dataloader))
    observation_shape, action_dim, policy_embedding_dim = infer_shapes(sample)
    use_identity_encoder = (
        entry_config.observation_encoder in {"identity", "no_encoder"}
        or (
            entry_config.observation_encoder == "auto"
            and entry_config.policy_mode == "single_policy"
            and len(observation_shape) == 1
        )
    )
    latent_dim = math.prod(observation_shape) if use_identity_encoder else 128
    model_config = ModelConfig(
        observation_shape=observation_shape,
        action_dim=action_dim,
        backbone=entry_config.backbone,
        observation_encoder=entry_config.observation_encoder,
        network_variant=entry_config.network_variant,
        latent_dim=latent_dim,
        policy_embedding_dim=max(policy_embedding_dim, entry_config.policy_embedding_dim),
        policy_mode=entry_config.policy_mode,
    )
    return ProjectConfig(
        data=entry_config.data,
        model=model_config,
        train=entry_config.train,
    )


def train(project_config: ProjectConfig) -> spt.Manager:
    run_dir = resolve_run_dir(project_config)
    save_project_config(project_config, run_dir)
    data_module = build_data_module(project_config, mode="fit")
    module = build_training_module(project_config.model, project_config.train)
    has_validation = data_module.val_dataloader() is not None
    logger = build_loggers(project_config, run_dir)
    callbacks = build_callbacks(project_config, run_dir, has_validation=has_validation)
    trainer = build_trainer(
        project_config,
        logger=logger,
        callbacks=callbacks,
        has_validation=has_validation,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=project_config.train.seed,
        ckpt_path=project_config.train.resume_ckpt_path,
    )
    manager()
    return manager


def evaluate(project_config: ProjectConfig) -> list[dict[str, float]]:
    if project_config.train.resume_ckpt_path is None:
        raise ValueError("resume_ckpt_path is required for run_mode=validate")

    run_dir = resolve_run_dir(project_config)
    save_project_config(project_config, run_dir)
    data_module = build_data_module(project_config, mode="validate")
    module = build_training_module(project_config.model, project_config.train)
    logger = build_loggers(project_config, run_dir)
    callbacks = build_callbacks(project_config, run_dir, has_validation=True)
    trainer = build_trainer(
        project_config,
        logger=logger,
        callbacks=callbacks,
        has_validation=True,
    )
    pl.seed_everything(project_config.train.seed, workers=True)
    results = trainer.validate(
        module,
        datamodule=data_module,
        ckpt_path=project_config.train.resume_ckpt_path,
    )
    metrics_path = run_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main() -> None:
    entry_config = tyro.cli(
        TrainEntryConfig,
        description="Train TD²-CFM on stable_worldmodel or OGBench datasets.",
    )
    project_config = build_project_config_from_sample(entry_config)
    if project_config.train.run_mode == "fit":
        train(project_config)
        return
    if project_config.train.run_mode == "validate":
        evaluate(project_config)
        return
    raise ValueError("train.run_mode must be one of: fit, validate")


if __name__ == "__main__":
    main()
