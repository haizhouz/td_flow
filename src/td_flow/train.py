from __future__ import annotations

from dataclasses import replace

import lightning as pl
import stable_pretraining as spt
import tyro
from lightning.pytorch.loggers import WandbLogger

from .config import DataConfig, ModelConfig, ProjectConfig, TrainEntryConfig
from .data import build_td2_dataloader, infer_shapes
from .module import build_training_module


def build_project_config_from_sample(
    entry_config: TrainEntryConfig,
) -> ProjectConfig:
    dataloader = build_td2_dataloader(entry_config.data, shuffle=True)
    sample = next(iter(dataloader))
    observation_shape, action_dim = infer_shapes(sample)
    model_config = ModelConfig(
        observation_shape=observation_shape,
        action_dim=action_dim,
        backbone=entry_config.backbone,
    )
    return ProjectConfig(
        data=entry_config.data,
        model=model_config,
        train=entry_config.train,
    )


def train(project_config: ProjectConfig) -> spt.Manager:
    train_loader = build_td2_dataloader(project_config.data, shuffle=True)
    val_data_config = replace(project_config.data, batch_size=project_config.data.batch_size)
    val_loader = build_td2_dataloader(val_data_config, shuffle=False)

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    module = build_training_module(project_config.model, project_config.train)
    logger = False
    if project_config.train.use_wandb:
        logger = WandbLogger(
            project=project_config.train.wandb_project,
            name=project_config.train.wandb_name,
            save_dir=project_config.train.wandb_save_dir,
            offline=project_config.train.wandb_offline,
            log_model=project_config.train.wandb_log_model,
        )
        logger.log_hyperparams(project_config.as_hparams())

    trainer = pl.Trainer(
        max_epochs=project_config.train.max_epochs,
        accelerator=project_config.train.accelerator,
        devices=project_config.train.devices,
        precision=project_config.train.precision,
        log_every_n_steps=project_config.train.log_every_n_steps,
        enable_checkpointing=project_config.train.enable_checkpointing,
        limit_train_batches=project_config.train.limit_train_batches,
        limit_val_batches=project_config.train.limit_val_batches,
        logger=logger,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=project_config.train.seed,
    )
    manager()
    return manager


def main() -> None:
    entry_config = tyro.cli(
        TrainEntryConfig,
        description="Train TD²-CFM on stable_worldmodel or OGBench datasets.",
    )
    project_config = build_project_config_from_sample(entry_config)
    train(project_config)


if __name__ == "__main__":
    main()
