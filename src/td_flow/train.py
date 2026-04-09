from __future__ import annotations

import math
from dataclasses import replace

import lightning as pl
import stable_pretraining as spt
import tyro
from lightning.pytorch.loggers import WandbLogger

from .config import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrainEntryConfig,
    resolve_paper_max_steps,
)
from .data import build_td2_dataloader, infer_shapes
from .module import build_training_module


def build_project_config_from_sample(
    entry_config: TrainEntryConfig,
) -> ProjectConfig:
    dataloader = build_td2_dataloader(entry_config.data, shuffle=True)
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
    train_loader = build_td2_dataloader(project_config.data, shuffle=True)
    val_loader = None
    if project_config.train.limit_val_batches not in (0, 0.0):
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

    trainer_kwargs = dict(
        accelerator=project_config.train.accelerator,
        devices=project_config.train.devices,
        precision=project_config.train.precision,
        log_every_n_steps=project_config.train.log_every_n_steps,
        enable_checkpointing=project_config.train.enable_checkpointing,
        limit_train_batches=project_config.train.limit_train_batches,
        limit_val_batches=project_config.train.limit_val_batches,
        logger=logger,
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
        if val_loader is not None:
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

    trainer = pl.Trainer(**trainer_kwargs)
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
