from __future__ import annotations

from dataclasses import asdict
from functools import partial
from types import MethodType
from typing import Any

import torch
import stable_pretraining as spt

from .config import ModelConfig, TrainConfig, resolve_paper_weight_decay
from .model import TD2CFMModel


def td2_cfm_forward(self: spt.Module, batch: dict, stage: str) -> dict:
    self.td2_cfm.set_loss_weight_step(self.global_step)
    return self.td2_cfm.compute_state(batch, stage=stage)


def _extract_scalar_metrics(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            metrics[key] = value.detach()
    return metrics


def _log_metrics(
    module: spt.Module,
    state: dict[str, Any] | None,
    *,
    prefix: str,
    batch_size: int,
    on_step: bool,
    on_epoch: bool,
) -> None:
    if not isinstance(state, dict):
        return
    metrics = _extract_scalar_metrics(state)
    if not metrics:
        return

    for key, value in metrics.items():
        module.log(
            f"{prefix}/{key}",
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=key == "loss",
            sync_dist=True,
            batch_size=batch_size,
        )
        if key == "loss":
            module.log(
                f"{prefix}_loss",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )


def td2_cfm_on_train_batch_end(
    self: spt.Module,
    outputs,
    batch,
    batch_idx: int,
) -> None:
    del batch_idx
    self.td2_cfm.update_targets()
    batch_size = int(batch["obs"].shape[0]) if isinstance(batch, dict) and "obs" in batch else 1
    _log_metrics(
        self,
        outputs,
        prefix="train",
        batch_size=batch_size,
        on_step=True,
        on_epoch=True,
    )


def td2_cfm_on_validation_batch_end(
    self: spt.Module,
    outputs,
    batch,
    batch_idx: int,
    dataloader_idx: int = 0,
) -> None:
    del batch_idx, dataloader_idx
    batch_size = int(batch["obs"].shape[0]) if isinstance(batch, dict) and "obs" in batch else 1
    _log_metrics(
        self,
        outputs,
        prefix="val",
        batch_size=batch_size,
        on_step=False,
        on_epoch=True,
    )


def build_training_module(
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> spt.Module:
    td2_model = TD2CFMModel(model_config)
    weight_decay = (
        train_config.weight_decay
        if train_config.weight_decay is not None
        else resolve_paper_weight_decay(model_config.policy_mode)
    )
    optim_config = {
        "optimizer": {
            "type": "AdamW",
            "lr": train_config.lr,
            "weight_decay": weight_decay,
            "betas": (train_config.adam_beta1, train_config.adam_beta2),
            "eps": train_config.adam_eps,
        },
        "interval": "step" if train_config.train_semantics == "paper" else "epoch",
    }
    if train_config.scheduler is not None:
        optim_config["scheduler"] = {"type": train_config.scheduler}
    else:
        optim_config["scheduler"] = partial(
            torch.optim.lr_scheduler.ConstantLR,
            factor=1.0,
            total_iters=1,
        )
    module = spt.Module(
        forward=td2_cfm_forward,
        hparams={
            "model": asdict(model_config),
            "train": asdict(train_config),
        },
        td2_cfm=td2_model,
        optim=optim_config,
    )
    module.on_train_batch_end = MethodType(td2_cfm_on_train_batch_end, module)
    module.on_validation_batch_end = MethodType(td2_cfm_on_validation_batch_end, module)
    return module
