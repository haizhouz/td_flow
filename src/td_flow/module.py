from __future__ import annotations

from dataclasses import asdict
from functools import partial
from types import MethodType

import torch
import stable_pretraining as spt

from .config import ModelConfig, TrainConfig, resolve_paper_weight_decay
from .model import TD2CFMModel


def td2_cfm_forward(self: spt.Module, batch: dict, stage: str) -> dict:
    return self.td2_cfm.compute_state(batch, stage=stage)


def td2_cfm_on_before_zero_grad(self: spt.Module, optimizer) -> None:
    del optimizer
    self.td2_cfm.update_targets()


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
    module.on_before_zero_grad = MethodType(td2_cfm_on_before_zero_grad, module)
    return module
