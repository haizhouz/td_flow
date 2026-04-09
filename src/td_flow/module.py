from __future__ import annotations

from dataclasses import asdict

import stable_pretraining as spt

from .config import ModelConfig, TrainConfig
from .model import TD2CFMModel


def td2_cfm_forward(self: spt.Module, batch: dict, stage: str) -> dict:
    return self.td2_cfm.compute_state(batch, stage=stage)


def build_training_module(
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> spt.Module:
    td2_model = TD2CFMModel(model_config)
    return spt.Module(
        forward=td2_cfm_forward,
        hparams={
            "model": asdict(model_config),
            "train": asdict(train_config),
        },
        td2_cfm=td2_model,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
            },
            "scheduler": {"type": train_config.scheduler},
            "interval": "epoch",
        },
    )

