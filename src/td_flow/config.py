from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BackboneConfig:
    kind: str = "mlp"
    hidden_dims: tuple[int, ...] = (256, 256)
    torchvision_name: str = "resnet18"
    low_resolution: bool = False


@dataclass
class DataConfig:
    dataset_name: str
    backend: str = "stablewm_hdf5"
    observation_key: str = "state"
    action_key: str = "action"
    goal_key: str | None = None
    batch_size: int = 64
    num_workers: int = 4
    frameskip: int = 1
    num_steps: int = 2
    cache_dir: str | None = None
    keys_to_load: tuple[str, ...] = ()

    def resolved_keys_to_load(self) -> list[str]:
        keys = set(self.keys_to_load)
        keys.add(self.observation_key)
        keys.add(self.action_key)
        if self.goal_key is not None:
            keys.add(self.goal_key)
        return sorted(keys)


@dataclass
class ModelConfig:
    observation_shape: tuple[int, ...]
    action_dim: int
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    latent_dim: int = 128
    context_dim: int = 128
    context_hidden_dims: tuple[int, ...] = (256, 256)
    vector_field_hidden_dims: tuple[int, ...] = (256, 256)
    time_embed_dim: int = 64
    gamma: float = 0.99
    polyak: float = 0.995
    ode_steps: int = 10
    time_eps: float = 1e-4


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    max_epochs: int = 50
    accelerator: str = "auto"
    devices: int | str = 1
    precision: str = "32-true"
    log_every_n_steps: int = 10
    seed: int = 0
    enable_checkpointing: bool = False
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    use_wandb: bool = False
    wandb_project: str = "td_flow"
    wandb_name: str | None = None
    wandb_save_dir: str = "."
    wandb_offline: bool = False
    wandb_log_model: bool = False


@dataclass
class PlanningConfig:
    horizon: int = 10
    receding_horizon: int = 5
    action_block: int = 1
    history_len: int = 1
    warm_start: bool = True
    num_samples: int = 300
    batch_size: int = 1
    n_steps: int = 15
    topk: int = 30
    var_scale: float = 1.0
    terminal_weight: float = 1.0
    rollout_discount: float = 1.0


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)

    def as_hparams(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainEntryConfig:
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
