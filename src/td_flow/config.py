from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def resolve_paper_weight_decay(policy_mode: str) -> float:
    if policy_mode == "single_policy":
        return 1e-3
    if policy_mode == "multi_policy":
        return 1e-2
    raise ValueError("policy_mode must be one of: single_policy, multi_policy")


def resolve_paper_max_steps(policy_mode: str) -> int:
    if policy_mode == "single_policy":
        return 3_000_000
    if policy_mode == "multi_policy":
        return 8_000_000
    raise ValueError("policy_mode must be one of: single_policy, multi_policy")


def resolve_paper_polyak(policy_mode: str) -> float:
    if policy_mode == "single_policy":
        return 0.999
    if policy_mode == "multi_policy":
        return 0.9999
    raise ValueError("policy_mode must be one of: single_policy, multi_policy")


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
    split: str = "train"
    observation_key: str = "state"
    action_key: str = "action"
    next_action_key: str | None = None
    goal_key: str | None = None
    policy_embedding_key: str | None = None
    batch_size: int = 1024
    num_workers: int = 4
    frameskip: int = 1
    num_steps: int = 2
    dir: str | None = None
    keys_to_load: tuple[str, ...] = ()

    def resolved_keys_to_load(self) -> list[str]:
        keys = set(self.keys_to_load)
        keys.add(self.observation_key)
        keys.add(self.action_key)
        if self.next_action_key is not None:
            keys.add(self.next_action_key)
        if self.goal_key is not None:
            keys.add(self.goal_key)
        if self.policy_embedding_key is not None:
            keys.add(self.policy_embedding_key)
        return sorted(keys)


@dataclass
class ModelConfig:
    observation_shape: tuple[int, ...]
    action_dim: int
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    observation_encoder: str = "auto"
    network_variant: str = "repo"
    latent_dim: int = 128
    policy_embedding_dim: int = 0
    context_dim: int = 128
    context_hidden_dims: tuple[int, ...] = (256, 256)
    vector_field_hidden_dims: tuple[int, ...] = (256, 256)
    time_embed_dim: int = 256
    gamma: float = 0.99
    direct_loss_weight: float | None = None
    bootstrap_loss_weight: float | None = None
    bootstrap_time_sampling: str = "uniform"
    bootstrap_time_late_prob: float = 0.5
    bootstrap_time_late_start: float = 0.9
    initialization: str = "default"
    polyak: float | None = None
    ode_steps: int = 10
    time_eps: float = 1e-4
    policy_mode: str = "single_policy"


@dataclass
class TrainConfig:
    run_mode: str = "fit"
    compile: bool = False
    cache_root: str = ".cache/td_flow"
    matmul_precision: str | None = "high"
    compile_cache_name: str | None = None
    train_semantics: str = "paper"
    lr: float = 1e-4
    weight_decay: float | None = None
    scheduler: str | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-4
    max_steps: int | None = None
    max_epochs: int | None = None
    val_check_interval: int | float | None = None
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = 10
    enable_progress_bar: bool = False
    seed: int = 0
    output_dir: str = "outputs"
    run_name: str | None = None
    use_csv_logger: bool = True
    enable_checkpointing: bool = True
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 1
    checkpoint_every_n_train_steps: int = 10_000
    checkpoint_save_last: bool = True
    resume: bool = False
    resume_ckpt_path: str | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = 0
    use_wandb: bool = False
    wandb_project: str = "td_flow"
    wandb_name: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str | None = None
    wandb_id: str | None = None
    wandb_resume: str | None = None
    wandb_save_dir: str | None = None
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
    policy_mode: str = "single_policy"
    observation_encoder: str = "auto"
    network_variant: str = "repo"
    policy_embedding_dim: int = 0
    gamma: float = 0.99
    polyak: float | None = None
    direct_loss_weight: float | None = None
    bootstrap_loss_weight: float | None = None
    bootstrap_time_sampling: str = "uniform"
    bootstrap_time_late_prob: float = 0.5
    bootstrap_time_late_start: float = 0.9
    initialization: str = "default"
