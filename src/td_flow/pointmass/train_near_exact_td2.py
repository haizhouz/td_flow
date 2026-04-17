from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader
import tyro
import wandb

from ..config import (
    BackboneConfig,
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrainConfig,
    resolve_paper_weight_decay,
)
from ..data import build_td2_dataloader, infer_shapes
from ..model import TD2CFMModel
from ..paths import sample_linear_probability_path, sample_source, sample_time
from ..train import save_project_config, timestamped_run_name


@dataclass
class NearExactPointmassTrainConfig:
    output_dir: str = "outputs"
    run_name: str = "pointmass-near-exact-td2"
    lr: float = 1e-4
    weight_decay: float | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-4
    outer_iterations: int = 20
    cached_batches_per_outer: int = 128
    inner_epochs: int = 20
    inner_batch_size: int = 1024
    checkpoint_every_outer: int = 1
    checkpoint_every_inner_epochs: int = 10
    seed: int = 0
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str = "td_flow"
    wandb_name: str | None = None
    wandb_offline: bool = False


@dataclass
class NearExactPointmassConfig:
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            dataset_name="pointmass-exorl-rnd-scripted-policy-relnoise10",
            backend="stablewm_hdf5",
            dir="/home/haizhou/Documents/td_flow/data/stablewm_cache",
            observation_key="observation",
            action_key="action",
            next_action_key="policy_action",
            batch_size=1024,
            num_workers=16,
            num_steps=2,
        )
    )
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    train: NearExactPointmassTrainConfig = field(default_factory=NearExactPointmassTrainConfig)
    policy_mode: str = "single_policy"
    state_only_conditioning: bool = False
    observation_encoder: str = "identity"
    network_variant: str = "paper"
    policy_embedding_dim: int = 0
    gamma: float = 0.99
    initialization: str = "default"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _next_batch(iterator: Iterator[dict[str, torch.Tensor]], dataloader: DataLoader) -> tuple[dict[str, torch.Tensor], Iterator[dict[str, torch.Tensor]]]:
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch, iterator


def _hard_update_targets(model: TD2CFMModel) -> None:
    model.target_encoder.load_state_dict(model.encoder.state_dict())
    model.target_context_encoder.load_state_dict(model.context_encoder.state_dict())
    model.target_vector_field.load_state_dict(model.vector_field.state_dict())


def _build_project_config(config: NearExactPointmassConfig) -> ProjectConfig:
    sample_dataloader = build_td2_dataloader(config.data, shuffle=True)
    sample = next(iter(sample_dataloader))
    observation_shape, action_dim, policy_embedding_dim = infer_shapes(sample)
    use_identity_encoder = (
        config.observation_encoder in {"identity", "no_encoder"}
        or (
            config.observation_encoder == "auto"
            and config.policy_mode == "single_policy"
            and len(observation_shape) == 1
        )
    )
    latent_dim = math.prod(observation_shape) if use_identity_encoder else 128
    model_config = ModelConfig(
        observation_shape=observation_shape,
        action_dim=action_dim,
        state_only_conditioning=config.state_only_conditioning,
        backbone=config.backbone,
        observation_encoder=config.observation_encoder,
        network_variant=config.network_variant,
        latent_dim=latent_dim,
        policy_embedding_dim=max(policy_embedding_dim, config.policy_embedding_dim),
        gamma=config.gamma,
        initialization=config.initialization,
        policy_mode=config.policy_mode,
    )
    train_config = TrainConfig(
        output_dir=config.train.output_dir,
        run_name=timestamped_run_name(config.train.run_name),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        adam_beta1=config.train.adam_beta1,
        adam_beta2=config.train.adam_beta2,
        adam_eps=config.train.adam_eps,
        max_steps=config.train.outer_iterations * config.train.inner_epochs,
        seed=config.train.seed,
        use_wandb=config.train.use_wandb,
        wandb_project=config.train.wandb_project,
        wandb_name=config.train.wandb_name,
        wandb_offline=config.train.wandb_offline,
    )
    return ProjectConfig(data=config.data, model=model_config, train=train_config)


@torch.no_grad()
def _build_frozen_supervision_dataset(
    model: TD2CFMModel,
    dataloader: DataLoader,
    *,
    cached_batches: int,
) -> dict[str, torch.Tensor]:
    iterator = iter(dataloader)
    entries: dict[str, list[torch.Tensor]] = {
        "obs": [],
        "action": [],
        "x_t": [],
        "t": [],
        "target": [],
        "loss_weight": [],
    }
    direct_weight, bootstrap_weight = model.loss_weights()

    for _ in range(int(cached_batches)):
        batch, iterator = _next_batch(iterator, dataloader)
        obs = batch["obs"].to(model.device).float()
        action = batch["action"].to(model.device).float()
        next_obs = batch["next_obs"].to(model.device).float()
        next_action = batch["next_action"].to(model.device).float()
        next_latent_target = model.encode_observation(next_obs, use_target=True).detach()

        direct_t = sample_time(
            obs.shape[0],
            device=model.device,
            dtype=next_latent_target.dtype,
            eps=model.cfg.time_eps,
        )
        bootstrap_t = model.sample_bootstrap_time(
            obs.shape[0],
            dtype=next_latent_target.dtype,
        )
        source = sample_source(
            obs.shape[0],
            model.latent_dim,
            device=model.device,
            dtype=next_latent_target.dtype,
        )

        direct_xt, direct_target = sample_linear_probability_path(
            source,
            next_latent_target,
            direct_t,
            eps=model.cfg.time_eps,
        )
        bootstrap_xt, bootstrap_target = model.bootstrap_target(
            next_latent_target,
            next_action,
            source,
            bootstrap_t,
        )

        entries["obs"].extend([obs.detach(), obs.detach()])
        entries["action"].extend([action.detach(), action.detach()])
        entries["x_t"].extend([direct_xt.detach(), bootstrap_xt.detach()])
        entries["t"].extend([direct_t.detach(), bootstrap_t.detach()])
        entries["target"].extend([direct_target.detach(), bootstrap_target.detach()])
        entries["loss_weight"].extend(
            [
                torch.full((obs.shape[0],), float(direct_weight), dtype=torch.float32, device=model.device),
                torch.full((obs.shape[0],), float(bootstrap_weight), dtype=torch.float32, device=model.device),
            ]
        )

    return {key: torch.cat(values, dim=0) for key, values in entries.items()}


def _training_loss(model: TD2CFMModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    obs = batch["obs"].float()
    action = batch["action"].float()
    x_t = batch["x_t"].float()
    t = batch["t"].float()
    target = batch["target"].float()
    loss_weight = batch["loss_weight"].float()

    state_latent = model.encode_observation(obs)
    prediction = model.compute_velocity(
        x_t,
        t,
        state_latent,
        action,
        use_target=False,
    )
    per_example = torch.mean((prediction - target) ** 2, dim=-1)
    return torch.mean(loss_weight * per_example)


def _save_checkpoint(
    model: TD2CFMModel,
    checkpoint_path: Path,
    *,
    global_step: int,
    outer_iteration: int,
    inner_epoch: int | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {f"td2_cfm.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "global_step": int(global_step),
        "outer_iteration": int(outer_iteration),
        "inner_epoch": None if inner_epoch is None else int(inner_epoch),
    }
    torch.save(payload, checkpoint_path)


def _iterate_frozen_batches(
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> Iterator[dict[str, torch.Tensor]]:
    length = next(iter(tensors.values())).shape[0]
    order = torch.randperm(length, device=next(iter(tensors.values())).device)
    for start in range(0, length, batch_size):
        batch_indices = order[start : start + batch_size]
        yield {key: value[batch_indices] for key, value in tensors.items()}


def run_near_exact_pointmass(config: NearExactPointmassConfig) -> Path:
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    project_config = _build_project_config(config)
    run_dir = Path(project_config.train.output_dir) / project_config.train.run_name
    save_project_config(project_config, run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    dataloader = build_td2_dataloader(project_config.data, shuffle=True)
    model = TD2CFMModel(project_config.model).to(_resolve_device(config.train.device))
    _hard_update_targets(model)

    weight_decay = (
        config.train.weight_decay
        if config.train.weight_decay is not None
        else resolve_paper_weight_decay(project_config.model.policy_mode)
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.train.lr,
        weight_decay=weight_decay,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_eps,
    )

    wandb_run = None
    if config.train.use_wandb:
        wandb_run = wandb.init(
            project=config.train.wandb_project,
            name=config.train.wandb_name or project_config.train.run_name,
            config=project_config.as_hparams(),
            dir=str(run_dir),
            mode="offline" if config.train.wandb_offline else "online",
        )

    global_step = 0
    for outer_iteration in range(1, config.train.outer_iterations + 1):
        frozen_dataset = _build_frozen_supervision_dataset(
            model,
            dataloader,
            cached_batches=config.train.cached_batches_per_outer,
        )
        cached_examples = next(iter(frozen_dataset.values())).shape[0]
        checkpoint_dir = run_dir / "checkpoints"

        epoch_losses: list[float] = []
        for inner_epoch in range(1, config.train.inner_epochs + 1):
            running_loss = 0.0
            batch_count = 0
            for batch in _iterate_frozen_batches(
                frozen_dataset,
                batch_size=config.train.inner_batch_size,
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = _training_loss(model, batch)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item())
                batch_count += 1
                global_step += 1
            epoch_loss = running_loss / max(batch_count, 1)
            epoch_losses.append(epoch_loss)
            epoch_metric = {
                "outer_iteration": int(outer_iteration),
                "inner_epoch": int(inner_epoch),
                "global_step": int(global_step),
                "cached_examples": int(cached_examples),
                "inner_epoch_loss": float(epoch_loss),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(epoch_metric, sort_keys=True) + "\n")
            if wandb_run is not None:
                wandb.log(epoch_metric, step=global_step)
            if inner_epoch % max(config.train.checkpoint_every_inner_epochs, 1) == 0 and config.train.checkpoint_every_inner_epochs > 0:
                inner_checkpoint = checkpoint_dir / f"outer={outer_iteration:04d}-inner={inner_epoch:04d}.ckpt"
                _save_checkpoint(
                    model,
                    inner_checkpoint,
                    global_step=global_step,
                    outer_iteration=outer_iteration,
                    inner_epoch=inner_epoch,
                )
                _save_checkpoint(
                    model,
                    checkpoint_dir / "last.ckpt",
                    global_step=global_step,
                    outer_iteration=outer_iteration,
                    inner_epoch=inner_epoch,
                )

        _hard_update_targets(model)
        metric = {
            "outer_iteration": int(outer_iteration),
            "global_step": int(global_step),
            "cached_examples": int(cached_examples),
            "outer_mean_inner_epoch_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            "outer_final_inner_epoch_loss": float(epoch_losses[-1]),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb.log(metric, step=global_step)

        if outer_iteration % max(config.train.checkpoint_every_outer, 1) == 0:
            checkpoint_path = checkpoint_dir / f"outer={outer_iteration:04d}.ckpt"
            _save_checkpoint(
                model,
                checkpoint_path,
                global_step=global_step,
                outer_iteration=outer_iteration,
            )
            _save_checkpoint(
                model,
                checkpoint_dir / "last.ckpt",
                global_step=global_step,
                outer_iteration=outer_iteration,
            )

    if wandb_run is not None:
        wandb_run.finish()
    return run_dir


def main() -> None:
    config = tyro.cli(
        NearExactPointmassConfig,
        description=(
            "Run a near-exact outer-loop TD2 experiment on pointmass by freezing the target, "
            "caching a fixed supervised TD dataset for each outer iteration, and optimizing "
            "the online model hard before a hard target update."
        ),
    )
    run_dir = run_near_exact_pointmass(config)
    print(str(run_dir))


if __name__ == "__main__":
    main()
