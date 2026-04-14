from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from ..paths import sample_linear_probability_path, sample_source
from .plot_circle_policy_conditioned_occupancy import (
    _draw_background,
    _episode_index_for_step,
    _resolve_device,
    _resolve_hdf5_path,
    _sample_valid_start_indices,
)
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


DIRECT_COLOR = "#f0d46d"
BOOTSTRAP_COLOR = "#ff8f79"
TRAJECTORY_COLOR = "#2f2f2f"
NEXT_COLOR = "#79c4ff"
START_COLOR = "#fff06a"


@dataclass
class AnalyzeToyCircleTD2TargetsConfig:
    checkpoint_path: str
    device: str = "auto"
    num_states: int = 3
    sample_count: int = 2048
    quiver_count: int = 128
    sample_batch_size: int = 1024
    min_future_steps: int = 100
    t_values: tuple[float, ...] = (0.5, 0.9, 0.99)
    future_overlay_steps: int = 100
    seed: int = 0
    output_dir: str | None = None


def _default_output_dir(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "toy_td2_targets"


def _future_trajectory(observations: h5py.Dataset, ep_offset: np.ndarray, ep_len: np.ndarray, start_index: int) -> np.ndarray:
    episode_index = _episode_index_for_step(ep_offset, start_index)
    episode_start = int(ep_offset[episode_index])
    local_index = int(start_index - episode_start)
    remaining = int(ep_len[episode_index]) - local_index
    return np.asarray(observations[start_index : start_index + remaining], dtype=np.float32)


def _mean_min_distance(samples_xy: np.ndarray, support_xy: np.ndarray) -> float:
    diffs = samples_xy[:, None, :] - support_xy[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.min(axis=1).mean())


@torch.no_grad()
def _sample_direct_xt_positions(
    model,
    next_observation: np.ndarray,
    *,
    t_value: float,
    device: torch.device,
    sample_count: int,
) -> np.ndarray:
    next_obs_tensor = torch.from_numpy(next_observation).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_latent = model.encode_observation(next_obs_tensor, use_target=True).expand(sample_count, -1)
    t_tensor = torch.full((sample_count,), float(t_value), device=device, dtype=next_latent.dtype)
    source = sample_source(sample_count, model.latent_dim, device=device, dtype=next_latent.dtype)
    direct_xt, _ = sample_linear_probability_path(
        source,
        next_latent,
        t_tensor,
        eps=model.cfg.time_eps,
    )
    return direct_xt[:, :2].detach().cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def _sample_bootstrap_xt_positions(
    model,
    next_observation: np.ndarray,
    next_action: np.ndarray,
    *,
    t_value: float,
    device: torch.device,
    sample_count: int,
) -> np.ndarray:
    next_obs_tensor = torch.from_numpy(next_observation).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_action_tensor = torch.from_numpy(next_action).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_latent = model.encode_observation(next_obs_tensor, use_target=True).expand(sample_count, -1)
    next_action_batch = next_action_tensor.expand(sample_count, -1)
    t_tensor = torch.full((sample_count,), float(t_value), device=device, dtype=next_latent.dtype)
    source = sample_source(sample_count, model.latent_dim, device=device, dtype=next_latent.dtype)
    bootstrap_xt, _ = model.bootstrap_target(
        next_latent,
        next_action_batch,
        source,
        t_tensor,
    )
    return bootstrap_xt[:, :2].detach().cpu().numpy().astype(np.float32, copy=False)


def _plot_positions(
    ax: plt.Axes,
    positions: np.ndarray,
    *,
    start_xy: np.ndarray,
    next_xy: np.ndarray,
    future_xy: np.ndarray,
    sample_color: str,
    title: str,
) -> None:
    _draw_background(ax)
    ax.scatter(positions[:, 0], positions[:, 1], s=5, c=sample_color, alpha=0.12, linewidths=0)
    ax.plot(future_xy[:, 0], future_xy[:, 1], color=TRAJECTORY_COLOR, linewidth=1.0, alpha=0.9, zorder=1)
    ax.scatter([float(start_xy[0])], [float(start_xy[1])], s=42, c=START_COLOR, edgecolors="#2f2f2f", linewidths=0.6, zorder=3)
    ax.scatter([float(next_xy[0])], [float(next_xy[1])], s=34, c=NEXT_COLOR, edgecolors="#2f2f2f", linewidths=0.5, zorder=3)
    ax.set_title(title, fontsize=10)


@torch.no_grad()
def _sample_online_velocity_xy(
    model,
    observation: np.ndarray,
    action: np.ndarray,
    xt_positions: np.ndarray,
    *,
    t_value: float,
    device: torch.device,
) -> np.ndarray:
    x_t = torch.from_numpy(np.asarray(xt_positions, dtype=np.float32)).to(device=device, dtype=torch.float32)
    obs_tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    state_latent = model.encode_observation(obs_tensor).expand(x_t.shape[0], -1)
    action_batch = action_tensor.expand(x_t.shape[0], -1)
    t_tensor = torch.full((x_t.shape[0],), float(t_value), device=device, dtype=x_t.dtype)
    velocity = model.compute_velocity(
        x_t,
        t_tensor,
        state_latent,
        action_batch,
    )
    return velocity[:, :2].detach().cpu().numpy().astype(np.float32, copy=False)


def _overlay_velocity_quiver(
    ax: plt.Axes,
    positions: np.ndarray,
    velocity_xy: np.ndarray,
    *,
    count: int,
    seed: int,
) -> None:
    if len(positions) == 0:
        return
    rng = np.random.default_rng(seed)
    chosen_count = min(int(count), len(positions))
    indices = rng.choice(len(positions), size=chosen_count, replace=False)
    pos = positions[indices]
    vel = velocity_xy[indices]
    norms = np.linalg.norm(vel, axis=-1, keepdims=True)
    unit_vel = vel / np.clip(norms, 1e-8, None)
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        unit_vel[:, 0],
        unit_vel[:, 1],
        color="#111111",
        angles="xy",
        scale_units="xy",
        scale=22.0,
        width=0.0032,
        alpha=0.75,
        zorder=4,
    )


def analyze_toy_circle_td2_targets(config: AnalyzeToyCircleTD2TargetsConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This analysis currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (2,):
        raise NotImplementedError("This analysis currently supports only 2D toy-circle observations.")
    if project_config.model.observation_encoder not in {"auto", "identity", "no_encoder"}:
        raise NotImplementedError("This analysis currently supports only identity-like observation encoders.")

    device = _resolve_device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)
    dataset_path = _resolve_hdf5_path(run_dir)

    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir(config.checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[project_config.data.observation_key]
        actions = handle[project_config.data.action_key]
        ep_offset = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        start_indices = _sample_valid_start_indices(
            ep_offset,
            ep_len,
            count=config.num_states,
            min_future_steps=config.min_future_steps,
            seed=config.seed,
        )

        fig, axes = plt.subplots(
            2 * config.num_states,
            len(config.t_values),
            figsize=(3.2 * len(config.t_values), 3.0 * 2 * config.num_states),
            squeeze=False,
        )

        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "device": str(device),
            "sample_count": config.sample_count,
            "quiver_count": config.quiver_count,
            "sample_batch_size": config.sample_batch_size,
            "min_future_steps": config.min_future_steps,
            "t_values": list(config.t_values),
            "future_overlay_steps": config.future_overlay_steps,
            "seed": config.seed,
            "states": [],
        }

        for state_index, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            next_observation = np.asarray(observations[start_index + 1], dtype=np.float32)
            next_action = np.asarray(actions[start_index + 1], dtype=np.float32)
            future_xy = _future_trajectory(observations, ep_offset, ep_len, start_index + 1)[: config.future_overlay_steps, :2]

            state_metrics: dict[str, object] = {
                "start_index": int(start_index),
                "observation": observation.tolist(),
                "next_observation": next_observation.tolist(),
                "next_action": next_action.tolist(),
                "t_values": {},
            }

            direct_row = 2 * state_index
            bootstrap_row = direct_row + 1
            axes[direct_row, 0].set_ylabel(f"State {state_index + 1}\nDirect", fontsize=11)
            axes[bootstrap_row, 0].set_ylabel(f"State {state_index + 1}\nBootstrap", fontsize=11)

            for col, t_value in enumerate(config.t_values):
                direct_positions = _sample_direct_xt_positions(
                    model,
                    next_observation,
                    t_value=float(t_value),
                    device=device,
                    sample_count=config.sample_count,
                )
                bootstrap_positions = _sample_bootstrap_xt_positions(
                    model,
                    next_observation,
                    next_action,
                    t_value=float(t_value),
                    device=device,
                    sample_count=config.sample_count,
                )
                direct_velocity = _sample_online_velocity_xy(
                    model,
                    observation,
                    next_action,
                    direct_positions,
                    t_value=float(t_value),
                    device=device,
                )
                bootstrap_velocity = _sample_online_velocity_xy(
                    model,
                    observation,
                    next_action,
                    bootstrap_positions,
                    t_value=float(t_value),
                    device=device,
                )

                _plot_positions(
                    axes[direct_row, col],
                    direct_positions,
                    start_xy=observation[:2],
                    next_xy=next_observation[:2],
                    future_xy=future_xy,
                    sample_color=DIRECT_COLOR,
                    title=f"direct x_t, t={t_value:.2f}",
                )
                _overlay_velocity_quiver(
                    axes[direct_row, col],
                    direct_positions,
                    direct_velocity,
                    count=config.quiver_count,
                    seed=config.seed + 1000 * state_index + 10 * col + 1,
                )
                _plot_positions(
                    axes[bootstrap_row, col],
                    bootstrap_positions,
                    start_xy=observation[:2],
                    next_xy=next_observation[:2],
                    future_xy=future_xy,
                    sample_color=BOOTSTRAP_COLOR,
                    title=f"bootstrap x_t, t={t_value:.2f}",
                )
                _overlay_velocity_quiver(
                    axes[bootstrap_row, col],
                    bootstrap_positions,
                    bootstrap_velocity,
                    count=config.quiver_count,
                    seed=config.seed + 1000 * state_index + 10 * col + 2,
                )

                state_metrics["t_values"][f"{t_value:.2f}"] = {
                    "direct_distance_to_future": _mean_min_distance(direct_positions, future_xy),
                    "bootstrap_distance_to_future": _mean_min_distance(bootstrap_positions, future_xy),
                    "direct_velocity_norm_mean": float(np.linalg.norm(direct_velocity, axis=-1).mean()),
                    "bootstrap_velocity_norm_mean": float(np.linalg.norm(bootstrap_velocity, axis=-1).mean()),
                }

            metadata["states"].append(state_metrics)

        fig.suptitle("Toy Circle TD2 Targets: Direct vs Bootstrap", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.985))

        figure_path = output_dir / "toy_circle_td2_targets.png"
        figure_path_json = figure_path.with_suffix(".json")
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure_path_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return figure_path


def main() -> None:
    config = tyro.cli(
        AnalyzeToyCircleTD2TargetsConfig,
        description="Visualize direct and bootstrap TD2 target samples separately on the toy circle dataset.",
    )
    figure_path = analyze_toy_circle_td2_targets(config)
    print(str(figure_path))


if __name__ == "__main__":
    main()
