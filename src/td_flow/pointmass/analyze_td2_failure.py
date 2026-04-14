from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import tyro

from ..paths import sample_source
from .plot_policy_conditioned_occupancy import _resolve_hdf5_path
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


MAZE_LIMIT = 0.3
MAZE_ARM_HALF_LENGTH = 0.18
MAZE_WALL_HALF_WIDTH = 0.02
BACKGROUND_COLOR = "#4e77aa"
WALL_COLOR = "#dce5ee"
SAMPLE_COLOR = "#f0d46d"
TARGET_COLOR = "#77e0a0"
START_COLOR = "#fff06a"


@dataclass
class AnalyzePointMassTD2FailureConfig:
    checkpoint_path: str
    device: str = "auto"
    num_states: int = 3
    sample_count: int = 2048
    sample_batch_size: int = 1024
    min_future_steps: int = 300
    t_values: tuple[float, ...] = (0.5, 0.75, 0.9, 0.99, 1.0)
    vector_field_t: float = 0.99
    vector_field_grid_size: int = 17
    vector_field_xy_radius: float = 0.04
    trajectory_overlay_steps: int = 300
    seed: int = 0
    output_dir: str | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_dir(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "td2_audit"


def _draw_maze_background(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.add_patch(
        Rectangle(
            (-MAZE_ARM_HALF_LENGTH, -MAZE_WALL_HALF_WIDTH),
            2 * MAZE_ARM_HALF_LENGTH,
            2 * MAZE_WALL_HALF_WIDTH,
            facecolor=WALL_COLOR,
            edgecolor="none",
            alpha=0.9,
        )
    )
    ax.add_patch(
        Rectangle(
            (-MAZE_WALL_HALF_WIDTH, -MAZE_ARM_HALF_LENGTH),
            2 * MAZE_WALL_HALF_WIDTH,
            2 * MAZE_ARM_HALF_LENGTH,
            facecolor=WALL_COLOR,
            edgecolor="none",
            alpha=0.9,
        )
    )
    ax.set_xlim(-MAZE_LIMIT, MAZE_LIMIT)
    ax.set_ylim(-MAZE_LIMIT, MAZE_LIMIT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_density(ax: plt.Axes, positions: np.ndarray, start_xy: np.ndarray, trajectory_xy: np.ndarray) -> None:
    _draw_maze_background(ax)
    bin_count = 72
    density, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=bin_count,
        range=[[-MAZE_LIMIT, MAZE_LIMIT], [-MAZE_LIMIT, MAZE_LIMIT]],
    )
    if density.max() > 0:
        normalized = (density.T / density.max()) ** 0.45
        overlay = np.zeros((bin_count, bin_count, 4), dtype=np.float32)
        overlay[..., :3] = np.asarray(mcolors.to_rgb(SAMPLE_COLOR), dtype=np.float32)
        overlay[..., 3] = np.clip(normalized * 0.95, 0.0, 0.95)
        ax.imshow(
            overlay,
            extent=(-MAZE_LIMIT, MAZE_LIMIT, -MAZE_LIMIT, MAZE_LIMIT),
            origin="lower",
            interpolation="bilinear",
            zorder=2,
        )
    ax.plot(
        trajectory_xy[:, 0],
        trajectory_xy[:, 1],
        color=TARGET_COLOR,
        linewidth=1.2,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        [float(start_xy[0])],
        [float(start_xy[1])],
        s=48,
        c=START_COLOR,
        edgecolors="#2f2f2f",
        linewidths=0.6,
        zorder=4,
    )


def _plot_quiver(
    ax: plt.Axes,
    grid_xy: np.ndarray,
    velocity_xy: np.ndarray,
    start_xy: np.ndarray,
    trajectory_xy: np.ndarray,
    next_xy: np.ndarray,
) -> None:
    _draw_maze_background(ax)
    ax.plot(
        trajectory_xy[:, 0],
        trajectory_xy[:, 1],
        color=TARGET_COLOR,
        linewidth=1.2,
        alpha=0.95,
        zorder=2,
    )
    ax.quiver(
        grid_xy[:, 0],
        grid_xy[:, 1],
        velocity_xy[:, 0],
        velocity_xy[:, 1],
        color=SAMPLE_COLOR,
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.004,
        alpha=0.8,
        zorder=3,
    )
    ax.scatter(
        [float(start_xy[0])],
        [float(start_xy[1])],
        s=44,
        c=START_COLOR,
        edgecolors="#2f2f2f",
        linewidths=0.6,
        zorder=4,
    )
    ax.scatter(
        [float(next_xy[0])],
        [float(next_xy[1])],
        s=32,
        c=TARGET_COLOR,
        edgecolors="#2f2f2f",
        linewidths=0.5,
        zorder=4,
    )


def _sample_valid_start_indices(
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    count: int,
    min_future_steps: int,
    seed: int,
) -> np.ndarray:
    valid_counts = np.maximum(ep_len.astype(np.int64) - int(min_future_steps), 0)
    total_valid = int(valid_counts.sum())
    if total_valid <= 0:
        raise ValueError(f"No valid start states found with min_future_steps={min_future_steps}")
    if count > total_valid:
        raise ValueError(f"Requested {count} states but only {total_valid} valid states are available.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total_valid, size=count, replace=False))
    cumulative = np.cumsum(valid_counts, dtype=np.int64)
    indices = np.empty(count, dtype=np.int64)
    for i, flat_index in enumerate(chosen):
        episode_index = int(np.searchsorted(cumulative, flat_index, side="right"))
        previous_total = 0 if episode_index == 0 else int(cumulative[episode_index - 1])
        local_index = int(flat_index - previous_total)
        indices[i] = int(ep_offset[episode_index]) + local_index
    return indices


def _episode_index_for_step(ep_offset: np.ndarray, step_index: int) -> int:
    return int(np.searchsorted(ep_offset, step_index, side="right") - 1)


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
def _sample_model_positions_for_t(
    model,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    t_end: float,
    device: torch.device,
    sample_count: int,
    batch_size: int,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(observation).to(device=device, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.from_numpy(action).to(device=device, dtype=torch.float32).unsqueeze(0)
    state_latent = model.encode_observation(obs_tensor)

    chunks: list[np.ndarray] = []
    remaining = sample_count
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        latent_batch = state_latent.expand(current_batch, -1)
        action_batch = action_tensor.expand(current_batch, -1)
        predictions = model.predict_next_latent(
            latent_batch,
            action_batch,
            t_end=float(t_end),
        )
        chunks.append(predictions[:, :2].detach().cpu().numpy().astype(np.float32, copy=False))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _sample_target_endpoint_positions(
    model,
    next_observation: np.ndarray,
    next_action: np.ndarray,
    *,
    device: torch.device,
    sample_count: int,
    batch_size: int,
) -> np.ndarray:
    next_obs_tensor = torch.from_numpy(next_observation).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_action_tensor = torch.from_numpy(next_action).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_latent = model.encode_observation(next_obs_tensor, use_target=True)

    chunks: list[np.ndarray] = []
    remaining = sample_count
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        latent_batch = next_latent.expand(current_batch, -1)
        action_batch = next_action_tensor.expand(current_batch, -1)
        predictions = model.predict_next_latent(
            latent_batch,
            action_batch,
            t_end=1.0,
            use_target=True,
        )
        chunks.append(predictions[:, :2].detach().cpu().numpy().astype(np.float32, copy=False))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _sample_bootstrap_xt_positions(
    model,
    next_observation: np.ndarray,
    next_action: np.ndarray,
    *,
    device: torch.device,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    next_obs_tensor = torch.from_numpy(next_observation).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_action_tensor = torch.from_numpy(next_action).to(device=device, dtype=torch.float32).unsqueeze(0)
    next_latent = model.encode_observation(next_obs_tensor, use_target=True).expand(sample_count, -1)
    next_action_batch = next_action_tensor.expand(sample_count, -1)
    t = torch.rand(sample_count, device=device, dtype=next_latent.dtype).clamp_(model.cfg.time_eps, 1.0 - model.cfg.time_eps)
    source = sample_source(
        sample_count,
        model.latent_dim,
        device=device,
        dtype=next_latent.dtype,
    )
    bootstrap_xt, _ = model.bootstrap_target(
        next_latent,
        next_action_batch,
        source,
        t,
    )
    return (
        bootstrap_xt[:, :2].detach().cpu().numpy().astype(np.float32, copy=False),
        t.detach().cpu().numpy().astype(np.float32, copy=False),
    )


@torch.no_grad()
def _vector_field_grid(
    model,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    *,
    device: torch.device,
    t_value: float,
    grid_size: int,
    xy_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    next_obs = np.asarray(next_observation, dtype=np.float32)
    x_values = np.linspace(next_obs[0] - xy_radius, next_obs[0] + xy_radius, grid_size, dtype=np.float32)
    y_values = np.linspace(next_obs[1] - xy_radius, next_obs[1] + xy_radius, grid_size, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    grid_xy = np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=-1)

    grid_latent = np.repeat(next_obs[None, :], repeats=grid_xy.shape[0], axis=0)
    grid_latent[:, :2] = grid_xy

    x_t = torch.from_numpy(grid_latent).to(device=device, dtype=torch.float32)
    obs_tensor = torch.from_numpy(observation).to(device=device, dtype=torch.float32).unsqueeze(0).expand(grid_xy.shape[0], -1)
    action_tensor = torch.from_numpy(action).to(device=device, dtype=torch.float32).unsqueeze(0).expand(grid_xy.shape[0], -1)
    state_latent = model.encode_observation(obs_tensor)
    t_tensor = torch.full((grid_xy.shape[0],), float(t_value), device=device, dtype=x_t.dtype)
    velocity = model.compute_velocity(
        x_t,
        t_tensor,
        state_latent,
        action_tensor,
    )
    return (
        grid_xy.astype(np.float32, copy=False),
        velocity[:, :2].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def analyze_pointmass_td2_failure(config: AnalyzePointMassTD2FailureConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This analysis currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (4,):
        raise NotImplementedError("This analysis currently supports only 4D pointmass observations.")
    if project_config.model.observation_encoder not in {"identity", "no_encoder"}:
        raise NotImplementedError("This analysis currently supports only identity observation encoders.")

    device = _resolve_device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)
    dataset_path = _resolve_hdf5_path(run_dir)

    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir(config.checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[project_config.data.observation_key]
        actions = handle[project_config.data.action_key]
        next_actions = handle[project_config.data.next_action_key] if project_config.data.next_action_key is not None else actions
        ep_offset = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        start_indices = _sample_valid_start_indices(
            ep_offset,
            ep_len,
            count=config.num_states,
            min_future_steps=config.min_future_steps,
            seed=config.seed,
        )

        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "device": str(device),
            "sample_count": config.sample_count,
            "sample_batch_size": config.sample_batch_size,
            "min_future_steps": config.min_future_steps,
            "t_values": list(config.t_values),
            "vector_field_t": config.vector_field_t,
            "vector_field_grid_size": config.vector_field_grid_size,
            "vector_field_xy_radius": config.vector_field_xy_radius,
            "trajectory_overlay_steps": config.trajectory_overlay_steps,
            "seed": config.seed,
            "states": [],
        }

        ode_fig, ode_axes = plt.subplots(
            config.num_states,
            len(config.t_values),
            figsize=(3.0 * len(config.t_values), 3.0 * config.num_states),
            squeeze=False,
        )
        bootstrap_fig, bootstrap_axes = plt.subplots(
            config.num_states,
            3,
            figsize=(9.0, 3.0 * config.num_states),
            squeeze=False,
        )
        field_fig, field_axes = plt.subplots(
            config.num_states,
            1,
            figsize=(4.4, 4.0 * config.num_states),
            squeeze=False,
        )

        for row, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            action = np.asarray(actions[start_index], dtype=np.float32)
            next_observation = np.asarray(observations[start_index + 1], dtype=np.float32)
            next_action = np.asarray(next_actions[start_index + 1], dtype=np.float32)
            future_observations = _future_trajectory(observations, ep_offset, ep_len, start_index)
            next_future_observations = future_observations[1:]
            overlay_trajectory_xy = future_observations[: config.trajectory_overlay_steps, :2]
            next_overlay_xy = next_future_observations[: config.trajectory_overlay_steps, :2]

            state_metrics: dict[str, object] = {
                "start_index": int(start_index),
                "observation": observation.tolist(),
                "action": action.tolist(),
                "next_observation": next_observation.tolist(),
                "next_action": next_action.tolist(),
                "ode_time_distances": {},
            }

            for col, t_end in enumerate(config.t_values):
                sample_positions = _sample_model_positions_for_t(
                    model,
                    observation,
                    action,
                    t_end=float(t_end),
                    device=device,
                    sample_count=config.sample_count,
                    batch_size=config.sample_batch_size,
                )
                _plot_density(
                    ode_axes[row][col],
                    sample_positions,
                    observation[:2],
                    overlay_trajectory_xy,
                )
                ode_axes[row][col].set_title(f"t_end={t_end:.2f}", fontsize=10)
                state_metrics["ode_time_distances"][f"{t_end:.2f}"] = _mean_min_distance(
                    sample_positions,
                    overlay_trajectory_xy,
                )

            target_endpoint_positions = _sample_target_endpoint_positions(
                model,
                next_observation,
                next_action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            bootstrap_xt_positions, bootstrap_t = _sample_bootstrap_xt_positions(
                model,
                next_observation,
                next_action,
                device=device,
                sample_count=config.sample_count,
            )
            true_discounted_positions = next_future_observations[
                np.random.default_rng(config.seed + row).choice(
                    len(next_future_observations),
                    size=config.sample_count,
                    replace=True,
                    p=np.power(project_config.model.gamma, np.arange(len(next_future_observations), dtype=np.float64))
                    / np.power(project_config.model.gamma, np.arange(len(next_future_observations), dtype=np.float64)).sum(),
                )
            ][:, :2]

            _plot_density(
                bootstrap_axes[row][0],
                target_endpoint_positions,
                next_observation[:2],
                next_overlay_xy,
            )
            bootstrap_axes[row][0].set_title("target endpoint\nfrom (s', a')", fontsize=10)
            _plot_density(
                bootstrap_axes[row][1],
                bootstrap_xt_positions,
                next_observation[:2],
                next_overlay_xy,
            )
            bootstrap_axes[row][1].set_title("bootstrap x_t\nsampled t ~ U[0,1]", fontsize=10)
            _plot_density(
                bootstrap_axes[row][2],
                true_discounted_positions,
                next_observation[:2],
                next_overlay_xy,
            )
            bootstrap_axes[row][2].set_title("true discounted\nfuture from s'", fontsize=10)

            state_metrics["target_endpoint_distance"] = _mean_min_distance(target_endpoint_positions, next_overlay_xy)
            state_metrics["bootstrap_xt_distance"] = _mean_min_distance(bootstrap_xt_positions, next_overlay_xy)
            state_metrics["bootstrap_t_stats"] = {
                "mean": float(bootstrap_t.mean()),
                "std": float(bootstrap_t.std()),
                "min": float(bootstrap_t.min()),
                "max": float(bootstrap_t.max()),
            }

            grid_xy, velocity_xy = _vector_field_grid(
                model,
                observation,
                action,
                next_observation,
                device=device,
                t_value=config.vector_field_t,
                grid_size=config.vector_field_grid_size,
                xy_radius=config.vector_field_xy_radius,
            )
            _plot_quiver(
                field_axes[row][0],
                grid_xy,
                velocity_xy,
                observation[:2],
                overlay_trajectory_xy,
                next_observation[:2],
            )
            field_axes[row][0].set_title(
                f"local vector field near true next state, t={config.vector_field_t:.2f}",
                fontsize=11,
            )

            center_index = int(np.argmin(np.linalg.norm(grid_xy - next_observation[:2], axis=1)))
            center_velocity = velocity_xy[center_index]
            true_step = next_future_observations[1, :2] - next_future_observations[0, :2]
            denom = max(float(np.linalg.norm(center_velocity) * np.linalg.norm(true_step)), 1e-12)
            state_metrics["center_velocity_xy"] = center_velocity.tolist()
            state_metrics["true_step_xy"] = true_step.tolist()
            state_metrics["center_velocity_cosine_to_true_step"] = float(np.dot(center_velocity, true_step) / denom)

            metadata["states"].append(state_metrics)

        ode_axes[0][0].set_ylabel("model ODE-time samples", fontsize=11)
        bootstrap_axes[0][0].set_ylabel("bootstrap audit", fontsize=11)
        field_axes[0][0].set_ylabel("vector field", fontsize=11)

        ode_fig.suptitle("TD2 audit: endpoint structure across ODE time", fontsize=14, y=0.995)
        ode_fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
        bootstrap_fig.suptitle("TD2 audit: bootstrap targets vs true continuation", fontsize=14, y=0.995)
        bootstrap_fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
        field_fig.suptitle("TD2 audit: local vector field geometry", fontsize=14, y=0.995)
        field_fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))

        ode_path = output_dir / "pointmass_td2_audit_ode_times.png"
        bootstrap_path = output_dir / "pointmass_td2_audit_bootstrap.png"
        field_path = output_dir / "pointmass_td2_audit_vector_field.png"
        ode_fig.savefig(ode_path, dpi=220, bbox_inches="tight")
        bootstrap_fig.savefig(bootstrap_path, dpi=220, bbox_inches="tight")
        field_fig.savefig(field_path, dpi=220, bbox_inches="tight")
        plt.close(ode_fig)
        plt.close(bootstrap_fig)
        plt.close(field_fig)

    metadata_path = output_dir / "pointmass_td2_audit.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_dir


def main() -> None:
    config = tyro.cli(
        AnalyzePointMassTD2FailureConfig,
        description="Diagnose why a pointmass TD2-CFM checkpoint fails to capture deterministic occupancy structure.",
    )
    output_dir = analyze_pointmass_td2_failure(config)
    print(str(output_dir))


if __name__ == "__main__":
    main()
