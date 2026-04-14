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
from .plot_policy_conditioned_occupancy import _resolve_hdf5_path
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


DIRECT_COLOR = "#f0d46d"
BOOTSTRAP_COLOR = "#ff8f79"


@dataclass
class AnalyzePointMassBootstrapPathOverTimeConfig:
    checkpoint_path: str
    device: str = "auto"
    num_states: int = 3
    sample_count: int = 2048
    min_future_steps: int = 300
    t_values: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
    )
    trajectory_overlay_steps: int = 300
    seed: int = 0
    output_dir: str | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_dir(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "bootstrap_path_over_time"


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


def _future_trajectory(
    observations: h5py.Dataset,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    start_index: int,
) -> np.ndarray:
    episode_index = _episode_index_for_step(ep_offset, start_index)
    episode_start = int(ep_offset[episode_index])
    local_index = int(start_index - episode_start)
    remaining = int(ep_len[episode_index]) - local_index
    return np.asarray(observations[start_index : start_index + remaining], dtype=np.float32)


def _mean_min_distance(samples_xy: np.ndarray, support_xy: np.ndarray) -> float:
    diffs = samples_xy[:, None, :] - support_xy[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.min(axis=1).mean())


def _xy_spread(samples_xy: np.ndarray) -> dict[str, float]:
    std_xy = samples_xy.std(axis=0)
    return {
        "std_x": float(std_xy[0]),
        "std_y": float(std_xy[1]),
        "std_mean": float(std_xy.mean()),
        "rms_radius": float(np.sqrt(np.mean(np.sum((samples_xy - samples_xy.mean(axis=0, keepdims=True)) ** 2, axis=-1)))),
    }


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


def analyze_pointmass_bootstrap_path_over_time(
    config: AnalyzePointMassBootstrapPathOverTimeConfig,
) -> Path:
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
            "min_future_steps": config.min_future_steps,
            "t_values": list(config.t_values),
            "trajectory_overlay_steps": config.trajectory_overlay_steps,
            "seed": config.seed,
            "states": [],
        }

        fig, axes = plt.subplots(
            2,
            config.num_states,
            figsize=(4.4 * config.num_states, 8.0),
            squeeze=False,
        )

        for col, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            action = np.asarray(actions[start_index], dtype=np.float32)
            next_observation = np.asarray(observations[start_index + 1], dtype=np.float32)
            next_action = np.asarray(next_actions[start_index + 1], dtype=np.float32)
            future_observations = _future_trajectory(observations, ep_offset, ep_len, start_index)
            next_overlay_xy = future_observations[1 : config.trajectory_overlay_steps + 1, :2]

            direct_distances: list[float] = []
            bootstrap_distances: list[float] = []
            direct_std_mean: list[float] = []
            bootstrap_std_mean: list[float] = []
            direct_rms_radius: list[float] = []
            bootstrap_rms_radius: list[float] = []

            state_metrics: dict[str, object] = {
                "start_index": int(start_index),
                "observation": observation.tolist(),
                "action": action.tolist(),
                "next_observation": next_observation.tolist(),
                "next_action": next_action.tolist(),
                "t_values": {},
            }

            for t_value in config.t_values:
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
                direct_distance = _mean_min_distance(direct_positions, next_overlay_xy)
                bootstrap_distance = _mean_min_distance(bootstrap_positions, next_overlay_xy)
                direct_spread = _xy_spread(direct_positions)
                bootstrap_spread = _xy_spread(bootstrap_positions)

                direct_distances.append(direct_distance)
                bootstrap_distances.append(bootstrap_distance)
                direct_std_mean.append(direct_spread["std_mean"])
                bootstrap_std_mean.append(bootstrap_spread["std_mean"])
                direct_rms_radius.append(direct_spread["rms_radius"])
                bootstrap_rms_radius.append(bootstrap_spread["rms_radius"])

                state_metrics["t_values"][f"{float(t_value):.2f}"] = {
                    "direct_distance_to_future": direct_distance,
                    "bootstrap_distance_to_future": bootstrap_distance,
                    "direct_spread": direct_spread,
                    "bootstrap_spread": bootstrap_spread,
                }

            t_axis = np.asarray(config.t_values, dtype=np.float32)
            axes[0][col].plot(t_axis, direct_distances, color=DIRECT_COLOR, linewidth=2.0, label="direct")
            axes[0][col].plot(t_axis, bootstrap_distances, color=BOOTSTRAP_COLOR, linewidth=2.0, label="bootstrap")
            axes[0][col].set_title(f"state {col + 1} distance", fontsize=11)
            axes[0][col].set_xlabel("t")
            axes[0][col].set_ylabel("mean min distance")
            axes[0][col].grid(alpha=0.25, linewidth=0.6)
            axes[0][col].legend(frameon=False, fontsize=9)

            axes[1][col].plot(t_axis, direct_std_mean, color=DIRECT_COLOR, linewidth=2.0, label="direct std_xy")
            axes[1][col].plot(t_axis, bootstrap_std_mean, color=BOOTSTRAP_COLOR, linewidth=2.0, label="bootstrap std_xy")
            axes[1][col].plot(t_axis, direct_rms_radius, color=DIRECT_COLOR, linewidth=1.2, linestyle="--", label="direct rms")
            axes[1][col].plot(t_axis, bootstrap_rms_radius, color=BOOTSTRAP_COLOR, linewidth=1.2, linestyle="--", label="bootstrap rms")
            axes[1][col].set_title(f"state {col + 1} spread", fontsize=11)
            axes[1][col].set_xlabel("t")
            axes[1][col].set_ylabel("spread")
            axes[1][col].grid(alpha=0.25, linewidth=0.6)
            axes[1][col].legend(frameon=False, fontsize=8)

            metadata["states"].append(state_metrics)

        fig.suptitle("Pointmass TD2 bootstrap path over time", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
        figure_path = output_dir / "pointmass_bootstrap_path_over_time.png"
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    metadata_path = output_dir / "pointmass_bootstrap_path_over_time.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_dir


def main() -> None:
    config = tyro.cli(
        AnalyzePointMassBootstrapPathOverTimeConfig,
        description="Track pointmass direct/bootstrap path distance and spread across flow time.",
    )
    output_dir = analyze_pointmass_bootstrap_path_over_time(config)
    print(str(output_dir))


if __name__ == "__main__":
    main()
