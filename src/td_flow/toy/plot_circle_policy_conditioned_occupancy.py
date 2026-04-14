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

from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


START_COLOR = "#fff06a"
MODEL_COLOR = "#f0d46d"
ROLLOUT_COLOR = "#79c4ff"
TRAJECTORY_COLOR = "#2f2f2f"


@dataclass
class ToyCirclePolicyConditionedOccupancyConfig:
    checkpoint_path: str
    device: str = "auto"
    num_states: int = 5
    sample_count: int = 2048
    sample_batch_size: int = 1024
    min_future_steps: int = 100
    gamma_override: float | None = None
    seed: int = 0
    output_path: str | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_path(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "toy_circle_policy_conditioned_occupancy.png"


def _resolve_hdf5_path(run_dir: Path) -> Path:
    project_config = load_project_config_from_run_dir(run_dir)
    dataset_dir = Path(project_config.data.dir or ".")
    dataset_path = dataset_dir / f"{project_config.data.dataset_name}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")
    return dataset_path


def _sample_valid_start_indices(
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    *,
    count: int,
    min_future_steps: int,
    seed: int,
) -> np.ndarray:
    valid_counts = np.maximum(ep_len.astype(np.int64) - int(min_future_steps), 0)
    total_valid = int(valid_counts.sum())
    if total_valid <= 0:
        raise ValueError(f"No valid states found with min_future_steps={min_future_steps}")
    if count > total_valid:
        raise ValueError(f"Requested {count} states but only {total_valid} valid states are available.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total_valid, size=count, replace=False))
    cumulative = np.cumsum(valid_counts, dtype=np.int64)
    global_indices = np.empty(count, dtype=np.int64)
    for i, flat_index in enumerate(chosen):
        episode_index = int(np.searchsorted(cumulative, flat_index, side="right"))
        previous_total = 0 if episode_index == 0 else int(cumulative[episode_index - 1])
        local_index = int(flat_index - previous_total)
        global_indices[i] = int(ep_offset[episode_index]) + local_index
    return global_indices


def _episode_index_for_step(ep_offset: np.ndarray, step_index: int) -> int:
    return int(np.searchsorted(ep_offset, step_index, side="right") - 1)


def _sample_ground_truth_positions(
    observations: h5py.Dataset,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    *,
    start_index: int,
    gamma: float,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episode_index = _episode_index_for_step(ep_offset, start_index)
    episode_start = int(ep_offset[episode_index])
    local_index = int(start_index - episode_start)
    remaining = int(ep_len[episode_index]) - local_index - 1
    if remaining <= 0:
        raise ValueError(f"start_index={start_index} has no future observations.")

    future_positions = np.asarray(
        observations[start_index + 1 : start_index + 1 + remaining, :2],
        dtype=np.float32,
    )
    discounts = np.power(gamma, np.arange(remaining, dtype=np.float64))
    probabilities = discounts / discounts.sum()
    rng = np.random.default_rng(seed)
    sampled_offsets = rng.choice(remaining, size=sample_count, replace=True, p=probabilities)
    return future_positions[sampled_offsets], future_positions


@torch.no_grad()
def _sample_model_positions(
    model,
    observation: np.ndarray,
    action: np.ndarray,
    *,
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
        current_batch = min(batch_size, remaining)
        predictions = model.predict_next_latent(
            state_latent.expand(current_batch, -1),
            action_tensor.expand(current_batch, -1),
        )
        chunks.append(predictions[:, :2].detach().cpu().numpy().astype(np.float32, copy=False))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


def _mean_min_distance(source: np.ndarray, target: np.ndarray) -> float:
    deltas = source[:, None, :] - target[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return float(distances.min(axis=1).mean())


def _draw_background(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 512, dtype=np.float64)
    ax.plot(np.cos(theta), np.sin(theta), color="#d9dde2", linewidth=1.0, zorder=0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_toy_circle_policy_conditioned_occupancy(config: ToyCirclePolicyConditionedOccupancyConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This script currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (2,):
        raise NotImplementedError("This script currently supports only 2D toy-circle observations.")
    if project_config.model.observation_encoder not in {"auto", "identity", "no_encoder"}:
        raise NotImplementedError("This script currently supports only identity-like observation encoders.")

    device = _resolve_device(config.device)
    dataset_path = _resolve_hdf5_path(run_dir)
    gamma = float(project_config.model.gamma if config.gamma_override is None else config.gamma_override)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "gamma": gamma,
        "num_states": config.num_states,
        "sample_count": config.sample_count,
        "sample_batch_size": config.sample_batch_size,
            "min_future_steps": config.min_future_steps,
            "gamma_override": None if config.gamma_override is None else float(config.gamma_override),
            "seed": config.seed,
            "states": [],
        }

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
            2,
            config.num_states,
            figsize=(3.0 * config.num_states, 6.1),
            squeeze=False,
        )

        model_from_start: list[float] = []
        rollout_from_start: list[float] = []
        model_to_rollout: list[float] = []
        rollout_to_model: list[float] = []

        for column, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            action = np.asarray(actions[start_index], dtype=np.float32)
            model_positions = _sample_model_positions(
                model,
                observation,
                action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            gt_samples, future_curve = _sample_ground_truth_positions(
                observations,
                ep_offset,
                ep_len,
                start_index=start_index,
                gamma=gamma,
                sample_count=config.sample_count,
                seed=config.seed + column,
            )

            _draw_background(axes[0, column])
            _draw_background(axes[1, column])
            axes[0, column].scatter(model_positions[:, 0], model_positions[:, 1], s=5, c=MODEL_COLOR, alpha=0.12, linewidths=0)
            axes[1, column].scatter(gt_samples[:, 0], gt_samples[:, 1], s=5, c=ROLLOUT_COLOR, alpha=0.12, linewidths=0)
            axes[0, column].plot(future_curve[:, 0], future_curve[:, 1], color=TRAJECTORY_COLOR, linewidth=1.0, alpha=0.85, zorder=1)
            axes[1, column].plot(future_curve[:, 0], future_curve[:, 1], color=TRAJECTORY_COLOR, linewidth=1.0, alpha=0.85, zorder=1)
            axes[0, column].scatter([float(observation[0])], [float(observation[1])], s=42, c=START_COLOR, edgecolors="#2f2f2f", linewidths=0.6, zorder=3)
            axes[1, column].scatter([float(observation[0])], [float(observation[1])], s=42, c=START_COLOR, edgecolors="#2f2f2f", linewidths=0.6, zorder=3)

            if column == 0:
                axes[0, column].set_ylabel("Model", fontsize=11)
                axes[1, column].set_ylabel("Rollout", fontsize=11)

            model_from_start.append(float(np.linalg.norm(model_positions - observation[:2], axis=-1).mean()))
            rollout_from_start.append(float(np.linalg.norm(gt_samples - observation[:2], axis=-1).mean()))
            model_to_rollout.append(_mean_min_distance(model_positions, future_curve))
            rollout_to_model.append(_mean_min_distance(gt_samples, model_positions))

            metadata["states"].append(
                {
                    "start_index": int(start_index),
                    "observation": observation.astype(float).tolist(),
                    "action": action.astype(float).tolist(),
                    "model_distance_from_start": model_from_start[-1],
                    "rollout_distance_from_start": rollout_from_start[-1],
                    "model_to_rollout_min_distance": model_to_rollout[-1],
                    "rollout_to_model_min_distance": rollout_to_model[-1],
                }
            )

    fig.suptitle("Toy Circle TD2 Occupancy Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata.update(
        {
            "mean_model_distance_from_start": float(np.mean(model_from_start)),
            "mean_rollout_distance_from_start": float(np.mean(rollout_from_start)),
            "mean_model_to_rollout_min_distance": float(np.mean(model_to_rollout)),
            "mean_rollout_to_model_min_distance": float(np.mean(rollout_to_model)),
        }
    )
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        ToyCirclePolicyConditionedOccupancyConfig,
        description="Plot toy-circle policy-conditioned occupancy comparison for a TD2 checkpoint.",
    )
    output_path = plot_toy_circle_policy_conditioned_occupancy(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
