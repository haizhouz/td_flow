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

from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


MAZE_LIMIT = 0.3
MAZE_ARM_HALF_LENGTH = 0.18
MAZE_WALL_HALF_WIDTH = 0.02
BACKGROUND_COLOR = "#4e77aa"
WALL_COLOR = "#dce5ee"
SAMPLE_COLOR = "#f0d46d"
START_COLOR = "#fff06a"


@dataclass
class PointMassOccupancyConfig:
    checkpoint_path: str
    device: str = "auto"
    num_states: int = 5
    sample_count: int = 4096
    sample_batch_size: int = 1024
    min_future_steps: int = 300
    output_path: str | None = None
    seed: int = 0
    include_ground_truth: bool = True


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_path(checkpoint_path: str) -> Path:
    run_dir = _checkpoint_run_dir(checkpoint_path)
    return run_dir / "pointmass_discounted_occupancy.png"


def _default_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _resolve_hdf5_path(run_dir: Path) -> Path:
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This script currently supports only stablewm_hdf5 checkpoints.")
    dataset_dir = Path(project_config.data.dir or ".")
    dataset_path = dataset_dir / f"{project_config.data.dataset_name}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")
    return dataset_path


def _sample_valid_start_indices(ep_offset: np.ndarray, ep_len: np.ndarray, count: int, min_future_steps: int, seed: int) -> np.ndarray:
    valid_counts = np.maximum(ep_len.astype(np.int64) - int(min_future_steps), 0)
    total_valid = int(valid_counts.sum())
    if total_valid <= 0:
        raise ValueError(f"No valid start states found with min_future_steps={min_future_steps}")
    if count > total_valid:
        raise ValueError(f"Requested {count} states but only {total_valid} valid start states are available.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(total_valid, size=count, replace=False)
    cumulative = np.cumsum(valid_counts, dtype=np.int64)

    global_indices = np.empty(count, dtype=np.int64)
    for i, flat_index in enumerate(np.sort(chosen)):
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
    start_index: int,
    gamma: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    episode_index = _episode_index_for_step(ep_offset, start_index)
    episode_start = int(ep_offset[episode_index])
    local_index = int(start_index - episode_start)
    remaining = int(ep_len[episode_index]) - local_index - 1
    if remaining <= 0:
        raise ValueError(f"start_index={start_index} has no future observations available.")

    future_xy = np.asarray(observations[start_index + 1 : start_index + 1 + remaining, :2], dtype=np.float32)
    discounts = np.power(gamma, np.arange(remaining, dtype=np.float64))
    probabilities = discounts / discounts.sum()
    rng = np.random.default_rng(seed)
    sampled_offsets = rng.choice(remaining, size=sample_count, replace=True, p=probabilities)
    return future_xy[sampled_offsets]


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
        current_batch = min(int(batch_size), remaining)
        latent_batch = state_latent.expand(current_batch, -1)
        action_batch = action_tensor.expand(current_batch, -1)
        predictions = model.predict_next_latent(latent_batch, action_batch)
        chunks.append(predictions[:, :2].detach().cpu().numpy().astype(np.float32, copy=False))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


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


def _plot_occupancy_cell(
    ax: plt.Axes,
    positions: np.ndarray,
    start_xy: np.ndarray,
) -> None:
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
    ax.scatter(
        [float(start_xy[0])],
        [float(start_xy[1])],
        s=48,
        c=START_COLOR,
        edgecolors="#2f2f2f",
        linewidths=0.6,
        zorder=3,
    )


def plot_pointmass_discounted_occupancy(config: PointMassOccupancyConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This script currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (4,):
        raise NotImplementedError("This script currently supports only 4D pointmass observations.")
    if project_config.model.observation_encoder not in {"identity", "no_encoder"}:
        raise NotImplementedError("This script currently supports only identity observation encoders.")

    dataset_path = _resolve_hdf5_path(run_dir)
    gamma = float(project_config.model.gamma)
    device = _resolve_device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as handle:
        observation_key = project_config.data.observation_key
        action_key = project_config.data.action_key
        observations = handle[observation_key]
        actions = handle[action_key]
        ep_offset = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)

        start_indices = _sample_valid_start_indices(
            ep_offset,
            ep_len,
            count=config.num_states,
            min_future_steps=config.min_future_steps,
            seed=config.seed,
        )

        row_count = 2 if config.include_ground_truth else 1
        fig, axes = plt.subplots(
            row_count,
            config.num_states,
            figsize=(3.2 * config.num_states, 3.1 * row_count + 0.6),
            squeeze=False,
        )

        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "gamma": gamma,
            "num_states": config.num_states,
            "sample_count": config.sample_count,
            "sample_batch_size": config.sample_batch_size,
            "min_future_steps": config.min_future_steps,
            "seed": config.seed,
            "device": str(device),
            "start_indices": start_indices.tolist(),
            "states": [],
        }

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
            _plot_occupancy_cell(axes[0][column], model_positions, observation[:2])
            axes[0][column].set_title(
                f"state {column + 1}\nxy=({observation[0]:.2f}, {observation[1]:.2f})",
                fontsize=10,
            )

            if config.include_ground_truth:
                ground_truth_positions = _sample_ground_truth_positions(
                    observations,
                    ep_offset,
                    ep_len,
                    start_index=start_index,
                    gamma=gamma,
                    sample_count=config.sample_count,
                    seed=config.seed + column + 1,
                )
                _plot_occupancy_cell(axes[1][column], ground_truth_positions, observation[:2])

            metadata["states"].append(
                {
                    "start_index": int(start_index),
                    "observation": observation.tolist(),
                    "action": action.tolist(),
                }
            )

        axes[0][0].set_ylabel("TD-Flow\nsamples", fontsize=11)
        if config.include_ground_truth:
            axes[1][0].set_ylabel("Ground\ntruth", fontsize=11)

        fig.suptitle(
            f"PointMass Loop Discounted Occupancies (gamma={gamma:.2f})",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    metadata_path = _default_metadata_path(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        PointMassOccupancyConfig,
        description="Plot pointmass loop discounted occupancy samples from a trained TD-Flow checkpoint.",
    )
    output_path = plot_pointmass_discounted_occupancy(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
