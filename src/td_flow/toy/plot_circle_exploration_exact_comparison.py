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

from ..rollout import (
    _checkpoint_run_dir,
    load_project_config_from_run_dir,
    load_td2_model,
)
from .plot_circle_policy_conditioned_occupancy import (
    _draw_background,
    _mean_min_distance,
    _resolve_device,
    _resolve_hdf5_path,
    _sample_model_positions,
    _sample_valid_start_indices,
)


DATASET_MODEL_COLOR = "#f0d46d"
BEHAVIOR_GT_COLOR = "#79c4ff"
OFFPOLICY_MODEL_COLOR = "#ff9b63"
POLICY_GT_COLOR = "#74d07a"
POLICY_CURVE_COLOR = "#2f2f2f"
START_COLOR = "#fff06a"


@dataclass
class ToyCircleExplorationExactComparisonConfig:
    dataset_only_checkpoint_path: str
    offpolicy_checkpoint_path: str
    device: str = "auto"
    num_states: int = 5
    sample_count: int = 2048
    sample_batch_size: int = 1024
    max_future_steps: int = 1024
    seed: int = 0
    show_policy_curve: bool = False
    output_path: str | None = None


def _default_output_path(offpolicy_checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(offpolicy_checkpoint_path) / "toy_circle_exploration_exact_comparison.png"


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _sample_discounted_offsets(
    gamma: float,
    *,
    sample_count: int,
    max_future_steps: int,
    seed: int,
) -> np.ndarray:
    offsets = np.arange(max_future_steps, dtype=np.int64)
    weights = np.power(float(gamma), offsets, dtype=np.float64)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(max_future_steps, size=sample_count, replace=True, p=probabilities).astype(np.int64, copy=False)


def _positions_from_theta(theta: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32, copy=False)


def _sample_behavior_successor_positions(
    observation: np.ndarray,
    current_action: np.ndarray,
    *,
    gamma: float,
    behavior_policy: dict[str, object],
    sample_count: int,
    max_future_steps: int,
    seed: int,
) -> np.ndarray:
    start_theta = float(np.arctan2(float(observation[1]), float(observation[0])))
    theta_after_first = float(_wrap_angle(np.asarray(start_theta + float(current_action[0]), dtype=np.float64)))
    offsets = _sample_discounted_offsets(
        gamma,
        sample_count=sample_count,
        max_future_steps=max_future_steps,
        seed=seed,
    )
    max_continuation_steps = int(offsets.max(initial=0))
    if max_continuation_steps <= 0:
        theta = np.full(sample_count, theta_after_first, dtype=np.float64)
        return _positions_from_theta(theta)

    behavior_kind = str(behavior_policy["kind"])
    if behavior_kind == "uniform_random":
        rng = np.random.default_rng(seed + 17)
        continuation_actions = rng.uniform(
            low=float(behavior_policy["action_low"]),
            high=float(behavior_policy["action_high"]),
            size=(sample_count, max_continuation_steps),
        ).astype(np.float64, copy=False)
    elif behavior_kind == "disjoint_uniform":
        rng = np.random.default_rng(seed + 17)
        action_low = float(behavior_policy["action_low"])
        action_high = float(behavior_policy["action_high"])
        excluded_low = float(behavior_policy["excluded_low"])
        excluded_high = float(behavior_policy["excluded_high"])
        intervals: list[tuple[float, float]] = []
        if action_low < excluded_low:
            intervals.append((action_low, min(excluded_low, action_high)))
        if excluded_high < action_high:
            intervals.append((max(excluded_high, action_low), action_high))

        widths = np.asarray(
            [max(0.0, high - low) for low, high in intervals],
            dtype=np.float64,
        )
        if len(intervals) == 0 or float(widths.sum()) <= 0.0:
            raise ValueError("disjoint_uniform leaves no valid continuation support.")

        flat_count = sample_count * max_continuation_steps
        probabilities = widths / widths.sum()
        interval_indices = rng.choice(len(intervals), size=flat_count, p=probabilities)
        flat_actions = np.empty(flat_count, dtype=np.float64)
        for interval_index, (low, high) in enumerate(intervals):
            mask = interval_indices == interval_index
            if np.any(mask):
                flat_actions[mask] = rng.uniform(low=low, high=high, size=int(mask.sum()))
        continuation_actions = flat_actions.reshape(sample_count, max_continuation_steps)
    elif behavior_kind == "constant_delta_theta":
        continuation_actions = np.full(
            (sample_count, max_continuation_steps),
            float(behavior_policy["delta_theta"]),
            dtype=np.float64,
        )
    else:
        raise NotImplementedError(
            "Exact behavior occupancy is currently implemented only for "
            "uniform_random, disjoint_uniform, and constant_delta_theta behavior policies."
        )

    cumulative = np.concatenate(
        [
            np.zeros((sample_count, 1), dtype=np.float64),
            np.cumsum(continuation_actions, axis=1, dtype=np.float64),
        ],
        axis=1,
    )
    theta = theta_after_first + cumulative[np.arange(sample_count), offsets]
    return _positions_from_theta(_wrap_angle(theta))


def _sample_policy_successor_positions(
    observation: np.ndarray,
    current_action: np.ndarray,
    *,
    gamma: float,
    policy_delta_theta: float,
    sample_count: int,
    max_future_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    start_theta = float(np.arctan2(float(observation[1]), float(observation[0])))
    theta_after_first = float(_wrap_angle(np.asarray(start_theta + float(current_action[0]), dtype=np.float64)))
    offsets = _sample_discounted_offsets(
        gamma,
        sample_count=sample_count,
        max_future_steps=max_future_steps,
        seed=seed,
    )
    theta = theta_after_first + offsets.astype(np.float64) * float(policy_delta_theta)
    positions = _positions_from_theta(_wrap_angle(theta))

    curve_offsets = np.arange(max_future_steps, dtype=np.float64)
    curve_theta = theta_after_first + curve_offsets * float(policy_delta_theta)
    policy_curve = _positions_from_theta(_wrap_angle(curve_theta))
    return positions, policy_curve


def _symmetric_min_distance(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * (_mean_min_distance(left, right) + _mean_min_distance(right, left))


def _draw_positions(
    ax: plt.Axes,
    positions: np.ndarray,
    *,
    start_xy: np.ndarray,
    color: str,
    policy_curve: np.ndarray | None = None,
) -> None:
    _draw_background(ax)
    ax.scatter(positions[:, 0], positions[:, 1], s=5, c=color, alpha=0.12, linewidths=0)
    if policy_curve is not None:
        ax.plot(policy_curve[:, 0], policy_curve[:, 1], color=POLICY_CURVE_COLOR, linewidth=1.0, alpha=0.9, zorder=1)
    ax.scatter(
        [float(start_xy[0])],
        [float(start_xy[1])],
        s=42,
        c=START_COLOR,
        edgecolors="#2f2f2f",
        linewidths=0.6,
        zorder=3,
    )


def plot_toy_circle_exploration_exact_comparison(
    config: ToyCircleExplorationExactComparisonConfig,
) -> Path:
    dataset_only_run_dir = _checkpoint_run_dir(config.dataset_only_checkpoint_path)
    offpolicy_run_dir = _checkpoint_run_dir(config.offpolicy_checkpoint_path)
    dataset_only_project_config = load_project_config_from_run_dir(dataset_only_run_dir)
    offpolicy_project_config = load_project_config_from_run_dir(offpolicy_run_dir)

    if dataset_only_project_config.data.dataset_name != offpolicy_project_config.data.dataset_name:
        raise ValueError("Both checkpoints must use the same dataset.")
    if dataset_only_project_config.data.action_key != offpolicy_project_config.data.action_key:
        raise ValueError("Both checkpoints must use the same current action key.")
    if dataset_only_project_config.data.next_action_key != dataset_only_project_config.data.action_key:
        raise ValueError("dataset-only checkpoint must use next_action_key=action.")
    if offpolicy_project_config.data.next_action_key == offpolicy_project_config.data.action_key:
        raise ValueError("off-policy checkpoint must use a non-dataset next_action_key.")

    dataset_path = _resolve_hdf5_path(dataset_only_run_dir)
    if dataset_path != _resolve_hdf5_path(offpolicy_run_dir):
        raise ValueError("Both checkpoints must point to the same HDF5 dataset.")

    dataset_metadata = json.loads(dataset_path.with_suffix(".json").read_text())
    if dataset_metadata.get("dataset_type") != "toy_circle_exploration":
        raise ValueError("This plot expects a toy_circle_exploration dataset.")

    behavior_policy = dataset_metadata["behavior_policy"]
    target_policy = dataset_metadata["target_policy"]
    policy_delta_theta = float(target_policy["delta_theta"])
    gamma = float(offpolicy_project_config.model.gamma)

    device = _resolve_device(config.device)
    dataset_only_model = load_td2_model(config.dataset_only_checkpoint_path, dataset_only_project_config, device=device)
    offpolicy_model = load_td2_model(config.offpolicy_checkpoint_path, offpolicy_project_config, device=device)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.offpolicy_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "dataset_only_checkpoint_path": str(Path(config.dataset_only_checkpoint_path).resolve()),
        "offpolicy_checkpoint_path": str(Path(config.offpolicy_checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "gamma": gamma,
        "num_states": config.num_states,
        "sample_count": config.sample_count,
        "sample_batch_size": config.sample_batch_size,
        "max_future_steps": config.max_future_steps,
        "seed": config.seed,
        "show_policy_curve": bool(config.show_policy_curve),
        "behavior_policy": behavior_policy,
        "policy_delta_theta": policy_delta_theta,
        "states": [],
    }

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[dataset_only_project_config.data.observation_key]
        actions = handle[dataset_only_project_config.data.action_key]
        ep_offset = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        start_indices = _sample_valid_start_indices(
            ep_offset,
            ep_len,
            count=config.num_states,
            min_future_steps=1,
            seed=config.seed,
        )

        fig, axes = plt.subplots(
            4,
            config.num_states,
            figsize=(3.0 * config.num_states, 12.0),
            squeeze=False,
        )

        dataset_only_distances: list[float] = []
        offpolicy_distances: list[float] = []

        for column, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            current_action = np.asarray(actions[start_index], dtype=np.float32)
            dataset_model_positions = _sample_model_positions(
                dataset_only_model,
                observation,
                current_action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            behavior_gt_positions = _sample_behavior_successor_positions(
                observation,
                current_action,
                gamma=gamma,
                behavior_policy=behavior_policy,
                sample_count=config.sample_count,
                max_future_steps=config.max_future_steps,
                seed=config.seed + column,
            )
            offpolicy_model_positions = _sample_model_positions(
                offpolicy_model,
                observation,
                current_action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            policy_gt_positions, policy_curve = _sample_policy_successor_positions(
                observation,
                current_action,
                gamma=gamma,
                policy_delta_theta=policy_delta_theta,
                sample_count=config.sample_count,
                max_future_steps=config.max_future_steps,
                seed=config.seed + 1000 + column,
            )

            dataset_distance = _symmetric_min_distance(dataset_model_positions, behavior_gt_positions)
            offpolicy_distance = _symmetric_min_distance(offpolicy_model_positions, policy_gt_positions)
            dataset_only_distances.append(dataset_distance)
            offpolicy_distances.append(offpolicy_distance)

            _draw_positions(
                axes[0, column],
                dataset_model_positions,
                start_xy=observation[:2],
                color=DATASET_MODEL_COLOR,
            )
            _draw_positions(
                axes[1, column],
                behavior_gt_positions,
                start_xy=observation[:2],
                color=BEHAVIOR_GT_COLOR,
            )
            _draw_positions(
                axes[2, column],
                offpolicy_model_positions,
                start_xy=observation[:2],
                color=OFFPOLICY_MODEL_COLOR,
                policy_curve=policy_curve if config.show_policy_curve else None,
            )
            _draw_positions(
                axes[3, column],
                policy_gt_positions,
                start_xy=observation[:2],
                color=POLICY_GT_COLOR,
                policy_curve=policy_curve if config.show_policy_curve else None,
            )

            axes[0, column].set_title(
                f"idx={int(start_index)}  a={float(current_action[0]):+.3f}\n"
                f"dataset symNN={dataset_distance:.3f}",
                fontsize=10,
            )
            axes[2, column].set_title(
                f"off-policy symNN={offpolicy_distance:.3f}",
                fontsize=10,
            )

            if column == 0:
                axes[0, column].set_ylabel("dataset-only\nmodel", fontsize=11)
                axes[1, column].set_ylabel("exact $m^\\mu$", fontsize=11)
                axes[2, column].set_ylabel("off-policy\nmodel", fontsize=11)
                axes[3, column].set_ylabel("exact $m^\\pi$", fontsize=11)

            metadata["states"].append(
                {
                    "start_index": int(start_index),
                    "observation": observation.astype(float).tolist(),
                    "current_action": current_action.astype(float).tolist(),
                    "dataset_only_symmetric_nn_distance": dataset_distance,
                    "offpolicy_symmetric_nn_distance": offpolicy_distance,
                }
            )

    fig.suptitle(
        "Toy circle exact occupancy comparison\n"
        "Rows 1-2 compare dataset-only model to exact behavior occupancy. "
        "Rows 3-4 compare off-policy model to exact policy occupancy.",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata.update(
        {
            "mean_dataset_only_symmetric_nn_distance": float(np.mean(dataset_only_distances)),
            "mean_offpolicy_symmetric_nn_distance": float(np.mean(offpolicy_distances)),
        }
    )
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        ToyCircleExplorationExactComparisonConfig,
        description="Compare toy-circle dataset-only and off-policy checkpoints against exact known occupancies.",
    )
    output_path = plot_toy_circle_exploration_exact_comparison(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
