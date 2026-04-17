from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class GenerateToyCircleExplorationDatasetConfig:
    output_hdf5_path: str
    num_episodes: int = 2048
    episode_length: int = 256
    radius: float = 1.0
    policy_delta_theta: float = 0.02
    behavior_policy_kind: str = "uniform_random"
    behavior_delta_theta: float = -0.02
    behavior_action_limit: float = 0.25
    behavior_exclusion_radius: float = 0.05
    seed: int = 0
    overwrite: bool = False


def _behavior_policy_metadata(config: GenerateToyCircleExplorationDatasetConfig) -> dict[str, object]:
    if config.behavior_policy_kind == "uniform_random":
        return {
            "kind": "uniform_random",
            "action_low": -float(config.behavior_action_limit),
            "action_high": float(config.behavior_action_limit),
        }
    if config.behavior_policy_kind == "constant_delta_theta":
        return {
            "kind": "constant_delta_theta",
            "delta_theta": float(config.behavior_delta_theta),
        }
    if config.behavior_policy_kind == "disjoint_uniform":
        low = float(config.policy_delta_theta - config.behavior_exclusion_radius)
        high = float(config.policy_delta_theta + config.behavior_exclusion_radius)
        return {
            "kind": "disjoint_uniform",
            "action_low": -float(config.behavior_action_limit),
            "action_high": float(config.behavior_action_limit),
            "excluded_low": low,
            "excluded_high": high,
        }
    raise ValueError(
        "behavior_policy_kind must be one of: uniform_random, constant_delta_theta, disjoint_uniform"
    )


def _sample_behavior_actions(
    rng: np.random.Generator,
    config: GenerateToyCircleExplorationDatasetConfig,
    *,
    size: int,
) -> np.ndarray:
    action_limit = float(config.behavior_action_limit)
    if action_limit <= 0.0:
        raise ValueError("behavior_action_limit must be positive.")

    if config.behavior_policy_kind == "uniform_random":
        return rng.uniform(
            low=-action_limit,
            high=action_limit,
            size=size,
        ).astype(np.float64, copy=False)

    if config.behavior_policy_kind == "constant_delta_theta":
        delta = float(config.behavior_delta_theta)
        if abs(delta) > action_limit + 1e-12:
            raise ValueError(
                "behavior_delta_theta must satisfy "
                "|behavior_delta_theta| <= behavior_action_limit."
            )
        return np.full(size, delta, dtype=np.float64)

    if config.behavior_policy_kind != "disjoint_uniform":
        raise ValueError(
            "behavior_policy_kind must be one of: uniform_random, constant_delta_theta, disjoint_uniform"
        )

    exclusion_radius = float(config.behavior_exclusion_radius)
    if exclusion_radius <= 0.0:
        raise ValueError(
            "behavior_exclusion_radius must be positive for disjoint_uniform."
        )

    excluded_low = float(config.policy_delta_theta - exclusion_radius)
    excluded_high = float(config.policy_delta_theta + exclusion_radius)
    intervals: list[tuple[float, float]] = []
    if -action_limit < excluded_low:
        intervals.append((-action_limit, min(excluded_low, action_limit)))
    if excluded_high < action_limit:
        intervals.append((max(excluded_high, -action_limit), action_limit))

    widths = np.asarray(
        [max(0.0, high - low) for low, high in intervals], dtype=np.float64
    )
    if len(intervals) == 0 or float(widths.sum()) <= 0.0:
        raise ValueError(
            "disjoint_uniform leaves no valid behavior support. "
            "Reduce behavior_exclusion_radius or increase behavior_action_limit."
        )

    probabilities = widths / widths.sum()
    interval_indices = rng.choice(len(intervals), size=size, p=probabilities)
    samples = np.empty(size, dtype=np.float64)
    for interval_index, (low, high) in enumerate(intervals):
        mask = interval_indices == interval_index
        if np.any(mask):
            samples[mask] = rng.uniform(low=low, high=high, size=int(mask.sum()))
    return samples


def generate_toy_circle_exploration_dataset(
    config: GenerateToyCircleExplorationDatasetConfig,
) -> Path:
    output_path = Path(config.output_hdf5_path)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = config.num_episodes * config.episode_length
    observations = np.zeros((total_steps, 2), dtype=np.float32)
    actions = np.zeros((total_steps, 1), dtype=np.float32)
    policy_actions = np.full((total_steps, 1), float(config.policy_delta_theta), dtype=np.float32)
    rewards = np.zeros((total_steps, 1), dtype=np.float32)
    discounts = np.ones((total_steps, 1), dtype=np.float32)
    ep_len = np.full(config.num_episodes, config.episode_length, dtype=np.int64)
    ep_offset = np.arange(config.num_episodes, dtype=np.int64) * config.episode_length

    rng = np.random.default_rng(config.seed)
    cursor = 0
    for _episode_index in range(config.num_episodes):
        theta = np.empty(config.episode_length, dtype=np.float64)
        theta[0] = float(rng.uniform(-np.pi, np.pi))
        behavior_actions = _sample_behavior_actions(
            rng,
            config,
            size=config.episode_length,
        )
        for step in range(1, config.episode_length):
            theta[step] = _wrap_angle(theta[step - 1] + behavior_actions[step - 1])

        observations[cursor : cursor + config.episode_length, 0] = (
            config.radius * np.cos(theta)
        ).astype(np.float32, copy=False)
        observations[cursor : cursor + config.episode_length, 1] = (
            config.radius * np.sin(theta)
        ).astype(np.float32, copy=False)
        actions[cursor : cursor + config.episode_length, 0] = behavior_actions.astype(
            np.float32,
            copy=False,
        )
        cursor += config.episode_length

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("observation", data=observations)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("policy_action", data=policy_actions)
        handle.create_dataset("reward", data=rewards)
        handle.create_dataset("discount", data=discounts)
        handle.create_dataset("ep_len", data=ep_len)
        handle.create_dataset("ep_offset", data=ep_offset)

    metadata = {
        **asdict(config),
        "output_hdf5_path": str(output_path.resolve()),
        "dataset_type": "toy_circle_exploration",
        "behavior_policy": _behavior_policy_metadata(config),
        "target_policy": {
            "kind": "constant_delta_theta",
            "delta_theta": float(config.policy_delta_theta),
        },
        "observation_shape": list(observations.shape[1:]),
        "action_shape": list(actions.shape[1:]),
        "total_steps": int(total_steps),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        GenerateToyCircleExplorationDatasetConfig,
        description="Generate a toy-circle exploration dataset with random behavior actions and relabeled policy actions.",
    )
    output_path = generate_toy_circle_exploration_dataset(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
