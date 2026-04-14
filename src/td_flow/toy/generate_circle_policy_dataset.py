from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro


@dataclass
class GenerateToyCirclePolicyDatasetConfig:
    output_hdf5_path: str
    num_episodes: int = 2048
    episode_length: int = 256
    radius: float = 1.0
    delta_theta: float = 0.02
    seed: int = 0
    overwrite: bool = False


def generate_toy_circle_policy_dataset(config: GenerateToyCirclePolicyDatasetConfig) -> Path:
    output_path = Path(config.output_hdf5_path)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = config.num_episodes * config.episode_length
    observations = np.zeros((total_steps, 2), dtype=np.float32)
    actions = np.zeros((total_steps, 1), dtype=np.float32)
    rewards = np.zeros((total_steps, 1), dtype=np.float32)
    discounts = np.ones((total_steps, 1), dtype=np.float32)
    ep_len = np.full(config.num_episodes, config.episode_length, dtype=np.int64)
    ep_offset = np.arange(config.num_episodes, dtype=np.int64) * config.episode_length

    rng = np.random.default_rng(config.seed)
    cursor = 0
    for episode_index in range(config.num_episodes):
        start_theta = float(rng.uniform(0.0, 2.0 * np.pi))
        theta = start_theta + config.delta_theta * np.arange(config.episode_length, dtype=np.float64)
        x = config.radius * np.cos(theta)
        y = config.radius * np.sin(theta)
        observations[cursor : cursor + config.episode_length, 0] = x.astype(np.float32, copy=False)
        observations[cursor : cursor + config.episode_length, 1] = y.astype(np.float32, copy=False)
        actions[cursor : cursor + config.episode_length, 0] = float(config.delta_theta)
        cursor += config.episode_length

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("observation", data=observations)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("reward", data=rewards)
        handle.create_dataset("discount", data=discounts)
        handle.create_dataset("ep_len", data=ep_len)
        handle.create_dataset("ep_offset", data=ep_offset)

    metadata = {
        **asdict(config),
        "output_hdf5_path": str(output_path.resolve()),
        "observation_shape": list(observations.shape[1:]),
        "action_shape": list(actions.shape[1:]),
        "total_steps": int(total_steps),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        GenerateToyCirclePolicyDatasetConfig,
        description="Generate a deterministic toy circle-policy HDF5 dataset.",
    )
    output_path = generate_toy_circle_policy_dataset(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
