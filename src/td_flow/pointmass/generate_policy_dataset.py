from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro

from .loop_policy import PointMassLoopPolicyConfig, scripted_pointmass_loop_action


DEFAULT_DNC_ROOT = "/home/haizhou/Documents/DnC-FBr"


@dataclass
class GeneratePointMassPolicyDatasetConfig:
    output_hdf5_path: str
    dnc_root: str = DEFAULT_DNC_ROOT
    num_episodes: int = 2000
    episode_length: int = 1001
    seed: int = 0
    overwrite: bool = False
    include_physics: bool = True
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _load_pointmass_module(dnc_root: str):
    if dnc_root not in sys.path:
        sys.path.append(dnc_root)
    from metamotivo.envs.dmc_tasks import pointmass

    return pointmass


def generate_pointmass_policy_dataset(config: GeneratePointMassPolicyDatasetConfig) -> Path:
    output_path = Path(config.output_hdf5_path)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pointmass = _load_pointmass_module(config.dnc_root)
    total_steps = config.num_episodes * config.episode_length

    observation_buffer = np.zeros((total_steps, 4), dtype=np.float32)
    action_buffer = np.zeros((total_steps, 2), dtype=np.float32)
    reward_buffer = np.zeros((total_steps, 1), dtype=np.float32)
    discount_buffer = np.ones((total_steps, 1), dtype=np.float32)
    physics_buffer = np.zeros((total_steps, 4), dtype=np.float32) if config.include_physics else None
    ep_len = np.full(config.num_episodes, config.episode_length, dtype=np.int64)
    ep_offset = np.arange(config.num_episodes, dtype=np.int64) * config.episode_length

    cursor = 0
    returns: list[float] = []
    for episode_index in range(config.num_episodes):
        env = pointmass.loop(
            random=config.seed + episode_index,
            environment_kwargs=dict(flat_observation=True),
        )
        time_step = env.reset()
        episode_return = 0.0

        for step_index in range(config.episode_length):
            observation = np.asarray(time_step.observation["observations"], dtype=np.float32)
            action = scripted_pointmass_loop_action(observation, config=config.policy)

            observation_buffer[cursor] = observation
            action_buffer[cursor] = action
            reward_buffer[cursor, 0] = 0.0 if time_step.reward is None else float(time_step.reward)
            discount_buffer[cursor, 0] = 1.0 if time_step.discount is None else float(time_step.discount)
            if physics_buffer is not None:
                physics_buffer[cursor] = np.asarray(env.physics.get_state(), dtype=np.float32)
            cursor += 1

            if step_index + 1 < config.episode_length:
                time_step = env.step(action)
                episode_return += float(time_step.reward)

        returns.append(episode_return)

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("observation", data=observation_buffer)
        handle.create_dataset("action", data=action_buffer)
        handle.create_dataset("reward", data=reward_buffer)
        handle.create_dataset("discount", data=discount_buffer)
        if physics_buffer is not None:
            handle.create_dataset("physics", data=physics_buffer)
        handle.create_dataset("ep_len", data=ep_len)
        handle.create_dataset("ep_offset", data=ep_offset)

    metadata = {
        "output_hdf5_path": str(output_path.resolve()),
        "num_episodes": config.num_episodes,
        "episode_length": config.episode_length,
        "seed": config.seed,
        "include_physics": config.include_physics,
        "policy": asdict(config.policy),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        GeneratePointMassPolicyDatasetConfig,
        description="Generate a pointmass loop HDF5 dataset from scripted-policy rollouts.",
    )
    output_path = generate_pointmass_policy_dataset(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
