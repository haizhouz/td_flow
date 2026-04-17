from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from .analyze_td2_failure import _sample_target_endpoint_positions
from .compare_action_conditioned_successors import (
    MODEL_COLOR,
    NEXT_COLOR,
    ROLLOUT_COLOR,
    START_COLOR,
    TRAJECTORY_COLOR,
    _deterministic_rollout_after_action,
    _mean_min_distance,
    _plot_density,
    _plot_rollout,
    _policy_action,
)
from .loop_policy import PointMassLoopPolicyConfig, TorchPointMassLoopScriptedPolicy
from .plot_policy_conditioned_occupancy import (
    DEFAULT_DNC_ROOT,
    _load_pointmass_module,
    _resolve_device,
    _resolve_hdf5_path,
    _sample_discounted_positions,
    _sample_model_positions,
    _set_env_state_from_observation,
)
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


@dataclass
class CompareBootstrapTargetOccupancyConfig:
    checkpoint_path: str
    start_index: int = 5_116_475
    dnc_root: str = DEFAULT_DNC_ROOT
    device: str = "auto"
    rollout_steps: int = 1000
    sample_count: int = 2048
    sample_batch_size: int = 1024
    seed: int = 0
    output_path: str | None = None
    baseline_mode: str = "auto"
    compile_policy: bool = False
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _default_output_path(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "pointmass_bootstrap_target_occupancy.png"


def _default_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _rollout_from_next_state(
    env,
    policy: TorchPointMassLoopScriptedPolicy,
    next_observation: np.ndarray,
    *,
    device: torch.device,
    rollout_steps: int,
    physics_state: np.ndarray | None,
) -> np.ndarray:
    _set_env_state_from_observation(env, next_observation, physics_state)
    current_obs = np.asarray(next_observation, dtype=np.float32).copy()
    positions: list[np.ndarray] = [current_obs[:2].copy()]
    for _ in range(max(int(rollout_steps) - 1, 0)):
        action = _policy_action(policy, current_obs, device=device)
        step = env.step(action)
        current_obs = np.asarray(step.observation["observations"], dtype=np.float32)
        positions.append(current_obs[:2].copy())
    return np.stack(positions, axis=0)


def _resolve_baseline_mode(project_config, requested_mode: str) -> str:
    if requested_mode == "auto":
        if project_config.data.next_action_key == project_config.data.action_key:
            return "dataset_episode"
        return "scripted_policy"
    if requested_mode not in {"scripted_policy", "dataset_episode"}:
        raise ValueError("baseline_mode must be one of: auto, scripted_policy, dataset_episode")
    return requested_mode


def _episode_index_for_step(ep_offset: np.ndarray, step_index: int) -> int:
    return int(np.searchsorted(ep_offset, step_index, side="right") - 1)


def _dataset_trajectory_from_index(
    observations,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    start_index: int,
) -> np.ndarray:
    episode_index = _episode_index_for_step(ep_offset, start_index)
    episode_start = int(ep_offset[episode_index])
    local_index = int(start_index - episode_start)
    remaining = int(ep_len[episode_index]) - local_index
    return np.asarray(observations[start_index : start_index + remaining, :2], dtype=np.float32)


def compare_bootstrap_target_occupancy(config: CompareBootstrapTargetOccupancyConfig) -> Path:
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
    policy = TorchPointMassLoopScriptedPolicy(config.policy).to(device)
    if config.compile_policy:
        policy = torch.compile(policy)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pointmass = _load_pointmass_module(config.dnc_root)
    env = pointmass.loop(random=config.seed, environment_kwargs=dict(flat_observation=True))
    env.reset()

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[project_config.data.observation_key]
        actions = handle[project_config.data.action_key]
        next_action_key = project_config.data.next_action_key
        next_action_ds = handle[next_action_key] if next_action_key is not None else None
        ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"], dtype=np.int64)
        physics_states = handle["physics"] if "physics" in handle else None
        if not (0 <= config.start_index < len(observations) - 1):
            raise ValueError(f"start_index={config.start_index} out of bounds for dataset of size {len(observations)}")
        baseline_mode = _resolve_baseline_mode(project_config, config.baseline_mode)

        observation = np.asarray(observations[config.start_index], dtype=np.float32)
        next_observation = np.asarray(observations[config.start_index + 1], dtype=np.float32)
        physics_state = None if physics_states is None else np.asarray(physics_states[config.start_index], dtype=np.float64)
        next_physics_state = None if physics_states is None else np.asarray(physics_states[config.start_index + 1], dtype=np.float64)
        action = np.asarray(actions[config.start_index], dtype=np.float32)
        next_action = (
            np.asarray(next_action_ds[config.start_index + 1], dtype=np.float32)
            if next_action_ds is not None
            else _policy_action(policy, next_observation, device=device)
        )

        action_tensor = torch.from_numpy(action).to(device=device, dtype=torch.float32).unsqueeze(0)
        model_positions, used_action = _sample_model_positions(
            model,
            observation,
            action_tensor,
            device=device,
            sample_count=config.sample_count,
            batch_size=config.sample_batch_size,
        )
        bootstrap_positions = _sample_target_endpoint_positions(
            model,
            next_observation,
            next_action,
            device=device,
            sample_count=config.sample_count,
            batch_size=config.sample_batch_size,
        )

        if baseline_mode == "dataset_episode":
            full_trajectory_xy = _dataset_trajectory_from_index(
                observations,
                ep_offset,
                ep_len,
                config.start_index,
            )
            continuation_trajectory_xy = _dataset_trajectory_from_index(
                observations,
                ep_offset,
                ep_len,
                config.start_index + 1,
            )
            next_xy = next_observation[:2].copy()
        else:
            full_trajectory_xy, next_xy = _deterministic_rollout_after_action(
                env,
                policy,
                observation,
                used_action,
                device=device,
                rollout_steps=config.rollout_steps,
                physics_state=physics_state,
            )
            continuation_trajectory_xy = _rollout_from_next_state(
                env,
                policy,
                next_observation,
                device=device,
                rollout_steps=config.rollout_steps,
                physics_state=next_physics_state,
            )
        full_rollout_positions = _sample_discounted_positions(
            trajectory_positions=full_trajectory_xy,
            gamma=gamma,
            sample_count=config.sample_count,
            seed=config.seed,
        )
        continuation_rollout_positions = _sample_discounted_positions(
            trajectory_positions=continuation_trajectory_xy,
            gamma=gamma,
            sample_count=config.sample_count,
            seed=config.seed + 1,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.2), squeeze=False)
    _plot_density(axes[0][0], model_positions, observation[:2], next_xy, full_trajectory_xy)
    _plot_density(axes[0][1], bootstrap_positions, observation[:2], next_xy, continuation_trajectory_xy)
    _plot_rollout(axes[1][0], full_rollout_positions, observation[:2], next_xy, full_trajectory_xy)
    _plot_rollout(axes[1][1], continuation_rollout_positions, observation[:2], next_xy, continuation_trajectory_xy)

    model_to_full = _mean_min_distance(model_positions, full_trajectory_xy)
    full_to_model = _mean_min_distance(full_trajectory_xy, model_positions)
    bootstrap_to_full = _mean_min_distance(bootstrap_positions, full_trajectory_xy)
    bootstrap_to_cont = _mean_min_distance(bootstrap_positions, continuation_trajectory_xy)
    cont_to_bootstrap = _mean_min_distance(continuation_trajectory_xy, bootstrap_positions)

    axes[0][0].set_title(
        f"Online Model m(.|s,a)\nm->full={model_to_full:.3f} full->m={full_to_model:.3f}",
        fontsize=11,
    )
    axes[0][1].set_title(
        f"Bootstrap Target m_target(.|s',a')\nb->full={bootstrap_to_full:.3f} b->cont={bootstrap_to_cont:.3f} cont->b={cont_to_bootstrap:.3f}",
        fontsize=11,
    )
    axes[1][0].set_title(
        "Logged Dataset Rollout" if baseline_mode == "dataset_episode" else "True Rollout: take a once, then follow π",
        fontsize=11,
    )
    axes[1][1].set_title(
        "Logged Dataset Continuation" if baseline_mode == "dataset_episode" else "True Continuation: start at s', follow π",
        fontsize=11,
    )

    for ax in axes.reshape(-1):
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        f"start={config.start_index}  a=({used_action[0]:.2f}, {used_action[1]:.2f})  a'=({next_action[0]:.2f}, {next_action[1]:.2f})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    metadata = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "start_index": int(config.start_index),
        "observation": observation.tolist(),
        "next_observation": next_observation.tolist(),
        "action": used_action.tolist(),
        "next_action": next_action.tolist(),
        "gamma": gamma,
        "rollout_steps": int(config.rollout_steps),
        "sample_count": int(config.sample_count),
        "sample_batch_size": int(config.sample_batch_size),
        "seed": int(config.seed),
        "device": str(device),
        "policy": asdict(config.policy),
        "baseline_mode": baseline_mode,
        "metrics": {
            "model_to_full_distance": model_to_full,
            "full_to_model_distance": full_to_model,
            "bootstrap_to_full_distance": bootstrap_to_full,
            "bootstrap_to_continuation_distance": bootstrap_to_cont,
            "continuation_to_bootstrap_distance": cont_to_bootstrap,
        },
    }
    _default_metadata_path(output_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        CompareBootstrapTargetOccupancyConfig,
        description="Compare online model occupancy, bootstrap-target occupancy, and true rollout support for one fixed pointmass state-action pair.",
    )
    output_path = compare_bootstrap_target_occupancy(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
