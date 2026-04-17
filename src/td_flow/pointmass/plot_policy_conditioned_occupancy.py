from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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

from .loop_policy import (
    PointMassLoopPolicyConfig,
    TorchPointMassLoopScriptedPolicy,
)
from .relabel_policy_actions import (
    _sample_bounded_relative_noise as _sample_bounded_relative_noise_torch,
)
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


MAZE_LIMIT = 0.3
MAZE_ARM_HALF_LENGTH = 0.18
MAZE_WALL_HALF_WIDTH = 0.02
BACKGROUND_COLOR = "#4e77aa"
WALL_COLOR = "#dce5ee"
SAMPLE_COLOR = "#f0d46d"
START_COLOR = "#fff06a"
DEFAULT_DNC_ROOT = "/home/haizhou/Documents/DnC-FBr"


@dataclass
class PointMassPolicyConditionedOccupancyConfig:
    checkpoint_path: str
    dnc_root: str = DEFAULT_DNC_ROOT
    device: str = "auto"
    num_states: int = 5
    sample_count: int = 2048
    sample_batch_size: int = 1024
    rollout_steps: int = 1000
    stochastic_rollouts: int = 64
    rollout_max_noise_fraction: float = 0.1
    seed: int = 0
    output_path: str | None = None
    use_dataset_initial_action: bool = False
    baseline_mode: str = "auto"
    compile_policy: bool = False
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_path(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "pointmass_policy_conditioned_occupancy.png"


def _default_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _resolve_hdf5_path(run_dir: Path) -> Path:
    project_config = load_project_config_from_run_dir(run_dir)
    dataset_dir = Path(project_config.data.dir or ".")
    dataset_path = dataset_dir / f"{project_config.data.dataset_name}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")
    return dataset_path


def _sample_random_indices(dataset_size: int, count: int, seed: int) -> np.ndarray:
    if count > dataset_size:
        raise ValueError(f"Requested {count} states but dataset only has {dataset_size}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=count, replace=False).astype(np.int64))


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


def _plot_occupancy_cell(ax: plt.Axes, positions: np.ndarray, start_xy: np.ndarray) -> None:
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


def _load_pointmass_module(dnc_root: str):
    if dnc_root not in sys.path:
        sys.path.append(dnc_root)
    from metamotivo.envs.dmc_tasks import pointmass

    return pointmass


def _set_env_state_from_observation(
    env,
    observation: np.ndarray,
    physics_state: np.ndarray | None = None,
) -> None:
    obs = np.asarray(observation, dtype=np.float32)
    env.reset()
    if physics_state is not None:
        with env.physics.reset_context():
            env.physics.set_state(np.asarray(physics_state, dtype=np.float64))
        return

    with env.physics.reset_context():
        env.physics.data.qpos[:2] = obs[:2]
        env.physics.data.qvel[:2] = obs[2:4]


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type != "mps" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def _sample_policy_action_tensor(
    policy: TorchPointMassLoopScriptedPolicy,
    observation: np.ndarray,
    *,
    device: torch.device,
    max_noise_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    observation_tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    action_tensor = policy(observation_tensor)
    if max_noise_fraction > 0.0:
        action_tensor = action_tensor + _sample_bounded_relative_noise_torch(
            action_chunk=action_tensor,
            max_noise_fraction=max_noise_fraction,
            generator=generator,
        )
    return torch.clamp(action_tensor, -1.0, 1.0)


def _sample_stochastic_rollout_positions(
    env,
    observation: np.ndarray,
    physics_state: np.ndarray | None = None,
    *,
    rollout_steps: int,
    rollout_count: int,
    policy: TorchPointMassLoopScriptedPolicy,
    device: torch.device,
    initial_action: np.ndarray,
    max_noise_fraction: float,
    gamma: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    torch_generator = _make_torch_generator(device, seed)
    positions_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []

    for _ in range(rollout_count):
        _set_env_state_from_observation(env, observation, physics_state)
        current_obs = np.asarray(observation, dtype=np.float32)
        noisy_positions: list[np.ndarray] = [current_obs[:2].copy()]

        if rollout_steps > 0:
            time_step = env.step(np.asarray(initial_action, dtype=np.float32))
            current_obs = np.asarray(time_step.observation["observations"], dtype=np.float32)
            noisy_positions.append(current_obs[:2].copy())

        for _ in range(max(rollout_steps - 1, 0)):
            action_tensor = _sample_policy_action_tensor(
                policy,
                current_obs,
                device=device,
                max_noise_fraction=max_noise_fraction,
                generator=torch_generator,
            )
            action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            time_step = env.step(action)
            current_obs = np.asarray(time_step.observation["observations"], dtype=np.float32)
            noisy_positions.append(current_obs[:2].copy())

        trajectory_positions = np.stack(noisy_positions, axis=0)

        positions_chunks.append(trajectory_positions)
        weight_chunks.append(np.power(gamma, np.arange(len(trajectory_positions), dtype=np.float64)))

    all_positions = np.concatenate(positions_chunks, axis=0)
    all_weights = np.concatenate(weight_chunks, axis=0)
    probabilities = all_weights / all_weights.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(all_positions), size=sample_count, replace=True, p=probabilities)
    return all_positions[sampled_indices]


def _sample_discounted_positions(
    trajectory_positions: np.ndarray,
    *,
    gamma: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    discounts = np.power(gamma, np.arange(len(trajectory_positions), dtype=np.float64))
    probabilities = discounts / discounts.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(trajectory_positions), size=sample_count, replace=True, p=probabilities)
    return trajectory_positions[sampled_indices]


def _episode_index_for_step(ep_offset: np.ndarray, step_index: int) -> int:
    return int(np.searchsorted(ep_offset, step_index, side="right") - 1)


def _dataset_future_positions(
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


def _resolve_baseline_mode(project_config, requested_mode: str) -> str:
    if requested_mode == "auto":
        if project_config.data.next_action_key == project_config.data.action_key:
            return "dataset_episode"
        return "scripted_policy"
    if requested_mode not in {"scripted_policy", "dataset_episode"}:
        raise ValueError("baseline_mode must be one of: auto, scripted_policy, dataset_episode")
    return requested_mode


@torch.no_grad()
def _sample_model_positions(
    model,
    observation: np.ndarray,
    action_tensor: torch.Tensor,
    *,
    device: torch.device,
    sample_count: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_tensor = torch.from_numpy(observation).to(device=device, dtype=torch.float32).unsqueeze(0)
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
    return (
        np.concatenate(chunks, axis=0),
        action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _dataset_action_tensor(
    action_dataset,
    start_index: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    action = np.asarray(action_dataset[start_index], dtype=np.float32)
    return torch.from_numpy(action).to(device=device, dtype=torch.float32).unsqueeze(0)


def plot_pointmass_policy_conditioned_occupancy(config: PointMassPolicyConditionedOccupancyConfig) -> Path:
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
        ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"], dtype=np.int64)
        physics_states = handle["physics"] if "physics" in handle else None
        start_indices = _sample_random_indices(len(observations), config.num_states, config.seed)
        baseline_mode = _resolve_baseline_mode(project_config, config.baseline_mode)

        fig, axes = plt.subplots(
            2,
            config.num_states,
            figsize=(3.2 * config.num_states, 6.8),
            squeeze=False,
        )

        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "gamma": gamma,
            "conditioning": "dataset_action_then_dataset_episode" if baseline_mode == "dataset_episode" else (
                "dataset_initial_action_then_noisy_policy_rollout"
                if config.use_dataset_initial_action
                else "shared_noisy_initial_action_then_noisy_policy_rollout"
            ),
            "baseline_mode": baseline_mode,
            "num_states": config.num_states,
            "sample_count": config.sample_count,
            "sample_batch_size": config.sample_batch_size,
            "rollout_steps": config.rollout_steps,
            "stochastic_rollouts": config.stochastic_rollouts,
            "rollout_max_noise_fraction": config.rollout_max_noise_fraction,
            "seed": config.seed,
            "device": str(device),
            "compile_policy": config.compile_policy,
            "policy": asdict(config.policy),
            "start_indices": start_indices.tolist(),
            "states": [],
        }

        for column, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            physics_state = None if physics_states is None else np.asarray(physics_states[start_index], dtype=np.float64)
            if baseline_mode == "dataset_episode" or config.use_dataset_initial_action:
                conditioning_action_tensor = _dataset_action_tensor(
                    actions,
                    start_index,
                    device=device,
                )
            else:
                conditioning_action_tensor = _sample_policy_action_tensor(
                    policy,
                    observation,
                    device=device,
                    max_noise_fraction=config.rollout_max_noise_fraction,
                    generator=_make_torch_generator(device, config.seed + column),
                )
            model_positions, action = _sample_model_positions(
                model,
                observation,
                conditioning_action_tensor,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            if baseline_mode == "dataset_episode":
                trajectory_positions = _dataset_future_positions(
                    observations,
                    ep_offset,
                    ep_len,
                    start_index,
                )
                rollout_positions = _sample_discounted_positions(
                    trajectory_positions,
                    gamma=gamma,
                    sample_count=config.sample_count,
                    seed=config.seed + column,
                )
            else:
                rollout_positions = _sample_stochastic_rollout_positions(
                    env,
                    observation,
                    physics_state,
                    rollout_steps=config.rollout_steps,
                    rollout_count=config.stochastic_rollouts,
                    policy=policy,
                    device=device,
                    initial_action=action,
                    max_noise_fraction=config.rollout_max_noise_fraction,
                    gamma=gamma,
                    sample_count=config.sample_count,
                    seed=config.seed + column,
                )

            _plot_occupancy_cell(axes[0][column], model_positions, observation[:2])
            axes[0][column].set_title(
                (
                    f"state {column + 1}\n"
                    f"xy=({observation[0]:.2f}, {observation[1]:.2f})\n"
                    f"a=({action[0]:.2f}, {action[1]:.2f})"
                ),
                fontsize=10,
            )
            _plot_occupancy_cell(axes[1][column], rollout_positions, observation[:2])

            metadata["states"].append(
                {
                    "start_index": int(start_index),
                    "observation": observation.tolist(),
                    "conditioning_action": action.tolist(),
                }
            )

        axes[0][0].set_ylabel("TD-Flow\npolicy-conditioned", fontsize=11)
        axes[1][0].set_ylabel(
            "Logged dataset\nepisode continuation"
            if baseline_mode == "dataset_episode"
            else (
                "Logged-a then noisy\npolicy rollout"
                if config.use_dataset_initial_action
                else "Shared-a noisy policy\nrollout"
            ),
            fontsize=11,
        )
        fig.suptitle(
            (
                f"PointMass Policy-Conditioned Samples vs Logged Dataset Continuation (gamma={gamma:.2f})"
                if baseline_mode == "dataset_episode"
                else (
                    f"PointMass Policy-Conditioned Samples vs Logged-a Then Noisy Rollout (gamma={gamma:.2f})"
                    if config.use_dataset_initial_action
                    else f"PointMass Policy-Conditioned Samples vs Shared-a Noisy Rollout (gamma={gamma:.2f})"
                )
            ),
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    metadata_path = _default_metadata_path(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    try:
        env.close()
    except Exception:
        pass
    return output_path


def main() -> None:
    config = tyro.cli(
        PointMassPolicyConditionedOccupancyConfig,
        description="Plot scripted-policy-conditioned pointmass occupancies from a trained TD-Flow checkpoint.",
    )
    output_path = plot_pointmass_policy_conditioned_occupancy(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
