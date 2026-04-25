from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
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
    td_jepa_root: str = "third_party/td_jepa"
    td3_checkpoint_model_path: str = (
        "outputs/pointmass-loop-td3-tdjepa-paper-compile-par16-5m-20260417-132244/checkpoint/model"
    )
    device: str = "auto"
    num_states: int = 5
    sample_count: int = 2048
    sample_batch_size: int = 1024
    rollout_steps: int = 1000
    stochastic_rollouts: int = 64
    rollout_num_workers: int = 1
    worker_torch_threads: int = 1
    rollout_max_noise_fraction: float = 0.1
    seed: int = 0
    output_path: str | None = None
    use_dataset_initial_action: bool = False
    baseline_mode: str = "auto"
    policy_source: str = "scripted"
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


def _sample_valid_start_indices(
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    valid_counts = np.maximum(ep_len.astype(np.int64) - 1, 0)
    total_valid = int(valid_counts.sum())
    if total_valid <= 0:
        raise ValueError("No valid start states found with at least one future observation.")
    if count > total_valid:
        raise ValueError(f"Requested {count} states but only {total_valid} valid start states are available.")

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


class _TD3RolloutPolicy(torch.nn.Module):
    def __init__(self, td3_model: torch.nn.Module) -> None:
        super().__init__()
        self.td3_model = td3_model

    @torch.no_grad()
    def forward(self, observation_tensor: torch.Tensor) -> torch.Tensor:
        action_tensor = self.td3_model.act(observation_tensor, mean=True)
        return torch.clamp(action_tensor, -1.0, 1.0)


def _load_rollout_policy(
    *,
    policy_source: str,
    policy_config: PointMassLoopPolicyConfig,
    td_jepa_root: str,
    td3_checkpoint_model_path: str,
    device: torch.device,
) -> torch.nn.Module:
    if policy_source == "scripted":
        return TorchPointMassLoopScriptedPolicy(policy_config).to(device)
    if policy_source == "td3":
        td_jepa_path = Path(td_jepa_root).resolve()
        if str(td_jepa_path) not in sys.path:
            sys.path.insert(0, str(td_jepa_path))
        from metamotivo.agents.td3.model import TD3Model

        td3_model = TD3Model.load(str(Path(td3_checkpoint_model_path).resolve()), device=device.type)
        td3_model.eval()
        return _TD3RolloutPolicy(td3_model).to(device)
    raise ValueError("policy_source must be one of: scripted, td3")


def _rollout_policy_label(policy_source: str) -> str:
    if policy_source == "scripted":
        return "scripted policy"
    if policy_source == "td3":
        return "TD3 policy"
    return policy_source


def _rollout_noise_label(max_noise_fraction: float) -> str:
    return "noisy" if max_noise_fraction > 0.0 else "deterministic"


@torch.no_grad()
def _sample_policy_action_tensor(
    policy: torch.nn.Module,
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
    policy: torch.nn.Module,
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
        noisy_positions: list[np.ndarray] = []

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

        if not noisy_positions:
            raise ValueError("rollout_steps must be positive to sample successor occupancy.")
        trajectory_positions = np.stack(noisy_positions, axis=0)

        positions_chunks.append(trajectory_positions)
        weight_chunks.append(np.power(gamma, np.arange(len(trajectory_positions), dtype=np.float64)))

    all_positions = np.concatenate(positions_chunks, axis=0)
    all_weights = np.concatenate(weight_chunks, axis=0)
    probabilities = all_weights / all_weights.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(all_positions), size=sample_count, replace=True, p=probabilities)
    return all_positions[sampled_indices]


def _sample_stochastic_rollout_positions_worker(args: dict[str, object]) -> np.ndarray:
    torch.set_num_threads(int(args["worker_torch_threads"]))
    device = _resolve_device(str(args["device"]))
    policy = _load_rollout_policy(
        policy_source=str(args["policy_source"]),
        policy_config=PointMassLoopPolicyConfig(**args["policy"]),  # type: ignore[arg-type]
        td_jepa_root=str(args["td_jepa_root"]),
        td3_checkpoint_model_path=str(args["td3_checkpoint_model_path"]),
        device=device,
    )
    if bool(args["compile_policy"]):
        policy = torch.compile(policy)

    pointmass = _load_pointmass_module(str(args["dnc_root"]))
    env = pointmass.loop(
        random=int(args["seed"]),
        environment_kwargs=dict(flat_observation=True),
    )
    env.reset()
    try:
        return _sample_stochastic_rollout_positions(
            env,
            np.asarray(args["observation"], dtype=np.float32),
            None if args["physics_state"] is None else np.asarray(args["physics_state"], dtype=np.float64),
            rollout_steps=int(args["rollout_steps"]),
            rollout_count=int(args["stochastic_rollouts"]),
            policy=policy,
            device=device,
            initial_action=np.asarray(args["initial_action"], dtype=np.float32),
            max_noise_fraction=float(args["rollout_max_noise_fraction"]),
            gamma=float(args["gamma"]),
            sample_count=int(args["sample_count"]),
            seed=int(args["seed"]),
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def _sample_discounted_positions(
    trajectory_positions: np.ndarray,
    *,
    gamma: float,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    if len(trajectory_positions) == 0:
        raise ValueError("trajectory_positions must contain at least one future state.")
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
    remaining = int(ep_len[episode_index]) - local_index - 1
    if remaining <= 0:
        raise ValueError(f"start_index={start_index} has no future observations available.")
    return np.asarray(observations[start_index + 1 : start_index + 1 + remaining, :2], dtype=np.float32)


def _resolve_baseline_mode(project_config, requested_mode: str) -> str:
    if requested_mode == "auto":
        if project_config.data.next_action_key is None or project_config.data.next_action_key == project_config.data.action_key:
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
    policy = _load_rollout_policy(
        policy_source=config.policy_source,
        policy_config=config.policy,
        td_jepa_root=config.td_jepa_root,
        td3_checkpoint_model_path=config.td3_checkpoint_model_path,
        device=device,
    )
    if config.compile_policy:
        policy = torch.compile(policy)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[project_config.data.observation_key]
        actions = handle[project_config.data.action_key]
        ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"], dtype=np.int64)
        physics_states = handle["physics"] if "physics" in handle else None
        start_indices = _sample_valid_start_indices(ep_offset, ep_len, config.num_states, config.seed)
        baseline_mode = _resolve_baseline_mode(project_config, config.baseline_mode)

        fig, axes = plt.subplots(
            2,
            config.num_states,
            figsize=(3.2 * config.num_states, 6.8),
            squeeze=False,
        )

        rollout_noise_label = _rollout_noise_label(config.rollout_max_noise_fraction)
        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "gamma": gamma,
            "conditioning": "dataset_action_then_dataset_episode" if baseline_mode == "dataset_episode" else (
                f"dataset_initial_action_then_{rollout_noise_label}_rollout_policy"
                if config.use_dataset_initial_action
                else f"shared_{rollout_noise_label}_policy_initial_action_then_{rollout_noise_label}_rollout_policy"
            ),
            "baseline_mode": baseline_mode,
            "num_states": config.num_states,
            "sample_count": config.sample_count,
            "sample_batch_size": config.sample_batch_size,
            "rollout_steps": config.rollout_steps,
            "stochastic_rollouts": config.stochastic_rollouts,
            "rollout_num_workers": config.rollout_num_workers,
            "worker_torch_threads": config.worker_torch_threads,
            "rollout_max_noise_fraction": config.rollout_max_noise_fraction,
            "seed": config.seed,
            "device": str(device),
            "compile_policy": config.compile_policy,
            "policy_source": config.policy_source,
            "td3_checkpoint_model_path": str(Path(config.td3_checkpoint_model_path).resolve()),
            "policy": asdict(config.policy),
            "start_indices": start_indices.tolist(),
            "states": [],
        }

        rollout_worker_count = (
            max(1, min(int(config.rollout_num_workers), int(config.num_states)))
            if baseline_mode != "dataset_episode"
            else 1
        )
        env = None
        if baseline_mode != "dataset_episode" and rollout_worker_count <= 1:
            pointmass = _load_pointmass_module(config.dnc_root)
            env = pointmass.loop(random=config.seed, environment_kwargs=dict(flat_observation=True))
            env.reset()

        state_records: list[dict[str, object]] = []
        rollout_worker_args: list[dict[str, object]] = []

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
            record: dict[str, object] = {
                "start_index": int(start_index),
                "observation": observation,
                "model_positions": model_positions,
                "conditioning_action": action,
                "rollout_positions": None,
            }
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
                record["rollout_positions"] = rollout_positions
            elif rollout_worker_count > 1:
                rollout_worker_args.append(
                    {
                        "dnc_root": config.dnc_root,
                        "td_jepa_root": config.td_jepa_root,
                        "td3_checkpoint_model_path": config.td3_checkpoint_model_path,
                        "device": str(device),
                        "policy_source": config.policy_source,
                        "compile_policy": config.compile_policy,
                        "policy": asdict(config.policy),
                        "observation": observation,
                        "physics_state": physics_state,
                        "initial_action": action,
                        "rollout_steps": config.rollout_steps,
                        "stochastic_rollouts": config.stochastic_rollouts,
                        "rollout_max_noise_fraction": config.rollout_max_noise_fraction,
                        "gamma": gamma,
                        "sample_count": config.sample_count,
                        "seed": config.seed + column,
                        "worker_torch_threads": config.worker_torch_threads,
                    }
                )
            else:
                assert env is not None
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
                record["rollout_positions"] = rollout_positions

            state_records.append(record)

        if rollout_worker_args:
            with ProcessPoolExecutor(max_workers=rollout_worker_count) as executor:
                for record, rollout_positions in zip(
                    state_records,
                    executor.map(_sample_stochastic_rollout_positions_worker, rollout_worker_args),
                ):
                    record["rollout_positions"] = rollout_positions

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        for column, record in enumerate(state_records):
            observation = np.asarray(record["observation"], dtype=np.float32)
            model_positions = np.asarray(record["model_positions"], dtype=np.float32)
            rollout_positions = np.asarray(record["rollout_positions"], dtype=np.float32)
            action = np.asarray(record["conditioning_action"], dtype=np.float32)

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
                    "start_index": int(record["start_index"]),
                    "observation": observation.tolist(),
                    "conditioning_action": action.tolist(),
                }
            )

        axes[0][0].set_ylabel("TD-Flow\npolicy-conditioned", fontsize=11)
        axes[1][0].set_ylabel(
            "Logged dataset\nepisode continuation"
            if baseline_mode == "dataset_episode"
            else (
                f"Logged-a then {rollout_noise_label}\npolicy rollout"
                if config.use_dataset_initial_action
                else f"Shared-a {rollout_noise_label} policy\nrollout"
            ),
            fontsize=11,
        )
        fig.suptitle(
            (
                f"PointMass Policy-Conditioned Samples vs Logged Dataset Continuation (gamma={gamma:.2f})"
                if baseline_mode == "dataset_episode"
                else (
                    f"PointMass Policy-Conditioned Samples vs Logged-a Then {rollout_noise_label.title()} {_rollout_policy_label(config.policy_source)} Rollout (gamma={gamma:.2f})"
                    if config.use_dataset_initial_action
                    else f"PointMass Policy-Conditioned Samples vs Shared-a {rollout_noise_label.title()} {_rollout_policy_label(config.policy_source)} Rollout (gamma={gamma:.2f})"
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
    return output_path


def main() -> None:
    config = tyro.cli(
        PointMassPolicyConditionedOccupancyConfig,
        description="Plot rollout-policy-conditioned pointmass occupancies from a trained TD-Flow checkpoint.",
    )
    output_path = plot_pointmass_policy_conditioned_occupancy(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
