from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from .loop_policy import PointMassLoopPolicyConfig, TorchPointMassLoopScriptedPolicy
from .plot_policy_conditioned_occupancy import (
    DEFAULT_DNC_ROOT,
    _draw_maze_background,
    _load_pointmass_module,
    _make_torch_generator,
    _resolve_device,
    _resolve_hdf5_path,
    _sample_discounted_positions,
    _sample_model_positions,
    _set_env_state_from_observation,
)
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


MODEL_COLOR = "#f0d46d"
ROLLOUT_COLOR = "#79c4ff"
TRAJECTORY_COLOR = "#111111"
START_COLOR = "#fff06a"
NEXT_COLOR = "#ffffff"


@dataclass
class CompareActionConditionedSuccessorsConfig:
    checkpoint_path: str
    start_index: int = 5_116_475
    dnc_root: str = DEFAULT_DNC_ROOT
    device: str = "auto"
    rollout_steps: int = 1000
    sample_count: int = 2048
    sample_batch_size: int = 1024
    seed: int = 0
    output_path: str | None = None
    compile_policy: bool = False
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _default_output_path(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "pointmass_action_conditioned_successors.png"


def _default_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _plot_density(ax: plt.Axes, positions: np.ndarray, start_xy: np.ndarray, next_xy: np.ndarray, trajectory_xy: np.ndarray) -> None:
    _draw_maze_background(ax)
    density, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=72,
        range=[[-0.3, 0.3], [-0.3, 0.3]],
    )
    if density.max() > 0:
        normalized = (density.T / density.max()) ** 0.45
        overlay = np.zeros((72, 72, 4), dtype=np.float32)
        overlay[..., :3] = np.asarray(mcolors.to_rgb(MODEL_COLOR), dtype=np.float32)
        overlay[..., 3] = np.clip(normalized * 0.95, 0.0, 0.95)
        ax.imshow(
            overlay,
            extent=(-0.3, 0.3, -0.3, 0.3),
            origin="lower",
            interpolation="bilinear",
            zorder=2,
        )
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color=TRAJECTORY_COLOR, linewidth=1.1, alpha=0.95, zorder=3)
    ax.scatter([float(start_xy[0])], [float(start_xy[1])], s=44, c=START_COLOR, edgecolors="#2f2f2f", linewidths=0.6, zorder=4)
    ax.scatter([float(next_xy[0])], [float(next_xy[1])], s=26, c=NEXT_COLOR, edgecolors="#2f2f2f", linewidths=0.5, zorder=4)


def _plot_rollout(ax: plt.Axes, sampled_positions: np.ndarray, start_xy: np.ndarray, next_xy: np.ndarray, trajectory_xy: np.ndarray) -> None:
    _draw_maze_background(ax)
    ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], s=4, c=ROLLOUT_COLOR, alpha=0.12, linewidths=0, zorder=2)
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color=TRAJECTORY_COLOR, linewidth=1.1, alpha=0.95, zorder=3)
    ax.scatter([float(start_xy[0])], [float(start_xy[1])], s=44, c=START_COLOR, edgecolors="#2f2f2f", linewidths=0.6, zorder=4)
    ax.scatter([float(next_xy[0])], [float(next_xy[1])], s=26, c=NEXT_COLOR, edgecolors="#2f2f2f", linewidths=0.5, zorder=4)


def _mean_min_distance(samples_xy: np.ndarray, support_xy: np.ndarray) -> float:
    diffs = samples_xy[:, None, :] - support_xy[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.min(axis=1).mean())


@torch.no_grad()
def _policy_action(policy: TorchPointMassLoopScriptedPolicy, observation: np.ndarray, *, device: torch.device) -> np.ndarray:
    obs_tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    action = policy(obs_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return np.clip(action, -1.0, 1.0)


def _rotate_ccw(action: np.ndarray) -> np.ndarray:
    return np.clip(np.array([-action[1], action[0]], dtype=np.float32), -1.0, 1.0)


def _rotate_cw(action: np.ndarray) -> np.ndarray:
    return np.clip(np.array([action[1], -action[0]], dtype=np.float32), -1.0, 1.0)


def _deterministic_rollout_after_action(
    env,
    policy: TorchPointMassLoopScriptedPolicy,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    device: torch.device,
    rollout_steps: int,
    physics_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    _set_env_state_from_observation(env, observation, physics_state)
    first_step = env.step(np.asarray(action, dtype=np.float32))
    current_obs = np.asarray(first_step.observation["observations"], dtype=np.float32)
    positions: list[np.ndarray] = [current_obs[:2].copy()]
    next_xy = current_obs[:2].copy()

    for _ in range(max(int(rollout_steps) - 1, 0)):
        policy_action = _policy_action(policy, current_obs, device=device)
        step = env.step(policy_action)
        current_obs = np.asarray(step.observation["observations"], dtype=np.float32)
        positions.append(current_obs[:2].copy())
    return np.stack(positions, axis=0), next_xy


def _action_variants(dataset_action: np.ndarray, policy_action: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("dataset", dataset_action.astype(np.float32, copy=False))]
    named_candidates = [
        ("policy", policy_action),
        ("neg_policy", -policy_action),
        ("rot_ccw", _rotate_ccw(policy_action)),
        ("rot_cw", _rotate_cw(policy_action)),
    ]
    for label, action in named_candidates:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if any(np.allclose(action, existing_action, atol=1e-6) for _, existing_action in variants):
            continue
        variants.append((label, action))
    return variants


def compare_action_conditioned_successors(config: CompareActionConditionedSuccessorsConfig) -> Path:
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
        physics_states = handle["physics"] if "physics" in handle else None
        if not (0 <= config.start_index < len(observations)):
            raise ValueError(f"start_index={config.start_index} out of bounds for dataset of size {len(observations)}")

        observation = np.asarray(observations[config.start_index], dtype=np.float32)
        physics_state = None if physics_states is None else np.asarray(physics_states[config.start_index], dtype=np.float64)
        dataset_action = np.asarray(actions[config.start_index], dtype=np.float32)
        policy_action = _policy_action(policy, observation, device=device)
        action_variants = _action_variants(dataset_action, policy_action)

        fig, axes = plt.subplots(
            2,
            len(action_variants),
            figsize=(3.6 * len(action_variants), 6.6),
            squeeze=False,
        )
        model_position_by_label: dict[str, np.ndarray] = {}
        rollout_position_by_label: dict[str, np.ndarray] = {}

        metadata: dict[str, object] = {
            "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "start_index": int(config.start_index),
            "observation": observation.tolist(),
            "dataset_action": dataset_action.tolist(),
            "policy_action": policy_action.tolist(),
            "gamma": gamma,
            "rollout_steps": int(config.rollout_steps),
            "sample_count": int(config.sample_count),
            "sample_batch_size": int(config.sample_batch_size),
            "seed": int(config.seed),
            "device": str(device),
            "policy": asdict(config.policy),
            "actions": [],
        }

        for column, (label, action) in enumerate(action_variants):
            action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
            model_positions, used_action = _sample_model_positions(
                model,
                observation,
                action_tensor,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            trajectory_xy, next_xy = _deterministic_rollout_after_action(
                env,
                policy,
                observation,
                used_action,
                device=device,
                rollout_steps=config.rollout_steps,
                physics_state=physics_state,
            )
            rollout_positions = _sample_discounted_positions(
                trajectory_positions=trajectory_xy,
                gamma=gamma,
                sample_count=config.sample_count,
                seed=config.seed + column,
            )
            model_position_by_label[label] = model_positions
            rollout_position_by_label[label] = rollout_positions

            model_to_rollout = _mean_min_distance(model_positions, trajectory_xy)
            rollout_to_model = _mean_min_distance(trajectory_xy, model_positions)

            _plot_density(axes[0][column], model_positions, observation[:2], next_xy, trajectory_xy)
            axes[0][column].set_title(
                (
                    f"{label}\n"
                    f"a=({used_action[0]:.2f}, {used_action[1]:.2f})\n"
                    f"m->r={model_to_rollout:.3f} r->m={rollout_to_model:.3f}"
                ),
                fontsize=10,
            )
            _plot_rollout(axes[1][column], rollout_positions, observation[:2], next_xy, trajectory_xy)

            metadata["actions"].append(
                {
                    "label": label,
                    "action": used_action.tolist(),
                    "next_xy": next_xy.tolist(),
                    "model_to_rollout_distance": model_to_rollout,
                    "rollout_to_model_distance": rollout_to_model,
                    "trajectory_length": int(len(trajectory_xy)),
                }
            )

        pairwise: list[dict[str, object]] = []
        labels = [label for label, _ in action_variants]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left = labels[i]
                right = labels[j]
                model_left = model_position_by_label[left]
                model_right = model_position_by_label[right]
                rollout_left = rollout_position_by_label[left]
                rollout_right = rollout_position_by_label[right]
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "model_pair_distance": 0.5 * (
                            _mean_min_distance(model_left, model_right) +
                            _mean_min_distance(model_right, model_left)
                        ),
                        "rollout_pair_distance": 0.5 * (
                            _mean_min_distance(rollout_left, rollout_right) +
                            _mean_min_distance(rollout_right, rollout_left)
                        ),
                    }
                )
        metadata["pairwise"] = pairwise

        axes[0][0].set_ylabel("model", fontsize=11)
        axes[1][0].set_ylabel("take a once,\nthen follow pi", fontsize=11)
        fig.suptitle(
            (
                f"Fixed state index {config.start_index}\n"
                f"xy=({observation[0]:.3f}, {observation[1]:.3f}) "
                f"vel=({observation[2]:.3f}, {observation[3]:.3f})"
            ),
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    metadata_path = _default_metadata_path(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        CompareActionConditionedSuccessorsConfig,
        description=(
            "Fix one pointmass state, compare learned successor samples under several current actions, "
            "and compare each to deterministic rollouts that take the action once then follow the scripted policy."
        ),
    )
    output_path = compare_action_conditioned_successors(config)
    print(output_path)


if __name__ == "__main__":
    main()
