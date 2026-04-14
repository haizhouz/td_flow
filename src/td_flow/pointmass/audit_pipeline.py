from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro

from ..data import build_td2_dataloader, build_td2_hdf5_dataset
from ..paths import sample_linear_probability_path, sample_source, sample_time
from .plot_policy_conditioned_occupancy import (
    DEFAULT_DNC_ROOT,
    _resolve_device,
    _resolve_hdf5_path,
    _rollout_policy_positions,
    _sample_bounded_relative_noise,
    _sample_discounted_positions,
    _sample_model_positions,
    _sample_random_indices,
    _sample_stochastic_rollout_positions,
    _set_env_state_from_observation,
)
from .loop_policy import (
    PointMassLoopPolicyConfig,
    TorchPointMassLoopScriptedPolicy,
    scripted_pointmass_loop_action,
)
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model


@dataclass
class AuditPointMassPipelineConfig:
    checkpoint_path: str
    dnc_root: str = DEFAULT_DNC_ROOT
    device: str = "auto"
    num_states: int = 5
    start_indices: tuple[int, ...] = ()
    sample_count: int = 2048
    sample_batch_size: int = 1024
    rollout_steps: int = 1000
    stochastic_rollouts: int = 64
    rollout_max_noise_fraction: float = 0.1
    continuity_rollouts: int = 8
    audit_batch_size: int = 256
    seed: int = 0
    output_path: str | None = None
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _default_output_path(checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(checkpoint_path) / "pointmass_pipeline_audit.json"


def _load_pointmass_module(dnc_root: str):
    if dnc_root not in sys.path:
        sys.path.append(dnc_root)
    from metamotivo.envs.dmc_tasks import pointmass

    return pointmass


def _mean_min_distance(samples_xy: np.ndarray, support_xy: np.ndarray) -> float:
    diffs = samples_xy[:, None, :] - support_xy[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.min(axis=1).mean())


def _symmetric_chamfer(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    return _mean_min_distance(a_xy, b_xy) + _mean_min_distance(b_xy, a_xy)


def _continuity_metrics(
    env,
    observation: np.ndarray,
    physics_state: np.ndarray | None,
    *,
    policy_config: PointMassLoopPolicyConfig,
    rollout_steps: int,
    rollout_count: int,
    max_noise_fraction: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    all_step_deltas: list[np.ndarray] = []
    first_step_jumps: list[float] = []
    first_step_types: list[int] = []

    for _ in range(rollout_count):
        _set_env_state_from_observation(env, observation, physics_state)
        current_obs = np.asarray(observation, dtype=np.float32)
        positions = [current_obs[:2].copy()]
        for step_index in range(rollout_steps):
            action = scripted_pointmass_loop_action(current_obs, config=policy_config)
            action = np.clip(
                action + _sample_bounded_relative_noise(
                    action,
                    max_noise_fraction=max_noise_fraction,
                    rng=rng,
                ),
                -1.0,
                1.0,
            )
            time_step = env.step(action)
            if step_index == 0:
                first_step_types.append(int(time_step.step_type))
            current_obs = np.asarray(time_step.observation["observations"], dtype=np.float32)
            positions.append(current_obs[:2].copy())

        trajectory_xy = np.stack(positions, axis=0)
        deltas = np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=-1)
        all_step_deltas.append(deltas)
        first_step_jumps.append(float(deltas[0]))

    flat_deltas = np.concatenate(all_step_deltas, axis=0)
    return {
        "mean_step_delta": float(flat_deltas.mean()),
        "p99_step_delta": float(np.quantile(flat_deltas, 0.99)),
        "max_step_delta": float(flat_deltas.max()),
        "mean_first_step_jump": float(np.mean(first_step_jumps)),
        "max_first_step_jump": float(np.max(first_step_jumps)),
        "first_step_types": first_step_types,
        "had_first_step_reset": bool(any(step_type == 0 for step_type in first_step_types)),
    }


def _dataset_alignment_metrics(
    dataset,
    raw_samples: dict[int, dict[str, np.ndarray]],
    *,
    raw_to_clip_index: dict[int, int],
) -> dict[str, object]:
    obs_errors: list[float] = []
    next_obs_errors: list[float] = []
    action_errors: list[float] = []
    next_action_errors: list[float] = []

    for raw_index, clip_index in raw_to_clip_index.items():
        sample = dataset[clip_index]
        raw = raw_samples[raw_index]
        obs_errors.append(float(torch.max(torch.abs(sample["obs"] - torch.as_tensor(raw["obs"]))).item()))
        next_obs_errors.append(float(torch.max(torch.abs(sample["next_obs"] - torch.as_tensor(raw["next_obs"]))).item()))
        action_errors.append(float(torch.max(torch.abs(sample["action"] - torch.as_tensor(raw["action"]))).item()))
        if "next_action" in raw:
            next_action_errors.append(
                float(torch.max(torch.abs(sample["next_action"] - torch.as_tensor(raw["next_action"]))).item())
            )

    return {
        "checked_indices": [{"raw_index": int(raw_index), "clip_index": int(clip_index)} for raw_index, clip_index in raw_to_clip_index.items()],
        "max_obs_error": max(obs_errors, default=0.0),
        "max_next_obs_error": max(next_obs_errors, default=0.0),
        "max_action_error": max(action_errors, default=0.0),
        "max_next_action_error": max(next_action_errors, default=0.0),
    }


def _build_raw_to_clip_index(
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    *,
    raw_indices: list[int],
    span: int,
) -> dict[int, int]:
    clip_counts = np.maximum(ep_len.astype(np.int64) - int(span) + 1, 0)
    clip_prefix = np.cumsum(clip_counts, dtype=np.int64)
    mapping: dict[int, int] = {}
    for raw_index in raw_indices:
        episode_index = int(np.searchsorted(ep_offset, raw_index, side="right") - 1)
        local_index = int(raw_index - int(ep_offset[episode_index]))
        if local_index < 0 or local_index + span > int(ep_len[episode_index]):
            raise ValueError(f"Raw index {raw_index} is not a valid clip start for span={span}.")
        previous_total = 0 if episode_index == 0 else int(clip_prefix[episode_index - 1])
        mapping[int(raw_index)] = previous_total + local_index
    return mapping


@torch.no_grad()
def _td2_target_sensitivity_metrics(model, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    obs = batch["obs"].to(model.device).float()
    action = batch["action"].to(model.device).float()
    next_obs = batch["next_obs"].to(model.device).float()
    next_action = batch["next_action"].to(model.device).float()

    state_latent = model.encode_observation(obs)
    next_latent = model.encode_observation(next_obs, use_target=True).detach()
    t = sample_time(obs.shape[0], device=model.device, dtype=state_latent.dtype, eps=model.cfg.time_eps)
    source = sample_source(obs.shape[0], model.latent_dim, device=model.device, dtype=state_latent.dtype)

    perm = torch.randperm(obs.shape[0], device=model.device)
    zero_action = torch.zeros_like(action)
    zero_next_action = torch.zeros_like(next_action)

    direct_xt, direct_target = sample_linear_probability_path(source, next_latent, t, eps=model.cfg.time_eps)
    _, direct_target_shuffled = sample_linear_probability_path(source, next_latent[perm], t, eps=model.cfg.time_eps)

    bootstrap_xt, bootstrap_target = model.bootstrap_target(next_latent, next_action, source, t)
    bootstrap_xt_zero_action, bootstrap_target_zero_action = model.bootstrap_target(next_latent, zero_next_action, source, t)
    bootstrap_xt_shuffled_action, bootstrap_target_shuffled_action = model.bootstrap_target(
        next_latent,
        next_action[perm],
        source,
        t,
    )
    bootstrap_xt_shuffled_state, bootstrap_target_shuffled_state = model.bootstrap_target(
        next_latent[perm],
        next_action,
        source,
        t,
    )

    online_velocity = model.compute_velocity(bootstrap_xt, t, state_latent, action, use_target=False)
    online_velocity_zero_action = model.compute_velocity(bootstrap_xt, t, state_latent, zero_action, use_target=False)
    online_velocity_shuffled_action = model.compute_velocity(bootstrap_xt, t, state_latent, action[perm], use_target=False)

    endpoint = model.predict_next_latent(state_latent, action, source=source)
    endpoint_zero_action = model.predict_next_latent(state_latent, zero_action, source=source)
    endpoint_shuffled_action = model.predict_next_latent(state_latent, action[perm], source=source)

    return {
        "direct_target_shuffle_next_obs_mean_l2": float((direct_target - direct_target_shuffled).norm(dim=-1).mean().item()),
        "bootstrap_xt_zero_next_action_mean_l2": float((bootstrap_xt - bootstrap_xt_zero_action).norm(dim=-1).mean().item()),
        "bootstrap_target_zero_next_action_mean_l2": float((bootstrap_target - bootstrap_target_zero_action).norm(dim=-1).mean().item()),
        "bootstrap_xt_shuffle_next_action_mean_l2": float((bootstrap_xt - bootstrap_xt_shuffled_action).norm(dim=-1).mean().item()),
        "bootstrap_target_shuffle_next_action_mean_l2": float((bootstrap_target - bootstrap_target_shuffled_action).norm(dim=-1).mean().item()),
        "bootstrap_xt_shuffle_next_obs_mean_l2": float((bootstrap_xt - bootstrap_xt_shuffled_state).norm(dim=-1).mean().item()),
        "bootstrap_target_shuffle_next_obs_mean_l2": float((bootstrap_target - bootstrap_target_shuffled_state).norm(dim=-1).mean().item()),
        "online_velocity_zero_current_action_mean_l2": float((online_velocity - online_velocity_zero_action).norm(dim=-1).mean().item()),
        "online_velocity_shuffle_current_action_mean_l2": float((online_velocity - online_velocity_shuffled_action).norm(dim=-1).mean().item()),
        "endpoint_zero_current_action_mean_l2": float((endpoint - endpoint_zero_action).norm(dim=-1).mean().item()),
        "endpoint_shuffle_current_action_mean_l2": float((endpoint - endpoint_shuffled_action).norm(dim=-1).mean().item()),
    }


def audit_pointmass_pipeline(config: AuditPointMassPipelineConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This audit currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (4,):
        raise NotImplementedError("This audit currently supports only 4D pointmass observations.")

    device = _resolve_device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)
    policy = TorchPointMassLoopScriptedPolicy(config.policy).to(device)
    dataset_path = _resolve_hdf5_path(run_dir)
    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pointmass = _load_pointmass_module(config.dnc_root)
    env = pointmass.loop(random=config.seed, environment_kwargs=dict(flat_observation=True))

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[project_config.data.observation_key]
        physics_states = handle["physics"] if "physics" in handle else None
        actions = handle[project_config.data.action_key]
        next_actions = None if project_config.data.next_action_key is None else handle[project_config.data.next_action_key]
        ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"], dtype=np.int64)

        if config.start_indices:
            start_indices = np.asarray(config.start_indices, dtype=np.int64)
        else:
            start_indices = _sample_random_indices(len(observations), config.num_states, config.seed)

        occupancy_states: list[dict[str, object]] = []
        continuity_states: list[dict[str, object]] = []
        raw_alignment_samples: dict[int, dict[str, np.ndarray]] = {}
        for column, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            physics_state = None if physics_states is None else np.asarray(physics_states[start_index], dtype=np.float64)
            raw_alignment_samples[int(start_index)] = {
                "obs": np.asarray(observations[start_index], dtype=np.float32),
                "next_obs": np.asarray(observations[start_index + 1], dtype=np.float32),
                "action": np.asarray(actions[start_index], dtype=np.float32),
            }
            if next_actions is not None:
                raw_alignment_samples[int(start_index)]["next_action"] = np.asarray(next_actions[start_index + 1], dtype=np.float32)
            model_positions, action = _sample_model_positions(
                model,
                policy,
                observation,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            deterministic_trajectory = _rollout_policy_positions(
                env,
                observation,
                physics_state,
                rollout_steps=config.rollout_steps,
                policy_config=config.policy,
            )
            deterministic_positions = _sample_discounted_positions(
                deterministic_trajectory,
                gamma=float(project_config.model.gamma),
                sample_count=config.sample_count,
                seed=config.seed + column,
            )
            stochastic_positions = _sample_stochastic_rollout_positions(
                env,
                observation,
                physics_state,
                rollout_steps=config.rollout_steps,
                rollout_count=config.stochastic_rollouts,
                policy_config=config.policy,
                max_noise_fraction=config.rollout_max_noise_fraction,
                gamma=float(project_config.model.gamma),
                sample_count=config.sample_count,
                seed=config.seed + column,
            )

            occupancy_states.append(
                {
                    "start_index": int(start_index),
                    "observation": observation.tolist(),
                    "policy_action": action.tolist(),
                    "mean_model_distance_from_start": float(np.linalg.norm(model_positions - observation[:2], axis=-1).mean()),
                    "mean_deterministic_rollout_distance_from_start": float(
                        np.linalg.norm(deterministic_positions - observation[:2], axis=-1).mean()
                    ),
                    "mean_stochastic_rollout_distance_from_start": float(
                        np.linalg.norm(stochastic_positions - observation[:2], axis=-1).mean()
                    ),
                    "model_to_deterministic_min_distance": _mean_min_distance(model_positions, deterministic_positions),
                    "deterministic_to_model_min_distance": _mean_min_distance(deterministic_positions, model_positions),
                    "model_to_stochastic_min_distance": _mean_min_distance(model_positions, stochastic_positions),
                    "stochastic_to_model_min_distance": _mean_min_distance(stochastic_positions, model_positions),
                    "deterministic_symmetric_chamfer": _symmetric_chamfer(model_positions, deterministic_positions),
                    "stochastic_symmetric_chamfer": _symmetric_chamfer(model_positions, stochastic_positions),
                }
            )

            continuity_states.append(
                {
                    "start_index": int(start_index),
                    **_continuity_metrics(
                        env,
                        observation,
                        physics_state,
                        policy_config=config.policy,
                        rollout_steps=config.rollout_steps,
                        rollout_count=config.continuity_rollouts,
                        max_noise_fraction=config.rollout_max_noise_fraction,
                        seed=config.seed + column,
                    ),
                }
            )

    td2_dataset = build_td2_hdf5_dataset(replace(project_config.data, num_workers=0))
    alignment_indices = [int(index) for index in start_indices.tolist()]
    raw_to_clip_index = _build_raw_to_clip_index(
        ep_offset,
        ep_len,
        raw_indices=alignment_indices,
        span=2,
    )
    dataset_alignment = _dataset_alignment_metrics(
        td2_dataset,
        raw_alignment_samples,
        raw_to_clip_index=raw_to_clip_index,
    )

    dataloader = build_td2_dataloader(
        replace(project_config.data, batch_size=config.audit_batch_size, num_workers=0),
        shuffle=False,
    )
    batch = next(iter(dataloader))
    td2_target_sensitivity = _td2_target_sensitivity_metrics(model, batch)

    summary = {
        "rollout_continuity_ok": not any(state["had_first_step_reset"] for state in continuity_states)
        and max(float(state["max_first_step_jump"]) for state in continuity_states) < 0.01,
        "dataset_alignment_ok": dataset_alignment["max_obs_error"] < 1e-6
        and dataset_alignment["max_next_obs_error"] < 1e-6
        and dataset_alignment["max_action_error"] < 1e-6
        and dataset_alignment["max_next_action_error"] < 1e-6,
        "bootstrap_depends_on_next_action": td2_target_sensitivity["bootstrap_target_shuffle_next_action_mean_l2"] > 1e-4
        or td2_target_sensitivity["bootstrap_target_zero_next_action_mean_l2"] > 1e-4,
        "bootstrap_depends_on_next_obs": td2_target_sensitivity["bootstrap_target_shuffle_next_obs_mean_l2"] > 1e-4,
        "online_endpoint_depends_on_current_action": td2_target_sensitivity["endpoint_shuffle_current_action_mean_l2"] > 1e-4
        or td2_target_sensitivity["endpoint_zero_current_action_mean_l2"] > 1e-4,
    }

    result = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "device": str(device),
        "config": asdict(config),
        "summary": summary,
        "occupancy_metrics": occupancy_states,
        "rollout_continuity": continuity_states,
        "dataset_alignment": dataset_alignment,
        "td2_target_sensitivity": td2_target_sensitivity,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        env.close()
    except Exception:
        pass
    return output_path


def main() -> None:
    config = tyro.cli(
        AuditPointMassPipelineConfig,
        description="Run a repeatable pointmass TD2 pipeline audit.",
    )
    output_path = audit_pointmass_pipeline(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
