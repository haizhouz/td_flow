from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro


DEFAULT_DATA_ROOT = "/home/haizhou/Documents/td_flow/data/stablewm_cache"


@dataclass
class MeasurePointMassTransitionSupportConfig:
    behavior_hdf5_path: str = f"{DEFAULT_DATA_ROOT}/pointmass-exorl-rnd-scripted-policy-relnoise10.h5"
    policy_hdf5_path: str = f"{DEFAULT_DATA_ROOT}/pointmass-loop-scripted-policy-only.h5"
    behavior_observation_key: str = "observation"
    behavior_action_key: str = "action"
    policy_observation_key: str = "observation"
    policy_action_key: str = "action"
    device: str = "auto"
    query_count: int = 256
    seed: int = 0
    dataset_chunk_size: int = 65536
    query_block_size: int = 32
    obs_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05)
    action_thresholds: tuple[float, ...] = (0.1, 0.2, 0.5)
    next_obs_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05)
    combined_thresholds: tuple[tuple[float, float, float], ...] = (
        (0.01, 0.1, 0.01),
        (0.02, 0.2, 0.02),
        (0.05, 0.5, 0.05),
    )
    full_distance_scales: tuple[float, float, float] = (0.05, 0.5, 0.05)
    output_path: str | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_output_path(config: MeasurePointMassTransitionSupportConfig) -> Path:
    behavior_path = Path(config.behavior_hdf5_path)
    return behavior_path.parent.parent / "outputs" / "pointmass_transition_support_diagnostic.json"


def _sample_query_indices(dataset_size: int, count: int, seed: int) -> np.ndarray:
    if count > dataset_size:
        raise ValueError(f"Requested {count} queries but dataset only has {dataset_size} valid transitions.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=count, replace=False).astype(np.int64))


def _summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def measure_transition_support(config: MeasurePointMassTransitionSupportConfig) -> Path:
    behavior_path = Path(config.behavior_hdf5_path)
    policy_path = Path(config.policy_hdf5_path)
    if not behavior_path.exists():
        raise FileNotFoundError(f"Behavior dataset not found: {behavior_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy dataset not found: {policy_path}")

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)

    with h5py.File(policy_path, "r") as policy_handle:
        policy_observations = policy_handle[config.policy_observation_key]
        policy_actions = policy_handle[config.policy_action_key]
        valid_policy_size = len(policy_observations) - 1
        query_indices = _sample_query_indices(valid_policy_size, config.query_count, config.seed)
        query_obs_np = np.asarray(policy_observations[query_indices], dtype=np.float32)
        query_action_np = np.asarray(policy_actions[query_indices], dtype=np.float32)
        query_next_obs_np = np.asarray(policy_observations[query_indices + 1], dtype=np.float32)

    query_obs = torch.from_numpy(query_obs_np).to(device=device, dtype=torch.float32)
    query_action = torch.from_numpy(query_action_np).to(device=device, dtype=torch.float32)
    query_next_obs = torch.from_numpy(query_next_obs_np).to(device=device, dtype=torch.float32)

    query_count = query_obs.shape[0]
    query_block_size = max(1, int(config.query_block_size))
    chunk_size = max(1, int(config.dataset_chunk_size))
    obs_scale, action_scale, next_obs_scale = [float(v) for v in config.full_distance_scales]

    best_obs = torch.full((query_count,), float("inf"), device=device)
    best_action = torch.full((query_count,), float("inf"), device=device)
    best_next_obs = torch.full((query_count,), float("inf"), device=device)
    best_full = torch.full((query_count,), float("inf"), device=device)
    best_state_action = torch.full((query_count,), float("inf"), device=device)
    best_state_next = torch.full((query_count,), float("inf"), device=device)

    threshold_triplets = list(config.combined_thresholds)
    threshold_hits = torch.zeros((len(threshold_triplets), query_count), dtype=torch.bool, device=device)

    with h5py.File(behavior_path, "r") as behavior_handle:
        behavior_observations = behavior_handle[config.behavior_observation_key]
        behavior_actions = behavior_handle[config.behavior_action_key]
        behavior_size = len(behavior_observations) - 1

        for start in range(0, behavior_size, chunk_size):
            stop = min(start + chunk_size, behavior_size)
            obs_chunk = torch.from_numpy(
                np.asarray(behavior_observations[start:stop], dtype=np.float32)
            ).to(device=device, dtype=torch.float32)
            action_chunk = torch.from_numpy(
                np.asarray(behavior_actions[start:stop], dtype=np.float32)
            ).to(device=device, dtype=torch.float32)
            next_obs_chunk = torch.from_numpy(
                np.asarray(behavior_observations[start + 1 : stop + 1], dtype=np.float32)
            ).to(device=device, dtype=torch.float32)

            for q_start in range(0, query_count, query_block_size):
                q_stop = min(q_start + query_block_size, query_count)
                query_obs_block = query_obs[q_start:q_stop]
                query_action_block = query_action[q_start:q_stop]
                query_next_obs_block = query_next_obs[q_start:q_stop]

                obs_distance = torch.cdist(query_obs_block, obs_chunk)
                action_distance = torch.cdist(query_action_block, action_chunk)
                next_obs_distance = torch.cdist(query_next_obs_block, next_obs_chunk)

                state_action_distance = torch.sqrt(
                    (obs_distance / obs_scale) ** 2 + (action_distance / action_scale) ** 2
                )
                state_next_distance = torch.sqrt(
                    (obs_distance / obs_scale) ** 2 + (next_obs_distance / next_obs_scale) ** 2
                )
                full_distance = torch.sqrt(
                    (obs_distance / obs_scale) ** 2
                    + (action_distance / action_scale) ** 2
                    + (next_obs_distance / next_obs_scale) ** 2
                )

                best_obs[q_start:q_stop] = torch.minimum(best_obs[q_start:q_stop], obs_distance.min(dim=1).values)
                best_action[q_start:q_stop] = torch.minimum(
                    best_action[q_start:q_stop], action_distance.min(dim=1).values
                )
                best_next_obs[q_start:q_stop] = torch.minimum(
                    best_next_obs[q_start:q_stop], next_obs_distance.min(dim=1).values
                )
                best_full[q_start:q_stop] = torch.minimum(best_full[q_start:q_stop], full_distance.min(dim=1).values)
                best_state_action[q_start:q_stop] = torch.minimum(
                    best_state_action[q_start:q_stop], state_action_distance.min(dim=1).values
                )
                best_state_next[q_start:q_stop] = torch.minimum(
                    best_state_next[q_start:q_stop], state_next_distance.min(dim=1).values
                )

                for threshold_index, (obs_thr, act_thr, next_thr) in enumerate(threshold_triplets):
                    hit = (
                        (obs_distance <= float(obs_thr))
                        & (action_distance <= float(act_thr))
                        & (next_obs_distance <= float(next_thr))
                    ).any(dim=1)
                    threshold_hits[threshold_index, q_start:q_stop] |= hit

    best_obs_np = best_obs.detach().cpu().numpy()
    best_action_np = best_action.detach().cpu().numpy()
    best_next_obs_np = best_next_obs.detach().cpu().numpy()
    best_full_np = best_full.detach().cpu().numpy()
    best_state_action_np = best_state_action.detach().cpu().numpy()
    best_state_next_np = best_state_next.detach().cpu().numpy()
    threshold_hits_np = threshold_hits.detach().cpu().numpy()

    metadata = {
        "behavior_hdf5_path": str(behavior_path.resolve()),
        "policy_hdf5_path": str(policy_path.resolve()),
        "device": str(device),
        "seed": int(config.seed),
        "query_count": int(config.query_count),
        "dataset_chunk_size": int(chunk_size),
        "query_block_size": int(query_block_size),
        "full_distance_scales": {
            "obs": obs_scale,
            "action": action_scale,
            "next_obs": next_obs_scale,
        },
        "query_index_summary": {
            "min": int(query_indices.min()),
            "max": int(query_indices.max()),
        },
        "nearest_distance_summary": {
            "obs": _summarize(best_obs_np),
            "action": _summarize(best_action_np),
            "next_obs": _summarize(best_next_obs_np),
            "state_action": _summarize(best_state_action_np),
            "state_next_obs": _summarize(best_state_next_np),
            "state_action_next_obs": _summarize(best_full_np),
        },
        "coverage": {
            "obs": {str(threshold): float(np.mean(best_obs_np <= threshold)) for threshold in config.obs_thresholds},
            "action": {
                str(threshold): float(np.mean(best_action_np <= threshold)) for threshold in config.action_thresholds
            },
            "next_obs": {
                str(threshold): float(np.mean(best_next_obs_np <= threshold)) for threshold in config.next_obs_thresholds
            },
            "state_action_next_obs": [
                {
                    "obs_threshold": float(obs_thr),
                    "action_threshold": float(act_thr),
                    "next_obs_threshold": float(next_thr),
                    "fraction": float(np.mean(threshold_hits_np[index])),
                }
                for index, (obs_thr, act_thr, next_thr) in enumerate(threshold_triplets)
            ],
        },
        "examples": [
            {
                "query_index": int(query_indices[index]),
                "obs": query_obs_np[index].tolist(),
                "action": query_action_np[index].tolist(),
                "next_obs": query_next_obs_np[index].tolist(),
                "best_obs_distance": float(best_obs_np[index]),
                "best_action_distance": float(best_action_np[index]),
                "best_next_obs_distance": float(best_next_obs_np[index]),
                "best_state_action_distance": float(best_state_action_np[index]),
                "best_state_next_obs_distance": float(best_state_next_np[index]),
                "best_state_action_next_obs_distance": float(best_full_np[index]),
            }
            for index in range(min(8, query_count))
        ],
    }

    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        MeasurePointMassTransitionSupportConfig,
        description="Measure one-step (s, a, s') support of a target policy rollout set inside the off-policy pointmass dataset.",
    )
    output_path = measure_transition_support(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
