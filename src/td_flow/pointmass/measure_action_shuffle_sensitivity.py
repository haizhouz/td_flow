from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro
from ..rollout import _checkpoint_run_dir, load_project_config_from_run_dir, load_td2_model
from .plot_policy_conditioned_occupancy import _resolve_device, _resolve_hdf5_path


@dataclass
class MeasureActionShuffleSensitivityConfig:
    checkpoint_path: str
    num_states: int = 64
    num_sources_per_state: int = 256
    sample_batch_size: int = 1024
    derangement_trials: int = 256
    seed: int = 0
    device: str = "auto"
    output_path: str | None = None


def _default_output_path(checkpoint_path: str) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.parent.parent / f"pointmass_action_shuffle_sensitivity_{checkpoint.stem}_fixed.json"


def _random_derangement(rng: np.random.Generator, length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("derangement requires length >= 2")
    while True:
        perm = rng.permutation(length)
        if not np.any(perm == np.arange(length)):
            return perm


def _best_action_derangement(
    actions: np.ndarray,
    *,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(actions.shape[0])
    best_perm: np.ndarray | None = None
    best_action_l2: np.ndarray | None = None
    best_score = -np.inf
    for _ in range(max(int(trials), 1)):
        perm = _random_derangement(rng, len(indices))
        deltas = actions - actions[perm]
        action_l2 = np.linalg.norm(deltas, axis=-1)
        score = float(action_l2.mean())
        if score > best_score:
            best_score = score
            best_perm = perm
            best_action_l2 = action_l2
    if best_perm is None or best_action_l2 is None:
        raise RuntimeError("failed to construct action derangement")
    return best_perm, best_action_l2


def _mean_min_distance(samples_xy: np.ndarray, support_xy: np.ndarray) -> float:
    diffs = samples_xy[:, None, :] - support_xy[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.min(axis=1).mean())


@torch.no_grad()
def _sample_endpoint_positions(
    model,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    device: torch.device,
    sample_count: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    state_latent = model.encode_observation(obs_tensor)
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(seed))

    chunks: list[np.ndarray] = []
    remaining = int(sample_count)
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        latent_batch = state_latent.expand(current_batch, -1)
        action_batch = action_tensor.expand(current_batch, -1)
        source = torch.randn(
            current_batch,
            model.latent_dim,
            generator=generator,
            device=device,
            dtype=latent_batch.dtype,
        )
        prediction = model.predict_next_latent(latent_batch, action_batch, source=source)
        chunks.append(prediction[:, :2].detach().cpu().numpy().astype(np.float32, copy=False))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _sample_endpoint_positions_matched(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    shuffled_actions: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_tensor = torch.from_numpy(np.asarray(observations, dtype=np.float32)).to(device=device, dtype=torch.float32)
    action_tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(device=device, dtype=torch.float32)
    shuffled_action_tensor = torch.from_numpy(np.asarray(shuffled_actions, dtype=np.float32)).to(device=device, dtype=torch.float32)
    state_latent = model.encode_observation(obs_tensor)
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(seed))

    original_chunks: list[np.ndarray] = []
    shuffled_chunks: list[np.ndarray] = []
    for start in range(0, len(observations), int(batch_size)):
        stop = min(start + int(batch_size), len(observations))
        latent_batch = state_latent[start:stop]
        action_batch = action_tensor[start:stop]
        shuffled_action_batch = shuffled_action_tensor[start:stop]
        source = torch.randn(
            stop - start,
            model.latent_dim,
            generator=generator,
            device=device,
            dtype=latent_batch.dtype,
        )
        original = model.predict_next_latent(latent_batch, action_batch, source=source)
        shuffled = model.predict_next_latent(latent_batch, shuffled_action_batch, source=source)
        original_chunks.append(original.detach().cpu().numpy().astype(np.float32, copy=False))
        shuffled_chunks.append(shuffled.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(original_chunks, axis=0), np.concatenate(shuffled_chunks, axis=0)


def measure_action_shuffle_sensitivity(config: MeasureActionShuffleSensitivityConfig) -> Path:
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "stablewm_hdf5":
        raise NotImplementedError("This script currently supports only stablewm_hdf5 checkpoints.")
    if project_config.model.observation_shape != (4,):
        raise NotImplementedError("This script currently supports only 4D pointmass observations.")
    if project_config.model.observation_encoder not in {"identity", "no_encoder"}:
        raise NotImplementedError("This script currently supports only identity observation encoders.")

    device = _resolve_device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)
    dataset_path = _resolve_hdf5_path(run_dir)
    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    with h5py.File(dataset_path, "r") as handle:
        observations = np.asarray(handle[project_config.data.observation_key], dtype=np.float32)
        actions = np.asarray(handle[project_config.data.action_key], dtype=np.float32)
        if config.num_states > len(observations):
            raise ValueError(f"Requested {config.num_states} states but dataset has only {len(observations)}")
        dataset_indices = np.sort(rng.choice(len(observations), size=config.num_states, replace=False).astype(np.int64))
        selected_observations = observations[dataset_indices]
        selected_actions = actions[dataset_indices]

    perm, action_l2 = _best_action_derangement(
        selected_actions,
        seed=config.seed,
        trials=config.derangement_trials,
    )
    shuffled_actions = selected_actions[perm]

    original_xy_by_state: list[np.ndarray] = []
    shuffled_xy_by_state: list[np.ndarray] = []
    for offset, dataset_index in enumerate(dataset_indices.tolist()):
        seed_offset = config.seed * 100_000 + offset
        original_xy = _sample_endpoint_positions(
            model,
            selected_observations[offset],
            selected_actions[offset],
            device=device,
            sample_count=config.num_sources_per_state,
            batch_size=config.sample_batch_size,
            seed=seed_offset,
        )
        shuffled_xy = _sample_endpoint_positions(
            model,
            selected_observations[offset],
            shuffled_actions[offset],
            device=device,
            sample_count=config.num_sources_per_state,
            batch_size=config.sample_batch_size,
            seed=seed_offset,
        )
        original_xy_by_state.append(original_xy)
        shuffled_xy_by_state.append(shuffled_xy)

    original_endpoints, shuffled_endpoints = _sample_endpoint_positions_matched(
        model,
        np.repeat(selected_observations, config.num_sources_per_state, axis=0),
        np.repeat(selected_actions, config.num_sources_per_state, axis=0),
        np.repeat(shuffled_actions, config.num_sources_per_state, axis=0),
        device=device,
        batch_size=config.sample_batch_size,
        seed=config.seed + 1_000_000,
    )

    matched_l2 = np.linalg.norm(original_endpoints - shuffled_endpoints, axis=-1)
    per_state: list[dict[str, object]] = []
    centroid_l2_values: list[float] = []
    symmetric_nn_values: list[float] = []
    action_cosines = np.sum(selected_actions * shuffled_actions, axis=-1) / (
        np.linalg.norm(selected_actions, axis=-1) * np.linalg.norm(shuffled_actions, axis=-1) + 1e-8
    )

    for state_index, dataset_index in enumerate(dataset_indices.tolist()):
        original_xy = original_xy_by_state[state_index]
        shuffled_xy = shuffled_xy_by_state[state_index]
        centroid_l2 = float(np.linalg.norm(original_xy.mean(axis=0) - shuffled_xy.mean(axis=0)))
        symmetric_nn_xy = 0.5 * (
            _mean_min_distance(original_xy, shuffled_xy) +
            _mean_min_distance(shuffled_xy, original_xy)
        )
        centroid_l2_values.append(centroid_l2)
        symmetric_nn_values.append(symmetric_nn_xy)
        state_slice = slice(
            state_index * config.num_sources_per_state,
            (state_index + 1) * config.num_sources_per_state,
        )
        per_state.append(
            {
                "dataset_index": int(dataset_index),
                "shuffled_from_index": int(dataset_indices[perm[state_index]]),
                "action_l2": float(action_l2[state_index]),
                "action_cosine": float(action_cosines[state_index]),
                "matched_endpoint_l2": float(matched_l2[state_slice].mean()),
                "centroid_l2": centroid_l2,
                "symmetric_nn_xy": symmetric_nn_xy,
            }
        )

    sorted_snn = np.sort(np.asarray(symmetric_nn_values, dtype=np.float64))
    p90_index = int(0.9 * (len(sorted_snn) - 1))
    metadata = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "num_states": int(config.num_states),
        "num_sources_per_state": int(config.num_sources_per_state),
        "sample_batch_size": int(config.sample_batch_size),
        "derangement_trials": int(config.derangement_trials),
        "seed": int(config.seed),
        "device": str(device),
        "dataset_indices": dataset_indices.tolist(),
        "metrics": {
            "mean_action_l2": float(action_l2.mean()),
            "median_action_l2": float(np.median(action_l2)),
            "mean_action_cosine": float(action_cosines.mean()),
            "median_action_cosine": float(np.median(action_cosines)),
            "mean_matched_endpoint_l2": float(matched_l2.mean()),
            "median_matched_endpoint_l2": float(np.median(matched_l2)),
            "mean_centroid_l2": float(np.mean(centroid_l2_values)),
            "median_centroid_l2": float(np.median(centroid_l2_values)),
            "mean_symmetric_nn_xy": float(np.mean(symmetric_nn_values)),
            "median_symmetric_nn_xy": float(np.median(symmetric_nn_values)),
            "p90_symmetric_nn_xy": float(sorted_snn[p90_index]),
        },
        "per_state": per_state,
    }
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        MeasureActionShuffleSensitivityConfig,
        description=(
            "Measure how much the learned successor endpoint distribution changes when current actions "
            "are replaced by a deranged, high-action-distance shuffle across sampled states."
        ),
    )
    output_path = measure_action_shuffle_sensitivity(config)
    print(output_path)


if __name__ == "__main__":
    main()
