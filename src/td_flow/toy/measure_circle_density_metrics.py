from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro

from ..rollout import (
    _checkpoint_run_dir,
    load_project_config_from_run_dir,
    load_td2_model,
)
from .plot_circle_exploration_exact_comparison import (
    _sample_behavior_successor_positions,
    _sample_policy_successor_positions,
)
from .plot_circle_policy_conditioned_occupancy import (
    _mean_min_distance,
    _resolve_device,
    _resolve_hdf5_path,
    _sample_model_positions,
    _sample_valid_start_indices,
)


@dataclass
class ToyCircleDensityMetricsConfig:
    dataset_only_checkpoint_path: str
    offpolicy_checkpoint_path: str
    device: str = "auto"
    num_states: int = 32
    sample_count: int = 4096
    sample_batch_size: int = 1024
    max_future_steps: int = 1024
    num_angle_bins: int = 128
    seed: int = 0
    output_path: str | None = None


def _default_output_path(offpolicy_checkpoint_path: str) -> Path:
    return _checkpoint_run_dir(offpolicy_checkpoint_path) / "toy_circle_density_metrics.json"


def _angles(positions: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2.0 * np.pi)


def _normalized_angle_histogram(positions: np.ndarray, num_bins: int) -> np.ndarray:
    hist, _ = np.histogram(
        _angles(positions),
        bins=num_bins,
        range=(0.0, 2.0 * np.pi),
    )
    hist = hist.astype(np.float64, copy=False)
    total = float(hist.sum())
    if total <= 0.0:
        raise ValueError("angle histogram is empty")
    return hist / total


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.abs(left - right).sum())


def _symmetric_min_distance(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * (_mean_min_distance(left, right) + _mean_min_distance(right, left))


def measure_toy_circle_density_metrics(config: ToyCircleDensityMetricsConfig) -> Path:
    dataset_only_run_dir = _checkpoint_run_dir(config.dataset_only_checkpoint_path)
    offpolicy_run_dir = _checkpoint_run_dir(config.offpolicy_checkpoint_path)
    dataset_only_project_config = load_project_config_from_run_dir(dataset_only_run_dir)
    offpolicy_project_config = load_project_config_from_run_dir(offpolicy_run_dir)

    if dataset_only_project_config.data.dataset_name != offpolicy_project_config.data.dataset_name:
        raise ValueError("Both checkpoints must use the same dataset.")
    if dataset_only_project_config.data.next_action_key != dataset_only_project_config.data.action_key:
        raise ValueError("dataset-only checkpoint must use next_action_key=action.")
    if offpolicy_project_config.data.next_action_key == offpolicy_project_config.data.action_key:
        raise ValueError("off-policy checkpoint must use a non-dataset next_action_key.")

    dataset_path = _resolve_hdf5_path(dataset_only_run_dir)
    if dataset_path != _resolve_hdf5_path(offpolicy_run_dir):
        raise ValueError("Both checkpoints must point to the same HDF5 dataset.")

    dataset_metadata = json.loads(dataset_path.with_suffix(".json").read_text())
    if dataset_metadata.get("dataset_type") != "toy_circle_exploration":
        raise ValueError("This metric expects a toy_circle_exploration dataset.")

    behavior_policy = dataset_metadata["behavior_policy"]
    target_policy = dataset_metadata["target_policy"]
    gamma = float(offpolicy_project_config.model.gamma)

    device = _resolve_device(config.device)
    dataset_only_model = load_td2_model(config.dataset_only_checkpoint_path, dataset_only_project_config, device=device)
    offpolicy_model = load_td2_model(config.offpolicy_checkpoint_path, offpolicy_project_config, device=device)

    output_path = Path(config.output_path) if config.output_path is not None else _default_output_path(config.offpolicy_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states_metadata: list[dict[str, object]] = []
    exact_mu_pi_tv: list[float] = []
    exact_mu_pi_support: list[float] = []
    dataset_model_mu_tv: list[float] = []
    offpolicy_model_pi_tv: list[float] = []
    dataset_model_pi_tv: list[float] = []
    offpolicy_model_mu_tv: list[float] = []

    with h5py.File(dataset_path, "r") as handle:
        observations = handle[dataset_only_project_config.data.observation_key]
        actions = handle[dataset_only_project_config.data.action_key]
        ep_offset = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        start_indices = _sample_valid_start_indices(
            ep_offset,
            ep_len,
            count=config.num_states,
            min_future_steps=1,
            seed=config.seed,
        )

        for column, start_index in enumerate(start_indices.tolist()):
            observation = np.asarray(observations[start_index], dtype=np.float32)
            current_action = np.asarray(actions[start_index], dtype=np.float32)

            dataset_model_positions = _sample_model_positions(
                dataset_only_model,
                observation,
                current_action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            offpolicy_model_positions = _sample_model_positions(
                offpolicy_model,
                observation,
                current_action,
                device=device,
                sample_count=config.sample_count,
                batch_size=config.sample_batch_size,
            )
            exact_mu_positions = _sample_behavior_successor_positions(
                observation,
                current_action,
                gamma=gamma,
                behavior_policy=behavior_policy,
                sample_count=config.sample_count,
                max_future_steps=config.max_future_steps,
                seed=config.seed + column,
            )
            exact_pi_positions, _ = _sample_policy_successor_positions(
                observation,
                current_action,
                gamma=gamma,
                policy_delta_theta=float(target_policy["delta_theta"]),
                sample_count=config.sample_count,
                max_future_steps=config.max_future_steps,
                seed=config.seed + 1000 + column,
            )

            dataset_model_hist = _normalized_angle_histogram(dataset_model_positions, config.num_angle_bins)
            offpolicy_model_hist = _normalized_angle_histogram(offpolicy_model_positions, config.num_angle_bins)
            exact_mu_hist = _normalized_angle_histogram(exact_mu_positions, config.num_angle_bins)
            exact_pi_hist = _normalized_angle_histogram(exact_pi_positions, config.num_angle_bins)

            mu_pi_tv = _total_variation(exact_mu_hist, exact_pi_hist)
            mu_pi_support = _symmetric_min_distance(exact_mu_positions, exact_pi_positions)
            dataset_mu_tv = _total_variation(dataset_model_hist, exact_mu_hist)
            offpolicy_pi_tv = _total_variation(offpolicy_model_hist, exact_pi_hist)
            dataset_pi_tv = _total_variation(dataset_model_hist, exact_pi_hist)
            offpolicy_mu_tv = _total_variation(offpolicy_model_hist, exact_mu_hist)

            exact_mu_pi_tv.append(mu_pi_tv)
            exact_mu_pi_support.append(mu_pi_support)
            dataset_model_mu_tv.append(dataset_mu_tv)
            offpolicy_model_pi_tv.append(offpolicy_pi_tv)
            dataset_model_pi_tv.append(dataset_pi_tv)
            offpolicy_model_mu_tv.append(offpolicy_mu_tv)

            states_metadata.append(
                {
                    "start_index": int(start_index),
                    "observation": observation.astype(float).tolist(),
                    "current_action": current_action.astype(float).tolist(),
                    "exact_mu_pi_tv": mu_pi_tv,
                    "exact_mu_pi_support_symmetric_nn": mu_pi_support,
                    "dataset_model_vs_exact_mu_tv": dataset_mu_tv,
                    "offpolicy_model_vs_exact_pi_tv": offpolicy_pi_tv,
                    "dataset_model_vs_exact_pi_tv": dataset_pi_tv,
                    "offpolicy_model_vs_exact_mu_tv": offpolicy_mu_tv,
                }
            )

    metrics = {
        "dataset_only_checkpoint_path": str(Path(config.dataset_only_checkpoint_path).resolve()),
        "offpolicy_checkpoint_path": str(Path(config.offpolicy_checkpoint_path).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "behavior_policy": behavior_policy,
        "target_policy": target_policy,
        "gamma": gamma,
        "num_states": config.num_states,
        "sample_count": config.sample_count,
        "num_angle_bins": config.num_angle_bins,
        "max_future_steps": config.max_future_steps,
        "seed": config.seed,
        "mean_exact_mu_pi_tv": float(np.mean(exact_mu_pi_tv)),
        "mean_exact_mu_pi_support_symmetric_nn": float(np.mean(exact_mu_pi_support)),
        "mean_dataset_model_vs_exact_mu_tv": float(np.mean(dataset_model_mu_tv)),
        "mean_offpolicy_model_vs_exact_pi_tv": float(np.mean(offpolicy_model_pi_tv)),
        "mean_dataset_model_vs_exact_pi_tv": float(np.mean(dataset_model_pi_tv)),
        "mean_offpolicy_model_vs_exact_mu_tv": float(np.mean(offpolicy_model_mu_tv)),
        "states": states_metadata,
    }
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        ToyCircleDensityMetricsConfig,
        description="Measure support and density differences for toy-circle dataset-only and off-policy checkpoints.",
    )
    output_path = measure_toy_circle_density_metrics(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
