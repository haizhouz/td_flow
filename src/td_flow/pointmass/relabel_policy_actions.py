from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro

from .loop_policy import PointMassLoopPolicyConfig, TorchPointMassLoopScriptedPolicy


@dataclass
class RelabelPointMassPolicyActionsConfig:
    input_hdf5_path: str
    output_hdf5_path: str
    observation_key: str = "observation"
    output_action_key: str = "policy_action"
    max_noise_fraction: float = 0.1
    chunk_size: int = 131072
    device: str = "auto"
    seed: int = 0
    compile_policy: bool = False
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sample_bounded_relative_noise(
    action_chunk: torch.Tensor,
    max_noise_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if max_noise_fraction <= 0.0:
        return torch.zeros_like(action_chunk)

    noise_direction = torch.randn(
        action_chunk.shape,
        generator=generator,
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )
    noise_direction_norm = torch.linalg.vector_norm(noise_direction, dim=-1, keepdim=True)
    noise_direction = noise_direction / noise_direction_norm.clamp_min(1e-12)

    max_noise_norm = max_noise_fraction * torch.linalg.vector_norm(action_chunk, dim=-1, keepdim=True)
    noise_radius = torch.rand(
        (action_chunk.shape[0], 1),
        generator=generator,
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )
    return noise_direction * noise_radius * max_noise_norm


def relabel_pointmass_policy_actions(config: RelabelPointMassPolicyActionsConfig) -> Path:
    input_path = Path(config.input_hdf5_path)
    output_path = Path(config.output_hdf5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(config.seed)

    policy = TorchPointMassLoopScriptedPolicy(config.policy).to(device)
    if config.compile_policy:
        policy = torch.compile(policy)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        for key in src.keys():
            if key == config.output_action_key:
                continue
            src.copy(src[key], dst, name=key)

        observations = src[config.observation_key]
        if observations.shape[-1] < 4:
            raise ValueError(
                f"Expected pointmass observations with at least 4 dims, got shape {observations.shape}"
            )

        output_dataset = dst.create_dataset(
            config.output_action_key,
            shape=(observations.shape[0], 2),
            dtype=np.float32,
        )

        for start in range(0, observations.shape[0], config.chunk_size):
            stop = min(start + config.chunk_size, observations.shape[0])
            obs_chunk = torch.from_numpy(np.asarray(observations[start:stop], dtype=np.float32)).to(device)
            with torch.no_grad():
                action_chunk = policy(obs_chunk)
                if config.max_noise_fraction > 0.0:
                    action_chunk = action_chunk + _sample_bounded_relative_noise(
                        action_chunk=action_chunk,
                        max_noise_fraction=config.max_noise_fraction,
                        generator=generator,
                    )
                action_chunk = torch.clamp(action_chunk, -1.0, 1.0)
            output_dataset[start:stop] = action_chunk.cpu().numpy().astype(np.float32, copy=False)

        metadata = {
            "input_hdf5_path": str(input_path.resolve()),
            "output_hdf5_path": str(output_path.resolve()),
            "observation_key": config.observation_key,
            "output_action_key": config.output_action_key,
            "max_noise_fraction": config.max_noise_fraction,
            "chunk_size": config.chunk_size,
            "device": str(device),
            "seed": config.seed,
            "compile_policy": config.compile_policy,
            "policy": asdict(config.policy),
        }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    config = tyro.cli(
        RelabelPointMassPolicyActionsConfig,
        description="Relabel a pointmass HDF5 dataset with scripted-policy actions plus bounded relative noise.",
    )
    output_path = relabel_pointmass_policy_actions(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
