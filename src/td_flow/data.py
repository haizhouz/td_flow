from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


def _as_float_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


class OGBenchNPZDataset(Dataset):
    def __init__(
        self,
        dataset: dict[str, object],
        *,
        observation_key: str = "state",
        action_key: str = "action",
        goal_key: str | None = None,
        policy_embedding_key: str | None = None,
        policy_embeddings: torch.Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        self.observation_key = observation_key
        self.action_key = action_key
        self.goal_key = goal_key
        self.policy_embedding_key = policy_embedding_key
        self.policy_embeddings = policy_embeddings
        self.observations = _as_float_tensor(self._resolve_key(observation_key, default_key="observations"))
        self.actions = _as_float_tensor(self._resolve_key(action_key, default_key="actions"))
        self.next_observations = _as_float_tensor(
            self._resolve_next_key(observation_key, default_key="next_observations")
        )
        terminals = dataset.get("terminals")
        if terminals is None:
            self.terminals = torch.zeros(self.actions.shape[0], dtype=torch.bool)
        else:
            self.terminals = _as_float_tensor(terminals).bool()

    def _resolve_key(self, requested_key: str, *, default_key: str) -> object:
        if requested_key in self.dataset:
            return self.dataset[requested_key]

        alias_map = {
            "state": "observations",
            "observation": "observations",
            "observations": "observations",
            "action": "actions",
            "actions": "actions",
        }
        resolved_key = alias_map.get(requested_key)
        if resolved_key is not None and resolved_key in self.dataset:
            return self.dataset[resolved_key]
        if default_key in self.dataset:
            return self.dataset[default_key]
        raise KeyError(f"Could not resolve key '{requested_key}' in OGBench dataset.")

    def _resolve_next_key(self, requested_key: str, *, default_key: str) -> object:
        direct_next_key = f"next_{requested_key}"
        if direct_next_key in self.dataset:
            return self.dataset[direct_next_key]

        alias_map = {
            "state": "next_observations",
            "observation": "next_observations",
            "observations": "next_observations",
            "action": "next_actions",
            "actions": "next_actions",
        }
        resolved_key = alias_map.get(requested_key)
        if resolved_key is not None and resolved_key in self.dataset:
            return self.dataset[resolved_key]
        if default_key in self.dataset:
            return self.dataset[default_key]
        raise KeyError(f"Could not resolve next-step key for '{requested_key}' in OGBench dataset.")

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obs = self.observations[index]
        next_obs = self.next_observations[index]
        action = self.actions[index]
        next_actions = None
        direct_next_action_key = f"next_{self.action_key}"
        if direct_next_action_key in self.dataset:
            next_actions = _as_float_tensor(self.dataset[direct_next_action_key])
        elif "next_actions" in self.dataset:
            next_actions = _as_float_tensor(self.dataset["next_actions"])

        if next_actions is not None:
            next_action = next_actions[index]
        elif index + 1 < len(self.actions) and not bool(self.terminals[index]):
            next_action = self.actions[index + 1]
        else:
            next_action = torch.zeros_like(action)

        goal = next_obs
        if self.goal_key is not None:
            goal = _as_float_tensor(self._resolve_key(self.goal_key, default_key="next_observations"))[index]

        policy_embedding = None
        if self.policy_embeddings is not None:
            policy_embedding = self.policy_embeddings[index]
        elif self.policy_embedding_key is not None and self.policy_embedding_key in self.dataset:
            policy_embedding = _as_float_tensor(self.dataset[self.policy_embedding_key])[index]

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "next_action": next_action,
            "goal": goal,
            **({"policy_embedding": policy_embedding} if policy_embedding is not None else {}),
        }


class TD2CFMDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        observation_key: str = "state",
        action_key: str = "action",
        goal_key: str | None = None,
        policy_embedding_key: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.observation_key = observation_key
        self.action_key = action_key
        self.goal_key = goal_key
        self.policy_embedding_key = policy_embedding_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        observations = _as_float_tensor(sample[self.observation_key])
        actions = _as_float_tensor(sample[self.action_key])

        if observations.shape[0] < 2:
            raise ValueError(
                "TD2CFMDataset requires observation sequences of length >= 2. "
                "Increase num_steps when constructing the HDF5Dataset."
            )

        if actions.shape[0] < 2:
            next_action = torch.zeros_like(actions[0])
        else:
            next_action = actions[1]

        output = {
            "obs": observations[0],
            "next_obs": observations[1],
            "action": actions[0],
            "next_action": next_action,
        }

        if self.goal_key is not None and self.goal_key in sample:
            goal = _as_float_tensor(sample[self.goal_key])
            output["goal"] = goal[-1] if goal.ndim > observations[0].ndim else goal
        else:
            output["goal"] = observations[-1]

        if self.policy_embedding_key is not None and self.policy_embedding_key in sample:
            policy_embedding = _as_float_tensor(sample[self.policy_embedding_key])
            output["policy_embedding"] = (
                policy_embedding[-1] if policy_embedding.ndim > 1 else policy_embedding
            )

        return output


def build_td2_hdf5_dataset(config: DataConfig) -> TD2CFMDataset:
    from stable_worldmodel.data import HDF5Dataset

    base_config = replace(config, num_steps=max(config.num_steps, 2))
    dataset = HDF5Dataset(
        name=base_config.dataset_name,
        frameskip=base_config.frameskip,
        num_steps=base_config.num_steps,
        cache_dir=base_config.dir,
        keys_to_load=base_config.resolved_keys_to_load(),
    )
    return TD2CFMDataset(
        dataset,
        observation_key=base_config.observation_key,
        action_key=base_config.action_key,
        goal_key=base_config.goal_key,
        policy_embedding_key=base_config.policy_embedding_key,
    )


def build_td2_ogbench_dataset(config: DataConfig) -> OGBenchNPZDataset:
    import ogbench

    dataset_dir = config.dir or "/home/haizhou/.ogbench/data"
    train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config.dataset_name,
        dataset_dir=dataset_dir,
        dataset_only=True,
    )
    if config.split == "train":
        dataset = train_dataset
    elif config.split == "val":
        dataset = val_dataset
    else:
        raise ValueError("OGBench split must be one of: train, val")
    return OGBenchNPZDataset(
        dataset,
        observation_key=config.observation_key,
        action_key=config.action_key,
        goal_key=config.goal_key,
        policy_embedding_key=config.policy_embedding_key,
    )


def build_td2_dataloader(config: DataConfig, *, shuffle: bool = True) -> DataLoader:
    if config.backend == "stablewm_hdf5":
        dataset = build_td2_hdf5_dataset(config)
    elif config.backend == "ogbench_npz":
        dataset = build_td2_ogbench_dataset(config)
    else:
        raise ValueError(f"Unsupported data backend: {config.backend}")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=shuffle,
    )


def infer_shapes(sample: dict[str, torch.Tensor]) -> tuple[tuple[int, ...], int, int]:
    observation_shape = tuple(sample["obs"].shape[1:])
    action_dim = int(sample["action"][0].numel())
    policy_embedding = sample.get("policy_embedding")
    policy_embedding_dim = 0 if policy_embedding is None else int(policy_embedding[0].numel())
    return observation_shape, action_dim, policy_embedding_dim
