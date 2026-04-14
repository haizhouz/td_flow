from __future__ import annotations

import json
from dataclasses import dataclass

import tyro

from .data import summarize_episode_lengths


@dataclass
class DatasetStatsConfig:
    dataset_name: str
    dataset_dir: str = "/home/haizhou/.ogbench/data"
    split: str = "train"
    add_info: bool = False


def run_stats(config: DatasetStatsConfig) -> dict[str, float | int | str]:
    import ogbench

    train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config.dataset_name,
        dataset_dir=config.dataset_dir,
        dataset_only=True,
        add_info=config.add_info,
    )
    if config.split == "train":
        dataset = train_dataset
    elif config.split == "val":
        dataset = val_dataset
    else:
        raise ValueError("split must be one of: train, val")

    stats = summarize_episode_lengths(dataset["terminals"])
    return {
        "dataset_name": config.dataset_name,
        "split": config.split,
        **stats,
    }


def main() -> None:
    config = tyro.cli(DatasetStatsConfig, description="Print OGBench episode and trajectory length statistics.")
    print(json.dumps(run_stats(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
