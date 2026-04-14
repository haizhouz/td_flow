from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

from .loop_policy import PointMassLoopPolicyConfig, scripted_pointmass_loop_action


DEFAULT_DNC_ROOT = "/home/haizhou/Documents/DnC-FBr"


@dataclass
class EvalPointMassLoopPolicyConfig:
    dnc_root: str = DEFAULT_DNC_ROOT
    num_episodes: int = 20
    episode_length: int = 1000
    seed: int = 0
    output_path: str | None = None
    policy: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()


def _load_pointmass_module(dnc_root: str):
    if dnc_root not in sys.path:
        sys.path.append(dnc_root)
    from metamotivo.envs.dmc_tasks import pointmass

    return pointmass


def evaluate_pointmass_loop_policy(config: EvalPointMassLoopPolicyConfig) -> dict[str, object]:
    pointmass = _load_pointmass_module(config.dnc_root)

    returns: list[float] = []
    trajectories: list[list[list[float]]] = []
    for episode_index in range(config.num_episodes):
        env = pointmass.loop(
            random=config.seed + episode_index,
            environment_kwargs=dict(flat_observation=True),
        )
        time_step = env.reset()
        episode_return = 0.0
        trajectory: list[list[float]] = []
        for _ in range(config.episode_length):
            observation = np.asarray(time_step.observation["observations"], dtype=np.float32)
            trajectory.append(observation.tolist())
            action = scripted_pointmass_loop_action(observation, config=config.policy)
            time_step = env.step(action)
            episode_return += float(time_step.reward)
        returns.append(episode_return)
        trajectories.append(trajectory)

    summary = {
        "config": asdict(config),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "returns": returns,
        "trajectories": trajectories,
    }

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    config = tyro.cli(
        EvalPointMassLoopPolicyConfig,
        description="Evaluate the scripted pointmass loop policy against the local DMC loop task.",
    )
    summary = evaluate_pointmass_loop_policy(config)
    print(json.dumps(
        {
            "mean_return": summary["mean_return"],
            "std_return": summary["std_return"],
            "min_return": summary["min_return"],
            "max_return": summary["max_return"],
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
