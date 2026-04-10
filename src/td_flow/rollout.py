from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import tyro

from .config import BackboneConfig, DataConfig, ModelConfig, PlanningConfig, ProjectConfig, TrainConfig
from .model import TD2CFMModel


XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALER = 10.0
GRIPPER_SCALER = 3.0


@dataclass
class RolloutConfig:
    checkpoint_path: str
    split: str = "val"
    start_index: int | None = None
    horizon: int = 8
    device: str = "cpu"
    output_dir: str | None = None
    seed: int = 0
    fps: int = 4
    add_info: bool = True
    visualize_info: bool = False


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_tuple(item) for key, item in value.items()}
    return value


def _checkpoint_run_dir(ckpt_path: str) -> Path:
    checkpoint_path = Path(ckpt_path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _default_rollout_dir(ckpt_path: str) -> Path:
    checkpoint_path = Path(ckpt_path)
    return checkpoint_path.parent / "rollout"


def load_project_config_from_run_dir(run_dir: Path) -> ProjectConfig:
    raw = json.loads((run_dir / "project_config.json").read_text())
    data_raw = dict(raw["data"])
    if "dir" not in data_raw and "cache_dir" in data_raw:
        data_raw["dir"] = data_raw.pop("cache_dir")
    else:
        data_raw.pop("cache_dir", None)

    model_raw = dict(raw["model"])
    backbone_raw = _to_tuple(model_raw.pop("backbone"))
    train_raw = _to_tuple(raw["train"])
    planning_raw = _to_tuple(raw.get("planning", {}))

    data = DataConfig(**_to_tuple(data_raw))
    model = ModelConfig(
        **_to_tuple(model_raw),
        backbone=BackboneConfig(**backbone_raw),
    )
    train = TrainConfig(**train_raw)
    planning = PlanningConfig(**planning_raw) if planning_raw else PlanningConfig()
    return ProjectConfig(data=data, model=model, train=train, planning=planning)


def load_td2_model(ckpt_path: str, project_config: ProjectConfig, device: torch.device) -> TD2CFMModel:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = TD2CFMModel(project_config.model).to(device)
    state_dict = {
        key.removeprefix("td2_cfm."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("td2_cfm.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def find_valid_start(terminals: np.ndarray, horizon: int, start_index: int | None, seed: int) -> int:
    max_start = len(terminals) - horizon - 1
    if max_start < 0:
        raise ValueError(f"Dataset is too short for horizon={horizon}.")

    def _is_valid(index: int) -> bool:
        return not bool(np.any(terminals[index : index + horizon]))

    if start_index is not None:
        if start_index < 0 or start_index > max_start:
            raise ValueError(f"start_index must be in [0, {max_start}]")
        if not _is_valid(start_index):
            raise ValueError(f"start_index={start_index} crosses a terminal transition for horizon={horizon}")
        return start_index

    valid_indices = [index for index in range(max_start + 1) if _is_valid(index)]
    if not valid_indices:
        raise ValueError(f"No valid rollout start found for horizon={horizon}")
    rng = np.random.default_rng(seed)
    return int(rng.choice(valid_indices))


def decode_cube_single_state(observation: np.ndarray, base_qpos: np.ndarray, base_qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if observation.shape[-1] != 28:
        raise ValueError(f"Expected 28D cube-single state observation, got shape {observation.shape}")

    qpos = np.array(base_qpos, copy=True)
    qvel = np.array(base_qvel, copy=True)

    qpos[:6] = observation[:6]
    qvel[:6] = observation[6:12]

    gripper_opening = float(np.clip(observation[17] / GRIPPER_SCALER, 0.0, 1.0))
    qpos[6] = gripper_opening * 0.8

    block_pos = observation[19:22] / XYZ_SCALER + XYZ_CENTER
    block_quat = np.array(observation[22:26], copy=True)
    quat_norm = np.linalg.norm(block_quat)
    if quat_norm > 1e-8:
        block_quat /= quat_norm
    else:
        block_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    qpos[14:17] = block_pos
    qpos[17:21] = block_quat
    qvel[14:20] = 0.0
    return qpos, qvel


def render_frame(env, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    env.unwrapped.set_state(qpos, qvel)
    frame = env.render()
    return np.asarray(frame, dtype=np.uint8)


def ensure_egl() -> None:
    if os.environ.get("MUJOCO_GL") is None and os.environ.get("DISPLAY") is None:
        os.environ["MUJOCO_GL"] = "egl"


def run_rollout(config: RolloutConfig) -> Path:
    ensure_egl()
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "ogbench_npz":
        raise NotImplementedError("This rollout script currently supports only ogbench_npz checkpoints.")
    if project_config.data.dataset_name != "cube-single-play-v0":
        raise NotImplementedError("This rollout script currently supports only cube-single-play-v0.")
    if project_config.model.observation_encoder not in {"identity", "no_encoder"}:
        raise NotImplementedError("This rollout script currently supports only identity observation encoders.")

    import ogbench

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        project_config.data.dataset_name,
        dataset_dir=project_config.data.dir or "/home/haizhou/.ogbench/data",
        add_info=config.add_info,
        render_mode="rgb_array",
        visualize_info=config.visualize_info,
    )
    env.reset(seed=config.seed)
    dataset = train_dataset if config.split == "train" else val_dataset
    start_index = find_valid_start(np.asarray(dataset["terminals"]), config.horizon, config.start_index, config.seed)

    device = torch.device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)

    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_rollout_dir(config.checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "dataset_name": project_config.data.dataset_name,
        "split": config.split,
        "start_index": start_index,
        "horizon": config.horizon,
    }
    (output_dir / "rollout_config.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    predicted_frames: list[Image.Image] = []
    predicted_obs = np.asarray(dataset["observations"][start_index], dtype=np.float32)
    predicted_qpos = np.asarray(dataset["qpos"][start_index], dtype=np.float32)
    predicted_qvel = np.asarray(dataset["qvel"][start_index], dtype=np.float32)

    initial_frame = render_frame(env, predicted_qpos, predicted_qvel)
    Image.fromarray(initial_frame).save(frames_dir / "frame_000.png")
    predicted_frames.append(Image.fromarray(initial_frame))

    for step in range(config.horizon):
        action = np.asarray(dataset["actions"][start_index + step], dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.from_numpy(predicted_obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(device=device, dtype=torch.float32)
            state_latent = model.encode_observation(state_tensor)
            next_prediction = model.predict_next_latent(state_latent, action_tensor).squeeze(0).cpu().numpy()

        predicted_qpos, predicted_qvel = decode_cube_single_state(next_prediction, predicted_qpos, predicted_qvel)
        predicted_frame = render_frame(env, predicted_qpos, predicted_qvel)
        Image.fromarray(predicted_frame).save(frames_dir / f"frame_{step + 1:03d}.png")
        predicted_frames.append(Image.fromarray(predicted_frame))
        predicted_obs = next_prediction.astype(np.float32)

    if predicted_frames:
        predicted_frames[0].save(
            output_dir / "predicted_rollout.gif",
            save_all=True,
            append_images=predicted_frames[1:],
            duration=max(int(1000 / max(config.fps, 1)), 1),
            loop=0,
        )

    try:
        env.close()
    except Exception:
        pass
    return output_dir


def main() -> None:
    config = tyro.cli(RolloutConfig, description="Render OGBench cube-single checkpoint rollouts as predicted frames.")
    output_dir = run_rollout(config)
    print(str(output_dir))


if __name__ == "__main__":
    main()
