from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import tyro

from .rollout import (
    _checkpoint_run_dir,
    decode_cube_single_state,
    ensure_egl,
    find_valid_start,
    load_project_config_from_run_dir,
    load_td2_model,
    render_frame,
)


@dataclass
class SuccessorGridConfig:
    checkpoint_path: str
    split: str = "val"
    start_index: int | None = None
    sample_count: int = 100
    grid_cols: int = 10
    device: str = "cpu"
    output_path: str | None = None
    seed: int = 0
    add_info: bool = True
    visualize_info: bool = False
    padding: int = 8
    label_tiles: bool = True
    sort_by_uncertainty: bool = True
    uncertainty_descending: bool = False


def _compute_uncertainty_scores(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if predictions.ndim != 2:
        raise ValueError(f"predictions must have shape (N, D), got {predictions.shape}")
    if predictions.shape[0] == 0:
        raise ValueError("predictions must be non-empty")

    centered = predictions - predictions.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[float(covariance)]], dtype=np.float64)
    regularized_covariance = covariance + 1e-6 * np.eye(covariance.shape[0], dtype=np.float64)
    precision = np.linalg.pinv(regularized_covariance)
    raw_scores = np.einsum("nd,dd,nd->n", centered, precision, centered, optimize=True) / predictions.shape[1]

    order = np.argsort(raw_scores)
    percentiles = np.empty_like(raw_scores, dtype=np.float64)
    if len(raw_scores) == 1:
        percentiles[0] = 0.0
    else:
        percentiles[order] = np.linspace(0.0, 1.0, num=len(raw_scores), endpoint=True)
    return raw_scores.astype(np.float32), percentiles.astype(np.float32)


def _sort_indices_by_uncertainty(percentiles: np.ndarray, *, descending: bool) -> np.ndarray:
    order = np.argsort(percentiles)
    if descending:
        order = order[::-1]
    return order


def _make_image_grid(
    frames: list[np.ndarray],
    *,
    cols: int,
    padding: int,
    label_tiles: bool,
    tile_labels: list[str] | None = None,
) -> Image.Image:
    if not frames:
        raise ValueError("frames must be non-empty")
    if tile_labels is not None and len(tile_labels) != len(frames):
        raise ValueError("tile_labels must match the number of frames")
    pil_frames = [Image.fromarray(frame) for frame in frames]
    frame_width = pil_frames[0].width
    frame_height = pil_frames[0].height
    rows = (len(pil_frames) + cols - 1) // cols
    label_height = 18 if label_tiles else 0

    canvas = Image.new(
        "RGB",
        (
            cols * frame_width + padding * (cols + 1),
            rows * (frame_height + label_height) + padding * (rows + 1),
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    for index, image in enumerate(pil_frames):
        row = index // cols
        col = index % cols
        x = padding + col * (frame_width + padding)
        y = padding + row * (frame_height + label_height + padding)
        if label_tiles:
            label = tile_labels[index] if tile_labels is not None else f"{index:03d}"
            draw.text((x, y), label, fill=(0, 0, 0))
            canvas.paste(image, (x, y + label_height))
        else:
            canvas.paste(image, (x, y))
    return canvas


def run_successor_grid(config: SuccessorGridConfig) -> Path:
    ensure_egl()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    run_dir = _checkpoint_run_dir(config.checkpoint_path)
    project_config = load_project_config_from_run_dir(run_dir)
    if project_config.data.backend != "ogbench_npz":
        raise NotImplementedError("This script currently supports only ogbench_npz checkpoints.")
    if project_config.data.dataset_name != "cube-single-play-v0":
        raise NotImplementedError("This script currently supports only cube-single-play-v0.")
    if project_config.model.observation_encoder not in {"identity", "no_encoder"}:
        raise NotImplementedError("This script currently supports only identity observation encoders.")

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
    start_index = find_valid_start(
        np.asarray(dataset["terminals"]),
        horizon=1,
        start_index=config.start_index,
        seed=config.seed,
    )

    device = torch.device(config.device)
    model = load_td2_model(config.checkpoint_path, project_config, device=device)
    obs = np.asarray(dataset["observations"][start_index], dtype=np.float32)
    action = np.asarray(dataset["actions"][start_index], dtype=np.float32)
    base_qpos = np.asarray(dataset["qpos"][start_index], dtype=np.float32)
    base_qvel = np.asarray(dataset["qvel"][start_index], dtype=np.float32)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    action_tensor = torch.from_numpy(action).unsqueeze(0).to(device=device, dtype=torch.float32)
    state_latent = model.encode_observation(obs_tensor)

    frames: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(config.sample_count):
            prediction = model.predict_next_latent(state_latent, action_tensor).squeeze(0).cpu().numpy()
            predictions.append(prediction.astype(np.float32, copy=False))
            qpos, qvel = decode_cube_single_state(prediction, base_qpos, base_qvel)
            frames.append(render_frame(env, qpos, qvel))

    prediction_array = np.stack(predictions, axis=0)
    raw_uncertainty, percentile_uncertainty = _compute_uncertainty_scores(prediction_array)
    tile_labels = [f"{index:03d} u={uncertainty:.2f}" for index, uncertainty in enumerate(percentile_uncertainty)]

    order = np.arange(len(frames))
    if config.sort_by_uncertainty:
        order = _sort_indices_by_uncertainty(percentile_uncertainty, descending=config.uncertainty_descending)

    ordered_frames = [frames[int(index)] for index in order]
    tile_labels = [
        f"{int(index):03d} u={percentile_uncertainty[int(index)]:.2f}"
        for index in order
    ]

    grid = _make_image_grid(
        ordered_frames,
        cols=max(config.grid_cols, 1),
        padding=max(config.padding, 0),
        label_tiles=config.label_tiles,
        tile_labels=tile_labels if config.label_tiles else None,
    )
    output_path = (
        Path(config.output_path)
        if config.output_path is not None
        else Path(config.checkpoint_path).parent / "successor_grid.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)

    meta = {
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "dataset_name": project_config.data.dataset_name,
        "split": config.split,
        "start_index": start_index,
        "sample_count": config.sample_count,
        "grid_cols": config.grid_cols,
        "sort_by_uncertainty": config.sort_by_uncertainty,
        "uncertainty_descending": config.uncertainty_descending,
        "uncertainty_metric": "mahalanobis_percentile_within_samples",
        "raw_uncertainty_mean": float(raw_uncertainty.mean()),
        "raw_uncertainty_std": float(raw_uncertainty.std()),
        "output_path": str(output_path.resolve()),
    }
    (output_path.parent / "successor_grid.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    try:
        env.close()
    except Exception:
        pass
    return output_path


def main() -> None:
    config = tyro.cli(
        SuccessorGridConfig,
        description="Render many successor samples from one fixed OGBench state-action pair as an image grid.",
    )
    output_path = run_successor_grid(config)
    print(str(output_path))


if __name__ == "__main__":
    main()
