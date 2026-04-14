from __future__ import annotations

import contextlib
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path

import lightning as pl
from loguru import logger
import stable_pretraining as spt
import torch
import tyro
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .config import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TrainEntryConfig,
    resolve_paper_max_steps,
)
from .data import build_td2_dataloader, infer_shapes
from .module import build_training_module


class ThroughputCallback(pl.Callback):
    def __init__(self, every_n_steps: int) -> None:
        self.every_n_steps = max(int(every_n_steps), 1)
        self._last_log_time: float | None = None
        self._last_logged_step = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._last_log_time = time.perf_counter()
        self._last_logged_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch_idx
        if not trainer.is_global_zero:
            return
        if self._last_log_time is None:
            self._last_log_time = time.perf_counter()
            self._last_logged_step = int(trainer.global_step)
            return

        global_step = int(trainer.global_step)
        if global_step <= self._last_logged_step:
            return
        if global_step % self.every_n_steps != 0:
            return

        batch_size = int(batch["obs"].shape[0]) if isinstance(batch, dict) and "obs" in batch else 1
        world_size = max(int(getattr(trainer, "world_size", 1) or 1), 1)
        elapsed = time.perf_counter() - self._last_log_time
        if elapsed <= 0:
            return

        step_delta = global_step - self._last_logged_step
        fps = (step_delta * batch_size * world_size) / elapsed
        metrics = {"train/fps": fps}
        active_loggers = trainer.loggers or []
        for active_logger in active_loggers:
            active_logger.log_metrics(metrics, step=global_step)
        self._last_log_time = time.perf_counter()
        self._last_logged_step = global_step


def resolve_run_dir(project_config: ProjectConfig) -> Path:
    run_name = project_config.train.run_name or project_config.data.dataset_name
    return Path(project_config.train.output_dir) / run_name


def timestamped_run_name(dataset_name: str) -> str:
    return f"{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


_RUN_NAME_TIMESTAMP_RE = re.compile(r"^(?P<base>.+)-(?P<ts>\d{8}-\d{6}(?:-\d{6})?)$")


def parse_run_name_timestamp(run_name: str) -> str | None:
    match = _RUN_NAME_TIMESTAMP_RE.match(run_name)
    if match is None:
        return None
    return match.group("ts")


def has_timestamped_suffix(run_name: str) -> bool:
    return parse_run_name_timestamp(run_name) is not None


def _checkpoint_run_dir_from_path(ckpt_path: str) -> Path:
    checkpoint_path = Path(ckpt_path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def find_latest_run_name(output_dir: Path, dataset_name: str) -> str | None:
    prefix = f"{dataset_name}-"
    candidates = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            if not path.is_dir() or not path.name.startswith(prefix):
                continue
            timestamp = parse_run_name_timestamp(path.name)
            if timestamp is None:
                continue
            candidates.append((timestamp, path.name))
    if not candidates:
        return None
    return max(candidates)[1]


def resolve_resume_run_name(output_dir: Path, run_name: str) -> str | None:
    exact_dir = output_dir / run_name
    if has_timestamped_suffix(run_name) and exact_dir.is_dir():
        return run_name

    latest_timestamped = find_latest_run_name(output_dir, run_name)
    if latest_timestamped is not None:
        return latest_timestamped

    if exact_dir.is_dir():
        return run_name

    return None


def resolve_latest_checkpoint_path(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_dir}")

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return str(last_checkpoint)

    candidates = sorted(
        (path for path in checkpoint_dir.glob("*.ckpt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found at {checkpoint_dir}")
    return str(candidates[0])


def normalize_project_config(project_config: ProjectConfig) -> ProjectConfig:
    output_dir = Path(project_config.train.output_dir)
    train_config = project_config.train

    run_name = train_config.run_name
    checkpoint_run_name = (
        _checkpoint_run_dir_from_path(train_config.resume_ckpt_path).name
        if train_config.resume_ckpt_path is not None
        else None
    )
    if train_config.run_mode == "validate" and checkpoint_run_name is not None:
        if run_name is not None and run_name != checkpoint_run_name:
            raise ValueError(
                f"resume_ckpt_path belongs to run '{checkpoint_run_name}', which does not match run_name '{run_name}'"
            )
        run_name = checkpoint_run_name
    elif run_name is None:
        if train_config.run_mode == "validate" and train_config.resume_ckpt_path is not None:
            run_name = checkpoint_run_name
        elif train_config.resume:
            if checkpoint_run_name is not None:
                run_name = checkpoint_run_name
            else:
                latest_run_name = find_latest_run_name(output_dir, project_config.data.dataset_name)
                if latest_run_name is None:
                    raise FileNotFoundError(
                        f"No prior run found for dataset prefix '{project_config.data.dataset_name}-' in {output_dir}"
                    )
                run_name = latest_run_name
        else:
            run_name = resolve_fresh_run_name(project_config, project_config.data.dataset_name)
    elif train_config.run_mode == "fit":
        if train_config.resume:
            if checkpoint_run_name is not None:
                if run_name != checkpoint_run_name and not checkpoint_run_name.startswith(f"{run_name}-"):
                    raise ValueError(
                        f"resume_ckpt_path belongs to run '{checkpoint_run_name}', which does not match run_name '{run_name}'"
                    )
                run_name = checkpoint_run_name
            else:
                resolved_run_name = resolve_resume_run_name(output_dir, run_name)
                if resolved_run_name is None:
                    raise FileNotFoundError(
                        f"No prior run found for run name or prefix '{run_name}' in {output_dir}"
                    )
                run_name = resolved_run_name
        elif not has_timestamped_suffix(run_name):
            run_name = resolve_fresh_run_name(project_config, run_name)

    resume_ckpt_path = train_config.resume_ckpt_path
    if train_config.run_mode == "fit":
        if train_config.resume:
            if resume_ckpt_path is None:
                resume_ckpt_path = resolve_latest_checkpoint_path(output_dir / run_name)
        else:
            resume_ckpt_path = None

    return replace(
        project_config,
        train=replace(
            train_config,
            run_name=run_name,
            resume_ckpt_path=resume_ckpt_path,
        ),
    )


def resolve_cache_root(project_config: ProjectConfig) -> Path:
    return Path(project_config.train.cache_root)


def resolve_cache_run_dir(project_config: ProjectConfig) -> Path:
    run_name = project_config.train.run_name or project_config.data.dataset_name
    return resolve_cache_root(project_config) / run_name


def resolve_wandb_state_dir(project_config: ProjectConfig) -> Path:
    if project_config.train.wandb_save_dir is not None:
        return Path(project_config.train.wandb_save_dir)
    return resolve_cache_run_dir(project_config) / "wandb"


def get_world_size() -> int:
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 1


def get_global_rank() -> int:
    for key in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def is_global_zero() -> bool:
    return get_global_rank() == 0


def resolve_distributed_launch_key(project_config: ProjectConfig, base_name: str) -> str | None:
    if get_world_size() <= 1:
        return None

    preferred = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("SLURM_JOB_ID")
    if preferred is not None and preferred.strip().lower() in {"", "none", "null", "nil"}:
        preferred = None
    shared_parts = (
        preferred or "",
        os.environ.get("SLURM_STEP_ID", ""),
        os.environ.get("MASTER_ADDR", ""),
        os.environ.get("MASTER_PORT", ""),
        str(get_world_size()),
        str(Path(project_config.train.output_dir).resolve()),
        base_name,
        project_config.train.run_mode,
    )
    raw_key = "|".join(shared_parts)
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]


def resolve_fresh_run_name(project_config: ProjectConfig, base_name: str) -> str:
    launch_key = resolve_distributed_launch_key(project_config, base_name)
    if launch_key is None:
        return timestamped_run_name(base_name)

    coordination_dir = resolve_cache_root(project_config) / ".run_name_coord"
    run_name_path = coordination_dir / f"{launch_key}.txt"
    lock_path = coordination_dir / f"{launch_key}.lock"
    if is_global_zero():
        with file_lock(lock_path):
            run_name = timestamped_run_name(base_name)
            run_name_path.write_text(run_name + "\n")
            return run_name

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        with file_lock(lock_path):
            if run_name_path.exists():
                run_name = run_name_path.read_text().strip()
                if run_name:
                    return run_name
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for global rank 0 to write coordinated run name at {run_name_path}")


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_project_config(project_config: ProjectConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "project_config.json"
    lock_path = run_dir / ".project_config.lock"
    payload = json.dumps(project_config.as_hparams(), indent=2, sort_keys=True) + "\n"
    with file_lock(lock_path):
        config_path.write_text(payload)


def resolve_wandb_run_id(project_config: ProjectConfig, state_dir: Path) -> str:
    if project_config.train.wandb_id is not None:
        return project_config.train.wandb_id

    state_dir.mkdir(parents=True, exist_ok=True)
    id_path = state_dir / "wandb_run_id.txt"
    lock_path = state_dir / ".wandb_run_id.lock"
    with file_lock(lock_path):
        if project_config.train.resume and id_path.exists():
            return id_path.read_text().strip()

        run_id = wandb.util.generate_id()
        id_path.write_text(run_id + "\n")
        return run_id


def resolve_wandb_resume(project_config: ProjectConfig) -> str | None:
    if project_config.train.wandb_offline:
        return None
    if project_config.train.wandb_resume is not None:
        return project_config.train.wandb_resume
    if (
        project_config.train.resume
        and project_config.train.resume_ckpt_path is not None
        and project_config.train.run_mode == "fit"
    ):
        return "must"
    return None


def resolve_checkpoint_global_step(ckpt_path: str | None) -> int | None:
    if ckpt_path is None:
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    global_step = checkpoint.get("global_step")
    if global_step is None:
        return None
    return int(global_step)


def resolve_wandb_resume_from(
    project_config: ProjectConfig,
    wandb_run_id: str,
) -> str | None:
    if project_config.train.wandb_offline:
        return None
    if project_config.train.wandb_resume is not None:
        return None
    if project_config.train.run_mode != "fit" or not project_config.train.resume:
        return None
    global_step = resolve_checkpoint_global_step(project_config.train.resume_ckpt_path)
    if global_step is None:
        return None
    return f"{wandb_run_id}?_step={global_step}"


def build_loggers(project_config: ProjectConfig, run_dir: Path):
    if not is_global_zero():
        return False

    loggers: list = []
    if project_config.train.use_csv_logger:
        loggers.append(
            CSVLogger(
                save_dir=str(run_dir),
                name="csv",
            )
        )
    if project_config.train.use_wandb:
        wandb_state_dir = resolve_wandb_state_dir(project_config)
        wandb_run_id = resolve_wandb_run_id(project_config, wandb_state_dir)
        wandb_save_dir = str(wandb_state_dir)
        wandb_resume_from = resolve_wandb_resume_from(project_config, wandb_run_id)
        wandb_resume = None if wandb_resume_from is not None else resolve_wandb_resume(project_config)
        wandb_init_id = None if wandb_resume_from is not None else wandb_run_id
        logger.info(
            "W&B enabled: project='{}' entity='{}' mode='{}' run_id='{}' save_dir='{}' resume='{}' resume_from='{}'",
            project_config.train.wandb_project,
            project_config.train.wandb_entity or "<default>",
            "offline" if project_config.train.wandb_offline else "online",
            wandb_run_id,
            wandb_save_dir,
            wandb_resume or "<none>",
            wandb_resume_from or "<none>",
        )
        loggers.append(
            WandbLogger(
                project=project_config.train.wandb_project,
                entity=project_config.train.wandb_entity,
                name=project_config.train.wandb_name or project_config.train.run_name,
                group=project_config.train.wandb_group,
                tags=list(project_config.train.wandb_tags),
                notes=project_config.train.wandb_notes,
                save_dir=wandb_save_dir,
                offline=project_config.train.wandb_offline,
                log_model=project_config.train.wandb_log_model,
                id=wandb_init_id,
                resume=wandb_resume,
                resume_from=wandb_resume_from,
                job_type=project_config.train.run_mode,
            )
        )
    if not loggers:
        return False
    for active_logger in loggers:
        active_logger.log_hyperparams(project_config.as_hparams())
    return loggers[0] if len(loggers) == 1 else loggers


def build_callbacks(
    project_config: ProjectConfig,
    run_dir: Path,
    *,
    has_validation: bool,
) -> list:
    callbacks: list = []
    if project_config.train.use_csv_logger or project_config.train.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        callbacks.append(ThroughputCallback(project_config.train.log_every_n_steps))
    if not project_config.train.enable_checkpointing:
        return callbacks

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = project_config.train.checkpoint_monitor if has_validation else None
    filename = "step={step}"
    if monitor == "val_loss":
        filename = "step={step}-val_loss={val_loss:.6f}"
    checkpoint_kwargs = dict(
        dirpath=str(checkpoint_dir),
        filename=filename,
        save_top_k=project_config.train.checkpoint_save_top_k,
        save_last=project_config.train.checkpoint_save_last,
        every_n_train_steps=project_config.train.checkpoint_every_n_train_steps,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    if monitor is not None:
        checkpoint_kwargs["monitor"] = monitor
        checkpoint_kwargs["mode"] = project_config.train.checkpoint_mode
    callbacks.append(ModelCheckpoint(**checkpoint_kwargs))
    return callbacks


def _resolve_train_data_config(project_config: ProjectConfig) -> DataConfig:
    return replace(project_config.data, split="train")


def _resolve_val_data_config(project_config: ProjectConfig) -> DataConfig:
    split = "val" if project_config.data.backend == "ogbench_npz" else project_config.data.split
    return replace(project_config.data, split=split)


def build_data_module(project_config: ProjectConfig, *, mode: str) -> spt.data.DataModule:
    if mode not in {"fit", "validate"}:
        raise ValueError("mode must be one of: fit, validate")

    if mode == "fit":
        train_loader = build_td2_dataloader(_resolve_train_data_config(project_config), shuffle=True)
        val_loader = None
        if project_config.train.limit_val_batches not in (0, 0.0):
            val_loader = build_td2_dataloader(_resolve_val_data_config(project_config), shuffle=False)
        return spt.data.DataModule(train=train_loader, val=val_loader)

    val_loader = build_td2_dataloader(_resolve_val_data_config(project_config), shuffle=False)
    return spt.data.DataModule(val=val_loader)


def has_validation_enabled(project_config: ProjectConfig) -> bool:
    return project_config.train.limit_val_batches not in (0, 0.0)


def build_trainer(
    project_config: ProjectConfig,
    *,
    logger,
    callbacks: list,
    has_validation: bool,
) -> pl.Trainer:
    run_dir = resolve_run_dir(project_config)
    trainer_kwargs = dict(
        default_root_dir=str(run_dir),
        accelerator=project_config.train.accelerator,
        devices=project_config.train.devices,
        precision=project_config.train.precision,
        log_every_n_steps=project_config.train.log_every_n_steps,
        enable_progress_bar=project_config.train.enable_progress_bar,
        enable_checkpointing=project_config.train.enable_checkpointing,
        limit_train_batches=project_config.train.limit_train_batches,
        limit_val_batches=project_config.train.limit_val_batches,
        logger=logger,
        callbacks=callbacks,
    )
    if project_config.train.train_semantics == "paper":
        max_steps = (
            project_config.train.max_steps
            if project_config.train.max_steps is not None
            else resolve_paper_max_steps(project_config.model.policy_mode)
        )
        trainer_kwargs.update(
            max_steps=max_steps,
            max_epochs=-1 if project_config.train.max_epochs is None else project_config.train.max_epochs,
            num_sanity_val_steps=0,
        )
        if has_validation:
            trainer_kwargs["check_val_every_n_epoch"] = None
            trainer_kwargs["val_check_interval"] = (
                project_config.train.val_check_interval
                if project_config.train.val_check_interval is not None
                else max_steps
            )
    else:
        trainer_kwargs["max_epochs"] = (
            project_config.train.max_epochs
            if project_config.train.max_epochs is not None
            else 50
        )
    return pl.Trainer(**trainer_kwargs)


def apply_runtime_acceleration(project_config: ProjectConfig) -> None:
    matmul_precision = project_config.train.matmul_precision
    if matmul_precision is None:
        return
    torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = matmul_precision in {"high", "medium"}
        torch.backends.cudnn.allow_tf32 = matmul_precision in {"high", "medium"}


def resolve_compile_cache_artifact_path(project_config: ProjectConfig) -> Path | None:
    cache_name = project_config.train.compile_cache_name or project_config.data.dataset_name
    return resolve_cache_root(project_config) / "compile" / cache_name / "cache_artifacts.bin"


def resolve_compile_cache_runtime_dir(project_config: ProjectConfig) -> Path | None:
    cache_name = project_config.train.compile_cache_name or project_config.data.dataset_name
    return resolve_cache_root(project_config) / "compile" / cache_name


def resolve_compile_cache_save_artifact_path(project_config: ProjectConfig) -> Path | None:
    return resolve_compile_cache_artifact_path(project_config)


def configure_compile_cache(project_config: ProjectConfig) -> Path | None:
    if not project_config.train.compile:
        return None

    load_artifact_path = resolve_compile_cache_artifact_path(project_config)
    runtime_cache_dir = resolve_compile_cache_runtime_dir(project_config)
    save_artifact_path = resolve_compile_cache_save_artifact_path(project_config)
    if runtime_cache_dir is None:
        return None

    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(runtime_cache_dir.resolve())

    if (
        load_artifact_path is not None
        and load_artifact_path.exists()
        and hasattr(torch.compiler, "load_cache_artifacts")
    ):
        torch.compiler.load_cache_artifacts(load_artifact_path.read_bytes())

    return save_artifact_path


def save_compile_cache(artifact_path: Path | None) -> None:
    if (
        artifact_path is None
        or not hasattr(torch.compiler, "save_cache_artifacts")
        or not is_global_zero()
    ):
        return
    artifacts = torch.compiler.save_cache_artifacts()
    if artifacts is None:
        return
    artifact_bytes, _cache_info = artifacts
    lock_path = artifact_path.parent / ".cache_artifacts.lock"
    with file_lock(lock_path):
        artifact_path.write_bytes(artifact_bytes)


def build_project_config_from_sample(
    entry_config: TrainEntryConfig,
) -> ProjectConfig:
    sample_data_config = replace(entry_config.data, split="train")
    dataloader = build_td2_dataloader(sample_data_config, shuffle=True)
    sample = next(iter(dataloader))
    observation_shape, action_dim, policy_embedding_dim = infer_shapes(sample)
    use_identity_encoder = (
        entry_config.observation_encoder in {"identity", "no_encoder"}
        or (
            entry_config.observation_encoder == "auto"
            and entry_config.policy_mode == "single_policy"
            and len(observation_shape) == 1
        )
    )
    latent_dim = math.prod(observation_shape) if use_identity_encoder else 128
    model_config = ModelConfig(
        observation_shape=observation_shape,
        action_dim=action_dim,
        backbone=entry_config.backbone,
        observation_encoder=entry_config.observation_encoder,
        network_variant=entry_config.network_variant,
        latent_dim=latent_dim,
        policy_embedding_dim=max(policy_embedding_dim, entry_config.policy_embedding_dim),
        gamma=entry_config.gamma,
        polyak=entry_config.polyak,
        direct_loss_weight=entry_config.direct_loss_weight,
        bootstrap_loss_weight=entry_config.bootstrap_loss_weight,
        bootstrap_time_sampling=entry_config.bootstrap_time_sampling,
        bootstrap_time_late_prob=entry_config.bootstrap_time_late_prob,
        bootstrap_time_late_start=entry_config.bootstrap_time_late_start,
        initialization=entry_config.initialization,
        policy_mode=entry_config.policy_mode,
    )
    return ProjectConfig(
        data=entry_config.data,
        model=model_config,
        train=entry_config.train,
    )


def train(project_config: ProjectConfig) -> spt.Manager:
    project_config = normalize_project_config(project_config)
    apply_runtime_acceleration(project_config)
    compile_cache_artifact = configure_compile_cache(project_config)
    run_dir = resolve_run_dir(project_config)
    save_project_config(project_config, run_dir)
    data_module = build_data_module(project_config, mode="fit")
    module = build_training_module(project_config.model, project_config.train)
    has_validation = has_validation_enabled(project_config)
    logger = build_loggers(project_config, run_dir)
    callbacks = build_callbacks(project_config, run_dir, has_validation=has_validation)
    trainer = build_trainer(
        project_config,
        logger=logger,
        callbacks=callbacks,
        has_validation=has_validation,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=project_config.train.seed,
        ckpt_path=project_config.train.resume_ckpt_path,
        compile=project_config.train.compile,
    )
    manager()
    save_compile_cache(compile_cache_artifact)
    return manager


def evaluate(project_config: ProjectConfig) -> list[dict[str, float]]:
    project_config = normalize_project_config(project_config)
    if project_config.train.resume_ckpt_path is None:
        raise ValueError("resume_ckpt_path is required for run_mode=validate")

    apply_runtime_acceleration(project_config)
    compile_cache_artifact = configure_compile_cache(project_config)
    run_dir = resolve_run_dir(project_config)
    save_project_config(project_config, run_dir)
    data_module = build_data_module(project_config, mode="validate")
    module = build_training_module(project_config.model, project_config.train)
    if project_config.train.compile:
        module.compile()
    logger = build_loggers(project_config, run_dir)
    callbacks = build_callbacks(project_config, run_dir, has_validation=True)
    trainer = build_trainer(
        project_config,
        logger=logger,
        callbacks=callbacks,
        has_validation=True,
    )
    pl.seed_everything(project_config.train.seed, workers=True)
    results = trainer.validate(
        module,
        datamodule=data_module,
        ckpt_path=project_config.train.resume_ckpt_path,
    )
    if is_global_zero():
        metrics_path = run_dir / "eval_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    save_compile_cache(compile_cache_artifact)
    if is_global_zero():
        print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main() -> None:
    entry_config = tyro.cli(
        TrainEntryConfig,
        description="Train TD²-CFM on stable_worldmodel or OGBench datasets.",
    )
    project_config = build_project_config_from_sample(entry_config)
    if project_config.train.run_mode == "fit":
        train(project_config)
        return
    if project_config.train.run_mode == "validate":
        evaluate(project_config)
        return
    raise ValueError("train.run_mode must be one of: fit, validate")


if __name__ == "__main__":
    main()
