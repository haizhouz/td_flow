# TD-Flow

TD²-CFM training and planning scaffold built on `stable_pretraining` and `stable_worldmodel`.

## Environment

Create or update the repo-local `uv` environment:

```bash
uv venv --python 3.10 --seed
uv sync
```

`--seed` matters here because `stable_pretraining` records the environment with `python -m pip freeze`.

## Training

The training entrypoint uses `tyro`, so CLI flags map directly onto nested dataclass fields.

Show the generated help:

```bash
uv run python -m td_flow.train --help
```

Run a small OGBench smoke test:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --data.batch-size 64 \
  --data.num-workers 0 \
  --train.output-dir outputs \
  --train.run-name cube-single-smoke \
  --train.max-steps 1 \
  --train.limit-train-batches 1 \
  --train.limit-val-batches 1
```

For `ogbench_npz`, the adapter now honors the same key-based interface as the HDF5 backend:

- `--data.observation-key`
- `--data.action-key`
- `--data.goal-key`
- `--data.policy-embedding-key`

Default aliases are resolved for OGBench automatically:

- `state` or `observation` -> `observations`
- `action` -> `actions`
- next observation -> `next_observations`

So later RGB/custom-key datasets can reuse the same CLI surface without a separate loader interface.

### Key Examples

Default OGBench state/action aliases:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --data.observation-key state \
  --data.action-key action
```

Equivalent explicit OGBench raw keys:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --data.observation-key observations \
  --data.action-key actions
```

Custom RGB-style observation key:

```bash
uv run python -m td_flow.train \
  --data.dataset-name my-cube-rgb-v0 \
  --data.backend ogbench_npz \
  --data.dir /path/to/dataset \
  --data.observation-key pixels \
  --data.action-key actions \
  --observation-encoder learned
```

Custom goal key:

```bash
uv run python -m td_flow.train \
  --data.dataset-name my-goal-dataset-v0 \
  --data.backend ogbench_npz \
  --data.dir /path/to/dataset \
  --data.observation-key state \
  --data.action-key action \
  --data.goal-key target
```

Custom policy embedding key:

```bash
uv run python -m td_flow.train \
  --data.dataset-name my-multipolicy-dataset-v0 \
  --data.backend ogbench_npz \
  --data.dir /path/to/dataset \
  --data.observation-key state \
  --data.action-key action \
  --data.policy-embedding-key policy_z \
  --policy-mode multi_policy \
  --policy-embedding-dim 128
```

The same key interface also applies to `stablewm_hdf5`:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend stablewm_hdf5 \
  --data.dir /path/to/stablewm/cache \
  --data.observation-key pixels \
  --data.action-key action \
  --data.goal-key goal
```

This writes:

- `outputs/<run_name>/project_config.json`
- CSV logs under `outputs/<run_name>/csv/`
- checkpoints under `outputs/<run_name>/checkpoints/`

## Rollout Visualization

Use the rollout entrypoint to render a checkpoint on OGBench `cube-single-play-v0` and save an autoregressive predicted trajectory seeded from one dataset state.

Example:

```bash
uv run python -m td_flow.rollout \
  --checkpoint-path outputs/cube-single-10k/checkpoints/last.ckpt \
  --split val \
  --horizon 8
```

By default this writes a `rollout/` directory inside the checkpoint folder, for example `outputs/<run_name>/checkpoints/rollout/`, with:

- `frames/frame_000.png`, ...
- `predicted_rollout.gif`
- `rollout_config.json`

Important limits for the current script:

- only `ogbench_npz` checkpoints are supported
- only `cube-single-play-v0` is supported
- only `identity` / `no_encoder` observation encoders are supported

You can override the destination or choose a deterministic start:

```bash
uv run python -m td_flow.rollout \
  --checkpoint-path outputs/cube-single-10k/checkpoints/last.ckpt \
  --output-dir /tmp/cube-rollout \
  --start-index 100 \
  --horizon 12
```

## Pointmass And Toy Utilities

Pointmass-specific experiment scripts now live under `td_flow.pointmass` and toy-circle helpers live under `td_flow.toy`.

Examples:

```bash
uv run python -m td_flow.pointmass.plot_policy_conditioned_occupancy --help
uv run python -m td_flow.pointmass.analyze_td2_failure --help
uv run python -m td_flow.toy.generate_circle_policy_dataset --help
uv run python -m td_flow.toy.plot_circle_policy_conditioned_occupancy --help
```

Pointmass helpers now also cover mixed loop-path datasets and off-policy diagnostics:

```bash
uv run python -m td_flow.pointmass.generate_policy_dataset --help
uv run python -m td_flow.pointmass.compare_bootstrap_target_occupancy --help
uv run python -m td_flow.pointmass.compare_action_conditioned_successors --help
```

`td_flow.pointmass.generate_policy_dataset` supports multiple scripted path modes per episode. For example, the current mixed dataset can alternate between the straight/diamond loop and the circular loop and stores the per-step mode in `policy_mode_id`.

Toy-circle helpers now include an exploration/off-policy benchmark with exact successor-measure references:

```bash
uv run python -m td_flow.toy.generate_circle_exploration_dataset --help
uv run python -m td_flow.toy.plot_circle_exploration_exact_comparison --help
uv run python -m td_flow.toy.measure_circle_density_metrics --help
```

`td_flow.toy.generate_circle_exploration_dataset` writes:

- `action`: behavior-policy action
- `policy_action`: target-policy action
- metadata JSON with explicit `behavior_policy` and `target_policy`

Supported behavior policies are:

- `uniform_random`
- `disjoint_uniform`
- `constant_delta_theta`

The exact toy comparison script evaluates:

- dataset-only models against exact `m^mu`
- off-policy models against exact `m^pi`

The density-metric script complements the support-style nearest-neighbor metric with angle-histogram total variation, which is the right diagnostic when two policies share the same circle support but induce different discounted densities.

### Toy Off-Policy Example

Generate a toy exploration dataset with a fixed behavior policy and relabeled target policy:

```bash
uv run python -m td_flow.toy.generate_circle_exploration_dataset \
  --output-hdf5-path data/stablewm_cache/toy-circle-exorl-opposite-policy.h5 \
  --behavior-policy-kind constant_delta_theta \
  --behavior-delta-theta -0.02 \
  --policy-delta-theta 0.02 \
  --overwrite
```

Train a dataset-only model:

```bash
uv run python -m td_flow.train \
  --data.dataset-name toy-circle-exorl-opposite-policy \
  --data.backend stablewm_hdf5 \
  --data.dir /home/haizhou/Documents/td_flow/data/stablewm_cache \
  --data.observation-key observation \
  --data.action-key action \
  --data.next-action-key action \
  --observation-encoder identity \
  --network-variant paper \
  --gamma 0.99
```

Train an off-policy model on the same dataset:

```bash
uv run python -m td_flow.train \
  --data.dataset-name toy-circle-exorl-opposite-policy \
  --data.backend stablewm_hdf5 \
  --data.dir /home/haizhou/Documents/td_flow/data/stablewm_cache \
  --data.observation-key observation \
  --data.action-key action \
  --data.next-action-key policy_action \
  --observation-encoder identity \
  --network-variant paper \
  --gamma 0.99
```

Compare both checkpoints against exact toy successor measures:

```bash
uv run python -m td_flow.toy.plot_circle_exploration_exact_comparison \
  --dataset-only-checkpoint-path outputs/toy-circle-dataset/checkpoints/last.ckpt \
  --offpolicy-checkpoint-path outputs/toy-circle-offpolicy/checkpoints/last.ckpt
uv run python -m td_flow.toy.measure_circle_density_metrics \
  --dataset-only-checkpoint-path outputs/toy-circle-dataset/checkpoints/last.ckpt \
  --offpolicy-checkpoint-path outputs/toy-circle-offpolicy/checkpoints/last.ckpt
```

Fresh runs always use a timestamped run name. If you do not set `--train.run-name`, the base name is the dataset name and the final run looks like `cube-single-play-v0-20260409-210000`. If you set `--train.run-name cube-single-smoke`, the final run name becomes `cube-single-smoke-20260409-210000`.

Resume a run from the latest checkpoint:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --train.output-dir outputs \
  --train.run-name cube-single-smoke \
  --train.resume
```

`--train.resume-ckpt-path` is optional in `fit` mode when `--train.resume` is set; if omitted, the trainer uses the latest checkpoint from the selected run directory. When `--train.run-name` is provided during resume, it is treated as an exact run name or a prefix to resolve the latest matching timestamped run.

## Distributed Training

The current distributed workflow follows standard DDP semantics:

- `--data.batch-size` is the per-rank local batch size, not the global batch size.
- Effective global batch size is approximately:
  - `data.batch_size * world_size`
- If you keep `--data.batch-size` fixed and add GPUs, you are increasing the effective batch size.

Examples:

- 1 GPU with `--data.batch-size 1024`:
  - effective batch size `1024`
- 4 GPUs with `--data.batch-size 256`:
  - effective batch size `1024`
- 4 GPUs with `--data.batch-size 1024`:
  - effective batch size `4096`

Fresh distributed launches coordinate a single shared timestamped run name across ranks. Resume is still explicit:

- fresh run:
  - no `--train.resume`
- resumed run:
  - `--train.resume`
  - optional `--train.resume-ckpt-path`

Distributed caveats:

- `--train.cache-root` and `--train.output-dir` should be on a shared filesystem for multi-node runs.
- Only global rank 0 writes:
  - local W&B state
  - `eval_metrics.json`
  - compile cache artifacts
- Validation in distributed mode is supported, but only rank 0 writes the final validation artifact.

Enable `torch.compile` for training or resumed training with the same switch-style flag:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --train.output-dir outputs \
  --train.run-name cube-single-smoke \
  --train.compile
```

Cached runtime files now default under `.cache/td_flow/`. Compile artifacts live under `.cache/td_flow/compile/`, and local W&B files live under `.cache/td_flow/<run_name>/wandb/`. Override or disable that with:

```bash
--train.cache-root /scratch/$USER/td_flow_cache
--train.compile-cache-name my-shared-cache
```

By default, the compile-cache namespace uses `data.dataset_name`, so repeated runs on the same dataset reuse the same compiled artifacts unless you override `--train.compile-cache-name`.
If a compatible compile artifact is present, the training entrypoint will preload it before compiling and save updated artifacts back after the run.

Enable W&B logging:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --train.output-dir outputs \
  --train.run-name cube-single-wandb \
  --train.use-wandb \
  --train.wandb-project td_flow \
  --train.wandb-offline
```

When W&B is enabled, local W&B files and `wandb_run_id.txt` are stored under the cache root by default. Only explicit `--train.resume` runs reuse that ID; fresh runs with the same `run_name` generate a new W&B run id. Resumed online runs continue from the checkpoint `global_step`.

Run checkpointed validation only:

```bash
uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --train.run-mode validate \
  --train.output-dir outputs \
  --train.run-name cube-single-smoke \
  --train.resume-ckpt-path outputs/cube-single-smoke/checkpoints/last.ckpt
```

For OGBench, validate mode uses the dataset's `val` split automatically and writes `eval_metrics.json` into the run directory.

## Dataset Stats

To inspect OGBench episode counts and trajectory lengths:

```bash
uv run python -m td_flow.dataset_stats \
  --dataset-name cube-single-play-v0 \
  --dataset-dir /home/haizhou/.ogbench/data \
  --split train
```

This prints JSON with:

- `num_transitions`
- `num_episodes`
- `min_length`
- `max_length`
- `mean_length`

## Config Tutorial

`tyro` exposes the nested config structure directly:

- `--data.*` controls dataset loading and batch construction.
- `--train.*` controls Lightning trainer settings and W&B logging.
- `--train.*` also controls CSV logging, checkpointing, resume, and validate-only runs.
- `--backbone.*` controls the encoder backbone used to build the TD²-CFM model.

Examples:

```bash
--policy-mode single_policy
--observation-encoder identity
--network-variant paper
--policy-embedding-dim 128
--data.observation-key state
--train.use-wandb
--train.wandb-project td_flow
--train.output-dir outputs
--train.run-name cube-single-paper
--train.resume
--train.resume-ckpt-path outputs/cube-single-paper/checkpoints/last.ckpt
--backbone.kind mlp
--backbone.hidden-dims 256 256
```

The model shape fields are inferred from the first batch, so you do not set `observation_shape` or `action_dim` on the CLI.

`--network-variant` selects the flow network family:

- `repo`: the current repo default FiLM residual U-Net approximation
- `paper`: the Table 5 architecture path
  - conditional encoder MLP: `(512, 512, 512)` single-policy or `(1024, 1024, 1024)` multi-policy
  - time embedding MLP: `(256, 256)`
  - U-Net block widths: `(512, 512, 512)` single-policy or `(1024, 1024, 1024)` multi-policy

`--observation-encoder` controls how observations are represented before the flow model:

- `auto`: use the repo default for the selected policy mode
- `identity`: use an identity observation encoder
- `learned`: use the learned backbone encoder
- `no_encoder`: alias for `identity`

Training defaults now follow paper-style step semantics:

- `--train.train-semantics paper` uses `max_steps` and step-based optimizer scheduling
- `--train.max-steps` is the primary training horizon
- `--train.max-epochs` is only for overriding Lightning behavior during debugging
- `--train.devices` defaults to `auto`, so Lightning uses the visible devices unless you override it
- in distributed training, `--data.batch-size` remains per-rank; adjust it manually if you want to preserve a target global batch size
- `--train.run-mode fit|validate` selects training or checkpointed validation
- `--train.use-csv-logger` defaults to `True`
- `--train.enable-checkpointing` defaults to `True`
- `--train.checkpoint-monitor` defaults to `val_loss`
- `--train.run-name` acts as a base name for fresh runs and an exact-name/prefix selector for resume
- `--train.resume` explicitly enables checkpoint resume for `fit`
- `--train.resume-ckpt-path` points to a specific checkpoint; for `fit`, it is only used when `--train.resume` is set
- `--train.log-every-n-steps` controls logger cadence and the `train/fps` throughput metric
- `--train.enable-progress-bar` defaults to `False` to avoid terminal corruption when Lightning, loguru, and W&B all write concurrently
- `--train.compile` enables `torch.compile` for both `fit` and checkpointed `validate`
- `--train.cache-root` is the shared root for local runtime/cache files, including compile caches and W&B local state
- `--train.compile-cache-name` overrides the default dataset-name cache namespace
- `--train.use-wandb` enables W&B logging in addition to CSV logging
- `--train.wandb-offline` keeps runs local; in offline mode W&B resume is intentionally disabled
- `--train.wandb-id`, `--train.wandb-resume`, `--train.wandb-group`, `--train.wandb-tags`, and `--train.wandb-notes` expose the common W&B run controls

When the corresponding field is left unset, paper defaults are resolved by `policy_mode`:

- single-policy:
  - `weight_decay = 1e-3`
  - `max_steps = 3_000_000`
  - target EMA coefficient `polyak = 0.999`
- multi-policy:
  - `weight_decay = 1e-2`
  - `max_steps = 8_000_000`
  - target EMA coefficient `polyak = 0.9999`

## Development

Common commands:

```bash
uv run python -m compileall src tests
uv run python -m unittest discover -s tests
uv lock
```

## SLURM

Launch through `srun` and keep the Python entrypoint unchanged:

```bash
srun \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --network-variant paper
```

For multi-GPU jobs, either rely on `--train.devices auto` with the allocated GPUs or set an explicit count such as `--train.devices 4`.

To preserve a paper-style global batch size across multiple GPUs, divide the local batch size by GPU count yourself. Example for 4 GPUs and target global batch `1024`:

```bash
srun \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:4 \
  uv run python -m td_flow.train \
  --data.dataset-name cube-single-play-v0 \
  --data.backend ogbench_npz \
  --data.dir /home/haizhou/.ogbench/data \
  --data.batch-size 256 \
  --train.devices 4
```
