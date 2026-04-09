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
  --data.cache-dir /home/haizhou/.ogbench/data \
  --data.batch-size 64 \
  --data.num-workers 0 \
  --train.max-steps 1 \
  --train.limit-train-batches 1 \
  --train.limit-val-batches 1
```

## Config Tutorial

`tyro` exposes the nested config structure directly:

- `--data.*` controls dataset loading and batch construction.
- `--train.*` controls Lightning trainer settings and W&B logging.
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
--backbone.kind mlp
--backbone.hidden-dims 256 256
```

The model shape fields are inferred from the first batch, so you do not set `observation_shape` or `action_dim` on the CLI.

`--network-variant` selects the flow network family:

- `repo`: the current repo default FiLM residual U-Net approximation
- `paper`: the paper-width U-Net MLP with Table 5 widths (`512` single-policy, `1024` multi-policy)

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
  --data.cache-dir /home/haizhou/.ogbench/data \
  --network-variant paper
```

For multi-GPU jobs, either rely on `--train.devices auto` with the allocated GPUs or set an explicit count such as `--train.devices 4`.
