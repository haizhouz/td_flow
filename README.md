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
  --train.max-epochs 1 \
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
--data.observation-key state
--train.use-wandb
--train.wandb-project td_flow
--backbone.kind mlp
--backbone.hidden-dims 256 256
```

The model shape fields are inferred from the first batch, so you do not set `observation_shape` or `action_dim` on the CLI.

## Development

Common commands:

```bash
uv run python -m compileall src tests
uv run python -m unittest discover -s tests
uv lock
```
