# Repository Guidelines

## Project Structure & Module Organization

- `src/td_flow/`: main package. Core files are `model.py`, `module.py`, `data.py`, `planner.py`, and `train.py`.
- `tests/`: `unittest` smoke tests for data loading, model architecture, training entrypoints, paths, ODE behavior, and planner interfaces.
- `doc/`: paper, implementation plan, and backbone notes.
- `README.md`: setup, training commands, and the `tyro` config tutorial.

## Build, Test, and Development Commands

- Create or refresh the local env:
  ```bash
  uv venv --python 3.10 --seed
  uv sync
  ```
- Run unit tests:
  ```bash
  uv run python -m unittest discover -s tests
  ```
- Run a syntax smoke check:
  ```bash
  uv run python -m compileall src tests
  ```
- Show the generated config CLI:
  ```bash
  uv run python -m td_flow.train --help
  ```
- Run a checkpointed smoke train:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --train.max-steps 1 --train.run-name smoke
  ```
- Run the same entrypoint with `torch.compile` enabled:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --train.run-name smoke --train.compile
  ```
- Run validate-only from a checkpoint:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --train.run-mode validate --train.resume-ckpt-path outputs/smoke/checkpoints/last.ckpt
  ```

## Coding Style & Naming Conventions

- Use Python 3.10-compatible code.
- Follow PEP 8 with 4-space indentation.
- Prefer small, explicit functions over implicit framework magic.
- Use `snake_case` for functions/files, `PascalCase` for classes, and clear config dataclass names such as `ModelConfig`.
- Keep interfaces aligned with installed backbones:
  - `stable_pretraining.Module.forward(self, batch, stage)`
  - `stable_worldmodel` planner models exposing `get_cost(info_dict, action_candidates)`

## Testing Guidelines

- Test framework: standard library `unittest`.
- Add focused tests in `tests/test_<area>.py`.
- Prioritize shape, interface, and smoke coverage before long training runs.
- New planner or dataset code should include at least one tensor-shape assertion test.
- For training-loop changes, cover checkpoint, resume, or validate-only behavior in `tests/test_train.py` when possible.

## Commit & Pull Request Guidelines

- Match the existing commit style: short imperative subject lines, e.g. `Add uv-based project environment`.
- Keep commits scoped: environment, docs, model scaffold, and training changes should be separable when practical.
- PRs should include:
  - what changed
  - why it changed
  - how it was verified
  - any dataset or environment assumptions

## Configuration Notes

- Use `uv` and the repo-local `.venv`; do not rely on the base environment.
- Seed `pip` into the virtualenv, because `stable_pretraining` calls `python -m pip freeze` during environment dumps.
- Training config is nested under `--data.*`, `--train.*`, and `--backbone.*` because the entrypoint uses `tyro`.
- Run artifacts are written under `--train.output-dir/--train.run-name`, including `project_config.json`, CSV logs, checkpoints, and `eval_metrics.json`.
- CSV logging is enabled by default; W&B is optional through `--train.use-wandb`.
- Checkpointing is enabled by default; resume and validate-only runs use `--train.resume-ckpt-path`.
- `--train.compile` is a boolean switch; use `--train.compile`, not `--train.compile true`.
- Compiled runs persist cache files under `--train.compile-cache-dir/<dataset_name>/` by default so repeated runs on the same dataset can reuse compatible compile artifacts.
- Use `--train.compile-cache-name` only when you want to override that default namespace explicitly.
- `.venv/`, `.pyc`, `__pycache__/`, and generated environment dumps should remain untracked.
