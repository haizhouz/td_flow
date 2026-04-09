# Repository Guidelines

## Project Structure & Module Organization

- `src/td_flow/`: main package. Core files are `model.py`, `module.py`, `data.py`, `planner.py`, and `train.py`.
- `tests/`: `unittest` smoke tests for paths, ODE behavior, and planner interfaces.
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
- `.venv/`, `.pyc`, `__pycache__/`, and generated environment dumps should remain untracked.
