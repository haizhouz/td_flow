# Repository Guidelines

## Project Structure & Module Organization

- `src/td_flow/`: main package. Core files are `model.py`, `module.py`, `data.py`, `planner.py`, `train.py`, `rollout.py`, and `dataset_stats.py`.
- `src/td_flow/pointmass/`: pointmass loop-policy experiments, mixed-path dataset builders, occupancy plots, and TD2 analysis scripts.
- `src/td_flow/toy/`: toy circle-policy and toy exploration dataset generation, exact successor-measure comparisons, and TD2 diagnostics.
- `tests/`: shared `unittest` coverage for data loading, model architecture, training entrypoints, paths, ODE behavior, and planner interfaces.
- `tests/pointmass/`: focused tests for pointmass utilities and plotting.
- `tests/toy/`: focused tests for toy dataset generation.
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
- Run focused toy exploration dataset tests:
  ```bash
  uv run python -m unittest tests.toy.test_generate_circle_exploration_dataset
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
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --data.dir /home/haizhou/.ogbench/data --train.max-steps 1 --train.run-name smoke
  ```
- Run the same entrypoint with `torch.compile` enabled:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --data.dir /home/haizhou/.ogbench/data --train.run-name smoke --train.compile
  ```
- Run validate-only from a checkpoint:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --data.dir /home/haizhou/.ogbench/data --train.run-mode validate --train.resume-ckpt-path outputs/smoke-20260409-210000/checkpoints/last.ckpt
  ```
- Run the same entrypoint with W&B logging:
  ```bash
  uv run python -m td_flow.train --data.dataset-name cube-single-play-v0 --data.backend ogbench_npz --data.dir /home/haizhou/.ogbench/data --train.run-name smoke --train.use-wandb --train.wandb-project td_flow --train.wandb-offline
  ```
- Render a checkpoint rollout as predicted frames:
  ```bash
  uv run python -m td_flow.rollout --checkpoint-path outputs/cube-single-10k/checkpoints/last.ckpt --split val --horizon 8
  ```
- Print OGBench episode-length stats:
  ```bash
  uv run python -m td_flow.dataset_stats --dataset-name cube-single-play-v0 --dataset-dir /home/haizhou/.ogbench/data --split train
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
- For rollout/rendering changes, add focused coverage in `tests/test_rollout.py` for checkpoint/config loading and dataset-state decoding.
- For dataset parsing utilities, add focused coverage in `tests/test_data.py` for episode-boundary reconstruction and summary statistics.

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
- Dataset roots are passed with `--data.dir`; avoid calling that field `cache` in docs or code because `train.cache-root` is a separate runtime/cache concept.
- For `ogbench_npz`, keep the same public key surface as `stablewm_hdf5`: `--data.observation-key`, `--data.action-key`, `--data.goal-key`, and `--data.policy-embedding-key`. Default aliases like `state -> observations` and `action -> actions` should work transparently.
- Run artifacts are written under `--train.output-dir/--train.run-name`, including `project_config.json`, CSV logs, checkpoints, and `eval_metrics.json`.
- Fresh runs always get a timestamped final run name. `--train.run-name` supplies the base prefix; otherwise the dataset name is used.
- `--train.resume` is the only flag that enables training-state resume. `--train.resume-ckpt-path` only points to a checkpoint; by itself it does not imply resume for `fit`.
- In `fit` mode, `--train.resume` without `--train.resume-ckpt-path` resumes the latest checkpoint from the resolved run directory.
- In `validate` mode, `--train.resume-ckpt-path` is required and determines the run directory unless an exact matching `--train.run-name` is provided.
- CSV logging is enabled by default; W&B is optional through `--train.use-wandb`.
- `--train.cache-root` is the shared root for local runtime/cache files such as compile artifacts and W&B local state.
- For distributed fresh launches, ranks coordinate a single shared timestamped run name under `cache_root/.run_name_coord/`.
- Multi-node runs require `--train.cache-root` and `--train.output-dir` to be on a shared filesystem if you want coordinated run names, checkpoints, and W&B local state.
- When W&B is enabled, `wandb_run_id.txt` is stored under the cache root by default. Only explicit `--train.resume` runs reuse that ID; resumed online runs continue from the checkpoint `global_step`.
- Only global rank 0 initializes local loggers, writes `eval_metrics.json`, and saves compile cache artifacts.
- Checkpointing is enabled by default. `--train.resume` turns checkpoint resume on for `fit`, and `--train.resume-ckpt-path` optionally selects a specific checkpoint. During resume, `--train.run-name` is treated as an exact run name or prefix selector. `validate` still loads from `--train.resume-ckpt-path`.
- `--train.log-every-n-steps` controls both metric logging cadence and the `train/fps` throughput metric.
- `--train.enable-progress-bar` defaults to `False` so long runs rely on CSV/W&B instead of a noisy live terminal bar.
- `--train.compile` is a boolean switch; use `--train.compile`, not `--train.compile true`.
- Compiled runs persist cache files under `--train.cache-root/compile/<dataset_name>/` by default so repeated runs on the same dataset can reuse compatible compile artifacts.
- Use `--train.compile-cache-name` only when you want to override that default namespace explicitly.
- `td_flow.rollout` currently targets OGBench `cube-single-play-v0` with `ogbench_npz` checkpoints and identity observation encoding. It renders an initial seeded state plus the predicted trajectory, and its default output path is `<checkpoint_dir>/rollout/`; update docs/tests if that support surface changes.
- `td_flow.dataset_stats` reconstructs trajectories from flat OGBench transitions via the `terminals` array; keep that interpretation consistent with the loaders.
- Off-policy TD2 runs are controlled by `--data.action-key` and `--data.next-action-key`; dataset-only behavior baselines should use `next_action_key=action`, while policy-conditioned off-policy runs should use the relabeled target action key such as `policy_action`.
- `--state-only-conditioning` zeros action inputs before context encoding. Use it only for explicit state-only ablations; it changes the learned object from `m(.|s,a)` to a state-only approximation.
- `--one-step-prediction-loss-weight` adds an auxiliary latent next-state prediction loss through the model's `one_step_head`.
- `--loss-weight-schedule`, `--loss-weight-warmup-steps`, and `--loss-weight-ramp-steps` control direct/bootstrap weighting schedules. The current custom schedule is `direct_warmup_linear`.
- Toy exploration HDF5 datasets generated by `td_flow.toy.generate_circle_exploration_dataset` store both `action` and `policy_action`, and the sibling JSON stores explicit `behavior_policy` and `target_policy` metadata. Keep these in sync if you extend the toy behavior-policy surface.
- `td_flow.toy.measure_circle_density_metrics` is the density-aware evaluator for toy-circle successor measures. Use it when support overlap on the circle makes nearest-neighbor occupancy comparisons misleading.
- Pointmass mixed-loop datasets generated by `td_flow.pointmass.generate_policy_dataset` may store `policy_mode_id`; preserve that key if you extend the straight/circle path mixture tooling.
- `.venv/`, `.pyc`, `__pycache__/`, and generated environment dumps should remain untracked.
