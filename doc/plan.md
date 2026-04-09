# TD-Flow Reproduction Plan

## Goal

Reproduce the paper’s proposed TD-Flow method from `doc/farebrother25a.pdf`, specifically `TD²-CFM`, on top of:

- `stable_pretraining` as the model/training backbone
- `stable_worldmodel` as the environment, dataset, planning, and evaluation backbone

This plan assumes the repo will become an integration project rather than a from-scratch standalone implementation. The scope is intentionally narrowed to the single strongest flow-matching variant in the paper instead of reproducing all ablations up front.

## Current Status

The repo now has a working TD$^2$-CFM scaffold with:

- `tyro` configuration
- `uv` environment management
- OGBench state-space smoke coverage
- optional `policy_embedding` input
- `network_variant=paper` for the paper-width MLP U-Net path
- paper-style step-based trainer semantics through `max_steps`

What remains is paper-alignment cleanup and broader benchmark coverage, not first-pass scaffolding.

## What I Verified

### `stable_pretraining`

Based on the public docs:

- It is a Lightning-based PyTorch framework.
- Data is expected to flow as dictionaries.
- The core training stack is:
  - dataset / dataloader
  - `spt.Module`
  - custom or built-in `forward(...)`
  - `spt.Manager`
  - callbacks for monitoring
- It exposes reusable backbones under `stable_pretraining.backbone`.

Relevant doc points:

- Docs: <https://rbalestr-lab.github.io/stable-pretraining/>
- The docs explicitly describe `DataModule`, `Module`, `Manager`, callback-based monitoring, and dictionary-shaped batches.

### `stable_worldmodel`

Based on the public docs:

- It provides the environment-facing layer:
  - `World`
  - dataset recording
  - `HDF5Dataset`
  - policy wrappers
  - MPC planning through `WorldModelPolicy`
  - solvers such as `CEMSolver`
- Recommended evaluation is `world.evaluate_from_dataset(...)`.
- A planning-compatible world model only needs to implement `get_cost(info, action_candidates)`.

Relevant doc points:

- Docs: <https://galilai-group.github.io/stable-worldmodel/>
- Quickstart: <https://galilai-group.github.io/stable-worldmodel/quick_start/>
- API pages used:
  - `World`
  - `Policy / PlanConfig / WorldModelPolicy`
  - `Solver / CEMSolver`
  - `Dataset / HDF5Dataset`

## Integration Strategy

### High-level split of responsibilities

Use `stable_pretraining` for:

- encoder / backbone construction
- TD-Flow training loop packaging
- optimizer / scheduler setup
- logging and online diagnostics
- checkpointing

Use `stable_worldmodel` for:

- environment instantiation
- offline dataset collection and loading
- goal-conditioned evaluation
- MPC planning using the trained TD-Flow model

### Why this split is the right one

The TD-Flow paper is fundamentally a training method for generative long-horizon prediction. `stable_pretraining` is the cleaner place to host:

- the learnable representation backbone
- the vector-field model
- the custom TD losses
- target-network updates

`stable_worldmodel` is the cleaner place to host:

- trajectories
- sequence datasets
- world interaction
- planning-time rollouts and evaluation

That keeps the reproduction aligned with each library’s intended API instead of forcing one library to do both jobs.

## Proposed Architecture

### 1. Data path

1. Use `stable_worldmodel.World(...)` to collect or load data from a supported benchmark.
2. Store offline data in `HDF5Dataset`.
3. Wrap samples into dictionary batches compatible with `stable_pretraining`.
4. Each training sample should expose at minimum:
   - `obs`
   - `action`
   - `next_obs`
   - optional `goal` or latent target fields
   - any mask / episode boundary metadata needed for bootstrapping

### 2. Representation backbone

Build the TD-Flow encoder on `stable_pretraining.backbone`:

- image observations:
  - start with a torchvision-style encoder from `stable_pretraining.backbone.from_torchvision(...)`
- low-dimensional observations:
  - start with `stable_pretraining.backbone.MLP`

The backbone output should be an encoded state, denoted `s` in the math notes.

### 3. TD-Flow model head

On top of the backbone, implement:

- a transition-conditioned vector-field network
- a noise sampler for `x0 ~ N(0, I)`
- a midpoint ODE sampler for the bootstrap path
- support for the paper’s main target:
  - `td2_cfm`

The paper details to preserve:

- Gaussian source distribution `m0 = N(0, I)`
- Gaussian linear conditional path for the standard CFM branch
- midpoint ODE sampling with 10 steps
- target network with Polyak averaging
- the `TD²-CFM` bootstrap target, which uses the previous vector field directly as the regression target
- single-policy input `s, a`
- multi-policy input `s, a, z` represented here as an optional `policy_embedding`
- step-based training semantics rather than an epoch-first loop

### 4. Training wrapper

Implement TD-Flow as a custom `stable_pretraining.Module`:

- define a custom `forward(self, batch, stage)` that:
  - encodes the current state / action context
  - samples `t`
  - constructs the direct term
  - constructs the bootstrap term using the target network
  - computes the `TD²-CFM` loss
  - returns:
    - `"loss"`
    - useful diagnostics like `"loss_direct"`, `"loss_bootstrap"`, `"latent_norm"`, `"vf_norm"`

Then run it through:

- `spt.data.DataModule`
- `spt.Manager`
- Lightning `Trainer`

Current implementation note:

- the trainer now defaults to paper-style step semantics via `max_steps`
- validation is optional and disabled by default unless explicitly requested
- `network_variant=paper` selects the paper-width architecture path

### 5. Planning-time adapter

Implement a small adapter around the trained TD-Flow model so it satisfies the `stable_worldmodel` planning interface:

- required method:
  - `get_cost(info, action_candidates) -> torch.Tensor`

This adapter will:

1. encode the current observation from `info`
2. evaluate candidate action sequences under the TD-Flow latent predictor
3. score them against goal distance, task reward proxy, or success objective
4. return per-sample costs for `CEMSolver` or other solvers

Then plug it into:

- `WorldModelPolicy`
- `PlanConfig`
- `CEMSolver`

## Proposed Repo Layout

```text
doc/
  farebrother25a.pdf
  plan.md

src/
  td_flow/
    __init__.py
    config.py
    data.py
    paths.py
    ode.py
    model.py
    losses.py
    module.py
    target.py
    planner.py
    train.py
    eval.py

tests/
  test_paths.py
  test_ode.py
  test_losses.py
  test_planner_adapter.py
```

## Execution Plan

### Phase 1. Environment and dependency wiring

Add project dependencies and confirm versions:

- `stable-pretraining`
- `stable-worldmodel`
- `torch`
- `lightning`

Deliverables:

- `pyproject.toml`
- import smoke test

### Phase 2. Dataset bridge

Build a bridge from `stable_worldmodel.data.HDF5Dataset` into the dictionary format expected by `stable_pretraining`.

Deliverables:

- sequence dataset wrapper
- collate logic if needed
- shape / dtype tests

### Phase 3. Backbone and vector field

Implement the encoder + TD-Flow head:

- observation encoder from `stable_pretraining.backbone`
- action conditioning
- time embedding
- vector-field network
- target-network copy / EMA update

Current status:

- `network_variant=paper` follows Table 5 for the conditional encoder, time-embedding MLP, and U-Net stage widths

Deliverables:

- model module
- target model utilities
- forward-shape tests

### Phase 4. TD-Flow loss

Implement the paper’s main flow-matching objective:

1. `TD²-CFM`

Deliverables:

- `TD²-CFM` loss implementation
- target-vector-field bootstrapping logic
- tests that confirm the `TD²-CFM` targets and bootstrap path are built correctly

### Phase 5. Training pipeline on `stable_pretraining`

Implement:

- custom `spt.Module`
- datamodule wiring
- callbacks for monitoring

Recommended early callbacks:

- loss logging
- latent norm / variance monitoring
- optional embedding-quality probe if we expose supervised labels or goals

Deliverables:

- runnable training script
- short smoke config

### Phase 6. Planning adapter on `stable_worldmodel`

Implement the planning-time `get_cost(...)` wrapper and connect it to:

- `CEMSolver`
- `WorldModelPolicy`
- `PlanConfig`

Deliverables:

- planning adapter
- evaluation script using `world.evaluate_from_dataset(...)`

### Phase 7. Benchmark reproduction

Start with one small benchmark before scaling:

- preferred first target: a low-dimensional or lightweight environment
- then scale to pixel tasks if the backbone path is stable

Deliverables:

- baseline config
- `TD²-CFM` reproduction run
- basic plots or tabular metrics

## Key Implementation Decisions

### Use latent-space TD-Flow, not raw-pixel vector fields

Raw pixel-space TD-Flow would be expensive and unnecessary for the first working reproduction. The cleaner path is:

- encode observation `obs` to an encoded state `s`
- run TD-Flow in that encoded-state space
- use the encoded-state predictor for planning cost estimation

This is also the natural point where `stable_pretraining` adds value.

### Keep `stable_worldmodel` as the outer control loop

Do not reimplement:

- environment management
- HDF5 dataset handling
- MPC solvers
- evaluation loops

Those are already the point of `stable_worldmodel`.

### Keep `stable_pretraining` as the training/runtime shell

Do not build a custom trainer first. Use:

- `spt.Module`
- `spt.Manager`
- Lightning trainer

That will keep the repo maintainable and closer to the user’s requested backbone.

## Why Only TD²-CFM

The paper’s own analysis and experiments make `TD²-CFM` the right initial target:

- it has lower gradient variance than `TD-CFM` and `TD-CFM(C)`
- it is the strongest overall flow variant in the reported results
- it remains effective under non-straight path geometries where `TD-CFM(C)` degrades

The other two methods are best treated as later ablations, not part of the first implementation target.

## First Concrete Milestone

The first milestone should be:

1. Load a `stable_worldmodel` dataset.
2. Train a latent `TD²-CFM` predictor with a `stable_pretraining` backbone.
3. Expose it as a `get_cost(...)` world model.
4. Run `CEMSolver` through `WorldModelPolicy`.
5. Evaluate with `world.evaluate_from_dataset(...)`.

If that works end to end, the repo will already have a real TD-Flow reproduction path rather than only isolated model code.

## Risks / Unknowns

### 1. API drift

Both libraries appear active. Their latest public docs may differ slightly from the installed version we end up using. We should pin versions early.

### 2. TD-Flow paper mismatch with world-model API

TD-Flow predicts discounted future-state distributions, while `stable_worldmodel` planners expect a `get_cost(...)` interface. We will need an explicit adapter from TD-Flow predictions to planning cost.

### 3. Evaluation target definition

The planning objective depends on environment/task:

- goal-image distance
- state distance
- learned reward proxy

We need to choose one benchmark-compatible cost first.

### 4. Backbone choice

For the first runnable version, the backbone should be simple. A large vision backbone can wait until the latent training path is stable.

## Recommended Next Step

Implement the repo in this order:

1. dependency scaffold
2. dataset bridge
3. latent `TD²-CFM` model
4. planning adapter
5. one environment end-to-end smoke run

That is the shortest path to a useful reproduction using the two requested backbones.
