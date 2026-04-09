# Backbone Notes

Implementation notes for integrating TD-Flow with:

- `stable_pretraining`
- `stable_worldmodel`

These notes are intentionally operational. They focus on the APIs and conventions that matter while building the repo.

## stable_pretraining

Docs:

- <https://rbalestr-lab.github.io/stable-pretraining/>

### Core mental model

`stable_pretraining` is the training shell, not the environment shell.

The library is organized around four pieces:

1. dictionary-shaped data
2. `spt.Module`
3. callbacks
4. Lightning `Trainer` wrapped by `spt.Manager`

### Data contract

Important rule: each sample should be a dictionary with named fields.

Examples from the docs use structures like:

- `{"image": ..., "label": ...}`

Implication for this repo:

- any `stable_worldmodel` dataset sample must be converted into a dictionary that TD-Flow can consume directly
- expected keys for us will likely be:
  - `obs`
  - `action`
  - `next_obs`
  - optional `goal`
  - optional `state`
  - optional masks / terminal flags

### Module contract

The main design choice in `stable_pretraining` is:

- define `forward(self, batch, stage)`
- do not build a custom Lightning `training_step` first unless we hit a real limitation

The docs explicitly say:

- the custom `forward` returns a dictionary
- training requires a `"loss"` key in that dictionary
- extra outputs are available to callbacks and logging

Implication for TD-Flow:

our custom forward should return at least:

- `"loss"`
- `"loss_direct"`
- `"loss_bootstrap"`
- `"embedding"` or `"latent"`
- `"vf_norm"` / `"latent_norm"` if useful

### Model composition

`spt.Module(...)` accepts model pieces as kwargs, for example:

- `backbone`
- `projector`
- custom losses
- custom `forward`
- optimizer config

Implication for TD-Flow:

we can package the whole TD-Flow learner into one `spt.Module` by passing:

- `backbone`
- `vector_field`
- `target_vector_field`
- optional `action_encoder`
- `forward=td_flow_forward`
- `optim={...}`

### Backbones

The public docs expose:

- `stable_pretraining.backbone.MLP`
- `stable_pretraining.backbone.from_torchvision(...)`
- other reusable backbones

Planned usage:

- low-dimensional observations: start with `MLP`
- pixel observations: start with `from_torchvision(...)`

### Callbacks

Callbacks are one of the main reasons to use `stable_pretraining` here.

The docs highlight:

- `OnlineProbe`
- `OnlineKNN`
- `RankMe`
- `LoggingCallback`
- checkpoint-related callbacks

Useful initial monitoring for TD-Flow:

- scalar losses
- latent norm / variance
- representation collapse indicators
- optional goal-prediction or reward-proxy probe if labels are available

### Training orchestration

The standard wiring is:

1. build dataloaders
2. wrap them in `spt.data.DataModule`
3. build `spt.Module`
4. build Lightning `Trainer`
5. run through `spt.Manager(trainer=..., module=..., data=...)`

Implication:

- keep our training entrypoint thin
- avoid building a separate bespoke trainer loop unless target-network updates or planning hooks force it

### What this means for TD-Flow

`stable_pretraining` should own:

- encoder backbone
- vector-field network
- target-network EMA update
- loss computation
- optimizer / scheduler
- training-time monitoring

It should not own:

- environment reset / stepping
- dataset recording
- MPC planning loop

## stable_worldmodel

Docs:

- <https://galilai-group.github.io/stable-worldmodel/>
- <https://galilai-group.github.io/stable-worldmodel/quick_start/>
- Policy API: <https://galilai-group.github.io/stable-worldmodel/api/policy/>
- Solver API: <https://galilai-group.github.io/stable-worldmodel/api/solver/>
- World API: <https://galilai-group.github.io/stable-worldmodel/api/world/>
- Dataset API: <https://galilai-group.github.io/stable-worldmodel/api/dataset/>

### Core mental model

`stable_worldmodel` is the outer world-model research shell:

- environments
- data collection
- HDF5 datasets
- planners / solvers
- evaluation

This is where we should integrate the trained TD-Flow model for control-time use.

### Installation note

The docs explicitly say the base install does not include all environment or training dependencies.

If we want the full stack, the docs recommend:

- `pip install stable-worldmodel[env,train]`

That matters because a plain install may be missing environment support.

### World

The typical entrypoint is:

- `world = swm.World(...)`

Key world-side methods we care about:

- `set_policy(...)`
- `record_dataset(...)`
- `evaluate(...)`
- `evaluate_from_dataset(...)`

### Dataset recording

The intended path for offline data collection is:

1. instantiate `World`
2. attach a policy
3. call `world.record_dataset(...)`

Important notes from docs:

- datasets are stored in HDF5
- default storage root is `$STABLEWM_HOME`
- default fallback location is `~/.stable_worldmodel/`

Implication:

- this repo should not invent its own first-pass dataset format
- use recorded HDF5 datasets directly unless there is a hard blocker

### HDF5Dataset

The docs show:

- `HDF5Dataset(name=..., frameskip=..., num_steps=..., keys_to_load=[...])`

Operational meaning:

- `frameskip` controls temporal stride
- `num_steps` controls returned sequence length
- it is already compatible with PyTorch `DataLoader`

This is likely the right starting dataset format for TD-Flow training.

Expected fields for first integration:

- `pixels` for image observations
- `state` for low-dimensional debugging
- `action`

### Policy interface

All policies must implement:

- `get_action(obs, **kwargs)` or the policy-specific info-dict form used by the provided classes

Important policy utility:

- `WorldModelPolicy(solver=..., config=PlanConfig(...))`

Important `PlanConfig` fields:

- `horizon`
- `receding_horizon`
- `history_len`
- `action_block`
- `warm_start`

Implication:

- when plugging TD-Flow into planning, our model does not need to be a policy directly
- it can stay a cost model and be wrapped through `WorldModelPolicy`

### Solver interface

The key planning abstraction is a solver that optimizes action candidates using a model with `get_cost(...)`.

Important solver note from the quickstart:

- the planning model can be any object implementing `get_cost(info, action_candidates)`

Important shape note from the docs:

- `action_candidates` are shaped like `(num_envs, num_samples, horizon, action_dim)`

The docs also describe cost outputs in solver examples as `(B, S)` and, in the quickstart prose, `(num_envs, num_samples, 1)`.

Practical conclusion:

- we should target a plain scalar cost per candidate sequence
- when implementing, confirm the installed version’s exact expected output shape before wiring the solver

### CEMSolver

The most relevant initial planner is:

- `CEMSolver(model=world_model, num_samples=300, ...)`

This is the shortest path for testing TD-Flow in MPC.

### Evaluation

`stable_worldmodel` supports:

- `world.evaluate(...)`
- `world.evaluate_from_dataset(...)`

The docs explicitly recommend `evaluate_from_dataset(...)` for fairer and solvable offline evaluation.

Why it matters:

- start states come from a recorded dataset
- goals are sampled a fixed offset later in the same trajectory
- this avoids evaluating on impossible or inconsistent tasks

Implication:

- our first benchmark should use `evaluate_from_dataset(...)`, not raw random-goal online evaluation

## Integration Decisions For This Repo

### 1. Use `stable_pretraining` for training only

Put TD-Flow learning inside:

- custom dataset wrapper or transform
- custom `spt.Module`
- custom `forward(batch, stage)`

### 2. Use `stable_worldmodel` for data and control

Put environment-facing functionality inside:

- dataset collection with `record_dataset(...)`
- loading with `HDF5Dataset`
- planning with `WorldModelPolicy + CEMSolver`
- evaluation with `evaluate_from_dataset(...)`

### 3. First model should be latent-space TD-Flow

Do not start with raw-pixel TD-Flow dynamics.

Start with:

- encoder backbone from `stable_pretraining`
- TD-Flow in latent space
- cost computed from latent rollout predictions relative to current goal

### 4. First world-model adapter should implement `get_cost(...)`

The cleanest interface boundary is:

- train a TD-Flow module
- wrap it in a planner adapter
- expose `get_cost(info, action_candidates)`

That keeps the training model and planning API decoupled.

## Open Questions To Resolve During Implementation

1. Which installed versions of both libraries are we actually targeting?
2. Does the installed `stable_worldmodel` solver expect cost shape `(B, S)` or `(B, S, 1)`?
3. What is the cleanest observation key for training:
   - `pixels`
   - `state`
   - both
4. Should the first reproduction target:
   - low-dimensional state first
   - pixel observations first
5. Where should target-network EMA updates live:
   - inside custom `forward`
   - callback
   - Lightning hook exposed through `spt.Module`

## Working Assumption

Until local code proves otherwise, the safest implementation assumption is:

- `stable_pretraining` hosts the TD-Flow learner
- `stable_worldmodel` hosts data collection, planning, and evaluation
- the boundary between them is a trained module exposing `get_cost(...)`
