#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haizhou/Documents/td_flow"
cd "$ROOT"

DATASET_DIR="$ROOT/data/stablewm_cache"
COMMON_ARGS=(
  --data.dataset-name pointmass-exorl-rnd-scripted-policy-relnoise10
  --data.backend stablewm_hdf5
  --data.dir "$DATASET_DIR"
  --data.observation-key observation
  --data.action-key action
  --data.next-action-key policy_action
  --data.num-workers 16
  --policy-mode single_policy
  --observation-encoder identity
  --network-variant paper
  --train.max-steps 60000
  --train.use-wandb
  --train.wandb-project td_flow
  --train.checkpoint-every-n-train-steps 10000
)

echo "[$(date -Iseconds)] Starting pointmass TD2 ablation suite"

echo "[$(date -Iseconds)] Run 1/3: gamma=0.95"
uv run python -m td_flow.train \
  "${COMMON_ARGS[@]}" \
  --gamma 0.95 \
  --train.run-name pointmass-exorl-offpolicy-g095-paper-60k

echo "[$(date -Iseconds)] Run 2/3: gamma=0.90"
uv run python -m td_flow.train \
  "${COMMON_ARGS[@]}" \
  --gamma 0.90 \
  --train.run-name pointmass-exorl-offpolicy-g090-paper-60k

echo "[$(date -Iseconds)] Run 3/3: gamma=0.99, direct/bootstrap weights 0.1/0.9"
uv run python -m td_flow.train \
  "${COMMON_ARGS[@]}" \
  --gamma 0.99 \
  --direct-loss-weight 0.1 \
  --bootstrap-loss-weight 0.9 \
  --train.run-name pointmass-exorl-offpolicy-g099-direct01-bootstrap09-paper-60k

echo "[$(date -Iseconds)] Ablation suite completed"
