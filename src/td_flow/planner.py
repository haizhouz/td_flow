from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stable_worldmodel.policy import BasePolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

from .config import PlanningConfig
from .model import TD2CFMModel


def _ensure_tensor(value: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class TD2CFMPlannerAdapter(nn.Module):
    def __init__(
        self,
        model: TD2CFMModel,
        *,
        observation_key: str = "pixels",
        goal_key: str = "goal",
        rollout_discount: float = 1.0,
        terminal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.observation_key = observation_key
        self.goal_key = goal_key
        self.rollout_discount = rollout_discount
        self.terminal_weight = terminal_weight

    @property
    def device(self) -> torch.device:
        return self.model.device

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        current_obs = _ensure_tensor(info_dict[self.observation_key], self.device)
        goal_obs = _ensure_tensor(info_dict[self.goal_key], self.device)
        actions = _ensure_tensor(action_candidates, self.device)

        batch_size, num_samples, horizon, action_dim = actions.shape
        flat_obs = current_obs.reshape(batch_size * num_samples, *current_obs.shape[2:])
        flat_goal = goal_obs.reshape(batch_size * num_samples, *goal_obs.shape[2:])
        flat_actions = actions.reshape(batch_size * num_samples, horizon, action_dim)

        with torch.no_grad():
            current_latent = self.model.encode_observation(flat_obs)
            goal_latent = self.model.encode_observation(flat_goal)

            costs = torch.zeros(
                batch_size * num_samples,
                device=self.device,
                dtype=current_latent.dtype,
            )
            discount = 1.0

            for step in range(horizon):
                current_latent = self.model.predict_next_latent(
                    current_latent,
                    flat_actions[:, step],
                    source=torch.zeros_like(current_latent),
                )
                step_cost = torch.mean((current_latent - goal_latent) ** 2, dim=-1)
                costs = costs + discount * step_cost
                discount *= self.rollout_discount

            if self.terminal_weight != 0.0:
                terminal_cost = torch.mean((current_latent - goal_latent) ** 2, dim=-1)
                costs = costs + self.terminal_weight * terminal_cost

        return costs.view(batch_size, num_samples)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.criterion(info_dict, action_candidates)


class TD2CFMPlanningPolicy(BasePolicy):
    def __init__(
        self,
        *,
        solver: CEMSolver,
        config: PlanConfig,
        observation_key: str,
        goal_key: str,
        process: dict | None = None,
        transform: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "world_model"
        self.solver = solver
        self.cfg = config
        self.observation_key = observation_key
        self.goal_key = goal_key
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[torch.Tensor] = []
        self._next_init: torch.Tensor | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env) -> None:
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        self.solver.configure(
            action_space=env.action_space,
            n_envs=n_envs,
            config=self.cfg,
        )
        self._action_buffer = []

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        del kwargs
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert self.observation_key in info_dict, (
            f"'{self.observation_key}' must be provided in info_dict"
        )
        assert self.goal_key in info_dict, (
            f"'{self.goal_key}' must be provided in info_dict"
        )

        info_dict = self._prepare_info(dict(info_dict))

        if len(self._action_buffer) == 0:
            outputs = self.solver(info_dict, init_action=self._next_init)
            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None
            plan = plan.reshape(
                self.env.num_envs,
                self.flatten_receding_horizon,
                -1,
            )
            self._action_buffer.extend(list(plan.transpose(0, 1)))

        action = self._action_buffer.pop(0)
        action = action.reshape(*self.env.action_space.shape)
        return action.detach().cpu().numpy()


def build_planning_policy(
    model: TD2CFMModel,
    planning_config: PlanningConfig,
    *,
    observation_key: str = "observation",
    goal_key: str = "target",
    device: str | torch.device = "cpu",
) -> TD2CFMPlanningPolicy:
    adapter = TD2CFMPlannerAdapter(
        model,
        observation_key=observation_key,
        goal_key=goal_key,
        rollout_discount=planning_config.rollout_discount,
        terminal_weight=planning_config.terminal_weight,
    )
    solver = CEMSolver(
        model=adapter,
        batch_size=planning_config.batch_size,
        num_samples=planning_config.num_samples,
        var_scale=planning_config.var_scale,
        n_steps=planning_config.n_steps,
        topk=planning_config.topk,
        device=device,
    )
    return TD2CFMPlanningPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=planning_config.horizon,
            receding_horizon=planning_config.receding_horizon,
            history_len=planning_config.history_len,
            action_block=planning_config.action_block,
            warm_start=planning_config.warm_start,
        ),
        observation_key=observation_key,
        goal_key=goal_key,
    )
