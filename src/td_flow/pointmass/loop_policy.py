from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


SQRT_TWO = float(np.sqrt(2.0))


def _sign_or_one(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def _torch_sign_or_one(value: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(value)
    return torch.where(value >= 0.0, ones, -ones)


def loop_tangent(x: float, y: float) -> np.ndarray:
    tangent = np.array([_sign_or_one(y), -_sign_or_one(x)], dtype=np.float32)
    return tangent / SQRT_TWO


def torch_loop_tangent(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack((_torch_sign_or_one(y), -_torch_sign_or_one(x)), dim=-1) / SQRT_TWO


def loop_outward_normal(x: float, y: float) -> np.ndarray:
    normal = np.array([_sign_or_one(x), _sign_or_one(y)], dtype=np.float32)
    return normal / SQRT_TWO


def torch_loop_outward_normal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack((_torch_sign_or_one(x), _torch_sign_or_one(y)), dim=-1) / SQRT_TWO


def circle_tangent(x: float, y: float) -> np.ndarray:
    radius = max(float(np.hypot(x, y)), 1e-6)
    tangent = np.array([y, -x], dtype=np.float32) / radius
    return tangent


def torch_circle_tangent(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    radius = torch.clamp(torch.sqrt(x.square() + y.square()), min=1e-6)
    return torch.stack((y / radius, -x / radius), dim=-1)


def circle_outward_normal(x: float, y: float) -> np.ndarray:
    radius = max(float(np.hypot(x, y)), 1e-6)
    return np.array([x, y], dtype=np.float32) / radius


def torch_circle_outward_normal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    radius = torch.clamp(torch.sqrt(x.square() + y.square()), min=1e-6)
    return torch.stack((x / radius, y / radius), dim=-1)


@dataclass(frozen=True)
class PointMassLoopPolicyConfig:
    diamond_radius: float = 0.24
    target_speed: float = 0.10
    radial_gain: float = 1.0
    velocity_gain: float = 20.0
    action_limit: float = 1.0


@dataclass(frozen=True)
class PointMassCirclePolicyConfig:
    radius: float = 0.24
    target_speed: float = 0.10
    radial_gain: float = 1.0
    velocity_gain: float = 20.0
    action_limit: float = 1.0


def desired_loop_velocity(
    observation: np.ndarray,
    config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig(),
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {obs.shape}")

    x, y = float(obs[0]), float(obs[1])
    tangent = loop_tangent(x, y)
    outward = loop_outward_normal(x, y)
    diamond_error = abs(x) + abs(y) - config.diamond_radius

    return (
        config.target_speed * tangent
        - config.radial_gain * diamond_error * outward
    ).astype(np.float32, copy=False)


def torch_desired_loop_velocity(
    observation: torch.Tensor,
    config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig(),
) -> torch.Tensor:
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {tuple(obs.shape)}")

    x = obs[..., 0]
    y = obs[..., 1]
    tangent = torch_loop_tangent(x, y)
    outward = torch_loop_outward_normal(x, y)
    diamond_error = torch.abs(x) + torch.abs(y) - config.diamond_radius
    return config.target_speed * tangent - config.radial_gain * diamond_error.unsqueeze(-1) * outward


def desired_circle_velocity(
    observation: np.ndarray,
    config: PointMassCirclePolicyConfig = PointMassCirclePolicyConfig(),
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {obs.shape}")

    x, y = float(obs[0]), float(obs[1])
    tangent = circle_tangent(x, y)
    outward = circle_outward_normal(x, y)
    radial_error = float(np.hypot(x, y)) - config.radius
    return (
        config.target_speed * tangent
        - config.radial_gain * radial_error * outward
    ).astype(np.float32, copy=False)


def torch_desired_circle_velocity(
    observation: torch.Tensor,
    config: PointMassCirclePolicyConfig = PointMassCirclePolicyConfig(),
) -> torch.Tensor:
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {tuple(obs.shape)}")

    x = obs[..., 0]
    y = obs[..., 1]
    tangent = torch_circle_tangent(x, y)
    outward = torch_circle_outward_normal(x, y)
    radial_error = torch.sqrt(x.square() + y.square()) - config.radius
    return config.target_speed * tangent - config.radial_gain * radial_error.unsqueeze(-1) * outward


def scripted_pointmass_loop_action(
    observation: np.ndarray,
    config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig(),
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {obs.shape}")

    velocity = obs[2:4]
    desired_velocity = desired_loop_velocity(obs, config=config)
    action = config.velocity_gain * (desired_velocity - velocity)
    return np.clip(action, -config.action_limit, config.action_limit).astype(np.float32, copy=False)


def torch_scripted_pointmass_loop_action(
    observation: torch.Tensor,
    config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig(),
) -> torch.Tensor:
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {tuple(obs.shape)}")

    velocity = obs[..., 2:4]
    desired_velocity = torch_desired_loop_velocity(obs, config=config)
    action = config.velocity_gain * (desired_velocity - velocity)
    return torch.clamp(action, -config.action_limit, config.action_limit)


def scripted_pointmass_circle_action(
    observation: np.ndarray,
    config: PointMassCirclePolicyConfig = PointMassCirclePolicyConfig(),
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {obs.shape}")

    velocity = obs[2:4]
    desired_velocity = desired_circle_velocity(obs, config=config)
    action = config.velocity_gain * (desired_velocity - velocity)
    return np.clip(action, -config.action_limit, config.action_limit).astype(np.float32, copy=False)


def torch_scripted_pointmass_circle_action(
    observation: torch.Tensor,
    config: PointMassCirclePolicyConfig = PointMassCirclePolicyConfig(),
) -> torch.Tensor:
    obs = torch.as_tensor(observation, dtype=torch.float32)
    if obs.shape[-1] < 4:
        raise ValueError(f"Expected observation with at least 4 entries, got shape {tuple(obs.shape)}")

    velocity = obs[..., 2:4]
    desired_velocity = torch_desired_circle_velocity(obs, config=config)
    action = config.velocity_gain * (desired_velocity - velocity)
    return torch.clamp(action, -config.action_limit, config.action_limit)


def scripted_pointmass_action(
    observation: np.ndarray,
    *,
    mode: str,
    loop_config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig(),
    circle_config: PointMassCirclePolicyConfig = PointMassCirclePolicyConfig(),
) -> np.ndarray:
    if mode in {"loop", "straight"}:
        return scripted_pointmass_loop_action(observation, config=loop_config)
    if mode == "circle":
        return scripted_pointmass_circle_action(observation, config=circle_config)
    raise ValueError(f"Unsupported scripted pointmass mode: {mode}")


class PointMassLoopScriptedPolicy:
    def __init__(self, config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()) -> None:
        self.config = config

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return scripted_pointmass_loop_action(observation, config=self.config)


class TorchPointMassLoopScriptedPolicy(nn.Module):
    def __init__(self, config: PointMassLoopPolicyConfig = PointMassLoopPolicyConfig()) -> None:
        super().__init__()
        self.config = config

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch_scripted_pointmass_loop_action(observation, config=self.config)
