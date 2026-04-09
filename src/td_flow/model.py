from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn as nn
import stable_pretraining as spt

from .config import BackboneConfig, ModelConfig
from .ode import midpoint_integrate
from .paths import sample_linear_probability_path, sample_source, sample_time
from .target import clone_as_target, ema_update


def _make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    activation: type[nn.Module] = nn.Mish,
) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.extend([nn.Linear(in_dim, out_dim), activation()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10_000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * exponent
        )
        angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class StableBackboneEncoder(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], cfg: BackboneConfig, latent_dim: int) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.cfg = cfg

        if cfg.kind == "mlp":
            in_channels = math.prod(observation_shape)
            self.backbone = spt.backbone.MLP(
                in_channels=in_channels,
                hidden_channels=[*cfg.hidden_dims, latent_dim],
                activation_layer=nn.Mish,
            )
            self.projection = nn.Identity()
            self.flatten_input = True
        elif cfg.kind == "torchvision":
            self.backbone = spt.backbone.from_torchvision(
                cfg.torchvision_name,
                low_resolution=cfg.low_resolution,
            )
            self.flatten_input = False
            with torch.no_grad():
                was_training = self.backbone.training
                self.backbone.eval()
                dummy = torch.zeros(1, *observation_shape)
                features = self.backbone(dummy)
                if was_training:
                    self.backbone.train()
            if features.ndim > 2:
                features = features.flatten(1)
            self.projection = (
                nn.Identity()
                if features.shape[-1] == latent_dim
                else nn.Linear(features.shape[-1], latent_dim)
            )
        else:
            raise ValueError(f"Unsupported backbone kind: {cfg.kind}")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = observation.float()
        if self.flatten_input:
            x = x.reshape(x.shape[0], -1)
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.flatten(1)
        return self.projection(features)


class ContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        context_dim: int,
    ) -> None:
        super().__init__()
        self.network = _make_mlp(
            latent_dim + action_dim,
            hidden_dims,
            context_dim,
        )

    def forward(self, state_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        flat_action = action.reshape(action.shape[0], -1).float()
        return self.network(torch.cat([state_latent, flat_action], dim=-1))


class VectorField(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        context_dim: int,
        time_embed_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.network = _make_mlp(
            latent_dim + context_dim + time_embed_dim,
            hidden_dims,
            latent_dim,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_embedding(t.float())
        return self.network(torch.cat([x_t, context, time_features], dim=-1))


class TD2CFMModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = StableBackboneEncoder(
            observation_shape=cfg.observation_shape,
            cfg=cfg.backbone,
            latent_dim=cfg.latent_dim,
        )
        self.context_encoder = ContextEncoder(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.context_hidden_dims,
            context_dim=cfg.context_dim,
        )
        self.vector_field = VectorField(
            latent_dim=cfg.latent_dim,
            context_dim=cfg.context_dim,
            time_embed_dim=cfg.time_embed_dim,
            hidden_dims=cfg.vector_field_hidden_dims,
        )

        self.target_encoder = clone_as_target(self.encoder)
        self.target_context_encoder = clone_as_target(self.context_encoder)
        self.target_vector_field = clone_as_target(self.vector_field)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def update_targets(self) -> None:
        ema_update(self.target_encoder, self.encoder, self.cfg.polyak)
        ema_update(self.target_context_encoder, self.context_encoder, self.cfg.polyak)
        ema_update(self.target_vector_field, self.vector_field, self.cfg.polyak)

    def encode_observation(self, observation: torch.Tensor, *, use_target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.encoder
        return encoder(observation)

    def encode_context(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        *,
        use_target: bool = False,
    ) -> torch.Tensor:
        context_encoder = self.target_context_encoder if use_target else self.context_encoder
        return context_encoder(state_latent, action)

    def compute_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        *,
        use_target: bool = False,
    ) -> torch.Tensor:
        context = self.encode_context(state_latent, action, use_target=use_target)
        vector_field = self.target_vector_field if use_target else self.vector_field
        return vector_field(x_t, t, context)

    def bootstrap_target(
        self,
        next_latent: torch.Tensor,
        next_action: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def vf(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return self.compute_velocity(
                x_t,
                tau,
                next_latent,
                next_action,
                use_target=True,
            )

        x_t = midpoint_integrate(vf, source, t, steps=self.cfg.ode_steps)
        v_t = vf(x_t, t)
        return x_t.detach(), v_t.detach()

    @torch.no_grad()
    def predict_next_latent(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        *,
        t_end: float = 1.0,
        use_target: bool = False,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            source = torch.zeros(
                state_latent.shape[0],
                self.cfg.latent_dim,
                device=state_latent.device,
                dtype=state_latent.dtype,
            )

        def vf(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return self.compute_velocity(
                x_t,
                tau,
                state_latent,
                action,
                use_target=use_target,
            )

        return midpoint_integrate(vf, source, t_end, steps=self.cfg.ode_steps)

    def compute_state(self, batch: dict[str, torch.Tensor], *, stage: str) -> dict[str, torch.Tensor]:
        training = stage.startswith("train") and self.training
        if training:
            self.update_targets()

        obs = batch["obs"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next_obs"].to(self.device).float()
        next_action = batch.get("next_action")
        if next_action is None:
            next_action = torch.zeros_like(action)
        next_action = next_action.to(self.device).float()

        state_latent = self.encode_observation(obs)
        next_latent_target = self.encode_observation(next_obs, use_target=True).detach()

        t = sample_time(
            obs.shape[0],
            device=self.device,
            dtype=state_latent.dtype,
            eps=self.cfg.time_eps,
        )
        source = sample_source(
            obs.shape[0],
            self.cfg.latent_dim,
            device=self.device,
            dtype=state_latent.dtype,
        )

        direct_xt, direct_target = sample_linear_probability_path(
            source,
            next_latent_target,
            t,
            eps=self.cfg.time_eps,
        )
        direct_prediction = self.compute_velocity(
            direct_xt,
            t,
            state_latent,
            action,
            use_target=False,
        )

        bootstrap_xt, bootstrap_target = self.bootstrap_target(
            next_latent_target,
            next_action,
            source,
            t,
        )
        bootstrap_prediction = self.compute_velocity(
            bootstrap_xt,
            t,
            state_latent,
            action,
            use_target=False,
        )

        direct_loss = torch.mean((direct_prediction - direct_target) ** 2)
        bootstrap_loss = torch.mean((bootstrap_prediction - bootstrap_target) ** 2)
        loss = (1.0 - self.cfg.gamma) * direct_loss + self.cfg.gamma * bootstrap_loss

        return {
            "loss": loss,
            "loss_direct": direct_loss.detach(),
            "loss_bootstrap": bootstrap_loss.detach(),
            "latent": state_latent.detach(),
            "latent_norm": state_latent.norm(dim=-1).mean().detach(),
            "vf_norm": direct_prediction.norm(dim=-1).mean().detach(),
        }

