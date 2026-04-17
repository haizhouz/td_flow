from __future__ import annotations

import math

import torch
import torch.nn as nn
import stable_pretraining as spt

from .config import BackboneConfig, ModelConfig, resolve_paper_polyak
from .ode import midpoint_integrate
from .paths import (
    sample_late_mixture_time,
    sample_linear_probability_path,
    sample_source,
    sample_time,
)
from .target import clone_as_target, ema_update

PAPER_SINGLE_WIDTH = 512
PAPER_MULTI_WIDTH = 1024


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


def _apply_orthogonal_init(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


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


class IdentityObservationEncoder(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.float().reshape(observation.shape[0], -1)


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
        policy_embedding_dim: int,
        hidden_dims: tuple[int, ...],
        context_dim: int,
    ) -> None:
        super().__init__()
        self.network = _make_mlp(
            latent_dim + action_dim + policy_embedding_dim,
            hidden_dims,
            context_dim,
        )

    def forward(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        policy_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_action = action.reshape(action.shape[0], -1).float()
        inputs = [state_latent, flat_action]
        if policy_embedding is not None:
            inputs.append(policy_embedding.reshape(policy_embedding.shape[0], -1).float())
        return self.network(torch.cat(inputs, dim=-1))


class PaperContextEncoder(ContextEncoder):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        policy_embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            policy_embedding_dim=policy_embedding_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            context_dim=hidden_dim,
        )


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.mod1 = nn.Linear(conditioning_dim, 2 * output_dim)
        self.mod2 = nn.Linear(conditioning_dim, 2 * output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.activation = nn.Mish()

    def _film(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        scale, shift = modulation.chunk(2, dim=-1)
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)
        h = self.input_proj(x)
        h = self.norm1(h)
        h = self._film(h, self.mod1(conditioning))
        h = self.activation(h)
        h = self.output_proj(h)
        h = self.norm2(h)
        h = self._film(h, self.mod2(conditioning))
        h = self.activation(h)
        return residual + h


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
        if len(hidden_dims) == 0:
            raise ValueError("vector_field_hidden_dims must be non-empty")

        self.raw_time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_encoder = _make_mlp(
            time_embed_dim,
            (time_embed_dim,),
            time_embed_dim,
        )
        self.conditioning_dim = context_dim + time_embed_dim
        self.down_blocks = nn.ModuleList()

        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.down_blocks.append(
                FiLMResidualBlock(
                    input_dim=in_dim,
                    output_dim=hidden_dim,
                    conditioning_dim=self.conditioning_dim,
                )
            )
            in_dim = hidden_dim

        self.mid_block = FiLMResidualBlock(
            input_dim=in_dim,
            output_dim=in_dim,
            conditioning_dim=self.conditioning_dim,
        )

        self.up_blocks = nn.ModuleList()
        for skip_dim in reversed(hidden_dims[:-1]):
            self.up_blocks.append(
                FiLMResidualBlock(
                    input_dim=in_dim + skip_dim,
                    output_dim=skip_dim,
                    conditioning_dim=self.conditioning_dim,
                )
            )
            in_dim = skip_dim

        self.final = nn.Linear(in_dim, latent_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_encoder(self.raw_time_embedding(t.float()))
        conditioning = torch.cat([context, time_features], dim=-1)

        h = x_t
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            h = block(h, conditioning)
            if index < len(self.down_blocks) - 1:
                skips.append(h)

        h = self.mid_block(h, conditioning)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(torch.cat([h, skip], dim=-1), conditioning)

        return self.final(h)


class PaperVectorField(VectorField):
    def __init__(
        self,
        *,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            context_dim=context_dim,
            time_embed_dim=256,
            hidden_dims=(hidden_dim, hidden_dim, hidden_dim),
        )


def _resolve_paper_hidden_dim(policy_mode: str) -> int:
    if policy_mode == "single_policy":
        return PAPER_SINGLE_WIDTH
    if policy_mode == "multi_policy":
        return PAPER_MULTI_WIDTH
    raise ValueError("policy_mode must be one of: single_policy, multi_policy")


class TD2CFMModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_mode = cfg.observation_encoder.lower()
        if encoder_mode == "auto":
            self.use_identity_encoder = cfg.policy_mode == "single_policy" and len(cfg.observation_shape) == 1
        elif encoder_mode in {"identity", "no_encoder"}:
            self.use_identity_encoder = True
        elif encoder_mode == "learned":
            self.use_identity_encoder = False
        else:
            raise ValueError(
                "observation_encoder must be one of: auto, identity, no_encoder, learned"
            )
        self.network_variant = cfg.network_variant.lower()
        if self.network_variant not in {"repo", "paper"}:
            raise ValueError("network_variant must be one of: repo, paper")
        self.target_polyak = (
            cfg.polyak
            if cfg.polyak is not None
            else resolve_paper_polyak(cfg.policy_mode)
        )
        self.initialization = cfg.initialization.lower()
        if self.initialization not in {"default", "orthogonal"}:
            raise ValueError("initialization must be one of: default, orthogonal")

        self.latent_dim = math.prod(cfg.observation_shape) if self.use_identity_encoder else cfg.latent_dim
        self.encoder = (
            IdentityObservationEncoder()
            if self.use_identity_encoder
            else StableBackboneEncoder(
                observation_shape=cfg.observation_shape,
                cfg=cfg.backbone,
                latent_dim=self.latent_dim,
            )
        )
        if self.network_variant == "paper":
            paper_hidden_dim = _resolve_paper_hidden_dim(cfg.policy_mode)
            self.context_dim = paper_hidden_dim
            self.context_encoder = PaperContextEncoder(
                latent_dim=self.latent_dim,
                action_dim=cfg.action_dim,
                policy_embedding_dim=cfg.policy_embedding_dim,
                hidden_dim=paper_hidden_dim,
            )
            self.vector_field = PaperVectorField(
                latent_dim=self.latent_dim,
                context_dim=paper_hidden_dim,
                hidden_dim=paper_hidden_dim,
            )
        else:
            self.context_dim = cfg.context_dim
            self.context_encoder = ContextEncoder(
                latent_dim=self.latent_dim,
                action_dim=cfg.action_dim,
                policy_embedding_dim=cfg.policy_embedding_dim,
                hidden_dims=cfg.context_hidden_dims,
                context_dim=cfg.context_dim,
            )
            self.vector_field = VectorField(
                latent_dim=self.latent_dim,
                context_dim=cfg.context_dim,
                time_embed_dim=cfg.time_embed_dim,
                hidden_dims=cfg.vector_field_hidden_dims,
            )
        self.one_step_head = _make_mlp(
            self.context_dim,
            (self.context_dim,),
            self.latent_dim,
        )

        if self.initialization == "orthogonal":
            _apply_orthogonal_init(self.encoder)
            _apply_orthogonal_init(self.context_encoder)
            _apply_orthogonal_init(self.vector_field)
            _apply_orthogonal_init(self.one_step_head)

        self.target_encoder = clone_as_target(self.encoder)
        self.target_context_encoder = clone_as_target(self.context_encoder)
        self.target_vector_field = clone_as_target(self.vector_field)
        self.loss_weight_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def update_targets(self) -> None:
        ema_update(self.target_encoder, self.encoder, self.target_polyak)
        ema_update(self.target_context_encoder, self.context_encoder, self.target_polyak)
        ema_update(self.target_vector_field, self.vector_field, self.target_polyak)

    def base_loss_weights(self) -> tuple[float, float]:
        direct_weight = self.cfg.direct_loss_weight
        bootstrap_weight = self.cfg.bootstrap_loss_weight
        if direct_weight is None and bootstrap_weight is None:
            return 1.0 - self.cfg.gamma, self.cfg.gamma
        if direct_weight is None or bootstrap_weight is None:
            raise ValueError(
                "direct_loss_weight and bootstrap_loss_weight must both be set or both be None."
            )
        if direct_weight < 0.0 or bootstrap_weight < 0.0:
            raise ValueError("loss weights must be non-negative.")
        if direct_weight == 0.0 and bootstrap_weight == 0.0:
            raise ValueError("At least one loss weight must be positive.")
        return float(direct_weight), float(bootstrap_weight)

    def set_loss_weight_step(self, step: int) -> None:
        self.loss_weight_step = max(int(step), 0)

    def loss_weights(self) -> tuple[float, float]:
        target_direct, target_bootstrap = self.base_loss_weights()
        schedule = self.cfg.loss_weight_schedule
        if schedule == "constant":
            return target_direct, target_bootstrap
        if schedule != "direct_warmup_linear":
            raise ValueError(f"Unsupported loss_weight_schedule: {schedule}")

        warmup_steps = self.cfg.loss_weight_warmup_steps
        ramp_steps = self.cfg.loss_weight_ramp_steps
        if warmup_steps < 0 or ramp_steps < 0:
            raise ValueError("loss_weight_warmup_steps and loss_weight_ramp_steps must be non-negative.")

        if self.loss_weight_step < warmup_steps:
            return 1.0, 0.0
        if ramp_steps == 0:
            return target_direct, target_bootstrap

        ramp_progress = min(max(self.loss_weight_step - warmup_steps, 0), ramp_steps) / float(ramp_steps)
        direct_weight = (1.0 - ramp_progress) * 1.0 + ramp_progress * target_direct
        bootstrap_weight = ramp_progress * target_bootstrap
        return float(direct_weight), float(bootstrap_weight)

    def sample_bootstrap_time(self, batch_size: int, *, dtype: torch.dtype) -> torch.Tensor:
        mode = self.cfg.bootstrap_time_sampling
        if mode == "uniform":
            return sample_time(
                batch_size,
                device=self.device,
                dtype=dtype,
                eps=self.cfg.time_eps,
            )
        if mode == "late_mixture":
            return sample_late_mixture_time(
                batch_size,
                device=self.device,
                dtype=dtype,
                late_prob=self.cfg.bootstrap_time_late_prob,
                late_start=self.cfg.bootstrap_time_late_start,
                eps=self.cfg.time_eps,
            )
        raise ValueError(f"Unsupported bootstrap_time_sampling mode: {mode}")

    def encode_observation(self, observation: torch.Tensor, *, use_target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.encoder
        return encoder(observation)

    def conditioning_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.cfg.state_only_conditioning:
            return action
        return torch.zeros_like(action)

    def encode_context(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        policy_embedding: torch.Tensor | None = None,
        *,
        use_target: bool = False,
    ) -> torch.Tensor:
        if self.cfg.policy_embedding_dim > 0 and policy_embedding is None:
            raise ValueError(
                "policy_embedding is required when policy_embedding_dim > 0."
            )
        context_encoder = self.target_context_encoder if use_target else self.context_encoder
        conditioned_action = self.conditioning_action(action)
        return context_encoder(state_latent, conditioned_action, policy_embedding)

    def compute_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        policy_embedding: torch.Tensor | None = None,
        *,
        use_target: bool = False,
    ) -> torch.Tensor:
        context = self.encode_context(
            state_latent,
            action,
            policy_embedding,
            use_target=use_target,
        )
        vector_field = self.target_vector_field if use_target else self.vector_field
        return vector_field(x_t, t, context)

    def predict_one_step_latent(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        policy_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encode_context(
            state_latent,
            action,
            policy_embedding,
            use_target=False,
        )
        return self.one_step_head(context)

    def bootstrap_target(
        self,
        next_latent: torch.Tensor,
        next_action: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
        policy_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def vf(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return self.compute_velocity(
                x_t,
                tau,
                next_latent,
                next_action,
                policy_embedding,
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
        policy_embedding: torch.Tensor | None = None,
        *,
        t_end: float = 1.0,
        use_target: bool = False,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            source = sample_source(
                state_latent.shape[0],
                self.latent_dim,
                device=state_latent.device,
                dtype=state_latent.dtype,
            )

        def vf(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return self.compute_velocity(
                x_t,
                tau,
                state_latent,
                action,
                policy_embedding,
                use_target=use_target,
            )

        return midpoint_integrate(vf, source, t_end, steps=self.cfg.ode_steps)

    def compute_state(self, batch: dict[str, torch.Tensor], *, stage: str) -> dict[str, torch.Tensor]:
        obs = batch["obs"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        next_obs = batch["next_obs"].to(self.device).float()
        next_action = batch.get("next_action")
        if next_action is None:
            next_action = torch.zeros_like(action)
        next_action = next_action.to(self.device).float()
        policy_embedding = batch.get("policy_embedding")
        if policy_embedding is not None:
            policy_embedding = policy_embedding.to(self.device).float()

        state_latent = self.encode_observation(obs)
        next_latent_target = self.encode_observation(next_obs, use_target=True).detach()

        direct_t = sample_time(
            obs.shape[0],
            device=self.device,
            dtype=state_latent.dtype,
            eps=self.cfg.time_eps,
        )
        bootstrap_t = self.sample_bootstrap_time(
            obs.shape[0],
            dtype=state_latent.dtype,
        )
        source = sample_source(
            obs.shape[0],
            self.latent_dim,
            device=self.device,
            dtype=state_latent.dtype,
        )

        direct_xt, direct_target = sample_linear_probability_path(
            source,
            next_latent_target,
            direct_t,
            eps=self.cfg.time_eps,
        )
        direct_prediction = self.compute_velocity(
            direct_xt,
            direct_t,
            state_latent,
            action,
            policy_embedding,
            use_target=False,
        )

        bootstrap_xt, bootstrap_target = self.bootstrap_target(
            next_latent_target,
            next_action,
            source,
            bootstrap_t,
            policy_embedding,
        )
        bootstrap_prediction = self.compute_velocity(
            bootstrap_xt,
            bootstrap_t,
            state_latent,
            action,
            policy_embedding,
            use_target=False,
        )
        next_latent_prediction = self.predict_one_step_latent(
            state_latent,
            action,
            policy_embedding,
        )

        direct_loss = torch.mean((direct_prediction - direct_target) ** 2)
        bootstrap_loss = torch.mean((bootstrap_prediction - bootstrap_target) ** 2)
        one_step_prediction_loss = torch.mean((next_latent_prediction - next_latent_target) ** 2)
        direct_weight, bootstrap_weight = self.loss_weights()
        loss = (
            direct_weight * direct_loss
            + bootstrap_weight * bootstrap_loss
            + self.cfg.one_step_prediction_loss_weight * one_step_prediction_loss
        )

        return {
            "loss": loss,
            "loss_direct": direct_loss.detach(),
            "loss_bootstrap": bootstrap_loss.detach(),
            "loss_one_step_prediction": one_step_prediction_loss.detach(),
            "loss_direct_weight": torch.tensor(direct_weight, device=self.device),
            "loss_bootstrap_weight": torch.tensor(bootstrap_weight, device=self.device),
            "loss_one_step_prediction_weight": torch.tensor(
                self.cfg.one_step_prediction_loss_weight,
                device=self.device,
            ),
            "latent": state_latent.detach(),
            "latent_norm": state_latent.norm(dim=-1).mean().detach(),
            "vf_norm": direct_prediction.norm(dim=-1).mean().detach(),
            "one_step_prediction_norm": next_latent_prediction.norm(dim=-1).mean().detach(),
        }
