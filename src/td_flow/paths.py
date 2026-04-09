from __future__ import annotations

import torch


def expand_time_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while t.ndim < ref.ndim:
        t = t.unsqueeze(-1)
    return t


def sample_time(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    eps: float = 1e-4,
) -> torch.Tensor:
    return torch.rand(batch_size, device=device, dtype=dtype).clamp_(eps, 1.0 - eps)


def sample_source(
    batch_size: int,
    latent_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device, dtype=dtype)


def sample_linear_probability_path(
    source: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_expanded = expand_time_like(t, source)
    xt = t_expanded * target + (1.0 - t_expanded) * source
    denom = expand_time_like((1.0 - t).clamp_min(eps), xt)
    ut = (target - xt) / denom
    return xt, ut

