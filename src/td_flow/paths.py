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


def sample_late_mixture_time(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    late_prob: float,
    late_start: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    if not 0.0 <= late_prob <= 1.0:
        raise ValueError("late_prob must be in [0, 1].")
    if not eps < late_start < 1.0:
        raise ValueError("late_start must be in (eps, 1).")

    uniform_time = sample_time(batch_size, device=device, dtype=dtype, eps=eps)
    if late_prob == 0.0:
        return uniform_time

    late_lower = max(float(late_start), float(eps))
    late_upper = 1.0 - float(eps)
    if late_lower >= late_upper:
        raise ValueError("late_start must be smaller than 1 - eps.")

    late_time = torch.rand(batch_size, device=device, dtype=dtype)
    late_time = late_lower + (late_upper - late_lower) * late_time
    late_mask = torch.rand(batch_size, device=device) < float(late_prob)
    return torch.where(late_mask, late_time, uniform_time)


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
    del eps
    ut = target - source
    return xt, ut
