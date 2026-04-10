from __future__ import annotations

from collections.abc import Callable

import torch
from torchdiffeq import odeint


def _manual_midpoint_integrate(
    vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    t_end: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    x_t = source
    dt = t_end / float(steps)
    dt_view = dt.view(-1, *([1] * (source.ndim - 1)))
    half_dt_view = 0.5 * dt_view

    for step in range(steps):
        t_i = dt * float(step)
        k1 = vector_field(x_t, t_i)
        midpoint = x_t + half_dt_view * k1
        k2 = vector_field(midpoint, t_i + 0.5 * dt)
        x_t = x_t + dt_view * k2

    return x_t


def midpoint_integrate(
    vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    t_end: torch.Tensor | float,
    *,
    steps: int,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")

    if not torch.is_tensor(t_end):
        t_end = torch.full(
            (source.shape[0],),
            float(t_end),
            device=source.device,
            dtype=source.dtype,
        )
    else:
        t_end = t_end.to(device=source.device, dtype=source.dtype)

    if torch.compiler.is_compiling():
        return _manual_midpoint_integrate(
            vector_field,
            source,
            t_end,
            steps=steps,
        )

    class _ScaledVectorField(torch.nn.Module):
        def __init__(
            self,
            vf: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            final_time: torch.Tensor,
        ) -> None:
            super().__init__()
            self.vf = vf
            self.final_time = final_time

        def forward(self, normalized_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            actual_t = normalized_t.to(dtype=x_t.dtype, device=x_t.device) * self.final_time
            scale = self.final_time.view(-1, *([1] * (x_t.ndim - 1)))
            return scale * self.vf(x_t, actual_t)

    integration_times = torch.tensor([0.0, 1.0], device=source.device, dtype=source.dtype)
    solution = odeint(
        _ScaledVectorField(vector_field, t_end),
        source,
        integration_times,
        method="midpoint",
        options={"step_size": 1.0 / float(steps)},
    )
    return solution[-1]
