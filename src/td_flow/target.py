from __future__ import annotations

import copy

import torch
import torch.nn as nn


def clone_as_target(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    target.eval()
    target.requires_grad_(False)
    return target


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, polyak: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(polyak).add_(source_param.data, alpha=1.0 - polyak)

    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)
