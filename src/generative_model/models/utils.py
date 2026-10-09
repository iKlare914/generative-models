import math

import torch as th
from torch import nn


def zero_init(module: nn.Module):
    """
    Initialize all parameters of a module with zero
    """
    for p in module.parameters():
        p = p.detach().zero_()
    return module


def position_embedding(x: th.Tensor, emb_dim: int):
    """
    Apply position embedding for input
    Args:
        x: Tensor [B]
        emb_dim: Dimension of embedding
    Returns:
        emb: Tensor [B, emb_dim]
    """
    if len(x.shape) != 1:
        raise ValueError("x must be 1D tensor")
    if emb_dim <= 0 or not isinstance(emb_dim, int) or emb_dim % 2:
        raise ValueError("emb_dim must be a positive even integer")
    d = emb_dim // 2
    scale = math.log(10000) / d
    scale = th.exp(th.arange(d, dtype=th.float32, device=x.device) * -scale)
    emb = th.outer(x.float(), scale)
    emb = th.cat(
        [th.sin(emb), th.cos(emb)],
        dim=-1
    )
    return emb
