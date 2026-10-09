import torch as th
from torch import nn

from .utils import zero_init


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim , emb_dim, exp_scale=6, is_zero_init=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.exp_scale = exp_scale
        self.affine_layer = zero_init(nn.Linear(emb_dim, exp_scale * hidden_dim)) if is_zero_init else nn.Linear(emb_dim, exp_scale * hidden_dim)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """
        Apply adaptive layer normalization
        Args:
            emb: Tensor [B, emb_dim]
        Returns:
            result: Tensor [B, 1, exp_scale * hidden_dim]
        """
        result = self.affine_layer(emb).unsqueeze(1) # [B, 1, exp_scale * hidden_dim]
        return result
