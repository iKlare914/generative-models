import torch as th
from torch import nn

from .utils import zero_init


class QKVMHAttention(nn.Module):
    def __init__(self, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: th.Tensor):
        """
        Args:
            qkv: Tensor [B, C, (H*W)]
        Returns:
            attention: Tensor [B, C, (H*W)]
        """
        raise NotImplementedError("Implement the new attention forward pass.")


class QKVMHACrossAttention(nn.Module):
    def __init__(self, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor, attention_mask: th.Tensor):
        """
        Args:
            q: Tensor [B, C, (H*W)]
            k: Tensor [B, L, C]
            v: Tensor [B, L, C]
            attention_mask: 0-1 Tensor [B, L]
        Returns:
            attention: Tensor [B, C, (H*W)]
        """
        raise NotImplementedError("Implement the new attention forward pass.")


class AttentionBlock(nn.Module):
    def __init__(self, channel, num_heads = 1, o_proj_zeroinit=False):
        super().__init__()
        self.channel = channel
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv1d(channel, 3 * channel, 1)
        self.o_proj = zero_init(nn.Conv1d(channel, channel, 1)) if o_proj_zeroinit else nn.Conv1d(channel, channel, 1)
        self.norm = nn.GroupNorm(32, channel)
        self.attention = QKVMHAttention(num_heads)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns: 
            result: Tensor [B, C, H, W]
        """
        raise NotImplementedError("Implement the new attention forward pass.")


class CrossAttentionBlock(nn.Module):
    def __init__(self, channel, feature_channel, num_heads = 1, o_proj_zeroinit=False):
        super().__init__()
        self.channel = channel
        self.feature_channel = feature_channel
        self.num_heads = num_heads
        self.q_proj = nn.Conv1d(channel, channel, 1)
        self.k_proj = nn.Linear(feature_channel, channel)
        self.v_proj = nn.Linear(feature_channel, channel)
        self.o_proj = zero_init(nn.Conv1d(channel, channel, 1)) if o_proj_zeroinit else nn.Conv1d(channel, channel, 1)
        self.norm = nn.GroupNorm(32, channel)
        self.attention = QKVMHACrossAttention(num_heads)

    def forward(self, x: th.Tensor, context: th.Tensor, attention_mask: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
            context: Tensor [B, L, C]
            attention_mask: 0-1 Tensor [B, L]
        Returns:
            result: Tensor [B, C, H, W]
        """
        raise NotImplementedError("Implement the new attention forward pass.")
