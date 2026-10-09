import math

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
        b, c, l = qkv.shape
        q, k, v = qkv.chunk(3, dim=1)
        if (c // 3) % self.num_heads != 0:
            raise ValueError(f"Channel({c // 3}) cannot be divided by num heads({self.num_heads})")
        head_channel = (c // 3) // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(head_channel))
        q = q.reshape(b * self.num_heads, head_channel, l) * scale
        k = k.reshape(b * self.num_heads, head_channel, l) * scale
        v = v.reshape(b * self.num_heads, head_channel, l)
        weight =  th.softmax(th.einsum("bcl,bct->blt", q, k), dim=-1)
        attention = th.einsum(
            "blt,bct->bcl",
            weight,
            v
        )
        return attention.reshape(b, c // 3, l)


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
        b, c, hw = q.shape
        if c % self.num_heads != 0:
            raise ValueError(f"Channel({c}) cannot be divided by num heads({self.num_heads})")
        head_channel = c // self.num_heads
        scale = 1/ math.sqrt(math.sqrt(head_channel))
        q = q.reshape(b * self.num_heads, head_channel, hw) * scale # [B * num_heads, head_channel, H*W]
        k = k.permute(0, 2, 1).reshape(b * self.num_heads, head_channel, -1) * scale # [B * num_heads, head_channel, L]
        v = v.permute(0, 2, 1).reshape(b * self.num_heads, head_channel, -1) # [B * num_heads, head_channel, L]
        attention_mask = attention_mask.unsqueeze(1).repeat_interleave(self.num_heads, dim=0).bool() # [B * num_heads, 1, L]
        weight = th.einsum(
            "bct,bcs->bts",
            q,
            k
        )
        weight = weight.masked_fill(~attention_mask, float("-inf")) # [B * num_heads, H*W, L]
        attention = th.einsum(
            "bts,bcs->bct",
            weight.softmax(dim=-1),
            v
        ) # [B * num_heads, head_channel, hw]
        return attention.reshape(b, c, hw)


class AttentionBlock(nn.Module):
    def __init__(self, channel, num_heads = 1, o_proj_zeroinit=False):
        super().__init__()
        self.channel = channel
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv1d(channel, 3 * channel, 1)
        self.o_proj = nn.Conv1d(channel, channel, 1)
        if o_proj_zeroinit:
            zero_init(self.o_proj)
        self.norm = nn.GroupNorm(32, channel)
        self.attention = QKVMHAttention(num_heads)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns: 
            result: Tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv_proj(self.norm(x))
        a = self.attention(qkv)
        out = self.o_proj(a) + x
        return out.reshape(b, c, h, w)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channel, feature_channel, num_heads = 1, o_proj_zeroinit=False):
        super().__init__()
        self.channel = channel
        self.feature_channel = feature_channel
        self.num_heads = num_heads
        self.q_proj = nn.Conv1d(channel, channel, 1)
        self.k_proj = nn.Linear(feature_channel, channel)
        self.v_proj = nn.Linear(feature_channel, channel)
        self.o_proj = nn.Conv1d(channel, channel, 1)
        if o_proj_zeroinit:
            zero_init(self.o_proj)
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
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        q = self.q_proj(self.norm(x))
        k = self.k_proj(context)
        v = self.v_proj(context)
        a = self.attention(q, k, v, attention_mask)
        out = self.o_proj(a) + x
        return out.reshape(b, c, h, w)
