import torch as th
from torch import nn
import torch.nn.functional as F
from .utils import zero_init
from typing import Literal
from generative_model.models.norms import AdaptiveLayerNorm
from generative_model.models.embedding import RoPE2D, getSinusoidalEmbedding2D

class QKVMHAttentionRoPE2D(nn.Module):
    def __init__(self, grid_size_h, grid_size_w, rope_theta=10000.0, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads
        self.rope = RoPE2D(grid_size_h, grid_size_w, rope_theta)

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor):
        """
        Args:
            q, k, v: Tensor [B, L, C]
        Returns:
            attention: Tensor [B, L, C]
        """
        b, l, c = q.shape
        h = self.num_heads
        if c % h != 0:
            raise ValueError(f"Channel num: {c} must can be divided by num heads: {h}")
        hc = c // h 
        
        q = self.rope(q.reshape(b, l, h, hc).transpose(1, 2)) # [B, H, L, HC]
        k = self.rope(k.reshape(b, l, h, hc).transpose(1, 2))
        v = v.reshape(b, l, h, hc).transpose(1, 2)

        # spda kernel fusion requires last dim with stride 1
        q, k, v = (
            o.contiguous() if o.stride(-1) != 1 else o
            for o in (q, k, v)
        )

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, l, -1) # [B, L, C]
        return attn


class QKVMHACrossAttentionRope2D(nn.Module):
    def __init__(self, grid_size_h, grid_size_w, rope_theta=10000.0, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads
        self.rope = RoPE2D(grid_size_h, grid_size_w, rope_theta)

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor, attention_mask: th.Tensor):
        """
        Args:
            q: Tensor [B, S, C]
            k: Tensor [B, L, C]
            v: Tensor [B, L, C]
            attention_mask: 0-1 Tensor [B, L]
        Returns:
            attention: Tensor [B, S, C]
        """
        s, l = q.shape[-2], k.shape[-2]
        b, c = q.shape[0], q.shape[-1]
        h = self.num_heads
        if c % h != 0:
            raise ValueError(f"Channel num: {c} must can be divided by num heads: {h}")
        hc = c // h

        # ref: https://github.com/Tencent-Hunyuan/HunyuanDiT/blob/cb709308d92e6c7e8d59d0dff41b74d35088db6a/hydit/modules/attn_layers.py#L301-L305
        # Use RoPE only for q (Image) in Cross attention
        q = self.rope(q.reshape(b, s, h, hc).transpose(1, 2)) # [B, H, S, HC]
        k = k.reshape(b, l, h, hc).transpose(1, 2)
        v = v.reshape(b, l, h, hc).transpose(1, 2)
        mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        # spda kernel fusion requires last dim with stride 1
        q, k, v = (
            o.contiguous() if o.stride(-1) != 1 else o
            for o in (q, k, v)
        )

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # [B, H, S, HC]
        return attn.transpose(1, 2).reshape(b, s, -1)

class QKVMHAttention(nn.Module):
    def __init__(self, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor):
        """
        Args:
            q, k, v: Tensor [B, L, C]
        Returns:
            attention: Tensor [B, L, C]
        """
        b, l, c = q.shape
        h = self.num_heads
        if c % h != 0:
            raise ValueError(f"Channel num: {c} must can be divided by num heads: {h}")
        hc = c // h 
        
        q = q.reshape(b, l, h, hc).transpose(1, 2) # [B, H, L, HC]
        k = k.reshape(b, l, h, hc).transpose(1, 2)
        v = v.reshape(b, l, h, hc).transpose(1, 2)

        # spda kernel fusion requires last dim with stride 1
        q, k, v = (
            o.contiguous() if o.stride(-1) != 1 else o
            for o in (q, k, v)
        )

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, l, -1) # [B, L, C]
        return attn


class QKVMHACrossAttention(nn.Module):
    def __init__(self, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor, attention_mask: th.Tensor):
        """
        Args:
            q: Tensor [B, S, C]
            k: Tensor [B, L, C]
            v: Tensor [B, L, C]
            attention_mask: 0-1 Tensor [B, L]
        Returns:
            attention: Tensor [B, S, C]
        """
        s, l = q.shape[-2], k.shape[-2]
        b, c = q.shape[0], q.shape[-1]
        h = self.num_heads
        if c % h != 0:
            raise ValueError(f"Channel num: {c} must can be divided by num heads: {h}")
        hc = c // h
        q = q.reshape(b, s, h, hc).transpose(1, 2) # [B, H, S, HC]
        k = k.reshape(b, l, h, hc).transpose(1, 2)
        v = v.reshape(b, l, h, hc).transpose(1, 2)
        mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()

        # spda kernel fusion requires last dim with stride 1
        q, k, v = (
            o.contiguous() if o.stride(-1) != 1 else o
            for o in (q, k, v)
        )

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # [B, H, S, HC]
        return attn.transpose(1, 2).reshape(b, s, -1)

class ConvAttentionBlock(nn.Module):
    def __init__(self, channel, num_heads=1, o_proj_zeroinit=False):
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
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        q, k ,v = self.qkv_proj(self.norm(x)).chunk(3, dim=1)
        q = q.transpose(1, 2) # [B, (H*W)], C]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn: th.Tensor = self.attention(q, k, v) # [B, (H*W), C]
        o = self.o_proj(attn.transpose(1, 2))
        return (o + x).reshape(b, c, h, w)



class ConvCrossAttentionBlock(nn.Module):
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
            context: Tensor [B, L, FC]
            attention_mask: 0-1 Tensor [B, L]
        Returns:
            result: Tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)
        q = self.q_proj(self.norm(x)).transpose(1, 2) # [B, S, C]
        k = self.k_proj(context) # [B, L ,C]
        v = self.v_proj(context)

        attn: th.Tensor = self.attention(q, k, v, attention_mask) # [B, S, C]
        o = self.o_proj(attn.transpose(1, 2))
        return (x + o).reshape(b, c, h, w)


