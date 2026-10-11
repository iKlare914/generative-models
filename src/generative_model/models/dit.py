import torch as th
from torch import nn
import math
from generative_model.models.attention import (
    AttentionBlock,
    AttentionBlockRoPE2D,
    CrossAttentionBlock,
    CrossAttentionBlockRoPE2D,
)
from generative_model.models.norms import AdaptiveLayerNorm
from generative_model.models.utils import ada_modulation
from typing import Literal
from generative_model.models.embedding import getSinusoidalEmbedding2D, TimeStepEmbedder, PatchEmbedding

# ref: https://github.com/facebookresearch/DiT/blob/main/models.py

# TODO
# 1. DiTBlock
# 2. DiT
# 3. Patchify -- UnPatchify 
# 4. 2d sinusoidal positional embedding - done
# 5. 2d-Rope - done
# 6. AdaLN condition injection - done
# 7. CrossAttention - done
# 8. weight init

class LastLayer(nn.Module):
    def __init__(self, patch_size: int, channel: int, d_model: int, d_time_emb: int, t_freq: int, use_adaln_zero: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaln = AdaptiveLayerNorm(d_model, d_model, exp_scale=2, is_zero_init=use_adaln_zero)
        self.linear = nn.Linear(d_model, channel * patch_size ** 2)
        self.time_emb_layer = TimeStepEmbedder(d_time_emb, d_model, t_freq)

    def forward(self, x: th.Tensor, t: th.Tensor):
        t_emb = self.time_emb_layer(t)
        scale, shift = self.adaln(t_emb).chunk(2, dim=-1)
        return self.linear(ada_modulation(self.norm(x), scale, shift)) # [B, L, C*p*p]

class DiTBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_time_emb: int,
            num_heads: int,
            grid_size_h: int,
            grid_size_w: int,
            *,
            extra_feature: bool = False,
            d_feature: int = None,
            use_rope: bool = True,
            t_freq: float = 10000.0,
            pe_freq: float = 10000.0,
            rope_cross_q: bool = False,
            rope_cross_k: bool = False,
            use_adaln_zero: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_time_emb = d_time_emb
        self.num_heads = num_heads
        self.extra_feature = extra_feature
        if extra_feature and d_feature is None:
            raise ValueError("d_feature must be set when using extra_feature")
        self.d_feature = d_feature
        self.t_freq = t_freq
        self.pe_freq = pe_freq

        # Independent time embeddings
        self.time_emb_layer = TimeStepEmbedder(t_dim=d_time_emb, hidden_dim=d_model, freq=t_freq)

        # Attentions
        if use_rope:
            self.attn = AttentionBlockRoPE2D(
                d_model=d_model, grid_size_h=grid_size_h, grid_size_w=grid_size_w,
                rope_theta=pe_freq, num_heads=num_heads,
            )
            if extra_feature:
                self.cross_attn = CrossAttentionBlockRoPE2D(
                    d_model=d_model, d_feature=d_feature,
                    grid_size_h=grid_size_h, grid_size_w=grid_size_w,
                    rope_theta=pe_freq, num_heads=num_heads,
                    cross_rope_q=rope_cross_q, cross_rope_k=rope_cross_k,
                )
        else:
            self.attn = AttentionBlock(d_model=d_model, num_heads=num_heads)
            if extra_feature:
                self.cross_attn = CrossAttentionBlock(
                    d_model=d_model, d_feature=d_feature, num_heads=num_heads,
                )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * d_model, d_model)
        )

        # Norms
        self.norm_1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if extra_feature:
            self.norm_c = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaln_norm = AdaptiveLayerNorm(d_model, d_model, exp_scale=9 if extra_feature else 6, is_zero_init=use_adaln_zero)

    def forward(self, x: th.Tensor, t: th.Tensor, c: th.Tensor = None, c_attention_mask: th.Tensor = None) -> th.Tensor:
        if self.extra_feature and (c is None or c_attention_mask is None):
            raise ValueError("missing c and c_attention_mask")
        t_emb = self.time_emb_layer(t)
        if self.extra_feature:
            scale_attn, shift_attn, gate_attn, scale_cattn, shift_cattn, gate_cattn, scale_ffn, shift_ffn, gate_ffn = self.adaln_norm(t_emb).chunk(9, dim=-1)
        else:
            scale_attn, shift_attn, gate_attn, scale_ffn, shift_ffn, gate_ffn = self.adaln_norm(t_emb).chunk(6, dim=-1)

        x = x + self.attn(ada_modulation(self.norm_1(x), scale_attn, shift_attn)) * gate_attn
        if self.extra_feature:
            x = x + self.cross_attn(ada_modulation(self.norm_c(x), scale_cattn, shift_cattn), c, c_attention_mask)* gate_cattn
        x = x + self.ffn(ada_modulation(self.norm_2(x), scale_ffn, shift_ffn)) * gate_ffn
        return x

class DiT(nn.Module):
    def __init__(
            self,
            image_size: tuple[int,int],
            patch_size: int,
            in_channel: int,
            d_model: int,
            d_time_emb: int,
            num_heads: int,
            num_layers: int,
            grid_size_h: int,
            grid_size_w: int,
            *,
            extra_feature: bool = False,
            d_feature: int = None,
            use_rope: bool = True,
            t_freq: float = 10000.0,
            pe_freq: float = 10000.0,
            rope_cross_q: bool = False,
            rope_cross_k: bool = False,
            use_adaln_zero: bool = True,
    ):
        super().__init__()
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError(f"image_size must contain two positive integers, got {image_size!r}")

        if type(patch_size) is not int or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")

        for i, size in enumerate(image_size):
            if type(size) is not int or size <= 0:
                raise ValueError(f"image_size[{i}] must be a positive integer")
            if size % patch_size != 0:
                raise ValueError(f"image_size[{i}] must be divisible by patch_size")

        if not use_rope:
            self.register_buffer(
                'pos_emb',
                getSinusoidalEmbedding2D(pe_freq, d_model, grid_size_h, grid_size_w),
                persistent=False
            )

        self.use_rope = use_rope
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel

        self.patch_embed = PatchEmbedding(in_channel, d_model, image_size, patch_size)
        self.dit_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.dit_layers.append(
                DiTBlock(
                    d_model,
                    d_time_emb,
                    num_heads,
                    grid_size_h,
                    grid_size_w,
                    extra_feature=extra_feature,
                    d_feature=d_feature,
                    use_rope=use_rope,
                    t_freq=t_freq,
                    pe_freq=pe_freq,
                    rope_cross_q=rope_cross_q,
                    rope_cross_k=rope_cross_k,
                    use_adaln_zero=use_adaln_zero,
                )
            )

        self.last_layer = LastLayer(patch_size, in_channel, d_model, d_time_emb, t_freq ,use_adaln_zero)

    def unpatchify(self, x: th.Tensor) -> th.Tensor:
        """
        Unpatchify Images
        Args:
            x: Tensor [B, L, C*p*p], L = HW/p**2
        Returns:
            image: Tensor [B, C, H, W]
        """
        b, l, _ = x.shape
        h, w = self.image_size
        p = self.patch_size
        c = self.in_channel

        x = x.view(b, h // p, w // p, p, p, c) # [B, H/p, Wp, p, p, C]
        x = x.permute(0, 5, 1, 3, 2, 4) # [B, C, H/p, p, W/p, p]

        return x.reshape(b, c, h, w)

    def forward(self, x: th.Tensor, t: th.Tensor, c: th.Tensor = None, c_attention_mask: th.Tensor = None):
        """
        Args:
            x: Tensor [B, C, H, W]
            t: timesteps Tensor [B]
            c: features that used to guide the model , Tensor [B, L, d_feature]
            c_attention_mask: attention mask for cross attention, bool tensor [B, L]
        """
        x = self.patch_embed(x) # [B, HW/p**2, d_model]
        if not self.use_rope:
            x = x + self.pos_emb

        for layer in self.dit_layers:
            x = layer(x, t, c, c_attention_mask)

        last_hidden_state = self.last_layer(x, t) # [B, HW/p**w, C*p*p]
        return self.unpatchify(last_hidden_state)
