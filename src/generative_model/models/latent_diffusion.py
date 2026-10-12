"""Compose a DiT with a frozen Stable Diffusion AutoencoderKL."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch as th
from torch import nn

from .dit import DiT

if TYPE_CHECKING:
    from diffusers import AutoencoderKL


class LatentDiffusion(nn.Module):
    """
    Latent diffusion using pretrained VAE
    """
    def __init__(self, dit: DiT, vae: AutoencoderKL):
        super().__init__()
        if dit.in_channel != vae.config.latent_channels:
            raise ValueError('DiT input channels must match VAE latent channels.')

        self.dit = dit
        self.vae = vae.eval().requires_grad_(False)
        self.scaling_factor = float(vae.config.scaling_factor)
        self.downsample_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    @classmethod
    def from_pretrained_vae(
        cls, dit: DiT,
        vae_model_name_or_path: str | Path = 'stabilityai/sdxl-vae',
        **vae_kwargs,
    ) -> LatentDiffusion:
        """Load the HF VAE; kwargs go to AutoencoderKL.from_pretrained.

        A local pretrained directory also works. Diffusers is imported only
        here, so importing other models does not require the optional extra.
        Move the resulting wrapper with ``model.to(device)`` before use.
        """
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(str(vae_model_name_or_path), **vae_kwargs)
        return cls(dit, vae)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """Unbatched (C, H, W) shape for latent noise and training targets."""
        return (self.dit.in_channel, *self.dit.image_size)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Expected unbatched image input shape."""
        h, w = self.dit.image_size
        return (self.vae.config.in_channels, h * self.downsample_factor, w * self.downsample_factor)

    def train(self, mode: bool = True) -> LatentDiffusion:
        # set DiT and VAE to train mode recursively
        super().train(mode)
        # freeze pretrained VAE
        self.vae.eval().requires_grad_(False)
        return self

    def _vae_input(self, tensor: th.Tensor) -> th.Tensor:
        # An outer model.to(dtype=...) may disalign the VAE's precision policy.
        parameter = next(self.vae.parameters())
        if self.vae.config.force_upcast and parameter.dtype != th.float32:
            self.vae.float()
            parameter = next(self.vae.parameters())
        return tensor.to(device=parameter.device, dtype=parameter.dtype)

    @th.no_grad()
    def encode_images(
        self, images: th.Tensor, *, sample_posterior: bool = True,
        generator: th.Generator | None = None,
    ) -> th.Tensor:
        """Encode images in [-1, 1] to scaled latents on the DiT device/dtype.

        Set sample_posterior=False to use the posterior mode deterministically.
        If supplied, the generator must match the VAE device.
        """
        if images.ndim != 4 or tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(f'Expected image shape [B, {self.image_shape}], got {tuple(images.shape)}.')
        images = self._vae_input(images)
        # disable outer autocast
        with th.autocast(device_type=images.device.type, enabled=False):
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
            latents = latents * self.scaling_factor
        parameter = next(self.dit.parameters())
        return latents.to(device=parameter.device, dtype=parameter.dtype)

    @th.no_grad()
    def decode_latents(self, latents: th.Tensor) -> th.Tensor:
        """Undo scaling and decode on the VAE device/dtype; no display clamping."""
        if latents.ndim != 4 or tuple(latents.shape[1:]) != self.latent_shape:
            raise ValueError(f'Expected latent shape [B, {self.latent_shape}], got {tuple(latents.shape)}.')

        latents = self._vae_input(latents)
        with th.autocast(device_type=latents.device.type, enabled=False):
            return self.vae.decode(latents / self.scaling_factor).sample

    def forward(
        self, z_t: th.Tensor, t: th.Tensor,
        context: th.Tensor | None = None,
        attention_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Predict latent noise/velocity; accepts existing CFG sampler arguments."""
        return self.dit(z_t, t, context, attention_mask)
