"""DDPM/DDIM sampling in latent space, returning decoded floating-point images.

Pass a LatentDiffusion as model and scaled latent noise to sample(). The inherited
loss methods still take scaled latents; only sample() performs VAE decoding.
Image display conversion remains the caller's responsibility.
"""

import torch as th

from generative_model.models.latent_diffusion import LatentDiffusion
from .diffusion import CFGDDIMSampler, CFGDDPMSampler, DDIMSampler, DDPMSampler


class LatentDDPMSampler(DDPMSampler):
    """Use the DDPM reverse process, then undo latent scaling and decode once."""

    @th.no_grad()
    def sample(self, x_T, guided=False, y=None, guidance_scale=1.0) -> th.Tensor:
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(x_T, guided=guided, y=y, guidance_scale=guidance_scale)
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)


class LatentDDIMSampler(DDIMSampler):
    """Use the DDIM reverse process, then undo latent scaling and decode once."""

    @th.no_grad()
    def sample(self, x_T, guided=False, y=None, guidance_scale=1.0) -> th.Tensor:
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(x_T, guided=guided, y=y, guidance_scale=guidance_scale)
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)


class CFGLatentDDPMSampler(CFGDDPMSampler):
    """Preserve CFG DDPM guidance and timestep behavior, returning decoded images."""

    @th.no_grad()
    def sample(self, x_T, prompts=None, guidance_scale=None, *, context=None, attention_mask=None):
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(
                x_T, prompts=prompts, guidance_scale=guidance_scale,
                context=context, attention_mask=attention_mask,
            )
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)


class CFGLatentDDIMSampler(CFGDDIMSampler):
    """Preserve CFG DDIM guidance and timestep behavior, returning decoded images."""

    @th.no_grad()
    def sample(self, x_T, prompts=None, guidance_scale=None, *, context=None, attention_mask=None):
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(
                x_T, prompts=prompts, guidance_scale=guidance_scale,
                context=context, attention_mask=attention_mask,
            )
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)
