"""Flow matching in latent space, returning decoded floating-point images.

Pass a LatentDiffusion as model and scaled latent noise to sample(). The inherited
loss methods still take scaled latents; only sample() performs VAE decoding.
Image display conversion remains the caller's responsibility.
"""

import torch as th

from generative_model.models.latent_diffusion import LatentDiffusion
from .flow_matching import CFGFMSampler, FMSampler


class LatentFMSampler(FMSampler):
    """Integrate the latent velocity field, then undo scaling and decode once."""

    @th.no_grad()
    def sample(self, x_0: th.Tensor) -> th.Tensor:
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(x_0)
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)


class CFGLatentFMSampler(CFGFMSampler):
    """Preserve CFG flow integration and time scaling, returning decoded images."""

    @th.no_grad()
    def sample(self, x_0, prompts=None, guidance_scale=None, *, context=None, attention_mask=None):
        if not isinstance(self.model, LatentDiffusion):
            raise TypeError('model must be a LatentDiffusion instance.')
        was_training = self.model.training
        self.model.eval()
        try:
            latents = super().sample(
                x_0, prompts=prompts, guidance_scale=guidance_scale,
                context=context, attention_mask=attention_mask,
            )
            return self.model.decode_latents(latents)
        finally:
            self.model.train(was_training)
