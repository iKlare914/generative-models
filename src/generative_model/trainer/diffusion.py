"""DDPM/DDIM trainers with shared training and checkpoint utilities."""

from generative_model.sampler.diffusion import CFGDDPMSampler, DDPMSampler, TimestepSampler
from .common import _CFGTrainer, _Trainer


class Trainer(_Trainer):
    """Train an unconditional diffusion model with DDPM or DDIM."""

    sampler_type = DDPMSampler
    timestep_sampler_type = TimestepSampler
    training_method = 'ddpm'


class CFGTrainer(_CFGTrainer):
    """Train a classifier-free guided diffusion model with DDPM or DDIM."""

    sampler_type = CFGDDPMSampler
    timestep_sampler_type = TimestepSampler
    training_method = 'ddpm'
