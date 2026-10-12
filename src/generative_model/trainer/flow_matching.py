"""Flow-matching trainers with continuous training timesteps."""

from generative_model.sampler.flow_matching import CFGFMSampler, FMSampler, FMTimestepSampler
from .common import _CFGTrainer, _Trainer


class FMTrainer(_Trainer):
    """Train an unconditional flow-matching model."""

    sampler_type = FMSampler
    timestep_sampler_type = FMTimestepSampler
    training_method = 'fm'


class CFGFMTrainer(_CFGTrainer):
    """Train a classifier-free guided flow-matching model."""

    sampler_type = CFGFMSampler
    timestep_sampler_type = FMTimestepSampler
    training_method = 'fm'
