"""Public training API."""

from .diffusion import CFGTrainer, Trainer
from .flow_matching import CFGFMTrainer, FMTrainer
from .latent_diffusion import CFGLatentTrainer, LatentTrainer
from .latent_flow_matching import CFGLatentFMTrainer, LatentFMTrainer
from .trainer_base import BaseTrainer

__all__ = [
    "BaseTrainer", "Trainer", "CFGTrainer", "FMTrainer", "CFGFMTrainer",
    "LatentTrainer", "CFGLatentTrainer", "LatentFMTrainer", "CFGLatentFMTrainer",
]
