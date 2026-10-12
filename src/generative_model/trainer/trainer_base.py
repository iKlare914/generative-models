"""Abstract trainer with checkpoint, EMA and gradient utilities."""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import torch as th


class BaseTrainer(ABC):
    """Checkpoint, EMA and gradient utilities shared by concrete trainers."""

    def __init__(self, *, model, optimizer, save_dir, training_method, resume=False,
                 use_ema=True, ema_decay=None, ema_target=None):
        if use_ema and ema_decay is None:
            raise ValueError('ema_decay must be provided when use_ema=True')
        if ema_decay is not None and not 0 <= ema_decay < 1:
            raise ValueError('ema_decay must be in [0, 1)')
        self.model = model
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        # Latent trainers average only the DiT, without duplicating the frozen VAE.
        self.ema_target = model if ema_target is None else ema_target
        self.ema_model = deepcopy(self.ema_target).eval().requires_grad_(False) if use_ema else None
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.use_bf16_autocast = False
        if self.device.type == "cuda" and th.cuda.is_available():
            with th.cuda.device(self.device):
                self.use_bf16_autocast = th.cuda.is_bf16_supported(including_emulation=False)
        self.optimizer = optimizer
        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.training_method = training_method
        self.resume_enabled = resume
        self.start_epoch = 0
        self.wandb_runid = uuid4().hex[:16]
        self.global_steps = 0

    @abstractmethod
    def train(self):
        """Run this trainer's training loop."""
        raise NotImplementedError

    @abstractmethod
    def log_samples(self, run, fixed_noise, epoch: int):
        """Generate and log samples together with their training epoch."""
        raise NotImplementedError

    def save(self, epoch: int, loss: float) -> None:
        """Save training state and optional EMA weights in the same checkpoint."""
        if self.save_dir is None:
            return

        filename = f"{epoch:08d}_{loss:012.6f}_checkpoint.pt"
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "last_epoch": epoch,
            "wandb_runid": self.wandb_runid,
            "global_steps": self.global_steps,
            "training_method": self.training_method,
        }
        if self.use_ema:
            checkpoint['ema_model'] = self.ema_model.state_dict()
        th.save(checkpoint, self.save_dir / filename)

    def load(self, path: str | Path) -> None:
        """Restore model and optimizer state dicts from a single .pt file."""
        checkpoint = th.load(path, map_location='cpu', weights_only=True)
        # Checkpoints created before FM support contain diffusion noise predictors.
        checkpoint_method = checkpoint.get('training_method', 'ddpm')
        if checkpoint_method != self.training_method:
            raise ValueError(
                f"Checkpoint training method is {checkpoint_method!r}, "
                f"but this trainer uses {self.training_method!r}."
            )
        self.model.load_state_dict(checkpoint["model"])
        if self.use_ema:
            self.ema_model.load_state_dict(checkpoint.get('ema_model', self.ema_target.state_dict()))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint['last_epoch']
        self.wandb_runid = checkpoint['wandb_runid']
        self.global_steps = checkpoint['global_steps']

    def resume(self) -> None:
        """Load the checkpoint with the latest epoch when resume is enabled."""
        if not self.resume_enabled or self.save_dir is None:
            return

        checkpoints = sorted(self.save_dir.glob("*_*_checkpoint.pt"))
        if not checkpoints:
            return

        self.load(checkpoints[-1])

    @th.no_grad()
    def update_ema(self) -> None:
        """Average parameters after an optimizer step and copy model buffers."""
        if not self.use_ema:
            return
        ema_parameters = list(self.ema_model.parameters())
        model_parameters = list(self.ema_target.parameters())
        th._foreach_lerp_(ema_parameters, model_parameters, 1 - self.ema_decay)

        # Buffers include running statistics and integer counters, not optimized weights.
        # Only BatchNorm need this
        # for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
        #     ema_buffer.copy_(buffer)

    def get_grad_norm(self) -> float:
        """Return the L2 norm of all gradients flattened into one vector.

        Parameters without gradients are skipped. Return 0.0 if none exist.
        """
        grads = [
            param.grad.detach().reshape(-1)
            for param in self.model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return 0.0
        grad_vector = th.cat(grads)
        return th.linalg.vector_norm(grad_vector, ord=2).item()

