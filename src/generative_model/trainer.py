from generative_model.load_dataset import getCifarLoader, DataLoader
from generative_model.logger import get_logger
from generative_model.sampler import DDIMSampler, DDPMSampler, TimestepSampler, display_image_uint8, FMSampler, FMTimestepSampler
from generative_model.config import TrainerConfig
from uuid import uuid4

import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import wandb
from pathlib import Path
from time import time
from datetime import datetime

logger = get_logger(__name__)

class Trainer():
    def __init__(
            self,
            *,
            model: nn.Module,
            data: DataLoader,
            diffusion_sampler: DDIMSampler | DDPMSampler | FMSampler,
            timestep_sampler: TimestepSampler | FMTimestepSampler,
            config: TrainerConfig 
    ):
        is_fm = isinstance(diffusion_sampler, FMSampler)
        expected_timesteps = FMTimestepSampler if is_fm else TimestepSampler
        if not isinstance(timestep_sampler, expected_timesteps):
            raise ValueError(f"{type(diffusion_sampler).__name__} requires {expected_timesteps.__name__}.")
        self.training_method = 'fm' if is_fm else 'ddpm'
        self.lr = config.lr
        self.dropout = config.dropout
        self.dataloader = data
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.weight_decay = config.weight_decay
        self.epoches = config.epoches
        self.save_interval_epoch = config.save_interval_epoch
        self.resume_enabled = config.resume
        self.guided = config.guided
        self.log_samples_enabled = config.log_samples
        self.save_dir: Path = config.save_dir 
        if self.save_dir:
                self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = self.save_dir.name if self.save_dir else 'dry-run'
        self.model = model
        self.device = next(model.parameters()).device
        self.diffusion_sampler = diffusion_sampler
        if config.use_torch_compile:
            self.model = th.compile(model, fullgraph=True)
        self.timestep_sampler = timestep_sampler
        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
        self.optimizer.zero_grad()
        self.start_epoch = 0 
        self.wandb_runid = uuid4().hex[:16]
        self.global_steps = 0

        if self.log_samples_enabled:
            generator = th.Generator().manual_seed(114514)
            self.fixed_noise = th.randn((9, 3, self.image_size, self.image_size), generator=generator).to(self.device)

    def train(self):
        self.resume() # resume checkpoint if resume enabled

        self.model.train()
        with wandb.init(project="Diffusion-experiment", name=self.name, id=self.wandb_runid, resume='allow') as run:
            run.define_metric("global_steps")
            run.define_metric("train/*", step_metric="global_steps")
            with tqdm(range(self.start_epoch + 1, self.epoches + 1), desc="Training Epoch") as pbar:
                new_steps = 0
                for epoch in pbar:
                    total_loss = 0.
                    for batch_img, batch_label in self.dataloader:
                        new_steps += 1
                        batch_img = batch_img.to(self.device)
                        # batch_label =  batch_label.to(self.device)
                        self.optimizer.zero_grad()
                        # weight currentlty unused
                        t, weights = self.timestep_sampler.sample(batch_img.shape[0], self.device)
                        loss = self.diffusion_sampler.get_loss(self.model, batch_img, t)
                        loss.backward()
                        total_loss += loss.item()
                        grad_norm = self.get_grad_norm()
                        if grad_norm >= 1:
                            logger.warning(f"Gradient norm({grad_norm}) > 1, clipped to 1")
                            grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2, error_if_nonfinite=True).item()
                        self.optimizer.step()

                        # log
                        run.log({
                            "train/loss": loss.item(),
                            "train/grad_norm": grad_norm,
                            "global_steps": self.global_steps + new_steps
                        })

                    if epoch % self.save_interval_epoch  == 0 or epoch == self.epoches:
                        self.global_steps += new_steps
                        new_steps = 0
                        self.save(epoch, total_loss)
                        if self.log_samples_enabled:
                            self.log_samples(run, self.fixed_noise)

    @th.no_grad()
    def log_samples(self, run, fixed_noise):
        self.model.eval()
        samples = self.diffusion_sampler.sample(fixed_noise.clone())

        # Convert [-1, 1] data to [0, 255]
        samples = display_image_uint8(samples)

        run.log({
            "global_steps": self.global_steps,
            "samples": [wandb.Image(img) for img in samples],
        })

        # turn on training
        self.model.train()

    def save(self, epoch: int, loss: float) -> None:
        """Save both state dicts under an epoch-sortable checkpoint filename."""
        if self.save_dir is None:
            return

        filename = f"{epoch:08d}_{loss:012.6f}_checkpoint.pt"
        th.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "last_epoch": epoch,
            "wandb_runid": self.wandb_runid,
            "global_steps": self.global_steps,
            "training_method": self.training_method,
        }, self.save_dir / filename)

    def load(self, path: str | Path) -> None:
        """Restore model and optimizer state dicts from a single .pt file."""
        checkpoint = th.load(path, map_location=self.device, weights_only=True)
        # Checkpoints created before FM support contain diffusion noise predictors.
        checkpoint_method = checkpoint.get('training_method', 'ddpm')
        if checkpoint_method != self.training_method:
            raise ValueError(
                f"Checkpoint training method is {checkpoint_method!r}, "
                f"but this trainer uses {self.training_method!r}."
            )
        self.model.load_state_dict(checkpoint["model"])
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
