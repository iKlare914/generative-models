from torch.utils.data import DataLoader
from generative_model.logger import get_logger
from generative_model.sampler import DDIMSampler, DDPMSampler, TimestepSampler, display_image_uint8, FMSampler, FMTimestepSampler
from generative_model.sampler import CFGDDPMSampler, CFGDDIMSampler, CFGFMSampler
from generative_model.config import TrainerConfig
from generative_model.text_encoder import get_cifar10_prompt
from uuid import uuid4
from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import wandb
from pathlib import Path

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """Checkpoint and gradient utilities; concrete trainers own their training setup."""

    def __init__(self, *, model, optimizer, save_dir, training_method, resume=False):
        self.model = model
        self.device = next(model.parameters()).device
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


class Trainer(BaseTrainer):
    """Train an unconditional diffusion or flow matching model."""

    def __init__(
        self, *, model: nn.Module, data: DataLoader,
        diffusion_sampler: DDIMSampler | DDPMSampler | FMSampler,
        timestep_sampler: TimestepSampler | FMTimestepSampler,
        config: TrainerConfig,
    ):
        is_fm = isinstance(diffusion_sampler, FMSampler)
        expected_timesteps = FMTimestepSampler if is_fm else TimestepSampler
        if not isinstance(timestep_sampler, expected_timesteps):
            raise ValueError(f"{type(diffusion_sampler).__name__} requires {expected_timesteps.__name__}.")
        if config.use_torch_compile:
            model = th.compile(model, fullgraph=True)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        super().__init__(
            model=model, optimizer=optimizer, save_dir=config.save_dir,
            training_method='fm' if is_fm else 'ddpm', resume=config.resume,
        )
        self.lr = config.lr
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.weight_decay = config.weight_decay
        self.epoches = config.epoches
        self.save_interval_epoch = config.save_interval_epoch
        self.guided = config.guided
        self.log_samples_enabled = config.log_samples
        self.name = self.save_dir.name if self.save_dir is not None else 'dry-run'
        self.dataloader = data
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_sampler.model = self.model
        self.timestep_sampler = timestep_sampler
        self.optimizer.zero_grad()
        if self.log_samples_enabled:
            generator = th.Generator().manual_seed(114514)
            self.fixed_noise = th.randn((9, 3, self.image_size, self.image_size), generator=generator).to(self.device)

    @th.no_grad()
    def log_samples(self, run, fixed_noise, epoch: int):
        self.model.eval()
        samples = self.diffusion_sampler.sample(fixed_noise.clone())
        samples = display_image_uint8(samples)
        run.log({
            "epoches": epoch,
            "global_steps": self.global_steps,
            "samples": [wandb.Image(img) for img in samples],
        })
        self.model.train()

    def train(self):
        self.resume() # resume checkpoint if resume enabled

        self.model.train()
        with wandb.init(project="Diffusion-experiment", name=self.name, id=self.wandb_runid, resume='allow') as run:
            run.define_metric("global_steps")
            run.define_metric("epoches")
            run.define_metric("train/*", step_metric="global_steps")
            run.define_metric("samples", step_metric="epoches")
            with tqdm(range(self.start_epoch + 1, self.epoches + 1), desc="Training Epoch") as pbar:
                new_steps = 0
                for epoch in pbar:
                    total_loss = 0.
                    run.log({"epoches": epoch})
                    for batch_img, batch_label in self.dataloader:
                        new_steps += 1
                        batch_img = batch_img.to(self.device)
                        # classifier guidance is currently not supported, so labels are ignored
                        # batch_label =  batch_label.to(self.device)
                        self.optimizer.zero_grad()
                        # weight currentlty unused
                        t, weights = self.timestep_sampler.sample(batch_img.shape[0], self.device)
                        with th.autocast(device_type="cuda", dtype=th.bfloat16, enabled=self.use_bf16_autocast):
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
                            self.log_samples(run, self.fixed_noise, epoch)


class CFGTrainer(BaseTrainer):
    """Train a CFG model and log generated images with their prompt captions."""

    def __init__(
        self, *, model: nn.Module, data: DataLoader,
        diffusion_sampler: CFGDDIMSampler | CFGDDPMSampler | CFGFMSampler,
        timestep_sampler: TimestepSampler | FMTimestepSampler,
        config: TrainerConfig, fixed_prompts: list[str] | None = None, num_fixed_samples: int = 9,
    ):
        is_fm = isinstance(diffusion_sampler, CFGFMSampler)
        expected_timesteps = FMTimestepSampler if is_fm else TimestepSampler
        if not isinstance(timestep_sampler, expected_timesteps):
            raise ValueError(f"{type(diffusion_sampler).__name__} requires {expected_timesteps.__name__}.")
        if config.use_torch_compile:
            model = th.compile(model, fullgraph=True)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        super().__init__(
            model=model, optimizer=optimizer, save_dir=config.save_dir,
            training_method='fm' if is_fm else 'ddpm', resume=config.resume,
        )
        self.lr = config.lr
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.weight_decay = config.weight_decay
        self.epoches = config.epoches
        self.save_interval_epoch = config.save_interval_epoch
        self.guided = config.guided
        self.log_samples_enabled = config.log_samples
        self.name = self.save_dir.name if self.save_dir is not None else 'dry-run'
        self.dataloader = data
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_sampler.model = self.model
        self.timestep_sampler = timestep_sampler
        self.optimizer.zero_grad()
        self.fixed_prompts = list(fixed_prompts) if fixed_prompts is not None else []
        if self.log_samples_enabled:
            if fixed_prompts is None:
                self.fixed_prompts, self.fixed_context, self.fixed_attention_mask = get_cifar10_prompt(
                    num_fixed_samples, self.diffusion_sampler.text_encoder,
                    max_length=self.diffusion_sampler.eval_max_length, device=self.device,
                )
            else:
                self.fixed_context, self.fixed_attention_mask = self.diffusion_sampler.encode_prompts(self.fixed_prompts)
            if self.diffusion_sampler.guidance_scale != 1:
                self.diffusion_sampler._empty_condition(self.fixed_context.shape[1])
            generator = th.Generator().manual_seed(114514)
            self.fixed_noise = th.randn(
                (len(self.fixed_prompts), 3, self.image_size, self.image_size), generator=generator,
            ).to(self.device)

    @th.no_grad()
    def log_samples(self, run, fixed_noise, epoch: int):
        self.model.eval()
        samples = self.diffusion_sampler.sample(
            fixed_noise.clone(), context=self.fixed_context, attention_mask=self.fixed_attention_mask,
        )
        samples = display_image_uint8(samples)
        run.log({
            "epoches": epoch,
            "global_steps": self.global_steps,
            "samples": [wandb.Image(img, caption=prompt) for img, prompt in zip(samples, self.fixed_prompts)],
        })
        self.model.train()

    def train(self):
        self.resume() # resume checkpoint if resume enabled

        self.model.train()
        with wandb.init(project="Diffusion-experiment", name=self.name, id=self.wandb_runid, resume='allow') as run:
            run.define_metric("global_steps")
            run.define_metric("epoches")
            run.define_metric("train/*", step_metric="global_steps")
            run.define_metric("samples", step_metric="epoches")
            with tqdm(range(self.start_epoch + 1, self.epoches + 1), desc="Training Epoch") as pbar:
                new_steps = 0
                for epoch in pbar:
                    run.log({"epoches": epoch})
                    total_loss = 0.
                    for batch_img, (context, attention_mask) in self.dataloader:
                        new_steps += 1
                        batch_img = batch_img.to(self.device)
                        context = context.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        self.optimizer.zero_grad()
                        # weight currentlty unused
                        t, weights = self.timestep_sampler.sample(batch_img.shape[0], self.device)
                        with th.autocast(device_type="cuda", dtype=th.bfloat16, enabled=self.use_bf16_autocast):
                            loss = self.diffusion_sampler.get_loss(
                                self.model, batch_img, t, context=context, attention_mask=attention_mask,
                            )
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
                            self.log_samples(run, self.fixed_noise, epoch)
