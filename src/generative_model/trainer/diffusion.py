"""DDPM trainers with independent training implementations."""

import torch as th
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from generative_model.config import TrainerConfig
from generative_model.dataset.text_encoder import get_cifar10_prompt
from generative_model.logger import get_logger
from generative_model.sampler.diffusion import (
    CFGDDPMSampler, DDPMSampler, TimestepSampler,
)
from generative_model.sampler.utils import display_image_uint8

from .trainer_base import BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    """Train an unconditional diffusion model with DDPM."""

    sampler_type = DDPMSampler
    timestep_sampler_type = TimestepSampler
    training_method = 'ddpm'

    def __init__(
        self, *, model: nn.Module, data: DataLoader,
        diffusion_sampler: DDPMSampler,
        timestep_sampler: TimestepSampler,
        config: TrainerConfig,
    ):
        if not isinstance(diffusion_sampler, self.sampler_type):
            raise ValueError(f"{type(self).__name__} received an incompatible sampler: {type(diffusion_sampler).__name__}.")
        expected_timesteps = self.timestep_sampler_type
        if not isinstance(timestep_sampler, expected_timesteps):
            raise ValueError(f"{type(diffusion_sampler).__name__} requires {expected_timesteps.__name__}.")
        if config.use_torch_compile:
            model = th.compile(model, fullgraph=True)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        super().__init__(
            model=model, optimizer=optimizer, save_dir=config.save_dir,
            training_method=self.training_method, resume=config.resume,
            use_ema=config.use_ema, ema_decay=config.ema_decay,
        )
        parameter = next(self.model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
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
                        self.update_ema()

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
    """Train a classifier-free guided diffusion model with DDPM."""

    sampler_type = CFGDDPMSampler
    timestep_sampler_type = TimestepSampler
    training_method = 'ddpm'

    def __init__(
        self, *, model: nn.Module, data: DataLoader,
        diffusion_sampler: CFGDDPMSampler,
        timestep_sampler: TimestepSampler,
        config: TrainerConfig, fixed_prompts: list[str] | None = None, num_fixed_samples: int = 9,
    ):
        if not isinstance(diffusion_sampler, self.sampler_type):
            raise ValueError(f"{type(self).__name__} received an incompatible sampler: {type(diffusion_sampler).__name__}.")
        expected_timesteps = self.timestep_sampler_type
        if not isinstance(timestep_sampler, expected_timesteps):
            raise ValueError(f"{type(diffusion_sampler).__name__} requires {expected_timesteps.__name__}.")
        if config.use_torch_compile:
            model = th.compile(model, fullgraph=True)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        super().__init__(
            model=model, optimizer=optimizer, save_dir=config.save_dir,
            training_method=self.training_method, resume=config.resume,
            use_ema=config.use_ema, ema_decay=config.ema_decay,
        )
        parameter = next(self.model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
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
                        self.update_ema()

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
