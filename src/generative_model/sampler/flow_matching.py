"""Flow-matching sampling and continuous timestep sampling."""

import math
from typing import Literal

import torch as th
from tqdm.auto import tqdm

from generative_model.dataset.text_encoder import CLIPTextEncoder
from .utils import validate_cfg_skip_steps


def _step(velocity, x, t, h, solver, randomness=0.0):
    """Advance a physical-time velocity field; add new solver branches here."""
    if solver == 'euler':
        if randomness > 0:
            vf = velocity(x, t)
            noise = th.randn_like(x)
            drift = h * ((1 + (1 - t) * t * math.pow(randomness, 2) / 2) * vf - x * (math.pow(randomness, 2) / 2) * (1 - t))
            diffusion = noise * (1 - t) * randomness * math.sqrt(h)
            return x + drift + diffusion
        else:
            return x + h * velocity(x, t)
    raise ValueError(f"Unsupported solver {solver!r}; only 'euler' is supported.")


class FMSampler():
    """
    CondOT flow matching with physical time in [0, 1].

    Training and sampling both pass 1000 * t to the UNet time embedding.
    solver currently supports only Euler, which is the default.
    """
    def __init__(self, step, model, device, randomness = 0.0, loss_fn = th.nn.MSELoss(), *, solver: Literal['euler'] = 'euler'):
        step = float(step)
        if not math.isfinite(step) or not 0 < step <= 1:
            raise ValueError("FM step must be finite and in (0, 1].")
        if solver != 'euler':
            raise ValueError("solver must be 'euler'.")
        self.solver = solver
        self.randomness = float(randomness)
        self.h = step
        self.model = model
        self.device = th.device(device)
        self.sample_nums = math.ceil(1 / step)
        self.loss_fn = loss_fn

    def q_sample(self, x_T: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Calculate x_t = alpha_t * x_T + beta_t * z
        Args:
            x_T: Image tensor sampled from dataset, shape [B, C, H ,W]
            t: time step tensor from U[0, 1), shape [B]
        Returns:
            x_t: Interpolated image tensor, shape [B, C, H, W]
            noise: Standard gaussian noise added to image, shape [B, C, H, W]
        """
        if t.ndim != 1 or t.shape[0] != x_T.shape[0]:
            raise ValueError("Expected one timestep per image, with shape [B].")
        x_T = x_T.to(self.device)
        t = t.to(self.device)
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise = th.randn_like(x_T, device=self.device)
        x_t = t * x_T + (1 - t) * noise
        return x_t, noise

    @th.no_grad()
    def sample(self, x_0: th.Tensor) -> th.Tensor:
        """
        Integrate from Gaussian noise using the configured solver (Euler).
        Args:
            x_0: Standard gaussian noise, shape [B, C, H, W]
        Returns:
            x_T: True image reconstructed from gaussian noise, shape [B, C, H, W]
        """
        x_0 = x_0.to(self.device)

        def velocity(x, t):
            t_v = th.full((x.shape[0],), t, device=self.device, dtype=th.float32)
            return self.model(x, t_v * 1000)

        with tqdm(range(self.sample_nums), desc="Sampling") as pbar:
            for index in pbar:
                # Derive time from the index to avoid accumulating rounding error.
                t = index * self.h
                step_size = min(self.h, 1 - t)
                x_0 = _step(velocity, x_0, t, step_size, self.solver, self.randomness)
        return x_0

    def get_loss(self, model, x_0: th.Tensor, t: th.Tensor):
        """Regress the noise-to-data velocity x_data - noise at continuous time t."""
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        x_t, noise = self.q_sample(x_0, t)
        vf = model(x_t, t * 1000)
        loss = self.loss_fn(vf, x_0 - noise)
        return loss


class FMTimestepSampler():
    def __init__(self, strategy: Literal['Uniform', 'Loss_weighted']):
        if strategy not in ['Uniform', 'Loss_weighted']:
            raise ValueError(f"Sampling strategy must be Uniform or Loss_weighted, but got {strategy}")
        self.strategy = strategy

    def sample(self, nums, device) -> tuple[th.Tensor, th.Tensor]:
        """
        Sample nums timesteps from full timesteps
        Args:
            nums: number of timesteps needed
            device: device that result sits on
        Returns:
            timesteps: Tensor [nums] sampled
            weights: Tensor [nums] for weighted loss
        """
        if self.strategy == 'Uniform':
            return th.rand((nums,), device=device, dtype=th.float32), th.ones((nums,), device=device, dtype=th.float32)
        elif self.strategy == 'Loss_weighted':
            raise NotImplementedError(f"{self.strategy} sampling is not implemented")


class CFGFMSampler:
    """Flow matching with cached text conditions and guided Euler sampling."""

    def __init__(
        self, step, model, device, randomness=0.0, loss_fn=th.nn.MSELoss(), *, guidance_scale=1.0, eval_max_length=77,
        use_cfgzero_star=False, skip_steps=None,
        text_encoder: CLIPTextEncoder | None = None,
        text_model_name="openai/clip-vit-base-patch32",
        solver: Literal['euler'] = 'euler',
    ):
        self.h = float(step)
        self.randomness = float(randomness)
        if not 0 < self.h <= 1:
            raise ValueError("FM step must be in (0, 1].")
        if solver != 'euler':
            raise ValueError("solver must be 'euler'.")
        self.solver = solver
        self.sample_nums = math.ceil(1 / self.h)
        skip_steps = validate_cfg_skip_steps(use_cfgzero_star, skip_steps, self.sample_nums)
        self.model = model
        self.device = th.device(device)
        self.loss_fn = th.nn.MSELoss() if loss_fn is None else loss_fn
        self.guidance_scale = guidance_scale
        self.use_cfgzero_star = use_cfgzero_star
        self.skip_steps = skip_steps
        self.eval_max_length = eval_max_length
        self.text_encoder = text_encoder if text_encoder is not None else CLIPTextEncoder(text_model_name)
        self._empty_conditions: dict[int, tuple[th.Tensor, th.Tensor]] = {}

    def encode_prompts(self, prompts, *, max_length=None):
        """Delegate prompt encoding to the supplied CLIPTextEncoder."""
        length = self.eval_max_length if max_length is None else max_length
        return self.text_encoder.encode_prompts(prompts, max_length=length, device=self.device)

    def _empty_condition(self, max_length):
        """Keep one empty-text feature on the sampler device for each used length."""
        if max_length not in self._empty_conditions:
            self._empty_conditions[max_length] = self.encode_prompts([""], max_length=max_length)
        return self._empty_conditions[max_length]

    def q_sample(self, x_0, t):
        t = t.view(-1, 1, 1, 1)
        noise = th.randn_like(x_0)
        return t * x_0 + (1 - t) * noise, noise

    def get_loss(self, model, x_0, t, context, attention_mask):
        """Regress image - noise using supplied text features and model time 1000 * t."""
        x_0, t = x_0.to(self.device), t.to(self.device)
        context = context.to(device=self.device, dtype=x_0.dtype)
        attention_mask = attention_mask.to(device=self.device, dtype=th.bool)
        x_t, noise = self.q_sample(x_0, t)
        prediction = model(x_t, t * 1000, context, attention_mask)
        return self.loss_fn(prediction, x_0 - noise)

    @th.no_grad()
    def sample(self, x_0, prompts=None, guidance_scale=None, *, context=None, attention_mask=None):
        """Integrate CFG velocities, holding noise fixed for skip_steps with or without CFG-Zero*."""
        x_t = x_0.to(self.device)
        if context is None:
            prompts = [prompts] * len(x_t) if isinstance(prompts, str) else list(prompts)
            if len(prompts) != len(x_t):
                raise ValueError("Provide one prompt per image, or one string for the whole batch.")
            context, attention_mask = self.encode_prompts(prompts)
        context = context.to(device=self.device, dtype=x_t.dtype)
        attention_mask = attention_mask.to(device=self.device, dtype=th.bool)
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        empty, empty_mask = (None, None)
        if scale != 1:
            empty, empty_mask = self._empty_condition(context.shape[1])
            empty = empty.to(x_t.dtype).expand(len(x_t), -1, -1)
            empty_mask = empty_mask.expand(len(x_t), -1)
        was_training = self.model.training
        self.model.eval()

        def velocity(x, t):
            model_t = th.full((len(x),), t * 1000, dtype=th.float32, device=self.device)
            if scale == 1:
                return self.model(x, model_t, context, attention_mask)
            if scale == 0:
                return self.model(x, model_t, empty, empty_mask)
            conditional: th.Tensor = self.model(x, model_t, context, attention_mask)
            unconditional: th.Tensor = self.model(x, model_t, empty, empty_mask)
            if not self.use_cfgzero_star:
                return unconditional + scale * (conditional - unconditional)
            output_dtype = conditional.dtype
            b, c, h, w = conditional.shape
            conditional = conditional.reshape(b, -1).float()
            unconditional = unconditional.reshape(b, -1).float()
            numerator = (conditional * unconditional).sum(dim=1, keepdim=True)
            denominator = unconditional.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
            adjust_scale = numerator / denominator
            res: th.Tensor = (1 - scale) * adjust_scale * unconditional + scale * conditional
            return res.reshape(b, c, h, w).to(output_dtype)
        try:
            # Keep the initial state fixed during zero-init, retaining the original time grid.
            for index in tqdm(range(self.skip_steps, self.sample_nums), desc="CFG sampling"):
                t = index * self.h
                step_size = min(self.h, 1 - t)
                x_t = _step(velocity, x_t, t, step_size, self.solver, self.randomness)
        finally:
            self.model.train(was_training)
        return x_t
