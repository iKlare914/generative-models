import math
import torch as th
from tqdm.auto import tqdm
from generative_model.logger import get_logger
from typing import Literal
import numpy as np

logger = get_logger(__name__)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def to_tensor(x, d_type=th.float32, device=device):
    """Convert array-like input to a tensor with the requested dtype and device."""
    return th.as_tensor(x, dtype=d_type, device=device)

def make_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    """Create a linear beta schedule."""
    if num_timesteps <= 50:
        raise ValueError("num_timesteps must be greater than 50 for a meaningful beta schedule.")
    scale = 1000 / num_timesteps
    beta_start = beta_start * scale
    beta_end = beta_end * scale
    betas = th.linspace(beta_start, beta_end, num_timesteps)
    if th.any(betas <= 0) or th.any(betas >= 1):
        raise ValueError("Beta values must be in the range (0, 1).")
    return betas

def display_image_uint8(img: th.Tensor):
    """
    Convert image from diffusion (range [-1, 1]) to normal uint8 image
    Args:
        img: Tensor [B, C ,H, W], value range from -1 to 1
    Returns:
        image: Tensor [B, C, H, W] with uint8 value
    """
    img = img.detach().cpu()
    img = (img + 1) * 127.5
    img = img.clamp(0, 255).type(th.uint8)
    return img

class DDPMSampler:
    def __init__(self, model, betas=None, d_type=th.float32, device=device, loss_fn=th.nn.MSELoss(), cond_fn=None, num_classes=None):
        """
        Args:
            model: The diffusion model to sample from.
            betas: Tensor or array of shape (T,). The beta schedule.
            d_type: The data type for the tensors.
            device: The device to run the sampling on.
        """
        self.model = model
        self.betas = to_tensor(betas, d_type=d_type, device=device) if betas is not None else None
        self.d_type = d_type
        self.cond_fn = cond_fn
        self.num_classes = num_classes
        self.device = th.device(device)
        if self.betas is not None:
            self._set_params(self.betas)
        self.loss_fn = loss_fn

    def load_config_from_model(self, model):
        """Load the configuration from a pre-trained model. If model has betas, use them; otherwise, use the default betas."""
        self.model = model.to(self.device)
        betas = getattr(model, "betas", None)
        self._set_params(self.betas if betas is None else betas)

    def _set_params(self, betas):
        """Compute schedule tensors using the configured dtype and device."""
        if betas is None:
            raise ValueError("Betas must be provided either in the model or as an argument.")
        self.betas = to_tensor(betas, d_type=self.d_type, device=self.device) if not isinstance(betas, th.Tensor) else betas.to(dtype=self.d_type, device=self.device)
        self.num_timesteps = len(self.betas)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = th.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = th.cat([th.tensor([1.0], dtype=self.d_type, device=self.device), self.alpha_bars[:-1]], dim=0)
        self.m1alpha_bars = 1 - self.alpha_bars
        self.m1alpha_bars_prev = 1 - self.alpha_bars_prev
        self.sqrt_m1alpha_bars = th.sqrt(self.m1alpha_bars)
        self.sqrt_alphas_bars = th.sqrt(self.alpha_bars)
        self.reciprocal_sqrt_alphas = th.rsqrt(self.alphas)
        self.reciprocal_sqrt_m1alpha_bars = th.rsqrt(self.m1alpha_bars)
        self.reciprocal_sqrt_alphas_bars = th.rsqrt(self.alpha_bars)
        self.sqrt_m1alpha_bars_prev = th.sqrt(self.m1alpha_bars_prev)
        
    def q_mean_std(self, x_0, t):
       """
       Compute the mean and standard deviation of q(x_t | x_0) for a given x_0 and timestep t.
       Args:
           x_0: The original data point [B, C, H, W].
           t: The timestep [T].

        Returns:
            mean: The mean of q(x_t | x_0).
            std: The standard deviation of q(x_t | x_0).
            x_t: A sample from q(x_t | x_0).
            noise: The standard gaussian noise added to x_0 by std to obtain x_t. [B, C, H, W]
       """
       if len(t.shape) != 1:
           raise ValueError(f"Expected t to be a 1D tensor, but got shape {t.shape}.")
       mean = self.sqrt_alphas_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_0
       std = self.sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
       noise = th.randn_like(x_0) 
       x_t = mean + noise * std
       return mean, std, x_t, noise

    def p_mean_std(self, x_t, t, guided=False, y=None, guidance_scale=1.0):
        """
        Compute the mean and standard deviation of p(x_{t-1} | x_t) for a given x_t and timestep t.
        Args:
            x_t: The noisy data point at timestep t, [B, C, H, W].
            t: The timestep, [B].
            guided: Whether to use guided sampling, defalult is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            mean: The mean of p(x_{t-1} | x_t).
            std: The standard deviation of p(x_{t-1} | x_t).
            x_t_prev: A sample from p(x_{t-1} | x_t).
            pred_standard_gaussian_noise: The predicted standard gaussian noise from the model, [B, C, H, W].
        """
        if len(t.shape) != 1:
            raise ValueError(f"Expected t to be a 1D tensor, but got shape {t.shape}.")
        pred_noise = self.model(x_t, t)
        pred_standard_gaussian_noise = pred_noise.clone()
        reciprocal_sqrt_m1alpha_bars_t = self.reciprocal_sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        betas_t = self.betas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_m1alpha_bars_prev_t = self.sqrt_m1alpha_bars_prev[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_m1alpha_bars_t = self.sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        reciprocal_sqrt_alphas_t = self.reciprocal_sqrt_alphas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if guided:
            if self.cond_fn is None or y is None:
                raise ValueError("Guided sampling requires a conditional function and labels y.")
            if self.cond_fn is not None and y is not None:
                pred_noise = pred_noise - self.cond_fn(x_t, t, y) * guidance_scale * sqrt_m1alpha_bars_t
        mean = reciprocal_sqrt_alphas_t * (x_t - betas_t * pred_noise * reciprocal_sqrt_m1alpha_bars_t)
        std = th.sqrt(betas_t) * sqrt_m1alpha_bars_prev_t * reciprocal_sqrt_m1alpha_bars_t
        noise = th.randn_like(x_t) * std
        x_t_prev = mean + noise
        return mean, std, x_t_prev, pred_standard_gaussian_noise


    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0) for a given x_0 and timestep t.
        Args:
            x_0: The original data point [B, C, H, W].
            t: The timestep [B].
        Returns:
            x_t: A sample from q(x_t | x_0).
            noise: The standard gaussian noise added to x_0 to obtain x_t. [B, C, H, W]
        """
        _, _ , x_t, noise = self.q_mean_std(x_0, t)
        return x_t, noise

    def p_sample(self, x_t, t, guided=False, y=None, guidance_scale=1.0):
        """
        Sample from p(x_{t-1} | x_t) for a given x_t and timestep t.
        Args:
            x_t: The noisy data point at timestep t, [B, C, H, W].
            t: The timestep, [T].
            guided: Whether to use guided sampling, default is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            x_t_prev: A sample from p(x_{t-1} | x_t).
            pred_standard_gaussian_noise: The predicted standard gaussian noise from the model, [B, C, H, W].
        """
        _, _, x_t_prev, pred_standard_gaussian_noise = self.p_mean_std(x_t, t, guided, y, guidance_scale)
        return x_t_prev, pred_standard_gaussian_noise


    def sample(self, x_T, guided=False, y=None, guidance_scale=1.0) -> th.Tensor:
        """
        Sample from the diffusion model starting from x_T and iteratively applying p_sample.
        Args:
            x_T: The initial noisy data point at timestep T, [B, C, H, W].
            guided: Whether to use guided sampling, default is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            x_0: The final denoised sample after T timesteps.
        """
        x_t = x_T
        with tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling") as progress:
            for step in progress:
                # progress.set_postfix_str(f'Timesteps: {step + 1} / {self.num_timesteps}')
                with th.no_grad():
                    t = th.full((x_t.shape[0],), step, dtype=th.long, device=x_t.device)
                    x_t, _ = self.p_sample(
                        x_t, t, guided=guided, y=y, guidance_scale=guidance_scale
                    )
        return x_t

    def get_loss(self, model, x_0, t):
        """
        Calculate loss for given samples
        """
        x_t, noise = self.q_sample(x_0, t)
        pred_noise = model(x_t, t)
        return self.loss_fn(pred_noise, noise)


class DDIMSampler(DDPMSampler):
    def __init__(self, model, spacing, randomness=0.0, betas=None, d_type=th.float32, device=device, loss_fn=th.nn.MSELoss() ,cond_fn=None, num_classes=None):
        super().__init__(model, betas, d_type, device, loss_fn, cond_fn, num_classes)
        self.spacing = spacing
        randomness = float(randomness)
        if not math.isfinite(randomness):
            raise ValueError("randomness must be finite.")
        self.randomness = min(max(randomness, 0.0), 1.0)
        self.loss_fn = loss_fn
        if self.randomness != randomness:
            logger.warning(
                "Randomness value %s is out of bounds [0, 1]. Clamped to %s.",
                randomness, self.randomness,
            )
        self.ddim_timesteps = self._do_timestep_mapping(self.spacing)
        self._transform_parameters()

    def _do_timestep_mapping(self, spacing):
        """
        Create a mapping from the ddpm timesteps to the new ddim timesteps based on the specified spacing.
        Args:
            spacing: The number of timesteps to skip in the original schedule.
        Returns:
            A tensor of shape [num_ddim_timesteps] containing the mapped timesteps.
        """
        selcted_timesteps = th.arange(0, self.num_timesteps, spacing, dtype=th.long, device=self.device)
        if selcted_timesteps[-1] != self.num_timesteps - 1:
            selcted_timesteps = th.cat([selcted_timesteps, th.tensor([self.num_timesteps - 1], dtype=th.long, device=self.device)])
        logger.info(
            "Selected %d timesteps from the original %d timesteps.",
            len(selcted_timesteps), self.num_timesteps,
        )
        return selcted_timesteps

    def _transform_parameters(self):
        """
        Transform the parameters of the original DDPM schedule to match the new DDIM schedule.
        """
        selected_alpha_bars = self.alpha_bars[self.ddim_timesteps]
        # Caution: prev here is relative to selected_alpha_bars, not the original alpha_bars
        selected_alpha_bars_prev = th.cat([th.tensor([1.0], dtype=self.d_type, device=self.device), selected_alpha_bars[:-1]], dim=0)
        # Caution: the betas here cannot be directly indexed from the original betas, since alpha is the major concerns
        ddim_betas = 1 - (selected_alpha_bars / selected_alpha_bars_prev)
        self._set_params(ddim_betas)
        self.sigmas = self.randomness * th.sqrt(self.betas) * th.sqrt(1 - selected_alpha_bars_prev) / th.sqrt(1 - selected_alpha_bars)

    def load_config_from_model(self, model):
        """Load the configuration from a pre-trained model. If model has betas, use them; otherwise, use the default betas."""
        self.model = model.to(self.device)
        self.betas = getattr(model, "betas", None)
        self.betas = to_tensor(self.betas, d_type=self.d_type, device=self.device)
        self._transform_parameters()

    def p_mean_std(self, x_t, t, guided=False, y=None, guidance_scale=1.0):
        """
        Compute the mean and standard deviation of p(x_{t-1} | x_t) using DDIM sampling for a given x_t and timestep t using the DDIM update rule.
        Args:
            x_t: The noisy data point at timestep t, [B, C, H, W].
            t: The timestep, [B].
            guided: Whether to use guided sampling, default is False.
            y: The labels for guided sampling, [B], required if guided is True.
            guidance_scale: The scale for guided sampling, default is 1.0.
        Returns:
            mean: The mean of p(x_{t-1} | x_t).
            std: The standard deviation of p(x_{t-1} | x_t).
            x_t_prev: A sample from p(x_{t-1} | x_t).
            pred_standard_gaussian_noise: The predicted standard gaussian noise from the model, [B, C, H, W].
        """
        if len(t.shape) != 1:
            raise ValueError(f"Expected t to be a 1D tensor, but got shape {t.shape}.")
        pred_noise = self.model(x_t, self.ddim_timesteps[t])
        pred_standard_gaussian_noise = pred_noise
        reciprocal_sqrt_alphas_t = self.reciprocal_sqrt_alphas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        m1alpha_bars_prev_t = self.m1alpha_bars_prev[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_m1alpha_bars_t = self.sqrt_m1alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sigma_t = self.sigmas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if guided:
            if self.cond_fn is None or y is None:
                raise ValueError("Guided sampling requires a conditional function and labels y.")
            if self.cond_fn is not None and y is not None:
                pred_noise = pred_noise - self.cond_fn(x_t, self.ddim_timesteps[t], y) * guidance_scale * sqrt_m1alpha_bars_t
        mean = reciprocal_sqrt_alphas_t * (x_t - pred_noise * sqrt_m1alpha_bars_t) + th.sqrt(m1alpha_bars_prev_t - th.square(sigma_t)) * pred_noise 
        std = sigma_t
        noise = th.randn_like(x_t) * std
        x_t_prev = mean + noise
        return mean, std, x_t_prev, pred_standard_gaussian_noise

    def get_loss(self, model, x_0, t):
        """
        Calculate loss for given samples
        """
        x_t, noise = self.q_sample(x_0, t)
        t = self.ddim_timesteps[t]
        pred_noise = model(x_t, t)
        return self.loss_fn(pred_noise, noise)

class TimestepSampler():
    def __init__(self, num_timestep, strategy: Literal['Uniform', 'Loss_weighted']):
        if strategy not in ['Uniform', 'Loss_weighted']:
            raise ValueError(f"Sampling strategy must be Uniform or Loss_weighted, but got {strategy}")
        self.num_timesteps = num_timestep
        self.strategy = strategy
        if strategy == 'Uniform':
            self.p = np.full((num_timestep,), 1.0 / num_timestep, dtype=np.float32)
        self.timesteps = np.arange(num_timestep)

    def sample(self, nums, device):
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
            t = np.random.choice(self.timesteps, nums, p=self.p)
            weights = np.ones((nums,), dtype=np.float32)
        elif self.strategy == 'Loss_weighted':
            raise NotImplementedError("Loss weighted sampling is not implemented")
        return th.tensor(t, dtype=th.long, device=device), th.tensor(weights, dtype=th.float32, device=device)
