"""Shared tensor, image and CFG sampling helpers."""

import torch as th

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def to_tensor(x, d_type=th.float32, device=device):
    """Convert array-like input to a tensor with the requested dtype and device."""
    return th.as_tensor(x, dtype=d_type, device=device)


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


def validate_cfg_skip_steps(use_cfgzero_star, skip_steps, total_steps):
    """Resolve optional skipping; CFG-Zero* requires an explicit skip count, including 0."""
    if skip_steps is None:
        if use_cfgzero_star:
            raise ValueError("skip_steps must be provided when use_cfgzero_star=True (use 0 to disable skipping).")
        skip_steps = 0
    if isinstance(skip_steps, bool) or not isinstance(skip_steps, int) or skip_steps < 0:
        raise ValueError("skip_steps must be a nonnegative integer.")
    if skip_steps >= total_steps:
        raise ValueError(f"skip_steps must be less than the total number of sampling steps ({total_steps}).")
    return skip_steps
