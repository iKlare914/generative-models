from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch as th
from datasets import Array3D, Dataset, Features, Image as DatasetImage, load_from_disk
from PIL import Image

if TYPE_CHECKING:
    from diffusers import AutoencoderKL


DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets"


def center_crop_arr(pil_image, image_size):
    """
    Resize preserving aspect ratio, then center crop to an int or (H, W).
    Based on the center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    while pil_image.width >= 2 * width and pil_image.height >= 2 * height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = max(width / pil_image.width, height / pil_image.height)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    crop_y = (pil_image.height - height) // 2
    crop_x = (pil_image.width - width) // 2
    return pil_image.crop((crop_x, crop_y, crop_x + width, crop_y + height))


def get_resized_dataset_dir(repo_name, resize, split=None, save_dir=None) -> Path:
    """Return save_dir/<repo_name>-<H>x<W>-<split>, using all for no split."""
    if (
        not isinstance(resize, tuple) or len(resize) != 2
        or any(type(size) is not int or size <= 0 for size in resize)
    ):
        raise ValueError("resize must be a (height, width) tuple of positive integers.")
    if not isinstance(repo_name, str):
        raise ValueError("Cannot infer the repo name; pass repo_name explicitly.")
    name = repo_name.rstrip("/").rsplit("/", 1)[-1]
    if not name or name in (".", "..") or "\\" in name:
        raise ValueError("repo_name must contain a valid repository name.")
    split_name = str(split) if split is not None else "all"
    if not split_name or split_name in (".", "..") or "/" in split_name or "\\" in split_name:
        raise ValueError("split must contain a valid split name.")
    height, width = resize
    parent = Path(save_dir) if save_dir is not None else DATASET_DIR
    return parent / f"{name}-{height}x{width}-{split_name}"


def resize_image_dataset(
    dataset: Dataset,
    resize: tuple[int, int],
    save_dir: str | Path | None = None,
    num_proc: int = 2,
    *,
    repo_name: str | None = None,
    split: str | None = None,
) -> Dataset:
    """Map image resizing/center cropping over a dataset and save it to disk.

    Args:
        dataset: A single Hugging Face Dataset with Image columns.
        resize: Target (height, width), keeping the original aspect ratio.
        save_dir: Parent directory, defaulting to DATASET_DIR.
        repo_name: Optional Hub repo ID/name. Defaults to info.dataset_name;
            supply it explicitly when the dataset has no name metadata.
        split: Optional suffix override, e.g. all for a combined dataset.

    Returns:
        The resized dataset, saved to save_dir/<repo_name>-<H>x<W>-<split>.
        Hub namespaces are omitted. The split suffix comes from dataset.split,
        or defaults to all when unset. All top-level Image columns are processed;
        other columns are preserved. Existing output raises FileExistsError.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a single Hugging Face Dataset.")
    if repo_name is None:
        repo_name = dataset.info.dataset_name
    output_dir = get_resized_dataset_dir(
        repo_name, resize, dataset.split if split is None else split, save_dir,
    )
    height, width = resize
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    image_columns = [name for name, feature in dataset.features.items() if isinstance(feature, DatasetImage)]
    if not image_columns:
        raise ValueError("dataset must contain at least one Image column.")
    dataset = dataset.with_format(None)
    for column in image_columns:
        feature = dataset.features[column]
        if not feature.decode:
            dataset = dataset.cast_column(column, replace(feature, decode=True))

    def transform(batch):
        return {
            column: [center_crop_arr(image, resize) if image is not None else None for image in batch[column]]
            for column in image_columns
        }

    resized = dataset.map(
        transform, batched=True, batch_size=32, writer_batch_size=32,
        desc=f"Resize images to {height}x{width}", num_proc=num_proc
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    resized.save_to_disk(output_dir)
    return resized


def get_latent_dataset_dir(repo_name, image_size, split=None, save_dir=None) -> Path:
    """Return save_dir/<repo_name>-<H>x<W>-<split>-latent."""
    size = (image_size, image_size) if type(image_size) is int else image_size
    resized_dir = get_resized_dataset_dir(repo_name, size, split, save_dir)
    return resized_dir.with_name(resized_dir.name + '-latent')


def to_latent(
    dataset: Dataset,
    vae: AutoencoderKL,
    save_dir: str | Path | None = None,
    batch_size: int = 8,
    *,
    repo_name: str | None = None,
    split: str | None = None,
    device: str | th.device | None = None,
) -> Dataset:
    """Encode an image Dataset with a pretrained HF VAE and save its posterior.

    All Image columns are converted to RGB, normalized from [0, 255] to [-1, 1],
    and replaced by {mean: float32[C,H,W], std: float32[C,H,W]}. Other columns
    and their feature metadata are preserved. No posterior sampling or latent
    scaling is performed. Images must already share a size divisible by the
    VAE downsampling factor; use resize_image_dataset first when needed.

    CUDA is preferred when device is omitted, otherwise CPU is used. Encoding
    runs without gradients or autocast, and respects the VAE's force_upcast
    setting. The supplied VAE's original device, dtype and training mode are
    restored afterwards; its requires_grad flags are unchanged.

    Output is DATASET_DIR (or save_dir)/<repo_name>-<H>x<W>-<split>-latent.
    Use separate save_dir roots for different VAE checkpoints. Existing output
    is never overwritten. Arrow batches are written in a temporary directory;
    only a completed save is published. The returned Dataset reads that cache.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError('dataset must be a single Hugging Face Dataset.')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    if len(dataset) == 0:
        raise ValueError('Cannot encode an empty dataset.')
    image_columns = [name for name, feature in dataset.features.items() if isinstance(feature, DatasetImage)]
    if not image_columns:
        raise ValueError('dataset must contain at least one Image column.')
    prepared = dataset.with_format(None)
    for column in image_columns:
        feature = prepared.features[column]
        if not feature.decode:
            prepared = prepared.cast_column(column, replace(feature, decode=True))
    first_image = prepared[0][image_columns[0]]
    if first_image is None:
        raise ValueError('Image columns cannot contain missing images.')
    width, height = first_image.size
    output_dir = get_latent_dataset_dir(
        dataset.info.dataset_name if repo_name is None else repo_name,
        (height, width), dataset.split if split is None else split, save_dir,
    )
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    downsample = 2 ** (len(vae.config.block_out_channels) - 1)
    if height % downsample or width % downsample:
        raise ValueError(f'Image height and width must be divisible by the VAE downsampling factor ({downsample}).')
    if vae.config.in_channels != 3:
        raise ValueError('to_latent requires a VAE with three input channels for RGB images.')
    latent_shape = (vae.config.latent_channels, height // downsample, width // downsample)
    features = Features(dict(prepared.features))
    for column in image_columns:
        features[column] = {
            'mean': Array3D(shape=latent_shape, dtype='float32'),
            'std': Array3D(shape=latent_shape, dtype='float32'),
        }

    parameter = next(vae.parameters())
    original_device, original_dtype = parameter.device, parameter.dtype
    was_training = vae.training
    device = th.device(device) if device is not None else th.device('cuda' if th.cuda.is_available() else 'cpu')
    dtype = th.float32 if getattr(vae.config, 'force_upcast', False) else original_dtype

    @th.no_grad()
    def transform(batch):
        result = {}
        for column in image_columns:
            images = []
            for image in batch[column]:
                if image is None or image.size != (width, height):
                    raise ValueError('All images must be present and have the same size; resize the dataset first.')
                pixels = np.array(image.convert('RGB'), dtype=np.float32)
                images.append(th.from_numpy(pixels).permute(2, 0, 1).contiguous() / 127.5 - 1)
            inputs = th.stack(images).to(device=device, dtype=dtype)
            with th.autocast(device_type=device.type, enabled=False):
                posterior = vae.encode(inputs).latent_dist
                mean = posterior.mean.float().cpu()
                # HF DiagonalGaussianDistribution already computes std = exp(0.5 * logvar).
                std = posterior.std.float().cpu()
            if tuple(mean.shape[1:]) != latent_shape or mean.shape != std.shape:
                raise ValueError('VAE posterior shape does not match its configured latent dimensions.')
            if not th.isfinite(mean).all() or not th.isfinite(std).all():
                raise ValueError('VAE produced non-finite posterior parameters.')
            result[column] = [
                {'mean': mean_i.numpy(), 'std': std_i.numpy()}
                for mean_i, std_i in zip(mean, std)
            ]
        return result

    try:
        vae.to(device=device, dtype=dtype).eval()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{output_dir.name}-', dir=output_dir.parent) as temporary:
            temporary = Path(temporary)
            encoded = prepared.map(
                transform, batched=True, batch_size=batch_size, writer_batch_size=batch_size,
                features=features, num_proc=None, load_from_cache_file=False,
                cache_file_name=str(temporary / 'encoded.arrow'), new_fingerprint=uuid4().hex,
                desc='Encode VAE posterior (mean/std)',
            )
            saved = temporary / 'dataset'
            encoded.save_to_disk(saved)
            metadata = {
                'image_size': [height, width], 'image_columns': image_columns,
                'representation': 'mean_std', 'scaled': False,
                'vae_model_name_or_path': str(getattr(vae.config, '_name_or_path', '')),
                'scaling_factor': float(vae.config.scaling_factor),
            }
            (saved / 'latent_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            if output_dir.exists():
                raise FileExistsError(f'Output directory already exists: {output_dir}')
            saved.rename(output_dir)
    finally:
        vae.to(device=original_device, dtype=original_dtype).train(was_training)
    return load_from_disk(output_dir)
