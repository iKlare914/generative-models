from dataclasses import replace
from pathlib import Path

from datasets import Dataset, Image as DatasetImage
from PIL import Image


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
