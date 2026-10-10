"""Initialize ImagenetDataset to download/cache ImageNet and inspect one sample.

Run from the project root:
    python scripts/download_imagenet.py
    python scripts/download_imagenet.py --split train --resize --image-size 256 128
"""

import argparse
from pathlib import Path

from generative_model.dataset.load_dataset import ImagenetDataset


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-name", default="clane9/imagenet-100")
    parser.add_argument("--split", default="all", help="Split to load; all combines all available splits")
    parser.add_argument("--resize", action="store_true", help="Resize and center crop images, then save a local cache")
    parser.add_argument("--image-size", type=positive_int, nargs="+", metavar="SIZE", help="One square size, or two values: H W; required with --resize")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface/datasets")
    args = parser.parse_args(argv)
    if args.image_size is not None and len(args.image_size) not in (1, 2):
        parser.error("--image-size expects one square size or two values: H W")
    if args.resize and args.image_size is None:
        parser.error("--resize requires --image-size")
    if args.image_size is not None:
        args.image_size = args.image_size[0] if len(args.image_size) == 1 else tuple(args.image_size)
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Loading {args.repo_name} (split={args.split}, resize={args.resize})", flush=True)
    print(f"Source cache: {args.cache_dir}", flush=True)
    dataset = ImagenetDataset(
        name=args.repo_name, split=args.split, resize=args.resize,
        image_size=args.image_size, cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset):
        image, label = dataset[0]
        print(f"First sample: shape={tuple(image.shape)}, dtype={image.dtype}, label={label}")
        print(f"Pixel range: [{image.min().item():.3f}, {image.max().item():.3f}]")


if __name__ == "__main__":
    main()
