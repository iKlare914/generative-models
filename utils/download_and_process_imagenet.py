"""Prepare an ImageNet image cache or a VAE posterior cache and inspect one sample.

Run from the project root:
    python utils/download_and_process_imagenet.py --split train --resize --image-size 256
    python utils/download_and_process_imagenet.py --split train --use-latent --image-size 256
"""

import argparse
from pathlib import Path

from generative_model.dataset.image_transforms import DATASET_DIR, get_latent_dataset_dir
from generative_model.dataset.load_dataset import ImagenetDataset, LatentImagenetDataset


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
    parser.add_argument('--use-latent', action=argparse.BooleanOptionalAction, default=False,
                        help='Normalize/encode images and save raw VAE mean/std; implies resizing to --image-size')
    parser.add_argument('--vae-model-name-or-path', default='stabilityai/sdxl-vae',
                        help='HF AutoencoderKL checkpoint; use the same VAE for training and sampling')
    parser.add_argument('--save-dir', type=Path, default=DATASET_DIR, help='Latent cache root (training --latent-cache-dir)')
    parser.add_argument('--batch-size', type=positive_int, default=8, help='VAE preprocessing batch size')
    parser.add_argument('--device', default=None, help='VAE device; omitted prefers CUDA, otherwise CPU')
    args = parser.parse_args(argv)
    if args.image_size is not None and len(args.image_size) not in (1, 2):
        parser.error("--image-size expects one square size or two values: H W")
    if (args.resize or args.use_latent) and args.image_size is None:
        parser.error('--resize and --use-latent require --image-size')
    if args.image_size is not None:
        args.image_size = args.image_size[0] if len(args.image_size) == 1 else tuple(args.image_size)
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Loading {args.repo_name} (split={args.split}, resize={args.resize})", flush=True)
    print(f"Source cache: {args.cache_dir}", flush=True)
    if args.use_latent:
        latent_dir = get_latent_dataset_dir(args.repo_name, args.image_size, args.split, args.save_dir)
        vae = None
        if not latent_dir.exists():
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
        dataset = LatentImagenetDataset(
            name=args.repo_name, split=args.split, image_size=args.image_size, cache_dir=args.cache_dir,
            vae=vae, save_dir=args.save_dir, batch_size=args.batch_size, device=args.device,
        )
        print(f'Latent cache: {latent_dir}', flush=True)
    else:
        dataset = ImagenetDataset(
            name=args.repo_name, split=args.split, resize=args.resize,
            image_size=args.image_size, cache_dir=args.cache_dir,
        )
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset):
        data, label = dataset[0]
        if args.use_latent:
            mean, std = data
            print(f'First sample: mean={tuple(mean.shape)}, std={tuple(std.shape)}, dtype={mean.dtype}, label={label}')
            print('Saved raw mean/std; posterior sampling and scaling_factor are applied by the trainer.')
        else:
            print(f"First sample: shape={tuple(data.shape)}, dtype={data.dtype}, label={label}")
            print(f"Pixel range: [{data.min().item():.3f}, {data.max().item():.3f}]")
    return dataset


if __name__ == "__main__":
    main()
