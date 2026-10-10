"""Sample nine RGB images and display a 3x3 PIL grid or save it to --output-path.

Example:
    python sample/diffusion_sample.py --model-path checkpoint.pt --image-size 32 --sample ddim
"""

import argparse
import math
from pathlib import Path

from PIL import Image
import torch

from generative_model.logger import get_logger
from generative_model.cli_config import parse_args_with_config
from generative_model.models.unet import UNet
from generative_model.sampler import DDIMSampler, DDPMSampler, FMSampler, display_image_uint8, make_beta_schedule

logger = get_logger('generative_model.sample')


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=Path, required=True, help='Training checkpoint; load ema_model weights when available, otherwise model')
    parser.add_argument('--output-path', type=Path, help='Save the 3x3 grid to this image file (e.g. samples.png) instead of opening a viewer; omitted means no output is saved')
    parser.add_argument('--image-size', type=positive_int, required=True, help='Square image side length used during training')
    parser.add_argument('--model-channels', type=positive_int, default=64)
    parser.add_argument('--embedding-channels', type=positive_int, default=256)
    parser.add_argument('--res-blocks', type=positive_int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--channel-mult', type=positive_int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attention-resolutions', type=positive_int, nargs='*', default=[16, 8], help='Feature-map side lengths, as in training; empty disables encoder/decoder attention')
    parser.add_argument('--num-heads', type=positive_int, default=4)
    parser.add_argument('--use-conv', action=argparse.BooleanOptionalAction, default=False, help='Use convolutions in resampling and residual shortcuts')
    parser.add_argument('--attn-o-proj-zeroinit', action=argparse.BooleanOptionalAction, default=False, help='Zero-initialize attention output projections')
    parser.add_argument('--sample', choices=['ddpm', 'ddim', 'fm'], default='ddpm')
    parser.add_argument('--fm-step', type=float, default=0.01, help='FM only: integration step size in (0, 1]')
    parser.add_argument('--solver', choices=['euler'], default='euler', help='FM only: solver (currently only euler is supported)')
    parser.add_argument('--timesteps', type=positive_int, default=1000, help='DDPM/DDIM only: original diffusion schedule length; must match training')
    parser.add_argument('--timestep-spacing', type=positive_int, default=20, help='DDIM only: stride through the original timesteps; the sampler also includes the last timestep')
    parser.add_argument('--randomness', type=float, default=0.0, help='Noise strength in [0, 1]: DDIM eta; FM sigma(t) = randomness * sqrt(1-t)')
    parser.add_argument('--device', default='cpu', help='Torch device, e.g. cpu, mps, or cuda; never silently falls back')
    parser.add_argument('--seed', type=int, default=42)
    args = parse_args_with_config(parser, argv)

    if not args.model_path.is_file():
        parser.error(f'--model-path is not a file: {args.model_path}')
    if args.output_path is not None:
        if args.output_path.is_dir():
            parser.error('--output-path must be an image file, not a directory')
        Image.init()
        image_format = Image.registered_extensions().get(args.output_path.suffix.lower())
        if image_format not in Image.SAVE:
            parser.error('--output-path must have a supported image extension, e.g. .png or .jpg')
    max_scale = 2 ** (len(args.channel_mult) - 1)
    if args.image_size % max_scale:
        parser.error(f'--image-size must be divisible by {max_scale}')
    resolutions = {args.image_size // 2 ** level for level in range(len(args.channel_mult))}
    if not set(args.attention_resolutions).issubset(resolutions):
        parser.error(f'--attention-resolutions must be selected from {sorted(resolutions)}')
    if args.model_channels % 32 or any(args.model_channels * mult % 32 for mult in args.channel_mult):
        parser.error('model channels at every level must be divisible by 32 (GroupNorm)')
    if any(args.model_channels * mult % args.num_heads for mult in args.channel_mult):
        parser.error('model channels at every level must be divisible by --num-heads')
    if args.embedding_channels % 2:
        parser.error('--embedding-channels must be even')
    if not 0 <= args.dropout <= 1:
        parser.error('--dropout must be between 0 and 1')
    if args.sample != 'fm' and args.timesteps <= 50:
        parser.error('--timesteps must exceed 50 for this beta schedule')
    if args.sample == 'fm' and (not math.isfinite(args.fm_step) or not 0 < args.fm_step <= 1):
        parser.error('--fm-step must be finite and in (0, 1]')
    if not math.isfinite(args.randomness) or not 0 <= args.randomness <= 1:
        parser.error('--randomness must be finite and between 0 and 1')
    return args


def make_grid(samples):
    """Convert nine diffusion tensors in [-1, 1] to one RGB PIL image."""
    pixels = display_image_uint8(samples)
    _, _, height, width = pixels.shape
    grid = Image.new('RGB', (width * 3, height * 3))
    for index, pixels_i in enumerate(pixels):
        image = Image.fromarray(pixels_i.permute(1, 2, 0).numpy())
        grid.paste(image, ((index % 3) * width, (index // 3) * height))
    return grid


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if args.output_path is None:
        logger.warning('--output-path was not provided; samples will NOT be saved. PIL will only attempt to open a viewer.')
    else:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.ones(1, device=device).add_(1)
    torch.manual_seed(args.seed)
    model = UNet(
        in_channel=3, out_channel=3, model_channel=args.model_channels,
        embedding_channel=args.embedding_channels, resblock_num=args.res_blocks,
        dropout=args.dropout, ch_mult=tuple(args.channel_mult),
        attention_resolution=tuple(args.attention_resolutions),
        num_heads=args.num_heads, image_size=args.image_size,
        use_conv=args.use_conv, attn_o_proj_zeroinit=args.attn_o_proj_zeroinit,
    )
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not any(key in checkpoint for key in ('ema_model', 'model')):
        raise ValueError('Checkpoint must contain an ema_model or model key holding the UNet state dict.')
    checkpoint_method = checkpoint.get('training_method', 'ddpm')
    expected_method = 'fm' if args.sample == 'fm' else 'ddpm'
    if checkpoint_method != expected_method:
        raise ValueError(
            f'Checkpoint training method is {checkpoint_method!r}, '
            f'which is incompatible with --sample {args.sample}.'
        )
    weights_key = 'ema_model' if 'ema_model' in checkpoint else 'model'
    state = checkpoint[weights_key]
    # torch.compile adds this prefix to training state dicts.
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, '_orig_mod.')
    model.load_state_dict(state)
    print(f'Loaded {weights_key} weights from {args.model_path}', flush=True)
    del checkpoint, state
    model.to(device).eval()

    if args.sample == 'fm':
        sampler = FMSampler(args.fm_step, model, device=device, solver=args.solver, randomness=args.randomness)
        num_steps = sampler.sample_nums
        print(f'FM solver: {args.solver}, randomness: {args.randomness:g}', flush=True)
    else:
        betas = make_beta_schedule(args.timesteps)
        if args.sample == 'ddim':
            sampler = DDIMSampler(
                model, spacing=args.timestep_spacing, randomness=args.randomness,
                betas=betas, device=device,
            )
        else:
            sampler = DDPMSampler(model, betas=betas, device=device)
        num_steps = sampler.num_timesteps
    noise = torch.randn((9, 3, args.image_size, args.image_size), device=device)
    print(f'Sampling 9 images with {args.sample.upper()} on {device}: {num_steps} steps', flush=True)
    samples = sampler.sample(noise)
    if not torch.isfinite(samples).all():
        raise RuntimeError('Sampling produced non-finite values; check the checkpoint and sampler settings.')
    grid = make_grid(samples)
    if args.output_path is not None:
        grid.save(args.output_path)
        print(f'Saved samples to {args.output_path.resolve()}', flush=True)
    else:
        grid.show(title=f'{args.sample.upper()} samples')
    return grid


if __name__ == '__main__':
    main()
