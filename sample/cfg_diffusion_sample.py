"""Show twelve samples in a 4x3 figure; an empty prompt means unconditional sampling.

Examples:
    python sample/cfg_diffusion_sample.py --model-path checkpoint.pt --image-size 32 --sample ddim
    python sample/cfg_diffusion_sample.py --random-init --image-size 32 --sample fm --step 1 --output-path cfg-smoke.png
"""

import argparse
import math
from pathlib import Path
import textwrap

import numpy as np
import torch

from generative_model.cli_config import parse_args_with_config
from generative_model.models.unet import CFGUNet
from generative_model.models.dit import DiT
from generative_model.models.latent_diffusion import LatentDiffusion
from generative_model.sampler import (
    CFGDDIMSampler, CFGDDPMSampler, CFGFMSampler, display_image_uint8, make_beta_schedule,
    validate_cfg_skip_steps, CFGLatentDDIMSampler, CFGLatentDDPMSampler, CFGLatentFMSampler,
)
from generative_model.dataset.text_encoder import CLIPTextEncoder


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--model-path', type=Path, help='CFG training checkpoint; load ema_model weights when available, otherwise model')
    weights.add_argument('--random-init', action='store_true', help='Use an untrained model for a smoke test')
    parser.add_argument('--output-path', type=Path, help='Save the titled figure instead of opening an interactive window')
    parser.add_argument('--image-size', type=positive_int, required=True)
    parser.add_argument('--model-channel', '--model-channels', type=positive_int, default=64)
    parser.add_argument('--embedding-channel', '--embedding-channels', type=positive_int, default=256)
    parser.add_argument('--d-feature', '--feature-channels', type=positive_int, default=512, help='Text feature width: DiT.d_feature / CFGUNet.feature_channel; must match training and CLIP')
    parser.add_argument('--resblock-num', '--res-blocks', type=positive_int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ch-mult', '--channel-mult', type=positive_int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attention-resolution', '--attention-resolutions', type=positive_int, nargs='*', default=[16, 8])
    parser.add_argument('--num-heads', type=positive_int, default=4)
    parser.add_argument('--use-conv', action=argparse.BooleanOptionalAction, default=False, help='Use convolutions in resampling and residual shortcuts')
    parser.add_argument('--attn-o-proj-zeroinit', action=argparse.BooleanOptionalAction, default=False, help='Zero-initialize attention output projections')
    parser.add_argument('--sample', choices=['ddpm', 'ddim', 'fm'], default='ddim')
    parser.add_argument('--step', '--fm-step', type=float, default=0.01)
    parser.add_argument('--use-cfgzero-star', '--use-cfgzero_star', dest='use_cfgzero_star',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Enable CFG-Zero* projection scaling for DDPM, DDIM or FM; requires --skip-steps')
    parser.add_argument('--skip-steps', type=int, default=None,
                        help='Initial sampling updates to skip, independently of CFG-Zero*; required when CFG-Zero* is enabled, otherwise defaults to 0. DDIM counts reduced-schedule steps.')
    parser.add_argument('--solver', choices=['euler'], default='euler', help='FM only: solver (currently only euler is supported)')
    parser.add_argument('--num-timesteps', '--timesteps', type=positive_int, default=1000, help='DDPM/DDIM: schedule length used in training')
    parser.add_argument('--spacing', '--timestep-spacing', type=positive_int, default=20, help='DDIM timestep stride')
    parser.add_argument('--randomness', type=float, default=0.0, help='Noise strength in [0, 1]: DDIM eta; FM sigma(t) = randomness * sqrt(1-t)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='CFG weight when --prompt is nonempty; empty text uses unconditional sampling')
    parser.add_argument('--eval-max-length', type=positive_int, default=77)
    parser.add_argument('--text-model-name', default='openai/clip-vit-base-patch32', help='CLIP model ID or local directory; CLIP runs on CPU')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', type=str, default='', help='Text prompt for all twelve images; empty string means unconditional')
    parser.add_argument('--model-type', choices=['unet', 'dit'], default='unet', help='Select model architecture independently of the dataset representation')
    parser.add_argument('--use-latent', action=argparse.BooleanOptionalAction, default=False, help='DiT training: read cached mean/std instead of encoding images; ignored for UNet and during sampling')
    parser.add_argument('--vae-model-name-or-path', '--vae-model-name', default='stabilityai/sdxl-vae', help='SD-style AutoencoderKL Hub ID or local directory; must match training')
    parser.add_argument('--vae-slicing', action=argparse.BooleanOptionalAction, default=True, help='Process VAE batch elements separately to reduce peak memory')
    parser.add_argument('--d-model', '--dit-hidden-size', type=positive_int, default=256)
    parser.add_argument('--num-layers', '--dit-depth', type=positive_int, default=8)
    parser.add_argument('--patch-size', '--dit-patch-size', type=positive_int, default=2, help='Patch size in latent pixels')
    parser.add_argument('--d-time-emb', '--dit-time-embedding', type=positive_int, default=256)
    args = parse_args_with_config(parser, argv)
    if args.model_type == 'unet':
        args.use_latent = False

    if args.model_path is not None and not args.model_path.is_file():
        parser.error(f'--model-path is not a file: {args.model_path}')
    if args.output_path is not None and args.output_path.is_dir():
        parser.error('--output-path must be a file')
    if args.model_type == 'dit':
        if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 4:
            parser.error('--d-model must be divisible by --num-heads and have a head dimension divisible by 4 (2D RoPE)')
        if args.d_time_emb % 2:
            parser.error('--d-time-emb must be even')
        # VAE downsampling and latent patch divisibility are checked after loading its config.
    else:
        max_scale = 2 ** (len(args.ch_mult) - 1)
        if args.image_size % max_scale:
            parser.error(f'--image-size must be divisible by {max_scale}')
        resolutions = {args.image_size // 2 ** level for level in range(len(args.ch_mult))}
        if not set(args.attention_resolution).issubset(resolutions):
            parser.error(f'--attention-resolution must be selected from {sorted(resolutions)}')
        if any(args.model_channel * mult % 32 for mult in args.ch_mult) or args.model_channel % 32:
            parser.error('model channels must be divisible by 32 (GroupNorm)')
        if any(args.model_channel * mult % args.num_heads for mult in args.ch_mult):
            parser.error('model channels must be divisible by --num-heads')
        if args.embedding_channel % 2:
            parser.error('--embedding-channel must be even')
    if not 0 <= args.dropout <= 1:
        parser.error('--dropout must be between 0 and 1')
    if not math.isfinite(args.randomness) or not 0 <= args.randomness <= 1:
        parser.error('--randomness must be finite and between 0 and 1')
    if args.sample != 'fm' and args.num_timesteps <= 50:
        parser.error('--num-timesteps must exceed 50 for this beta schedule')
    if args.sample == 'fm' and (not math.isfinite(args.step) or not 0 < args.step <= 1):
        parser.error('--step must be finite and in (0, 1]')
    if args.sample == 'fm':
        total_steps = math.ceil(1 / args.step)
    elif args.sample == 'ddim':
        total_steps = len(range(0, args.num_timesteps, args.spacing))
        total_steps += (args.num_timesteps - 1) % args.spacing != 0
    else:
        total_steps = args.num_timesteps
    try:
        args.skip_steps = validate_cfg_skip_steps(args.use_cfgzero_star, args.skip_steps, total_steps)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error('--guidance-scale must be finite and nonnegative')
    if not 2 <= args.eval_max_length <= 77:
        parser.error('--eval-max-length must be between 2 and 77')
    return args


def make_figure(samples, prompts, *, sample_method, guidance_scale, random_init=False, solver=None, randomness=0.0):
    """Draw twelve separate image axes, each with its own prompt title."""
    import matplotlib.pyplot as plt

    pixels = display_image_uint8(samples).permute(0, 2, 3, 1).numpy()
    figure, axes = plt.subplots(4, 3, figsize=(10.5, 13), layout='constrained')
    title = f'{sample_method.upper()} | CFG scale {guidance_scale:g}'
    if sample_method == 'fm' and solver is not None:
        title += f' | {solver.upper()}'
    if sample_method in ('fm', 'ddim'):
        title += f' | randomness {randomness:g}'
    if random_init:
        title += ' | Untrained model'
    figure.suptitle(title, fontsize=15)
    for index, axis in enumerate(axes.flat):
        axis.imshow(pixels[index], interpolation='nearest')
        prompt_title = prompts[index]
        if prompt_title == '':
            prompt_title = 'No prompt (unconditional)'
        axis.set_title(textwrap.fill(prompt_title, width=30), fontsize=10, pad=8)
        axis.set_axis_off()
    return figure


def build_model(args):
    if args.model_type == 'dit':
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path).float()
        if vae.config.shift_factor not in (None, 0) or vae.config.latents_mean is not None or vae.config.latents_std is not None:
            raise ValueError('This entry point requires an SD-style VAE with scalar latent scaling only.')
        downsample = 2 ** (len(vae.config.block_out_channels) - 1)
        if args.image_size % (downsample * args.patch_size):
            raise ValueError(f'--image-size must be divisible by VAE downsampling * patch size ({downsample * args.patch_size}).')
        if vae.config.in_channels != 3 or vae.config.out_channels != 3:
            raise ValueError('The image loaders and sample grids require an RGB VAE.')
        if args.vae_slicing:
            vae.enable_slicing()
        latent_size = args.image_size // downsample
        grid_size = latent_size // args.patch_size
        dit = DiT(
            image_size=(latent_size, latent_size), patch_size=args.patch_size,
            in_channel=vae.config.latent_channels, d_model=args.d_model,
            d_time_emb=args.d_time_emb, num_heads=args.num_heads,
            num_layers=args.num_layers, grid_size_h=grid_size, grid_size_w=grid_size,
            extra_feature=True, d_feature=args.d_feature,
        )
        return LatentDiffusion(dit, vae)
    return CFGUNet(
        in_channel=3, out_channel=3, model_channel=args.model_channel,
        embedding_channel=args.embedding_channel, feature_channel=args.d_feature,
        resblock_num=args.resblock_num, dropout=args.dropout, ch_mult=tuple(args.ch_mult),
        attention_resolution=tuple(args.attention_resolution), num_heads=args.num_heads,
        image_size=args.image_size,
        use_conv=args.use_conv, attn_o_proj_zeroinit=args.attn_o_proj_zeroinit,
    )


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.output_path is not None:
        matplotlib.use('Agg')
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    torch.ones(1, device=device).add_(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(args)
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not any(key in checkpoint for key in ('ema_model', 'model')):
            raise ValueError('Checkpoint must contain an ema_model or model key holding model weights.')
        expected_method = ('latent_' if args.model_type == 'dit' else '') + ('fm' if args.sample == 'fm' else 'ddpm')
        if checkpoint.get('training_method', 'ddpm') != expected_method:
            raise ValueError(f'Checkpoint training method does not match --sample {args.sample}.')
        weights_key = 'ema_model' if 'ema_model' in checkpoint else 'model'
        state = checkpoint[weights_key]
        # torch.compile adds this prefix to training state dicts.
        if args.model_type == 'dit':
            # Non-EMA checkpoints contain dit.* and vae.*; compiled DiT adds dit._orig_mod.*.
            if 'model' in checkpoint:
                full_state = {
                    key.replace('dit._orig_mod.', 'dit.', 1) if key.startswith('dit._orig_mod.') else key: value
                    for key, value in checkpoint['model'].items()
                }
                model.load_state_dict(full_state)
                del full_state
            if weights_key == 'ema_model':
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, '_orig_mod.')
                model.dit.load_state_dict(state)
        else:
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, '_orig_mod.')
            model.load_state_dict(state)
        print(f'Loaded {weights_key} weights from {args.model_path}', flush=True)
        del checkpoint, state
    else:
        print('Using an untrained model: this checks execution and layout, not image quality.', flush=True)
    model.to(device).eval()
    noise_shape = model.latent_shape if args.model_type == 'dit' else (3, args.image_size, args.image_size)
    encoder = CLIPTextEncoder(args.text_model_name)
    guidance_scale = args.guidance_scale if args.prompt else 0.0
    options = dict(device=device, text_encoder=encoder, guidance_scale=guidance_scale,
                   eval_max_length=args.eval_max_length,
                   use_cfgzero_star=args.use_cfgzero_star, skip_steps=args.skip_steps)
    if args.sample == 'fm':
        sampler_class = CFGLatentFMSampler if args.model_type == 'dit' else CFGFMSampler
        sampler = sampler_class(args.step, model, solver=args.solver, randomness=args.randomness, **options)
        num_steps = sampler.sample_nums - sampler.skip_steps
        print(f'FM solver: {args.solver}, randomness: {args.randomness:g}', flush=True)
    elif args.sample == 'ddim':
        sampler_class = CFGLatentDDIMSampler if args.model_type == 'dit' else CFGDDIMSampler
        sampler = sampler_class(model, spacing=args.spacing, randomness=args.randomness,
                                 betas=make_beta_schedule(args.num_timesteps), **options)
        num_steps = sampler.num_timesteps - sampler.skip_steps
    else:
        sampler_class = CFGLatentDDPMSampler if args.model_type == 'dit' else CFGDDPMSampler
        sampler = sampler_class(model, betas=make_beta_schedule(args.num_timesteps), **options)
        num_steps = sampler.num_timesteps - sampler.skip_steps
    print(f'CFG-Zero*: {args.use_cfgzero_star}, skipped steps: {sampler.skip_steps}, '
          f'sampling updates: {num_steps}', flush=True)
    prompts = [args.prompt] * 12
    noise = torch.randn((12, *noise_shape), device=device)
    print(f'Sampling 12 images with prompt={args.prompt!r}, guidance_scale={guidance_scale:g}', flush=True)
    samples = sampler.sample(noise, prompts=prompts)
    for index, prompt in enumerate(prompts, start=1):
        print(f'{index}: {prompt}', flush=True)
    if not torch.isfinite(samples).all():
        raise RuntimeError('Sampling produced non-finite values; check the checkpoint and schedule.')
    figure = make_figure(samples, prompts, sample_method=args.sample,
                         guidance_scale=guidance_scale, random_init=args.random_init,
                         solver=args.solver, randomness=args.randomness)
    if args.output_path is not None:
        figure.savefig(args.output_path, dpi=160)
        print(f'Saved samples to {args.output_path.resolve()}', flush=True)
        plt.close(figure)
    else:
        plt.show()
    return figure


if __name__ == '__main__':
    main()
