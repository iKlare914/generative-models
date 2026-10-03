"""Show nine prompted CFG samples and three empty-prompt samples in separate 4x3 subplots.

Examples:
    python scripts/cfg_diffusion_sample.py --model-path checkpoint.pt --image-size 32 --sample ddim
    python scripts/cfg_diffusion_sample.py --random-init --image-size 32 --sample fm --fm-step 1 --output-path cfg-smoke.png
"""

import argparse
import math
from pathlib import Path
import textwrap

import numpy as np
import torch

from generative_model.model import CFGUNet
from generative_model.sampler import CFGDDIMSampler, CFGDDPMSampler, CFGFMSampler, display_image_uint8, make_beta_schedule
from generative_model.text_encoder import CLIPTextEncoder, get_cifar10_prompt


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--model-path', type=Path, help='CFG training checkpoint')
    weights.add_argument('--random-init', action='store_true', help='Use an untrained model for a smoke test')
    parser.add_argument('--output-path', type=Path, help='Save the titled figure instead of opening an interactive window')
    parser.add_argument('--image-size', type=positive_int, required=True)
    parser.add_argument('--model-channels', type=positive_int, default=64)
    parser.add_argument('--embedding-channels', type=positive_int, default=256)
    parser.add_argument('--feature-channels', type=positive_int, default=512, help='CLIP text hidden size; must match training')
    parser.add_argument('--res-blocks', type=positive_int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--channel-mult', type=positive_int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attention-resolutions', type=positive_int, nargs='*', default=[16, 8])
    parser.add_argument('--num-heads', type=positive_int, default=4)
    parser.add_argument('--sample', choices=['ddpm', 'ddim', 'fm'], default='ddim')
    parser.add_argument('--fm-step', type=float, default=0.01)
    parser.add_argument('--solver', choices=['euler'], default='euler', help='FM only: solver (currently only euler is supported)')
    parser.add_argument('--timesteps', type=positive_int, default=1000, help='DDPM/DDIM: schedule length used in training')
    parser.add_argument('--timestep-spacing', type=positive_int, default=20, help='DDIM timestep stride')
    parser.add_argument('--randomness', type=float, default=0.0, help='Noise strength in [0, 1]: DDIM eta; FM sigma(t) = randomness * sqrt(1-t)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='CFG weight for the first nine samples; bottom row always uses 0')
    parser.add_argument('--eval-max-length', type=positive_int, default=77)
    parser.add_argument('--text-model-name', default='openai/clip-vit-base-patch32', help='CLIP model ID or local directory; CLIP runs on CPU')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    if args.model_path is not None and not args.model_path.is_file():
        parser.error(f'--model-path is not a file: {args.model_path}')
    if args.output_path is not None and args.output_path.is_dir():
        parser.error('--output-path must be a file')
    max_scale = 2 ** (len(args.channel_mult) - 1)
    if args.image_size % max_scale:
        parser.error(f'--image-size must be divisible by {max_scale}')
    resolutions = {args.image_size // 2 ** level for level in range(len(args.channel_mult))}
    if not set(args.attention_resolutions).issubset(resolutions):
        parser.error(f'--attention-resolutions must be selected from {sorted(resolutions)}')
    if any(args.model_channels * mult % 32 for mult in args.channel_mult) or args.model_channels % 32:
        parser.error('model channels must be divisible by 32 (GroupNorm)')
    if any(args.model_channels * mult % args.num_heads for mult in args.channel_mult):
        parser.error('model channels must be divisible by --num-heads')
    if args.embedding_channels % 2:
        parser.error('--embedding-channels must be even')
    if not 0 <= args.dropout <= 1:
        parser.error('--dropout must be between 0 and 1')
    if not math.isfinite(args.randomness) or not 0 <= args.randomness <= 1:
        parser.error('--randomness must be finite and between 0 and 1')
    if args.sample != 'fm' and args.timesteps <= 50:
        parser.error('--timesteps must exceed 50 for this beta schedule')
    if args.sample == 'fm' and (not math.isfinite(args.fm_step) or not 0 < args.fm_step <= 1):
        parser.error('--fm-step must be finite and in (0, 1]')
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
        prompt_title = prompts[index] if index < 9 else f'No prompt {index - 8} ("")'
        axis.set_title(textwrap.fill(prompt_title, width=30), fontsize=10, pad=8)
        axis.set_axis_off()
    return figure


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
    model = CFGUNet(
        in_channel=3, out_channel=3, model_channel=args.model_channels,
        embedding_channel=args.embedding_channels, feature_channel=args.feature_channels,
        resblock_num=args.res_blocks, dropout=args.dropout, ch_mult=tuple(args.channel_mult),
        attention_resolution=tuple(args.attention_resolutions), num_heads=args.num_heads,
        image_size=args.image_size,
    )
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
        expected_method = 'fm' if args.sample == 'fm' else 'ddpm'
        if checkpoint.get('training_method', 'ddpm') != expected_method:
            raise ValueError(f'Checkpoint training method does not match --sample {args.sample}.')
        # torch.compile adds this prefix to training state dicts.
        state = checkpoint['model']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, '_orig_mod.')
        model.load_state_dict(state)
        del checkpoint, state
    else:
        print('Using an untrained model: this checks execution and layout, not image quality.', flush=True)
    model.to(device).eval()
    encoder = CLIPTextEncoder(args.text_model_name)
    options = dict(device=device, text_encoder=encoder, guidance_scale=args.guidance_scale,
                   eval_max_length=args.eval_max_length)
    if args.sample == 'fm':
        sampler = CFGFMSampler(args.fm_step, model, solver=args.solver, randomness=args.randomness, **options)
        num_steps = sampler.sample_nums
        print(f'FM solver: {args.solver}, randomness: {args.randomness:g}', flush=True)
    elif args.sample == 'ddim':
        sampler = CFGDDIMSampler(model, spacing=args.timestep_spacing, randomness=args.randomness,
                                 betas=make_beta_schedule(args.timesteps), **options)
        num_steps = sampler.num_timesteps
    else:
        sampler = CFGDDPMSampler(model, betas=make_beta_schedule(args.timesteps), **options)
        num_steps = sampler.num_timesteps

    prompts, context, mask = get_cifar10_prompt(9, encoder, max_length=args.eval_max_length, device=device)
    noise = torch.randn((12, 3, args.image_size, args.image_size), device=device)
    print(f'Sampling 9 prompted + 3 unconditional images with {args.sample.upper()}: {num_steps} steps', flush=True)
    for index, prompt in enumerate(prompts, start=1):
        print(f'{index}: {prompt}', flush=True)
    conditional = sampler.sample(noise[:9], context=context, attention_mask=mask)
    unconditional = sampler.sample(noise[9:], prompts=['', '', ''], guidance_scale=0.0)
    samples = torch.cat([conditional, unconditional])
    if not torch.isfinite(samples).all():
        raise RuntimeError('Sampling produced non-finite values; check the checkpoint and schedule.')
    figure = make_figure(samples, prompts, sample_method=args.sample,
                         guidance_scale=args.guidance_scale, random_init=args.random_init,
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
