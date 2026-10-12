"""Train a classifier-free guidance DDPM or flow matching model on CIFAR-10 or ImageNet.

Example:
    python train/cfg_diffusion_train.py --image-size 32 --epoches 1 --device cuda --ema-decay 0.9999
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from generative_model.config import TrainerConfig
from generative_model.dataset.image_transforms import DATASET_DIR, get_latent_dataset_dir
from generative_model.dataset.load_dataset import getCifarCLIPLoader, getImagenetCLIPLoader, getLatentImagenetCLIPLoader
from generative_model.cli_config import parse_args_with_config
from generative_model.models.unet import CFGUNet
from generative_model.models.dit import DiT
from generative_model.models.latent_diffusion import LatentDiffusion
from generative_model.sampler import (
    CFGDDPMSampler, CFGFMSampler, FMTimestepSampler, TimestepSampler, make_beta_schedule,
    CFGLatentDDPMSampler, CFGLatentFMSampler, validate_cfg_skip_steps,
)
from generative_model.dataset.text_encoder import CLIPTextEncoder
from generative_model.trainer import CFGTrainer, CFGFMTrainer, CFGLatentTrainer, CFGLatentFMTrainer


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-size', type=positive_int, required=True, help='Square RGB image size; latent spatial size is inferred from the VAE')
    parser.add_argument('--epoches', '--epochs', type=positive_int, default=100, help='Total target epochs, including epochs completed before resume')
    parser.add_argument('--batch-size', type=positive_int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--model-channel', '--model-channels', type=positive_int, default=64)
    parser.add_argument('--embedding-channel', '--embedding-channels', type=positive_int, default=256)
    parser.add_argument('--d-feature', '--feature-channels', type=positive_int, default=512, help='Text feature width: DiT.d_feature / CFGUNet.feature_channel; must match CLIP')
    parser.add_argument('--resblock-num', '--res-blocks', type=positive_int, default=2)
    parser.add_argument('--ch-mult', '--channel-mult', type=positive_int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attention-resolution', '--attention-resolutions', type=positive_int, nargs='*', default=[16, 8], help='Actual feature-map side lengths in encoder AND decoder, not downsampling factors; empty disables their attention. Bottleneck retains attention.')
    parser.add_argument('--num-heads', type=positive_int, default=4)
    parser.add_argument('--use-conv', action=argparse.BooleanOptionalAction, default=False, help='Use convolutions in resampling and residual shortcuts')
    parser.add_argument('--attn-o-proj-zeroinit', action=argparse.BooleanOptionalAction, default=False, help='Zero-initialize attention output projections')
    parser.add_argument('--method', choices=['ddpm', 'fm'], default='ddpm')
    parser.add_argument('--num-timesteps', '--timesteps', type=positive_int, default=1000, help='DDPM only: diffusion schedule length')
    parser.add_argument('--step', '--fm-step', type=float, default=0.01, help='FM only: integration step size in (0, 1] for logged samples; does not discretize training time')
    parser.add_argument('--solver', choices=['euler'], default='euler', help='FM only: solver (currently only euler is supported)')
    parser.add_argument('--dataset', default='uoft-cs/cifar10', help='Hugging Face dataset repository, matching --dataset-type')
    parser.add_argument('--text-model-name', default='openai/clip-vit-base-patch32', help='CLIP model ID or local directory; always encoded on CPU')
    parser.add_argument('--label-drop-rate', type=float, default=0.2, help='Probability of using empty-text conditioning in the dataset')
    parser.add_argument('--train-max-length', type=positive_int, default=16, help='Training token length, including BOS/EOS')
    parser.add_argument('--eval-max-length', type=positive_int, default=77, help='Token length for logged samples')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='CFG weight for logged samples; 1 means conditional prediction only')
    parser.add_argument('--use-cfgzero-star', action=argparse.BooleanOptionalAction, default=False, help='Enable CFG-Zero* for logged samples only; does not affect training loss')
    parser.add_argument('--skip-steps', type=int, default=None, help='Initial updates skipped by logged sampling; required with CFG-Zero*')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/huggingface/datasets'))
    parser.add_argument('--save-dir', type=Path, default=Path('checkpoints/cifar10-cfg'))
    parser.add_argument('--save-interval-epoch', type=positive_int, default=10)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True, help='Maintain EMA model weights')
    parser.add_argument('--ema-decay', type=float, help='EMA decay in [0, 1); required when EMA is enabled')
    parser.add_argument('--use-torch-compile', action=argparse.BooleanOptionalAction, default=False, help='Compile the training model with torch.compile')
    parser.add_argument('--log-samples', action=argparse.BooleanOptionalAction, default=False, help='Generate nine prompt-captioned images whenever a checkpoint is saved')
    parser.add_argument('--device', default='cuda', help='Torch device, e.g. cuda, cuda:0, or cpu; never silently falls back')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--model-type', choices=['unet', 'dit'], default='unet', help='Select model architecture independently of the dataset representation')
    parser.add_argument('--use-latent', action=argparse.BooleanOptionalAction, default=False, help='DiT training: read cached mean/std instead of encoding images; ignored for UNet and during sampling')
    parser.add_argument('--vae-model-name-or-path', '--vae-model-name', default='stabilityai/sdxl-vae', help='SD-style AutoencoderKL Hub ID or local directory; must match training')
    parser.add_argument('--vae-slicing', action=argparse.BooleanOptionalAction, default=True, help='Process VAE batch elements separately to reduce peak memory')
    parser.add_argument('--d-model', '--dit-hidden-size', type=positive_int, default=256)
    parser.add_argument('--num-layers', '--dit-depth', type=positive_int, default=8)
    parser.add_argument('--patch-size', '--dit-patch-size', type=positive_int, default=2, help='Patch size in latent pixels')
    parser.add_argument('--d-time-emb', '--dit-time-embedding', type=positive_int, default=256)
    parser.add_argument('--in-channel', type=positive_int, default=None, help='DiT latent channels; omitted reads VAE config, explicit value must match it')
    parser.add_argument('--grid-size-h', type=positive_int, default=None, help='DiT patch grid height; omitted derives from image/VAE/patch sizes')
    parser.add_argument('--grid-size-w', type=positive_int, default=None, help='DiT patch grid width; omitted derives from image/VAE/patch sizes')
    parser.add_argument('--extra-feature', action=argparse.BooleanOptionalAction, default=True, help='DiT text cross-attention; true for the CFG entry, false for the unconditional entry')
    parser.add_argument('--use-rope', action=argparse.BooleanOptionalAction, default=True, help='DiT: use 2D RoPE; false uses additive 2D sinusoidal embeddings')
    parser.add_argument('--t-freq', type=float, default=10000.0, help='DiT time embedding frequency base')
    parser.add_argument('--pe-freq', type=float, default=10000.0, help='DiT spatial positional embedding frequency base')
    parser.add_argument('--rope-cross-q', action=argparse.BooleanOptionalAction, default=False, help='Apply 2D RoPE to image queries in DiT cross-attention')
    parser.add_argument('--rope-cross-k', action=argparse.BooleanOptionalAction, default=False, help='Apply 2D RoPE to text keys; requires text length equal to image token count')
    parser.add_argument('--use-adaln-zero', action=argparse.BooleanOptionalAction, default=True, help='Zero-initialize DiT AdaLN modulation layers')
    parser.add_argument('--sample-posterior', action=argparse.BooleanOptionalAction, default=True, help='Latent training: sample the VAE posterior; false uses its mean')
    parser.add_argument('--dataset-type', choices=['cifar10', 'imagenet'], default='cifar10')
    parser.add_argument('--latent-cache-dir', type=Path, default=DATASET_DIR, help='Root containing cached ImageNet mean/std datasets, matching preprocessing --save-dir')
    parser.add_argument('--dataset-split', default='train', help='Training split; e.g. train[:8] for a CIFAR smoke test')
    # Sampling-only option: tolerate it in reused configs/commands, but never train on it.
    parser.add_argument('--prompt', default=None, help=argparse.SUPPRESS)
    args = parse_args_with_config(parser, argv)
    if args.model_type == 'unet':
        args.use_latent = False
    del args.prompt
    if args.use_ema and args.ema_decay is None:
        parser.error('--ema-decay is required when EMA is enabled; use --no-use-ema to disable it')
    if args.ema_decay is not None and not 0 <= args.ema_decay < 1:
        parser.error('--ema-decay must be finite and in [0, 1)')
    if args.model_type == 'dit':
        if args.d_model % args.num_heads:
            parser.error('--d-model must be divisible by --num-heads')
        if args.use_rope and (args.d_model // args.num_heads) % 4:
            parser.error('With --use-rope, --d-model / --num-heads must be divisible by 4')
        if not args.use_rope and args.d_model % 4:
            parser.error('Without RoPE, --d-model must be divisible by 4 for 2D sinusoidal embeddings')
        if not args.extra_feature:
            parser.error('--extra-feature must be true for this entry point')
        if not np.isfinite(args.t_freq) or args.t_freq <= 0 or not np.isfinite(args.pe_freq) or args.pe_freq <= 0:
            parser.error('--t-freq and --pe-freq must be finite and positive')
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
            parser.error('model channels at every level must be divisible by 32 (GroupNorm)')
        if any(args.model_channel * mult % args.num_heads for mult in args.ch_mult):
            parser.error('model channels at every level must be divisible by --num-heads')
        if args.embedding_channel % 2:
            parser.error('--embedding-channel must be even')
    if args.use_latent and args.dataset_type != 'imagenet':
        parser.error('--use-latent currently requires --dataset-type imagenet; use --no-use-latent for online encoding of CIFAR images')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not 0 <= args.dropout <= 1:
        parser.error('--dropout must be between 0 and 1')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight-decay must be finite and nonnegative')
    if args.method == 'ddpm' and args.num_timesteps <= 50:
        parser.error('--num-timesteps must exceed 50 for this beta schedule')
    if args.method == 'fm' and (not np.isfinite(args.step) or not 0 < args.step <= 1):
        parser.error('--step must be finite and in (0, 1]')
    if args.resume and not any(args.save_dir.glob('*_*_checkpoint.pt')):
        parser.error('--resume requires a checkpoint in --save-dir')
    if not 0 <= args.label_drop_rate <= 1:
        parser.error('--label-drop-rate must be between 0 and 1')
    if not 2 <= args.train_max_length <= 77 or not 2 <= args.eval_max_length <= 77:
        parser.error('--train-max-length and --eval-max-length must be between 2 and 77')
    if not np.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error('--guidance-scale must be finite and nonnegative')
    total_steps = int(np.ceil(1 / args.step)) if args.method == 'fm' else args.num_timesteps
    try:
        args.skip_steps = validate_cfg_skip_steps(args.use_cfgzero_star, args.skip_steps, total_steps)
    except ValueError as exc:
        parser.error(str(exc))
    return args


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
        in_channel = vae.config.latent_channels if args.in_channel is None else args.in_channel
        grid_size_h = grid_size if args.grid_size_h is None else args.grid_size_h
        grid_size_w = grid_size if args.grid_size_w is None else args.grid_size_w
        if in_channel != vae.config.latent_channels:
            raise ValueError('--in-channel must match the VAE latent_channels.')
        if (grid_size_h, grid_size_w) != (grid_size, grid_size):
            raise ValueError('--grid-size-h and --grid-size-w must match the latent image patch grid.')
        if args.use_rope and args.rope_cross_k and any(length != grid_size_h * grid_size_w for length in (args.train_max_length, args.eval_max_length)):
            raise ValueError('--rope-cross-k requires text token lengths to equal the image patch token count.')
        dit = DiT(
            image_size=(latent_size, latent_size), patch_size=args.patch_size,
            in_channel=in_channel, d_model=args.d_model,
            d_time_emb=args.d_time_emb, num_heads=args.num_heads,
            num_layers=args.num_layers, grid_size_h=grid_size_h, grid_size_w=grid_size_w,
            extra_feature=args.extra_feature, d_feature=args.d_feature,
            use_rope=args.use_rope, t_freq=args.t_freq, pe_freq=args.pe_freq,
            rope_cross_q=args.rope_cross_q, rope_cross_k=args.rope_cross_k,
            use_adaln_zero=args.use_adaln_zero,
        )
        return LatentDiffusion(dit, vae)
    return CFGUNet(
        in_channel=3, out_channel=3, model_channel=args.model_channel,
        embedding_channel=args.embedding_channel, feature_channel=args.d_feature,
        resblock_num=args.resblock_num,
        dropout=args.dropout, ch_mult=tuple(args.ch_mult),
        attention_resolution=tuple(args.attention_resolution),
        num_heads=args.num_heads, image_size=args.image_size,
        use_conv=args.use_conv, attn_o_proj_zeroinit=args.attn_o_proj_zeroinit,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.use_latent:
        latent_dir = get_latent_dataset_dir(args.dataset, args.image_size, args.dataset_split, args.latent_cache_dir)
        if not latent_dir.is_dir():
            raise FileNotFoundError(f'Latent cache not found: {latent_dir}. Run utils/download_and_process_imagenet.py --use-latent first.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; training stopped without falling back to CPU.')
    # Fail before downloading data if the requested device cannot execute a kernel.
    torch.ones(1, device=device).add_(1)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_MODE'] = args.wandb_mode
    os.environ['WANDB_DIR'] = str(args.save_dir.resolve())
    print(json.dumps(vars(args), indent=2, default=str), flush=True)

    config = TrainerConfig(
        lr=args.lr, batch_size=args.batch_size, dropout=args.dropout if args.model_type == 'unet' else 0.0,
        weight_decay=args.weight_decay, epoches=args.epoches,
        save_interval_epoch=args.save_interval_epoch, resume=args.resume,
        log_samples=args.log_samples, save_dir=args.save_dir,
        use_torch_compile=args.use_torch_compile,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        image_size=args.image_size, guided=True,
    )
    encoder = CLIPTextEncoder(args.text_model_name)
    model = build_model(args).to(device)
    if args.model_type == 'dit':
        trainer_class = CFGLatentFMTrainer if args.method == 'fm' else CFGLatentTrainer
    else:
        trainer_class = CFGFMTrainer if args.method == 'fm' else CFGTrainer
    if args.method == 'fm':
        sampler_class = CFGLatentFMSampler if args.model_type == 'dit' else CFGFMSampler
        sampler = sampler_class(
            args.step, model, device=device, text_encoder=encoder, solver=args.solver,
            guidance_scale=args.guidance_scale, eval_max_length=args.eval_max_length,
            use_cfgzero_star=args.use_cfgzero_star, skip_steps=args.skip_steps,
        )
        timestep_sampler = FMTimestepSampler('Uniform')
    else:
        sampler_class = CFGLatentDDPMSampler if args.model_type == 'dit' else CFGDDPMSampler
        sampler = sampler_class(
            model, betas=make_beta_schedule(args.num_timesteps), device=device, text_encoder=encoder,
            guidance_scale=args.guidance_scale, eval_max_length=args.eval_max_length,
            use_cfgzero_star=args.use_cfgzero_star, skip_steps=args.skip_steps,
        )
        timestep_sampler = TimestepSampler(args.num_timesteps, 'Uniform')
    if args.use_latent:
        loader_fn = getLatentImagenetCLIPLoader
    else:
        loader_fn = getImagenetCLIPLoader if args.dataset_type == 'imagenet' else getCifarCLIPLoader
    loader = loader_fn(
        args.dataset, args.dataset_split, batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, cache_dir=str(args.cache_dir),
        **({"save_dir": args.latent_cache_dir} if args.use_latent else {}),
        text_encoder=encoder, max_length=args.train_max_length, label_drop_rate=args.label_drop_rate,
    )
    print(f'Training on {len(loader.dataset)} images; {len(loader)} batches per epoch', flush=True)
    fixed_prompts = loader.dataset.prompts[:9] if args.dataset_type == 'imagenet' else None
    trainer = trainer_class(
        model=model, data=loader, diffusion_sampler=sampler,
        timestep_sampler=timestep_sampler, config=config, fixed_prompts=fixed_prompts,
        **({"sample_posterior": args.sample_posterior, "use_latent_dataset": args.use_latent} if args.model_type == 'dit' else {}),
    )
    trainer.train()
    return trainer


if __name__ == '__main__':
    main()
