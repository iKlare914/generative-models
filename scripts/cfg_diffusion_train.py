"""Train a classifier-free guidance DDPM or flow matching model on CIFAR-10.

Example:
    python scripts/cfg_diffusion_train.py --image-size 32 --epochs 1 --device cuda --ema-decay 0.9999
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from generative_model.config import TrainerConfig
from generative_model.dataset.load_dataset import getCifarCLIPLoader
from generative_model.cli_config import parse_args_with_config
from generative_model.models.unet import CFGUNet
from generative_model.sampler import CFGDDPMSampler, CFGFMSampler, FMTimestepSampler, TimestepSampler, make_beta_schedule
from generative_model.trainer import CFGTrainer
from generative_model.dataset.text_encoder import CLIPTextEncoder


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image-size', type=positive_int, required=True, help='Square input image side length; CIFAR-10 is 32')
    parser.add_argument('--epochs', type=positive_int, default=100, help='Total target epochs, including epochs completed before resume')
    parser.add_argument('--batch-size', type=positive_int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--model-channels', type=positive_int, default=64)
    parser.add_argument('--embedding-channels', type=positive_int, default=256)
    parser.add_argument('--feature-channels', type=positive_int, default=512, help='CLIP text hidden size; must match the text model')
    parser.add_argument('--res-blocks', type=positive_int, default=2)
    parser.add_argument('--channel-mult', type=positive_int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attention-resolutions', type=positive_int, nargs='*', default=[16, 8], help='Actual feature-map side lengths in encoder AND decoder, not downsampling factors; empty disables their attention. Bottleneck retains attention.')
    parser.add_argument('--num-heads', type=positive_int, default=4)
    parser.add_argument('--use-conv', action=argparse.BooleanOptionalAction, default=False, help='Use convolutions in resampling and residual shortcuts')
    parser.add_argument('--attn-o-proj-zeroinit', action=argparse.BooleanOptionalAction, default=False, help='Zero-initialize attention output projections')
    parser.add_argument('--method', choices=['ddpm', 'fm'], default='ddpm')
    parser.add_argument('--timesteps', type=positive_int, default=1000, help='DDPM only: diffusion schedule length')
    parser.add_argument('--fm-step', type=float, default=0.01, help='FM only: integration step size in (0, 1] for logged samples; does not discretize training time')
    parser.add_argument('--solver', choices=['euler'], default='euler', help='FM only: solver (currently only euler is supported)')
    parser.add_argument('--dataset', default='uoft-cs/cifar10', help='Hugging Face CIFAR-format dataset repository')
    parser.add_argument('--text-model-name', default='openai/clip-vit-base-patch32', help='CLIP model ID or local directory; always encoded on CPU')
    parser.add_argument('--label-drop-rate', type=float, default=0.2, help='Probability of using empty-text conditioning in the dataset')
    parser.add_argument('--train-max-length', type=positive_int, default=16, help='Training token length, including BOS/EOS')
    parser.add_argument('--eval-max-length', type=positive_int, default=77, help='Token length for logged samples')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='CFG weight for logged samples; 1 means conditional prediction only')
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
    args = parse_args_with_config(parser, argv)
    if args.use_ema and args.ema_decay is None:
        parser.error('--ema-decay is required when EMA is enabled; use --no-use-ema to disable it')
    if args.ema_decay is not None and not 0 <= args.ema_decay < 1:
        parser.error('--ema-decay must be finite and in [0, 1)')
    max_scale = 2 ** (len(args.channel_mult) - 1)
    if args.image_size % max_scale:
        parser.error(f'--image-size must be divisible by {max_scale}')
    resolutions = {args.image_size // 2 ** level for level in range(len(args.channel_mult))}
    if not set(args.attention_resolutions).issubset(resolutions):
        parser.error(f'--attention-resolutions must be selected from {sorted(resolutions)}')
    if any(args.model_channels * mult % 32 for mult in args.channel_mult) or args.model_channels % 32:
        parser.error('model channels at every level must be divisible by 32 (GroupNorm)')
    if any(args.model_channels * mult % args.num_heads for mult in args.channel_mult):
        parser.error('model channels at every level must be divisible by --num-heads')
    if args.embedding_channels % 2:
        parser.error('--embedding-channels must be even')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not 0 <= args.dropout <= 1:
        parser.error('--dropout must be between 0 and 1')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight-decay must be finite and nonnegative')
    if args.method == 'ddpm' and args.timesteps <= 50:
        parser.error('--timesteps must exceed 50 for this beta schedule')
    if args.method == 'fm' and (not np.isfinite(args.fm_step) or not 0 < args.fm_step <= 1):
        parser.error('--fm-step must be finite and in (0, 1]')
    if args.resume and not any(args.save_dir.glob('*_*_checkpoint.pt')):
        parser.error('--resume requires a checkpoint in --save-dir')
    if not 0 <= args.label_drop_rate <= 1:
        parser.error('--label-drop-rate must be between 0 and 1')
    if not 2 <= args.train_max_length <= 77 or not 2 <= args.eval_max_length <= 77:
        parser.error('--train-max-length and --eval-max-length must be between 2 and 77')
    if not np.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error('--guidance-scale must be finite and nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
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
        lr=args.lr, batch_size=args.batch_size, dropout=args.dropout,
        weight_decay=args.weight_decay, epoches=args.epochs,
        save_interval_epoch=args.save_interval_epoch, resume=args.resume,
        log_samples=args.log_samples, save_dir=args.save_dir,
        use_torch_compile=args.use_torch_compile,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        image_size=args.image_size, guided=True,
    )
    encoder = CLIPTextEncoder(args.text_model_name)
    model = CFGUNet(
        in_channel=3, out_channel=3, model_channel=args.model_channels,
        embedding_channel=args.embedding_channels, feature_channel=args.feature_channels,
        resblock_num=args.res_blocks,
        dropout=args.dropout, ch_mult=tuple(args.channel_mult),
        attention_resolution=tuple(args.attention_resolutions),
        num_heads=args.num_heads, image_size=args.image_size,
        use_conv=args.use_conv, attn_o_proj_zeroinit=args.attn_o_proj_zeroinit,
    ).to(device)
    if args.method == 'fm':
        sampler = CFGFMSampler(
            args.fm_step, model, device=device, text_encoder=encoder, solver=args.solver,
            guidance_scale=args.guidance_scale, eval_max_length=args.eval_max_length,
        )
        timestep_sampler = FMTimestepSampler('Uniform')
    else:
        sampler = CFGDDPMSampler(
            model, betas=make_beta_schedule(args.timesteps), device=device, text_encoder=encoder,
            guidance_scale=args.guidance_scale, eval_max_length=args.eval_max_length,
        )
        timestep_sampler = TimestepSampler(args.timesteps, 'Uniform')
    loader = getCifarCLIPLoader(
        args.dataset, 'train', batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, cache_dir=str(args.cache_dir),
        text_encoder=encoder, max_length=args.train_max_length, label_drop_rate=args.label_drop_rate,
    )
    print(f'Training on {len(loader.dataset)} images; {len(loader)} batches per epoch', flush=True)
    trainer = CFGTrainer(
        model=model, data=loader, diffusion_sampler=sampler,
        timestep_sampler=timestep_sampler, config=config,
    )
    trainer.train()
    return trainer


if __name__ == '__main__':
    main()
