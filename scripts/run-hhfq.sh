#!/usr/bin/env bash
set -euo pipefail

# Run from any directory using the existing project environment.
# Additional CLI arguments override the defaults below.
# EMA is enabled by default; pass --ema-decay VALUE or --no-use-ema.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/diffusion_train.py" \
  --use-ema \
  --dataset "marcosv/ffhq-dataset" \
  --image-size 1024 \
  --epochs 8 \
  --batch-size 8 \
  --attention-resolutions 512 256 \
  --device cuda \
  --res-blocks 3 \
  --num-heads 4 \
  --timesteps 1000 \
  --log-samples \
  --model-channels 256 \
  --embedding-channels 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  --wandb-mode online \
  --save-interval-epoch 2 \
  --cache-dir "$SCRIPT_DIR/../.cache/huggingface/datasets" \
  --save-dir "$SCRIPT_DIR/../checkpoints/ffhq-20260924-1" \
  "$@"
