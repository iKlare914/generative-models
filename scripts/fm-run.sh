#!/usr/bin/env bash
set -euo pipefail

# Train flow matching from any directory using the existing project environment.
# Additional CLI arguments override the defaults below.
# EMA is enabled by default; pass --ema-decay VALUE or --no-use-ema.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/diffusion_train.py" \
  --use-ema \
  --method fm \
  --fm-step 0.01 \
  --image-size 32 \
  --epochs 16 \
  --batch-size 64 \
  --attention-resolutions 16 8 \
  --device cuda \
  --res-blocks 3 \
  --num-heads 4 \
  --log-samples \
  --model-channels 128 \
  --embedding-channels 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  --wandb-mode online \
  --save-interval-epoch 1 \
  --cache-dir "$SCRIPT_DIR/../.cache/huggingface/datasets" \
  --save-dir "$SCRIPT_DIR/../checkpoints/cifar10-fm" \
  "$@"
