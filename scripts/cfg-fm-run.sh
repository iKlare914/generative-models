#!/usr/bin/env bash
set -euo pipefail

# CFG entry point. Additional CLI arguments override defaults below.
# Install optional dependencies once: uv sync --extra clip-cfg
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/cfg_diffusion_train.py" \
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
  --save-dir "$SCRIPT_DIR/../checkpoints/cifar10-cfg-fm" \
  --feature-channels 512 \
  --guidance-scale 3.0 \
  --eval-max-length 77 \
  --label-drop-rate 0.2 \
  --train-max-length 16 \
  "$@"
