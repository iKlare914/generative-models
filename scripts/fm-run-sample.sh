#!/usr/bin/env bash
set -euo pipefail

# Sample with flow matching from any directory. Pass --model-path /path/to/fm-checkpoint.pt.
# Architecture defaults match scripts/fm-run.sh; additional CLI arguments override defaults.
# Use --output-path samples-fm.png to save the grid instead of opening a viewer.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/diffusion_sample.py" \
  --sample fm \
  --fm-step 0.01 \
  --image-size 32 \
  --attention-resolutions 16 8 \
  --device cuda \
  --res-blocks 3 \
  --num-heads 4 \
  --model-channels 128 \
  --embedding-channels 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  "$@"
