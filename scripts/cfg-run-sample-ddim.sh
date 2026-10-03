#!/usr/bin/env bash
set -euo pipefail

# CFG entry point. Additional CLI arguments override defaults below.
# Pass --model-path checkpoint.pt, or --random-init for an untrained smoke test.
# Use --output-path samples.png to save the 4x3 titled figure.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/cfg_diffusion_sample.py" \
  --image-size 32 \
  --attention-resolutions 16 8 \
  --device cuda \
  --res-blocks 3 \
  --num-heads 4 \
  --timesteps 1000 \
  --model-channels 128 \
  --embedding-channels 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  --sample ddim \
  --timestep-spacing 10 \
  --randomness 0.0 \
  --feature-channels 512 \
  --guidance-scale 4.0 \
  --eval-max-length 77 \
  "$@"
