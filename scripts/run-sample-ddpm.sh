#!/usr/bin/env bash
set -euo pipefail

# Override YAML settings with CLI arguments; --config selects a different YAML file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/diffusion_sample.py" \
  --config "$SCRIPT_DIR/../config/test/run-sample-ddpm.yaml" "$@"
