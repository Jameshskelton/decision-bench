#!/usr/bin/env bash
# Fallback and cross-check path for d3: its own AutoModel code at the versions its card lists.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-d3
.venv-d3/bin/pip install -q -U pip
.venv-d3/bin/pip install -q "transformers==5.17.0" torch torchvision pillow opencv-python-headless \
  safetensors accelerate flash-linear-attention
