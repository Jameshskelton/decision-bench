#!/usr/bin/env bash
# d1-2B needs transformers>=5.19 (its card); d3 pins 5.17.0, so they get separate venvs.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-d1
.venv-d1/bin/pip install -q -U pip
.venv-d1/bin/pip install -q "transformers>=5.19" torch torchvision pillow accelerate
