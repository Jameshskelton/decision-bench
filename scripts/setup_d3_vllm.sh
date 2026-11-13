#!/usr/bin/env bash
# d3 on vLLM's System One endpoint. Support is on vLLM main since 2026-10-14 and not in a
# release yet, so this installs a nightly. Its card says the CUDA path has not been run on
# NVIDIA by the authors: check the cross-check step in README before trusting it.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv-vllm --python 3.12 --seed
uv pip install --python .venv-vllm/bin/python -U vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
.venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
