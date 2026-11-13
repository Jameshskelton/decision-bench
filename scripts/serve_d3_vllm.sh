#!/usr/bin/env bash
# Port 8000. --max-model-len 32768: without it vLLM profiles d3's full 262,144 positions at
# startup (bidirectional attention, no chunked prefill). Our states are far shorter.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-vllm/bin/vllm serve vllm-sr/d3 --max-model-len 32768 --port 8000 "$@"
