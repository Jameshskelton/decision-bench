#!/usr/bin/env bash
# Port 8001, so it can sit next to the vLLM server during the cross-check (d3 is ~62 GiB
# per replica; two fit on one 141 GB H200 only if vLLM runs with --gpu-memory-utilization 0.5).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-d3/bin/python servers/hf_systemone_server.py --model vllm-sr/d3 --port 8001 --max-batch 1 "$@"
