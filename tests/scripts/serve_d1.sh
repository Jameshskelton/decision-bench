#!/usr/bin/env bash
# Port 8010. Batches concurrent requests through system_one_batch (max 32, 2 ms window).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-d1/bin/python servers/hf_systemone_server.py --model LiquidAI/d1-2B --port 8010 --compile "$@"
