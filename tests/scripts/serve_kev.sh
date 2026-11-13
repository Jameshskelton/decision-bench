#!/usr/bin/env bash
# Port 8008. Weights pinned to the Kev 1.0 tag. The shipped temperature (T=2.41) stays on:
# we measure the model as its maker ships it, then recalibrate ourselves in bench.analyze.
set -euo pipefail
cd "$(dirname "$0")/../kev"
exec uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8008
