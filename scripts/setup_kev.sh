#!/usr/bin/env bash
# Kev-4B server (Apache-2.0). Needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d kev ] || git clone https://github.com/jaredpalmer/kev.git
cd kev && uv sync --extra serve
echo "kev code at $(git rev-parse HEAD)"
