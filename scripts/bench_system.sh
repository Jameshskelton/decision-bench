#!/usr/bin/env bash
# Full benchmark for one system whose server is already up (or Jev).
#   scripts/bench_system.sh kev-4b "kev @ $(git -C kev rev-parse --short HEAD)"
set -euo pipefail
cd "$(dirname "$0")/.."
SYSTEM="${1:?usage: bench_system.sh <system> [runtime description]}"
RUNTIME="${2:-}"
PY=.venv-harness/bin/python
$PY -m bench.manifest --system "$SYSTEM" --runtime "$RUNTIME" >/dev/null
$PY -m bench.run smoke    --system "$SYSTEM"
$PY -m bench.run accuracy --system "$SYSTEM" --datasets d2_calib d2_test d1_test --repeats 3
$PY -m bench.run load     --system "$SYSTEM"
echo "done: results/$SYSTEM"
