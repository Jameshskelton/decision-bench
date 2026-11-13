#!/usr/bin/env bash
# Block until a system answers the smoke test (servers can take minutes to load and compile).
set -uo pipefail
cd "$(dirname "$0")/.."
SYSTEM="${1:?usage: wait_for.sh <system>}"
for i in $(seq 1 120); do
  if .venv-harness/bin/python -m bench.run smoke --system "$SYSTEM" >/dev/null 2>&1; then
    echo "$SYSTEM is up"; exit 0
  fi
  sleep 10
done
echo "$SYSTEM did not come up in 20 minutes"; exit 1
