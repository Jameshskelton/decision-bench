#!/usr/bin/env bash
# Harness venv: data prep, runner, analysis. No GPU libraries.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-harness
.venv-harness/bin/pip install -q -U pip
.venv-harness/bin/pip install -q -r requirements-harness.txt
echo "harness ready: source .venv-harness/bin/activate"
