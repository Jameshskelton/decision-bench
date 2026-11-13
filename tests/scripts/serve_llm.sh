#!/usr/bin/env bash
# Port 8020. The prompted-LLM baseline. Usage: scripts/serve_llm.sh <hf-model-id>
# Pick an instruct model of similar size to d3 so the comparison is "decision fine-tune vs.
# prompting"; set the same id as systems.llm-baseline.model in config.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: serve_llm.sh <hf-model-id>}"
exec .venv-vllm/bin/vllm serve "$MODEL" --port 8020 --max-model-len 16384 --max-logprobs 20
