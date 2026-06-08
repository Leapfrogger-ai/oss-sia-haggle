#!/usr/bin/env bash
# Serve BASE + TUNED on one OpenAI-compatible endpoint (:8000).
#   request model="Qwen/Qwen3-14B" -> base        (Cell A / C)
#   request model="haggle"          -> base+adapter (Cell B / D)
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
ADAPTER="${1:-./haggle-adapter}"
echo "== ensure vLLM is installed (separate from training deps) =="
python -c "import vllm" 2>/dev/null || pip install -q -U vllm openai
# Set SERVE_TOOLS=1 to also enable OpenAI tool-calling — needed only if you run the SIA
# META agent on this same local model (fully-on-box, no Nebius). Harmless for target/eval.
TOOLS=""
[ "${SERVE_TOOLS:-0}" = "1" ] && TOOLS="--enable-auto-tool-choice --tool-call-parser hermes"
echo "Serving Qwen/Qwen3-14B + LoRA adapter '$ADAPTER' on :8000 ${TOOLS:+(+tool-calling)}..."
vllm serve Qwen/Qwen3-14B \
  --enable-lora --lora-modules "haggle=${ADAPTER}" --max-lora-rank 16 \
  --max-model-len 4096 --gpu-memory-utilization 0.92 ${TOOLS} \
  --host 0.0.0.0 --port 8000
