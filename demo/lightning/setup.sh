#!/usr/bin/env bash
# Run from the repo root inside a Lightning AI Studio (L40S, "AI development").
# Installs deps and regenerates the task + fine-tune data from source (no Nebius needed).
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
echo "Repo root: $(pwd)"

echo "== install TRAINING deps (vLLM is installed separately by serve.sh to avoid a transformers pin clash) =="
# Pin numpy to 2.x: the cloudspace's prebuilt pandas/torch are built against NumPy 2 (96-byte
# dtype); letting a dep pull NumPy 1.x triggers a "dtype size changed 96 vs 88" ABI crash.
pip install -q -U "transformers>=4.51" "peft>=0.13" "datasets>=3.0" \
    "accelerate>=1.0" "bitsandbytes>=0.44" "numpy>=2.0,<3"

echo "== regenerate data from source (CodaLab download) =="
python tasks/craigslist-bargains/build_task.py
python tasks/craigslist-bargains/build_finetune_data.py --n 2000 --val 200

echo
echo "Data ready:"
wc -l tasks/craigslist-bargains/finetune/train.jsonl tasks/craigslist-bargains/finetune/val.jsonl

echo "== preflight: fast env check before the 30-min finetune =="
python demo/lightning/preflight.py

echo "Next: python demo/lightning/finetune_lora.py"
