#!/usr/bin/env bash
# Run from the repo root inside a Lightning AI Studio (L40S, "AI development").
# Installs deps and regenerates the task + fine-tune data from source (no Nebius needed).
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
echo "Repo root: $(pwd)"

echo "== install TRAINING deps (vLLM is installed separately by serve.sh to avoid a transformers pin clash) =="
pip install -q -U "transformers>=4.51" "peft>=0.13" "datasets>=3.0" \
    "accelerate>=1.0" "bitsandbytes>=0.44"
# Reconcile NumPy LAST so it wins. The cloudspace base env is compiled against NumPy 1.x
# (scikit-learn<2, scipy<1.28, matplotlib<2, mistral-common<2.4 — and transformers imports
# sklearn at import time), so the whole stack must stay on numpy<2. numpy 1.26.4 satisfies
# every prebuilt constraint; pandas is downgraded to a numpy-1-compatible build (the base's
# pandas 3.0 is numpy-2-only and triggers the "dtype size 96 vs 88" ABI crash).
pip install -q "numpy==1.26.4" "pandas==2.2.3"

echo "== regenerate data from source (CodaLab download) =="
python tasks/craigslist-bargains/build_task.py
python tasks/craigslist-bargains/build_finetune_data.py --n 2000 --val 200

echo
echo "Data ready:"
wc -l tasks/craigslist-bargains/finetune/train.jsonl tasks/craigslist-bargains/finetune/val.jsonl

echo "== preflight: fast env check before the 30-min finetune =="
python demo/lightning/preflight.py

echo "Next: python demo/lightning/finetune_lora.py"
