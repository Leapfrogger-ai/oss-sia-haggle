# Lightning bundle — validated WEIGHTS proof (served base-vs-tuned)

This produces the thing the Nebius loss curve could only *imply*: the **served, ±10%-accuracy
delta between the base and the LoRA-fine-tuned model** — fully in your control, no Nebius
serving/export gate. Everything runs on **one Lightning L40S Studio**.

## Why this works
- We **fine-tune on Lightning** (your `train.jsonl` is portable; base model pulls from HF —
  Qwen3-14B is ungated). No need to export weights from Nebius (which is 403-blocked).
- vLLM `--enable-lora` serves **base AND tuned on one endpoint** → switch per request:
  - `model="Qwen/Qwen3-14B"` → base  (**Cell A**, and **C** with an evolved prompt)
  - `model="haggle"` → base+adapter (**Cell B**, and **D** with an evolved prompt)

## Setup (in the Lightning Studio terminal)
Create a Studio: **AI development · GPU · L40S (48 GB)**. Then:
```bash
git clone https://github.com/Leapfrogger-ai/oss-sia-haggle && cd oss-sia-haggle
bash demo/lightning/setup.sh                 # deps + regenerate data from source (~3 min)
python demo/lightning/finetune_lora.py       # QLoRA on Qwen3-14B → ./haggle-adapter (~30 min)
bash demo/lightning/serve.sh &               # vLLM serves base + adapter on :8000 (first run downloads ~28GB)
#   wait until it logs "Application startup complete", then:
python demo/lightning/eval_ab.py             # Cell A vs Cell B accuracy → the weights delta
```
Output is the validated weights axis:
```
Cell A  base  Qwen3-14B + seed prompt : XX.X%
Cell B  tuned (LoRA)    + seed prompt : YY.Y%
Δ weights = +Z.Z pts
```
…and `demo/lightning/ab_results.json` (with deal/no-deal/MAPE breakdown).

## Optional — full 2×2 / dual hill-climb (Cells C & D)
Point SIA's harness at the local endpoint. The meta agent still runs on Nebius (GLM-5),
so this one step needs your Token Factory key:
```bash
pip install -e ".[pydantic-ai]"
export NEBIUS_API_KEY="<your token-factory key>"      # for the GLM-5 META agent only
export NEBIUS_BASE_URL="http://localhost:8000/v1"     # TARGET agent hits local vLLM
# base curve (Cells A→C):
SIA_TARGET_MODEL="Qwen/Qwen3-14B" sia run --task_dir ./tasks/craigslist-bargains \
  --meta-agent-profile glm-meta --target-agent-profile llama70b-nebius-target \
  --max_gen 5 --run_id 100 --no-web
# tuned curve (Cells B→D):
SIA_TARGET_MODEL="haggle" sia run --task_dir ./tasks/craigslist-bargains \
  --meta-agent-profile glm-meta --target-agent-profile llama70b-nebius-target \
  --max_gen 5 --run_id 101 --no-web
```
Overlay `runs/run_100` vs `runs/run_101` → the **dual hill-climb** (both climb = harness;
tuned curve sits above = weights; top of tuned = compounding D).

## Optional — live haggle demo
```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"haggle","messages":[{"role":"user","content":"ITEM: bike ... NEGOTIATION ..."}],
  "chat_template_kwargs":{"enable_thinking":false}}'
# repeat with "model":"Qwen/Qwen3-14B" to show base vs tuned side-by-side, live
```

## Credit budget (L40S @ 2.14 credits/hr, you have 12)
setup ~0.1 · finetune ~0.5h · serve+eval ~0.5h · optional dual-climb ~1h  →  **~2–4 credits total.**

## Gotchas baked in
- **Qwen3 “thinking”**: every request sets `enable_thinking:false` so the model emits the JSON directly.
- **vLLM LoRA**: `--max-lora-rank 16` matches the adapter; `--gpu-memory-utilization 0.92` for headroom.
- First serve downloads the 14B base (~28 GB) — a few minutes, one time.
