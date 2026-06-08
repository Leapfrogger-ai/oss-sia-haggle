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

## Full 2×2 / dual hill-climb on ONE base (Cells C & D) — keep `serve.sh` running

Run SIA's harness loop against the **live local 14B endpoint**, twice (base vs tuned). SIA
goes in its **own venv** so its deps can't disturb the carefully-fixed serving/training env.

```bash
# one-time: SIA in an ISOLATED venv (does NOT touch the cloudspace numpy/torch/vllm env)
python -m venv ~/sia-venv && source ~/sia-venv/bin/activate
pip install -q 'sia-agent[pydantic-ai]'

# META (GLM-5) uses Token Factory; TARGET agent is pointed at the local vLLM via env:
export NEBIUS_API_KEY="<your token-factory key>"      # for the GLM-5 meta agent
export NEBIUS_BASE_URL="http://localhost:8000/v1"     # target agent hits local vLLM (same box)
cd ~/oss-sia-haggle

# Cell A→C : harness on the BASE 14B
SIA_TARGET_MODEL="Qwen/Qwen3-14B" sia run --task_dir ./tasks/craigslist-bargains \
  --meta-agent-profile glm-meta --target-agent-profile qwen14b-local-target \
  --max_gen 5 --run_id 100 --no-web
# Cell B→D : harness on the TUNED 14B (base + adapter, served as "haggle")
SIA_TARGET_MODEL="haggle" sia run --task_dir ./tasks/craigslist-bargains \
  --meta-agent-profile glm-meta --target-agent-profile qwen14b-local-target \
  --max_gen 5 --run_id 101 --no-web

# overlay the two curves + print the full 2×2  (uses cloudspace python for matplotlib)
deactivate
python demo/lightning/dualclimb_chart.py     # -> demo/lightning/dualclimb.png
```
**Reading it:** each curve climbing = the *harness* axis; the tuned curve sitting above the
base curve = the *weights* axis; the top of the tuned curve (D) = compounding. **All four
cells are the same Qwen3-14B weights on the same endpoint, scored by the same `evaluate.py`
on the same 60 negotiations** — only "adapter on/off" and "seed vs SIA-evolved prompt" change.

> Sanity check after each `sia run`: `cat runs/run_100/gen_1/results.json | grep accuracy` —
> if it's ~25% (the no-deal floor) the target hit `<think>` again; confirm `serve.sh` is up and
> the env vars are exported in the SIA-venv shell.

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
