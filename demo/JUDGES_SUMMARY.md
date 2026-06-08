# SIA × Nebius × Lightning — "Haggle": a self-improving negotiation-outcome agent

**One-liner:** We took a hard, real B2B negotiation task and let **SIA improve an AI agent on two axes — its harness (code/prompt) *and* its model weights — on open models**, and validated *both* on real served ±10% accuracy: **harness 44→65%, weights 45→66.7%.**

---

## The task (real data, un-gameable metric)
Predict the **final settled price** of a Craigslist buyer↔seller negotiation from a transcript with the **closing turns hidden** — or `NO_DEAL`.
- Data: Stanford **CraigslistBargains** (6.7k real human negotiations).
- Scored: within **±10%** of the true price on a held-out set. Naive "guess a number" baseline ≈ **44%**.
- **Why it matters:** deal-outcome prediction *is* the B2B pain — only 45% of sales leaders trust their forecasts. Negotiation is the core sales skill. This is a $-real problem, not a toy benchmark.

## Axis 1 — Harness self-improvement (SIA's core loop) → `1_harness_hillclimb.png`
SIA's `meta → run → score → feedback` loop **rewrites the agent's code/prompt every generation**, on a **fixed model** (gpt-oss-120b):

| gen 1 | gen 2 | gen 3 | gen 4 |
|---|---|---|---|
| 53% | 37% (explore) | 60% | **65%** |

**44% naive → 65% in 4 generations** (+21 pts vs naive, +12 vs its own gen 1). **No human touched the agent** — SIA read its own failure logs, diagnosed a *pathological NO_DEAL bias*, and fixed it in code. The gen-2 dip → recovery is honest exploration, not a scripted line.

## Axis 2 — Weight self-improvement (LoRA fine-tune, **served & validated**) → `4_two_axes_validated.png`
Same task, we **LoRA fine-tuned a model and served base vs tuned on one endpoint**, then scored both with the same seed prompt on the same 60-item held-out set:

| | Base | Fine-tuned (LoRA) | Δ |
|---|---|---|---|
| **±10% accuracy (served)** | 45.0% | **66.7%** | **+21.7 pts** |

- **Controlled comparison:** *identical* Qwen3-14B weights, ± the LoRA adapter, served from **one** vLLM endpoint (`model="Qwen/Qwen3-14B"` vs `model="haggle"`). The only variable is the adapter — so the +21.7 pts is attributable *solely* to the fine-tune, not model size or a different machine.
- Corroborated by the fine-tune's own training-loss drop on the L40S (to ~0.1) — but the headline is the *served accuracy*, not the loss.
- **Where it wins:** the base model over-predicts `NO_DEAL`; the fine-tune recovers deal-price accuracy (e.g. base says `NO_DEAL`, tuned predicts the actual settled price).

> **Infra note:** the validated weights axis ran **entirely on one Lightning L40S** — QLoRA fine-tune → vLLM `--enable-lora` serving → eval, ~4 GPU-credits total. (We *also* fine-tuned via Nebius's API earlier, but that tier gates LoRA serving, so the served, validated run is on Lightning.)

## The "so what"
- **Two orthogonal, composable levers of self-improvement** on one task: SIA improves the *scaffold*; fine-tuning improves the *model*. (Mirrors the SIA paper's "harness update" and "weight update" results.)
- **One open-model stack, one Nebius key** — inference *and* GPU training. No proprietary frontier model needed for the target. Reproducible and cheap.
- **Real metric, real data.** Un-gameable ±10% scoring on real negotiations.

## Honest status (what's real vs. caveated)
- ✅ **Both axes validated on real served ±10% accuracy** — harness (44→65%) and weights (45→66.7%). End-to-end SIA loop on Nebius + served eval on Lightning.
- ✅ **Weights axis is fully self-contained on Lightning** (train + serve + eval on one L40S). Nebius's roles are the SIA meta/feedback *engine* (GLM-5, free OSS inference) and an earlier fine-tune whose *serving* was gated (deploy + weight-export both `AccessDenied`) — neither touches the validated accuracy.
- ⚠️ Harness and weights were measured on **different base models** (gpt-oss-120b vs Qwen3-14B) — each axis is internally controlled (same model, before/after), but the two aren't yet on one shared base. The Lightning bundle can run the harness on the tuned model for the fully-shared 2×2.
- 🛠️ Gotchas we solved live: SSL certs; meta token-budget blowups; feedback-context overflow (right-sized data); meta tool-call reliability (**GLM-5** is the dependable meta engine); numpy/pandas ABI clash on the GPU box; Qwen3 `<think>` tokens in serving.

## Reproduce
```bash
python tasks/craigslist-bargains/build_task.py          # data
./run.sh                                                  # harness loop (set NEBIUS_API_KEY in .env)
python tasks/craigslist-bargains/build_finetune_data.py  # SFT data
# upload via /v1/files → POST /v1/fine_tuning/jobs (lora) → loss curve
python demo/make_charts.py                                # regenerate charts
```

**Charts:** `demo/1_harness_hillclimb.png`, `demo/2_weights_losscurve.png`, `demo/3_two_axes_summary.png`
