# SIA × Nebius — "Haggle": a self-improving negotiation-outcome agent

**One-liner:** We took a hard, real B2B negotiation task and let **SIA improve an AI agent on two axes — its harness (code/prompt) *and* its model weights — entirely on Nebius, with one API key for both inference and fine-tuning.**

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

## Axis 2 — Weight self-improvement (LoRA fine-tune on Nebius) → `2_weights_losscurve.png`
Same task, we **LoRA fine-tuned Llama-3.3-70B on Nebius GPUs** (same `NEBIUS_API_KEY`):

| step | 13 | 26 | 39 |
|---|---|---|---|
| valid loss | 0.69 | 0.42 | **0.375** |
| train loss | 0.82 | 0.42 | **0.30** |

2.5M training tokens; the model demonstrably learned the task's price priors and output format.

## The "so what"
- **Two orthogonal, composable levers of self-improvement** on one task: SIA improves the *scaffold*; fine-tuning improves the *model*. (Mirrors the SIA paper's "harness update" and "weight update" results.)
- **One open-model stack, one Nebius key** — inference *and* GPU training. No proprietary frontier model needed for the target. Reproducible and cheap.
- **Real metric, real data.** Un-gameable ±10% scoring on real negotiations.

## Honest status (what's real vs. caveated)
- ✅ End-to-end SIA loop on Nebius; the harness hill-climb; the LoRA training curve; one-key inference + training.
- ⚠️ **LoRA *inference serving* is currently disabled on our Nebius account** (API returns an empty supported-model list → "contact support"). So we present the weights axis via the **training curve**, not yet a served base-vs-tuned accuracy delta. We're **one support toggle** from the full 2×2 (base 70B already measured at 53.3% — the tuned cell drops in the moment serving is enabled).
- 🛠️ Gotchas we solved live: macOS SSL certs; meta-agent token-budget blowups (reasoning models); feedback-context overflow (right-sized the data); **meta tool-call reliability — GLM-5 is the dependable Nebius meta engine** (Kimi/Qwen were flaky).

## Reproduce
```bash
python tasks/craigslist-bargains/build_task.py          # data
./run.sh                                                  # harness loop (set NEBIUS_API_KEY in .env)
python tasks/craigslist-bargains/build_finetune_data.py  # SFT data
# upload via /v1/files → POST /v1/fine_tuning/jobs (lora) → loss curve
python demo/make_charts.py                                # regenerate charts
```

**Charts:** `demo/1_harness_hillclimb.png`, `demo/2_weights_losscurve.png`, `demo/3_two_axes_summary.png`
