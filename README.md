# oss-sia-haggle

A [SIA](https://github.com/hexo-ai/sia) self-improvement task: **predict the final
agreed price of a Craigslist negotiation from a *truncated* transcript** — or detect
when no deal is reached. Built to run on the [Nebius Token Factory](https://docs.tokenfactory.nebius.com)
OpenAI-compatible API.

SIA runs a loop in which a meta-agent writes a task-specific agent, the agent is run
and scored, and a feedback agent rewrites it for the next generation. This repo is the
**task** (data builder, task spec, seed agent, evaluator) plus the **Nebius provider**
config — point SIA's `--task_dir` at it.

## The task

Each example is a buyer↔seller negotiation over one item. The agent sees the item, both
parties' private **target** prices, and the transcript **with the last 2 message turns
removed** (the closing agreement/acceptance is hidden). It must predict the **final
sale price**, or `NO_DEAL`.

Because the acceptance is hidden, a price mentioned mid-conversation is only a
*candidate* — a naive "guess the last number" baseline scores ~44%, leaving real room
to improve through inference. Design highlights:

| | |
|---|---|
| **Public** (agent sees) | item + buyer/seller targets + truncated dialogue; labeled `train_examples.jsonl` demos |
| **Private** (evaluator only) | full records + `final_price`/`deal` labels; held-out `test` split |
| **Splits** | train → public demos · validation → per-generation signal (subsampled to 150) · test → final held-out eval |
| **Metric** | within-±10% tolerance accuracy (+ per-class accuracy, MAE, MAPE) |

## Quickstart

```bash
# 1. Install SIA + the pydantic-ai engine (to run the meta/feedback agent on Nebius)
pip install 'sia-agent[claude]' 'pydantic-ai>=1.0'

# 2. Build the dataset — downloads the source splits and writes data/ + reference/
python tasks/craigslist-bargains/build_task.py

# 3. Keys: copy the template and add your Nebius key
cp .env.example .env        # then edit .env: NEBIUS_API_KEY=...

# 4. Run — the wrapper loads .env and runs SIA entirely on Nebius
./run.sh                    # MAX_GEN / RUN_ID env vars override; extra args pass through
```

`run.sh` sources `.env` (so keys never need re-exporting) and runs:

```bash
python -m sia run --task_dir ./tasks/craigslist-bargains \
        --meta-agent-profile nebius-meta \
        --target-agent-profile gptoss-nebius-target \
        --max_gen 5 --run_id 1
```

Verify the Nebius/meta wiring first with `python smoke_meta_nebius.py`. `profiles/nebius-meta.json`
runs the meta/feedback author on Nebius via `pydantic-ai`; reasoning-heavy models may need a
larger output cap — set `SIA_META_MAX_TOKENS` (the bundled impl defaults low).

**Meta/feedback author profiles:**

| Profile | Author model | Notes |
|---|---|---|
| `nebius-meta` | gpt-oss / Kimi on Nebius | fully on Nebius; open models can be unreliable at the tool-driven code authoring |
| `tokenrouter-claude-meta` | Claude Opus 4.8 via [TokenRouter](https://www.tokenrouter.com/docs) | reliable authoring through an OpenAI-compatible gateway; set `TOKENROUTER_API_KEY` |

Both keep the **target** model on Nebius. Override `run.sh`'s default by passing
`--meta-agent-profile tokenrouter-claude-meta` (later flags win), e.g. `./run.sh --meta-agent-profile tokenrouter-claude-meta`.

`providers/nebius.json` overrides SIA's bundled Nebius provider with the canonical
`https://api.tokenfactory.nebius.com/v1/` base URL; it's read from `./providers/` when
you run from the repo root (or set `$SIA_PROVIDERS_DIR`). Swap the target profile for
`qwen-nebius-target` / `kimi-nebius-target` to evaluate a different Nebius model. The
seed agent (`reference/reference_target_agent.py`) calls Nebius via the OpenAI SDK.

> **Meta/feedback engine:** the bundled `claude` engine is Anthropic-only, so it needs
> `ANTHROPIC_API_KEY`. To run the meta/feedback agent on Nebius too, install
> `pydantic-ai` (or `openhands-ai`) and pass a Nebius meta profile via
> `--meta-agent-profile`.

### Final held-out test evaluation

After a run, score the best generation's agent on the untouched test split:

```bash
# generate predictions for the test inputs, then:
python tasks/craigslist-bargains/data/public/evaluate.py \
       --gen-dir <dir-with-test-predictions> --split test
```

## Layout

```
tasks/craigslist-bargains/
├── build_task.py                    # fetches source + writes all data/reference artifacts
├── data/public/task.md              # task spec the meta/feedback agent reads
├── data/public/evaluate.py          # scorer (writes results.json; --split validation|test)
└── reference/reference_target_agent.py   # seed agent (Nebius / OpenAI SDK)
providers/nebius.json                # Nebius Token Factory provider override
```

Everything under `data/` (and `reference/SAMPLE_TASK_DESCRIPTIONS.md`, `meta.json`) is
**generated** by `build_task.py` and gitignored — run the builder to materialize it.

## Data & licensing

The code here is MIT-licensed (see `LICENSE`). The underlying **CraigslistBargains**
dataset (He et al., 2018; Stanford NLP CocoA) is **not redistributed** in this repo —
`build_task.py` fetches it from its original CodaLab source at build time. The dataset
card lists its license as "unknown"; consult the [original source](https://stanfordnlp.github.io/cocoa/)
before any downstream redistribution.

```bibtex
@inproceedings{he2018decoupling,
  title  = {Decoupling Strategy and Generation in Negotiation Dialogues},
  author = {He, He and Chen, Derek and Balakrishnan, Anusha and Liang, Percy},
  booktitle = {EMNLP},
  year   = {2018}
}
```
