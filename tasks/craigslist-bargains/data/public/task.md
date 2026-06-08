# Task: Predict the final sale price of a Craigslist negotiation

## Objective

Each example is a buyer↔seller negotiation over a single Craigslist item. You are
given the item, both parties' private **target** prices, and the negotiation
transcript **with the last two message turns removed** — the closing agreement /
acceptance is deliberately hidden. Your job is to predict the **final agreed sale
price**, or output **`NO_DEAL`** if you believe the two never reach an agreement.

Because the closing turns are hidden, a price mentioned mid-conversation is only a
*candidate* — it may be countered or rejected. You must infer where the negotiation
actually lands. The settled price typically falls between the buyer's and seller's
targets.

## Input — `negotiations.jsonl` (in your `--dataset_dir`)

One JSON object per line:

```json
{
  "id": "C_ff9abf...",
  "category": "bike",
  "title": "Nirve beach comber bike",
  "description": "....",
  "list_price": 250.0,
  "buyer_target": 230.0,
  "seller_target": 250.0,
  "dialogue": [
    {"speaker": "seller", "text": "Ask me anything about the bike ..."},
    {"speaker": "buyer",  "text": "Hi, I'm interested in the bike"}
  ]
}
```

There are **no answer fields** in this file — predict every `id`.

## Labeled examples — `train_examples.jsonl` (same dir)

A sample of labeled negotiations in the **identical input format** plus the answer
fields `final_price` (float or `null`), `deal` (1/0), and `answer` (the price string
or `"NO_DEAL"`). Use these freely for few-shot prompting, to calibrate a price prior
(e.g. typical fraction of list price), or to tune your no-deal heuristic. They are a
*different* set of negotiations from the ones you are scored on.

## Output — write `results/predictions.json` into your `--working_dir`

```json
{
  "details": [
    {"id": "C_ff9abf...", "prediction": 240.0},
    {"id": "C_91d391...", "prediction": "NO_DEAL"}
  ]
}
```

`prediction` is a number (the predicted final price) or the string `"NO_DEAL"`.
Provide a prediction for every `id` in `negotiations.jsonl`. (A CSV `predictions.csv`
with columns `id,prediction` is also accepted.)

Also write an `agent_execution.json` trajectory log into `--working_dir` so the
improvement step can see how predictions were made.

> **Model note:** some chat models (e.g. Qwen3) emit hidden `<think>` reasoning that can
> crowd out your JSON answer within the token budget — yielding empty/`NO_DEAL` predictions.
> If your target model does this, disable it (pass
> `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`) and/or raise `max_tokens`.

## Scoring (`evaluate.py`)

- **Deal example:** correct if your predicted price is within **±10%** of the true
  final price.
- **No-deal example:** correct if you output `NO_DEAL`.
- **Headline metric:** `accuracy_percent` over all examples. The evaluator also
  reports per-class accuracy (deal vs. no-deal), price MAE, and MAPE.

A naive "guess a number from the transcript" baseline scores only ~44% — there is
substantial room to improve through better price inference and no-deal detection.
