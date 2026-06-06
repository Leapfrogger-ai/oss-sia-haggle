#!/usr/bin/env python3
"""Seed target agent for craigslist-bargains (final-sale-price prediction).

Runs against Nebius Token Factory (OpenAI-compatible endpoint). Reads the truncated
negotiations from <dataset_dir>/negotiations.jsonl, asks the model to predict the
final agreed price (or "NO_DEAL"), and writes:

  <working_dir>/results/predictions.json   # {"details": [{"id", "prediction"}, ...]}
  <working_dir>/agent_execution.json       # per-item prompt/response trajectory

Deliberately simple baseline — the SIA feedback agent is expected to improve the
prompt, add few-shot examples from train_examples.jsonl, calibrate a price prior,
sharpen no-deal detection, etc.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Provider wiring (Nebius Token Factory, OpenAI-compatible). The SIA feedback agent
# may change MODEL to match the target profile.
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.getenv("SIA_TARGET_MODEL", "openai/gpt-oss-120b-fast")
CONCURRENCY = 5
MAX_DIALOGUE_CHARS = 4000

client = OpenAI(base_url=BASE_URL, api_key=os.getenv("NEBIUS_API_KEY"))

SYSTEM = (
    "You predict the outcome of a Craigslist price negotiation between a buyer and a "
    "seller. You are given the item, both parties' private target prices, and the "
    "negotiation transcript WITH THE FINAL TURNS REMOVED — the closing agreement is "
    "hidden, so you must infer where the negotiation lands. Predict the final agreed "
    "sale price, or NO_DEAL if you believe no agreement is reached. The settled price "
    "usually falls between the buyer's and seller's targets."
)


def build_prompt(rec: dict) -> str:
    turns = "\n".join(f"{t['speaker']}: {t['text']}" for t in rec.get("dialogue", []))
    turns = turns[:MAX_DIALOGUE_CHARS] or "(no visible dialogue)"
    return (
        f"ITEM: {rec.get('title','')}\n"
        f"Category: {rec.get('category','')}\n"
        f"Listed price: {rec.get('list_price')}\n"
        f"Description: {str(rec.get('description',''))[:600]}\n\n"
        f"Buyer target: {rec.get('buyer_target')}\n"
        f"Seller target: {rec.get('seller_target')}\n\n"
        f"NEGOTIATION (closing turns hidden):\n{turns}\n\n"
        'Respond with JSON only: {"prediction": <number>}  or  {"prediction": "NO_DEAL"}'
    )


def parse_prediction(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group()).get("prediction")
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                if re.sub(r"[^a-z]", "", val.lower()) == "nodeal":
                    return "NO_DEAL"
                num = re.search(r"-?\d+(?:\.\d+)?", val.replace(",", ""))
                if num:
                    return float(num.group())
        except json.JSONDecodeError:
            pass
    if "NO_DEAL" in text.upper():
        return "NO_DEAL"
    num = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(num.group()) if num else "NO_DEAL"


def _create(messages):
    """Call the chat endpoint, preferring JSON mode but degrading gracefully."""
    try:
        return client.chat.completions.create(
            model=MODEL, max_tokens=300, temperature=0.0, messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:  # noqa: BLE001 — some models reject response_format; retry plain
        return client.chat.completions.create(
            model=MODEL, max_tokens=300, temperature=0.0, messages=messages,
        )


def predict_one(idx: int, rec: dict) -> dict:
    prompt = build_prompt(rec)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        resp = _create(messages)
        raw = (resp.choices[0].message.content or "").strip()
        pred = parse_prediction(raw)
    except Exception as exc:  # noqa: BLE001 — log and continue so one failure ≠ whole run
        raw, pred = f"ERROR: {exc}", "NO_DEAL"
    return {"idx": idx, "id": rec["id"], "prediction": pred, "raw": raw, "prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--working_dir", type=Path, required=True)
    args = ap.parse_args()

    data_file = args.dataset_dir / "negotiations.jsonl"
    records = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(records)} negotiations from {data_file} | model={MODEL} @ {BASE_URL}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(lambda p: predict_one(*p), enumerate(records)))
    results.sort(key=lambda r: r["idx"])

    out_dir = args.working_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(
        {"model": MODEL,
         "details": [{"id": r["id"], "prediction": r["prediction"]} for r in results]},
        indent=2), encoding="utf-8")

    (args.working_dir / "agent_execution.json").write_text(json.dumps(
        [{"id": r["id"], "prompt": r["prompt"], "response": r["raw"], "prediction": r["prediction"]}
         for r in results], indent=2, ensure_ascii=False), encoding="utf-8")

    n_nodeal = sum(1 for r in results if r["prediction"] == "NO_DEAL")
    print(f"Predicted {len(results)} ({n_nodeal} NO_DEAL). Wrote {out_dir/'predictions.json'}")


if __name__ == "__main__":
    main()
