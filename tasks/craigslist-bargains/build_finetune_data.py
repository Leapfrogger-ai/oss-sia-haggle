#!/usr/bin/env python3
"""Build SFT data for a Nebius LoRA fine-tune of the craigslist price-prediction task.

Produces OpenAI-style chat JSONL (one {"messages": [...]} per line) that mirrors EXACTLY
the system/user format the target agent uses at eval time (truncated transcript -> predict
final price or NO_DEAL), with the assistant turn = the ground-truth answer. Training rows
come only from the TRAIN split (test/validation are never touched).

  python build_finetune_data.py --n 2000        # -> finetune/train.jsonl + finetune/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Reuse the canonical parser/truncation so SFT inputs match the eval distribution exactly.
from build_task import TRUNCATE_LAST_N_TURNS, ensure_raw, parse_record, stratified_sample  # noqa: E402

ROOT = Path(__file__).resolve().parent
MAX_DIALOGUE_CHARS = 4000

SYSTEM = (
    "You predict the outcome of a Craigslist price negotiation between a buyer and a "
    "seller. You are given the item, both parties' private target prices, and the "
    "negotiation transcript WITH THE FINAL TURNS REMOVED — the closing agreement is "
    "hidden, so you must infer where the negotiation lands. Predict the final agreed "
    "sale price, or NO_DEAL if you believe no agreement is reached. The settled price "
    "usually falls between the buyer's and seller's targets."
)


def user_prompt(rec: dict) -> str:
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


def to_chat(rec: dict) -> dict:
    lab = rec["label"]
    answer = {"prediction": lab["final_price"]} if lab["deal"] else {"prediction": "NO_DEAL"}
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(rec["input"])},
        {"role": "assistant", "content": json.dumps(answer)},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/tmp/cb_raw"))
    ap.add_argument("--n", type=int, default=2000, help="training rows (stratified)")
    ap.add_argument("--val", type=int, default=200, help="validation rows (stratified)")
    args = ap.parse_args()

    ensure_raw(args.raw_dir)
    raw = json.loads((args.raw_dir / "train.json").read_text(encoding="utf-8"))
    recs = [r for r in (parse_record(ex) for ex in raw) if r is not None]
    # need a visible transcript to learn from
    recs = [r for r in recs if len(r["input"]["dialogue"]) >= 2]
    print(f"usable train records: {len(recs)} (truncate_last_n={TRUNCATE_LAST_N_TURNS})")

    sample = stratified_sample(recs, args.n + args.val, seed=42)
    random.Random(7).shuffle(sample)
    val = sample[: args.val]
    train = sample[args.val :]

    out = ROOT / "finetune"
    out.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        deals = sum(r["label"]["deal"] for r in rows)
        print(f"  → finetune/{name}.jsonl: {len(rows)} rows (deals={deals}, no_deals={len(rows)-deals})")


if __name__ == "__main__":
    main()
