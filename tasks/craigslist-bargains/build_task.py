#!/usr/bin/env python3
"""Build the craigslist-bargains SIA task from the raw CocoA/CodaLab JSON splits.

Task A, Variant 3 — "predict the settled price from a truncated negotiation":

  PUBLIC  (data/public/, the agent sees):
    - train_examples.jsonl : labeled demonstrations (truncated input + answer)
    - negotiations.jsonl   : the per-generation eval set (subsampled validation, inputs only)
    - sample_submission.csv: the expected submission format
  PRIVATE (data/private/, only evaluate.py sees):
    - validation_labels.json : full records + labels for the validation split (loop scoring)
    - test_labels.json       : full records + labels for the test split (final scoring)
    - test/negotiations.jsonl: truncated test inputs, shaped like a dataset_dir for the
                               one-off final evaluation run

Input transform (identical for validation + test prediction sets, and for the
demonstrated inputs in train):
  * keep only `message` turns; drop the structured offer/accept/reject/quit events
  * remove the last 2 message turns (hides the closing agreement / acceptance)

Label: final_price = outcome.offer.price when reward==1, else the string "NO_DEAL".

Run:  python build_task.py --raw-dir /tmp/cb_raw
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

# Raw CocoA/CraigslistBargains splits (the HF repo is a loader script that points here).
CODALAB_URLS = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
    "test": "https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/parsed.json",
}

TRUNCATE_LAST_N_TURNS = 2          # remove the closing agreement/acceptance
TRAIN_DEMO_CAP = 1000              # stratified sample of train kept as public demos
VALIDATION_LOOP_SIZE = 150         # stratified per-generation eval set (cost control)
SEED = 42
NO_DEAL = "NO_DEAL"


def _flatten(value) -> str:
    """CodaLab item fields are sometimes lists (e.g. multiple description blobs)."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).strip()
    return "" if value is None else str(value).strip()


def _num(value):
    try:
        f = float(value)
        return round(f, 2)
    except (TypeError, ValueError):
        return None


def parse_record(ex: dict) -> dict | None:
    """Turn one raw CocoA dialogue into a normalized record (input + label + provenance)."""
    scenario = ex.get("scenario") or {}
    kbs = scenario.get("kbs") or []
    if len(kbs) != 2:
        return None

    item = kbs[0].get("item", {}) or {}
    roles = {}
    for kb in kbs:
        personal = kb.get("personal", {}) or {}
        role = personal.get("Role")
        if role in ("buyer", "seller"):
            roles[role] = _num(personal.get("Target"))
    if "buyer" not in roles or "seller" not in roles:
        return None

    # role lookup by agent index, for tagging message speakers
    idx_role = {i: (kb.get("personal", {}) or {}).get("Role") for i, kb in enumerate(kbs)}

    messages, structured = [], []
    for e in ex.get("events", []):
        action = e.get("action")
        agent = e.get("agent")
        speaker = idx_role.get(agent, "unknown") if isinstance(agent, int) else "unknown"
        if action == "message":
            text = e.get("data")
            if isinstance(text, str) and text.strip():
                messages.append({"speaker": speaker, "text": text.strip()})
        elif action in ("offer", "accept", "reject", "quit"):
            data = e.get("data")
            price = data.get("price") if isinstance(data, dict) else None
            structured.append({"action": action, "speaker": speaker, "price": _num(price)})

    # Truncate: hide the last N message turns (the closing agreement).
    visible = messages[:-TRUNCATE_LAST_N_TURNS] if len(messages) > TRUNCATE_LAST_N_TURNS else []
    removed = messages[len(visible):]

    outcome = ex.get("outcome") or {}
    reward = outcome.get("reward")
    price = _num((outcome.get("offer") or {}).get("price"))
    deal = 1 if (reward == 1 and price is not None) else 0
    final_price = price if deal else None

    rid = ex.get("uuid") or scenario.get("uuid")
    item_record = {
        "id": rid,
        "category": scenario.get("category", ""),
        "title": _flatten(item.get("Title")),
        "description": _flatten(item.get("Description")),
        "list_price": _num(item.get("Price")),
        "buyer_target": roles["buyer"],
        "seller_target": roles["seller"],
        "dialogue": visible,
    }
    return {
        "input": item_record,
        "label": {"final_price": final_price, "deal": deal,
                  "answer": (str(final_price) if deal else NO_DEAL)},
        "provenance": {
            "full_dialogue": messages,
            "removed_turns": removed,
            "structured_events": structured,
            "outcome": outcome,
        },
    }


def stratified_sample(records: list[dict], k: int, seed: int) -> list[dict]:
    """Sample k records preserving the deal/no-deal ratio."""
    if k >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for r in records:
        by_class.setdefault(r["label"]["deal"], []).append(r)
    sample = []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = round(k * len(items) / len(records))
        sample.extend(items[:max(1, n)])
    rng.shuffle(sample)
    return sample[:k]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_raw(raw_dir: Path) -> None:
    """Download the raw splits from CodaLab if they're not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, url in CODALAB_URLS.items():
        dest = raw_dir / f"{split}.json"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {split} from CodaLab …")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — known, fixed https URLs


def write_sample_descriptions(records: list[dict], dest: Path) -> None:
    """Emit a few real (truncated input + answer) examples for the meta-agent prompt."""
    deals = [r for r in records if r["label"]["deal"] == 1 and len(r["input"]["dialogue"]) >= 3]
    nodeals = [r for r in records if r["label"]["deal"] == 0 and len(r["input"]["dialogue"]) >= 3]
    picks = deals[:2] + nodeals[:1]
    out = ["# Sample negotiations (truncated input + true answer)\n",
           "Each example shows exactly what the agent sees (item, targets, transcript with the",
           "last 2 turns hidden) and the held-out answer. The *removed closing turns* are shown",
           "only to illustrate why the answer is not directly readable from the visible text.\n"]
    for i, r in enumerate(picks, 1):
        inp, lab, prov = r["input"], r["label"], r["provenance"]
        out.append(f"## Sample {i}: {inp['title'][:70]}  ({inp['category']})")
        out.append(f"- Listed price: {inp['list_price']}  |  Buyer target: {inp['buyer_target']}  "
                   f"|  Seller target: {inp['seller_target']}\n")
        out.append("**Visible transcript (what you predict from):**")
        out += [f"- {t['speaker']}: {t['text'][:140]}" for t in inp["dialogue"]]
        out.append("")
        if prov["removed_turns"]:
            out.append("**Hidden closing turns (NOT shown to the agent):**")
            out += [f"- {t['speaker']}: {t['text'][:140]}" for t in prov["removed_turns"]]
            out.append("")
        tail = f"  (deal closed at ${lab['final_price']})" if lab["deal"] else "  (no agreement reached)"
        out.append(f"**Answer:** {lab['answer']}{tail}\n\n---\n")
    dest.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/tmp/cb_raw"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    public = root / "data" / "public"
    private = root / "data" / "private"
    reference = root / "reference"
    test_input_dir = private / "test"
    for d in (public, private, test_input_dir, reference):
        d.mkdir(parents=True, exist_ok=True)

    ensure_raw(args.raw_dir)

    parsed = {}
    for split in ("train", "validation", "test"):
        raw = json.loads((args.raw_dir / f"{split}.json").read_text(encoding="utf-8"))
        recs = [r for r in (parse_record(ex) for ex in raw) if r is not None]
        parsed[split] = recs
        dist = Counter(r["label"]["deal"] for r in recs)
        print(f"{split}: {len(recs)} records  (deals={dist.get(1,0)}, no_deals={dist.get(0,0)})")

    # ── PUBLIC ────────────────────────────────────────────────────────────────
    # train demos: stratified sample of (truncated input + answer)
    train_demos = stratified_sample(parsed["train"], TRAIN_DEMO_CAP, SEED)
    write_jsonl(public / "train_examples.jsonl",
                [{**r["input"], "final_price": r["label"]["final_price"],
                  "deal": r["label"]["deal"], "answer": r["label"]["answer"]}
                 for r in train_demos])
    print(f"  → public/train_examples.jsonl: {len(train_demos)} labeled demos "
          f"(stratified sample of {len(parsed['train'])} train)")

    # per-generation eval set: subsampled validation, inputs only
    val_loop = stratified_sample(parsed["validation"], VALIDATION_LOOP_SIZE, SEED)
    write_jsonl(public / "negotiations.jsonl", [r["input"] for r in val_loop])
    print(f"  → public/negotiations.jsonl: {len(val_loop)} validation inputs (loop eval set)")

    with (public / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        for r in val_loop[:3]:
            w.writerow([r["input"]["id"], "0.0"])
        w.writerow(["<id>", "NO_DEAL"])

    # ── PRIVATE ───────────────────────────────────────────────────────────────
    def labels_blob(recs):
        return {r["input"]["id"]: {"final_price": r["label"]["final_price"],
                                   "deal": r["label"]["deal"],
                                   "answer": r["label"]["answer"],
                                   "provenance": r["provenance"]} for r in recs}

    (private / "validation_labels.json").write_text(
        json.dumps(labels_blob(parsed["validation"]), ensure_ascii=False, indent=2), encoding="utf-8")
    (private / "test_labels.json").write_text(
        json.dumps(labels_blob(parsed["test"]), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → private/validation_labels.json: {len(parsed['validation'])} labels")
    print(f"  → private/test_labels.json: {len(parsed['test'])} labels")

    # truncated test inputs, shaped as a dataset_dir for the final eval run
    write_jsonl(test_input_dir / "negotiations.jsonl", [r["input"] for r in parsed["test"]])
    print(f"  → private/test/negotiations.jsonl: {len(parsed['test'])} test inputs (final eval)")

    # ── REFERENCE ─────────────────────────────────────────────────────────────
    write_sample_descriptions(parsed["validation"], reference / "SAMPLE_TASK_DESCRIPTIONS.md")
    print("  → reference/SAMPLE_TASK_DESCRIPTIONS.md: 3 worked examples")

    (root / "meta.json").write_text(json.dumps({
        "task": "craigslist-bargains",
        "variant": "A3-truncated-price-prediction",
        "truncate_last_n_turns": TRUNCATE_LAST_N_TURNS,
        "train_demo_cap": TRAIN_DEMO_CAP,
        "validation_loop_size": VALIDATION_LOOP_SIZE,
        "seed": SEED,
        "counts": {s: len(parsed[s]) for s in parsed},
    }, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
