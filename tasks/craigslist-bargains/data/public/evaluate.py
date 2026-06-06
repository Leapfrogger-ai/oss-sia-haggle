#!/usr/bin/env python3
"""Evaluate craigslist-bargains submissions (final-sale-price prediction).

The target agent reads the negotiations and writes predictions; this script scores
them against the held-out labels in data/private/.

Scoring:
  * deal examples   → correct if the predicted price is within ±10% of the true
                      final price (TOLERANCE).
  * no-deal examples→ correct if the prediction is the string "NO_DEAL".
Headline metric: accuracy_percent. Also reports MAE / MAPE / within-10% over the
deal subset, and a deal vs. no-deal breakdown.

Usage:
    python evaluate.py --gen-dir runs/run_1/gen_1                 # scores vs validation
    python evaluate.py --gen-dir <dir> --split test              # scores vs test (final)
    python evaluate.py --submission preds.json --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

TOLERANCE = 0.10            # ±10% counts as correct for a deal
NO_DEAL = "NO_DEAL"
TASK_DIR = Path(__file__).resolve().parent.parent.parent   # data/public/ -> task root
PRIVATE = TASK_DIR / "data" / "private"


# ── loading ────────────────────────────────────────────────────────────────────
def load_labels(split: str) -> dict:
    path = PRIVATE / (f"{split}_labels.json")
    if not path.is_file():
        raise FileNotFoundError(f"Labels not found for split '{split}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def eval_id_set(split: str, labels: dict) -> list:
    """The ids actually presented to the agent for this split — only these are scored.

    Validation is subsampled for the loop, so scoring against every private label
    would unfairly penalize the agent for rows it was never asked to predict.
    """
    input_file = {
        "validation": TASK_DIR / "data" / "public" / "negotiations.jsonl",
        "test": PRIVATE / "test" / "negotiations.jsonl",
    }.get(split)
    if input_file and input_file.is_file():
        ids = [json.loads(line)["id"]
               for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [i for i in ids if i in labels]
    return list(labels)


def find_submission(gen_dir: Path) -> Path | None:
    """Locate a predictions file written by the target agent inside gen_dir."""
    if not gen_dir.is_dir():
        return None
    results = gen_dir / "results"
    search_roots = [results, gen_dir] if results.is_dir() else [gen_dir]
    patterns = ["predictions*.json", "submission*.json", "results*.json",
                "predictions*.csv", "submission*.csv"]
    for root in search_roots:
        for pat in patterns:
            matches = sorted(root.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]
    # last resort: a single json/csv anywhere in the search roots
    for root in search_roots:
        files = list(root.glob("*.json")) + list(root.glob("*.csv"))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
    return None


def parse_submission(path: Path) -> dict:
    """Return {id: raw_prediction_string}. Accepts several shapes."""
    preds: dict[str, str] = {}
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row.get("id") or row.get("question_id")
                if rid is not None:
                    preds[str(rid)] = str(row.get("prediction") or row.get("model_answer") or "")
        return preds

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):                       # [{"id":..,"prediction":..}, ...]
        rows = data
    elif "details" in data:                          # {"details":[...]}
        rows = data["details"]
    elif "predictions" in data and isinstance(data["predictions"], dict):
        return {str(k): str(v) for k, v in data["predictions"].items()}
    else:                                            # {id: pred, ...}
        return {str(k): str(v) for k, v in data.items()}
    for row in rows:
        rid = row.get("id", row.get("question_id"))
        if rid is not None:
            preds[str(rid)] = str(row.get("prediction", row.get("model_answer", "")))
    return preds


# ── prediction normalization ─────────────────────────────────────────────────
# comma-grouped thousands (needs ≥1 group) OR a plain number; the plain form must be
# the fallback so "2500.0" isn't truncated to "250" by the 1-3-digit group.
_NUM = re.compile(r"-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def normalize(raw: str):
    """Return ("no_deal", None) | ("price", float) | ("invalid", None)."""
    if raw is None:
        return ("invalid", None)
    s = str(raw).strip()
    if not s:
        return ("invalid", None)
    if re.sub(r"[^a-z]", "", s.lower()) == "nodeal":
        return ("no_deal", None)
    m = _NUM.search(s.replace("$", ""))
    if m:
        try:
            return ("price", float(m.group().replace(",", "")))
        except ValueError:
            return ("invalid", None)
    return ("invalid", None)


# ── scoring ────────────────────────────────────────────────────────────────────
def evaluate(labels: dict, preds: dict, eval_ids: list) -> dict:
    correct = incorrect = missing = invalid = 0
    abs_errs, pct_errs, within = [], [], 0
    n_deals = sum(1 for i in eval_ids if labels[i]["deal"] == 1)
    deal_correct = nodeal_correct = 0
    details = []

    for rid in eval_ids:
        lab = labels[rid]
        true_deal = lab["deal"]
        true_price = lab["final_price"]
        raw = preds.get(str(rid))
        if raw is None:
            missing += 1
            details.append({"id": rid, "status": "missing", "true": lab["answer"]})
            continue
        kind, value = normalize(raw)

        if true_deal == 0:                                   # ground truth: no deal
            ok = (kind == "no_deal")
            nodeal_correct += int(ok)
        else:                                                # ground truth: a price
            if kind == "price":
                err = abs(value - true_price)
                abs_errs.append(err)
                if true_price:
                    pct_errs.append(err / abs(true_price))
                ok = true_price > 0 and err / abs(true_price) <= TOLERANCE
                within += int(ok)
                deal_correct += int(ok)
            elif kind == "invalid":
                invalid += 1
                details.append({"id": rid, "status": "invalid", "pred": raw, "true": lab["answer"]})
                incorrect += 1
                continue
            else:  # predicted NO_DEAL on a real deal
                ok = False
        correct += int(ok)
        incorrect += int(not ok)
        details.append({"id": rid, "status": "correct" if ok else "wrong",
                        "pred": raw, "true": lab["answer"]})

    total = len(eval_ids)
    acc = correct / total if total else 0.0
    n_nodeals = total - n_deals
    return {
        "total_questions": total,
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "invalid": invalid,
        "accuracy": round(acc, 4),
        "accuracy_percent": round(100 * acc, 2),
        "deal_accuracy_percent": round(100 * deal_correct / n_deals, 2) if n_deals else None,
        "nodeal_accuracy_percent": round(100 * nodeal_correct / n_nodeals, 2) if n_nodeals else None,
        "price_mae": round(sum(abs_errs) / len(abs_errs), 2) if abs_errs else None,
        "price_mape_percent": round(100 * sum(pct_errs) / len(pct_errs), 2) if pct_errs else None,
        "within_10pct": within,
        "n_deals": n_deals,
        "n_nodeals": n_nodeals,
        "tolerance": TOLERANCE,
        "timestamp": datetime.now().isoformat(),
        "details": details[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score craigslist-bargains predictions.")
    ap.add_argument("--gen-dir", type=Path, help="Generation dir containing the submission")
    ap.add_argument("--submission", type=Path, help="Explicit submission file (json/csv)")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    args = ap.parse_args()

    sub = args.submission or (find_submission(args.gen_dir) if args.gen_dir else None)
    if sub is None:
        raise SystemExit("No submission file found (looked for predictions*.json/csv).")
    print(f"Submission: {sub}")
    print(f"Scoring against split: {args.split}")

    labels = load_labels(args.split)
    preds = parse_submission(sub)
    eval_ids = eval_id_set(args.split, labels)
    results = evaluate(labels, preds, eval_ids)

    out_dir = args.gen_dir if args.gen_dir else sub.parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "details"}, indent=2))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
