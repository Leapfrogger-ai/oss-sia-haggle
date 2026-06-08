#!/usr/bin/env python3
"""Cell A (base) vs Cell B (tuned) — the validated weights delta.

Runs the SAME seed agent over the SAME 60-item held-out set against the local vLLM
endpoint, once on the base model and once on the LoRA-tuned model, scores both with
the repo's evaluate.py, and prints the accuracy delta. Run AFTER serve.sh is up.

  python demo/lightning/eval_ab.py
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "tasks/craigslist-bargains/data/public"
EVAL = PUB / "evaluate.py"
BASE_URL = "http://localhost:8000/v1"
client = OpenAI(base_url=BASE_URL, api_key="dummy")


def wait_ready(timeout=180):
    """Block until vLLM answers /v1/models — avoids scoring connection errors as NO_DEAL."""
    for _ in range(timeout // 3):
        try:
            urllib.request.urlopen(f"{BASE_URL}/models", timeout=3)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return False

SYSTEM = (
    "You predict the outcome of a Craigslist price negotiation between a buyer and a "
    "seller. You are given the item, both parties' private target prices, and the "
    "negotiation transcript WITH THE FINAL TURNS REMOVED — the closing agreement is "
    "hidden, so you must infer where the negotiation lands. Predict the final agreed "
    "sale price, or NO_DEAL if you believe no agreement is reached. The settled price "
    "usually falls between the buyer's and seller's targets."
)


def build_prompt(rec):
    turns = "\n".join(f"{t['speaker']}: {t['text']}" for t in rec.get("dialogue", []))[:4000] or "(none)"
    return (f"ITEM: {rec.get('title','')}\nCategory: {rec.get('category','')}\n"
            f"Listed price: {rec.get('list_price')}\nDescription: {str(rec.get('description',''))[:600]}\n\n"
            f"Buyer target: {rec.get('buyer_target')}\nSeller target: {rec.get('seller_target')}\n\n"
            f"NEGOTIATION (closing turns hidden):\n{turns}\n\n"
            'Respond with JSON only: {"prediction": <number>}  or  {"prediction": "NO_DEAL"}')


def parse(text):
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group()).get("prediction")
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, str) and re.sub(r"[^a-z]", "", v.lower()) == "nodeal":
                return "NO_DEAL"
        except json.JSONDecodeError:
            pass
    if "NO_DEAL" in text.upper():
        return "NO_DEAL"
    n = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(n.group()) if n else "NO_DEAL"


def predict_one(model, rec):
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(rec)}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3: no <think>
        )
        return {"id": rec["id"], "prediction": parse(r.choices[0].message.content or "")}
    except Exception as e:  # noqa: BLE001
        return {"id": rec["id"], "prediction": "NO_DEAL", "_err": str(e)[:80]}


def run(tag, model):
    recs = [json.loads(l) for l in (PUB / "negotiations.jsonl").read_text().splitlines() if l.strip()]
    print(f"[{tag}] model={model}  predicting {len(recs)} ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        preds = list(pool.map(lambda r: predict_one(model, r), recs))
    errs = [p for p in preds if "_err" in p]
    if errs:
        print(f"  ⚠️  {len(errs)}/{len(preds)} requests ERRORED (scored as NO_DEAL!) "
              f"— e.g. {errs[0]['_err']}")
        if len(errs) > len(preds) // 2:
            sys.exit(f"  ✗ majority of requests failed — is `serve.sh` up? Aborting (results would be bogus).")
    gendir = ROOT / "demo/lightning/runs" / tag
    (gendir / "results").mkdir(parents=True, exist_ok=True)
    (gendir / "results/predictions.json").write_text(json.dumps({"details": preds}))
    subprocess.run([sys.executable, str(EVAL), "--gen-dir", str(gendir)], check=True,
                   stdout=subprocess.DEVNULL)
    res = json.loads((gendir / "results.json").read_text())
    return res["accuracy_percent"], res


def main():
    if not wait_ready():
        sys.exit("✗ vLLM not reachable at :8000 after 180s — start `serve.sh` first.")
    print("server ready ✓")
    a, ra = run("A_base", "Qwen/Qwen3-14B")
    b, rb = run("B_tuned", "haggle")
    out = {"A_base_seedprompt": a, "B_tuned_seedprompt": b, "weights_delta_pts": round(b - a, 2),
           "A_detail": {k: ra[k] for k in ("deal_accuracy_percent", "nodeal_accuracy_percent", "price_mape_percent")},
           "B_detail": {k: rb[k] for k in ("deal_accuracy_percent", "nodeal_accuracy_percent", "price_mape_percent")}}
    (ROOT / "demo/lightning/ab_results.json").write_text(json.dumps(out, indent=2))
    print("\n========== WEIGHTS AXIS (served, validated) ==========")
    print(f"  Cell A  base  Qwen3-14B + seed prompt : {a:.2f}%")
    print(f"  Cell B  tuned (LoRA)    + seed prompt : {b:.2f}%")
    print(f"  Δ weights = {b - a:+.2f} pts")
    print("======================================================")
    print("saved demo/lightning/ab_results.json")


if __name__ == "__main__":
    main()
