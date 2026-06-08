#!/usr/bin/env python3
"""#6 — Qualitative side-by-side: where the BASE model fails (base prediction vs gold).

Uses existing artifacts only (no serving needed):
  - base 70B predictions: runs/cellA_base70b/results/predictions.json
  - gold + hidden closing turns: tasks/.../data/private/validation_labels.json
  - inputs (item, targets, transcript): tasks/.../data/public/negotiations.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
preds = {d["id"]: d["prediction"] for d in
         json.load(open(ROOT / "runs/cellA_base70b/results/predictions.json"))["details"]}
gold = json.load(open(ROOT / "tasks/craigslist-bargains/data/private/validation_labels.json"))
inputs = {json.loads(l)["id"]: json.loads(l)
          for l in (ROOT / "tasks/craigslist-bargains/data/public/negotiations.jsonl").read_text().splitlines() if l.strip()}

TOL = 0.10
def wrong(pid):
    g = gold[pid]; p = preds.get(pid)
    if g["deal"] == 0:
        return str(p).upper().replace("_", "").replace(" ", "") != "NODEAL"
    try:
        return abs(float(p) - g["final_price"]) / g["final_price"] > TOL
    except (TypeError, ValueError):
        return True

wrongs = [i for i in inputs if i in gold and wrong(i)]
# bucket the interesting failure modes
nd_as_deal = [i for i in wrongs if gold[i]["deal"] == 1 and "DEAL" in str(preds.get(i)).upper()]
deal_as_nd = [i for i in wrongs if gold[i]["deal"] == 0]
far_price  = [i for i in wrongs if gold[i]["deal"] == 1 and "DEAL" not in str(preds.get(i)).upper()]

print(f"base-70B wrong on {len(wrongs)}/{len(inputs)}  "
      f"(predicted NO_DEAL on real deals: {len(nd_as_deal)}, "
      f"predicted a price on real no-deals: {len(deal_as_nd)}, "
      f"price off >10%: {len(far_price)})\n")

def card(pid):
    inp, g = inputs[pid], gold[pid]
    closing = g["provenance"].get("removed_turns", [])
    lines = [f"### {inp['title'][:60]}  ({inp['category']})",
             f"- List ${inp['list_price']} | buyer target ${inp['buyer_target']} | seller target ${inp['seller_target']}",
             f"- **Base model predicted:** `{preds.get(pid)}`   →   **Truth:** `{g['answer']}`  ❌",
             "- Hidden closing turns (what the model had to infer):"]
    lines += [f"    > {t['speaker']}: {t['text'][:150]}" for t in closing] or ["    > (none)"]
    return "\n".join(lines)

picks = (nd_as_deal[:2] + far_price[:2] + deal_as_nd[:1])
out = ["# Qualitative: where the base Llama-3.3-70B fails (base prediction vs gold)\n",
       f"Base model accuracy on the 60-item held-out set: **53.3%** — it is wrong on **{len(wrongs)}** cases.",
       "These are the cases the fine-tune targets. The LoRA's held-out loss drop (0.69→0.375) is",
       "the model learning to assign higher probability to exactly these gold answers.\n",
       "_Note: this compares base-vs-gold. A base-vs-tuned comparison needs the tuned model served (gated)._\n",
       "---\n"]
for pid in picks:
    out.append(card(pid)); out.append("")
(Path(__file__).parent / "qualitative_base_vs_gold.md").write_text("\n".join(out))
print("wrote demo/qualitative_base_vs_gold.md  (", len(picks), "illustrative cases )")
print("\n".join(out[:0]))  # quiet
print("--- preview ---\n" + card(picks[0]) if picks else "no wrong cases?")
