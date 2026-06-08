#!/usr/bin/env python3
"""Bar chart of the validated weights axis (Cell A base vs Cell B tuned). Run after eval_ab.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
d = json.load(open(R / "ab_results.json"))
a, b = d["A_base_seedprompt"], d["B_tuned_seedprompt"]

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(["Base\nQwen3-14B", "Fine-tuned\n(LoRA)"], [a, b],
              color=["#9aa0a6", "#0F9D58"], width=0.6)
for bar, v in zip(bars, [a, b]):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%",
            ha="center", fontsize=15, fontweight="bold")
ax.annotate(f"Δ weights = {b - a:+.1f} pts", xy=(1, b), xytext=(0.5, max(a, b) + 6),
            ha="center", fontsize=13, color="#0F9D58", fontweight="bold")
ax.set_ylabel("±10% accuracy on held-out negotiations")
ax.set_title("Weights axis (validated, served on Lightning)\nsame seed prompt · same 60-item eval set",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, max(a, b) + 14)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(R / "ab_chart.png", dpi=160, bbox_inches="tight")
print(f"wrote {R / 'ab_chart.png'}   (A={a:.1f}%  B={b:.1f}%  Δ={b-a:+.1f})")
