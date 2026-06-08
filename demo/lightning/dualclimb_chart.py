#!/usr/bin/env python3
"""The full single-model 2×2 / dual hill-climb — everything on ONE Qwen3-14B base.

Overlays SIA's harness curve on the BASE model (runs/run_100) vs the TUNED model
(runs/run_101), anchored at the seed-prompt baselines A & B from ab_results.json.
Run after the two `sia run` harness jobs + eval_ab.py.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GREEN, BLUE, MUTE = "#0F9D58", "#4285F4", "#9aa0a6"


def curve(run):
    fs = sorted(glob.glob(str(ROOT / f"runs/{run}/gen_*/results.json")),
                key=lambda p: int(p.split("gen_")[1].split("/")[0]))
    return [json.load(open(f))["accuracy_percent"] for f in fs]


def main():
    base, tuned = curve("run_100"), curve("run_101")
    ab = json.load(open(ROOT / "demo/lightning/ab_results.json"))
    A, B = ab["A_base_seedprompt"], ab["B_tuned_seedprompt"]
    if not base or not tuned:
        raise SystemExit("Missing harness runs — run sia run --run_id 100 (base) and 101 (tuned) first.")

    xb = list(range(0, len(base) + 1))   # gen 0 = seed-prompt baseline
    yb = [A] + base
    yt = [B] + tuned
    C, D = max(base), max(tuned)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xb, yb, "-o", color=MUTE, lw=2.4, ms=8, label="base Qwen3-14B")
    ax.plot(xb, yt, "-o", color=GREEN, lw=2.6, ms=9, label="fine-tuned Qwen3-14B (LoRA)")
    for x, y in [(0, A), (max(xb[1:], key=lambda i: yb[i]), C)]:
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", color=MUTE, fontweight="bold")
    for x, y in [(0, B), (max(xb[1:], key=lambda i: yt[i]), D)]:
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=GREEN, fontweight="bold")
    ax.annotate("A", (0, A), textcoords="offset points", xytext=(-16, -4), color=MUTE, fontsize=13, fontweight="bold")
    ax.annotate("B", (0, B), textcoords="offset points", xytext=(-16, -4), color=GREEN, fontsize=13, fontweight="bold")
    ax.annotate("C", (yb.index(C), C), textcoords="offset points", xytext=(10, -4), color=MUTE, fontsize=13, fontweight="bold")
    ax.annotate("D", (yt.index(D), D), textcoords="offset points", xytext=(10, -4), color=GREEN, fontsize=13, fontweight="bold")
    ax.set_title("Full 2×2 on one base — harness × weights (all Qwen3-14B, one endpoint)", fontweight="bold")
    ax.set_xlabel("SIA generation  (0 = seed prompt)"); ax.set_ylabel("±10% accuracy on held-out set")
    ax.set_xticks(xb); ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "demo/lightning/dualclimb.png", dpi=160, bbox_inches="tight")

    print("\n========== FULL 2×2 (all Qwen3-14B, served, same eval set) ==========")
    print(f"            base model        fine-tuned (LoRA)")
    print(f"  seed       A = {A:5.1f}%       B = {B:5.1f}%")
    print(f"  evolved    C = {C:5.1f}%       D = {D:5.1f}%")
    print(f"  harness gain (A->C): {C-A:+.1f}   weights gain (A->B): {B-A:+.1f}   both (A->D): {D-A:+.1f}")
    print("====================================================================")
    print("wrote demo/lightning/dualclimb.png")


if __name__ == "__main__":
    main()
