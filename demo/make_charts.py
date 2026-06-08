#!/usr/bin/env python3
"""Generate judge-ready demo charts from the run + fine-tune data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
harness = json.load(open(D / "harness_data.json"))
ft = json.load(open(D / "ft_data.json"))
SERVED = {"base": 45.0, "tuned": 66.67}   # Lightning Qwen3-14B served A/B (validated ±10% acc)

NAIVE = 44.0  # naive "guess a number" baseline from task.md
INK, ACCENT, ACCENT2, MUTE = "#1a1a2e", "#0F9D58", "#4285F4", "#9aa0a6"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white"})


# ── 1. Harness hill-climb (gpt-oss, the complete run) ───────────────────────────
def harness_chart():
    c = harness["run_6_gptoss"]
    gens = [d["gen"] for d in c]
    acc = [d["acc"] for d in c]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(NAIVE, ls="--", color=MUTE, lw=1.5)
    ax.text(gens[0], NAIVE + 0.6, "naive baseline ~44%", color=MUTE, fontsize=10)
    ax.plot(gens, acc, "-o", color=ACCENT, lw=2.6, ms=9)
    for g, a in zip(gens, acc):
        ax.annotate(f"{a:.0f}%", (g, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    best = max(acc)
    ax.annotate("best — agent fixed its own\nNO_DEAL over-bias",
                (gens[acc.index(best)], best), textcoords="offset points",
                xytext=(-12, -38), fontsize=9.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.set_title("Harness self-improvement — SIA rewrites the agent each generation")
    ax.set_xlabel("Generation"); ax.set_ylabel("Accuracy (±10%) on held-out negotiations")
    ax.set_xticks(gens); ax.set_ylim(30, 75); ax.grid(axis="y", alpha=.3)
    ax.text(0.99, -0.16, "Same model (gpt-oss-120b). SIA changed only code/prompt.",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTE)
    fig.tight_layout(); fig.savefig(D / "1_harness_hillclimb.png", dpi=160, bbox_inches="tight")


# ── 2. Weights loss curve (Llama-3.3-70B LoRA) ──────────────────────────────────
def weights_chart():
    c = ft["llama70b"]
    steps = [d["step"] for d in c]
    tr = [d["train_loss"] for d in c]; va = [d["valid_loss"] for d in c]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, tr, "-o", color=ACCENT2, lw=2.6, ms=9, label="train loss")
    ax.plot(steps, va, "-s", color=ACCENT, lw=2.6, ms=9, label="validation loss")
    for s, v in zip(steps, va):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=ACCENT)
    ax.annotate(f"valid {va[0]:.2f} → {va[-1]:.2f}", (steps[-1], va[-1]),
                textcoords="offset points", xytext=(-150, 8), fontsize=11,
                fontweight="bold", color=INK)
    ax.set_title("Weight self-improvement — LoRA fine-tune on Nebius GPUs")
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
    ax.set_xticks(steps); ax.grid(axis="y", alpha=.3); ax.legend()
    ax.text(0.99, -0.16, "Llama-3.3-70B · LoRA r16 · 2.5M tokens · same NEBIUS_API_KEY as inference",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTE)
    fig.tight_layout(); fig.savefig(D / "2_weights_losscurve.png", dpi=160, bbox_inches="tight")


# ── 3. Two-axes summary (side by side) ──────────────────────────────────────────
def combined_chart():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    c = harness["run_6_gptoss"]; g = [d["gen"] for d in c]; acc = [d["acc"] for d in c]
    a1.axhline(NAIVE, ls="--", color=MUTE, lw=1.3)
    a1.plot(g, acc, "-o", color=ACCENT, lw=2.6, ms=9)
    for gg, aa in zip(g, acc):
        a1.annotate(f"{aa:.0f}%", (gg, aa), textcoords="offset points", xytext=(0, 9), ha="center", fontweight="bold")
    a1.set_title("AXIS 1 · Harness  (code/prompt)"); a1.set_xlabel("Generation"); a1.set_ylabel("Accuracy %")
    a1.set_xticks(g); a1.set_ylim(30, 75); a1.grid(axis="y", alpha=.3)
    a1.text(g[0], NAIVE + .6, "naive ~44%", color=MUTE, fontsize=9)

    c2 = ft["llama70b"]; s = [d["step"] for d in c2]; va = [d["valid_loss"] for d in c2]; tr = [d["train_loss"] for d in c2]
    a2.plot(s, tr, "-o", color=ACCENT2, lw=2.6, ms=9, label="train")
    a2.plot(s, va, "-s", color=ACCENT, lw=2.6, ms=9, label="validation")
    a2.set_title("AXIS 2 · Weights  (LoRA fine-tune)"); a2.set_xlabel("Training step"); a2.set_ylabel("Loss")
    a2.set_xticks(s); a2.grid(axis="y", alpha=.3); a2.legend()
    fig.suptitle("SIA × Nebius — two axes of self-improvement on one task, one API key",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(D / "3_two_axes_summary.png", dpi=160, bbox_inches="tight")


# ── 4. Validated two-axes (harness curve + SERVED weights bar) ──────────────────
def validated_chart():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    c = harness["run_6_gptoss"]; g = [d["gen"] for d in c]; acc = [d["acc"] for d in c]
    a1.axhline(NAIVE, ls="--", color=MUTE, lw=1.3)
    a1.plot(g, acc, "-o", color=ACCENT, lw=2.6, ms=9)
    for gg, aa in zip(g, acc):
        a1.annotate(f"{aa:.0f}%", (gg, aa), textcoords="offset points", xytext=(0, 9), ha="center", fontweight="bold")
    a1.set_title("AXIS 1 · Harness  (code/prompt)"); a1.set_xlabel("Generation"); a1.set_ylabel("Accuracy %")
    a1.set_xticks(g); a1.set_ylim(30, 75); a1.grid(axis="y", alpha=.3)
    a1.text(g[0], NAIVE + .6, "naive ~44%", color=MUTE, fontsize=9)

    base, tuned = SERVED["base"], SERVED["tuned"]
    bars = a2.bar(["base\nQwen3-14B", "fine-tuned\n(LoRA)"], [base, tuned],
                  color=[MUTE, ACCENT], width=0.55)
    for bar, v in zip(bars, [base, tuned]):
        a2.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=14, fontweight="bold")
    a2.annotate(f"Δ = {tuned - base:+.1f} pts", xy=(1, tuned), xytext=(0.5, max(base, tuned) + 6),
                ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    a2.set_title("AXIS 2 · Weights  (LoRA fine-tune, served)"); a2.set_ylabel("Accuracy %")
    a2.set_ylim(0, max(base, tuned) + 14); a2.grid(axis="y", alpha=.3)
    fig.suptitle("SIA × Nebius × Lightning — two axes of self-improvement, both validated on served accuracy",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(D / "4_two_axes_validated.png", dpi=160, bbox_inches="tight")


# ── 5. Standalone served weights bar (A vs B, same 14B base) ────────────────────
def served_bar():
    base, tuned = SERVED["base"], SERVED["tuned"]
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    bars = ax.bar(["base\nQwen3-14B", "fine-tuned\n(LoRA)"], [base, tuned],
                  color=[MUTE, ACCENT], width=0.55)
    for bar, v in zip(bars, [base, tuned]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%",
                ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.annotate(f"Δ = {tuned - base:+.1f} pts", xy=(1, tuned), xytext=(0.5, max(base, tuned) + 6),
                ha="center", fontsize=13, color=ACCENT, fontweight="bold")
    ax.set_ylabel("±10% accuracy on held-out negotiations")
    ax.set_title("Served base vs fine-tuned — identical Qwen3-14B,\n± the LoRA adapter, one endpoint", fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, max(base, tuned) + 14); ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(D / "5_weights_served.png", dpi=160, bbox_inches="tight")


harness_chart(); weights_chart(); combined_chart(); validated_chart(); served_bar()
print("wrote:", *(p.name for p in sorted(D.glob("*.png"))))
