#!/usr/bin/env python3
"""Figure: compliance- and refusal-state effective dimensionality by attack."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
DATA = str(ROOT / "02-data" / "pr_layers.json")
OUT = str(ROOT / "00-paper" / "figures" / "fig_decomp")

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_COLOR = "#2b6cb0"
R_COLOR = "#dd6b20"
ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
SIX = {"deepseek-v2-lite", "deepseek-moe-16b-chat", "qwen1.5-moe-a2.7b-chat",
       "llama-4-scout", "mixtral-8x7b", "olmoe-1b-7b"}

data = json.load(open(DATA))
cells, hook = data["cells"], data["hook_layer"]
prc = {a: [] for a in ATTACKS}
prr = {a: [] for a in ATTACKS}
for key, layers in cells.items():
    m, a = key.split("|")
    if m not in SIX:
        continue
    L = hook[m]
    e = min(layers, key=lambda x: abs(x["layer"] - L))
    prc[a].append(e["PR_C"])
    prr[a].append(e["PR_R"])

for a in ATTACKS:
    print(f"  {a:9s} PR_C median {np.median(prc[a]):6.1f}   "
          f"PR_R median {np.median(prr[a]):6.1f}   (n={len(prc[a])})")

fig, ax = plt.subplots(figsize=(3.35, 2.55))
rng = np.random.default_rng(0)
for i, a in enumerate(ATTACKS):
    for off, vals, color in [(-0.18, prc[a], C_COLOR), (0.18, prr[a], R_COLOR)]:
        med = float(np.median(vals))
        ax.bar(i + off, med, width=0.32, color=color, alpha=0.30,
               edgecolor="none", zorder=2)
        xs = i + off + rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(xs, vals, s=13, color=color, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=4)
        ax.plot([i + off - 0.16, i + off + 0.16], [med, med],
                color=color, lw=2.2, zorder=5)

ax.set_xticks(range(len(ATTACKS)))
ax.set_xticklabels(ATTACKS)
ax.set_ylabel("effective dimensionality")
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(0, 47)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

handles = [Line2D([0], [0], marker="o", color=C_COLOR, ls="", markersize=5,
                  label="compliance state (PR$_C$)"),
           Line2D([0], [0], marker="o", color=R_COLOR, ls="", markersize=5,
                  label="refusal state (PR$_R$)")]
ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right",
          handletextpad=0.2, borderpad=0.2)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf and {OUT}.png")
