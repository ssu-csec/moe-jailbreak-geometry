#!/usr/bin/env python3
"""Figure: leave-one-attack-out transfer."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
OUT = str(ROOT / "00-paper" / "figures" / "fig_transfer")

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
ROUTING = [0.66, 0.67, 0.50, 0.50]
RESIDUAL = [0.71, 0.72, 0.51, 0.58]
WITHIN = 0.88

RT_COLOR, RS_COLOR = "#2b6cb0", "#b0b0b0"

fig, ax = plt.subplots(figsize=(3.35, 2.55))
x = np.arange(len(ATTACKS))
w = 0.34
ax.bar(x - w / 2, ROUTING, w, color=RT_COLOR, label="routing probe")
ax.bar(x + w / 2, RESIDUAL, w, color=RS_COLOR, label="residual probe")
ax.axhline(0.5, ls="--", color="#444444", lw=1.0, zorder=0)
ax.axhline(WITHIN, ls="--", color="#2f9e44", lw=1.0, zorder=0)
ax.text(3.5, WITHIN + 0.015, "within attack", fontsize=7, color="#2f9e44",
        ha="right", va="bottom")
ax.text(3.5, 0.5 + 0.015, "chance", fontsize=7, color="#444444",
        ha="right", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([f"held out:\n{a}" for a in ATTACKS])
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0.0, 1.05)
ax.set_xlim(-0.6, 3.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=8, handletextpad=0.3, columnspacing=1.4,
           bbox_to_anchor=(0.5, 1.03))

fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf and {OUT}.png")
