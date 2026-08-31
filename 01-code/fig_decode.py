#!/usr/bin/env python3
"""Figure: within-attack routing decode."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
OUT = str(ROOT / "00-paper" / "figures" / "fig_decode")

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
ROUTING = [0.85, 0.95, 0.73, 0.99]
DISCRETE = [0.71, 0.77, 0.63, 0.90]
RESIDUAL = [0.94, 0.95, 0.75, 0.98]

RT_COLOR, DC_COLOR, RS_COLOR = "#2b6cb0", "#90b8dd", "#b0b0b0"

fig, ax = plt.subplots(figsize=(3.35, 2.55))
x = np.arange(len(ATTACKS))
w = 0.26
ax.bar(x - w, ROUTING, w, color=RT_COLOR, label="router logits")
ax.bar(x, DISCRETE, w, color=DC_COLOR, label="selected experts")
ax.bar(x + w, RESIDUAL, w, color=RS_COLOR, label="residual stream")
ax.axhline(0.5, ls="--", color="#444444", lw=1.0, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(ATTACKS)
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0.0, 1.08)
ax.set_xlim(-0.55, 3.55)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           fontsize=8, handletextpad=0.3, columnspacing=1.1,
           bbox_to_anchor=(0.5, 1.04))

fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf and {OUT}.png")
