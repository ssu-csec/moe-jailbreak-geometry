#!/usr/bin/env python3
"""Figure: cross-attack transfer matrix for the routing probe."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
OUT = str(ROOT / "00-paper" / "figures" / "fig_matrix")
ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
M = np.array([
    [0.83, 0.70, 0.50, 0.53],
    [0.60, 0.91, 0.51, 0.52],
    [0.47, 0.50, 0.69, 0.51],
    [0.53, 0.51, 0.50, 0.96],
])

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.0, 2.7))
im = ax.imshow(M, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="equal")
for i in range(4):
    for j in range(4):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.72 else "black")
ax.set_xticks(range(4))
ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_yticks(range(4))
ax.set_yticklabels(ATTACKS)
ax.set_xlabel("tested on")
ax.set_ylabel("trained on")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("balanced accuracy", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf and {OUT}.png")
