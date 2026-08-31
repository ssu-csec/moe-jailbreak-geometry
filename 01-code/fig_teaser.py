#!/usr/bin/env python3
"""Figure 1 (teaser): attack-specific geometry of the jailbroken state in MoE routing.

For one model (llama-4-scout), all four jailbreak attacks' router-logit activations
(every MoE layer's per-expert scores at the last prompt token, concatenated) are
projected into a shared 2D frame via PCA-50 + t-SNE on the combined data. Each
attack occupies a distinct region with its own internal structure, visualizing the
central claim that the jailbroken compliance geometry is attack-specific.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
DATA = str(ROOT / "02-data" / "router_tsne_llama.json")
OUT = str(ROOT / "00-paper" / "figures" / "fig1_teaser")

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

ATTACK_COLORS = {
    "PAP": "#1f77b4",       # blue
    "hill": "#2ca02c",      # green
    "prefill": "#d62728",   # red
    "roleplay": "#9467bd",  # purple
}


def main():
    d = json.load(open(DATA))
    attacks = d["attacks"]
    pts = d["points"]
    model = d["model"]
    nL = d["n_layers"]
    nE = d["n_experts"]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    for atk in attacks:
        P = np.array(pts[atk] if isinstance(pts[atk], list) else pts[atk]["all"])
        ax.scatter(P[:, 0], P[:, 1], s=10, c=ATTACK_COLORS[atk], alpha=0.45,
                   edgecolors="none", label=atk)

    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)

    leg = ax.legend(loc="best", fontsize=8, frameon=False,
                    markerscale=1.4, handletextpad=0.3, title="attack")
    leg.get_title().set_fontsize(8)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
