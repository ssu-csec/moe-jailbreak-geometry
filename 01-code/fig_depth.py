#!/usr/bin/env python3
"""Figure: the jailbroken state versus network depth."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
DATA = str(ROOT / "02-data" / "routing-analysis" / "layerwise_data.json")
OUT = str(ROOT / "00-paper" / "figures" / "fig_depth")

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_DEC = "#2b6cb0"
C_TR = "#dd6b20"
C_PERS = "#2f9e44"
C_CROSS = "#6b7280"
GRID = np.linspace(0.0, 1.0, 41)


def model_depths(r, kind):
    """Fractional depth axis (0..1) for one model's per-layer curve."""
    if kind == "routing":
        n = r["n_layers"]
        return np.arange(n) / (n - 1)
    return np.asarray(r["grid"], dtype=float) / (r["n_layers"] - 1)


def pooled(rows, key, kind):
    """Median and IQR across models of one per-layer curve on the common grid."""
    M = np.vstack([np.interp(GRID, model_depths(r, kind),
                             np.asarray(r[key], dtype=float)) for r in rows])
    return np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)


def thirds(v):
    """early/mid/late means, exactly as analyze_layerwise.thirds."""
    n = len(v)
    e, m = n // 3, 2 * n // 3
    return np.mean(v[:e]), np.mean(v[e:m]), np.mean(v[m:])


def report_thirds(rows, key):
    """Median over models of per-model thirds: reproduces analyze_layerwise.report."""
    t = np.array([thirds(np.asarray(r[key], dtype=float)) for r in rows])
    return np.median(t, 0)


def main():
    d = json.load(open(DATA))
    rt, rs = d["routing"], d["residual"]

    print(f"models: {len(rs)} residual, {len(rt)} routing")
    print("  exact-reproduction check (median over models of per-model thirds):")
    for label, rows, key in [
            ("residual decode   ", rs, "dec"),
            ("residual LOO-cross ", rs, "cr"),
            ("residual pers-cos  ", rs, "pe"),
            ("residual cross-cos ", rs, "cc"),
            ("residual null95   ", rs, "n95"),
            ("routing  decode    ", rt, "dec"),
            ("routing  LOO-cross ", rt, "cr")]:
        e, m, l = report_thirds(rows, key)
        print(f"    {label} early/mid/late = {e:+.2f} / {m:+.2f} / {l:+.2f}")

    rs_dec, rs_cr = pooled(rs, "dec", "residual"), pooled(rs, "cr", "residual")
    rt_dec, rt_cr = pooled(rt, "dec", "routing"), pooled(rt, "cr", "routing")
    rs_pe, rs_cc = pooled(rs, "pe", "residual"), pooled(rs, "cc", "residual")
    rs_n95 = pooled(rs, "n95", "residual")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.35, 4.2), sharex=True)

    ax0.axhline(0.5, ls="--", color="#444444", lw=1.0, zorder=0)
    ax0.text(0.99, 0.512, "chance", fontsize=7, color="#444444",
             ha="right", va="bottom")
    ax0.fill_between(GRID, rs_dec[1], rs_dec[2], color=C_DEC, alpha=0.15, lw=0)
    ax0.fill_between(GRID, rs_cr[1], rs_cr[2], color=C_TR, alpha=0.15, lw=0)
    ax0.plot(GRID, rs_dec[0], color=C_DEC, lw=2.2, label="decode, residual")
    ax0.plot(GRID, rs_cr[0], color=C_TR, lw=2.2, label="transfer, residual")
    ax0.plot(GRID, rt_dec[0], color=C_DEC, lw=1.3, ls="--",
             label="decode, routing")
    ax0.plot(GRID, rt_cr[0], color=C_TR, lw=1.3, ls="--",
             label="transfer, routing")
    ax0.set_ylabel("balanced accuracy")
    ax0.set_ylim(0.42, 1.0)
    ax0.set_xlim(0, 1)
    ax0.legend(fontsize=7, frameon=False, loc="center right",
               handlelength=1.9, handletextpad=0.5, labelspacing=0.3)
    ax0.text(0.02, 0.97, "(a)", transform=ax0.transAxes,
             fontweight="bold", va="top", ha="left")

    ax1.axhline(0.0, ls="--", color="#444444", lw=1.0, zorder=0)
    ax1.fill_between(GRID, -rs_n95[0], rs_n95[0], color="#9ca3af", alpha=0.22,
                     lw=0, zorder=0.5,
                     label="shuffled null $|cos|_{95}$")
    ax1.fill_between(GRID, rs_pe[1], rs_pe[2], color=C_PERS, alpha=0.15, lw=0)
    ax1.fill_between(GRID, rs_cc[1], rs_cc[2], color=C_CROSS, alpha=0.18, lw=0)
    ax1.plot(GRID, rs_pe[0], color=C_PERS, lw=2.2, label="persuasion pair")
    ax1.plot(GRID, rs_cc[0], color=C_CROSS, lw=2.2, label="cross-family")
    ax1.set_ylabel("decision-direction cosine")
    ax1.set_xlabel("relative network depth")
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=7, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 0.90),
               handlelength=1.9, handletextpad=0.5, labelspacing=0.3)
    ax1.text(0.02, 0.97, "(b)", transform=ax1.transAxes,
             fontweight="bold", va="top", ha="left")

    for ax in (ax0, ax1):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.align_ylabels([ax0, ax1])
    fig.tight_layout(pad=0.5, h_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
