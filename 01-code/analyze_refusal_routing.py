#!/usr/bin/env python3
"""Refusal-side re-analysis of MoE routing: dimensionality, centroid movement, and expert sets."""
import os
from pathlib import Path
from itertools import combinations
import numpy as np

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
MIN_N = 20
SEED = 0
N_RESAMPLE = 12
TOPM = 8
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))


def pr(A):
    """participation ratio of rows of A (n,d): effective dimensionality."""
    A = np.asarray(A, np.float64)
    A = A - A.mean(0)
    n, d = A.shape
    G = (A @ A.T) if n <= d else (A.T @ A)
    ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    s = ev.sum()
    return float(s * s / (ev * ev).sum()) if s > 0 else 1.0


def pr_nc(Xl, n_sub, rng):
    """PR of one layer's logits (n,E), averaged over N_RESAMPLE subsamples of n_sub."""
    return float(np.mean([pr(Xl[rng.choice(len(Xl), n_sub, replace=False)])
                          for _ in range(N_RESAMPLE)]))


def sel_freq(se, E):
    """selected_experts (n,L,k) -> (L,E) per-layer selection frequency."""
    n, L, k = se.shape
    mh = np.zeros((n, L, E), np.float32)
    np.put_along_axis(mh, np.clip(se, 0, E - 1), 1.0, axis=2)
    return mh.mean(0)


def load_cell(m, a):
    f = ROOT / m / f"{a}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    k = set(d.files)
    if "R_router_logits" not in k or "C_router_logits" not in k:
        d.close()
        return None
    o = {"R_rl": d["R_router_logits"].astype(np.float32),
         "C_rl": d["C_router_logits"].astype(np.float32)}
    if "R_selected_experts" in k:
        o["R_se"] = d["R_selected_experts"].astype(np.int64)
        o["C_se"] = d["C_selected_experts"].astype(np.int64)
    d.close()
    if min(len(o["R_rl"]), len(o["C_rl"])) < MIN_N:
        return None
    return o


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    models = args.models or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    print(f"routing-dir: {ROOT}  ({len(models)} models)\n")

    A_rows, B_rows, C_rows = [], [], []
    for m in models:
        cells = {a: load_cell(m, a) for a in ATTACKS}
        cells = {a: c for a, c in cells.items() if c is not None}
        if len(cells) < 2:
            continue
        atts = list(cells)
        E, L = cells[atts[0]]["R_rl"].shape[2], cells[atts[0]]["R_rl"].shape[1]

        sep = np.zeros(L)
        for c in cells.values():
            sep += np.linalg.norm(c["R_rl"].mean(0) - c["C_rl"].mean(0), axis=1)
        Lpk = int(np.argmax(sep))

        n_sub = min(min(len(c["R_rl"]), len(c["C_rl"])) for c in cells.values())
        n_sub = min(n_sub, 150)
        prR, prC = {}, {}
        for a, c in cells.items():
            prR[a] = pr_nc(c["R_rl"][:, Lpk, :], n_sub, rng)
            prC[a] = pr_nc(c["C_rl"][:, Lpk, :], n_sub, rng)
        A_rows.append((m, Lpk, n_sub, prR, prC))

        muR = {a: c["R_rl"][:, Lpk, :].mean(0) for a, c in cells.items()}
        muC = {a: c["C_rl"][:, Lpk, :].mean(0) for a, c in cells.items()}
        gap = float(np.mean([np.linalg.norm(muR[a] - muC[a]) for a in atts]))
        dR, dC = [], []
        for a, b in combinations(atts, 2):
            dR.append(np.linalg.norm(muR[a] - muR[b]) / gap)
            dC.append(np.linalg.norm(muC[a] - muC[b]) / gap)
        B_rows.append((m, gap, float(np.mean(dR)), float(np.mean(dC))))

        if all("R_se" in c for c in cells.values()):
            ref_sets = {}
            for a, c in cells.items():
                fR = sel_freq(c["R_se"], E)[Lpk]
                fC = sel_freq(c["C_se"], E)[Lpk]
                delta = fR - fC
                ref_sets[a] = set(int(e) for e in np.argsort(-delta)[:TOPM])
            jac = [len(ref_sets[a] & ref_sets[b]) / len(ref_sets[a] | ref_sets[b])
                   for a, b in combinations(atts, 2)]
            C_rows.append((m, ref_sets, float(np.mean(jac))))

    SH = "=" * 92
    print(SH)
    print("PART A -- routing participation ratio (effective dim of router-logit cloud), n-controlled")
    print("  v2 mirror: is refused-routing PR attack-flat while complied-routing PR swings?")
    print(SH)
    print(f"  {'model':22s} {'Lpk':>4s} {'nsub':>5s}   {'PR refused (by attack)':32s}  {'spread':>7s}")
    for m, Lpk, ns, prR, prC in A_rows:
        rv = "  ".join(f"{a[:4]}={prR[a]:.1f}" for a in prR)
        cv = "  ".join(f"{a[:4]}={prC[a]:.1f}" for a in prC)
        sR = max(prR.values()) - min(prR.values())
        sC = max(prC.values()) - min(prC.values())
        print(f"  {m:22s} {Lpk:4d} {ns:5d}   R: {rv}")
        print(f"  {'':22s} {'':>4s} {'':>5s}   C: {cv}")
        print(f"  {'':22s}  -> refused PR spread {sR:.1f}   complied PR spread {sC:.1f}"
              f"   {'(refused flatter)' if sR < sC else '(complied flatter)'}")
    if A_rows:
        sR_all = [max(r[3].values()) - min(r[3].values()) for r in A_rows]
        sC_all = [max(r[4].values()) - min(r[4].values()) for r in A_rows]
        print(f"  POOLED: median refused-PR spread {np.median(sR_all):.2f}  "
              f"complied-PR spread {np.median(sC_all):.2f}  "
              f"({sum(a < b for a, b in zip(sR_all, sC_all))}/{len(sR_all)} models refused flatter)")

    print(f"\n{SH}")
    print("PART B -- cross-attack centroid movement (router logits), normalized by refused-complied gap")
    print("  small = attack-stable. attributes the cross-attack transfer failure.")
    print(SH)
    print(f"  {'model':22s} {'gap':>8s} {'dR (refused move)':>18s} {'dC (complied move)':>19s}")
    for m, gap, dr, dc in B_rows:
        tag = "refused more stable" if dr < dc else "complied more stable"
        print(f"  {m:22s} {gap:8.2f} {dr:18.2f} {dc:19.2f}   {tag}")
    if B_rows:
        drs = [r[2] for r in B_rows]
        dcs = [r[3] for r in B_rows]
        print(f"  POOLED median: dR {np.median(drs):.2f}  dC {np.median(dcs):.2f}  "
              f"({sum(a < b for a, b in zip(drs, dcs))}/{len(drs)} models: refused moves less)")

    print(f"\n{SH}")
    print("PART C -- refusal expert set (top-8 experts refused over-selects) and its cross-attack overlap")
    print("  Jaccard over the 4 attacks: 1.0 = identical refusal-expert set, 0 = disjoint")
    print(SH)
    for m, ref_sets, jac in C_rows:
        print(f"  {m:22s} mean pairwise Jaccard = {jac:.2f}")
        for a, s in ref_sets.items():
            print(f"  {'':22s}   {a:9s} experts {sorted(s)}")
    if C_rows:
        print(f"  POOLED median Jaccard = {np.median([r[2] for r in C_rows]):.2f}"
              f"   (high = attack-invariant refusal-expert set)")


if __name__ == "__main__":
    main()
