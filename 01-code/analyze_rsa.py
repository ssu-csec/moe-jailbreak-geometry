#!/usr/bin/env python3
"""Cross-model representational similarity analysis of the shared base harmful requests."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
DEPTHS = [0.5, 0.7, 0.9]


def base_of(qid):
    return str(qid).split("__")[0]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def collect(root, model):
    """-> per_attack[depth][base_id][attack] = mean residual vector (float32)."""
    sums = {fr: {} for fr in DEPTHS}
    for atk in ATTACKS:
        f = Path(root) / model / f"{atk}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        for side in ("R", "C"):
            if f"{side}_residual" not in d.files:
                continue
            res = d[f"{side}_residual"]
            bids = [base_of(q) for q in d[f"{side}_query_ids"]]
            L = res.shape[1]
            for fr in DEPTHS:
                sl = res[:, int(round(fr * (L - 1))), :].astype(np.float32)
                for i, b in enumerate(bids):
                    k = (b, atk)
                    if k in sums[fr]:
                        sums[fr][k][0] += sl[i]
                        sums[fr][k][1] += 1
                    else:
                        sums[fr][k] = [sl[i].copy(), 1]
            del res
        d.close()
    per_attack = {fr: {} for fr in DEPTHS}
    for fr in DEPTHS:
        for (b, a), (s, c) in sums[fr].items():
            per_attack[fr].setdefault(b, {})[a] = s / c
    return per_attack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-dir",
                    default=os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    root = Path(args.routing_dir)
    if not root.exists():
        sys.exit(f"routing dir not found: {root}")
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())

    M = {}
    for m in models:
        M[m] = collect(root, m)
        print(f"  collected {m:24s}  {len(M[m][DEPTHS[0]])} base_ids", flush=True)

    common = None
    for m in models:
        s = set(M[m][DEPTHS[0]].keys())
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"\ncommon base_ids across {len(models)} models: {len(common)}")
    if len(common) < 10:
        sys.exit("too few common base_ids")

    SH = "=" * 90
    print(f"\n{SH}\nPART 1 -- CROSS-MODEL RSA: are the harmful requests laid out the same way?")
    print("  Spearman r of the base x base residual RSM between model pairs (1.0 = identical geometry)")
    print(SH)
    for fr in DEPTHS:
        rsms = {}
        for m in models:
            pa = M[m][fr]
            X = np.stack([np.mean(list(pa[b].values()), axis=0) for b in common])
            X = X - X.mean(0, keepdims=True)
            X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
            R = X @ X.T
            rsms[m] = R[np.triu_indices(len(common), k=1)]
        print(f"\n  depth {fr:.1f}   (rows/cols = models)")
        offs = []
        for i, mi in enumerate(models):
            cells = []
            for j, mj in enumerate(models):
                r = 1.0 if i == j else spearman(rsms[mi], rsms[mj])
                cells.append(r)
                if j > i:
                    offs.append(r)
            print(f"    {mi:22s} " + " ".join(f"{c:5.2f}" for c in cells))
        print(f"    mean off-diagonal RSM correlation: {np.mean(offs):.3f}")

    print(f"\n{SH}\nPART 2 -- ATTACK-INVARIANCE  (cross-check for experiment A)")
    print("  within = mean cosine among one base request's per-attack residual vectors")
    print("  across = mean cosine between different base requests")
    print("  within >> across  =>  residual reflects request content, not the attack wrapper")
    print(SH)
    rng = np.random.default_rng(0)
    for fr in DEPTHS:
        print(f"\n  depth {fr:.1f}")
        for m in models:
            pa = M[m][fr]
            gm = np.mean([v for b in pa for v in pa[b].values()], axis=0)

            def cen(v):
                w = v - gm
                return w / max(float(np.linalg.norm(w)), 1e-9)

            withins = []
            for b in pa:
                vs = [cen(v) for v in pa[b].values()]
                for x in range(len(vs)):
                    for y in range(x + 1, len(vs)):
                        withins.append(float(vs[x] @ vs[y]))
            flat = [(b, cen(v)) for b in pa for v in pa[b].values()]
            across = []
            while len(across) < 3000:
                i, j = rng.integers(len(flat), size=2)
                if flat[i][0] != flat[j][0]:
                    across.append(float(flat[i][1] @ flat[j][1]))
            w, a = float(np.mean(withins)), float(np.mean(across))
            print(f"    {m:22s} within {w:+.3f}  across {a:+.3f}  gap {w - a:+.3f}"
                  f"  (n_within_pairs={len(withins)})")

    print(f"\n{SH}")
    print("SUMMARY: PART 1 mean off-diagonal = is the harm-request geometry model-universal.")
    print("  PART 2 gap > 0 = the residual already separates requests by content under any attack.")


if __name__ == "__main__":
    main()
