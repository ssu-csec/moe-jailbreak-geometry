#!/usr/bin/env python3
"""Expert-identity analysis: which experts carry the refused/complied signal, and is that set sparse and consistent across attacks?"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("needs scikit-learn:  pip install scikit-learn")

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
N_CAP = 300
MIN_N = 12
SEED = 0
TOPN = [1, 2, 3, 5, 8, 12]


def lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def cohens_d(R, C):
    """Signed Cohen's d per expert: (R, C) each (n, E) -> (E,)."""
    nR, nC = len(R), len(C)
    sp = np.sqrt(((nR - 1) * R.var(0, ddof=1) + (nC - 1) * C.var(0, ddof=1))
                 / (nR + nC - 2))
    sp = np.where(sp < 1e-9, 1e-9, sp)
    return (R.mean(0) - C.mean(0)) / sp


def cv_bacc(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(cross_val_score(lr(), X, y, cv=skf,
                                 scoring="balanced_accuracy").mean())


def topn_curve(X, y):
    """Honest top-N probe: rank experts by |Cohen-d| on each train fold,
    score the top-N subset on the held-out fold."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {n: [] for n in TOPN}
    out["all"] = []
    for tr, te in skf.split(X, y):
        ytr = y[tr]
        order = np.argsort(-np.abs(cohens_d(X[tr][ytr == 0], X[tr][ytr == 1])))
        for n in TOPN:
            f = order[:n]
            m = lr().fit(X[tr][:, f], ytr)
            out[n].append(balanced_accuracy_score(y[te], m.predict(X[te][:, f])))
        m = lr().fit(X[tr], ytr)
        out["all"].append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return {k: float(np.mean(v)) for k, v in out.items()}


def part_ratio(d):
    """Participation ratio of d^2: ~E means spread, ~few means concentrated."""
    a = d ** 2
    s = a.sum()
    return float(s * s / (a ** 2).sum()) if s > 0 else 0.0


def pairs_mean(vecs):
    """Mean off-diagonal Pearson r among a list of vectors."""
    r = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            r.append(float(np.corrcoef(vecs[i], vecs[j])[0, 1]))
    return float(np.mean(r)) if r else float("nan")


def load(path):
    d = np.load(path, allow_pickle=True)
    R = d["R_router_logits"].astype(np.float32)
    C = d["C_router_logits"].astype(np.float32)
    d.close()
    nR, nC = len(R), len(C)
    if min(nR, nC) < MIN_N:
        return None
    rng = np.random.default_rng(SEED)
    R = R[rng.choice(nR, min(nR, N_CAP), replace=False)]
    C = C[rng.choice(nC, min(nC, N_CAP), replace=False)]
    X = np.concatenate([R, C], 0)
    y = np.array([0] * len(R) + [1] * len(C))
    dvec = np.stack([cohens_d(X[y == 0, i, :], X[y == 1, i, :])
                     for i in range(X.shape[1])])
    per_layer = [cv_bacc(X[:, i, :], y) for i in range(X.shape[1])]
    bi = int(np.argmax(per_layer))
    return dict(X=X, y=y, dvec=dvec, per_layer=per_layer, bi=bi,
                E=X.shape[2], nR=nR, nC=nC,
                topn=topn_curve(X[:, bi, :], y))


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

    cells = {}
    for m in models:
        for a in ATTACKS:
            f = root / m / f"{a}.npz"
            if not f.exists():
                continue
            c = load(f)
            if c is not None:
                cells[(m, a)] = c
            print(f"  loaded {m:24s} {a:9s}"
                  f"{'' if c else '  (too few -> skipped)'}", flush=True)
    if not cells:
        sys.exit("no probeable cells")

    SH = "=" * 96
    print(f"\n{SH}\nPART 1 -- SPARSITY: how many experts recover the within-attack signal")
    print("  top-N = probe restricted to the N most discriminative experts (selected on train folds)")
    print(SH)
    print(f"  {'model':22s} {'attack':9s} {'L*':>4s}  " +
          "  ".join(f"top{n:<2d}" for n in TOPN) + f"  {'allE':>5s}  top3/all")
    for (m, a), c in sorted(cells.items()):
        t = c["topn"]
        ch = lambda v: (v - 0.5)
        rec = ch(t[3]) / ch(t["all"]) if ch(t["all"]) > 0.02 else float("nan")
        row = "  ".join(f"{t[n]:5.2f}" for n in TOPN)
        print(f"  {m:22s} {a:9s} {c['bi']:4d}  {row}  {t['all']:5.2f}  {rec:6.2f}")
    print("  top3/all = fraction of the above-chance signal recovered by just 3 experts")

    print(f"\n{SH}\nPART 2 -- CONSISTENCY: is the discriminative expert set shared across attacks?")
    print("  Pearson r of per-expert Cohen-d vectors between attacks, at each model's global layer")
    print(SH)
    for m in models:
        present = [a for a in ATTACKS if (m, a) in cells]
        if len(present) < 2:
            continue
        glay = int(np.argmax(np.mean(
            [cells[(m, a)]["per_layer"] for a in present], axis=0)))
        dv = {a: cells[(m, a)]["dvec"][glay] for a in present}
        print(f"  {m}   (global layer L{glay})")
        for atr in present:
            row = "  ".join(
                f"{ate}:{np.corrcoef(dv[atr], dv[ate])[0, 1]:+.2f}"
                for ate in present if ate != atr)
            print(f"    {atr:9s} {row}")
        sem = [a for a in present if a != "prefill"]
        msem = pairs_mean([dv[a] for a in sem]) if len(sem) > 1 else float("nan")
        mall = pairs_mean([dv[a] for a in present])
        print(f"    mean r: all-pairs {mall:+.2f}   "
              f"semantic-only (PAP/hill/roleplay) {msem:+.2f}")

    print(f"\n{SH}\nPART 3 -- DEPTH: is it the same experts across layers (within an attack)?")
    print(SH)
    for (m, a), c in sorted(cells.items()):
        stab = pairs_mean(list(c["dvec"]))
        print(f"  {m:22s} {a:9s}  mean cross-layer r of the d-vector {stab:+.2f}"
              f"  (high = a depth-persistent expert set)")

    print(f"\n{SH}\nPART 4 -- DIRECTION: the expert-space refusal vector + target experts")
    print("  PR = participation ratio of d^2 (low = sparse).  +d: refused-favored, -d: complied-favored")
    print(SH)
    for (m, a), c in sorted(cells.items()):
        d = c["dvec"][c["bi"]]
        order = np.argsort(-np.abs(d))
        top = "  ".join(f"e{e}({d[e]:+.2f})" for e in order[:4])
        print(f"  {m:22s} {a:9s}  L{c['bi']}  E={c['E']:3d}  PR={part_ratio(d):5.1f}"
              f"   top: {top}")

    print(f"\n{SH}")
    print(f"SUMMARY: {len(cells)} cells.  PART 1 top3/all = is refusal sparse; "
          f"PART 2 mean r = is the expert set attack-invariant.")
    print("  Both high -> a nameable refusal-expert set -> causal routing-steer experiment is on.")


if __name__ == "__main__":
    main()
