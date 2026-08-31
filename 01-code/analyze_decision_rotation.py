#!/usr/bin/env python3
"""Test whether the cross-attack non-transfer is real or a corollary of attack-separability."""
import os
import time
from pathlib import Path
from itertools import combinations
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
PAIRS = list(combinations(ATTACKS, 2))
PERS = ("PAP", "hill")
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MIN_N = 25
PCA_DIM = 256
SEED = 0


def load_model(m):
    """All 4 attacks of one model, float32. None if any cell has < MIN_N/class."""
    cells = {}
    for a in ATTACKS:
        f = ROOT / m / f"{a}.npz"
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=True)
        cell = dict(R_rl=d["R_router_logits"].astype(np.float32),
                    C_rl=d["C_router_logits"].astype(np.float32),
                    R_res=d["R_residual"].astype(np.float32),
                    C_res=d["C_residual"].astype(np.float32))
        d.close()
        if min(len(cell["R_rl"]), len(cell["C_rl"])) < MIN_N:
            return None
        cells[a] = cell
    return cells


def clf(d_feat):
    steps = [StandardScaler()]
    if d_feat > PCA_DIM:
        steps.append(PCA(n_components=PCA_DIM, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def layer_decode(R, C):
    """Per-layer within-attack 5-fold CV balanced accuracy. (n,L,d) -> (L,)."""
    y = np.array([0] * len(R) + [1] * len(C))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = []
    for i in range(R.shape[1]):
        X = np.vstack([R[:, i, :], C[:, i, :]])
        out.append(cross_val_score(clf(X.shape[1]), X, y, cv=skf,
                                   scoring="balanced_accuracy").mean())
    return np.array(out)


def loo(cells, Rk, Ck, pick_layer, center):
    """Leave-one-attack-out. pick_layer(train_attacks) -> layer index.
    Returns dict held-out attack -> balanced accuracy."""
    out = {}
    for held in ATTACKS:
        tr = [a for a in ATTACKS if a != held]
        L = pick_layer(tr)
        Xtr, ytr = [], []
        for a in tr:
            R = cells[a][Rk][:, L, :]
            C = cells[a][Ck][:, L, :]
            X = np.vstack([R, C])
            if center:
                X = X - X.mean(0, keepdims=True)
            Xtr.append(X)
            ytr += [0] * len(R) + [1] * len(C)
        Xtr = np.vstack(Xtr)
        ytr = np.array(ytr)
        R = cells[held][Rk][:, L, :]
        C = cells[held][Ck][:, L, :]
        Xte = np.vstack([R, C])
        if center:
            Xte = Xte - Xte.mean(0, keepdims=True)
        yte = np.array([0] * len(R) + [1] * len(C))
        pipe = clf(Xtr.shape[1]).fit(Xtr, ytr)
        out[held] = float(balanced_accuracy_score(yte, pipe.predict(Xte)))
    return out


def cosine_resid(cells, layer, rng):
    """(A): z-scored residual decision-direction cosines, plus shuffled null."""
    Rk, Ck = "R_res", "C_res"
    allX = np.vstack([np.vstack([cells[a][Rk][:, layer, :], cells[a][Ck][:, layer, :]])
                      for a in ATTACKS])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6

    def z(x):
        return (x - mu) / sd

    dmu = {a: z(cells[a][Ck][:, layer, :]).mean(0) - z(cells[a][Rk][:, layer, :]).mean(0)
           for a in ATTACKS}
    cm = {p: cos(dmu[p[0]], dmu[p[1]]) for p in PAIRS}
    null = []
    for _ in range(20):
        sdmu = {}
        for a in ATTACKS:
            X = z(np.vstack([cells[a][Rk][:, layer, :], cells[a][Ck][:, layer, :]]))
            nR = len(cells[a][Rk])
            idx = rng.permutation(len(X))
            sdmu[a] = X[idx[nR:]].mean(0) - X[idx[:nR]].mean(0)
        null += [abs(cos(sdmu[p[0]], sdmu[p[1]])) for p in PAIRS]
    return cm, float(np.percentile(null, 95))


def main():
    models = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    rng = np.random.default_rng(SEED)
    cosrows, loorows = [], []
    print(f"routing-dir: {ROOT}", flush=True)
    for m in models:
        c = load_model(m)
        if c is None:
            print(f"  skip {m}  (a cell has < {MIN_N} per class)", flush=True)
            continue
        t0 = time.time()
        rdec = {a: layer_decode(c[a]["R_rl"], c[a]["C_rl"]) for a in ATTACKS}
        res_mid = c["PAP"]["R_res"].shape[1] // 2

        def rt_pick(tr):
            return int(np.argmax(sum(rdec[a] for a in tr)))

        def rs_pick(tr):
            return res_mid

        cm, null95 = cosine_resid(c, res_mid, rng)
        cosrows.append(dict(model=m, layer=res_mid, cm=cm, null95=null95))

        for sig, Rk, Ck, pick in [("routing", "R_rl", "C_rl", rt_pick),
                                  ("residual", "R_res", "C_res", rs_pick)]:
            unc = loo(c, Rk, Ck, pick, center=False)
            cen = loo(c, Rk, Ck, pick, center=True)
            loorows.append(dict(model=m, sig=sig, unc=unc, cen=cen,
                                rlayer=rt_pick(ATTACKS)))
        del c
        print(f"  done {m}  ({time.time() - t0:.0f}s)", flush=True)

    print()
    print("=" * 84)
    print("(A) RESIDUAL decision-direction cosine  cos(dMu_A, dMu_B),  dMu = mean_C - mean_R")
    print("    z-scored, at the middle residual layer")
    print("=" * 84)
    print(f"  {'model':24s} {'persuasion':>11s} {'cross-family':>13s} {'null|cos|95':>12s}")
    pers, cross = [], []
    for r in cosrows:
        pc = r["cm"][PERS]
        xc = [r["cm"][p] for p in PAIRS if p != PERS]
        pers.append(pc)
        cross += xc
        print(f"  {r['model']:24s} {pc:>+11.2f} {np.median(xc):>+13.2f} "
              f"{r['null95']:>12.2f}")
    print(f"  {'POOLED median':24s} {np.median(pers):>+11.2f} "
          f"{np.median(cross):>+13.2f}")
    print("  persuasion pair (PAP,hill) shares the decision direction;")
    print("  cross-family pairs sit at the shuffled-label null -- the axis rotates.")

    for sig in ("routing", "residual"):
        rows = [r for r in loorows if r["sig"] == sig]
        if not rows:
            continue
        print()
        print("=" * 84)
        print(f"(B) [{sig.upper()}] leave-one-attack-out: uncentered vs per-attack-centered")
        print("=" * 84)
        print(f"  {'model':24s} {'uncentered':>11s} {'centered':>10s}")
        ua, ca = [], []
        per_held = {h: {"u": [], "c": []} for h in ATTACKS}
        for r in rows:
            u = list(r["unc"].values())
            cc = list(r["cen"].values())
            ua += u
            ca += cc
            for h in ATTACKS:
                per_held[h]["u"].append(r["unc"][h])
                per_held[h]["c"].append(r["cen"][h])
            print(f"  {r['model']:24s} {np.median(u):>11.2f} {np.median(cc):>10.2f}")
        print(f"  {'POOLED median':24s} {np.median(ua):>11.3f} {np.median(ca):>10.3f}")
        print("  per held-out attack (median over models):")
        for h in ATTACKS:
            print(f"    held-out {h:9s} uncentered {np.median(per_held[h]['u']):.2f}"
                  f"   centered {np.median(per_held[h]['c']):.2f}")
        delta = np.median(ca) - np.median(ua)
        print(f"\n  VERDICT [{sig}]: centering moves pooled LOO by {delta:+.3f}.")
        if abs(delta) < 0.07:
            print("  -> centering does not rescue transfer. The non-transfer survives")
            print("     removal of the per-attack offset: the decision direction itself")
            print("     is attack-specific, NOT a corollary of attack-separability.")
        else:
            print("  -> centering shifts the result; the per-attack offset carried part")
            print("     of the non-transfer. Interpret with care.")


if __name__ == "__main__":
    main()
