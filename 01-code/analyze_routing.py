#!/usr/bin/env python3
"""Analyze whether the MoE routing decision carries the refused/complied signal, and where."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("needs scikit-learn:  pip install scikit-learn")

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
N_CAP = 200
PCA_DIM = 256
BIG_LAYERS = 3
MIN_N = 12
SEED = 0


def clf(n_pca):
    steps = [StandardScaler()]
    if n_pca:
        steps.append(PCA(n_components=n_pca, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def pca_dim(n_train, n_feat):
    """Largest safe PCA dim for a fit on n_train rows (0 = skip PCA)."""
    if n_feat <= PCA_DIM:
        return 0
    return max(2, min(PCA_DIM, n_train - 5, n_feat))


def cv_bacc(X, y, pca=False):
    X = np.asarray(X, np.float32)
    n_pca = pca_dim((len(y) * 4) // 5, X.shape[1]) if pca else 0
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(cross_val_score(clf(n_pca), X, y, cv=skf,
                                 scoring="balanced_accuracy").mean())


def shuffle_null(X, y, n_perm=15):
    rng = np.random.default_rng(SEED)
    return float(np.percentile([cv_bacc(X, rng.permutation(y))
                                for _ in range(n_perm)], 95))


def multihot(sel, E):
    n, L, k = sel.shape
    mh = np.zeros((n, L, E), np.float32)
    np.put_along_axis(mh, np.clip(sel, 0, E - 1).astype(np.int64), 1.0, axis=2)
    return mh


def depth_layers(L, n):
    return sorted(set(int(round(x)) for x in np.linspace(0, L - 1, n + 2)[1:-1]))


def analyze_cell(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    nR, nC = len(d["R_router_logits"]), len(d["C_router_logits"])
    rng = np.random.default_rng(SEED)
    iR = rng.choice(nR, min(nR, N_CAP), replace=False)
    iC = rng.choice(nC, min(nC, N_CAP), replace=False)
    y = np.array([0] * len(iR) + [1] * len(iC))
    o = {"nR": nR, "nC": nC, "y": y, "ok": min(nR, nC) >= MIN_N}

    def grab(name):
        if f"R_{name}" not in keys:
            return None
        return np.concatenate([d[f"R_{name}"][iR].astype(np.float32),
                               d[f"C_{name}"][iC].astype(np.float32)], 0)

    if not o["ok"]:
        d.close()
        return o

    rl = grab("router_logits")
    E, L = rl.shape[2], rl.shape[1]
    o["router_per_layer"] = [cv_bacc(rl[:, i, :], y) for i in range(L)]
    bi = int(np.argmax(o["router_per_layer"]))
    o["router_best"] = (o["router_per_layer"][bi], bi)
    o["router_shuf95"] = shuffle_null(rl[:, bi, :], y)

    sel = grab("selected_experts")
    mh = multihot(sel.astype(np.int64), E) if sel is not None else None
    if mh is not None:
        o["selexp_per_layer"] = [cv_bacc(mh[:, i, :], y) for i in range(L)]
        si = int(np.argmax(o["selexp_per_layer"]))
        o["selexp_best"] = (o["selexp_per_layer"][si], si)
        fR, fC = mh[y == 0].mean(0), mh[y == 1].mean(0)
        o["sel_tv"] = (0.5 * np.abs(fR - fC).sum(1)).tolist()
        pk = int(np.argmax(o["sel_tv"]))
        delta = fR[pk] - fC[pk]
        top = np.argsort(-np.abs(delta))[:5]
        o["sel_peak"] = (pk, o["sel_tv"][pk],
                         [(int(e), float(delta[e])) for e in top])

    for name in ("routed_out", "shared_out", "residual"):
        sig = grab(name)
        if sig is None:
            o[name] = None
            continue
        layers = depth_layers(sig.shape[1], BIG_LAYERS)
        scores = [(cv_bacc(sig[:, i, :], y, pca=True), i) for i in layers]
        o[name] = max(scores)
        if name == "residual":
            o["tf_resid"] = sig[:, sig.shape[1] // 2, :].astype(np.float16)

    o["rl_full"] = rl.astype(np.float16)
    if mh is not None:
        o["mh_full"] = mh.astype(np.float16)
    d.close()
    return o


def _slice(c, key, layer):
    v = c[key]
    return np.asarray(v[:, layer, :] if v.ndim == 3 else v, np.float32)


def transfer(cells, model, key, layer, pca):
    """Train probe on attack A, test on B, both sliced at a fixed layer."""
    present = [a for a in ATTACKS
               if (model, a) in cells and key in cells[(model, a)]]
    out = {}
    for atr in present:
        ca = cells[(model, atr)]
        Xtr, ytr = _slice(ca, key, layer), ca["y"]
        n_pca = pca_dim(len(ytr), Xtr.shape[1]) if pca else 0
        pipe = clf(n_pca).fit(Xtr, ytr)
        for ate in present:
            if ate == atr:
                continue
            cb = cells[(model, ate)]
            out[(atr, ate)] = float(balanced_accuracy_score(
                cb["y"], pipe.predict(_slice(cb, key, layer))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-dir", default=os.environ.get(
        "CGEO_ROUTING", "./data/cache_moe_routing"))
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    root = Path(args.routing_dir)
    if not root.exists():
        sys.exit(f"routing dir not found: {root}")
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())

    cells = {}
    print(f"routing-dir: {root}", flush=True)
    for m in models:
        for a in ATTACKS:
            f = root / m / f"{a}.npz"
            if not f.exists():
                continue
            t0 = time.time()
            cells[(m, a)] = analyze_cell(f)
            c = cells[(m, a)]
            print(f"  [{time.time() - t0:5.0f}s] {m:24s} {a:9s} "
                  f"nR/nC={c['nR']}/{c['nC']}"
                  f"{'' if c['ok'] else '  (too few -> skipped)'}", flush=True)
    if not cells:
        sys.exit("no {model}/{attack}.npz found")

    SH = "=" * 100
    print(f"\n{SH}\nPART 1 -- SIGNAL COMPARISON: refused-vs-complied probe (5-fold balanced-acc)")
    print("  routerL = gate logits | selExp = top-k expert choice | routed/shared = pathway out")
    print("  residual = full stream (reference ceiling).  d-model signals probed on top-256 PCs.")
    print(SH)
    print(f"  {'model':22s} {'attack':9s} {'routerL':>9s} {'selExp':>8s} "
          f"{'routedO':>8s} {'sharedO':>8s} {'residual':>9s} {'shuf95':>7s}  verdict")
    for (m, a), c in sorted(cells.items()):
        if not c["ok"]:
            print(f"  {m:22s} {a:9s}   skipped")
            continue
        rb = c["router_best"][0]
        sb = c["selexp_best"][0] if "selexp_best" in c else float("nan")
        ro = c["routed_out"][0] if c.get("routed_out") else float("nan")
        so = c["shared_out"][0] if c.get("shared_out") else float("nan")
        re = c["residual"][0] if c.get("residual") else float("nan")
        route = max(rb, sb if sb == sb else 0.0)
        if route >= 0.75 and route > c["router_shuf95"] and route >= re - 0.10:
            v = "GO routing"
        elif route >= 0.65 and route > c["router_shuf95"]:
            v = "partial"
        elif (ro == ro and ro >= 0.75) or (so == so and so >= 0.75):
            v = "NO-GO (in expert-compute)"
        else:
            v = "NO-GO"
        cn = lambda x: f"{x:>8.3f}" if x == x else f"{'-':>8s}"
        print(f"  {m:22s} {a:9s} {rb:9.3f} {cn(sb)} {cn(ro)} {cn(so)} "
              f"{re:9.3f} {c['router_shuf95']:7.2f}  {v}")
    print("  verdict: GO = routing decodes >=0.75, beats shuffle-95, within 0.10 of residual")

    print(f"\n{SH}\nPART 2 -- LAYER LOCUS of the routing-logit signal (per-layer balanced-acc)")
    print(SH)
    for (m, a), c in sorted(cells.items()):
        if not c["ok"]:
            continue
        pl = c["router_per_layer"]
        L = len(pl)
        e, mid = np.mean(pl[:L // 3]), np.mean(pl[L // 3:2 * L // 3])
        late = np.mean(pl[2 * L // 3:])
        top = sorted(range(L), key=lambda i: -pl[i])[:3]
        ts = ", ".join(f"L{i}={pl[i]:.2f}" for i in top)
        print(f"  {m:22s} {a:9s}  early/mid/late {e:.2f}/{mid:.2f}/{late:.2f}"
              f"   peak: {ts}")

    print(f"\n{SH}\nPART 3 -- EXPERT SELECTION: do refused prompts pick different experts?")
    print("  TV = total-variation distance between refused/complied expert-selection"
          " frequency (0 = identical, 1 = disjoint)")
    print(SH)
    for (m, a), c in sorted(cells.items()):
        if not c["ok"] or "sel_peak" not in c:
            continue
        pk, tv, top = c["sel_peak"]
        ts = ", ".join(f"e{e}({dl:+.2f})" for e, dl in top)
        print(f"  {m:22s} {a:9s}  peak TV {tv:.3f} at L{pk}   "
              f"refused-vs-complied freq delta: {ts}")

    print(f"\n{SH}\nPART 4 -- CROSS-ATTACK TRANSFER: train probe on attack A, test on attack B")
    print("  routing-logit vs residual-stream, at each model's globally-best routing layer")
    print(SH)
    for m in models:
        present = [a for a in ATTACKS if (m, a) in cells and cells[(m, a)]["ok"]]
        if len(present) < 2:
            continue
        rlay = int(np.argmax(np.mean(
            [cells[(m, a)]["router_per_layer"] for a in present], axis=0)))
        tr = transfer(cells, m, "rl_full", rlay, pca=False)
        tre = transfer(cells, m, "tf_resid", 0, pca=True)
        if not tr:
            continue
        off_r, off_e = list(tr.values()), list(tre.values())
        print(f"  {m}   (routing layer L{rlay})")
        for atr in present:
            row = "  ".join(f"{atr}>{ate}:{tr[(atr, ate)]:.2f}"
                            for ate in present if ate != atr)
            print(f"    routing   {row}")
        print(f"    routing off-diagonal median {np.median(off_r):.3f}   "
              f"residual {np.median(off_e):.3f}   "
              f"gap {np.median(off_r) - np.median(off_e):+.3f}")

    print(f"\n{SH}")
    probed = [c for c in cells.values() if c["ok"]]
    print(f"SUMMARY: {len(probed)} cells probed of {len(cells)}.  "
          f"PART 1 verdict column is the go/no-go; PART 4 transfer gap decides the paper.")


if __name__ == "__main__":
    main()
