#!/usr/bin/env python3
"""Measure where, with network depth, the jailbroken state becomes attack-specific."""
import os
import time
from pathlib import Path
from itertools import combinations
import json
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
CROSS_HELD = ["prefill", "roleplay"]
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MIN_N = 25
PCA_DIM = 256
SEED = 0
N_GRID = 12


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


def pca_dim(n_train, n_feat):
    """Largest safe PCA dim for a fit on n_train rows (0 = skip PCA)."""
    if n_feat <= PCA_DIM:
        return 0
    return max(2, min(PCA_DIM, n_train - 5, n_feat))


def clf(n_pca):
    steps = [StandardScaler()]
    if n_pca:
        steps.append(PCA(n_components=n_pca, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def decode_at(R, C, L):
    X = np.vstack([R[:, L, :], C[:, L, :]])
    y = np.array([0] * len(R) + [1] * len(C))
    n_pca = pca_dim((len(y) * 4) // 5, X.shape[1])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(cross_val_score(clf(n_pca), X, y, cv=skf,
                                 scoring="balanced_accuracy").mean())


def loo_at(cells, Rk, Ck, L, held_set):
    accs = []
    for held in held_set:
        tr = [a for a in ATTACKS if a != held]
        Xtr, ytr = [], []
        for a in tr:
            R = cells[a][Rk][:, L, :]
            C = cells[a][Ck][:, L, :]
            Xtr.append(np.vstack([R, C]))
            ytr += [0] * len(R) + [1] * len(C)
        Xtr = np.vstack(Xtr)
        ytr = np.array(ytr)
        R = cells[held][Rk][:, L, :]
        C = cells[held][Ck][:, L, :]
        Xte = np.vstack([R, C])
        yte = np.array([0] * len(R) + [1] * len(C))
        n_pca = pca_dim(len(ytr), Xtr.shape[1])
        pipe = clf(n_pca).fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, pipe.predict(Xte)))
    return float(np.mean(accs))


def resid_cosine(cells, L, rng):
    """z-scored residual decision-direction cosines at layer L:
    (persuasion-pair cosine, cross-family median cosine, shuffled null95)."""
    Rk, Ck = "R_res", "C_res"
    allX = np.vstack([np.vstack([cells[a][Rk][:, L, :], cells[a][Ck][:, L, :]])
                      for a in ATTACKS])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6

    def z(x):
        return (x - mu) / sd

    dmu = {a: z(cells[a][Ck][:, L, :]).mean(0) - z(cells[a][Rk][:, L, :]).mean(0)
           for a in ATTACKS}
    cm = {p: cos(dmu[p[0]], dmu[p[1]]) for p in PAIRS}
    crossv = float(np.median([cm[p] for p in PAIRS if p != PERS]))
    null = []
    for _ in range(20):
        sdmu = {}
        for a in ATTACKS:
            X = z(np.vstack([cells[a][Rk][:, L, :], cells[a][Ck][:, L, :]]))
            nR = len(cells[a][Rk])
            idx = rng.permutation(len(X))
            sdmu[a] = X[idx[nR:]].mean(0) - X[idx[:nR]].mean(0)
        null += [abs(cos(sdmu[p[0]], sdmu[p[1]])) for p in PAIRS]
    return cm[PERS], crossv, float(np.percentile(null, 95))


def thirds(vals):
    n = len(vals)
    e, m = n // 3, 2 * n // 3
    return (float(np.mean(vals[:e])), float(np.mean(vals[e:m])),
            float(np.mean(vals[m:])))


def main():
    models = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    rng = np.random.default_rng(SEED)
    rt_rows, rs_rows, cos_rows = [], [], []
    print(f"routing-dir: {ROOT}", flush=True)

    for m in models:
        c = load_model(m)
        if c is None:
            print(f"  skip {m}  (< {MIN_N}/class)", flush=True)
            continue
        t0 = time.time()

        Lrt = c["PAP"]["R_rl"].shape[1]
        dec, allv, cr = [], [], []
        for L in range(Lrt):
            dec.append(np.mean([decode_at(c[a]["R_rl"], c[a]["C_rl"], L)
                                for a in ATTACKS]))
            allv.append(loo_at(c, "R_rl", "C_rl", L, ATTACKS))
            cr.append(loo_at(c, "R_rl", "C_rl", L, CROSS_HELD))
        rt_rows.append(dict(model=m, n_layers=Lrt, dec=dec, allv=allv, cr=cr))

        Lrs = c["PAP"]["R_res"].shape[1]
        grid = sorted(set(np.linspace(0, Lrs - 1, N_GRID).round().astype(int)))
        dec, allv, cr, pe, cc, n95v = [], [], [], [], [], []
        for L in grid:
            dec.append(np.mean([decode_at(c[a]["R_res"], c[a]["C_res"], L)
                                for a in ATTACKS]))
            allv.append(loo_at(c, "R_res", "C_res", L, ATTACKS))
            cr.append(loo_at(c, "R_res", "C_res", L, CROSS_HELD))
            p, x, n95 = resid_cosine(c, L, rng)
            pe.append(p)
            cc.append(x)
            n95v.append(n95)
        rs_rows.append(dict(model=m, n_layers=Lrs, grid=[int(g) for g in grid],
                            dec=dec, allv=allv, cr=cr, pe=pe, cc=cc, n95=n95v))

        bi = int(np.argmax(dec))
        p, x, n95 = resid_cosine(c, grid[bi], rng)
        cos_rows.append(dict(model=m, layer=int(grid[bi]), dec=dec[bi],
                             pers=p, cross=x, null95=n95))
        del c
        print(f"  done {m}  ({time.time() - t0:.0f}s)", flush=True)

    json.dump({"routing": rt_rows, "residual": rs_rows, "decode_best": cos_rows},
              open("/tmp/layerwise_data.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("saved /tmp/layerwise_data.json", flush=True)

    def report(rows, name, has_cos):
        print()
        print("=" * 88)
        print(f"[{name}]  experiment 2: decode and transfer vs depth (early/mid/late thirds)")
        print("=" * 88)
        cols = f"  {'model':22s} {'third':6s} {'decode':>8s} {'LOO-all':>9s} {'LOO-cross':>10s}"
        if has_cos:
            cols += f" {'pers-cos':>9s} {'cross-cos':>10s}"
        print(cols)
        agg = {t: {k: [] for k in ("dec", "all", "cr", "pe", "cc")}
               for t in ("early", "mid", "late")}
        for r in rows:
            de, al, crr = thirds(r["dec"]), thirds(r["allv"]), thirds(r["cr"])
            pet = thirds(r["pe"]) if has_cos else (0, 0, 0)
            cct = thirds(r["cc"]) if has_cos else (0, 0, 0)
            for i, t in enumerate(("early", "mid", "late")):
                line = (f"  {r['model']:22s} {t:6s} {de[i]:>8.2f} "
                        f"{al[i]:>9.2f} {crr[i]:>10.2f}")
                if has_cos:
                    line += f" {pet[i]:>+9.2f} {cct[i]:>+10.2f}"
                print(line)
                agg[t]["dec"].append(de[i])
                agg[t]["all"].append(al[i])
                agg[t]["cr"].append(crr[i])
                agg[t]["pe"].append(pet[i])
                agg[t]["cc"].append(cct[i])
        print("  " + "-" * 60)
        for t in ("early", "mid", "late"):
            line = (f"  {'POOLED median':22s} {t:6s} "
                    f"{np.median(agg[t]['dec']):>8.2f} "
                    f"{np.median(agg[t]['all']):>9.2f} "
                    f"{np.median(agg[t]['cr']):>10.2f}")
            if has_cos:
                line += (f" {np.median(agg[t]['pe']):>+9.2f} "
                         f"{np.median(agg[t]['cc']):>+10.2f}")
            print(line)

    report(rt_rows, "ROUTING", has_cos=False)
    report(rs_rows, "RESIDUAL", has_cos=True)

    print()
    print("=" * 88)
    print("Experiment 1(A) re-run: residual decision-direction cosine at the")
    print("decode-best residual layer (principled layer, per model)")
    print("=" * 88)
    print(f"  {'model':22s} {'layer':>6s} {'decode':>8s} {'persuasion':>11s} "
          f"{'cross-family':>13s} {'null|cos|95':>12s}")
    for r in cos_rows:
        print(f"  {r['model']:22s} {r['layer']:>6d} {r['dec']:>8.2f} "
              f"{r['pers']:>+11.2f} {r['cross']:>+13.2f} {r['null95']:>12.2f}")
    print(f"  {'POOLED median':22s} {'':>6s} {'':>8s} "
          f"{np.median([r['pers'] for r in cos_rows]):>+11.2f} "
          f"{np.median([r['cross'] for r in cos_rows]):>+13.2f}")


if __name__ == "__main__":
    main()
