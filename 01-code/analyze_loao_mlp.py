#!/usr/bin/env python3
"""Leave-one-attack-out transfer with a small MLP probe (non-linear).

Mirror of analyze_loao.py with a 2-layer MLP head replacing the logistic head.
Tests whether a modestly non-linear probe finds the cross-attack signature
that a linear probe misses.  Cohen's d, layer selection, and bootstrap CIs
follow the linear pipeline.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MODELS = ["deepseek-moe-16b-chat", "deepseek-v2-lite", "llama-4-scout",
          "mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b-chat"]
SEED = 0
N_CAP = 200
PCA_DIM = 256
HIDDEN = 64
ALPHA = 1e-3       # L2 in MLPClassifier
MAX_ITER = 400


def mlp_clf(pca=False):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=PCA_DIM, random_state=SEED))
    steps.append(MLPClassifier(
        hidden_layer_sizes=(HIDDEN,),
        activation="relu",
        alpha=ALPHA,
        max_iter=MAX_ITER,
        random_state=SEED,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.15,
    ))
    return make_pipeline(*steps)


def load(model):
    rng = np.random.default_rng(SEED)
    out = {}
    for a in ATTACKS:
        f = ROOT / model / f"{a}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if "R_router_logits" not in d.files:
            d.close()
            continue
        nR, nC = len(d["R_router_logits"]), len(d["C_router_logits"])
        if min(nR, nC) < 25:
            d.close()
            continue
        iR = rng.choice(nR, min(nR, N_CAP), replace=False)
        iC = rng.choice(nC, min(nC, N_CAP), replace=False)
        rl = np.concatenate([d["R_router_logits"][iR], d["C_router_logits"][iC]], 0).astype(np.float32)
        res = np.concatenate([d["R_residual"][iR], d["C_residual"][iC]], 0).astype(np.float32)
        y = np.array([0] * len(iR) + [1] * len(iC))
        out[a] = (rl, res, y)
        d.close()
    return out


def best_layer(data, attacks):
    """Routing layer maximizing mean within-attack MLP decode over the given attacks.

    Matches analyze_loao.py: layer is chosen on the routing signal only,
    then the same layer index is used for both routing and residual probes.
    """
    L = data[attacks[0]][0].shape[1]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, bl = -1.0, 0
    for li in range(L):
        accs = []
        for a in attacks:
            X = data[a][0][:, li, :]
            y = data[a][2]
            fold = []
            for tr_i, te_i in skf.split(X, y):
                pipe = mlp_clf(pca=False).fit(X[tr_i], y[tr_i])
                fold.append(balanced_accuracy_score(y[te_i], pipe.predict(X[te_i])))
            accs.append(float(np.mean(fold)))
        if np.mean(accs) > best:
            best, bl = float(np.mean(accs)), li
    return bl


def loao(data, key_idx, pca):
    atts = list(data)
    out = {}
    for held in atts:
        train = [a for a in atts if a != held]
        bl = best_layer(data, train)
        Xtr = np.concatenate([data[a][key_idx][:, bl, :] for a in train], 0)
        ytr = np.concatenate([data[a][2] for a in train], 0)
        pipe = mlp_clf(pca).fit(Xtr, ytr)
        out[held] = float(balanced_accuracy_score(
            data[held][2], pipe.predict(data[held][key_idx][:, bl, :])))
    return out


def within_attack_routing(data):
    """Cross-validated within-attack MLP accuracy on router logits.

    Routing only: PCA is off, so the per-attack 5-fold CV (4/5 of one
    attack's samples) never runs into n_components > min(n_samples, n_feat).
    """
    atts = list(data)
    out = {}
    for a in atts:
        L = data[a][0].shape[1]
        best = -1.0
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for li in range(L):
            X = data[a][0][:, li, :]
            y = data[a][2]
            fold = []
            for tr_i, te_i in skf.split(X, y):
                pipe = mlp_clf(pca=False).fit(X[tr_i], y[tr_i])
                fold.append(balanced_accuracy_score(y[te_i], pipe.predict(X[te_i])))
            if np.mean(fold) > best:
                best = float(np.mean(fold))
        out[a] = best
    return out


def boot_ci(vals, k=4000):
    rng = np.random.default_rng(SEED)
    meds = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(k)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


print("=" * 88)
print("LEAVE-ONE-ATTACK-OUT transfer, MLP probe (hidden=%d, alpha=%.0e)" % (HIDDEN, ALPHA))
print("=" * 88)

rt_cells, rs_cells, wt_cells = {}, {}, {}
for m in MODELS:
    data = load(m)
    if len(data) < 4:
        print(f"  {m:22s}  fewer than 4 attacks, skipped", flush=True)
        continue
    rt = loao(data, 0, pca=False)
    rs = loao(data, 1, pca=True)
    wt = within_attack_routing(data)
    for a in ATTACKS:
        if a in rt:
            rt_cells[(m, a)] = rt[a]
            rs_cells[(m, a)] = rs[a]
            wt_cells[(m, a)] = wt[a]
    print(f"  {m}", flush=True)
    print(f"    within   routing  " + "  ".join(f"{a}={wt[a]:.2f}" for a in ATTACKS if a in wt), flush=True)
    print(f"    LOAO     routing  " + "  ".join(f"{a}={rt[a]:.2f}" for a in ATTACKS if a in rt), flush=True)
    print(f"    LOAO     residual " + "  ".join(f"{a}={rs[a]:.2f}" for a in ATTACKS if a in rs), flush=True)

print("-" * 88)
rt_all, rs_all, wt_all = list(rt_cells.values()), list(rs_cells.values()), list(wt_cells.values())
rt_ci, rs_ci, wt_ci = boot_ci(rt_all), boot_ci(rs_all), boot_ci(wt_all)
print(f"POOLED MLP leave-one-attack-out balanced accuracy ({len(rt_all)} held-out cells, chance 0.50):")
print(f"  within  routing : median {np.median(wt_all):.3f}  95% CI [{wt_ci[0]:.3f}, {wt_ci[1]:.3f}]")
print(f"  LOAO    routing : median {np.median(rt_all):.3f}  95% CI [{rt_ci[0]:.3f}, {rt_ci[1]:.3f}]")
print(f"  LOAO    residual: median {np.median(rs_all):.3f}  95% CI [{rs_ci[0]:.3f}, {rs_ci[1]:.3f}]")

print()
print("per held-out attack, MLP LOAO (median over models, 95% CI):")
for a in ATTACKS:
    rv = [rt_cells[(m, a)] for m in MODELS if (m, a) in rt_cells]
    sv = [rs_cells[(m, a)] for m in MODELS if (m, a) in rs_cells]
    if rv:
        rc, sc = boot_ci(rv), boot_ci(sv)
        print(f"  held-out {a:9s} routing {np.median(rv):.3f} [{rc[0]:.2f},{rc[1]:.2f}]   "
              f"residual {np.median(sv):.3f} [{sc[0]:.2f},{sc[1]:.2f}]")
