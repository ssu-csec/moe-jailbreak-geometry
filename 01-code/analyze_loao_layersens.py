#!/usr/bin/env python3
"""Layer-selection sensitivity for the leave-one-attack-out probe.

The default LOAO chooses the readout layer by maximizing mean within-attack
decode across the three training attacks.  Here we contrast that choice with
two attack-agnostic fallbacks: a fixed mid-depth layer (50%) and a fixed
late-depth layer (90%).  A defender who does not trust the layer-selection
heuristic could use either.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MODELS = ["deepseek-moe-16b-chat", "deepseek-v2-lite", "llama-4-scout",
          "mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b-chat"]
SEED = 0
N_CAP = 200
PCA_DIM = 256


def clf(pca=False):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=PCA_DIM, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
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
    L = data[attacks[0]][0].shape[1]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, bl = -1.0, 0
    for li in range(L):
        accs = [cross_val_score(clf(), data[a][0][:, li, :], data[a][2], cv=skf,
                                scoring="balanced_accuracy").mean()
                for a in attacks]
        if np.mean(accs) > best:
            best, bl = float(np.mean(accs)), li
    return bl


def loao_at_layer(data, key_idx, pca, layer_fn):
    """Held-out attack -> balanced accuracy at the layer chosen by layer_fn(train_attacks, L)."""
    atts = list(data)
    out = {}
    for held in atts:
        train = [a for a in atts if a != held]
        L = data[train[0]][key_idx].shape[1]
        bl = layer_fn(data, train, L)
        Xtr = np.concatenate([data[a][key_idx][:, bl, :] for a in train], 0)
        ytr = np.concatenate([data[a][2] for a in train], 0)
        pipe = clf(pca).fit(Xtr, ytr)
        out[held] = float(balanced_accuracy_score(
            data[held][2], pipe.predict(data[held][key_idx][:, bl, :])))
    return out


def boot_ci(vals, k=4000):
    rng = np.random.default_rng(SEED)
    meds = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(k)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def layer_train_best(data, train, L):
    return best_layer({a: data[a] for a in train}, train)


def layer_mid(data, train, L):
    return int(round(0.5 * (L - 1)))


def layer_late(data, train, L):
    return int(round(0.9 * (L - 1)))


CHOICES = [("train-best", layer_train_best),
           ("mid (0.50)", layer_mid),
           ("late (0.90)", layer_late)]


def main():
    print("=" * 92)
    print("LAYER-SELECTION SENSITIVITY for the leave-one-attack-out probe")
    print("rows: pooled median balanced accuracy across 24 held-out cells (95% CI)")
    print("=" * 92)
    rt_all = {n: [] for n, _ in CHOICES}
    rs_all = {n: [] for n, _ in CHOICES}
    rt_per = {n: {a: [] for a in ATTACKS} for n, _ in CHOICES}
    rs_per = {n: {a: [] for a in ATTACKS} for n, _ in CHOICES}

    for m in MODELS:
        data = load(m)
        if len(data) < 4:
            print(f"  {m:22s}  fewer than 4 attacks, skipped")
            continue
        print(f"  {m}")
        for name, fn in CHOICES:
            rt = loao_at_layer(data, 0, pca=False, layer_fn=fn)
            rs = loao_at_layer(data, 1, pca=True, layer_fn=fn)
            for a in ATTACKS:
                if a in rt:
                    rt_all[name].append(rt[a])
                    rs_all[name].append(rs[a])
                    rt_per[name][a].append(rt[a])
                    rs_per[name][a].append(rs[a])
            print(f"    {name:12s} routing " +
                  "  ".join(f"{a}={rt[a]:.2f}" for a in ATTACKS if a in rt))
            print(f"    {name:12s} residual " +
                  "  ".join(f"{a}={rs[a]:.2f}" for a in ATTACKS if a in rs))

    print()
    print("-" * 92)
    print(f"  {'layer choice':14s}  {'routing pooled':>22s}  {'residual pooled':>22s}")
    for name, _ in CHOICES:
        r, s = rt_all[name], rs_all[name]
        rc, sc = boot_ci(r), boot_ci(s)
        print(f"  {name:14s}  {np.median(r):.3f} [{rc[0]:.2f},{rc[1]:.2f}]  "
              f"            {np.median(s):.3f} [{sc[0]:.2f},{sc[1]:.2f}]")

    print()
    print("per held-out attack (routing, median across models):")
    print(f"  {'layer choice':14s} " + " ".join(f"{a:>10s}" for a in ATTACKS))
    for name, _ in CHOICES:
        line = f"  {name:14s} " + " ".join(
            f"{np.median(rt_per[name][a]) if rt_per[name][a] else float('nan'):>10.2f}"
            for a in ATTACKS)
        print(line)
    print()
    print("per held-out attack (residual, median across models):")
    print(f"  {'layer choice':14s} " + " ".join(f"{a:>10s}" for a in ATTACKS))
    for name, _ in CHOICES:
        line = f"  {name:14s} " + " ".join(
            f"{np.median(rs_per[name][a]) if rs_per[name][a] else float('nan'):>10.2f}"
            for a in ATTACKS)
        print(line)


if __name__ == "__main__":
    main()
