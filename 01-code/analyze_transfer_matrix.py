#!/usr/bin/env python3
"""Cross-attack transfer matrix for the routing probe."""
import os
from pathlib import Path

import numpy as np
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


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


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
        iR = rng.choice(nR, min(nR, N_CAP), replace=False)
        iC = rng.choice(nC, min(nC, N_CAP), replace=False)
        rl = np.concatenate([d["R_router_logits"][iR], d["C_router_logits"][iC]], 0).astype(np.float32)
        y = np.array([0] * len(iR) + [1] * len(iC))
        out[a] = (rl, y)
        d.close()
    return out


def best_layer(data):
    L = next(iter(data.values()))[0].shape[1]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, bl = -1.0, 0
    for li in range(L):
        accs = [cross_val_score(clf(), rl[:, li, :], y, cv=skf,
                                scoring="balanced_accuracy").mean()
                for rl, y in data.values()]
        if np.mean(accs) > best:
            best, bl = float(np.mean(accs)), li
    return bl


mats = []
for m in MODELS:
    data = load(m)
    if len(data) < 4:
        continue
    bl = best_layer(data)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    M = np.full((4, 4), np.nan)
    for i, ai in enumerate(ATTACKS):
        Xi, yi = data[ai][0][:, bl, :], data[ai][1]
        for j, aj in enumerate(ATTACKS):
            if i == j:
                M[i, j] = cross_val_score(clf(), Xi, yi, cv=skf,
                                          scoring="balanced_accuracy").mean()
            else:
                pipe = clf().fit(Xi, yi)
                M[i, j] = balanced_accuracy_score(
                    data[aj][1], pipe.predict(data[aj][0][:, bl, :]))
    mats.append(M)
    print(f"  {m:24s} layer {bl}", flush=True)

med = np.nanmedian(np.stack(mats), axis=0)
print("\ncross-attack routing transfer (rows = train attack, cols = test attack),")
print("median over the six models, chance 0.50:")
print("            " + "".join(f"{a:>10s}" for a in ATTACKS))
for i, a in enumerate(ATTACKS):
    print(f"  {a:9s} " + "".join(f"{med[i, j]:10.2f}" for j in range(4)))
print("\ndiagonal = within-attack; off-diagonal = single-attack transfer")
