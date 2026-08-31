#!/usr/bin/env python3
"""AUROC (threshold-free) for the dense residual LOAO probe at fixed depth.

Answers reviewer R1-C5: a near-chance *balanced accuracy* could in principle come
from threshold miscalibration rather than a genuine direction mismatch. AUROC is
threshold-independent, so if AUROC is also ~0.5 at the same held-out cells, the
non-transfer is real (the direction does not separate), not a calibration artifact.

Reads the dense residual caches produced by extract_routing.py (dense mode) and
reports, at the paper's fixed depth-50% and depth-90% readouts, both the balanced
accuracy and the AUROC, pooled over the models in CGEO_MODELS and per held-out attack.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MODELS = [m.strip() for m in os.environ.get("CGEO_MODELS", "llama-3.1-8b,mistral-7b").split(",") if m.strip()]
SEED, N_CAP, PCA_DIM = 0, 200, 256


def clf(nc):
    steps = [StandardScaler()]
    if nc:
        steps.append(PCA(n_components=nc, random_state=SEED))
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
        nR, nC = len(d["R_residual"]), len(d["C_residual"])
        if min(nR, nC) < 25:
            continue
        iR = rng.choice(nR, min(nR, N_CAP), replace=False)
        iC = rng.choice(nC, min(nC, N_CAP), replace=False)
        res = np.concatenate([d["R_residual"][iR], d["C_residual"][iC]], 0).astype(np.float32)
        y = np.array([0] * len(iR) + [1] * len(iC))
        out[a] = (res, y)
    return out


def loao_fixed(data, layer):
    atts = list(data)
    L = data[atts[0]][0].shape[1]
    li = max(0, min(L - 1, layer))
    res = {}
    for held in atts:
        tr = [a for a in atts if a != held]
        Xtr = np.concatenate([data[a][0][:, li, :] for a in tr], 0)
        ytr = np.concatenate([data[a][1] for a in tr], 0)
        nc = min(PCA_DIM, Xtr.shape[1], max(1, Xtr.shape[0] - 1))
        p = clf(nc).fit(Xtr, ytr)
        Xte, yte = data[held][0][:, li, :], data[held][1]
        bacc = balanced_accuracy_score(yte, p.predict(Xte))
        auroc = roc_auc_score(yte, p.predict_proba(Xte)[:, 1])
        res[held] = (float(bacc), float(auroc))
    return res


def med(v):
    return float(np.median(v)) if v else float("nan")


print("=" * 78)
print("DENSE LOAO -- balanced accuracy vs AUROC (threshold-free) at fixed depth")
print(f"models: {MODELS}")
print("=" * 78)
for depth_name, frac in [("mid-50%", 0.5), ("late-90%", 0.9)]:
    bacc_all, auroc_all = [], []
    per = {a: ([], []) for a in ATTACKS}
    for m in MODELS:
        data = load(m)
        if len(data) < 4:
            print(f"  [skip] {m}: <4 attacks")
            continue
        L = data[list(data)[0]][0].shape[1]
        r = loao_fixed(data, round(frac * L))
        for a in ATTACKS:
            if a in r:
                bacc_all.append(r[a][0])
                auroc_all.append(r[a][1])
                per[a][0].append(r[a][0])
                per[a][1].append(r[a][1])
    print(f"\n[{depth_name}] pooled ({len(bacc_all)} cells):  "
          f"bal-acc median {med(bacc_all):.3f}   AUROC median {med(auroc_all):.3f}")
    for a in ATTACKS:
        if per[a][0]:
            print(f"    held-out {a:9s} bal-acc {med(per[a][0]):.3f}   AUROC {med(per[a][1]):.3f}")
