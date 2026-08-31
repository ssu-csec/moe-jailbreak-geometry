#!/usr/bin/env python3
"""Within-family (cross-template) transfer, from the dense residual caches.

Answers Reviewer 3: how does transfer scale with attack similarity? We split an
attack's templates into two halves, train a comply-vs-refuse probe on one half's
templates, and test on the other half (symmetric, averaged both directions). This
is a within-family / cross-template transfer, the strongest similarity level short
of within-attack. Contrast it with the cross-FAMILY leave-one-attack-out (~0.55).

Env: CGEO_ROUTING, CGEO_MODELS (comma sep), WF_ATTACK (default roleplay).
Reads <CGEO_ROUTING>/<model>/<attack>.npz written by extract_routing.py.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
MODELS = [m.strip() for m in os.environ.get("CGEO_MODELS", "llama-3.1-8b,mistral-7b").split(",") if m.strip()]
ATTACK = os.environ.get("WF_ATTACK", "roleplay")
SEED, PCA_DIM = 0, 256


def clf(nc):
    steps = [StandardScaler()]
    if nc:
        steps.append(PCA(n_components=nc, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def tmpl_of(qid):
    parts = str(qid).split("__")
    if ATTACK in parts:
        i = parts.index(ATTACK)
        return parts[i + 1] if i + 1 < len(parts) else parts[-1]
    return parts[-1]


def load(model):
    d = np.load(ROOT / model / f"{ATTACK}.npz", allow_pickle=True)
    return (d["R_residual"].astype(np.float32), d["C_residual"].astype(np.float32),
            [str(x) for x in d["R_query_ids"]], [str(x) for x in d["C_query_ids"]])


def within_family(model, layer):
    R, C, Rid, Cid = load(model)
    tmpls = sorted(set(tmpl_of(i) for i in Rid + Cid))
    A, B = set(tmpls[0::2]), set(tmpls[1::2])

    def sel(grp):
        ri = [i for i, q in enumerate(Rid) if tmpl_of(q) in grp]
        ci = [i for i, q in enumerate(Cid) if tmpl_of(q) in grp]
        X = np.concatenate([R[ri][:, layer, :], C[ci][:, layer, :]], 0)
        y = np.array([0] * len(ri) + [1] * len(ci))
        return X, y

    Xa, ya = sel(A)
    Xb, yb = sel(B)
    if min((ya == 0).sum(), (ya == 1).sum(), (yb == 0).sum(), (yb == 1).sum()) < 20:
        return None
    res = []
    for Xtr, ytr, Xte, yte in [(Xa, ya, Xb, yb), (Xb, yb, Xa, ya)]:
        nc = min(PCA_DIM, Xtr.shape[1], max(1, Xtr.shape[0] - 1))
        p = clf(nc).fit(Xtr, ytr)
        res.append((balanced_accuracy_score(yte, p.predict(Xte)),
                    roc_auc_score(yte, p.predict_proba(Xte)[:, 1])))
    bacc = np.mean([r[0] for r in res])
    auroc = np.mean([r[1] for r in res])
    return bacc, auroc, len(tmpls), len(ya), len(yb)


print(f"WITHIN-FAMILY (cross-template) transfer on {ATTACK}: "
      f"train on half the templates, test on the other half (symmetric mean)")
print("=" * 78)
for depth_name, frac in [("mid-50%", 0.5), ("late-90%", 0.9)]:
    baccs, aurocs = [], []
    for m in MODELS:
        R, _, _, _ = load(m)
        layer = round(frac * R.shape[1])
        out = within_family(m, layer)
        if out is None:
            print(f"  [{depth_name}] {m}: too few per class, skipped")
            continue
        b, a, nt, na, nb = out
        baccs.append(b)
        aurocs.append(a)
        print(f"  [{depth_name}] {m:16s} bal-acc {b:.3f}  AUROC {a:.3f}  "
              f"({nt} templates; {na} vs {nb} prompts)")
    if baccs:
        print(f"  [{depth_name}] POOLED median: bal-acc {np.median(baccs):.3f}  "
              f"AUROC {np.median(aurocs):.3f}")
