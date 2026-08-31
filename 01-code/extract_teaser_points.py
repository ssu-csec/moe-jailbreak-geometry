#!/usr/bin/env python3
"""Extract the routing points behind Figure 1 (the teaser)."""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing")
MODEL = "llama-4-scout"
ATTACK_A = "roleplay"
ATTACK_B = "hill"
SEED = 0
N_CAP = 200

rng = np.random.default_rng(SEED)


def load(attack):
    d = np.load(f"{ROOT}/{MODEL}/{attack}.npz", allow_pickle=True)
    R = np.asarray(d["R_router_logits"], np.float32)
    C = np.asarray(d["C_router_logits"], np.float32)
    d.close()
    iR = rng.choice(len(R), min(len(R), N_CAP), replace=False)
    iC = rng.choice(len(C), min(len(C), N_CAP), replace=False)
    X = np.concatenate([R[iR], C[iC]], 0)
    y = np.array([0] * len(iR) + [1] * len(iC))
    return X, y


def probe():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def cv_acc(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), probe())
    return float(cross_val_score(pipe, X, y, cv=skf,
                                 scoring="balanced_accuracy").mean())


XA, yA = load(ATTACK_A)
XB, yB = load(ATTACK_B)
L = XA.shape[1]

best, layer = -1.0, 0
for li in range(L):
    m = 0.5 * (cv_acc(XA[:, li, :], yA) + cv_acc(XB[:, li, :], yB))
    if m > best:
        best, layer = m, li

gA, gB = XA[:, layer, :], XB[:, layer, :]

scA = StandardScaler().fit(gA)
lrA = probe().fit(scA.transform(gA), yA)
scB = StandardScaler().fit(gB)
lrB = probe().fit(scB.transform(gB), yB)


def axes(g):
    return (lrA.decision_function(scA.transform(g)),
            lrB.decision_function(scB.transform(g)))


aA, aB = axes(gA)
bA, bB = axes(gB)

out = {
    "model": MODEL, "attack_a": ATTACK_A, "attack_b": ATTACK_B, "layer": int(layer),
    "decode": {
        "within_a": round(cv_acc(gA, yA), 3),
        "within_b": round(cv_acc(gB, yB), 3),
        "a_probe_on_b": round(float(balanced_accuracy_score(
            yB, lrA.predict(scA.transform(gB)))), 3),
        "b_probe_on_a": round(float(balanced_accuracy_score(
            yA, lrB.predict(scB.transform(gA)))), 3),
    },
    "panel_a": [[round(float(x), 3), round(float(y), 3), int(l)]
                for x, y, l in zip(aA, aB, yA)],
    "panel_b": [[round(float(x), 3), round(float(y), 3), int(l)]
                for x, y, l in zip(bA, bB, yB)],
}

sys.stderr.write(f"model={MODEL}  a={ATTACK_A}  b={ATTACK_B}  layer={layer}\n")
sys.stderr.write(f"decode={out['decode']}  "
                 f"n_a={len(out['panel_a'])}  n_b={len(out['panel_b'])}\n")
json.dump(out, sys.stdout)
