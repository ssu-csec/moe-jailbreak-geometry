#!/usr/bin/env python3
"""Attack separability: 4-way linear classifier on per-model activations.

For each model, pool every attack's complied activations into one set and train
a 5-fold cross-validated multiclass logistic regression to identify which of
the four jailbreak attacks each prompt belongs to (chance = 0.25).
Do the same on the refused activations. The resulting per-model accuracies are
the numbers reported in the Attack Separability appendix table.

Reads residual-stream caches; writes per-model results to stdout.
"""
import json
import os
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path(os.environ.get("CGEO_DATA", "./data"))
CACHE_DIR = DATA_ROOT / "cache"
GATE_DIR = DATA_ROOT / "gate11_out"
ATTACKS = ["PAP", "hill", "prefill", "roleplay"]
MODELS = ["deepseek-v2-lite", "deepseek-moe-16b-chat",
          "qwen1.5-moe-a2.7b-chat", "llama-4-scout",
          "mixtral-8x7b", "olmoe-1b-7b"]
SEED = 0


def hook_layer(model):
    """Per-model readout layer from gate9_expert_{model}.json."""
    p = GATE_DIR / f"gate9_expert_{model}.json"
    return int(json.load(open(p)).get("hook_layer", -1)) if p.exists() else None


def logreg_cv_acc(X, y, folds=5):
    """Cross-validated multiclass logistic-regression accuracy."""
    n = len(y)
    if n < folds * 2 or len(set(y)) < 2:
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.1, random_state=SEED))
    try:
        sc = cross_val_score(clf, X, y, cv=min(folds, n // 4), scoring="accuracy")
        return float(sc.mean())
    except Exception:
        return None


def main():
    print(f"{'model':23s}{'complied':>12s}{'refused':>12s}")
    print("-" * 47)
    c_all, r_all = [], []
    for model in MODELS:
        L = hook_layer(model)
        if L is None:
            continue
        Xc_list, yc_list, Xr_list, yr_list = [], [], [], []
        for ai, atk in enumerate(ATTACKS):
            path = CACHE_DIR / model / f"{atk}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            R = d["R_acts"][:, L, :].astype(np.float32)
            C = d["C_acts"][:, L, :].astype(np.float32)
            if len(R) < 30 or len(C) < 30:
                continue
            Xc_list.append(C); yc_list += [ai] * len(C)
            Xr_list.append(R); yr_list += [ai] * len(R)
        if len(set(yc_list)) < 2:
            continue
        Xc = np.concatenate(Xc_list); yc = np.array(yc_list)
        Xr = np.concatenate(Xr_list); yr = np.array(yr_list)
        cacc = logreg_cv_acc(Xc, yc)
        racc = logreg_cv_acc(Xr, yr)
        if cacc is not None:
            c_all.append(cacc)
        if racc is not None:
            r_all.append(racc)
        cs = f"{cacc:.3f}" if cacc else "    -"
        rs = f"{racc:.3f}" if racc else "    -"
        print(f"{model:23s}{cs:>12s}{rs:>12s}")
    print("-" * 47)
    print(f"{'median':23s}{statistics.median(c_all):>12.3f}{statistics.median(r_all):>12.3f}")
    print(f"\n(chance = 0.25; higher = activations encode attack identity)")


if __name__ == "__main__":
    main()
