#!/usr/bin/env python3
"""Leave-one-attack-out transfer for the routing and residual-stream probes.

Default (MoE) mode reproduces the paper's numbers exactly. Setting CGEO_DENSE=1
switches to the dense-baseline mode used for the rebuttal: the npz files carry
only the residual stream (no router logits), so only the residual leave-one-
attack-out probe is run, and its readout layer is chosen from the residual
signal itself (there is no routing signal to select from). Everything else --
the L2 logistic probe, 256-component PCA, StandardScaler, per-fold layer choice
from the training attacks only, and the 4000-resample bootstrap CI -- is
identical to the MoE residual path, so the dense pooled median is directly
comparable to the paper's residual 0.62 [0.51, 0.73].

Env:
  CGEO_ROUTING  cache dir (default ./data/cache_moe_routing)
  CGEO_DENSE    "1" to run dense (residual-only) mode
  CGEO_MODELS   comma-separated model list override
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
DENSE = os.environ.get("CGEO_DENSE", "0").lower() not in ("0", "", "false", "no")

_MODELS_ENV = os.environ.get("CGEO_MODELS", "").strip()
if _MODELS_ENV:
    MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()]
elif DENSE:
    MODELS = ["mistral-7b", "llama-3.1-8b"]
else:
    MODELS = ["deepseek-moe-16b-chat", "deepseek-v2-lite", "llama-4-scout",
              "mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b-chat"]
SEED = 0
N_CAP = 200
PCA_DIM = 256


def make_clf(pca_components=None):
    steps = [StandardScaler()]
    if pca_components:
        steps.append(PCA(n_components=pca_components, random_state=SEED))
    steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def load(model, dense=False):
    """model -> {attack: (routing_or_None, residual, y)}."""
    rng = np.random.default_rng(SEED)
    out = {}
    for a in ATTACKS:
        f = ROOT / model / f"{a}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        has_routing = "R_router_logits" in d.files
        if not dense and not has_routing:
            d.close()
            continue
        nR, nC = len(d["R_residual"]), len(d["C_residual"])
        if min(nR, nC) < 25:
            d.close()
            continue
        iR = rng.choice(nR, min(nR, N_CAP), replace=False)
        iC = rng.choice(nC, min(nC, N_CAP), replace=False)
        res = np.concatenate([d["R_residual"][iR], d["C_residual"][iC]], 0).astype(np.float32)
        y = np.array([0] * len(iR) + [1] * len(iC))
        if has_routing:
            rl = np.concatenate([d["R_router_logits"][iR], d["C_router_logits"][iC]], 0).astype(np.float32)
        else:
            rl = None
        out[a] = (rl, res, y)
        d.close()
    return out


def best_layer(data, attacks, select_key_idx):
    """Layer maximizing mean within-attack decode over `attacks`, on `select_key_idx`.

    select_key_idx=0 selects on the routing signal (MoE, matching the paper);
    select_key_idx=1 selects on the residual signal (dense, no routing available).
    """
    L = data[attacks[0]][select_key_idx].shape[1]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, bl = -1.0, 0
    for li in range(L):
        accs = [cross_val_score(make_clf(), data[a][select_key_idx][:, li, :], data[a][2],
                                cv=skf, scoring="balanced_accuracy").mean()
                for a in attacks]
        if np.mean(accs) > best:
            best, bl = float(np.mean(accs)), li
    return bl


def loao(data, key_idx, pca, select_key_idx):
    """Held-out attack -> balanced accuracy. Layer chosen from training attacks only."""
    atts = list(data)
    out = {}
    for held in atts:
        train = [a for a in atts if a != held]
        bl = best_layer(data, train, select_key_idx)
        Xtr = np.concatenate([data[a][key_idx][:, bl, :] for a in train], 0)
        ytr = np.concatenate([data[a][2] for a in train], 0)
        pca_nc = None
        if pca:
            # guard against thin dense cells: n_components <= min(n_samples-1, n_features)
            pca_nc = min(PCA_DIM, Xtr.shape[1], max(1, Xtr.shape[0] - 1))
        pipe = make_clf(pca_nc).fit(Xtr, ytr)
        out[held] = float(balanced_accuracy_score(
            data[held][2], pipe.predict(data[held][key_idx][:, bl, :])))
    return out


def loao_fixed(data, key_idx, pca, layer):
    """Leave-one-attack-out at a FIXED layer (no per-fold selection).

    Used for the dense-vs-MoE comparison at the paper's fixed depth-50% / depth-90%
    readouts, which removes the layer-selection asymmetry (dense has no routing
    signal to select the readout layer from, unlike the MoE residual probe)."""
    atts = list(data)
    L = data[atts[0]][key_idx].shape[1]
    li = max(0, min(L - 1, layer))
    out = {}
    for held in atts:
        train = [a for a in atts if a != held]
        Xtr = np.concatenate([data[a][key_idx][:, li, :] for a in train], 0)
        ytr = np.concatenate([data[a][2] for a in train], 0)
        pca_nc = min(PCA_DIM, Xtr.shape[1], max(1, Xtr.shape[0] - 1)) if pca else None
        pipe = make_clf(pca_nc).fit(Xtr, ytr)
        out[held] = float(balanced_accuracy_score(
            data[held][2], pipe.predict(data[held][key_idx][:, li, :])))
    return out


def boot_ci(vals, k=4000):
    rng = np.random.default_rng(SEED)
    meds = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(k)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


print("=" * 86)
mode = "DENSE (residual-only)" if DENSE else "MoE (routing + residual)"
print(f"LEAVE-ONE-ATTACK-OUT transfer [{mode}] (layer chosen from training attacks only)")
print("=" * 86)
rt_cells, rs_cells = {}, {}
rs_mid_cells, rs_late_cells = {}, {}
for m in MODELS:
    data = load(m, dense=DENSE)
    if len(data) < 4:
        print(f"  {m:22s}  fewer than 4 attacks, skipped")
        continue
    # Residual probe: dense selects its layer from the residual itself; MoE from routing.
    rs = loao(data, 1, pca=True, select_key_idx=(1 if DENSE else 0))
    for a in ATTACKS:
        if a in rs:
            rs_cells[(m, a)] = rs[a]
    if DENSE:
        # Also report fixed depth-50%/90% -- removes the layer-selection asymmetry,
        # giving a clean apples-to-apples comparison with the paper's MoE fixed-depth
        # residual LOAO (mid 0.585, late 0.667).
        L = data[list(data)[0]][1].shape[1]
        mid_l, late_l = round(0.5 * L), round(0.9 * L)
        rs_mid = loao_fixed(data, 1, pca=True, layer=mid_l)
        rs_late = loao_fixed(data, 1, pca=True, layer=late_l)
        for a in ATTACKS:
            if a in rs_mid:
                rs_mid_cells[(m, a)] = rs_mid[a]
            if a in rs_late:
                rs_late_cells[(m, a)] = rs_late[a]
        print(f"  {m:22s}  (L={L}, mid={mid_l}, late={late_l})")
        print(f"    train-best " + "  ".join(f"{a}={rs[a]:.2f}" for a in ATTACKS if a in rs))
        print(f"    mid-50%    " + "  ".join(f"{a}={rs_mid[a]:.2f}" for a in ATTACKS if a in rs_mid))
        print(f"    late-90%   " + "  ".join(f"{a}={rs_late[a]:.2f}" for a in ATTACKS if a in rs_late))
    else:
        rt = loao(data, 0, pca=False, select_key_idx=0)
        for a in ATTACKS:
            if a in rt:
                rt_cells[(m, a)] = rt[a]
        print(f"  {m:22s}")
        print(f"    routing  " + "  ".join(f"{a}={rt[a]:.2f}" for a in ATTACKS if a in rt))
        print(f"    residual " + "  ".join(f"{a}={rs[a]:.2f}" for a in ATTACKS if a in rs))

print("-" * 86)
rt_all, rs_all = list(rt_cells.values()), list(rs_cells.values())
if not rs_all:
    print("[warn] no residual cells collected -- nothing to pool. "
          "Check that the npz caches exist and cleared the min-count gate.")
    raise SystemExit(0)
rs_ci = boot_ci(rs_all)
print(f"POOLED leave-one-attack-out balanced accuracy ({len(rs_all)} held-out cells, chance 0.50):")
if rt_all:
    rt_ci = boot_ci(rt_all)
    print(f"  routing : median {np.median(rt_all):.3f}  95% CI [{rt_ci[0]:.3f}, {rt_ci[1]:.3f}]  "
          f"range [{min(rt_all):.2f}, {max(rt_all):.2f}]")
if DENSE:
    for label, cells in [("residual train-best", rs_cells),
                         ("residual mid-50%   ", rs_mid_cells),
                         ("residual late-90%  ", rs_late_cells)]:
        vals = list(cells.values())
        if not vals:
            continue
        ci = boot_ci(vals)
        print(f"  {label}: median {np.median(vals):.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"range [{min(vals):.2f}, {max(vals):.2f}]")
else:
    print(f"  residual: median {np.median(rs_all):.3f}  95% CI [{rs_ci[0]:.3f}, {rs_ci[1]:.3f}]  "
          f"range [{min(rs_all):.2f}, {max(rs_all):.2f}]")
print()
print("per held-out attack (median over models, 95% CI):")
for a in ATTACKS:
    sv = [rs_cells[(m, a)] for m in MODELS if (m, a) in rs_cells]
    if not sv:
        continue
    sc = boot_ci(sv)
    if rt_all:
        rv = [rt_cells[(m, a)] for m in MODELS if (m, a) in rt_cells]
        rc = boot_ci(rv)
        print(f"  held-out {a:9s} routing {np.median(rv):.3f} [{rc[0]:.2f},{rc[1]:.2f}]   "
              f"residual {np.median(sv):.3f} [{sc[0]:.2f},{sc[1]:.2f}]")
    else:
        print(f"  held-out {a:9s} residual {np.median(sv):.3f} [{sc[0]:.2f},{sc[1]:.2f}]")
