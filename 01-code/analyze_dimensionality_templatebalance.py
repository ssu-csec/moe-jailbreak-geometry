#!/usr/bin/env python3
"""Template-balance robustness check for roleplay PR_C.

PAP and hill carry four templates each; roleplay nominally has eighty but
only a subset of them produce compliance in any given model.  This script
recomputes PR_C for roleplay after restricting the complied set to its
top-four most-frequent templates, holding the subsample size constant
between the original and restricted runs so the dimensionality bound is
comparable.  A roleplay PR_C that stays low under restriction would support
the response-shape reading; one that rises would point to template diversity
as the dominant driver.
"""
import os

import numpy as np

ATT = ["PAP", "hill", "prefill", "roleplay"]
HOOK = {"deepseek-v2-lite": 26, "deepseek-moe-16b-chat": 27,
        "qwen1.5-moe-a2.7b-chat": 23, "llama-4-scout": 47,
        "mixtral-8x7b": 31, "olmoe-1b-7b": 15}
DATA = os.environ.get("CGEO_DATA", "./data")
MODELS = list(HOOK)
SEED = 0
K_KEEP = 4   # match PAP/hill template count


def pr(A):
    A = np.asarray(A, np.float64)
    A = A - A.mean(0)
    n, d = A.shape
    G = (A @ A.T) if n <= d else (A.T @ A)
    ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    s = ev.sum()
    return float(s * s / (ev * ev).sum()) if s > 0 else 1.0


def pr_nc(X, n_sub, k=10, rng=None):
    rng = rng or np.random.default_rng(SEED)
    return float(np.mean([pr(X[rng.choice(len(X), n_sub, replace=False)])
                          for _ in range(k)]))


def load_layer(m, a):
    f = f"{DATA}/cache/{m}/{a}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    li = [int(x) for x in d["layer_indices"]]
    if HOOK[m] not in li:
        d.close()
        return None
    i = li.index(HOOK[m])
    R = d["R_acts"][:, i, :].astype(np.float64)
    C = d["C_acts"][:, i, :].astype(np.float64)
    qR = d["R_query_ids"][:]
    qC = d["C_query_ids"][:]
    d.close()
    return R, C, qR, qC


def template_of(qid):
    return qid.split("__")[-1]


def restrict_to_top_k(X, qids, k):
    """Keep rows whose template is among the top-k most frequent in qids."""
    templates = np.array([template_of(q) for q in qids])
    counts = {}
    for t in templates:
        counts[t] = counts.get(t, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    keep_set = {t for t, _ in top}
    mask = np.array([t in keep_set for t in templates])
    return X[mask], qids[mask], top


print("=" * 96)
print(f"Template-balance check: roleplay PR_C after restricting to top-{K_KEEP} templates")
print(f"(top-{K_KEEP} chosen by per-model complied-count; same n_sub between original and restricted runs)")
print("=" * 96)
print(f"  {'model':22s} {'attack':9s} {'orig_n':>7s} {'restr_n':>7s} {'n_sub':>6s} "
      f"{'PR_C_orig':>9s} {'PR_C_restr':>10s} {'delta':>7s}")
rng = np.random.default_rng(SEED)
agg = []

for m in MODELS:
    for a in ATT:
        la = load_layer(m, a)
        if la is None:
            continue
        R, C, qR, qC = la
        if len(C) < 25 or len(R) < 25:
            continue
        if a == "roleplay":
            C_restr, qC_restr, top = restrict_to_top_k(C, qC, K_KEEP)
            R_restr, _, _ = restrict_to_top_k(R, qR, K_KEEP)
            if len(C_restr) < 25:
                continue
            n_sub = min(len(C), len(C_restr), 120)
            prc_orig = pr_nc(C, n_sub, rng=np.random.default_rng(SEED))
            prc_restr = pr_nc(C_restr, n_sub, rng=np.random.default_rng(SEED))
            print(f"  {m:22s} {a:9s} {len(C):7d} {len(C_restr):7d} {n_sub:6d} "
                  f"{prc_orig:9.1f} {prc_restr:10.1f} {prc_restr - prc_orig:+7.1f}",
                  flush=True)
            agg.append((m, prc_orig, prc_restr, [t for t, _ in top]))
        else:
            # Persuasion / prefill baselines: original PR_C only, cap at 120.
            n_sub = min(len(C), 120)
            prc = pr_nc(C, n_sub, rng=np.random.default_rng(SEED))
            print(f"  {m:22s} {a:9s} {len(C):7d} {'-':>7s} {n_sub:6d} "
                  f"{prc:9.1f} {'-':>10s} {'-':>7s}",
                  flush=True)

print()
print("-" * 96)
print(f"roleplay summary (median across {len(agg)} models):")
orig = [x[1] for x in agg]
restr = [x[2] for x in agg]
print(f"  PR_C original (top-{K_KEEP} subsample budget) : median {np.median(orig):.1f}, "
      f"range [{min(orig):.1f}, {max(orig):.1f}]")
print(f"  PR_C restricted to top-{K_KEEP} templates     : median {np.median(restr):.1f}, "
      f"range [{min(restr):.1f}, {max(restr):.1f}]")
print(f"  delta median: {np.median(np.array(restr) - np.array(orig)):+.1f}")
print()
print("retained templates per model:")
for m, _, _, top in agg:
    print(f"  {m:22s} {top}")
