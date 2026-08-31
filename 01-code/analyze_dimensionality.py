#!/usr/bin/env python3
"""Participation ratio of the compliance and refusal states on the 6-model panel."""
import os

import numpy as np

ATT = ["PAP", "hill", "prefill", "roleplay"]
HOOK = {"deepseek-v2-lite": 26, "deepseek-moe-16b-chat": 27,
        "qwen1.5-moe-a2.7b-chat": 23, "llama-4-scout": 47,
        "mixtral-8x7b": 31, "olmoe-1b-7b": 15}
DATA = os.environ.get("CGEO_DATA", "./data")
MODELS = list(HOOK)
rng = np.random.default_rng(0)


def pr(A):
    A = np.asarray(A, np.float64)
    A = A - A.mean(0)
    n, d = A.shape
    G = (A @ A.T) if n <= d else (A.T @ A)
    ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    s = ev.sum()
    return float(s * s / (ev * ev).sum()) if s > 0 else 1.0


def pr_nc(X, n_sub, k=10):
    return float(np.mean([pr(X[rng.choice(len(X), n_sub, replace=False)])
                          for _ in range(k)]))


def layer_acts(m, a):
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
    d.close()
    return R, C


print("=" * 78)
print("v2 dimensionality, 6-model panel: absolute PR at hook layer, n-controlled")
print("=" * 78)
print(f"  {'model':22s} {'attack':9s} {'nsub':>6s} {'PR_C':>8s} {'PR_R':>8s}")
rows = {}
for m in MODELS:
    for a in ATT:
        la = layer_acts(m, a)
        if la is None:
            continue
        R, C = la
        ns = min(len(R), len(C))
        if ns < 25:
            continue
        ns = min(ns, 120)
        prc, prr = pr_nc(C, ns), pr_nc(R, ns)
        rows[(m, a)] = (prc, prr)
        print(f"  {m:22s} {a:9s} {ns:6d} {prc:8.1f} {prr:8.1f}", flush=True)

print("\nper-attack median (6-model panel):")
for a in ATT:
    cs = [v for (m, aa), v in rows.items() if aa == a]
    if cs:
        print(f"  {a:9s} PR_C {np.median([c[0] for c in cs]):6.1f}   "
              f"PR_R {np.median([c[1] for c in cs]):6.1f}   (n={len(cs)})")

pers = [v[0] for (m, a), v in rows.items() if a in ("PAP", "hill")]
pre = [v[0] for (m, a), v in rows.items() if a == "prefill"]
if pers and pre:
    print(f"\n  HEADLINE: persuasion (PAP+hill) median PR_C = {np.median(pers):.1f}, "
          f"prefill median PR_C = {np.median(pre):.1f}")
