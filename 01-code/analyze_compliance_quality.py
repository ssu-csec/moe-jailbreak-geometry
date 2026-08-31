#!/usr/bin/env python3
"""Test whether the attack sets the quality of compliance, measured by judge disagreement."""
import json
import os
from collections import defaultdict

import numpy as np

ATT = ["PAP", "hill", "prefill", "roleplay"]
HOOK = {"deepseek-v2-lite": 26, "deepseek-moe-16b-chat": 27, "qwen1.5-moe-a2.7b-chat": 23,
        "llama-4-scout": 47, "mixtral-8x7b": 31, "olmoe-1b-7b": 15, "gpt-oss-20b": 23,
        "phi-3.5-moe-instruct": 31}
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


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def judge_counts(m, a):
    """cross-tab of (hb outcome, gpt4o outcome), read from record FIELDS."""
    cc = defaultdict(int)
    n_rec = n_both = 0
    for split in ("refused", "complied"):
        p = f"{DATA}/dataset-permodel/{m}/{a}_{split}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            n_rec += 1
            r = json.loads(line)
            hb, g4 = r.get("outcome"), r.get("gpt4o_outcome")
            if hb in ("refused", "complied") and g4 in ("refused", "complied"):
                cc[(hb, g4)] += 1
                n_both += 1
    return cc, n_rec, n_both


def pr_ratio(m, a):
    f = f"{DATA}/cache/{m}/{a}.npz"
    if not os.path.exists(f):
        return float("nan")
    d = np.load(f, allow_pickle=True)
    li = [int(x) for x in d["layer_indices"]]
    if HOOK[m] not in li:
        d.close()
        return float("nan")
    i = li.index(HOOK[m])
    R = d["R_acts"][:, i, :].astype(np.float64)
    C = d["C_acts"][:, i, :].astype(np.float64)
    d.close()
    ns = min(len(R), len(C))
    if ns < 25:
        return float("nan")
    ns = min(ns, 120)
    return pr_nc(C, ns) / pr_nc(R, ns)


print("=" * 100)
print("COMPLIANCE QUALITY  --  judge disagreement (no degeneracy heuristic) vs v2 dimensionality")
print("=" * 100)
print(f"  {'model':22s} {'attack':9s} {'n_both':>7s} {'cov':>6s} {'disagree':>9s} "
      f"{'soft-comply':>12s} {'RC:CR':>11s} {'PR ratio':>9s}")

rows = {}
for m in MODELS:
    for a in ATT:
        cc, n_rec, n = judge_counts(m, a)
        if n < 40:
            print(f"  {m:22s} {a:9s} {n:7d}  -- judge labels absent or too few, skipped")
            continue
        rr, rc = cc[("refused", "refused")], cc[("refused", "complied")]
        cr, ccx = cc[("complied", "refused")], cc[("complied", "complied")]
        disagree = (rc + cr) / n
        soft = rc / (rc + ccx) if (rc + ccx) else float("nan")
        prr = pr_ratio(m, a)
        rows[(m, a)] = dict(n=n, disagree=disagree, soft=soft, rc=rc, cr=cr, prr=prr)
        print(f"  {m:22s} {a:9s} {n:7d} {n / n_rec:6.2f} {disagree:9.3f} "
              f"{soft:12.3f} {str(rc) + ':' + str(cr):>11s} {prr:9.2f}", flush=True)

SH = "=" * 100
print(f"\n{SH}\nPART 1 -- per-attack medians  (does the attack set compliance quality?)")
print(SH)
print(f"  {'attack':9s} {'disagree':>9s} {'soft-comply':>12s} {'PR ratio':>9s}   n")
for a in ATT:
    cs = [v for (m, aa), v in rows.items() if aa == a]
    if not cs:
        continue
    print(f"  {a:9s} {np.median([c['disagree'] for c in cs]):9.3f} "
          f"{np.median([c['soft'] for c in cs]):12.3f} "
          f"{np.nanmedian([c['prr'] for c in cs]):9.2f}   {len(cs)}")
print("  (soft-comply high + PR ratio low for the same attack => quality axis)")

print(f"\n{SH}\nPART 2 -- link across all cells: low PR ratio <-> soft (judge-ambiguous) compliance?")
print(SH)
xs = [(v['prr'], v['disagree'], v['soft']) for v in rows.values()
      if v['prr'] == v['prr']]
if len(xs) > 5:
    p, d, s = zip(*xs)
    print(f"  Spearman(PR ratio, soft-comply)  = {spear(p, s):+.3f}   (n={len(xs)} cells)")
    print(f"  Spearman(PR ratio, disagreement) = {spear(p, d):+.3f}")
    print("  negative => the low-dimensional complied cloud is the judge-ambiguous one")

print(f"\n{SH}\nPART 3 -- within-model attack ranking (controls for model)")
print("  Spearman of the 4 attacks' PR-ratio rank vs soft-comply rank, per model")
print(SH)
agr = []
for m in MODELS:
    cs = [(a, rows[(m, a)]) for a in ATT
          if (m, a) in rows and rows[(m, a)]['prr'] == rows[(m, a)]['prr']
          and rows[(m, a)]['soft'] == rows[(m, a)]['soft']]
    if len(cs) < 3:
        continue
    r = spear([c[1]['prr'] for c in cs], [c[1]['soft'] for c in cs])
    agr.append(r)
    print(f"  {m:22s} attacks={[c[0] for c in cs]}  Spearman(PR,soft)={r:+.2f}")
if agr:
    print(f"  median within-model Spearman = {np.median(agr):+.2f}  "
          f"({sum(x < 0 for x in agr)}/{len(agr)} models negative)")

print(f"\n{SH}")
print("READ: if prefill/roleplay show low PR ratio AND high soft-comply while PAP/hill")
print("show the opposite, the attack sets compliance QUALITY -- a degeneracy-free claim.")
