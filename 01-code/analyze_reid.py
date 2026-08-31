#!/usr/bin/env python3
"""Lexical (TF-IDF) baseline for cross-attack base-request re-identification."""
import json, os
from itertools import combinations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ATT = ["PAP", "hill", "prefill", "roleplay"]
HOOK = {"deepseek-v2-lite": 26, "deepseek-moe-16b-chat": 27, "qwen1.5-moe-a2.7b-chat": 23,
        "llama-4-scout": 47, "mixtral-8x7b": 31, "olmoe-1b-7b": 15}
DATA = os.environ.get("CGEO_DATA", "./data")
MODELS = list(HOOK)


def reid(d1, d2):
    common = sorted(set(d1) & set(d2))
    if len(common) < 20:
        return None
    X1 = np.array([d1[b] for b in common], np.float64)
    X2 = np.array([d2[b] for b in common], np.float64)
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True) + 1e-9
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9
    S = X1 @ X2.T
    n = len(common)
    return n, float(np.mean((-S).argsort(1)[:, 0] == np.arange(n)))


def attack_center(bm):
    out = {}
    for a, dd in bm.items():
        gm = np.mean(list(dd.values()), 0)
        out[a] = {b: v - gm for b, v in dd.items()}
    return out


prompts = {}
for m in MODELS:
    for a in ATT:
        for o in ("refused", "complied"):
            p = f"{DATA}/dataset-permodel/{m}/{a}_{o}.jsonl"
            if not os.path.exists(p):
                continue
            for line in open(p):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    prompts[str(r["id"])] = r["prompt"]
ids = sorted(prompts)
vec = TfidfVectorizer(min_df=2, stop_words="english", ngram_range=(1, 2), max_features=8000)
T = vec.fit_transform([prompts[i] for i in ids])
row = {i: k for k, i in enumerate(ids)}
print(f"TF-IDF: {len(ids)} unique prompts, vocab {T.shape[1]}\n")

print("=" * 86)
print("EXP A firm-up -- cross-attack base re-ID: activation vs lexical (TF-IDF), attack-centered")
print("=" * 86)
print(f"  {'model':22s} {'pair':17s} {'n':>5s} {'act top1':>9s} {'lex top1':>9s} {'act-lex':>8s}")
gaps, acts, lexs = [], [], []
for m in MODELS:
    act_bm, lex_bm = {}, {}
    for a in ATT:
        f = f"{DATA}/cache/{m}/{a}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        li = [int(x) for x in d["layer_indices"]]
        if HOOK[m] not in li:
            d.close(); continue
        idx = li.index(HOOK[m])
        X = np.concatenate([d["R_acts"][:, idx, :], d["C_acts"][:, idx, :]], 0).astype(np.float64)
        qids = [str(x) for x in d["R_query_ids"]] + [str(x) for x in d["C_query_ids"]]
        d.close()
        accA, accL = {}, {}
        for x, i in zip(X, qids):
            b = i.split("__")[0]
            accA.setdefault(b, []).append(x)
            if i in row:
                accL.setdefault(b, []).append(T[row[i]].toarray()[0])
        act_bm[a] = {b: np.mean(v, 0) for b, v in accA.items()}
        lex_bm[a] = {b: np.mean(v, 0) for b, v in accL.items() if v}
    actC, lexC = attack_center(act_bm), attack_center(lex_bm)
    for a, b in combinations([a for a in ATT if a in act_bm], 2):
        ra = reid(actC[a], actC[b])
        rl = reid(lexC.get(a, {}), lexC.get(b, {}))
        if not ra or not rl:
            continue
        print(f"  {m:22s} {a+'/'+b:17s} {ra[0]:5d} {ra[1]:9.3f} {rl[1]:9.3f} {ra[1]-rl[1]:+8.3f}")
        gaps.append(ra[1] - rl[1]); acts.append(ra[1]); lexs.append(rl[1])

print("-" * 86)
print(f"  POOLED ({len(gaps)} cells):  median activation top1 {np.median(acts):.3f}   "
      f"median lexical top1 {np.median(lexs):.3f}")
print(f"  activation > lexical in {sum(g > 0 for g in gaps)}/{len(gaps)} cells   "
      f"median gap {np.median(gaps):+.3f}")
print("  read: activation below lexical => the residual is more attack-entangled than the prompt text")
