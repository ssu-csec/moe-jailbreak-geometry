#!/usr/bin/env python3
"""Extract 2D t-SNE coordinates from MoE router logits for Figure 1 (teaser).

Reads per-(model, attack) router-logit caches and writes one JSON per requested
(model, method) combination to 02-data/. The teaser figure (Figure 1) reads from
router_tsne_llama.json by default; the same pipeline can be re-run for other
models by uncommenting the corresponding calls in main().
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
ROUTING_ROOT = Path(os.environ.get("CGEO_ROUTING", "./data/cache_moe_routing"))
OUT_DIR = REPO_ROOT / "02-data"
ATTACKS = ["PAP", "hill", "prefill", "roleplay"]


def run(model, method, out_path):
    cache = ROUTING_ROOT / model
    X = []
    labels_attack = []
    for atk in ATTACKS:
        d = np.load(cache / f"{atk}.npz")
        R = d["R_router_logits"].astype(np.float32)
        C = d["C_router_logits"].astype(np.float32)
        R_flat = R.reshape(R.shape[0], -1)
        C_flat = C.reshape(C.shape[0], -1)
        X.append(R_flat); labels_attack += [atk] * len(R_flat)
        X.append(C_flat); labels_attack += [atk] * len(C_flat)
    X = np.vstack(X)
    print(f"\n{model} / {method}: combined {X.shape}")

    if method == "PCA":
        Xc = X - X.mean(0)
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(Xc)
        evr = [float(v) for v in pca.explained_variance_ratio_]
        method_meta = "multi-layer router concat + PCA"
    elif method == "tSNE":
        Xc = X - X.mean(0)
        if Xc.shape[1] > 50:
            pca = PCA(n_components=50, random_state=0)
            Xc = pca.fit_transform(Xc)
        tsne = TSNE(n_components=2, perplexity=30, random_state=0,
                    init="pca", learning_rate="auto", max_iter=1500)
        Z = tsne.fit_transform(Xc)
        evr = None
        method_meta = "multi-layer router concat + PCA50 + t-SNE"
    else:
        raise ValueError(method)

    out = {atk: [] for atk in ATTACKS}
    for i, atk in enumerate(labels_attack):
        out[atk].append([float(Z[i, 0]), float(Z[i, 1])])

    payload = {
        "model": model,
        "substrate": "router_logits",
        "method": method_meta,
        "n_layers": int(R.shape[1]),
        "n_experts": int(R.shape[2]),
        "n_components": 2,
        "explained_variance_ratio": evr,
        "attacks": ATTACKS,
        "points": out,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"saved {out_path}")
    for a in ATTACKS:
        print(f"  {a}: n={len(out[a])}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run("llama-4-scout", "tSNE", str(OUT_DIR / "router_tsne_llama.json"))
    # additional models, uncomment as needed:
    # run("deepseek-v2-lite",         "tSNE", str(OUT_DIR / "router_tsne_deepseek-v2-lite.json"))
    # run("deepseek-moe-16b-chat",    "tSNE", str(OUT_DIR / "router_tsne_deepseek-moe-16b-chat.json"))
    # run("qwen1.5-moe-a2.7b-chat",   "tSNE", str(OUT_DIR / "router_tsne_qwen1.5-moe-a2.7b-chat.json"))
    # run("mixtral-8x7b",             "tSNE", str(OUT_DIR / "router_tsne_mixtral-8x7b.json"))
    # run("olmoe-1b-7b",              "tSNE", str(OUT_DIR / "router_tsne_olmoe-1b-7b.json"))


if __name__ == "__main__":
    main()
