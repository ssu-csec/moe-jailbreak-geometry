#!/usr/bin/env python3
"""Compute participation ratio (PR) per layer slice for cached activations.

Read-only over /root/compliance-cohesion/data. Writes JSON to /tmp.
"""
import json
import os
import sys

import numpy as np

from pathlib import Path

DATA_ROOT = Path(os.environ.get("CGEO_DATA", "./data"))
CACHE_DIR = str(DATA_ROOT / "cache")
REPO_ROOT = Path(os.environ.get("CGEO_ROOT", Path(__file__).resolve().parent.parent))
OUT_PATH = str(REPO_ROOT / "02-data" / "pr_layers.json")

# Per-model readout (hook) layer used in the paper's main analyses.
# Values match those documented in Methodology and Table 1.
HOOK_LAYER = {
    "deepseek-v2-lite": 26,
    "deepseek-moe-16b-chat": 27,
    "qwen1.5-moe-a2.7b-chat": 23,
    "llama-4-scout": 47,
    "mixtral-8x7b": 31,
    "olmoe-1b-7b": 15,
    "gpt-oss-20b": 23,
    "phi-3.5-moe-instruct": 31,
}

MODELS = [
    "deepseek-v2-lite",
    "llama-4-scout",
    "deepseek-moe-16b-chat",
    "qwen1.5-moe-a2.7b-chat",
    "olmoe-1b-7b",
    "gpt-oss-20b",
    "mixtral-8x7b",
    "phi-3.5-moe-instruct",
]
ATTACKS = ["PAP", "hill", "prefill", "roleplay"]

# Cells the task explicitly says are absent / not usable.
SKIP = {
    ("gpt-oss-20b", "prefill"),
    ("gpt-oss-20b", "roleplay"),
    ("phi-3.5-moe-instruct", "prefill"),
}


def participation_ratio(A):
    """PR of a matrix A of shape (m, d) via the m x m Gram matrix."""
    A = np.asarray(A, dtype=np.float64)
    m = A.shape[0]
    A = A - A.mean(axis=0)
    G = A @ A.T / m
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 1e-10]
    s = ev.sum()
    s2 = (ev ** 2).sum()
    if s2 <= 0.0:
        return float("nan")
    return float((s ** 2) / s2)


def main():
    cells = {}
    notes = []

    for model in MODELS:
        for attack in ATTACKS:
            key = f"{model}|{attack}"
            path = os.path.join(CACHE_DIR, model, f"{attack}.npz")

            if (model, attack) in SKIP:
                if os.path.exists(path):
                    notes.append(f"SKIP (per task spec) {key}: file present but excluded")
                else:
                    notes.append(f"SKIP (per task spec) {key}: file absent")
                continue

            if not os.path.exists(path):
                notes.append(f"MISSING unexpectedly {key}: {path}")
                continue

            d = np.load(path, allow_pickle=True)
            C = d["C_acts"]
            R = d["R_acts"]
            layer_indices = np.asarray(d["layer_indices"]).astype(int)
            n_C = int(C.shape[0])
            n_R = int(R.shape[0])
            L = int(C.shape[1])

            if L != len(layer_indices) or R.shape[1] != L:
                notes.append(
                    f"WARN {key}: layer mismatch C.L={L} R.L={R.shape[1]} "
                    f"len(layer_indices)={len(layer_indices)}"
                )
            if n_C < 3 or n_R < 3:
                notes.append(f"WARN {key}: tiny sample n_C={n_C} n_R={n_R}")

            entries = []
            for li in range(L):
                entries.append({
                    "layer": int(layer_indices[li]),
                    "PR_C": participation_ratio(C[:, li, :]),
                    "PR_R": participation_ratio(R[:, li, :]),
                    "n_C": n_C,
                    "n_R": n_R,
                })
            cells[key] = entries
            print(f"done {key}: L={L} n_C={n_C} n_R={n_R}", flush=True)

    hook_layer = {m: HOOK_LAYER[m] for m in MODELS if m in HOOK_LAYER}
    for m in MODELS:
        if m not in HOOK_LAYER:
            notes.append(f"WARN no HOOK_LAYER entry for {m}")

    out = {"cells": cells, "hook_layer": hook_layer}
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nwrote {OUT_PATH}", flush=True)
    print(f"cells: {len(cells)}  hook_layer entries: {len(hook_layer)}", flush=True)
    if notes:
        print("NOTES:", flush=True)
        for n in notes:
            print("  " + n, flush=True)
    else:
        print("NOTES: none", flush=True)


if __name__ == "__main__":
    main()
