#!/usr/bin/env python3
"""Extract MoE routing and residual-stream activations at the last prompt token.

For each (model, attack) pair, captures every MoE layer's router logits and
every decoder layer's residual-stream output at the last prompt token, and
writes the result to ``$CGEO_ROUTING/<model>/<attack>.npz`` with fields the
paper's analysis scripts read:

    R_router_logits / C_router_logits     (n, n_moe_layers, n_experts)   fp16
    R_selected_experts / C_selected_experts (n, n_moe_layers, top_k)     int16
    R_residual / C_residual               (n, n_decoder_layers, d_model) fp16
    R_query_ids / C_query_ids             (n,)                           object
    R_outcomes / C_outcomes               (n,)                           int8 (0=refused, 1=complied)
    gate_layer_indices                    (n_moe_layers,)                int32
    residual_layer_count                  ()                             int32
    meta                                  (1,)                           object dict

Input format under ``$CGEO_DATA/dataset-permodel/<model>/<attack>_{refused,complied}.jsonl``
(each line a JSON object with at least ``id`` and ``prompt``).

Requires ``torch`` and ``transformers``; this is the GPU-side script that
produces the routing caches the analysis pipeline reads.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("CGEO_DATA", "./data"))
ROUTING_ROOT = Path(os.environ.get("CGEO_ROUTING", str(DATA_ROOT / "cache_moe_routing")))

MODEL_CFG = {
    "deepseek-v2-lite":       dict(repo="deepseek-ai/DeepSeek-V2-Lite-Chat",
                                   trc=False, moe_class="DeepseekV2Moe",
                                   gate_attr="gate", top_k=6, verified=True),
    "deepseek-moe-16b-chat":  dict(repo="deepseek-ai/deepseek-moe-16b-chat",
                                   trc=True, moe_class="DeepseekMoE",
                                   gate_attr="gate", top_k=6, verified=False,
                                   attn_impl="eager"),
    "qwen1.5-moe-a2.7b-chat": dict(repo="Qwen/Qwen1.5-MoE-A2.7B-Chat",
                                   trc=False, moe_class="Qwen2MoeSparseMoeBlock",
                                   gate_attr="gate", top_k=4, verified=False),
    "llama-4-scout":          dict(repo="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                                   trc=False, moe_class="Llama4TextMoe",
                                   gate_attr="router", top_k=1, verified=False),
    "mixtral-8x7b":           dict(repo="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                   trc=False, moe_class="MixtralSparseMoeBlock",
                                   gate_attr="gate", top_k=2, verified=False),
    "olmoe-1b-7b":            dict(repo="allenai/OLMoE-1B-7B-0924-Instruct",
                                   trc=False, moe_class="OlmoeSparseMoeBlock",
                                   gate_attr="gate", top_k=8, verified=False),
    "phi-3.5-moe-instruct":   dict(repo="microsoft/Phi-3.5-MoE-instruct",
                                   trc=False, moe_class="PhimoeSparseMoeBlock",
                                   gate_attr="gate", top_k=2, verified=False),
    "gpt-oss-20b":            dict(repo="openai/gpt-oss-20b",
                                   trc=False, moe_class="GptOssExperts",
                                   gate_attr="router", top_k=4, verified=False),
    # Dense baselines: residual-stream only, no MoE routing. Added for the
    # dense-vs-MoE leave-one-attack-out comparison (rebuttal). The residual hook
    # below is architecture-generic, so these run through the identical residual
    # path as the MoE models; only the MoE-specific capture is skipped.
    "mistral-7b":             dict(repo="mistralai/Mistral-7B-Instruct-v0.3",
                                   trc=False, dense=True, verified=True),
    "llama-3.1-8b":           dict(repo="meta-llama/Llama-3.1-8B-Instruct",
                                   trc=False, dense=True, verified=True),
}


def load_split(model, attack):
    """Return (refused, complied), each a list of (id, prompt)."""
    base = DATA_ROOT / "dataset-permodel" / model
    out = {}
    for label in ("refused", "complied"):
        rows = []
        with open(base / f"{attack}_{label}.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append((r["id"], r["prompt"]))
        out[label] = rows
    return out["refused"], out["complied"]


def gate_weight(block, gate_attr):
    """Return the (n_experts, d_model) gate weight matrix for a MoE block."""
    g = getattr(block, gate_attr)
    w = getattr(g, "weight", None)
    if w is not None and w.dim() == 2:
        return w
    for sub in ("gate", "linear", "router", "classifier", "wg"):
        gg = getattr(g, sub, None)
        if gg is not None and getattr(gg, "weight", None) is not None and gg.weight.dim() == 2:
            return gg.weight
    raise RuntimeError(f"cannot locate a 2-D gate weight under .{gate_attr} "
                       f"(type {type(g).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODEL_CFG))
    ap.add_argument("--attacks", nargs="+",
                    default=["PAP", "hill", "prefill", "roleplay"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()

    cfg = MODEL_CFG[args.model]
    if not cfg["verified"]:
        print(f"[warn] gate-weight access for {args.model} is best-effort -- "
              f"check the printed n_experts is sane.", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {cfg['repo']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], trust_remote_code=cfg["trc"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    load_kw = dict(dtype=torch.bfloat16, device_map="auto",
                   trust_remote_code=cfg["trc"], low_cpu_mem_usage=True)
    if cfg.get("attn_impl"):
        load_kw["attn_implementation"] = cfg["attn_impl"]
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["repo"], **load_kw)
    except TypeError:
        load_kw["torch_dtype"] = load_kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(cfg["repo"], **load_kw)
    model.eval()

    # Locate MoE blocks and their containing decoder layers. Dense models have
    # none, so the routing capture is skipped and only the residual is recorded.
    dense = bool(cfg.get("dense", False))
    if dense:
        moe_blocks, moe_layer_idx, Ws = [], [], []
        n_experts, top_k = 0, 0
    else:
        moe = []
        for name, mod in model.named_modules():
            if type(mod).__name__ == cfg["moe_class"]:
                m = re.search(r"layers?\.(\d+)\.", name)
                layer_idx = int(m.group(1)) if m else len(moe)
                moe.append((layer_idx, mod))
        moe.sort(key=lambda x: x[0])
        if not moe:
            sys.exit(f"no '{cfg['moe_class']}' modules found in {args.model}")
        moe_layer_idx = [i for i, _ in moe]
        moe_blocks = [b for _, b in moe]

        Ws = [gate_weight(b, cfg["gate_attr"]).detach().float() for b in moe_blocks]
        n_experts = Ws[0].shape[0]
        top_k = int(cfg["top_k"])

    # All decoder layers (model.model.layers); the residual stream is each layer's output.
    decoder_layers = list(model.model.layers)
    n_decoder = len(decoder_layers)
    d_model = model.config.hidden_size
    if dense:
        print(f"[arch] DENSE model; {n_decoder} decoder layers, d_model={d_model} "
              f"(residual stream only)", flush=True)
    else:
        print(f"[arch] {len(moe_blocks)} MoE blocks at layers {moe_layer_idx[0]}..{moe_layer_idx[-1]} "
              f"({n_experts} experts, top_k={top_k}); {n_decoder} decoder layers, d_model={d_model}",
              flush=True)

    state = {"active": False}
    router_cap = {}     # id(moe_block) -> (B, n_experts) at last prompt token
    residual_cap = {}   # decoder_layer_idx -> (B, d_model) at last prompt token

    def make_router_hook(block, W):
        def pre_hook(_mod, inp):
            if state["active"]:
                last = inp[0][:, -1, :].to(W.dtype)
                router_cap[id(block)] = (last @ W.t()).detach().to("cpu", torch.float16)
        return pre_hook

    def make_residual_hook(layer_idx):
        def post_hook(_mod, _inp, output):
            if state["active"]:
                # decoder layer output is either a tensor or a tuple whose first element is the hidden state
                h = output[0] if isinstance(output, tuple) else output
                residual_cap[layer_idx] = h[:, -1, :].detach().to("cpu", torch.float16)
        return post_hook

    handles = []
    if not dense:
        handles += [b.register_forward_pre_hook(make_router_hook(b, W))
                    for b, W in zip(moe_blocks, Ws)]
    handles += [dl.register_forward_hook(make_residual_hook(i))
                for i, dl in enumerate(decoder_layers)]

    ROUTING_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = ROUTING_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for attack in args.attacks:
            refused, complied = load_split(args.model, attack)
            if len(refused) < 5 or len(complied) < 5:
                print(f"[{attack}] too few prompts ({len(refused)}R/{len(complied)}C), skip",
                      flush=True)
                continue
            items = [(i, p, 0) for i, p in refused] + [(i, p, 1) for i, p in complied]
            print(f"\n[{attack}] {len(refused)} refused + {len(complied)} complied", flush=True)

            n_items = len(items)
            if not dense:
                router_logits = np.zeros((n_items, len(moe_blocks), n_experts), dtype=np.float16)
            residual = np.zeros((n_items, n_decoder, d_model), dtype=np.float16)

            t0 = time.time()
            for s in range(0, n_items, args.batch_size):
                batch = items[s:s + args.batch_size]
                texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True)
                         for _, p, _ in batch]
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          add_special_tokens=False, max_length=args.max_length).to(model.device)
                state["active"] = True
                with torch.no_grad():
                    model(**enc, use_cache=False)
                state["active"] = False

                resid_stack = np.stack([residual_cap[i].numpy() for i in range(n_decoder)], axis=1)
                residual[s:s + len(batch)] = resid_stack[:len(batch)]
                if not dense:
                    router_stack = np.stack([router_cap[id(b)].numpy() for b in moe_blocks], axis=1)
                    router_logits[s:s + len(batch)] = router_stack[:len(batch)]

            ids = np.array([i for i, _, _ in items], dtype=object)
            y = np.array([lab for _, _, lab in items], dtype=np.int8)

            arrays = dict(
                R_residual=residual[y == 0],
                C_residual=residual[y == 1],
                R_query_ids=ids[y == 0],
                C_query_ids=ids[y == 1],
                R_outcomes=y[y == 0],
                C_outcomes=y[y == 1],
                residual_layer_count=np.int32(n_decoder),
                meta=np.array([{"model": args.model, "attack": attack,
                                "dense": bool(dense),
                                "n_experts": int(n_experts),
                                "top_k": int(top_k),
                                "n_moe_layers": len(moe_blocks),
                                "n_decoder_layers": int(n_decoder),
                                "d_model": int(d_model),
                                "capture_position": "last_prompt_token"}],
                              dtype=object),
            )
            if not dense:
                # selected_experts: top-k by router logits per (sample, MoE layer)
                selected_experts = np.argpartition(-router_logits, kth=min(top_k, n_experts - 1),
                                                   axis=-1)[..., :top_k].astype(np.int16)
                arrays.update(
                    R_router_logits=router_logits[y == 0],
                    C_router_logits=router_logits[y == 1],
                    R_selected_experts=selected_experts[y == 0],
                    C_selected_experts=selected_experts[y == 1],
                    gate_layer_indices=np.array(moe_layer_idx, dtype=np.int32),
                )
            np.savez(out_dir / f"{attack}.npz", **arrays)
            print(f"[{attack}] saved {attack}.npz  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
