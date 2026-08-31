# Legible but Not Localized: Jailbreak Geometry in Mixture-of-Experts Language Models Is Attack-Specific

Official code and data for our **EMNLP 2026 Main Conference** paper.

**Paper:** Coming soon · **Code and data:** This repository

## Overview

Activation-based jailbreak defenses often assume that successful jailbreaks leave a stable internal signature that generalizes across attacks. We test this assumption across six open-weight mixture-of-experts language models and four jailbreak attacks.

We find that jailbreak behavior is readily distinguishable within each attack, but its internal location is attack-specific: probes trained on known attacks fail to generalize to a held-out attack. This pattern appears in both router logits and residual-stream representations, showing that activation-based defenses cannot be assumed to detect jailbreaks they were not explicitly designed for.

## Reproduction Code and Data

This repository contains the analysis code, base harmful requests, and derived results for the paper.

The bundle is sufficient to regenerate every figure and table in the paper from the released logs and JSONs.
Raw activation caches and routing extractions are not part of the bundle because of their size; they are available to academic users on request.

## Directory Structure

```
.
├── 01-code/                   Python analysis, extraction, and figure scripts
├── 02-data/                   Processed per-cell logs and pooled JSONs
│   ├── routing-analysis/      Routing, RSA, re-ID, layerwise, etc. logs
│   └── dense-baseline/        Dense-model control (Llama-3.1-8B, Mistral-7B): LOAO, AUROC, within-family, second-judge logs + per-cell labels
├── requirements.txt
├── LICENSE
└── README.md
```

## Environment

Tested under Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional, only required for re-running the routing extraction in `01-code/extract_routing.py`:

```bash
pip install torch transformers
```

## Configuration via Environment Variables

The analysis scripts read the raw caches through environment variables; all default to local relative paths so the released bundle is self-contained.

| Variable        | Default                       | Purpose                                                  |
|-----------------|-------------------------------|----------------------------------------------------------|
| `CGEO_DATA`     | `./data`                      | Per-model dataset directory (response JSONLs, judge labels). |
| `CGEO_ROUTING`  | `./data/cache_moe_routing`    | Per-(model, attack) routing extraction caches.           |

Set them to point at the raw caches if you have them locally:

```bash
export CGEO_DATA=/path/to/dataset
export CGEO_ROUTING=/path/to/cache_moe_routing
```

## Reproducing the Figures

The released `02-data/` already contains the pooled JSONs and logs that the figure scripts read.
The rendered outputs are also included in `figures/`.
To regenerate them:

```bash
python 01-code/fig_teaser.py      # Figure 1 (teaser)
python 01-code/fig_decode.py      # Figure 2 (within-attack decode)
python 01-code/fig_transfer.py    # Figure 3 (LOAO transfer)
python 01-code/fig_matrix.py      # Figure 4 (cross-attack transfer matrix)
python 01-code/fig_depth.py       # Figure 5 (depth sweep)
python 01-code/fig_decomp.py      # Figure 6 (effective dimensionality)
```

## Re-running Analyses

If `CGEO_DATA` and `CGEO_ROUTING` are pointed at the raw caches, all analyses can be re-derived from scratch:

```bash
python 01-code/analyze_routing.py            # within-attack routing decode
python 01-code/analyze_experts.py            # per-expert sparsity
python 01-code/analyze_loao.py               # leave-one-attack-out transfer
python 01-code/analyze_loao_layersens.py     # layer-selection robustness
python 01-code/analyze_loao_mlp.py           # MLP-probe robustness
python 01-code/analyze_transfer_matrix.py    # cross-attack transfer matrix
python 01-code/analyze_layerwise.py          # depth sweep
python 01-code/analyze_decision_rotation.py  # comply-vs-refuse direction cosine
python 01-code/analyze_refusal_routing.py    # refusal-routing follow-up
python 01-code/analyze_dimensionality.py     # participation ratio
python 01-code/analyze_dimensionality_templatebalance.py  # template-balance check
python 01-code/analyze_rsa.py                # representational-similarity analysis
python 01-code/analyze_reid.py               # cross-attack re-identification + TF-IDF baseline
python 01-code/analyze_compliance_quality.py # compliance-quality robustness
```

## Data Availability

- **Released here:** all per-cell logs, pooled JSONs, figure outputs, and analysis/figure scripts, plus the dense-model baseline derived logs and per-cell labels in `02-data/dense-baseline/`.
- **On request to academic users:** raw activation caches (`cache_moe_routing/`) and per-model wrapped-prompt datasets used by the analysis scripts, including the dense-baseline caches and responses on the object volume `objvol-863tveulvjpf` under `dense-loao/`.
- **Wrapper scripts and base harmful requests:** drawn from HarmBench, AdvBench, StrongREJECT, and HarmEval; wrappers are deterministically reproducible from those sources.

## License

Code and analysis artifacts are released under the MIT License (see `LICENSE`).
The benchmark harm-prompt content is governed by the licenses of the upstream datasets cited in the paper.
