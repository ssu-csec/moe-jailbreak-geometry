# Dense-model baseline (rebuttal artifact)

Derived results for the dense-transformer control requested by Reviewers 1 and 3.
Two dense models, **Llama-3.1-8B-Instruct** and **Mistral-7B-Instruct-v0.3**, run
through the same residual-stream leave-one-attack-out pipeline as the MoE panel in
the paper: the same four attacks (PAP, hill, prefill, roleplay), the same four
benchmarks, the same L2 logistic probe with StandardScaler, PCA-256, and a
4000-resample bootstrap, and the same Claude Sonnet 4.6 judge (via OpenRouter).
The pipeline is identical except for layer selection, so the fixed-depth rows are
the strictly like-for-like comparison with MoE.

## Files

| File | Content |
|------|---------|
| `loao_dense_3depth.log` | Leave-one-attack-out balanced accuracy, per model and pooled, at three depths (train-best, fixed mid-50%, fixed late-90%), with 95% bootstrap CIs and the per-held-out-attack breakdown. |
| `loao_llama-3.1-8b.log`, `loao_mistral-7b.log` | Per-model leave-one-attack-out detail. |
| `auroc_dense.log` | Balanced accuracy vs. threshold-free AUROC at fixed depth, per held-out attack. |
| `within_family.log` | Within-family (cross-template) transfer for roleplay and hill: train on half the templates, test on the other half. |
| `second_judge_agreement.log` | Second-judge validation: GPT-4o vs. Claude Sonnet 4.6 agreement and Cohen's kappa, per model and attack. |
| `manifest_<model>.json` | Per-model run manifest: model id, judge, base-request count, survivors, kept template names, and per-attack complied/refused/unclear counts. |
| `labels/<model>/<attack>.jsonl` | Per-cell labels, one record per prompt: `{id, base_id, outcome}` where `outcome` is the judge's `complied`/`refused` verdict. |

## How the numbers map to the rebuttal

- Pooled residual LOAO: fixed mid-50% **0.632** [0.52, 0.71], fixed late-90% **0.648** [0.50, 0.68] (`loao_dense_3depth.log`), against MoE 0.585 / 0.667.
- Per held-out attack: PAP **0.736**, hill **0.645**, prefill **0.605** (Mistral 0.71, Llama 0.50), roleplay **0.546** with its own interval **[0.51, 0.58]** excluding the 0.62 transfer level (`loao_dense_3depth.log`).
- Roleplay is a direction mismatch, not a threshold artifact: cross-family AUROC **0.54 to 0.56**, but within-family AUROC **0.75 to 0.80** (`auroc_dense.log`, `within_family.log`).
- Second judge agrees with the primary judge at **92.9% to 97.9%** (kappa 0.86 to 0.95), and **98% to 100%** on prefill (`second_judge_agreement.log`).
- Survivors after the vanilla-refusal filter: **370/386** (Llama) and **328/386** (Mistral) (`manifest_*.json`).

## What is not here, and why

Per the paper's release policy (Ethical Considerations), no raw harmful responses
are released. These files contain only derived numerical features and prompt-level
metadata (judge label and query identifier); the wrapped prompt text is stripped
from the label files. The raw model responses, the judge verdict transcripts, the
per-(model, attack) activation caches (`*.npz`, roughly 1.1 GB), and the wrapped
prompts live on the VESSL object volume `objvol-863tveulvjpf` under `dense-loao/`,
and are available to academic users on request, matching the policy for the MoE
caches in the top-level README.

**Canonical sources.** The Mistral labels and manifest come from the
Mistral-7B-Instruct-**v0.3** run (`dense-loao/out-mistral/`). A stale v0.1 smoke
run left a partial `dense-loao/out/dataset-permodel/mistral-7b/` (40 base requests,
PAP only); it is not used. The Mistral `.npz` caches in `dense-loao/out/cache/` and
`dense-loao/out-mistral/cache/` are byte-identical v0.3 caches, and the leave-one-attack-out
analysis reads the caches, so the reported numbers are unaffected by the stale dataset directory.

## Regenerating

Scripts are in `01-code/`. With `CGEO_DATA` and `CGEO_ROUTING` pointed at the raw
caches:

```bash
# 1. build per-model dataset (generate + judge; needs OPENROUTER_API_KEY)
python 01-code/build_dense_dataset.py --model meta-llama/Llama-3.1-8B-Instruct \
    --model-short llama-3.1-8b --raw-dir <wrapped-prompts> --out-dir <CGEO_DATA>/dataset-permodel
# 2. extract residual-stream caches
python 01-code/extract_routing.py            # dense branch: residual-only npz
# 3. leave-one-attack-out at three depths
CGEO_DENSE=1 CGEO_MODELS=llama-3.1-8b,mistral-7b python 01-code/analyze_loao.py
# 4. AUROC (threshold-free) and within-family transfer
CGEO_MODELS=llama-3.1-8b,mistral-7b python 01-code/analyze_auroc.py
WF_ATTACK=roleplay CGEO_MODELS=llama-3.1-8b,mistral-7b python 01-code/analyze_within_family.py
# 5. second-judge validation
SECOND_JUDGE=openai/gpt-4o python 01-code/rejudge_second.py <cache_dir>
```
