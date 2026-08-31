#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Dense-baseline LOAO on VESSL: build dataset -> extract residuals -> LOAO.
# Runbook, NOT an auto-launcher. Each MODE prints what it will do; job-create
# steps require an explicit `go` to actually submit (they cost GPU + judge API).
#
# Usage:
#   run_dense_loao_vessl.sh upload            # upload code + raw prompts to obj volume
#   run_dense_loao_vessl.sh smoke [go]        # 1-attack, --max-base 40, <$1 (validate end-to-end)
#   run_dense_loao_vessl.sh full  [go]        # both dense models, 4 attacks
#   run_dense_loao_vessl.sh status <job-slug>
#   run_dense_loao_vessl.sh fetch             # download results from obj volume
#
# Prereqs (once): org/team set; Mistral + Llama HF licenses accepted by the
# account behind HF_TOKEN; OPENROUTER_API_KEY + HF_TOKEN available to the job.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- config ---
OBJVOL="objvol-863tveulvjpf"                 # c1-probe object volume
SPEC="resourcespec-a100x1"                   # A100 SXM x1 (betelgeuse); L40S also ample for 7-8B
IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
PREFIX="dense-loao"                          # path prefix inside the object volume
MOUNT="/work"                                # object volume mount point in the job
WORKDIR="${MOUNT}/${PREFIX}"
RAW_LOCAL="/Users/haehyun/work/ongoing/internal-state-of-jailbroken/raw-data/raw"
CODE_LOCAL="/Users/haehyun/work/ongoing/legible-not-localized/01-code"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.1"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"

# Secret injection: HF_TOKEN and OPENROUTER_API_KEY live in the VESSL secret
# store. Set SECRET_ENV to how this org exposes them to a job -- confirm which
# form applies before the paid run:
# Org secrets (vesslctl secret list): huggingface->HF_TOKEN, OpenRouter->OPEN_ROUTER_KEY.
# We remap OpenRouter to OPENROUTER_API_KEY via --secret KEY=NAME so the driver finds it.
SECRET_ENV="${SECRET_ENV:---secret HF_TOKEN=huggingface --secret OPENROUTER_API_KEY=OpenRouter}"

MODE="${1:-help}"
GO="${2:-}"

run_or_echo() {  # submit only when GO=go, else print the command
  if [ "$GO" = "go" ]; then eval "$1"; else printf '[dry-run] %s\n  (append '\''go'\'' to submit)\n' "$1"; fi
}

# per-job command: pip deps -> build dataset -> extract residuals -> LOAO. $1=repo $2=short $3=extra
job_cmd() {
  local repo="$1" short="$2" extra="${3:-}"
  cat <<EOF
set -e
pip install -q 'transformers>=4.44' accelerate 'scikit-learn>=1.3' 'openai>=1.50' numpy
cd ${WORKDIR}
export CGEO_DATA=${WORKDIR}/out
export CGEO_ROUTING=${WORKDIR}/out/cache
python build_dense_dataset.py --model ${repo} --model-short ${short} \\
    --raw-dir ${WORKDIR}/raw --out-dir \$CGEO_DATA/dataset-permodel ${extra}
python extract_routing.py ${short} --attacks PAP hill prefill roleplay --batch-size 8
CGEO_DENSE=1 CGEO_MODELS=${short} python analyze_loao.py | tee \$CGEO_DATA/loao_${short}.txt
EOF
}

case "$MODE" in
  upload)
    echo "# Upload pipeline code + wrapped prompts to ${OBJVOL} under ${PREFIX}/"
    run_or_echo "vesslctl volume upload ${OBJVOL} ${CODE_LOCAL}/build_dense_dataset.py --remote-prefix ${PREFIX}/"
    run_or_echo "vesslctl volume upload ${OBJVOL} ${CODE_LOCAL}/extract_routing.py     --remote-prefix ${PREFIX}/"
    run_or_echo "vesslctl volume upload ${OBJVOL} ${CODE_LOCAL}/analyze_loao.py         --remote-prefix ${PREFIX}/"
    run_or_echo "vesslctl volume upload ${OBJVOL} ${RAW_LOCAL}                           --remote-prefix ${PREFIX}/raw/"
    ;;

  smoke)
    echo "# SMOKE: ${MISTRAL} on PAP, --max-base 40 (validates HF gating, forward, OpenRouter judge, extract, analyze)"
    run_or_echo "vesslctl job create -n dense-loao-smoke -r ${SPEC} -i ${IMAGE} \
      --object-volume ${OBJVOL}:${MOUNT} --working-dir ${WORKDIR} ${SECRET_ENV} \
      --cmd \"$(job_cmd ${MISTRAL} mistral-7b '--attacks PAP --max-base 40')\""
    ;;

  full)
    echo "# FULL: both dense siblings, all 4 attacks"
    run_or_echo "vesslctl job create -n dense-loao-mistral -r ${SPEC} -i ${IMAGE} \
      --object-volume ${OBJVOL}:${MOUNT} --working-dir ${WORKDIR} ${SECRET_ENV} \
      --cmd \"$(job_cmd ${MISTRAL} mistral-7b)\""
    run_or_echo "vesslctl job create -n dense-loao-llama -r ${SPEC} -i ${IMAGE} \
      --object-volume ${OBJVOL}:${MOUNT} --working-dir ${WORKDIR} ${SECRET_ENV} \
      --cmd \"$(job_cmd ${LLAMA} llama-3.1-8b)\""
    ;;

  status) vesslctl job show "${2:?job-slug}"; vesslctl job logs "${2}" ;;
  fetch)  run_or_echo "vesslctl volume download ${OBJVOL} ./dense-loao-results --remote-prefix ${PREFIX}/out/" ;;
  *) sed -n '2,17p' "$0" ;;
esac
