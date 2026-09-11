#!/bin/bash
# Submit a single stage with its standard GH200 resources, e.g.
#   bash slurm/submit_stage.sh train
#   bash slurm/submit_stage.sh evaluate --dependency=afterok:123456
#   CH4HSI_EXTRA_SETS="--set run_name=mf_only --set features=[mf,mf_snr]" bash slurm/submit_stage.sh train
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
cd "${CH4HSI_REPO}"
STAGE=${1:?usage: submit_stage.sh <stage> [extra sbatch flags]}
shift
mapfile -t COMMON < <(sbatch_common)
case "${STAGE}" in
  setup-env) script=(slurm/00_setup_env.sbatch) ;;
  preflight) script=(slurm/01_preflight.sbatch) ;;
  *)         script=(slurm/job.sbatch "${STAGE}") ;;
esac
# shellcheck disable=SC2046
jid=$(sbatch "${COMMON[@]}" --job-name="ch4-${STAGE}" $(stage_resources "${STAGE}") "$@" "${script[@]}")
echo "${STAGE} -> ${jid%%;*}"
