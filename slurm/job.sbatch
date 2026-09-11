#!/bin/bash
#SBATCH --job-name=ch4hsi
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --export=ALL
#
# Generic stage runner. Resources come from stage_resources() in common.sh via submit_all.sh or
# submit_stage.sh, e.g.   bash slurm/submit_stage.sh train
set -euo pipefail
STAGE=${1:?usage: job.sbatch <stage> [extra ch4hsi args]}
shift || true
HPC_DIR="${CH4HSI_HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm}"
[ -f "${HPC_DIR}/common.sh" ] || HPC_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "${HPC_DIR}/common.sh" ] || { echo "FATAL: common.sh not found in ${HPC_DIR}" >&2; exit 2; }
source "${HPC_DIR}/common.sh"
ch4hsi_activate
case "${CH4HSI_LOCAL}" in /lscratch/*) trap 'rm -rf "${CH4HSI_LOCAL}"' EXIT ;; esac
echo "   stage: ${STAGE}  python: $(command -v python)  cpus: ${SLURM_CPUS_PER_TASK:-?}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ch4 "${STAGE}" "$@"
echo "== $(date -Is) done ${STAGE}"
