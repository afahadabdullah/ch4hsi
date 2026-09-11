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
if [ -n "${CH4HSI_HPC_DIR:-}" ] && [ -f "${CH4HSI_HPC_DIR}/common.sh" ]; then
  HPC_DIR="${CH4HSI_HPC_DIR}"
elif [ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)/common.sh" ]; then
  HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -f "${PWD}/slurm/common.sh" ]; then
  HPC_DIR="${PWD}/slurm"
elif [ -f "${PWD}/common.sh" ]; then
  HPC_DIR="${PWD}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
  HPC_DIR="${SLURM_SUBMIT_DIR}/slurm"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
  HPC_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "FATAL: common.sh not found. Run from the ch4hsi repository root." >&2
  exit 2
fi
source "${HPC_DIR}/common.sh"
ch4hsi_activate
case "${CH4HSI_LOCAL}" in /lscratch/*) trap 'rm -rf "${CH4HSI_LOCAL}"' EXIT ;; esac
echo "   stage: ${STAGE}  python: $(command -v python)  cpus: ${SLURM_CPUS_PER_TASK:-?}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ch4 "${STAGE}" "$@"
echo "== $(date +"%Y-%m-%dT%H:%M:%S%z") done ${STAGE}"
