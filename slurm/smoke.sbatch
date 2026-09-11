#!/bin/bash
#SBATCH --job-name=ch4-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=v100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#
# End-to-end smoke test on one GH200 with synthetic EMIT-format data (~10 min). Run after
# 00_setup_env.sbatch and before downloading real data. From the repo root:
#   sbatch --dependency=afterok:<setup_job> slurm/smoke.sbatch     (or just: sbatch slurm/smoke.sbatch)
set -euo pipefail
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
export CH4HSI_SMOKE="${CH4HSI_SMOKE:-${CH4HSI_DATA}/smoke}"
rm -rf "${CH4HSI_SMOKE}"
python -m pytest -q tests
for stage in make-fake-data preprocess split train evaluate mdl diagnose report; do
  echo "=== ${stage}"
  python -m ch4hsi "${stage}" --config configs/smoke.yaml --set "paths.data_root=${CH4HSI_SMOKE}" \
    --set "paths.local_scratch=${CH4HSI_LOCAL}"
done
cat "${CH4HSI_SMOKE}/runs/smoke/REPORT.md"
