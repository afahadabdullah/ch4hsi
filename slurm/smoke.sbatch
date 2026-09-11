#!/bin/bash
#SBATCH --job-name=ch4-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=grace
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
HPC_DIR="${CH4HSI_HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm}"
[ -f "${HPC_DIR}/common.sh" ] || HPC_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${HPC_DIR}/common.sh"
ch4hsi_activate
export CH4HSI_SMOKE="${CH4HSI_SMOKE:-${CH4HSI_DATA}/smoke}"
rm -rf "${CH4HSI_SMOKE}"
python -m pytest -q tests
for stage in make-fake-data preprocess split train evaluate mdl report; do
  echo "=== ${stage}"
  python -m ch4hsi "${stage}" --config configs/smoke.yaml --set "paths.data_root=${CH4HSI_SMOKE}" \
    --set "paths.local_scratch=${CH4HSI_LOCAL}"
done
cat "${CH4HSI_SMOKE}/runs/smoke/REPORT.md"
