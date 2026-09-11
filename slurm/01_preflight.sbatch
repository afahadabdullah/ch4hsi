#!/bin/bash
#SBATCH --job-name=ch4-preflight
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=compute
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#
# Prove this node class can reach every endpoint before anything expensive is queued. The download
# stages depend on this with afterok, so missing compute-node egress stops the chain here.
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

echo "--- egress ---"
for url in \
  "https://cmr.earthdata.nasa.gov/search/collections.json?short_name=EMITL1BRAD" \
  "https://urs.earthdata.nasa.gov/" \
  "https://data.lpdaac.earthdatacloud.nasa.gov/" \
  "https://earth.gov/ghgcenter/api/stac/collections/emit-ch4plume-v1"
do
  require_egress "${url}"
done

echo "--- credentials ---"
if [ -r "${HOME}/.netrc" ] && grep -q "urs.earthdata.nasa.gov" "${HOME}/.netrc"; then
  perm=$(stat -c %a "${HOME}/.netrc")
  [ "${perm}" = 600 ] || echo "WARNING: chmod 600 ~/.netrc (currently ${perm})"
  echo "   ~/.netrc has an Earthdata entry"
elif [ -n "${EARTHDATA_USERNAME:-}" ] && [ -n "${EARTHDATA_PASSWORD:-}" ]; then
  echo "   EARTHDATA_USERNAME/PASSWORD set"
else
  echo "FATAL: no Earthdata credentials (~/.netrc or EARTHDATA_USERNAME/PASSWORD)" >&2; exit 2
fi

echo "--- environment ---"
ch4hsi_activate
python -c 'import earthaccess, ch4hsi; print("earthaccess", earthaccess.__version__, "| ch4hsi", ch4hsi.__version__)'
echo "--- storage ---"
df -h "${CH4HSI_DATA}" | tail -1
echo "preflight OK"
