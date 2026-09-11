#!/bin/bash
# Optional: x86_64 download-only environment for gpulogin1 (or CPU_PARTITION=compute).
# Needed only if the preflight shows compute nodes lack egress and you run stages 1-2 on the login node
# (slurm/download_on_login.sh). No torch: the download / catalogue / preprocess stages do not import it.
#   bash slurm/setup_login_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
case "$(uname -m)" in x86_64) ;; *) echo "FATAL: this builds the x86_64 env; on grace use 00_setup_env.sbatch" >&2; exit 2;; esac
prefix="${CH4HSI_ENV_X86_64}"
export CONDA_PKGS_DIRS="${CH4HSI_DATA}/.cache/conda" PIP_CACHE_DIR="${CH4HSI_DATA}/.cache/pip" PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
set +u
module load miniforge 2>/dev/null || module load Conda 2>/dev/null || { echo "FATAL: miniforge/Conda module unavailable" >&2; exit 2; }
set -u
ensure_directory "$(dirname "${prefix}")"
(
  flock 9
  [ -x "${prefix}/bin/python" ] || conda create --yes --prefix "${prefix}" python=3.11 pip
  conda env update --yes --prefix "${prefix}" --file "${CH4HSI_REPO}/environment-download.yml"
  "${prefix}/bin/python" -m pip install --no-deps mag1c
  "${prefix}/bin/python" -m pip install --no-deps -e "${CH4HSI_REPO}"
  "${prefix}/bin/python" -c 'import earthaccess, rasterio, netCDF4, ch4hsi; print("x86 download environment OK")'
) 9>"${prefix}.setup.lock"
