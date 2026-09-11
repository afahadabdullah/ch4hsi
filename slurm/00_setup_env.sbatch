#!/bin/bash
#SBATCH --job-name=ch4-setup-env
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=grace
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#
# Build (once) the aarch64 environment on a GH200 node: gpulogin1 is x86_64, so a Grace env cannot be
# built there. Idempotent and flock-serialised; submit_all.sh runs it first and skips work if the env
# already imports torch with CUDA. Submit from the repo root:  sbatch slurm/00_setup_env.sbatch
set -euo pipefail
HPC_DIR="${CH4HSI_HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm}"
[ -f "${HPC_DIR}/common.sh" ] || HPC_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "${HPC_DIR}/common.sh" ] || { echo "FATAL: common.sh not found (submit from the repo root)" >&2; exit 2; }
source "${HPC_DIR}/common.sh"

case "$(uname -m)" in
  aarch64|arm64)
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # aarch64 (sbsa) CUDA wheels
    prefix="${CH4HSI_ENV_AARCH64}"
    ;;
  *)
    # x86_64 (e.g. V100, A100, RTX, etc.)
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"   # x86_64 CUDA wheels
    prefix="${CH4HSI_ENV_X86_64}"
    ;;
esac

export CONDA_PKGS_DIRS="${CH4HSI_DATA}/.cache/conda" PIP_CACHE_DIR="${CH4HSI_DATA}/.cache/pip"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
set +u
module load miniforge 2>/dev/null || module load Conda 2>/dev/null || { echo "FATAL: miniforge/Conda module unavailable" >&2; exit 2; }
set -u
command -v conda >/dev/null || { echo "FATAL: conda not found after module load" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

check() {
  "${prefix}/bin/python" - <<'EOF'
import torch, numpy, scipy, pandas, rasterio, netCDF4, h5py, pyproj, earthaccess, ch4hsi
from ch4hsi.physics.target import load_lut
assert torch.cuda.is_available(), "torch built without usable CUDA on this node"
wl, rads, _ = load_lut("mag1c")
n_g = torch.cuda.device_count()
devs = [torch.cuda.get_device_name(i) for i in range(n_g)]
print(f"OK torch {torch.__version__} cuda {torch.version.cuda} ({n_g} GPUs: {devs}); "
      f"bf16={torch.cuda.is_bf16_supported()}; LUT {rads.shape}")
EOF
}

ensure_directory "$(dirname "${prefix}")"
(
  flock 9
  if [ -x "${prefix}/bin/python" ] && check 2>/dev/null; then
    echo "environment already complete: ${prefix}"; exit 0
  fi
  if [ ! -x "${prefix}/bin/python" ]; then
    echo "Creating one-time $(uname -m) environment: ${prefix}"
    conda create --yes --prefix "${prefix}" python=3.11 pip
  fi
  conda env update --yes --prefix "${prefix}" --file "${CH4HSI_REPO}/environment.yml"
  py="${prefix}/bin/python"
  "${py}" -m pip install --upgrade pip
  if ! "${py}" -m pip install torch --index-url "${TORCH_INDEX}" || \
     ! "${py}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "pip CUDA wheel unavailable/unusable on this node; falling back to conda-forge pytorch-gpu"
    "${py}" -m pip uninstall -y torch || true
    conda install --yes --prefix "${prefix}" -c conda-forge pytorch-gpu
  fi
  "${py}" -m pip install --no-deps mag1c          # only for its ch4.hdr / ch4.lut absorption LUT
  "${py}" -m pip install --no-deps -e "${CH4HSI_REPO}"
  check
) 9>"${prefix}.setup.lock"
