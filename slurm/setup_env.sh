#!/bin/bash
# One-time environment setup on a Prism login node (x86_64 -> runs on `compute` and `dgx` partitions).
# For the `grace` (GH200, aarch64) partition, run this script inside an interactive grace job with
#   CH4HSI_ENV=/explore/nobackup/people/$USER/envs/ch4hsi-arm TORCH_INDEX=https://download.pytorch.org/whl/cu128
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CH4HSI_REPO=${CH4HSI_REPO:-$(dirname "$HERE")}
source "$HERE/env.sh"
# V100 (sm_70) is still in the cu126 wheels; A100/H100 also work with them.
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

set +u
module load miniforge 2>/dev/null || module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$(dirname "$CH4HSI_ENV")"
if [ ! -d "$CH4HSI_ENV" ]; then
    conda env create -p "$CH4HSI_ENV" -f "$CH4HSI_REPO/environment.yml"
fi
conda activate "$CH4HSI_ENV"
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"
# mag1c is only needed for its ch4.hdr/ch4.lut absorption LUT; --no-deps avoids re-resolving torch
pip install --no-deps mag1c
pip install -e "$CH4HSI_REPO"
python - <<'EOF'
import torch, numpy, rasterio, netCDF4, earthaccess
from ch4hsi.physics.target import load_lut
wl, rads, conc = load_lut("mag1c")
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| LUT", rads.shape, f"{wl.min():.0f}-{wl.max():.0f} nm")
EOF
mkdir -p "$CH4HSI_DATA" "$CH4HSI_REPO/logs"
[ -f "$HOME/.netrc" ] || echo "WARNING: create ~/.netrc with your Earthdata login (see README) before downloading"
echo "environment ready: $CH4HSI_ENV"
