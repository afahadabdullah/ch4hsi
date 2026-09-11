#!/bin/bash
# End-to-end smoke test on synthetic EMIT-format data (~10 min on one GPU). Run this before
# downloading real data:   sbatch -p compute -G1 -c 4 --mem=32G -t 01:00:00 slurm/smoke.sbatch
#SBATCH -J ch4_smoke
#SBATCH -o logs/%x_%j.out
set -euo pipefail
source "${CH4HSI_REPO:-$HOME/ch4hsi}/slurm/env.sh"
ch4hsi_activate
export CH4HSI_SMOKE=${CH4HSI_SMOKE:-/explore/nobackup/people/$USER/ch4hsi_smoke}
rm -rf "$CH4HSI_SMOKE"
S="python -m ch4hsi"
C="--config configs/smoke.yaml --set paths.data_root=$CH4HSI_SMOKE"
python -m pytest -q tests || python tests/test_physics.py
for stage in make-fake-data preprocess split train evaluate mdl report; do
    echo "=== $stage"; $S $stage $C
done
cat "$CH4HSI_SMOKE/runs/smoke/REPORT.md"
