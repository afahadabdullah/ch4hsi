#!/bin/bash
# Generic stage runner. Resources are passed by submit_all.sh on the sbatch command line, e.g.
#   sbatch -p compute -c 8 --mem=32G -t 04:00:00 slurm/job.sbatch preprocess
#SBATCH -J ch4hsi
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --export=ALL
set -euo pipefail
STAGE=${1:?usage: job.sbatch <stage> [extra ch4hsi args]}
shift || true
source "${CH4HSI_REPO:-$HOME/ch4hsi}/slurm/env.sh"
ch4hsi_activate
case "$CH4HSI_LOCAL" in /lscratch/*) trap 'rm -rf "$CH4HSI_LOCAL"' EXIT;; esac
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_TASK_ID:-} stage=$STAGE"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ch4 "$STAGE" "$@"
echo "[$(date)] done $STAGE"
