#!/bin/bash
# Fallback when the preflight shows compute-node egress is blocked: run stages 1-2 on gpulogin1
# (x86_64 env from setup_login_env.sh). Resumable: complete files are skipped, partials restart.
#   tmux new -s ch4dl 'bash slurm/download_on_login.sh'
#   bash slurm/submit_all.sh --from preprocess
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
ch4hsi_activate
renice -n 10 $$ >/dev/null 2>&1 || true
ch4 fetch-labels                        2>&1 | tee -a logs/login-fetch-labels.log
ch4 resolve-scenes                      2>&1 | tee -a logs/login-resolve-scenes.log
ch4 download --task-id 0 --num-tasks 1  2>&1 | tee -a logs/login-download.log
