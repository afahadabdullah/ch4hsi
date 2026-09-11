#!/bin/bash
# Fallback when compute nodes have no outbound internet: run stages 1-2 on the login node.
# Resumable (existing/complete files are skipped). Run inside tmux/screen:
#   tmux new -s ch4dl  'bash slurm/download_on_login.sh'
# then:  bash slurm/submit_all.sh --from preprocess
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CH4HSI_REPO=${CH4HSI_REPO:-$(dirname "$HERE")}
source "$HERE/env.sh"
ch4hsi_activate
mkdir -p logs
renice -n 10 $$ >/dev/null 2>&1 || true
ch4 fetch-labels                          2>&1 | tee -a logs/login_fetch_labels.log
ch4 resolve-scenes                        2>&1 | tee -a logs/login_resolve.log
ch4 download --task-id 0 --num-tasks 1    2>&1 | tee -a logs/login_download.log
