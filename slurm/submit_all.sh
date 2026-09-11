#!/bin/bash
# Submit the whole pipeline as a Slurm dependency chain on NCCS Prism.
#
#   bash slurm/submit_all.sh                      # everything
#   bash slurm/submit_all.sh --from preprocess    # data already downloaded (e.g. on the login node)
#   bash slurm/submit_all.sh --from train --to evaluate
#   CH4HSI_EXTRA_SETS="--set run_name=unet_synth --set train.synth_aug.enabled=true" bash slurm/submit_all.sh --from train
#
# Stages: fetch-labels resolve-scenes download preprocess split train evaluate mdl report
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CH4HSI_REPO=${CH4HSI_REPO:-$(dirname "$HERE")}
source "$HERE/env.sh"
mkdir -p "$CH4HSI_REPO/logs"
cd "$CH4HSI_REPO"

ORDER=(fetch-labels resolve-scenes download preprocess split train evaluate mdl report)
FROM=${ORDER[0]}; TO=${ORDER[-1]}
while [ $# -gt 0 ]; do
    case $1 in
        --from) FROM=$2; shift 2;;
        --to) TO=$2; shift 2;;
        *) echo "unknown arg $1"; exit 1;;
    esac
done
idx() { local i; for i in "${!ORDER[@]}"; do [ "${ORDER[$i]}" = "$1" ] && echo "$i" && return; done; echo "bad stage $1" >&2; exit 1; }
I0=$(idx "$FROM"); I1=$(idx "$TO")

CPU="-p $CPU_PARTITION $SBATCH_ACCOUNT_OPT"
GPU="-p $GPU_PARTITION -G1 $SBATCH_ACCOUNT_OPT"
declare -A RES=(
    [fetch-labels]="$CPU -c 4 --mem=8G -t 03:00:00"
    [resolve-scenes]="$CPU -c 2 --mem=8G -t 06:00:00"
    [download]="$CPU -c 4 --mem=8G -t 12:00:00 --array=0-$((DOWNLOAD_TASKS-1))%$DOWNLOAD_CONCURRENCY"
    [preprocess]="$CPU -c 8 --mem=48G -t 10:00:00 --array=0-$((PREPROCESS_TASKS-1))"
    [split]="$CPU -c 4 --mem=32G -t 01:00:00"
    [train]="$GPU -c 10 --mem=96G -t 1-00:00:00 --requeue"
    [evaluate]="$GPU -c 8 --mem=96G -t 06:00:00"
    [mdl]="$GPU -c 8 --mem=96G -t 12:00:00 --array=0-$((MDL_TASKS-1))"
    [report]="$CPU -c 2 --mem=8G -t 00:30:00"
)
dep=""
for ((i=I0; i<=I1; i++)); do
    s=${ORDER[$i]}
    # shellcheck disable=SC2086
    jid=$(sbatch --parsable -J "ch4_$s" $dep ${RES[$s]} slurm/job.sbatch "$s")
    jid=${jid%%;*}
    echo "submitted $s -> $jid"
    dep="--dependency=afterok:$jid"
    if [ "$s" = "mdl" ]; then       # merge MDL shards + fit POD curves
        jid=$(sbatch --parsable -J ch4_mdl-fit $dep $CPU -c 2 --mem=8G -t 00:30:00 slurm/job.sbatch mdl-fit)
        echo "submitted mdl-fit -> $jid"; dep="--dependency=afterok:$jid"
    fi
done
echo "monitor: squeue -u $USER ; logs in $CH4HSI_REPO/logs/"
