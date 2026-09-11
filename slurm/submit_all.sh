#!/bin/bash
# Submit the pipeline to NCCS Prism GH200 (`grace`) nodes as a Slurm dependency chain.
# Run with bash (Prism's default login shell is tcsh):
#
#   bash slurm/submit_all.sh                          # setup-env → preflight → … → diagnose → report
#   bash slurm/submit_all.sh --from preprocess        # data already downloaded (e.g. on the login node)
#   bash slurm/submit_all.sh --from train --to evaluate
#   CH4HSI_EXTRA_SETS="--set run_name=unet_synth --set train.synth_aug.enabled=true" \
#       bash slurm/submit_all.sh --from train
#
# Re-running is safe: setup-env, download and preprocess skip completed work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
cd "${CH4HSI_REPO}"

ORDER=(setup-env preflight fetch-labels resolve-scenes download preprocess split train evaluate mdl diagnose report)
FROM=${ORDER[0]}; TO=${ORDER[-1]}
while [ $# -gt 0 ]; do
  case $1 in
    --from) FROM=$2; shift 2 ;;
    --to)   TO=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done
idx() { local i; for i in "${!ORDER[@]}"; do [ "${ORDER[$i]}" = "$1" ] && { echo "$i"; return; }; done; echo "bad stage $1" >&2; exit 1; }
I0=$(idx "${FROM}"); I1=$(idx "${TO}")
[ -z "${SG_ACCOUNT:-}${SLURM_ACCOUNT:-}" ] || echo "NOTE: not passing an account; Prism uses your default Grace allocation." >&2
mapfile -t COMMON < <(sbatch_common)

script_for() {
  case "$1" in
    setup-env) echo "slurm/00_setup_env.sbatch" ;;
    preflight) echo "slurm/01_preflight.sbatch" ;;
    *)         echo "slurm/job.sbatch $1" ;;
  esac
}

dep=()
for ((i=I0; i<=I1; i++)); do
  s=${ORDER[$i]}
  # shellcheck disable=SC2046
  jid=$(sbatch "${COMMON[@]}" ${dep[@]+"${dep[@]}"} --job-name="ch4-${s}" $(stage_resources "${s}") $(script_for "${s}"))
  jid=${jid%%;*}
  printf '%-15s %s   [%s]\n' "${s}" "${jid}" "$(stage_resources "${s}" | grep -o -- '--partition=[^ ]*')"
  dep=(--dependency=afterok:"${jid}")
  if [ "${s}" = mdl ]; then        # merge MDL shards + fit POD curves
    # shellcheck disable=SC2046
    jid=$(sbatch "${COMMON[@]}" ${dep[@]+"${dep[@]}"} --job-name=ch4-mdl-fit $(stage_resources mdl-fit) slurm/job.sbatch mdl-fit)
    printf '%-15s %s\n' "mdl-fit" "${jid}"
    dep=(--dependency=afterok:"${jid}")
  fi
done
echo "monitor: squeue --me   |   logs: ${CH4HSI_REPO}/logs/"
