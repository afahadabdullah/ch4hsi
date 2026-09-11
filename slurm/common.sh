#!/bin/bash
# Shared NCCS Prism settings for ch4hsi (conventions follow the StormGrid scripts/hpc setup).
#
# Target hardware: Prism Grace Hopper nodes, partition `grace`
#   1x NVIDIA GH200 (H100, 96 GB HBM3) + 72-core Grace CPU (aarch64), 480 GB RAM, 1 TB /lscratch NVMe
# The login node (gpulogin1) is x86_64, so the aarch64 job environment is built *inside a grace job*
# (00_setup_env.sbatch); an optional x86 download-only env serves login-node fallbacks.
#
# Optional overrides go in slurm/site.env (git-ignored; see site.env.example).

CH4HSI_HPC_DIR="${CH4HSI_HPC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CH4HSI_REPO="${CH4HSI_REPO:-$(cd "${CH4HSI_HPC_DIR}/.." && pwd)}"
export CH4HSI_HPC_DIR CH4HSI_REPO
[ -f "${CH4HSI_HPC_DIR}/site.env" ] && source "${CH4HSI_HPC_DIR}/site.env"

# ---------------------------------------------------------------- storage
if [ -z "${CH4HSI_DATA:-}" ]; then
  # Prism's current shared high-capacity path first, then the older /explore layout.
  if [ -d "/panfs/ccds02/nobackup/people/${USER}" ]; then
    CH4HSI_DATA="/panfs/ccds02/nobackup/people/${USER}/ch4hsi"
  elif [ -d "/explore/nobackup/people/${USER}" ]; then
    CH4HSI_DATA="/explore/nobackup/people/${USER}/ch4hsi"
  else
    CH4HSI_DATA="${CH4HSI_REPO}/data"
  fi
fi
export CH4HSI_DATA

# PanFS can report EEXIST during simultaneous recursive creation (array tasks); recheck final state.
ensure_directory() {
  local d=$1
  [ -d "${d}" ] || mkdir -p "${d}" 2>/dev/null || [ -d "${d}" ]
}
ensure_directory "${CH4HSI_DATA}"
ensure_directory "${CH4HSI_REPO}/logs"

# ---------------------------------------------------------------- partitions / resources
# Default partition & resource selection keyed by host architecture or user override in site.env:
case "$(uname -m)" in
  aarch64|arm64)
    export GPU_PARTITION="${GPU_PARTITION:-grace}"
    export CPU_PARTITION="${CPU_PARTITION:-grace}"
    export CH4HSI_CONFIG="${CH4HSI_CONFIG:-${CH4HSI_REPO}/configs/gh200.yaml}"
    export TRAIN_GPUS="${TRAIN_GPUS:-1}"
    export TRAIN_CPUS="${TRAIN_CPUS:-32}"
    export TRAIN_MEM="${TRAIN_MEM:-240G}"
    ;;
  *)
    # x86 architecture (e.g. 2x V100, 20 CPU cores, 380GB RAM on Prism/Discover compute partition)
    export GPU_PARTITION="${GPU_PARTITION:-${SLURM_JOB_PARTITION:-compute}}"
    export CPU_PARTITION="${CPU_PARTITION:-compute}"
    export CH4HSI_CONFIG="${CH4HSI_CONFIG:-${CH4HSI_REPO}/configs/x86_v100.yaml}"
    export TRAIN_GPUS="${TRAIN_GPUS:-2}"
    export TRAIN_CPUS="${TRAIN_CPUS:-20}"
    export TRAIN_MEM="${TRAIN_MEM:-380G}"
    ;;
esac

export DOWNLOAD_TASKS="${DOWNLOAD_TASKS:-8}" DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-4}"
export PREPROCESS_TASKS="${PREPROCESS_TASKS:-16}" MDL_TASKS="${MDL_TASKS:-4}"

# Environments are keyed by architecture (aarch64 or x86_64).
export CH4HSI_ENV_AARCH64="${CH4HSI_ENV_AARCH64:-${CH4HSI_DATA}/.envs/ch4hsi-aarch64}"
export CH4HSI_ENV_X86_64="${CH4HSI_ENV_X86_64:-${CH4HSI_DATA}/.envs/ch4hsi-x86_64}"
ch4hsi_env_prefix() {
  case "$(uname -m)" in
    aarch64|arm64) echo "${CH4HSI_ENV_AARCH64}" ;;
    *)             echo "${CH4HSI_ENV_X86_64}" ;;
  esac
}

# stage -> sbatch resource flags (single source of truth for submit_all.sh / submit_stage.sh)
stage_resources() {
  local cpu_part=""
  local gpu_part=""
  [ -n "${CPU_PARTITION:-}" ] && [ "${CPU_PARTITION}" != "none" ] && cpu_part="--partition=${CPU_PARTITION}"
  [ -n "${GPU_PARTITION:-}" ] && [ "${GPU_PARTITION}" != "none" ] && gpu_part="--partition=${GPU_PARTITION}"
  local cpu="${cpu_part} --nodes=1 --ntasks=1"
  local gpu="${gpu_part} --nodes=1 --ntasks=1 --gpus=1"
  local gpu_train="${gpu_part} --nodes=1 --ntasks=1 --gpus=${TRAIN_GPUS}"
  case "$1" in
    setup-env)      echo "${gpu} --cpus-per-task=8  --mem=32G  --time=02:00:00" ;;
    preflight)      echo "${cpu} --cpus-per-task=1  --mem=4G   --time=00:10:00" ;;
    fetch-labels)   echo "${cpu} --cpus-per-task=4  --mem=8G   --time=03:00:00" ;;
    resolve-scenes) echo "${cpu} --cpus-per-task=2  --mem=8G   --time=06:00:00" ;;
    download)       echo "${cpu} --cpus-per-task=4  --mem=8G   --time=12:00:00 --array=0-$((DOWNLOAD_TASKS-1))%${DOWNLOAD_CONCURRENCY}" ;;
    preprocess)     echo "${cpu} --cpus-per-task=8  --mem=48G  --time=10:00:00 --array=0-$((PREPROCESS_TASKS-1))" ;;
    split)          echo "${cpu} --cpus-per-task=8  --mem=64G  --time=01:00:00" ;;
    train)          echo "${gpu_train} --cpus-per-task=${TRAIN_CPUS} --mem=${TRAIN_MEM} --time=1-00:00:00 --requeue" ;;
    evaluate)       echo "${gpu} --cpus-per-task=16 --mem=160G --time=06:00:00" ;;
    mdl)            echo "${gpu} --cpus-per-task=16 --mem=160G --time=12:00:00 --array=0-$((MDL_TASKS-1))" ;;
    diagnose)       echo "${gpu} --cpus-per-task=8  --mem=96G  --time=02:00:00" ;;
    baseline-lr)    echo "${cpu} --cpus-per-task=8  --mem=64G  --time=02:00:00" ;;
    mdl-fit|report) echo "${cpu} --cpus-per-task=2  --mem=8G   --time=00:30:00" ;;
    fetch-enh|mf-check|aviris-fetch) echo "${cpu} --cpus-per-task=4 --mem=32G --time=06:00:00" ;;
    aviris-eval)    echo "${gpu} --cpus-per-task=16 --mem=240G --time=06:00:00" ;;
    *) echo "unknown stage $1" >&2; return 2 ;;
  esac
}

# Common sbatch flags. Slurm spools the batch script under /var/spool/slurmd, so the job cannot find
# this directory from $0; pass it explicitly. No --account: Prism resolves the caller's Grace
# allocation, and a stale account/partition pairing makes Slurm reject otherwise valid jobs.
sbatch_common() {
  local -a a=(--parsable "--chdir=${CH4HSI_REPO}" "--export=ALL,CH4HSI_HPC_DIR=${CH4HSI_HPC_DIR},CH4HSI_REPO=${CH4HSI_REPO},CH4HSI_DATA=${CH4HSI_DATA}")
  [ -z "${CH4HSI_MAIL:-}" ] || a+=("--mail-user=${CH4HSI_MAIL}" --mail-type=FAIL)
  printf '%s\n' "${a[@]}"
}

# ---------------------------------------------------------------- python environment
# GDAL/pyproj may disagree on their data dir when a stale PROJ_LIB comes in via --export=ALL.
configure_proj_data() {
  local prefix=$1 c
  for c in "${prefix}/share/proj" "${prefix}/Library/share/proj"; do
    if [ -r "${c}/proj.db" ]; then export PROJ_DATA="${c}" PROJ_LIB="${c}"; return 0; fi
  done
  c="$("${prefix}/bin/python" -c 'from pathlib import Path; import pyproj; p=Path(pyproj.__file__).parent/"proj_dir"/"share"/"proj"; print(p if (p/"proj.db").is_file() else "")' 2>/dev/null || true)"
  [ -n "${c}" ] && export PROJ_DATA="${c}" PROJ_LIB="${c}"
  return 0
}

ch4hsi_activate() {
  local had_u=0; [[ $- == *u* ]] && had_u=1
  set +u   # module / conda activation scripts are not nounset-safe
  if command -v module >/dev/null 2>&1; then
    module load miniforge 2>/dev/null || module load Conda 2>/dev/null || true
  fi
  local prefix; prefix="$(ch4hsi_env_prefix)"
  if [ ! -x "${prefix}/bin/python" ]; then
    echo "FATAL: no ch4hsi environment for $(uname -m) at ${prefix}." >&2
    echo "  aarch64 (grace): sbatch slurm/00_setup_env.sbatch   |   x86_64 (login): bash slurm/setup_login_env.sh" >&2
    exit 2
  fi
  # Use the env's interpreter directly (no `conda activate`), and keep ~/.local out of the path.
  unset PYTHONHOME PYTHONPATH
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  export PATH="${prefix}/bin:${PATH}"
  hash -r
  configure_proj_data "${prefix}"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  if [ -n "${SLURM_JOB_ID:-}" ] && [ -d /lscratch ]; then     # node-local NVMe (1 TB on grace nodes)
    export CH4HSI_LOCAL="/lscratch/${USER}/${SLURM_JOB_ID}"
    mkdir -p "${CH4HSI_LOCAL}" 2>/dev/null || export CH4HSI_LOCAL="${TMPDIR:-/tmp}"
  else
    export CH4HSI_LOCAL="${TMPDIR:-/tmp}"
  fi
  cd "${CH4HSI_REPO}" || exit 1
  if [ "${had_u}" = 1 ]; then set -u; fi
}

# ch4 <stage> [args]: run a stage with the site config, data root and node-local scratch
ch4() {
  python -m ch4hsi "$@" --config "${CH4HSI_CONFIG}" --set "paths.data_root=${CH4HSI_DATA}" \
    --set "paths.local_scratch=${CH4HSI_LOCAL:-/tmp}" ${CH4HSI_EXTRA_SETS:-}
}

# ---------------------------------------------------------------- egress
# Prism's x86 compute image has an older curl than Grace; only use --retry-all-errors if supported.
require_egress() {
  local url=$1
  if ! curl -L --fail --silent --show-error --max-time 25 --range 0-0 --output /dev/null "${url}"; then
    echo "FATAL: ${url} is unreachable from $(hostname)." >&2
    echo "Run the download stages on an egress-capable login/transfer host (slurm/download_on_login.sh)" >&2
    echo "or contact NCCS support about compute-node egress." >&2
    return 3
  fi
  echo "   reachable: ${url}"
}

echo "== $(date +"%Y-%m-%dT%H:%M:%S%z") host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} =="
echo "   repository: ${CH4HSI_REPO}"
echo "   data:       ${CH4HSI_DATA}"
