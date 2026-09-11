# Site settings for NCCS Prism (ADAPT). Sourced by every job and by submit_all.sh — edit once.
# Prism partitions (NCCS docs): compute = V100 x86_64 nodes (default), dgx = 8x A100,
# grace = GH200 (aarch64!), grace-cpuonly = Grace CPU (aarch64). A conda env built on x86 login
# nodes only runs on compute/dgx; for grace build a second env on a grace node (see README).

export CH4HSI_REPO=${CH4HSI_REPO:-$HOME/ch4hsi}
export CH4HSI_DATA=${CH4HSI_DATA:-/explore/nobackup/people/$USER/ch4hsi}
export CH4HSI_ENV=${CH4HSI_ENV:-/explore/nobackup/people/$USER/envs/ch4hsi}
export CH4HSI_CONFIG=${CH4HSI_CONFIG:-$CH4HSI_REPO/configs/default.yaml}

export CPU_PARTITION=${CPU_PARTITION:-compute}
export GPU_PARTITION=${GPU_PARTITION:-compute}
export SBATCH_ACCOUNT_OPT=${SBATCH_ACCOUNT_OPT:-}        # e.g. "-A s1234" if your allocation needs one

export DOWNLOAD_TASKS=${DOWNLOAD_TASKS:-8}                 # array size for the L1B download
export DOWNLOAD_CONCURRENCY=${DOWNLOAD_CONCURRENCY:-4}     # simultaneous array tasks (be nice to LP DAAC)
export PREPROCESS_TASKS=${PREPROCESS_TASKS:-16}
export MDL_TASKS=${MDL_TASKS:-4}

ch4hsi_activate() {
    local had_u=0; [[ $- == *u* ]] && had_u=1
    set +u   # module / conda activation scripts are not nounset-safe
    module purge >/dev/null 2>&1 || true
    module load miniforge >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CH4HSI_ENV"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
    # node-local NVMe for training data staging
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -d /lscratch ]; then
        export CH4HSI_LOCAL=/lscratch/$USER/$SLURM_JOB_ID
        mkdir -p "$CH4HSI_LOCAL" 2>/dev/null || export CH4HSI_LOCAL=${TMPDIR:-/tmp}
    else
        export CH4HSI_LOCAL=${TMPDIR:-/tmp}
    fi
    cd "$CH4HSI_REPO" || exit 1
    if [ "$had_u" = 1 ]; then set -u; fi
}

# ch4hsi <stage> with the site config and node-local scratch
ch4() {
    python -m ch4hsi "$@" --config "$CH4HSI_CONFIG" --set "paths.data_root=$CH4HSI_DATA" \
        --set "paths.local_scratch=$CH4HSI_LOCAL" ${CH4HSI_EXTRA_SETS:-}
}
