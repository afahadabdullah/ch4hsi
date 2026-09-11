# NCCS Prism jobs (GH200 / `grace` partition)

Conventions follow the StormGrid `scripts/hpc` setup: PanFS data root, `miniforge` module, no `--account`
(Prism resolves your Grace allocation), `--chdir`/`--export` passed explicitly, a network preflight before
any download, `PYTHONNOUSERSITE=1` + explicit PROJ data, flock-serialised one-time env creation, and
idempotent / resumable stages.

| | |
|---|---|
| Node | 1× NVIDIA GH200 (H100 96 GB) · 72-core Grace CPU (**aarch64**) · 480 GB RAM · 1 TB `/lscratch` |
| Data root | `/panfs/ccds02/nobackup/people/$USER/ch4hsi` (fallback `/explore/nobackup/people/$USER/ch4hsi`) |
| Envs | `$CH4HSI_DATA/.envs/ch4hsi-aarch64` (built *on a grace node* by `00_setup_env.sbatch`, torch cu128) · optional `ch4hsi-x86_64` for gpulogin1 downloads |
| Config | `configs/gh200.yaml` layered over `configs/default.yaml` (batch 96, bf16, 24 loader workers, 512-px inference tiles) |

## Run (bash; Prism's default login shell is tcsh)

```bash
cd ~/ch4hsi
bash slurm/submit_all.sh                     # setup-env → preflight → labels → scenes → download → preprocess
                                             #   → split → train → evaluate → mdl → mdl-fit → diagnose → report
sbatch slurm/smoke.sbatch                    # synthetic end-to-end test (after setup-env has finished once)
bash slurm/submit_stage.sh train             # one stage with its standard resources
bash slurm/submit_all.sh --from preprocess   # resume the chain from any stage
```

If `01_preflight` fails on egress, build the x86 env and download from the login node, then continue:

```bash
bash slurm/setup_login_env.sh
tmux new -s ch4dl 'bash slurm/download_on_login.sh'
bash slurm/submit_all.sh --from preprocess
```

Interactive work on a GH200 from tcsh:
`srun -p grace --gpus=1 -c 16 --mem=64G -t 2:00:00 --pty tcsh` → `source slurm/activate.csh`.

## Resources (edit `stage_resources` in `common.sh`)

| stage | partition | GPUs | cores | mem | time | array |
|---|---|---|---|---|---|---|
| setup-env | grace | 1 | 8 | 32G | 2 h | |
| preflight | grace | – | 1 | 4G | 10 min | |
| fetch-labels / resolve-scenes | grace | – | 4 / 2 | 8G | 3 h / 6 h | |
| download | grace | – | 4 | 8G | 12 h | 8, `%4` |
| preprocess | grace | – | 8 | 48G | 10 h | 16 |
| split | grace | – | 8 | 64G | 1 h | |
| train | grace | 1 | 32 | 240G | 24 h, `--requeue` | |
| evaluate | grace | 1 | 16 | 160G | 6 h | |
| mdl | grace | 1 | 16 | 160G | 12 h | 4 |
| diagnose | grace | 1 | 8 | 96G | 2 h | |
| mdl-fit / report | grace | – | 2 | 8G | 30 min | |
| baseline-lr (optional) | grace | – | 8 | 64G | 2 h | |

CPU-only stages request no GPU on `grace`. To keep them off GPU nodes set `CPU_PARTITION=grace-cpuonly`
(same aarch64 env) in `slurm/site.env`; `CPU_PARTITION=compute` (x86) also works for every stage except
train/evaluate/mdl, using the x86 env.

## Files

| file | purpose |
|---|---|
| `common.sh` | paths, partitions, per-stage resources, env activation, egress check, `ch4` wrapper |
| `site.env.example` | optional overrides → copy to `site.env` (git-ignored) |
| `00_setup_env.sbatch` | one-time aarch64 env on a GH200 (conda-forge + torch cu128 wheel, fallback conda-forge `pytorch-gpu`), verifies CUDA/bf16 |
| `01_preflight.sbatch` | Earthdata / LP DAAC / GHG Center reachability, `~/.netrc`, env import, free space |
| `job.sbatch` | generic stage runner (cleans its `/lscratch` dir on exit) |
| `submit_all.sh`, `submit_stage.sh` | dependency chain / single stage |
| `setup_login_env.sh`, `download_on_login.sh` | x86 login-node fallback for downloads |
| `smoke.sbatch` | tests + synthetic end-to-end run on one GH200 |
| `activate.csh` | interactive tcsh activation |
