# ch4hsi — methane plume detection in EMIT hyperspectral radiance

Matched-filter baseline + U-Net segmentation on NASA EMIT L1B radiance, labelled with EMIT L2B methane plume
complexes, evaluated with precision/recall, IoU, plume-level and scene-level metrics, and a minimum detection limit from
synthetic plume injection. Built to run end-to-end on **NCCS Prism** with Slurm. See [PLAN.md](PLAN.md) for the design.

```
L1B radiance ──► column-wise matched filter (ppm·m) ──► 18 sensor-agnostic features ──► GLT ortho ──► U-Net
      │                                                                                   │
EMIT L2B plume complexes (COG/GeoJSON) ──► rasterised labels ───────────────────────► metrics + MDL (injection)
```

## Quick start on Prism

```bash
ssh adaptlogin.nccs.nasa.gov        # then: ssh gpulogin1  (Prism login)
git clone <your-repo> ~/ch4hsi && cd ~/ch4hsi
vi slurm/env.sh                     # paths, partitions (defaults: /explore/nobackup/people/$USER, partition "compute")

# Earthdata credentials (https://urs.earthdata.nasa.gov) for LP DAAC downloads
echo "machine urs.earthdata.nasa.gov login <USER> password <PASS>" >> ~/.netrc && chmod 600 ~/.netrc

bash slurm/setup_env.sh             # conda env + torch (cu126) + mag1c LUT + this package
sbatch -p compute -G1 -c 4 --mem=32G -t 01:00:00 slurm/smoke.sbatch   # synthetic end-to-end test, ~10 min
bash slurm/check_network.sh         # can compute nodes reach Earthdata?

bash slurm/submit_all.sh            # full chain: labels → scenes → download → preprocess → split → train → evaluate → mdl → report
# if compute nodes have no internet:
tmux new -s dl 'bash slurm/download_on_login.sh'   &&   bash slurm/submit_all.sh --from preprocess
```

Results land in `$CH4HSI_DATA/runs/<run_name>/`: `REPORT.md` (tables, figures, auto-filled resume bullets),
`metrics_test.json`, `mdl.json`, `per_plume_test.csv`, `figures/`.

## Stages (`python -m ch4hsi <stage> --config ... --set key=value`)

| stage | what it does | output |
|---|---|---|
| `fetch-labels` | plume-complex catalogue (LP DAAC `EMITL2BCH4PLM` or GHG Center STAC) + COG/GeoJSON | `catalog/plumes.csv` |
| `resolve-scenes` | CMR search for the L1B scenes of sampled plumes; plume-free negatives over O&G regions | `catalog/scenes.csv` |
| `download` | resumable, size-checked L1B download (Slurm array) | `raw/*.nc` |
| `preprocess` | column-wise MF, features, GLT ortho, label rasterisation (Slurm array) | `scenes/<id>/*.npy` |
| `split` | geo-blocked train/val/test + normalisation stats | `splits/` |
| `train` | U-Net, AMP, early stopping, resumes from `last.pt` | `runs/<run>/best.pt` |
| `evaluate` | val-tuned thresholds; test metrics for U-Net and MF baselines; figures | `metrics_test.json` |
| `mdl`, `mdl-fit` | synthetic injection into radiance → POD curves → MDL50/90 | `mdl.json`, `pod_curve.png` |
| `report` | REPORT.md | |
| `fetch-enh`, `mf-check` | optional: compare our MF with EMIT L2B CH4ENH | `catalog/mf_crosscheck.json` |
| `aviris-fetch`, `aviris-eval` | experimental cross-sensor test on AVIRIS-NG | `aviris_summary.json` |

Useful overrides: `--set run_name=unet_synth --set train.synth_aug.enabled=true`, `--set features=[mf,mf_snr]`
(MF-only ablation), `--set scenes.max_positive_scenes=150`, `--set labels.source=ghgc_stac`,
`--set train.model=smp` (needs `segmentation_models_pytorch`). With `submit_all.sh`, pass them through
`CH4HSI_EXTRA_SETS="--set ..."`.

## Prism notes

- Partitions (NCCS docs): `compute` (4× V100 32 GB, x86_64, default), `dgx` (8× A100), `grace` (GH200, **aarch64** —
  build a separate env there: `CH4HSI_ENV=.../ch4hsi-arm TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash slurm/setup_env.sh`
  inside an interactive grace job, then set `GPU_PARTITION=grace` and `CH4HSI_ENV` accordingly).
- Training stages scenes to node-local `/lscratch` (`train.stage_to_local`), removed at job exit.
- `train` is submitted with `--requeue` and resumes from `last.pt`.
- Re-running `mdl`: delete old `runs/<run>/mdl_records*.csv` first (the fit merges every shard file it finds).

## Tests

```bash
pytest -q tests        # physics (MF recovers injected ppm·m, GLT, plume mass conservation, units), metrics, numpy pipeline
```

## Caveats

- Labels are the EMIT team's reviewed plume complexes: the model learns that delineation, and uncatalogued real plumes
  count as false positives (precision is a lower bound; plume-free scenes give a clean false-alarm rate).
- MDL depends on the assumed wind speed (default 3 m/s) and the idealised plume model; report it with those assumptions.
- The CH4 absorption LUT is taken from the `mag1c` package (BSD-3, Foote et al. 2020).

## References

Thompson et al. 2016 (column-wise MF, AVIRIS-NG); Foote et al. 2020 (mag1c); Thorpe et al. 2023 (EMIT methane);
Varon et al. 2018 (IME flux); Růžička et al. 2023 (STARCOP, ML plume segmentation); EMIT L2B GHG User Guide.
