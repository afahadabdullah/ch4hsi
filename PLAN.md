# Project plan — Hyperspectral methane plume detection (EMIT / AVIRIS-NG)

**Goal.** A small, credible, end-to-end project on public data: detect methane plumes in EMIT L1B radiance with a
physics baseline (column-wise matched filter) and a learned segmentation model, and report precision/recall, IoU and an
estimated minimum detection limit (MDL). Everything runs as Slurm jobs on NCCS Prism.

## 1. Data

| role | product | access | notes |
|---|---|---|---|
| input | **EMIT L1B radiance** `EMITL1BRAD` v001 (`EMIT_L1B_RAD_*.nc`, 285 bands, 1280×1242 px, 60 m) | LP DAAC via `earthaccess` (Earthdata login) | ~1.8 GB/scene; GLT for orthorectification is in the file |
| labels | **EMIT L2B CH4 plume complexes** `EMITL2BCH4PLM` v002 (COG + GeoJSON per complex) | LP DAAC | manually reviewed complexes; GeoJSON lists source scenes |
| labels (alt) | US GHG Center STAC `emit-ch4plume-v1` | `https://earth.gov/ghgcenter/api/stac` (public) | same complexes (v001 COGs), no scene list → matched by time + bbox |
| check | EMIT L2B CH4 enhancement `EMITL2BCH4ENH` | LP DAAC | optional: validate our MF against the operational map |
| transfer test | AVIRIS-NG L1B radiance `AVIRIS-NG_L1B_radiance_2095` + ORNL DAAC plume lists | ORNL DAAC | optional / stretch |

**Scene set (default).** 300 scenes that contain ≥1 plume complex (sampled across the catalogue) + 150 plume-free
"hard negative" scenes over oil/gas/coal regions (Permian, Turkmenistan, Algeria, Iran, W Kazakhstan, Four Corners,
Libya/Egypt, Shanxi), cloud cover ≤ 40 %. ≈ 0.8 TB raw, ≈ 70 GB preprocessed. Halve `scenes.max_positive_scenes` if
the `/explore/nobackup` quota is tight.

**Label semantics.** A pixel is positive if it lies inside a plume complex (COG valid & > `labels.min_ppmm`) after
reprojection onto the scene's GLT grid. Pixels outside catalogued complexes are negatives — unlabelled real plumes
make measured precision a *lower bound*; the plume-free scenes give a clean false-alarm rate.

## 2. Methods

1. **Target spectrum.** Unit CH4 absorption `k(λ) = ∂ln L/∂(ppm·m)` from the mag1c high-resolution LUT, convolved with
   each scene's band centres/FWHM (2122–2488 nm window, good bands only).
2. **Baseline: column-wise matched filter** (per detector column, as in the EMIT operational algorithm): mean/covariance
   per column with shrinkage, target `t = μ ⊙ k`, albedo correction, two-pass background re-estimation that masks
   positive outliers. Output in ppm·m and a per-column noise-equivalent σ.
3. **Features (sensor-agnostic, 18 channels):** MF (ppm·m), MF/σ, 12 binned SWIR residuals `L/μ − 1`, albedo,
   visible RGB. All orthorectified with the GLT and stored per scene as memory-mapped `.npy`.
4. **Model: U-Net** (≈7.8 M params, base 32, depth 4) on 128×128 tiles, 50 % plume-centred sampling, BCE(pos_weight) +
   Dice, AdamW + cosine LR, AMP, resumable checkpoints. Optional ablation: physics-consistent synthetic plume
   augmentation in feature space (`train.synth_aug.enabled`).
5. **Split:** 1° lat/lon blocks keyed by plume location (repeat overpasses of one facility never straddle train/test),
   70/15/15. Normalisation stats from train only.

## 3. Evaluation (test scenes; thresholds frozen on validation)

- **Pixel:** precision, recall, F1, IoU, average precision (PR curve).
- **Plume-level:** recall = labelled complexes hit by a prediction; precision = predicted components touching a complex.
- **Scene-level:** AUROC for "scene contains a plume" (max score) and false alarms per 1000 km² on plume-free scenes.
- **Baselines:** MF with val-tuned threshold and a fixed 1000 ppm·m threshold, same smoothing/min-size post-processing.
- **Recall vs plume strength** (per-plume csv + figure).

## 4. Minimum detection limit

Inject Gaussian plumes (Briggs class-D spread, mass-conserving, turbulent texture) of known source rate Q (100–5000 kg/h,
U = 3 m/s) into the **radiance** of plume-free test scenes with Beer–Lambert `L' = L·exp(k·ΔX)`, re-run the full chain,
and record detection with the frozen thresholds. Fit a logistic POD in log Q → **MDL50 / MDL90 with bootstrap CIs** for
the model and the MF baseline; also POD vs peak ppm·m and an analytic noise-based MDL
(`Q = U·IME/L` for a 9-pixel 3σ plume) as a sanity check.

## 5. Compute plan on NCCS Prism

| stage | where | resources | wall-time (est.) |
|---|---|---|---|
| fetch-labels, resolve-scenes | CPU (or login node) | 2–4 cores | < 3 h |
| download (array ×8, 4 concurrent) | CPU (or login node) | 4 cores | 6–12 h for 0.8 TB |
| preprocess (array ×16) | CPU | 8 cores, 48 GB | ~1–2 h |
| split | CPU | 4 cores | minutes |
| train | 1 GPU (V100/A100/H100) | 10 cores, 96 GB | 1–3 h |
| evaluate | 1 GPU | | < 1 h |
| mdl (array ×4) + mdl-fit | 1 GPU each | | 1–2 h |

Run `slurm/check_network.sh` first: if compute nodes lack outbound internet, run stages 1–2 on the login node with
`slurm/download_on_login.sh`, then `slurm/submit_all.sh --from preprocess`.

## 6. Milestones

1. Env + smoke test on synthetic EMIT-format data (`slurm/smoke.sbatch`) — day 1.
2. Catalogue + 50-scene pilot download; `mf-check` against EMIT L2B CH4ENH (expect r > 0.8) — days 2–3.
3. Full download/preprocess; split; first U-Net run — week 1.
4. Evaluation + MDL; ablations (no-RGB, MF-only input, synth-aug) — week 2.
5. Write-up: REPORT.md figures → README / short blog post; optional AVIRIS-NG transfer test.

## 7. Risks & mitigations

- *Label noise / unlabelled plumes* → plume-free scenes for clean FA rate; report precision as a lower bound.
- *Our MF ≠ operational MF* → `mf-check`; labels come from complexes, not from our MF.
- *Class imbalance* → plume-centred tile sampling, pos_weight, Dice; thresholds tuned on val.
- *Leakage across overpasses* → geo-block split.
- *Storage quota* → reduce scene count or `preprocess.delete_raw` (keep raw for test negatives needed by MDL).
- *Prism specifics* (partition names, internet on compute nodes, aarch64 Grace nodes) → all in `slurm/env.sh`.

## 8. Resume bullets (numbers auto-filled into REPORT.md)

- Built an end-to-end methane plume detection pipeline on NASA EMIT imaging-spectrometer radiance (column-wise
  matched filter → U-Net segmentation) trained on EMIT L2B plume-complex labels on NCCS GPU clusters.
- On geographically held-out scenes: pixel F1/IoU of X/Y vs A/B for the matched-filter baseline; plume-level recall R at
  precision P; scene AUROC Z.
- Estimated a minimum detection limit of ~N kg/h (50 % POD at 3 m/s) via Beer–Lambert synthetic plume injection into
  L1B radiance, vs M kg/h for the matched filter.
