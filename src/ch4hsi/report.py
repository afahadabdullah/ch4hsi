"""Stage 8: assemble REPORT.md (metrics table, MDL, figures, resume bullets) for a run."""
from __future__ import annotations

from pathlib import Path

from .config import Cfg, run_dir
from .utils import get_logger, load_json

log = get_logger(__name__)


def _f(v, p=3):
    return "–" if v is None or v != v else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def _ci(c):
    return f"{c[0]:.0f}–{c[1]:.0f}" if c else "–"


def _cir(c):
    return f"{c[0]:.2f}–{c[1]:.2f}" if c else "–"


def run(cfg: Cfg):
    out = run_dir(cfg)
    m = load_json(out / "metrics_test.json")
    mdl = load_json(out / "mdl.json") if (out / "mdl.json").exists() else None
    split = load_json(Path(cfg.paths.splits) / "splits.json")["summary"]
    L = []
    L.append(f"# Methane plume detection — run `{cfg.run_name}`\n")
    L.append("## Data\n")
    L.append("| split | scenes | with plumes | plume-free | plume pixels |\n|---|---|---|---|---|")
    for k in ("train", "val", "test"):
        s = split[k]
        L.append(f"| {k} | {s['scenes']} | {s['pos']} | {s['neg']} | {s['pos_px']} |")
    L.append("\n## Test metrics (thresholds frozen on validation)\n")
    L.append("| method | threshold | pixel P | pixel R | F1 | IoU | AP | plume R | plume P | scene AUROC | FA / 1000 km² |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    names = {"model": m.get("model_label", "U-Net"), "mf": "Matched filter (val-tuned)", "mf_fixed": "Matched filter (fixed)"}
    for k, lab in names.items():
        r = m[k]
        t = m["thresholds"][k]["threshold"]
        L.append(f"| {lab} | {_f(t, 2 if k == 'model' else 0)} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | "
                 f"{_f(r['iou'])} | {_f(r['ap'])} | {_f(r['plume_recall'])} | {_f(r['plume_precision'])} | "
                 f"{_f(r['scene_auroc'])} | {_f(r['false_alarms_per_1000km2_neg'], 2)} |")
    if mdl:
        L.append(f"\n## Minimum detection limit (synthetic injection, U = {mdl['wind_ms']} m/s, "
                 f"{mdl['n_injections']} injections in {mdl['n_scenes']} plume-free test scenes)\n")
        L.append("| method | MDL50 (kg/h) | 90% CI | MDL90 (kg/h) | 90% CI | MDL50 peak (ppm·m) |\n|---|---|---|---|---|---|")
        for k, lab in names.items():
            f = mdl[k].get("flux", {})
            pk = mdl[k].get("peak_ppmm", {})
            ci50, ci90 = f.get("mdl50_ci"), f.get("mdl90_ci")
            L.append(f"| {lab} | {_f(f.get('mdl50'), 0)} | {_ci(ci50)} | "
                     f"{_f(f.get('mdl90'), 0)} | {_ci(ci90)} | {_f(pk.get('mdl50'), 0)} |")
        n = mdl["noise"]
        L.append(f"\nMedian column noise-equivalent enhancement σ = {n['sigma_median_ppmm']:.0f} ppm·m; "
                 f"analytic 3σ / 9-pixel MDL ≈ {n['analytic_mdl_kgph_3sigma_9px']:.0f} kg/h.")
    ci_rows = [(lab, m[k].get("ci90", {})) for k, lab in names.items() if m[k].get("ci90")]
    if ci_rows:
        L.append("\n90% scene-bootstrap confidence intervals:\n")
        L.append("| method | F1 | IoU | plume recall | plume precision |\n|---|---|---|---|---|")
        for lab, c in ci_rows:
            L.append(f"| {lab} | {_cir(c.get('f1'))} | {_cir(c.get('iou'))} | {_cir(c.get('plume_recall'))} | "
                     f"{_cir(c.get('plume_precision'))} |")
    L.append("\n## Figures\n")
    for fig in sorted((out / "figures").glob("*.png")):
        L.append(f"![{fig.stem}](figures/{fig.name})")
    if (out / "diagnostics" / "DIAGNOSTICS.md").exists():
        L.append("\nFull diagnostic suite (training curves, threshold sweeps, ROC, calibration, error maps, false-alarm "
                 "analysis): [diagnostics/DIAGNOSTICS.md](diagnostics/DIAGNOSTICS.md)")
    # resume bullets with the numbers filled in
    u, b = m["model"], m["mf"]
    L.append("\n## Resume bullets (auto-filled)\n")
    L.append(f"- Built an end-to-end methane plume detection pipeline on NASA EMIT L1B imaging-spectrometer radiance "
             f"(column-wise matched filter → sensor-agnostic spectral features → U-Net segmentation), trained on "
             f"{split['train']['pos']} scenes with EMIT L2B plume-complex labels on NCCS Prism GPUs.")
    L.append(f"- On geographically held-out scenes: pixel F1 {_f(u['f1'], 2)} / IoU {_f(u['iou'], 2)} for the U-Net vs "
             f"{_f(b['f1'], 2)} / {_f(b['iou'], 2)} for the tuned matched-filter baseline; plume-level recall "
             f"{_f(u['plume_recall'], 2)} at precision {_f(u['plume_precision'], 2)}.")
    if mdl and mdl["model"].get("flux"):
        L.append(f"- Quantified a minimum detection limit of ~{mdl['model']['flux']['mdl50']:.0f} kg/h (50% POD; "
                 f"{mdl['model']['flux']['mdl90']:.0f} kg/h at 90%) at {mdl['wind_ms']} m/s via Beer–Lambert plume "
                 f"injection into radiance, vs {_f(mdl['mf'].get('flux', {}).get('mdl50'), 0)} kg/h for the matched filter.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", out / "REPORT.md")
