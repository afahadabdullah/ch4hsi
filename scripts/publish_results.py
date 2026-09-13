#!/usr/bin/env python
"""Publish a finished run's numbers and figures into the repo (docs/results/<run>/ + README block).

    python scripts/publish_results.py --run unet_v1 -c configs/x86_v100.yaml          # copy + rewrite README
    python scripts/publish_results.py --run unet_v1 --check                           # print, write nothing

Run outputs live under $CH4HSI_DATA (not in git). This copies the small, presentable pieces into the
repo so the README shows real results:

    docs/results/<run>/REPORT.md            the run's report
    docs/results/<run>/DIAGNOSTICS.md       the diagnostics index
    docs/results/<run>/*.json               metrics_test.json, mdl.json, thresholds.json, splits summary
    docs/results/<run>/figures/*.png        evaluate's figures (PR curve, examples, POD)
    docs/results/<run>/diagnostics/*.png    the selected diagnostic panels (+ a few error maps)

and rewrites the block between the `ch4hsi:results` markers in README.md with a metrics table (90 %
scene-bootstrap CIs), the MDL line, and links. The block is created if the markers are absent.
Re-running after a new run replaces the block in place, so the README never drifts from the numbers.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ch4hsi.config import load_config, run_dir  # noqa: E402
from ch4hsi.utils import get_logger, load_json  # noqa: E402

log = get_logger("publish_results")
START, END = "<!-- ch4hsi:results:start -->", "<!-- ch4hsi:results:end -->"
DEFAULT_PANELS = ["02_threshold_sweep", "03_roc", "06_per_scene_iou", "07_plume_detection",
                  "08_false_alarms", "11_mdl"]
NAMES = {"model": None, "mf": "Matched filter (val-tuned)", "mf_fixed": "Matched filter (1000 ppm·m)"}


def _f(v, p=3):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.{p}f}"


def _ci(c, p=2):
    return f"{c[0]:.{p}f}–{c[1]:.{p}f}" if c else "–"


def _pick_maps(map_dir: Path, per_kind: int) -> list[Path]:
    """Up to `per_kind` error maps of each kind (best / worst / false-alarm)."""
    if per_kind <= 0 or not map_dir.exists():
        return []
    out = []
    for kind in ("best", "worst", "false-alarm"):
        out += sorted(map_dir.glob(f"{kind}_*.png"))[:per_kind]
    return out


def _rewrite_diagnostics(path: Path, kept_maps: set[str]) -> str:
    """Point image links at the copied layout and drop links to maps that were not copied."""
    lines = []
    for ln in path.read_text().splitlines():
        if "](maps/" in ln:
            if Path(ln.split("](maps/", 1)[1].rstrip(")")).name not in kept_maps:
                continue
            ln = ln.replace("](maps/", "](diagnostics/maps/")
        elif ln.startswith("![") and "](" in ln and "](diagnostics/" not in ln:
            ln = ln.replace("](", "](diagnostics/", 1)
        lines.append(ln)
    return "\n".join(lines) + "\n"


def copy_tree(src: Path, dst: Path, pattern="*.png", limit=None, check=False) -> list[Path]:
    files = sorted(src.glob(pattern))[:limit] if src.exists() else []
    if not check:
        for f in files:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.name)
    return files


def results_block(cfg, out: Path, run: str, doc_rel: str, panels: list[str]) -> str:
    m = load_json(out / "metrics_test.json")
    label = m.get("model_label", "U-Net")
    mdl = load_json(out / "mdl.json") if (out / "mdl.json").exists() else None
    sp = Path(cfg.paths.splits) / "splits.json"
    summary = load_json(sp)["summary"] if sp.exists() else None
    L = [START, "", "## Results", "",
         f"Run `{run}` on EMIT L1B radiance with EMIT L2B plume-complex labels; thresholds frozen on the "
         f"validation scenes, metrics on geographically held-out test scenes "
         f"({m.get('n_test_scenes', '?')} scenes)."]
    if summary:
        L += ["", "| split | scenes | with plumes | plume-free | plume pixels |", "|---|---|---|---|---|"]
        L += [f"| {k} | {s['scenes']} | {s['pos']} | {s['neg']} | {s['pos_px']:,} |"
              for k, s in ((k, summary[k]) for k in ("train", "val", "test") if k in summary)]
    L += ["", "| method | pixel F1 | 90% CI | IoU | AP | plume recall | 90% CI | plume precision | scene AUROC | FA / 1000 km² |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, nm in NAMES.items():
        r = m.get(k)
        if not r:
            continue
        ci = r.get("ci90", {})
        L.append(f"| {nm or label} | {_f(r['f1'])} | {_ci(ci.get('f1'))} | {_f(r['iou'])} | {_f(r['ap'])} | "
                 f"{_f(r['plume_recall'])} | {_ci(ci.get('plume_recall'))} | {_f(r['plume_precision'])} | "
                 f"{_f(r['scene_auroc'])} | {_f(r['false_alarms_per_1000km2_neg'], 2)} |")
    L.append("")
    L.append("Plume recall counts labelled plume complexes with at least one detected pixel; false alarms are "
             "predicted components on plume-free scenes. Uncatalogued real plumes count as false positives, so "
             "precision is a lower bound.")
    if mdl:
        f_m = (mdl.get("model") or {}).get("flux", {})
        f_b = (mdl.get("mf") or {}).get("flux", {})
        if f_m.get("mdl50"):
            L += ["", f"**Minimum detection limit** ({mdl['n_injections']:,} synthetic injections into the radiance of "
                      f"{mdl['n_scenes']} plume-free test scenes, U = {mdl['wind_ms']} m/s): "
                      f"MDL50 ≈ {f_m['mdl50']:.0f} kg/h ({_ci(f_m.get('mdl50_ci'), 0)}), MDL90 ≈ "
                      f"{f_m.get('mdl90', float('nan')):.0f} kg/h for {label}, vs "
                      f"{_f(f_b.get('mdl50'), 0)} kg/h (MDL50) for the matched filter. Median column noise-equivalent "
                      f"σ = {mdl['noise']['sigma_median_ppmm']:.0f} ppm·m."]
    L += ["", f"Full report: [{doc_rel}/REPORT.md]({doc_rel}/REPORT.md) · "
              f"diagnostics: [{doc_rel}/DIAGNOSTICS.md]({doc_rel}/DIAGNOSTICS.md)", ""]
    for name in panels:
        if (out / "diagnostics" / f"{name}.png").exists():
            L.append(f"![{name}]({doc_rel}/diagnostics/{name}.png)")
    L += ["", END]
    return "\n".join(L)


def splice(readme: Path, block: str) -> str:
    text = readme.read_text() if readme.exists() else "# ch4hsi\n"
    if START in text and END in text:
        head, rest = text.split(START, 1)
        return head + block + rest.split(END, 1)[1]
    # first run: insert before the notes/tests tail if we recognise it, else append
    for anchor in ("## HPC cloud notes", "## Prism notes", "## Tests", "## Caveats", "## References"):
        if f"\n{anchor}" in text:
            head, tail = text.split(f"\n{anchor}", 1)
            return f"{head}\n{block}\n\n{anchor}{tail}"
    return text.rstrip() + "\n\n" + block + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run_name under $CH4HSI_DATA/runs")
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--set", "-s", action="append", default=[])
    ap.add_argument("--docs", default=str(REPO / "docs" / "results"))
    ap.add_argument("--panels", nargs="*", default=DEFAULT_PANELS, help="diagnostic figures to embed in the README")
    ap.add_argument("--maps", type=int, default=1, help="error maps to copy per kind (best/worst/false-alarm)")
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    ap.add_argument("--check", action="store_true", help="print what would happen, write nothing")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set + [f"run_name={a.run}"])
    out = run_dir(cfg)
    if not (out / "metrics_test.json").exists():
        raise SystemExit(f"{out}/metrics_test.json not found — run `ch4hsi evaluate` for this run first")
    dst = Path(a.docs) / a.run
    doc_rel = str(dst.relative_to(REPO)) if dst.is_relative_to(REPO) else str(dst)

    copied = []
    if not a.check:
        dst.mkdir(parents=True, exist_ok=True)
    for name in ("REPORT.md", "metrics_test.json", "mdl.json", "thresholds.json", "diagnostics/diagnostics_summary.json"):
        src = out / name
        if src.exists():
            copied.append(src)
            if not a.check:
                if src.name == "REPORT.md":     # DIAGNOSTICS.md sits beside it in the published copy
                    (dst / "REPORT.md").write_text(src.read_text().replace("(diagnostics/DIAGNOSTICS.md)",
                                                                          "(DIAGNOSTICS.md)"))
                else:
                    shutil.copy(src, dst / Path(name).name)
    dsrc = out / "diagnostics"
    copied += copy_tree(out / "figures", dst / "figures", check=a.check)
    copied += copy_tree(dsrc, dst / "diagnostics", check=a.check)
    maps = _pick_maps(dsrc / "maps", a.maps)
    copied += maps
    if maps and not a.check:
        (dst / "diagnostics" / "maps").mkdir(parents=True, exist_ok=True)
        for f in maps:
            shutil.copy(f, dst / "diagnostics" / "maps" / f.name)
    if (dsrc / "DIAGNOSTICS.md").exists():
        copied.append(dsrc / "DIAGNOSTICS.md")
        if not a.check:
            (dst / "DIAGNOSTICS.md").write_text(_rewrite_diagnostics(dsrc / "DIAGNOSTICS.md", {f.name for f in maps}))

    mb = sum(f.stat().st_size for f in copied) / 1e6
    log.info("%s %d files (%.1f MB) -> %s", "would copy" if a.check else "copied", len(copied), mb, dst)
    if mb > 25:
        log.warning("that is a lot to commit; trim with --panels / --maps or keep the figures out of git")

    block = results_block(cfg, out, a.run, doc_rel, a.panels)
    readme = Path(a.readme)
    if a.check:
        print(block)
        return 0
    new = splice(readme, block)
    readme.write_text(new)
    log.info("README results block %s", "updated" if START in readme.read_text() else "added")
    log.info("review with: git diff -- README.md && git status --short docs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
