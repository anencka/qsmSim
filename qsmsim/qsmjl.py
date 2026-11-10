"""Cross-check qsmSim's algorithms against QSM.jl (reference implementation).

    python -m qsmsim.qsmjl outputs/<run>            # prepare -> run Julia -> compare
    python -m qsmsim.qsmjl outputs/<run> --no-run   # re-score existing Julia outputs

Uses the run's own fitted total field, mask, noise weights and magnitude-edge
weights, so both implementations see identical inputs. Both are oracle-tuned
the same way (lowest RMSE on the oracle local field) because their
regularization parameters are not on the same scale (QSM.jl uses gradients
per mm and its own objective scaling). Writes <run>/qsmjl/comparison.md and
figures. Requires Julia 1.10 + the environment in ../julia (see README).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from . import inversion, metrics as mt, phantom as ph
from .echofit import noise_weights

ROOT = Path(__file__).resolve().parent.parent
JL_DIR = ROOT / "julia"
JULIA = os.environ.get("QSMSIM_JULIA", str(Path.home() / ".juliaup/bin/julia"))
JULIA_CHANNEL = os.environ.get("QSMSIM_JULIA_CHANNEL", "+1.10")

BGS = ["oracle", "vsharp", "pdf"]
INVS = ["tkd", "cfl2", "tv", "medi"]
GRIDS = {
    "tkd": [0.1, 0.15, 0.19, 0.25, 0.3],
    "cfl2": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "tv": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "medi": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
JL_NAMES = {"tkd": "tkd", "cfl2": "tikh(reg=:gradient)", "tv": "tv(W)", "medi": "tv(W, Wtv=edges)"}


def _nifti(a, path, vox):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.asarray(a), np.diag([*vox, 1.0])), path)


def _read(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def prepare(run: Path):
    meta = json.loads((run / "metrics.json").read_text())
    cfg = meta["config"]
    vols = np.load(run / "volumes.npz")
    vox = (cfg["voxel_mm"],) * 3
    mask = vols["mask"].astype(bool)

    ndir = run / "nifti"
    ndir.mkdir(exist_ok=True)
    for k in ("total_field_est", "mask", "eval_mask", "chi_true", "local_field_oracle"):
        if not (ndir / f"{k}.nii.gz").exists():
            _nifti(vols[k], ndir / f"{k}.nii.gz", vox)

    inp = run / "qsmjl" / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    _nifti(noise_weights(vols["field_std"], mask), inp / "weights.nii.gz", vox)
    medi_p = meta["recon"].get("oracle/medi", {}).get("params", {})
    G = inversion.edge_weights(vols["magnitude"], mask, medi_p.get("edge_fraction", 0.1))
    for a, g in zip("xyz", G):
        _nifti(g, inp / f"edge_{a}.nii.gz", vox)

    vs = {"max_radius_mm": 12.0, "thr": 0.05, **cfg["params"].get("vsharp", {})}
    pdfp = {"maxiter": 150, "tol": 1e-3, **cfg["params"].get("pdf", {})}
    radii = np.arange(vs["max_radius_mm"], vox[0] - 1e-6, -vox[0])
    tkd_thr = meta["recon"].get("oracle/tkd", {}).get("params", {}).get("thr", 0.19)
    toml = [
        f"vsz = [{vox[0]}, {vox[1]}, {vox[2]}]",
        f"bdir = [{', '.join(f'{b:.10f}' for b in meta['b0_dir'])}]",
        f"vsharp_r = [{', '.join(f'{r:.4f}' for r in radii)}]",
        f"vsharp_thr = {vs['thr']}",
        f"pdf_pad = [{', '.join(str(int(np.ceil(0.125 * n))) for n in mask.shape)}]",
        f"pdf_tol = {pdfp['tol']}",
        f"pdf_maxit = {int(pdfp['maxiter'])}",
        f"tkd_same_thr = {tkd_thr}",
    ] + [f"grid_{k} = [{', '.join(repr(float(v)) for v in g)}]" for k, g in GRIDS.items()]
    (inp / "config.toml").write_text("\n".join(toml) + "\n")


def run_julia(run: Path):
    cmd = [JULIA, JULIA_CHANNEL, "-t", "auto", f"--project={JL_DIR}", str(JL_DIR / "qsmjl_run.jl"), str(run)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _rois(cfg):
    P = ph.build_phantom(tuple(cfg["fov_mm"]), cfg["voxel_mm"], cfg["oversample"], cfg["texture_ppm"],
                         cfg.get("texture_sigma_mm", 3.0), seed=cfg["seed"] + 1)
    return ph.roi_masks(P.labels, cfg["oversample"])


def _diff_pct(a, b, t, m):
    """Difference between two estimates as % of the referenced truth norm."""
    a, b, t = (mt.reference(x, m) for x in (a, b, t))
    return 100.0 * np.linalg.norm(a[m] - b[m]) / np.linalg.norm(t[m])


def compare(run: Path):
    jd = run / "qsmjl"
    meta = json.loads((run / "metrics.json").read_text())
    cfg = meta["config"]
    vols = np.load(run / "volumes.npz")
    vox = cfg["voxel_mm"]
    rois = _rois(cfg)
    ev = vols["eval_mask"].astype(bool) & (_read(jd / "mask_vsharp.nii.gz") != 0)
    chi_t, loc_t = vols["chi_true"], vols["local_field_true"]
    best_jl = tomllib.loads((jd / "best.toml").read_text())

    # local fields
    lf = []
    for b, label in [("vsharp", "matched"), ("vsharp_default", "QSM.jl defaults"),
                     ("pdf", "matched"), ("pdf_default", "QSM.jl defaults")]:
        l_jl = _read(jd / f"local_{b}.nii.gz")
        m = ev & (_read(jd / f"mask_{b}.nii.gz") != 0)
        mine_key = f"local_field_{b.split('_')[0]}"
        row = dict(bg=b, setting=label, voxels=int(m.sum()),
                   jl=mt.nrmse(mt.reference(l_jl, m), mt.reference(loc_t, m), m))
        if b in ("vsharp", "pdf") and mine_key in vols:
            row["mine"] = mt.nrmse(mt.reference(vols[mine_key], m), mt.reference(loc_t, m), m)
            row["agree"] = _diff_pct(vols[mine_key], l_jl, loc_t, m)
        lf.append(row)

    # susceptibility
    rows, chis_jl = [], {}
    for b in BGS:
        for k in INVS:
            key = f"chi_{b}_{k}"
            if key not in vols:
                continue
            x_jl = _read(jd / f"{key}.nii.gz")
            chis_jl[(b, k)] = x_jl
            r_m = mt.evaluate(vols[key], chi_t, ev, rois)
            r_j = mt.evaluate(x_jl, chi_t, ev, rois)
            rows.append(dict(bg=b, inv=k, mine=r_m, jl=r_j, agree=_diff_pct(vols[key], x_jl, chi_t, ev),
                             p_mine=meta["recon"][f"{b}/{k}"]["params"], p_jl=best_jl.get(k),
                             s_mine=meta["recon"][f"{b}/{k}"]["seconds"]))
    tkd_same = None
    if (jd / "chi_oracle_tkd_samethr.nii.gz").exists() and "chi_oracle_tkd" in vols:
        tkd_same = _diff_pct(vols["chi_oracle_tkd"], _read(jd / "chi_oracle_tkd_samethr.nii.gz"), chi_t, ev)

    timing = {}
    with open(jd / "timing.csv") as fh:
        for r in csv.DictReader(fh):
            timing[r["step"]] = float(r["seconds"])
    sweep = list(csv.DictReader(open(jd / "sweep.csv")))

    from . import plotting as pl
    figs = [_fig_chi(jd / "fig_chi_oracle.png", vols, chis_jl, ev, vox, "oracle", pl),
            _fig_chi(jd / "fig_chi_pdf.png", vols, chis_jl, ev, vox, "pdf", pl),
            _fig_metrics(jd / "fig_metrics.png", rows, pl)]
    _report(jd / "comparison.md", meta, lf, rows, tkd_same, timing, sweep, best_jl, int(ev.sum()), figs)
    json.dump(dict(local_field=lf, chi=[{**r, "mine": {k: v for k, v in r["mine"].items() if k != "roi"},
                                         "jl": {k: v for k, v in r["jl"].items() if k != "roi"}} for r in rows],
                   tkd_same_threshold_diff_pct=tkd_same, best_jl=best_jl),
              open(jd / "comparison.json", "w"), indent=2, default=float)
    print(f"wrote {jd / 'comparison.md'}")


def _fig_chi(path, vols, chis_jl, ev, vox, bg, pl):
    import matplotlib.pyplot as plt
    t = mt.reference(vols["chi_true"], ev)
    sl = pl.Slicer(t.shape, vox, ev)
    nan = lambda v: np.where(ev, v, np.nan)
    invs = [k for k in INVS if (bg, k) in chis_jl]
    fig, axs = plt.subplots(len(invs), 3, figsize=(9.5, 2.9 * len(invs) + 0.6), constrained_layout=True, squeeze=False)
    for r, k in enumerate(invs):
        a = mt.reference(vols[f"chi_{bg}_{k}"], ev)
        b = mt.reference(chis_jl[(bg, k)], ev)
        im = pl._img(axs[r, 0], sl.ax(nan(a)), "gray", *pl.CHI_WIN, "qsmSim" if r == 0 else None)
        pl._img(axs[r, 1], sl.ax(nan(b)), "gray", *pl.CHI_WIN, "QSM.jl" if r == 0 else None)
        imd = pl._img(axs[r, 2], sl.ax(nan(a - b)), pl.DIVERGING, -0.05, 0.05, "qsmSim - QSM.jl" if r == 0 else None)
        axs[r, 0].text(-0.06, 0.5, pl.LABELS[k], rotation=90, va="center", ha="right",
                       transform=axs[r, 0].transAxes, color=pl.INK, fontsize=10)
    pl._cbar(fig, im, axs[:, :2].ravel().tolist(), "chi (ppm)")
    pl._cbar(fig, imd, axs[:, 2].tolist(), "difference (ppm)")
    fig.suptitle(f"qsmSim vs QSM.jl on the {pl.LABELS[bg]} local field (each oracle-tuned)", color=pl.INK, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _fig_metrics(path, rows, pl):
    import matplotlib.pyplot as plt
    panels = [("rmse", "RMSE (%) - lower is better"), ("hfen", "HFEN (%) - lower is better"),
              ("ssim", "SSIM - higher is better"), ("deep_gm_slope", "deep-GM slope - 1 is ideal")]
    labels = [f"{pl.LABELS[r['bg']]} + {pl.LABELS[r['inv']]}" for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axs = plt.subplots(1, 4, figsize=(13, 0.32 * len(rows) + 1.9), sharey=True)
    series = [("mine", "qsmSim", pl.SERIES[0], "o", -0.15), ("jl", "QSM.jl", pl.SERIES[1], "s", 0.15)]
    for ax, (key, title) in zip(axs, panels):
        for src, _, col, mk, off in series:
            ax.plot([r[src][key] for r in rows], y + off, ls="", marker=mk, color=col, markersize=7,
                    markeredgecolor=pl.SURFACE, markeredgewidth=1.2, zorder=3)
        if key == "deep_gm_slope":
            ax.axvline(1.0, color=pl.AXIS, lw=1.2, zorder=1)
        ax.set_title(title, loc="left", fontsize=9, color=pl.INK2)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", length=0, labelcolor=pl.INK2)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", markersize=7, markeredgecolor=pl.SURFACE,
                          markeredgewidth=1.2, label=lab) for _, lab, c, mk, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("qsmSim vs QSM.jl - same inputs, same evaluation mask", color=pl.INK, fontsize=11,
                 x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _report(path, meta, lf, rows, tkd_same, timing, sweep, best_jl, nvox, figs):
    L = ["# qsmSim vs QSM.jl", ""]
    L += [f"Run: `{meta['config']['name']}`, {meta['config']['voxel_mm']} mm, {meta['config']['b0_tesla']} T, "
          f"SNR {meta['config']['snr']}. Both implementations get the same fitted total field, brain mask, "
          f"noise weights and magnitude-edge weights. Metrics use one evaluation mask ({nvox} voxels), "
          "and each regularization parameter is oracle-tuned on the oracle local field.", ""]
    L += ["QSM.jl calls: `vsharp`, `pdf(W, Dkernel=:k)`, `tkd`, `tikh(reg=:gradient)`, `tv(W)`, "
          "`tv(W, Wtv=edges)` (MEDI-like). \"Agreement\" = ||qsmSim - QSM.jl|| as % of ||truth|| "
          "(after the same referencing), on the same scale as RMSE.", ""]
    L += ["## Local field (nRMSE vs true local field, %)", "",
          "| method | QSM.jl setting | voxels | qsmSim | QSM.jl | agreement |", "|---|---|---:|---:|---:|---:|"]
    for r in lf:
        L.append(f"| {r['bg'].split('_')[0]} | {r['setting']} | {r['voxels']} | "
                 f"{r.get('mine', float('nan')):.1f} | {r['jl']:.1f} | {r.get('agree', float('nan')):.1f} |")
    L += ["", "## Susceptibility", "",
          "| background | inversion | RMSE qsmSim | RMSE QSM.jl | HFEN qsmSim | HFEN QSM.jl | SSIM qsmSim | SSIM QSM.jl "
          "| slope qsmSim | slope QSM.jl | agreement |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        a, b = r["mine"], r["jl"]
        L.append(f"| {r['bg']} | {r['inv']} | {a['rmse']:.1f} | {b['rmse']:.1f} | {a['hfen']:.1f} | {b['hfen']:.1f} | "
                 f"{a['ssim']:.3f} | {b['ssim']:.3f} | {a['deep_gm_slope']:.2f} | {b['deep_gm_slope']:.2f} | {r['agree']:.1f} |")
    if tkd_same is not None:
        L += ["", f"TKD at the identical threshold on the identical oracle field: implementations differ by "
                  f"**{tkd_same:.2f} %** of ||truth||."]
    L += ["", "## Tuned parameters (each in its own units)", "", "| inversion | qsmSim | QSM.jl |", "|---|---|---|"]
    for k in INVS:
        p = next((r["p_mine"] for r in rows if r["inv"] == k), {})
        key = {"tkd": "thr", "cfl2": "lam", "tv": "alpha", "medi": "alpha"}[k]
        L.append(f"| {k} | {key} = {p.get(key, float('nan')):g} | "
                 f"{'thr' if k == 'tkd' else 'lambda'} = {best_jl.get(k, float('nan')):g} ({JL_NAMES[k]}) |")
    L += ["", "## QSM.jl sweep (oracle local field)", "", "| inversion | value | RMSE % |", "|---|---:|---:|"]
    L += [f"| {s['inversion']} | {float(s['value']):g} | {float(s['rmse']):.1f} |" for s in sweep]
    L += ["", "## Run time (s, per call)", "", "| step | qsmSim | QSM.jl |", "|---|---:|---:|"]
    for r in rows:
        step = f"{r['bg']} + {r['inv']}"
        L.append(f"| {step} | {r['s_mine']:.1f} | {timing.get(step, float('nan')):.1f} |")
    for k in ("vsharp", "vsharp_default", "pdf", "pdf_default"):
        L.append(f"| background: {k} | | {timing.get(k, float('nan')):.1f} |")
    L += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs] + [""]
    path.write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m qsmsim.qsmjl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--no-run", action="store_true", help="skip Julia; re-score existing outputs")
    a = p.parse_args(argv)
    run = a.run.resolve()
    if not a.no_run:
        prepare(run)
        run_julia(run)
    compare(run)


if __name__ == "__main__":
    main()
