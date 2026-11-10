"""End-to-end QSM simulation: phantom -> fields -> multi-echo GRE -> field fit
-> background removal -> dipole inversion -> metrics, figures, report."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import background as bg
from . import dipole as dp
from . import echofit, inversion, metrics, phantom as ph, signal as sg
from .tfi import tfi

BG_METHODS = ("oracle", "sharp", "vsharp", "pdf", "lbv")
INV_METHODS = ("tkd", "cfl2", "tv", "medi")
SINGLE_STEP = ("tfi",)             # invert the total field directly (no background removal)

DEFAULT_PARAMS = {
    "sharp": dict(radius_mm=6.0, thr=0.05, domain="crop"),
    "vsharp": dict(max_radius_mm=12.0, thr=0.05, domain="crop"),
    "pdf": dict(maxiter=150, tol=1e-3),
    "lbv": dict(maxiter=1000, tol=1e-6),
    "tkd": dict(thr=0.19),
    "cfl2": dict(lam=1e-2),
    "tv": dict(alpha=1e-4, mu1_factor=100.0, mu2=0.1, maxiter=150, tol=1e-3),
    "medi": dict(alpha=1e-4, edge_fraction=0.1, mu1_factor=100.0, mu2=0.1, maxiter=150, tol=1e-3),
    # data: "brain" (same field PDF sees) or "head" (also scalp/soft tissue with signal)
    "tfi": dict(lam=3e-3, pb=30.0, bg_weight=0.0, pad_factor=1.5, outer=25, cg_iters=80, tol=3e-3,
                edge_fraction=0.1, data="brain"),
}

# regularization grids for --sweep (oracle tuning: each value scored against truth)
SWEEP = {
    "tkd": ("thr", [0.1, 0.15, 0.19, 0.25, 0.3]),
    "cfl2": ("lam", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
    "tv": ("alpha", [3e-5, 1e-4, 3e-4, 1e-3]),
    "medi": ("alpha", [1e-5, 3e-5, 1e-4, 3e-4]),
    "tfi": ("lam", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
}


@dataclass
class SimConfig:
    name: str = "default"
    # phantom / grid
    fov_mm: tuple = (240.0, 240.0, 204.0)
    voxel_mm: float = 1.5
    oversample: int = 1             # forward-simulate on a voxel_mm/oversample grid, then complex-average
    texture_ppm: float = 0.005
    texture_sigma_mm: float = 3.0   # spatial scale of the smooth chi texture inside tissue
    mask_erode_mm: float = 3.0
    # scanner / acquisition
    b0_tesla: float = 3.0
    tilt_deg: tuple = (0.0, 0.0)    # head pitch (about x), roll (about y)
    te1_ms: float = 4.0
    dte_ms: float = 4.0
    n_echoes: int = 6
    snr: float = 100.0              # for M0 = 1 at TE = 0
    phase_sign: float = 1.0
    transceive_amp: float = float(np.pi)
    shim_hz: dict = field(default_factory=lambda: {"x": 6.0, "y": -10.0, "z": 12.0, "z2": 15.0, "xy": 5.0})
    # simulation numerics
    pad_factor: float = 2.0
    seed: int = 0
    # processing
    background: tuple = ("oracle", "vsharp", "pdf")
    inversion: tuple = ("tkd", "cfl2", "tv", "medi")
    single_step: tuple = ("tfi",)
    params: dict = field(default_factory=dict)   # per-method overrides of DEFAULT_PARAMS
    sweep: bool = False
    reference: str = "mean"         # "mean" or "csf"
    save_nifti: bool = True

    def method_params(self, m):
        p = dict(DEFAULT_PARAMS.get(m, {}))
        p.update(self.params.get(m, {}))
        return p


class Log:
    def __init__(self):
        self.t0 = time.time()
        self.lines = []

    def __call__(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.lines.append(line)


@dataclass
class Truth:
    chi: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    total_field: np.ndarray
    local_field: np.ndarray
    m0: np.ndarray
    rois: dict
    voxel_size: tuple
    b0_dir: np.ndarray


def simulate(cfg: SimConfig, log=print):
    """Phantom, true fields and the noisy multi-echo signal on the acquisition grid."""
    os_ = int(cfg.oversample)
    P = ph.build_phantom(cfg.fov_mm, cfg.voxel_mm, os_, cfg.texture_ppm, cfg.texture_sigma_mm, seed=cfg.seed + 1)
    log(f"phantom: fine grid {P.grid.shape} @ {P.grid.voxel_size[0]:.3g} mm")
    b0_dir = dp.b0_from_tilt(cfg.tilt_deg)
    hzppm = sg.hz_per_ppm(cfg.b0_tesla)

    tot_f = dp.forward_field(P.chi, P.grid.voxel_size, b0_dir, cfg.pad_factor,
                             background=ph.TISSUES[ph.L.AIR][0])
    if cfg.shim_hz:
        tot_f += sg.smooth_poly_field(P.grid, cfg.shim_hz) / np.float32(hzppm)
    log("forward model: total field (tissue + air/bone + shim)")

    rng = np.random.default_rng(cfg.seed)
    phi0_f = sg.transceive_phase(P.grid, rng, cfg.transceive_amp)
    tes = sg.echo_times(cfg.te1_ms, cfg.dte_ms, cfg.n_echoes)
    echoes = []
    for te in tes:
        e = sg.gre_echo(P.m0, P.r2s, tot_f, phi0_f, te, cfg.b0_tesla, cfg.phase_sign)
        echoes.append(sg.add_noise(ph.block_mean(e, os_).astype(np.complex64), cfg.snr, rng))
    S = np.stack(echoes, -1)
    del echoes, phi0_f
    log(f"signal: {len(tes)} echoes, TE {tes[0]*1e3:.1f}-{tes[-1]*1e3:.1f} ms, SNR {cfg.snr}")

    vox = (cfg.voxel_mm,) * 3
    chi = ph.block_mean(P.chi, os_).astype(np.float32)
    m0 = ph.block_mean(P.m0, os_).astype(np.float32)
    tot = ph.block_mean(tot_f, os_).astype(np.float32)
    del tot_f
    icr = ph.block_mean(np.isin(P.labels, ph.INTRACRANIAL).astype(np.float32), os_) >= 0.5
    mask = ph.brain_mask(icr, cfg.voxel_mm, cfg.mask_erode_mm)
    rois = ph.roi_masks(P.labels, os_)
    c = os_ // 2
    labels = P.labels[c::os_, c::os_, c::os_]
    del P
    local = dp.forward_field(np.where(mask, chi, 0), vox, b0_dir, cfg.pad_factor) * mask
    log(f"truth: brain mask {int(mask.sum())} voxels; local field from chi inside the mask")
    return Truth(chi, labels, mask, tot, local.astype(np.float32), m0, rois, vox, b0_dir), S, tes


def remove_background(method, fit, truth, cfg, info=None):
    """info (optional dict): solver diagnostics where the method has them (PDF: CG iterations, final residual)."""
    p = cfg.method_params(method)
    vox, mask = truth.voxel_size, truth.mask
    if method == "oracle":
        # true local field + the actual field-estimation error (noise, partial volume)
        return ((truth.local_field + fit.field_ppm - truth.total_field) * mask).astype(np.float32), mask
    if method == "sharp":
        return bg.sharp(fit.field_ppm, mask, vox, **p)
    if method == "vsharp":
        step = cfg.voxel_mm
        radii = np.arange(p["max_radius_mm"], step - 1e-6, -step)
        return bg.vsharp(fit.field_ppm, mask, vox, radii_mm=radii, thr=p["thr"], domain=p["domain"])
    if method == "pdf":
        return bg.pdf(fit.field_ppm, mask, vox, truth.b0_dir, weights=fit.weights, info=info, **p)
    if method == "lbv":
        return bg.lbv(fit.field_ppm, mask, vox, **p)
    raise ValueError(method)


def invert(method, local, mask, fit, truth, cfg, override=None):
    p = cfg.method_params(method)
    p.update(override or {})
    vox, b0 = truth.voxel_size, truth.b0_dir
    if method == "tkd":
        return inversion.tkd(local, mask, vox, b0, **p), {}
    if method == "cfl2":
        return inversion.cfl2(local, mask, vox, b0, **p), {}
    if method == "tv":
        chi, it = inversion.tv_admm(local, mask, vox, b0, weights=fit.weights, **p)
        return chi, {"iterations": it}
    if method == "medi":
        chi, it = inversion.medi_like(local, mask, vox, fit.magnitude, b0, weights=fit.weights, **p)
        return chi, {"iterations": it}
    raise ValueError(method)


def head_field(S, tes, truth, cfg, log):
    """Field fit over every voxel with signal (brain + scalp/soft tissue), for
    TFI's data="head" option. Signal mask: first-echo magnitude > 5 noise
    sigmas, morphologically opened (no hole filling - that would pull in the
    air-filled sinuses)."""
    from scipy import ndimage
    head = ndimage.binary_opening(np.abs(S[..., 0]) > 5.0 / cfg.snr) | truth.mask
    fit = echofit.fit_field(S, tes, head, truth.voxel_size, cfg.b0_tesla, cfg.phase_sign)
    log(f"TFI head data: field fit on {int(head.sum())} voxels ({int((head & ~truth.mask).sum())} outside the brain)")
    return head, fit


def run_single_step(method, fit, truth, cfg, tfi_data=None, override=None):
    p = cfg.method_params(method)
    p.update(override or {})
    if method != "tfi":
        raise ValueError(method)
    data = p.pop("data")
    if data == "head":
        head, hfit = tfi_data
        return tfi(hfit.field_ppm, truth.mask, truth.voxel_size, fit.magnitude, truth.b0_dir,
                   weights=hfit.weights, data_mask=head, **p)
    return tfi(fit.field_ppm, truth.mask, truth.voxel_size, fit.magnitude, truth.b0_dir, weights=fit.weights, **p)


def run(cfg: SimConfig, outdir: Path):
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    log = Log()
    truth, S, tes = simulate(cfg, log)

    fit = echofit.fit_field(S, tes, truth.mask, truth.voxel_size, cfg.b0_tesla, cfg.phase_sign)
    ferr = (fit.field_ppm - truth.total_field)[truth.mask]
    half_cycle = 0.5 / (sg.hz_per_ppm(cfg.b0_tesla) * cfg.dte_ms * 1e-3)
    n_cycle_err = int(np.count_nonzero(np.abs(ferr) > half_cycle))
    log(f"field fit: median |error| {np.median(np.abs(ferr)) * 1e3:.2f} ppb, "
        f"predicted median std {np.median(fit.field_std_ppm[truth.mask]) * 1e3:.2f} ppb, "
        f"voxels off by >= 1 aliasing cycle: {n_cycle_err}")
    first_echo = S[..., 0].copy()
    tfi_data = None
    if "tfi" in cfg.single_step and cfg.method_params("tfi")["data"] == "head":
        tfi_data = head_field(S, tes, truth, cfg, log)
    del S

    locals_, masks = {}, {}
    for b in cfg.background:
        locals_[b], masks[b] = remove_background(b, fit, truth, cfg)
        log(f"background removal: {b} ({int(masks[b].sum())} voxels)")

    eval_mask = np.logical_and.reduce([masks[b] for b in cfg.background])
    ref_mask = truth.rois["csf"]
    regions = metrics.regions(truth.labels, truth.mask, eval_mask, cfg.voxel_mm)
    evaluate = lambda chi: metrics.evaluate(chi, truth.chi, eval_mask, truth.rois, cfg.reference, ref_mask, regions)
    results = {"field_fit": dict(median_abs_err_ppb=float(np.median(np.abs(ferr)) * 1e3),
                                 p99_abs_err_ppb=float(np.percentile(np.abs(ferr), 99) * 1e3),
                                 max_abs_err_ppb=float(np.abs(ferr).max() * 1e3),
                                 voxels_cycle_error=n_cycle_err)}
    results["local_field"] = {}
    for b in cfg.background:
        a = metrics.reference(locals_[b], eval_mask)
        t = metrics.reference(truth.local_field, eval_mask)
        results["local_field"][b] = dict(nrmse=metrics.nrmse(a, t, eval_mask))

    sweep_results, chosen = {}, {}
    if cfg.sweep:
        sb = "oracle" if "oracle" in locals_ else cfg.background[0]
        for m in cfg.inversion:
            key, grid = SWEEP[m]
            rows = []
            for v in grid:
                chi, _ = invert(m, locals_[sb], masks[sb], fit, truth, cfg, {key: v})
                r = evaluate(chi)
                rows.append(dict(value=v, rmse=r["rmse"], hfen=r["hfen"], ssim=r["ssim"]))
                log(f"sweep {m} {key}={v:g}: RMSE {r['rmse']:.1f}%")
            best = min(rows, key=lambda r: r["rmse"])
            chosen[m] = {key: best["value"]}
            sweep_results[m] = dict(param=key, background=sb, rows=rows, best=best["value"])
        for m in cfg.single_step:
            key, grid = SWEEP[m]
            rows = []
            for v in grid:
                chi, _, _ = run_single_step(m, fit, truth, cfg, tfi_data, {key: v})
                r = evaluate(chi)
                rows.append(dict(value=v, rmse=r["rmse"], hfen=r["hfen"], ssim=r["ssim"]))
                log(f"sweep {m} {key}={v:g}: RMSE {r['rmse']:.1f}%")
            best = min(rows, key=lambda r: r["rmse"])
            chosen[m] = {key: best["value"]}
            sweep_results[m] = dict(param=key, background="total", rows=rows, best=best["value"])
        results["sweep"] = sweep_results

    chis = {}
    results["recon"] = {}
    for b in cfg.background:
        for m in cfg.inversion:
            t = time.time()
            chi, info = invert(m, locals_[b], masks[b], fit, truth, cfg, chosen.get(m))
            r = evaluate(chi)
            r.update(info, seconds=time.time() - t,
                     params={**cfg.method_params(m), **chosen.get(m, {})})
            results["recon"][f"{b}/{m}"] = r
            chis[(b, m)] = chi
            log(f"{b:>6s} + {m:<4s}: RMSE {r['rmse']:5.1f}%  HFEN {r['hfen']:5.1f}%  SSIM {r['ssim']:.3f}  "
                f"deep-GM slope {r['deep_gm_slope']:.2f}  ({r['seconds']:.0f}s)")
    chi_fov = {}
    for m in cfg.single_step:
        t = time.time()
        chi, chi_fov[m], info = run_single_step(m, fit, truth, cfg, tfi_data, chosen.get(m))
        r = evaluate(chi)
        r.update(iterations=info["outer"], cg_iterations=int(sum(info["cg"])), seconds=time.time() - t,
                 params={**cfg.method_params(m), **chosen.get(m, {})})
        results["recon"][f"total/{m}"] = r
        chis[("total", m)] = chi
        log(f"single-step {m}: RMSE {r['rmse']:5.1f}%  HFEN {r['hfen']:5.1f}%  SSIM {r['ssim']:.3f}  "
            f"deep-GM slope {r['deep_gm_slope']:.2f}  ({r['seconds']:.0f}s)")

    # --- save -------------------------------------------------------------
    vols = dict(chi_true=truth.chi, mask=truth.mask.astype(np.uint8), eval_mask=eval_mask.astype(np.uint8),
                labels=truth.labels, magnitude=fit.magnitude, first_echo_phase=np.angle(first_echo).astype(np.float32),
                total_field_true=truth.total_field, total_field_est=fit.field_ppm, field_std=fit.field_std_ppm,
                local_field_true=truth.local_field)
    for b in cfg.background:
        vols[f"local_field_{b}"] = locals_[b]
    for (b, m), chi in chis.items():
        vols[f"chi_{b}_{m}"] = chi
    for m, chi in chi_fov.items():
        vols[f"chi_{m}_fov"] = chi                      # incl. the estimated background sources
    np.savez_compressed(outdir / "volumes.npz", **vols)
    if cfg.save_nifti:
        _save_nifti(vols, outdir / "nifti", truth.voxel_size)

    meta = dict(config=asdict(cfg), echo_times_ms=[float(t * 1e3) for t in tes],
                b0_dir=truth.b0_dir.tolist(), eval_mask_voxels=int(eval_mask.sum()))
    (outdir / "metrics.json").write_text(json.dumps({**meta, **results}, indent=2, default=float))
    with open(outdir / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["background", "inversion", "rmse_pct", "hfen_pct", "ssim", "deep_gm_slope", "seconds"])
        for k, r in results["recon"].items():
            w.writerow([*k.split("/"), f"{r['rmse']:.2f}", f"{r['hfen']:.2f}", f"{r['ssim']:.4f}",
                        f"{r['deep_gm_slope']:.3f}", f"{r['seconds']:.1f}"])

    from . import plotting
    figs = plotting.make_all(outdir / "figures", cfg, truth, fit, first_echo, locals_, chis, eval_mask, results)
    log(f"figures: {', '.join(p.name for p in figs)}")
    _write_report(outdir, cfg, results, meta, figs, log)
    return results


def _save_nifti(vols, d, voxel_size):
    try:
        import nibabel as nib
    except ImportError:
        print("nibabel not installed - skipping NIfTI export")
        return
    d.mkdir(exist_ok=True)
    aff = np.diag([*voxel_size, 1.0])          # arrays are (L->R, P->A, I->S) = RAS+
    for k, v in vols.items():
        nib.save(nib.Nifti1Image(np.asarray(v), aff), d / f"{k}.nii.gz")


def _write_report(outdir, cfg, results, meta, figs, log):
    L = [f"# QSM simulation: {cfg.name}", ""]
    L += [f"- Grid {cfg.voxel_mm} mm isotropic, oversample x{cfg.oversample}, FOV {tuple(cfg.fov_mm)} mm",
          f"- B0 {cfg.b0_tesla} T, direction {np.round(meta['b0_dir'], 3).tolist()} (tilt {tuple(cfg.tilt_deg)} deg)",
          f"- {cfg.n_echoes} echoes, TE {meta['echo_times_ms'][0]:.1f}-{meta['echo_times_ms'][-1]:.1f} ms, SNR {cfg.snr}",
          f"- Field fit error (in mask): median {results['field_fit']['median_abs_err_ppb']:.2f} ppb, "
          f"99th pct {results['field_fit']['p99_abs_err_ppb']:.2f} ppb, max {results['field_fit']['max_abs_err_ppb']:.1f} ppb "
          f"({results['field_fit']['voxels_cycle_error']} voxels off by >= 1 aliasing cycle)",
          f"- Evaluation mask: {meta['eval_mask_voxels']} voxels (intersection of background-removal masks); "
          f"reference: {cfg.reference}", ""]
    L += ["## Local field (nRMSE vs true local field, %)", "", "| background | nRMSE |", "|---|---:|"]
    L += [f"| {b} | {r['nrmse']:.1f} |" for b, r in results["local_field"].items()]
    L += ["", "## Susceptibility", "",
          "| background | inversion | RMSE % | HFEN % | SSIM | deep-GM slope | params | s |",
          "|---|---|---:|---:|---:|---:|---|---:|"]
    for k, r in results["recon"].items():
        b, m = k.split("/")
        key = SWEEP.get(m, (None,))[0]
        pv = f"{key}={r['params'][key]:g}" if key else ""
        L.append(f"| {b} | {m} | {r['rmse']:.1f} | {r['hfen']:.1f} | {r['ssim']:.3f} | "
                 f"{r['deep_gm_slope']:.2f} | {pv} | {r['seconds']:.0f} |")
    regs = next(iter(results["recon"].values())).get("regions", {})
    if regs:
        L += ["", "## Regional RMS error (ppb)", "",
              "| method | " + " | ".join(regs) + " |", "|---|" + "---:|" * len(regs)]
        for k, r in results["recon"].items():
            L.append(f"| {k} | " + " | ".join(f"{r['regions'][n]:.1f}" for n in regs) + " |")
    L += ["", "## ROI means (ppm, referenced)", ""]
    first = next(iter(results["recon"].values()))["roi"]
    names = list(first)
    L.append("| method | " + " | ".join(names) + " |")
    L.append("|---|" + "---:|" * len(names))
    L.append("| truth | " + " | ".join(f"{first[n]['true']:.3f}" for n in names) + " |")
    for k, r in results["recon"].items():
        L.append(f"| {k} | " + " | ".join(f"{r['roi'][n]['est']:.3f}" if n in r["roi"] else "" for n in names) + " |")
    L += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
    L += ["", "## Log", "", "```", *log.lines, "```", ""]
    (outdir / "report.md").write_text("\n".join(L))
