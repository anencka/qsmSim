"""Figures. Image panels: grayscale for chi/magnitude/phase, a blue-gray-red
diverging map for signed fields and errors. Charts: small multiples with one
series per background-removal method (fixed categorical order + distinct
markers, so identity never relies on color alone)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import metrics as mt  # noqa: E402
from .phantom import L  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5, fixed order
MARKERS = ["o", "s", "D", "P", "X", "^"]
DIVERGING = LinearSegmentedColormap.from_list(
    "bgr", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#ef9a99", "#e34948", "#8f1d1c"])
BG_ORDER = ["oracle", "sharp", "vsharp", "pdf", "lbv", "total"]
LABELS = {"oracle": "Oracle", "sharp": "SHARP", "vsharp": "V-SHARP", "pdf": "PDF", "lbv": "LBV", "total": "Total field", "tfi": "TFI",
          "tkd": "TKD", "cfl2": "CF-L2", "tv": "TV-ADMM", "medi": "MEDI-like"}
CHI_WIN = (-0.10, 0.20)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "axes.titlecolor": INK,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": ["DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _color(bg, present):
    """Color follows the entity: fixed slot by BG_ORDER among methods present."""
    order = [b for b in BG_ORDER if b in present]
    i = order.index(bg)
    return SERIES[i], MARKERS[i]


def _idx(coord_mm, n, vox):
    return int(np.clip(round(coord_mm / vox + (n - 1) / 2), 0, n - 1))


class Slicer:
    """Axial / coronal / sagittal slices through fixed anatomical points,
    cropped to a bounding box."""

    def __init__(self, shape, vox, box_mask, axial_z=0.0, coronal_y=-16.0, sagittal_x=34.0, margin=4):
        self.k = _idx(axial_z, shape[2], vox)
        self.j = _idx(coronal_y, shape[1], vox)
        self.i = _idx(sagittal_x, shape[0], vox)
        idx = np.argwhere(box_mask)
        lo = np.maximum(idx.min(0) - margin, 0)
        hi = np.minimum(idx.max(0) + margin + 1, shape)
        self.lo, self.hi = lo, hi

    def ax(self, v):
        return v[self.lo[0]:self.hi[0], self.lo[1]:self.hi[1], self.k].T

    def cor(self, v):
        return v[self.lo[0]:self.hi[0], self.j, self.lo[2]:self.hi[2]].T

    def sag(self, v):
        return v[self.i, self.lo[1]:self.hi[1], self.lo[2]:self.hi[2]].T

    def views(self, v):
        return [("axial", self.ax(v)), ("coronal", self.cor(v)), ("sagittal", self.sag(v))]


def _img(ax, a, cmap="gray", vmin=None, vmax=None, title=None):
    im = ax.imshow(a, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, color=INK2)
    return im


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    return cb


def fig_phantom(path, truth, fit, first_echo, vox):
    head = truth.labels != L.AIR
    sl = Slicer(truth.chi.shape, vox, head)
    rows = [("chi (ppm)", np.where(head, truth.chi, np.nan), "gray", CHI_WIN),
            ("RSS magnitude", fit.magnitude, "gray", (0, np.percentile(fit.magnitude, 99.5))),
            ("phase, echo 1 (rad)", np.where(np.abs(first_echo) > 0.05, np.angle(first_echo), np.nan), "gray", (-np.pi, np.pi))]
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for r, (name, v, cmap, (lo, hi)) in enumerate(rows):
        for c, (vn, a) in enumerate(sl.views(v)):
            im = _img(axs[r, c], a, cmap, lo, hi, f"{name} - {vn}" if c == 0 else vn)
        _cbar(fig, im, axs[r, :].tolist(), name)
    fig.suptitle("Phantom and simulated data", color=INK, fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_fields(path, truth, fit, locals_, eval_mask, vox):
    m = truth.mask
    sl = Slicer(m.shape, vox, m)
    nan = lambda v, mm=m: np.where(mm, v, np.nan)
    tot_t, tot_e = nan(truth.total_field), nan(fit.field_ppm)
    tlim = np.nanpercentile(np.abs(tot_t), 99)
    loc_t = nan(mt.reference(truth.local_field, eval_mask), eval_mask)
    llim = 0.05
    bgs = list(locals_)
    ncol = 1 + len(bgs)
    fig, axs = plt.subplots(3, ncol, figsize=(3.2 * ncol + 1, 9.5), constrained_layout=True, squeeze=False)
    im = _img(axs[0, 0], sl.ax(tot_t), DIVERGING, -tlim, tlim, "total field: true")
    _img(axs[0, 1], sl.ax(tot_e), DIVERGING, -tlim, tlim, "total field: fitted")
    _cbar(fig, im, axs[0, :2].tolist(), "ppm")
    err = nan(fit.field_ppm - truth.total_field)
    if ncol > 2:
        im2 = _img(axs[0, 2], sl.ax(err) * 1e3, DIVERGING, -10, 10, "fit error")
        _cbar(fig, im2, axs[0, 2], "ppb")
    for a in axs[0, 3:]:
        a.set_axis_off()
    im = _img(axs[1, 0], sl.ax(loc_t), DIVERGING, -llim, llim, "local field: true")
    _img(axs[2, 0], sl.sag(loc_t), DIVERGING, -llim, llim, "local field: true, sagittal")
    for c, b in enumerate(bgs, start=1):
        est = nan(mt.reference(locals_[b], eval_mask), eval_mask)
        _img(axs[1, c], sl.ax(est), DIVERGING, -llim, llim, LABELS[b])
        imd = _img(axs[2, c], sl.ax(est - loc_t), DIVERGING, -llim / 2, llim / 2, f"{LABELS[b]} - true")
    _cbar(fig, im, axs[1, :].tolist(), "ppm")
    _cbar(fig, imd, axs[2, :].tolist(), "error (ppm)")
    fig.suptitle("Field maps (local fields referenced to their mean in the evaluation mask)", color=INK, fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_chi_grid(path, truth, chis, eval_mask, vox, view="ax", error=False):
    bgs = list(dict.fromkeys(b for b, _ in chis))
    invs = list(dict.fromkeys(m for b, m in chis if b != "total"))   # columns: two-step inversions
    singles = [m for b, m in chis if b == "total"]                   # single-step methods fill the "total" row
    if not invs:
        invs = singles
    sl = Slicer(truth.chi.shape, vox, eval_mask)
    get = getattr(sl, view)
    t = mt.reference(truth.chi, eval_mask)
    nan = lambda v: np.where(eval_mask, v, np.nan)
    ncol = len(invs) + (0 if error else 1)
    fig, axs = plt.subplots(len(bgs), ncol, figsize=(2.6 * ncol + 1, 2.9 * len(bgs) + 0.6),
                            constrained_layout=True, squeeze=False)
    for r, b in enumerate(bgs):
        c0 = 0
        if not error:
            _img(axs[r, 0], get(nan(t)), "gray", *CHI_WIN, "truth" if r == 0 else None)
            c0 = 1
        axs[r, 0].text(-0.06, 0.5, LABELS[b], rotation=90, va="center", ha="right",
                       transform=axs[r, 0].transAxes, color=INK, fontsize=10)
        row = singles if b == "total" else invs
        for c in range(c0, ncol):
            m = row[c - c0] if c - c0 < len(row) else None
            title = (LABELS[m] if m and (r == 0 or b == "total") else None)
            if m is None or (b, m) not in chis:
                axs[r, c].set_axis_off()
                continue
            x = mt.reference(chis[(b, m)], eval_mask)
            if error:
                im = _img(axs[r, c], get(nan(x - t)), DIVERGING, -0.1, 0.1, title)
            else:
                im = _img(axs[r, c], get(nan(x)), "gray", *CHI_WIN, title)
    _cbar(fig, im, axs.ravel().tolist(), "chi error (ppm)" if error else "chi (ppm)")
    vname = {"ax": "axial", "cor": "coronal", "sag": "sagittal"}[view]
    fig.suptitle(f"{'Susceptibility error (estimate - truth)' if error else 'Susceptibility'} - {vname}; "
                 "rows: background removal, columns: inversion", color=INK, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _series_legend(fig, bgs, y=0.94):
    handles = [plt.Line2D([], [], color=_color(b, bgs)[0], marker=_color(b, bgs)[1], ls="",
                          markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS[b]) for b in bgs]
    fig.legend(handles=handles, loc="upper center", ncol=len(bgs), bbox_to_anchor=(0.5, y))


def _methods(cfg):
    bgs = [b for b in BG_ORDER if b in cfg.background or (b == "total" and cfg.single_step)]
    return bgs, list(cfg.inversion) + list(cfg.single_step)


def fig_metrics(path, results, cfg):
    bgs, invs = _methods(cfg)
    panels = [("rmse", "RMSE (%) - lower is better"), ("hfen", "HFEN (%) - lower is better"),
              ("ssim", "SSIM - higher is better"), ("deep_gm_slope", "deep-GM slope - 1 is ideal")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 0.55 * len(invs) + 1.8), sharey=True)
    y = np.arange(len(invs))[::-1]
    off = np.linspace(-0.18, 0.18, len(bgs)) if len(bgs) > 1 else [0.0]
    for ax, (key, title) in zip(axs, panels):
        for b, o in zip(bgs, off):
            col, mk = _color(b, bgs)
            pts = [(results["recon"][f"{b}/{m}"][key], yy + o) for m, yy in zip(invs, y)
                   if f"{b}/{m}" in results["recon"]]
            if pts:
                ax.plot(*zip(*pts), ls="", marker=mk, color=col, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        if key == "deep_gm_slope":
            ax.axvline(1.0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(title, loc="left", fontsize=9, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [LABELS[m] for m in invs])
        ax.set_ylim(-0.6, len(invs) - 0.4)
        ax.tick_params(axis="y", length=0, labelcolor=INK2)
    fig.suptitle("Reconstruction metrics by inversion (rows) and background removal (markers)",
                 color=INK, fontsize=11, x=0.01, ha="left", y=0.99)
    _series_legend(fig, bgs, y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_roi(path, results, cfg):
    bgs, invs = _methods(cfg)
    fig, axs = plt.subplots(1, len(invs), figsize=(3.3 * len(invs), 4.6), sharex=True, sharey=True, squeeze=False)
    axs = axs[0]
    names = [n for n in mt.DEEP_GM + ["white_matter", "csf"]]
    for ax, m in zip(axs, invs):
        lim = [-0.06, 0.22]
        ax.plot(lim, lim, color=AXIS, lw=1.2, zorder=1)
        for b in bgs:
            if f"{b}/{m}" not in results["recon"]:
                continue
            col, mk = _color(b, bgs)
            roi = results["recon"][f"{b}/{m}"]["roi"]
            t = [roi[n]["true"] for n in names if n in roi]
            e = [roi[n]["est"] for n in names if n in roi]
            ax.plot(t, e, ls="", marker=mk, color=col, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=3)
        ax.set_title(LABELS[m], loc="left", color=INK)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("true ROI mean (ppm)")
    axs[0].set_ylabel("estimated ROI mean (ppm)")
    fig.suptitle("Deep grey matter, WM and CSF ROI means vs truth (line = identity)",
                 color=INK, fontsize=11, x=0.01, ha="left", y=0.99)
    _series_legend(fig, bgs, y=0.93)
    fig.tight_layout(rect=(0, 0.02, 1, 0.87))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_sweep(path, sweep):
    ms = list(sweep)
    fig, axs = plt.subplots(1, len(ms), figsize=(3.3 * len(ms), 3.2), squeeze=False)
    for ax, m in zip(axs[0], ms):
        s = sweep[m]
        v = [r["value"] for r in s["rows"]]
        e = [r["rmse"] for r in s["rows"]]
        ax.plot(v, e, color=SERIES[0], lw=1.5, marker="o", markersize=6, markeredgecolor=SURFACE)
        i = int(np.argmin(e))
        ax.annotate(f"{v[i]:g}", (v[i], e[i]), textcoords="offset points", xytext=(0, 8),
                    ha="center", color=INK2, fontsize=8)
        if m != "tkd":
            ax.set_xscale("log")
        ax.set_title(LABELS[m], loc="left")
        ax.set_xlabel(s["param"])
    axs[0][0].set_ylabel("RMSE (%)")
    fig.suptitle(f"Regularization sweep (on {LABELS[sweep[ms[0]]['background']]} local field)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_all(d: Path, cfg, truth, fit, first_echo, locals_, chis, eval_mask, results):
    d = Path(d)
    vox = truth.voxel_size[0]
    out = []

    def f(name):
        p = d / name
        out.append(p)
        return p

    fig_phantom(f("01_phantom.png"), truth, fit, first_echo, vox)
    fig_fields(f("02_fields.png"), truth, fit, locals_, eval_mask, vox)
    fig_chi_grid(f("03_chi_axial.png"), truth, chis, eval_mask, vox, "ax")
    fig_chi_grid(f("04_chi_sagittal.png"), truth, chis, eval_mask, vox, "sag")
    fig_chi_grid(f("05_chi_error_axial.png"), truth, chis, eval_mask, vox, "ax", error=True)
    fig_metrics(f("06_metrics.png"), results, cfg)
    fig_roi(f("07_roi.png"), results, cfg)
    if results.get("sweep"):
        fig_sweep(f("08_sweep.png"), results["sweep"])
    return out
