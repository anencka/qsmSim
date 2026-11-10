"""Experiment 1 metrics (Experiment_1.md, plan section 3 item 5).

An Evaluator is built once per (dataset, evaluation mask) from the truth bundle
and scores any number of chi maps:

* global: qsmsim.metrics.evaluate (nRMSE, HFEN, SSIM, ROI means, deep-GM slope,
  regional RMS near sinuses / cortical rim / deep on the analytic phantoms)
* per region (brain = eval mask, dmv_territory, vein_core, vein_shell, deep_gm, plus any extra_regions):
  nRMSE, RMS error, bias, Pearson r, regression slope, SSIM, HFEN
* realized resolution: Gaussian-equivalent effective FWHM (brain, DMV territory),
  and the vein line-spread function (radially averaged cross-section profiles
  of the thin veins, which are line sources) per diameter bin
* smoothness: gradient-energy ratio, high-frequency spectral energy ratio,
  error autocorrelation FWHM (3dFWHMx estimator)
* veins: per-center susceptibility-moment and peak recovery, CNR against the
  local background ring, binned by local diameter and angle to B0; detection
  AUC (vein core vs shell voxels)
* partial volume (forward model v2, given the tissue fractions per voxel; GPU_Plan.md 2.6), under "pv":
  every region's metrics for pure voxels (one tissue label >= 90 %) and boundary voxels separately; the
  cortical-ribbon bias curve (chi error by cortical-GM fraction, 10 % bins); per deep-GM nucleus the error
  within 1 voxel of its edge by nucleus fraction; vein moment / peak recovery by diameter / voxel size
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

from .. import metrics

FWHM_SIGMAS_MM = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
DIAM_BINS = ((0.0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 9.9))
ANGLE_BINS = ((0, 45), (45, 70), (70, 91))
# vein cross-sections scale with the voxel: half width max(4 mm, 4 voxels), 33 x 33 samples;
# moment integrated within 0.625 x half width, background ring at 0.75-1.0 x half width
XS_HALF_MIN_MM, XS_HALF_VOX, XS_SAMPLES = 4.0, 4.0, 33
XS_MOMENT_FRAC, XS_RING_FRAC = 0.625, (0.75, 1.0)
CENTER_STEP_MM = 1.0        # cross-section spacing along each vein
END_SKIP_MM = 1.5
# partial volume (v2): a voxel is pure when one tissue label fills >= 90 % of it (prepare_v2's boundary count)
PV_PURE = 0.9
PV_NOT_TISSUE = ("intracranial", "dmv")     # fraction maps that are not tissue labels
RIBBON_EDGES = np.round(np.linspace(0.0, 1.0, 11), 2)              # cortical-GM fraction bins (lo, hi]
DGM_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)                    # nucleus fraction bins [lo, hi), last closed
DV_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99.0))     # vein diameter / voxel size


def _bin_name(lo, hi, unit):
    return f"{lo:g}-{hi:g}{unit}"


class Evaluator:
    def __init__(self, truth, eval_mask, rois, regions, net, b0=(0, 0, 1), fractions=None, extra_regions=None):
        self.vox = float(truth["voxel_mm"])
        self.mask = eval_mask
        self.chi_true = truth["chi"]
        self.rois = rois
        self.regions_global = regions
        self.t = metrics.reference(self.chi_true, eval_mask)
        dist, rad = truth["dmv_dist"], truth["dmv_r"]
        core = eval_mask & (dist <= rad + 0.5 * self.vox)
        R = {"brain": eval_mask,
             "dmv_territory": eval_mask & (dist <= 5.0),
             "vein_core": core,
             "vein_shell": eval_mask & (dist >= 1.0) & (dist <= 3.0) & ~core}
        dgm = [rois[n] for n in metrics.DEEP_GM if n in rois]
        if dgm:
            R["deep_gm"] = eval_mask & np.logical_or.reduce(dgm)
        for k, m in (extra_regions or {}).items():      # e.g. the control run's voxel classes (control.py)
            R[k] = eval_mask & m
        self.R = {k: v for k, v in R.items() if v.sum() >= 10}
        # SSIM/HFEN of the truth side, and blurred truths for the effective FWHM
        self.lt = ndimage.gaussian_laplace(np.where(eval_mask, self.t, 0), 1.5)
        self.blur = {}
        for name in ("brain", "dmv_territory"):
            if name in self.R:
                m = self.R[name]
                self.blur[name] = [self._gauss(self.t, s)[m] for s in FWHM_SIGMAS_MM]
        self._init_veins(net, b0)
        self._init_spectrum()
        self.pv = self._init_pv(fractions) if fractions is not None else None

    # ---------------------------------------------------------------- helpers
    def _gauss(self, a, sigma_mm):
        if sigma_mm <= 0:
            return np.where(self.mask, a, 0)
        return ndimage.gaussian_filter(np.where(self.mask, a, 0), sigma_mm / self.vox)

    def _init_spectrum(self):
        m = self.R.get("dmv_territory")
        self.spec_box = None
        if m is None:
            return
        idx = np.argwhere(m)
        lo, hi = idx.min(0), idx.max(0) + 1
        self.spec_box = tuple(slice(a, b) for a, b in zip(lo, hi))
        shp = tuple(b - a for a, b in zip(lo, hi))
        w = np.ones(shp, np.float32)
        for d, n in enumerate(shp):
            w *= np.hanning(n).astype(np.float32).reshape([-1 if i == d else 1 for i in range(3)])
        self.spec_w = w * self.mask[self.spec_box]
        k = np.sqrt(sum(np.fft.fftfreq(n).reshape([-1 if i == d else 1 for i in range(3)]) ** 2
                        for d, n in enumerate(shp)))
        self.spec_hf = k > 0.25                                      # above half Nyquist
        T = np.abs(np.fft.fftn(self.t[self.spec_box] * self.spec_w)) ** 2
        self.spec_t_hf = float(T[self.spec_hf].sum())

    def _init_veins(self, net, b0):
        self.xs = None
        if net is None or len(net) == 0:
            return
        b = np.asarray(b0, float) / np.linalg.norm(b0)
        P, rad, tan, vid = [], [], [], []
        for i, (p, r) in enumerate(zip(net.points, net.radii)):
            seg = np.diff(p, axis=0)
            ln = np.linalg.norm(seg, axis=1)
            s = np.concatenate([[0], np.cumsum(ln)])
            if s[-1] < 2 * END_SKIP_MM + CENTER_STEP_MM:
                continue
            at = np.arange(END_SKIP_MM, s[-1] - END_SKIP_MM + 1e-9, CENTER_STEP_MM)
            k = np.clip(np.searchsorted(s, at, side="right") - 1, 0, len(seg) - 1)
            fr = (at - s[k]) / ln[k]
            P.append(p[k] + fr[:, None] * seg[k])
            rad.append(r[k] + fr * (r[k + 1] - r[k]))
            tan.append(seg[k] / ln[k][:, None])
            vid.append(np.full(len(k), i))
        if not P:
            return
        P, rad, tan, vid = map(np.concatenate, (P, rad, tan, vid))
        a = np.cross(tan, np.where(np.abs(tan[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        u = a / np.linalg.norm(a, axis=1, keepdims=True)
        v = np.cross(tan, u)
        self.xs_half = max(XS_HALF_MIN_MM, XS_HALF_VOX * self.vox)
        g = np.linspace(-self.xs_half, self.xs_half, XS_SAMPLES)
        step = g[1] - g[0]
        A, B = np.meshgrid(g, g, indexing="ij")
        self.xs_rho = np.sqrt(A**2 + B**2).ravel()
        self.xs_dA = step * step
        pts = P[:, None, :] + A.ravel()[None, :, None] * u[:, None, :] + B.ravel()[None, :, None] * v[:, None, :]
        o = np.array([-(n - 1) / 2.0 * self.vox for n in self.mask.shape])
        self.xs_coords = ((pts - o) / self.vox).reshape(-1, 3).T
        self.xs_n = len(P)
        inside = ndimage.map_coordinates(self.mask.astype(np.float32), self.xs_coords, order=0).reshape(len(P), -1)
        self.xs_ok = inside[:, self.xs_rho <= self.xs_half].min(1) > 0        # whole cross-section in the mask
        self.xs_diam = 2 * rad
        self.xs_angle = np.degrees(np.arccos(np.clip(np.abs(tan @ b), 0, 1)))
        self.xs_vid = vid
        self.xs = True
        self.xs_true = self._cross_sections(self.t)

    def _cross_sections(self, x):
        prof = ndimage.map_coordinates(x, self.xs_coords, order=1).reshape(self.xs_n, -1)
        ring = (self.xs_rho >= XS_RING_FRAC[0] * self.xs_half) & (self.xs_rho <= XS_RING_FRAC[1] * self.xs_half)
        base = np.median(prof[:, ring], axis=1)
        p = prof - base[:, None]
        moment = p[:, self.xs_rho <= XS_MOMENT_FRAC * self.xs_half].sum(1) * self.xs_dA
        center = p[:, np.argmin(self.xs_rho)]
        noise = prof[:, ring].std(1)
        return dict(profile=p, moment=moment, peak=center, ring_std=noise)

    def _init_pv(self, fractions):
        """Partial-volume voxel classes and bins from the per-voxel tissue fractions (v2 prep)."""
        tissue = [v for k, v in fractions.items() if k not in PV_NOT_TISSUE]
        pure = np.max(np.stack(tissue), 0) >= PV_PURE
        pv = dict(classes={"pure": self.mask & pure, "boundary": self.mask & ~pure})
        pv["region"] = {name: {c: m & cm for c, cm in pv["classes"].items() if (m & cm).sum() >= 10}
                        for name, m in self.R.items()}
        ribbon = {}
        gm = fractions.get("cortical_gm")
        if gm is not None:
            for lo, hi in zip(RIBBON_EDGES[:-1], RIBBON_EDGES[1:]):
                m = self.mask & (gm > lo) & (gm <= hi)
                if m.sum() >= 3:
                    ribbon[f"gm{100 * lo:g}-{100 * hi:g}"] = m
        pv["ribbon"] = ribbon
        dgm = {}
        for nuc in metrics.DEEP_GM:
            f = fractions.get(nuc)
            if f is None or not (f >= 0.5).any():
                continue
            core = f >= 0.5
            band = self.mask & ndimage.binary_dilation(core) & ~ndimage.binary_erosion(core)  # within 1 voxel of the edge
            bins = {}
            for i, (lo, hi) in enumerate(zip(DGM_EDGES[:-1], DGM_EDGES[1:])):
                m = band & (f >= lo) & ((f < hi) if i < len(DGM_EDGES) - 2 else (f <= hi))
                if m.sum() >= 3:
                    bins[f"f{100 * lo:g}-{100 * hi:g}"] = m
            if band.sum() >= 3:
                dgm[nuc] = dict(edge=band, bins=bins)
        pv["deep_gm"] = dgm
        return pv

    # ---------------------------------------------------------------- scoring
    def __call__(self, chi):
        x = metrics.reference(chi, self.mask)
        out = metrics.evaluate(chi, self.chi_true, self.mask, self.rois, regions=self.regions_global)
        lx = ndimage.gaussian_laplace(np.where(self.mask, x, 0), 1.5)
        ssim_map = _ssim_map(x, self.t, self.mask)
        reg = {}
        for name, m in self.R.items():
            xm, tm = x[m].astype(np.float64), self.t[m].astype(np.float64)
            e = xm - tm
            d = dict(nrmse=float(100 * np.linalg.norm(e) / max(np.linalg.norm(tm), 1e-12)),
                     rms_ppb=float(1e3 * np.sqrt(np.mean(e**2))), bias_ppb=float(1e3 * e.mean()),
                     pearson_r=float(np.corrcoef(xm, tm)[0, 1]) if tm.std() > 0 else float("nan"),
                     slope=float(np.polyfit(tm, xm, 1)[0]) if tm.std() > 0 else float("nan"),
                     ssim=float(ssim_map[m].mean()),
                     hfen=float(100 * np.linalg.norm(lx[m] - self.lt[m]) / max(np.linalg.norm(self.lt[m]), 1e-12)),
                     grad_energy_ratio=_grad_ratio(x, self.t, m),
                     error_fwhm_mm=_acf_fwhm(np.where(m, x - self.t, 0), m, self.vox))
            if name in self.blur:
                d.update(_eff_fwhm(xm, self.blur[name]))
            reg[name] = d
        out["region"] = reg
        if self.spec_box is not None:
            X = np.abs(np.fft.fftn(x[self.spec_box] * self.spec_w)) ** 2
            out["region"]["dmv_territory"]["hf_energy_ratio"] = float(X[self.spec_hf].sum() / max(self.spec_t_hf, 1e-30))
        per_vein = None
        if self.xs is not None:
            out["veins"], per_vein = self._vein_metrics(x)
        if self.pv is not None:
            out["pv"] = self._pv_metrics(x, ssim_map, lx, per_vein)
        return out, per_vein

    def _pv_metrics(self, x, ssim_map, lx, per_vein):
        pv = self.pv
        e = (x - self.t).astype(np.float64)
        res = dict(voxels={c: int(m.sum()) for c, m in pv["classes"].items()})
        res["region"] = {name: {c: _basic(x, self.t, m, ssim_map, lx, self.lt) for c, m in cls.items()}
                         for name, cls in pv["region"].items() if cls}
        res["ribbon"] = {k: _bias(e, m) for k, m in pv["ribbon"].items()}
        res["deep_gm"] = {nuc: dict(edge=_bias(e, d["edge"]), **{k: _bias(e, m) for k, m in d["bins"].items()})
                          for nuc, d in pv["deep_gm"].items()}
        if per_vein is not None:
            ok, dv = per_vein["ok"], self.xs_diam / self.vox
            mom = per_vein["moment_rec"] / np.where(ok, per_vein["moment_true"], 1)
            peak = per_vein["peak_rec"] / np.where(ok, per_vein["peak_true"], 1)
            veins = {}
            for lo, hi in DV_BINS:
                sel = ok & (dv >= lo) & (dv < hi)
                if sel.sum() >= 3:
                    veins[f"dv{lo:g}-{hi:g}"] = dict(n=int(sel.sum()), moment_ratio=float(np.nanmedian(mom[sel])),
                                                      peak_ratio=float(np.nanmedian(peak[sel])))
            res["veins"] = veins
        return res

    def _vein_metrics(self, x):
        rec = self._cross_sections(x)
        tru = self.xs_true
        ok = self.xs_ok & (np.abs(tru["moment"]) > 1e-9)
        mom = np.where(ok, rec["moment"] / np.where(ok, tru["moment"], 1), np.nan)
        peak = np.where(ok, rec["peak"] / np.where(ok, tru["peak"], 1), np.nan)
        cnr = np.where(ok, rec["peak"] / np.maximum(rec["ring_std"], 1e-9), np.nan)
        res = {"n_cross_sections": int(ok.sum())}
        bins = {}
        for lo, hi in DIAM_BINS:
            for alo, ahi in ANGLE_BINS + ((0, 91),):
                sel = ok & (self.xs_diam >= lo) & (self.xs_diam < hi) & (self.xs_angle >= alo) & (self.xs_angle < ahi)
                key = f"d{_bin_name(lo, hi, 'mm')}_a{_bin_name(alo, min(ahi, 90), 'deg')}"
                if sel.sum() < 3:
                    continue
                b = dict(n=int(sel.sum()), moment_ratio=float(np.nanmedian(mom[sel])),
                         peak_ratio=float(np.nanmedian(peak[sel])), cnr=float(np.nanmedian(cnr[sel])))
                if (alo, ahi) == (0, 91):
                    b["lsf_fwhm_mm"] = _radial_fwhm(rec["profile"][sel].mean(0), self.xs_rho, self.xs_half)
                    b["lsf_fwhm_true_mm"] = _radial_fwhm(tru["profile"][sel].mean(0), self.xs_rho, self.xs_half)
                bins[key] = b
        res["bins"] = bins
        res["moment_ratio_median"] = float(np.nanmedian(mom)) if ok.any() else float("nan")
        res["peak_ratio_median"] = float(np.nanmedian(peak)) if ok.any() else float("nan")
        core, shell = self.R.get("vein_core"), self.R.get("vein_shell")
        if core is not None and shell is not None:
            res["auc_core_vs_shell"] = _auc(x[core], x[shell])
        per = dict(vein=self.xs_vid, diam=self.xs_diam.astype(np.float32), angle=self.xs_angle.astype(np.float32),
                   ok=ok, moment_rec=rec["moment"].astype(np.float32), moment_true=tru["moment"].astype(np.float32),
                   peak_rec=rec["peak"].astype(np.float32), peak_true=tru["peak"].astype(np.float32),
                   ring_std=rec["ring_std"].astype(np.float32))
        return res, per


def _basic(x, t, m, ssim_map, lx, lt):
    """Region metrics without the smoothness / resolution estimates (partial-volume classes)."""
    xm, tm = x[m].astype(np.float64), t[m].astype(np.float64)
    e = xm - tm
    return dict(n=int(m.sum()), nrmse=float(100 * np.linalg.norm(e) / max(np.linalg.norm(tm), 1e-12)),
                rms_ppb=float(1e3 * np.sqrt(np.mean(e**2))), bias_ppb=float(1e3 * e.mean()),
                pearson_r=float(np.corrcoef(xm, tm)[0, 1]) if tm.std() > 0 else float("nan"),
                slope=float(np.polyfit(tm, xm, 1)[0]) if tm.std() > 0 else float("nan"),
                ssim=float(ssim_map[m].mean()),
                hfen=float(100 * np.linalg.norm(lx[m] - lt[m]) / max(np.linalg.norm(lt[m]), 1e-12)))


def _bias(e, m):
    """Mean and RMS chi error (ppb) over a voxel set."""
    v = e[m]
    return dict(n=int(m.sum()), bias_ppb=float(1e3 * v.mean()), rms_ppb=float(1e3 * np.sqrt(np.mean(v**2))))


def _ssim_map(x, t, mask, sigma=1.5, K1=0.01, K2=0.03):
    L = float(t[mask].max() - t[mask].min())
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    x = np.where(mask, x, 0).astype(np.float32)
    t = np.where(mask, t, 0).astype(np.float32)
    g = lambda a: ndimage.gaussian_filter(a, sigma)
    mx, mt = g(x), g(t)
    vx, vt, cxt = g(x * x) - mx * mx, g(t * t) - mt * mt, g(x * t) - mx * mt
    return ((2 * mx * mt + C1) * (2 * cxt + C2)) / ((mx**2 + mt**2 + C1) * (vx + vt + C2))


def _grad_ratio(x, t, m):
    num = den = 0.0
    for ax in range(3):
        mm = m & np.roll(m, -1, ax)
        num += float(np.sum((np.roll(x, -1, ax) - x)[mm].astype(np.float64) ** 2))
        den += float(np.sum((np.roll(t, -1, ax) - t)[mm].astype(np.float64) ** 2))
    return num / max(den, 1e-30)


def _acf_fwhm(e, m, vox):
    """3dFWHMx-style: per axis, Gaussian ACF from the variance of first differences;
    returns the geometric mean FWHM (mm) of the equivalent smoothing kernel."""
    v = float(np.var(e[m]))
    if v <= 0:
        return float("nan")
    fw = []
    for ax in range(3):
        mm = m & np.roll(m, -1, ax)
        vd = float(np.var((np.roll(e, -1, ax) - e)[mm]))
        rho = 1 - vd / (2 * v)
        if rho <= 0 or rho >= 1:
            fw.append(np.nan if rho >= 1 else 0.0)
            continue
        s2 = -vox**2 / (4 * np.log(rho))
        fw.append(np.sqrt(8 * np.log(2) * s2))
    fw = np.array(fw)
    return float(np.exp(np.mean(np.log(np.maximum(fw, 1e-6))))) if np.all(np.isfinite(fw)) else float("nan")


def _eff_fwhm(xm, blurred):
    """Best Gaussian sigma (with a free gain) matching the truth to the recon."""
    best = None
    for s, b in zip(FWHM_SIGMAS_MM, blurred):
        bb = b.astype(np.float64)
        g = float(bb @ xm / max(bb @ bb, 1e-30))
        r = float(np.linalg.norm(xm - g * bb) / max(np.linalg.norm(xm), 1e-30))
        if best is None or r < best[2]:
            best = (s, g, r)
    s, g, r = best
    return dict(eff_fwhm_mm=float(2.3548 * s), eff_gain=g, eff_resid=r)


def _radial_fwhm(profile, rho, half):
    edges = np.linspace(0, half, 17)
    rr = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([profile[(rho >= a) & (rho < b)].mean() if np.any((rho >= a) & (rho < b)) else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
    p0 = prof[0]
    if not np.isfinite(p0) or p0 <= 0:
        return float("nan")
    below = np.flatnonzero(prof < p0 / 2)
    if len(below) == 0:
        return float("nan")
    k = below[0]
    if k == 0:
        return float(2 * rr[0])
    r_half = rr[k - 1] + (p0 / 2 - prof[k - 1]) * (rr[k] - rr[k - 1]) / (prof[k] - prof[k - 1])
    return float(2 * r_half)


def _auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))
