"""Reconstruction quality metrics (computed inside an evaluation mask after
referencing both maps the same way - QSM is only defined up to a constant).

rmse   100 * ||x - t|| / ||t||  (normalized RMSE in %, as in the 2016 QSM challenge)
hfen   same, after a Laplacian-of-Gaussian filter (sigma 1.5 voxels)
ssim   mean structural similarity inside the mask (Gaussian window sigma 1.5,
       K1=0.01, K2=0.03, dynamic range = truth range inside the mask)
roi    per-ROI means, their errors, and the slope/intercept of ROI means
       (estimate vs truth) - slope < 1 means underestimated contrast
regions RMS error (ppb) near the sinuses / cortical rim / deep brain
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def reference(x, mask, how="mean", ref_mask=None):
    """Remove the reference value: mean over mask, or over ref_mask (e.g. CSF)."""
    m = mask if how == "mean" else ref_mask & mask
    return np.where(mask, x - x[m].mean(), 0).astype(np.float32)


def nrmse(x, t, mask):
    return 100.0 * np.linalg.norm(x[mask] - t[mask]) / np.linalg.norm(t[mask])


def hfen(x, t, mask, sigma=1.5):
    lx = ndimage.gaussian_laplace(np.where(mask, x, 0), sigma)
    lt = ndimage.gaussian_laplace(np.where(mask, t, 0), sigma)
    return nrmse(lx, lt, mask)


def ssim(x, t, mask, sigma=1.5, K1=0.01, K2=0.03):
    L = float(t[mask].max() - t[mask].min())
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    x = np.where(mask, x, 0).astype(np.float64)
    t = np.where(mask, t, 0).astype(np.float64)
    g = lambda a: ndimage.gaussian_filter(a, sigma)
    mx, mt = g(x), g(t)
    vx = g(x * x) - mx * mx
    vt = g(t * t) - mt * mt
    cxt = g(x * t) - mx * mt
    s = ((2 * mx * mt + C1) * (2 * cxt + C2)) / ((mx**2 + mt**2 + C1) * (vx + vt + C2))
    return float(s[mask].mean())


def roi_stats(x, t, rois, mask):
    out = {}
    for name, r in rois.items():
        m = r & mask
        if m.sum() < 3:
            continue
        out[name] = dict(true=float(t[m].mean()), est=float(x[m].mean()),
                         err=float(x[m].mean() - t[m].mean()), n=int(m.sum()))
    return out


def roi_fit(stats, names=None):
    names = names or list(stats)
    tt = np.array([stats[n]["true"] for n in names if n in stats])
    ee = np.array([stats[n]["est"] for n in names if n in stats])
    slope, icpt = np.polyfit(tt, ee, 1)
    return float(slope), float(icpt)


DEEP_GM = ["caudate", "putamen", "globus_pallidus", "thalamus", "red_nucleus",
           "substantia_nigra", "dentate"]


def regions(labels, brain_mask, eval_mask, voxel_mm, sinus_mm=25.0, rim_mm=8.0):
    """Evaluation regions: near the paranasal sinuses (where background fields
    are steepest), the cortical rim (near the mask edge), and the deep brain."""
    from .phantom import L
    d_sinus = ndimage.distance_transform_edt(labels != L.SINUS) * voxel_mm
    d_edge = ndimage.distance_transform_edt(brain_mask) * voxel_mm
    near = eval_mask & (d_sinus < sinus_mm)
    rim = eval_mask & (d_edge < rim_mm) & ~near
    return {"near_sinuses": near, "cortical_rim": rim, "deep": eval_mask & ~near & ~rim}


def evaluate(chi, chi_true, mask, rois, ref="mean", ref_mask=None, regions=None):
    x = reference(chi, mask, ref, ref_mask)
    t = reference(chi_true, mask, ref, ref_mask)
    stats = roi_stats(x, t, rois, mask)
    slope, icpt = roi_fit(stats, DEEP_GM)
    out = dict(rmse=nrmse(x, t, mask), hfen=hfen(x, t, mask), ssim=ssim(x, t, mask),
               deep_gm_slope=slope, deep_gm_intercept=icpt, roi=stats)
    if regions:   # RMS error in ppb per region
        out["regions"] = {k: float(1e3 * np.sqrt(np.mean((x[r] - t[r]) ** 2))) for k, r in regions.items() if r.any()}
    return out
