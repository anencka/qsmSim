"""P3 for Experiment 1: the QSM Reconstruction Challenge 2.0 Sim1 head at 3 T.

* chi: RC2 Sim1 ground truth (a whole-head model), set to 0 outside the
  provided brain mask, so there is no background field (Q-P3 (a)).
* M0, R2*: log-linear monoexponential fit of the provided 7 T 4-echo magnitude
  (TE 4/12/20/28 ms, magnitude^2 weights); R2*(3 T) = R2*(7 T) * 3/7, clipped.
  M0 is scaled so the WM median is 0.70 (as in the analytic phantom); the SNR
  definition (first-echo WM magnitude) makes the absolute scale irrelevant.
* Everything is centered in a FOV padded to a multiple of 12 master voxels and
  interpolated to the master grid with cubic splines (chi, M0, R2*) or nearest
  neighbour (segmentation, mask). The 0.999 mm RC2 voxel is treated as 1.0 mm.
Labels keep the RC2 segmentation codes (label.mat order, 1-based).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .phantom import Grid
from .rc2 import ROOT as RC2_ROOT

TE_7T = np.array([4.0, 12.0, 20.0, 28.0]) * 1e-3
LABELS = {"caudate": 1, "globus_pallidus": 2, "putamen": 3, "red_nucleus": 4, "dentate": 5,
          "substantia_nigra": 6, "thalamus": 7, "white_matter": 8, "cortical_gm": 9, "csf": 10,
          "veins": 11, "calcification": 16}
WM, CSF = 8, 10
FILES = ["Sim1GT/Chi.nii.gz", "Sim1GT/Segmentation.nii.gz", "Sim1GT/label.mat", "Sim1GT/FilestructureForEval.mat",
         "Sim1/MaskBrainExtracted.nii.gz", "Sim1/Magnitude.nii.gz"]


def _nii(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def load(root=RC2_ROOT):
    root = Path(root)
    return dict(chi=_nii(root / "Sim1GT/Chi.nii.gz").astype(np.float32),
                seg=np.rint(_nii(root / "Sim1GT/Segmentation.nii.gz")).astype(np.uint8),
                mask=_nii(root / "Sim1/MaskBrainExtracted.nii.gz") > 0,
                mag=_nii(root / "Sim1/Magnitude.nii.gz").astype(np.float32))


def fit_m0_r2s(mag, te=TE_7T, floor_frac=0.02):
    """Weighted log-linear fit per voxel -> (M0, R2*) at the acquisition field."""
    s = np.maximum(mag, 1e-6).astype(np.float64)
    y = np.log(s)
    w = s**2
    sw, swt, swtt = w.sum(-1), (w * te).sum(-1), (w * te**2).sum(-1)
    swy, swty = (w * y).sum(-1), (w * te * y).sum(-1)
    det = sw * swtt - swt**2
    det = np.where(det > 0, det, np.inf)
    slope = (sw * swty - swt * swy) / det
    icpt = (swtt * swy - swt * swty) / det
    m0 = np.exp(icpt)
    r2s = -slope
    valid = mag[..., 0] > floor_frac * np.percentile(mag[..., 0], 99)
    m0 = np.where(valid, m0, 0)
    r2s = np.where(valid, r2s, 0)
    return m0.astype(np.float32), r2s.astype(np.float32)


def pad_to(a, shape, value=0):
    out = np.full(shape, value, a.dtype)
    lo = [(n - m) // 2 for n, m in zip(shape, a.shape)]
    out[tuple(slice(l, l + m) for l, m in zip(lo, a.shape))] = a
    return out, lo


def build(master_mm=0.5, fov_mm=(168.0, 216.0, 216.0), r2s_scale=3.0 / 7.0, r2s_max=150.0, root=RC2_ROOT):
    """Bulk P3 on the master grid. Returns a dict with Grid, labels, chi, m0, r2s,
    brain mask (master), and the 1 mm working-grid volumes used for vein seeding."""
    d = load(root)
    m0, r2s7 = fit_m0_r2s(d["mag"])
    r2s = np.clip(r2s7 * r2s_scale, 0, r2s_max)
    wm_med = np.median(m0[(d["seg"] == WM) & d["mask"]])
    m0 *= np.float32(0.70 / wm_med)
    chi = np.where(d["mask"], d["chi"], 0).astype(np.float32)
    shape1 = tuple(int(round(f)) for f in fov_mm)                     # 1 mm working grid
    chi1, lo = pad_to(chi, shape1)
    m01, _ = pad_to(m0, shape1)
    r2s1, _ = pad_to(r2s.astype(np.float32), shape1)
    seg1, _ = pad_to(d["seg"], shape1)
    mask1, _ = pad_to(d["mask"], shape1)
    z = 1.0 / master_mm
    if abs(z - round(z)) > 1e-9:
        raise ValueError("master spacing must divide 1 mm")
    z = int(round(z))
    if z == 1:
        chi_m, m0_m, r2s_m, seg_m, mask_m = chi1, m01, r2s1, seg1, mask1
    else:
        up = lambda a, order: ndimage.zoom(a, z, order=order, grid_mode=True, mode="grid-constant",
                                           output=np.float32 if order else a.dtype)
        mask_m = up(mask1.astype(np.uint8), 0) > 0
        chi_m = up(chi1, 3) * mask_m                                    # no chi (no field) outside the brain
        m0_m = np.maximum(up(m01, 3), 0)
        r2s_m = np.clip(up(r2s1, 3), 0, r2s_max)
        seg_m = up(seg1, 0)
    grid = Grid(tuple(n * z for n in shape1), (master_mm,) * 3)
    grid1 = Grid(shape1, (1.0,) * 3)
    return dict(grid=grid, labels=seg_m, chi=chi_m.astype(np.float32), m0=m0_m.astype(np.float32),
                r2s=r2s_m.astype(np.float32), mask=mask_m, grid1=grid1, seg1=seg1, mask1=mask1,
                chi1=chi1, m01=m01, r2s1=r2s1,
                rc2_offset_1mm=lo, rc2_shape=d["chi"].shape)


def lateral_ventricles(seg, mask, voxel_mm=1.0, erode_mm=8.0, min_ml=2.0):
    """CSF components well inside the brain (after eroding the mask), >= min_ml."""
    r = int(round(erode_mm / voxel_mm))
    inner = ndimage.binary_erosion(mask, iterations=r)
    lab, n = ndimage.label((seg == CSF) & inner)
    if n == 0:
        return np.zeros(seg.shape, bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * voxel_mm**3 / 1e3
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_ml]
    return np.isin(lab, keep)


def crop_to_rc2(vol_1mm, offset, shape):
    """Undo the FOV padding on a 1 mm volume (for the official RC2 metrics)."""
    return vol_1mm[tuple(slice(o, o + s) for o, s in zip(offset, shape))]


def sim_for_eval(root=RC2_ROOT):
    """The subset of rc2.load_sim that rc2.evaluate needs (no frequency/phase files)."""
    import scipy.io as sio
    root = Path(root)
    fs = sio.loadmat(root / "Sim1GT/FilestructureForEval.mat", squeeze_me=True,
                     struct_as_record=False)["filesstructure"]
    return dict(chi=_nii(root / fs.chi_crop).astype(np.float64), mask=_nii(root / fs.maskEroded).astype(np.float32),
                seg=np.rint(_nii(root / fs.Segment)).astype(np.int16), calc_moment=float(fs.CalcMoment))
