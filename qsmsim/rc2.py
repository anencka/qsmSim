"""QSM Reconstruction Challenge 2.0 (Bilgic et al., MRM 2021): data loading
and a Python port of the official evaluation metrics.

Port of functions4challenge/evaluation/ from the challenge's data share:
  EvaluateRecon_ChallengeFinalMetrics.m   (variant="stage1": what produced
                                           master_stage1_anonymized.csv)
  EvaluateRecon_ChallengeFinalMetrics_vs2.m (variant="vs2": tissue and DGM
                                           masks eroded, calcification v2)
  compute_rmse_detrend_v1.m, compute_linearityDeepGM.m,
  compute_calcification_metrics_v0.m / _v2.m, dilatemask.m,
  Combine_ChallengeSim1Sim2Metrics.m
Indices: MATLAB's 1-based inclusive ranges are mapped to 0-based slices.
Data: qsmsim-fetch-rc2 (qsmsim/rc2_download.py)  ->  data/rc2/ or QSMSIM_RC2_DIR
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage

def _default_root():
    """The challenge files: QSMSIM_RC2_DIR; else data/rc2 in a source checkout; else ~/.cache/qsmsim/rc2."""
    if os.environ.get("QSMSIM_RC2_DIR"):
        return Path(os.environ["QSMSIM_RC2_DIR"])
    src = Path(__file__).resolve().parent.parent
    if (src / "pyproject.toml").exists() or (src / "data" / "rc2").exists():
        return src / "data" / "rc2"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qsmsim" / "rc2"


ROOT = _default_root()
METRICS = ["rmse", "rmse_detrend", "rmse_detrend_Tissue", "rmse_detrend_Blood", "rmse_detrend_DGM",
           "DeviationFromLinearSlope", "CalcStreak", "DeviationFromCalcMoment"]
TE_MS = (4.0, 12.0, 20.0, 28.0)


def _nii(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def load_sim(sim, root=ROOT):
    """Inputs and ground truth for 'Sim1' or 'Sim2' (arrays in file order)."""
    root = Path(root)
    fs = sio.loadmat(root / f"{sim}GT" / "FilestructureForEval.mat", squeeze_me=True,
                     struct_as_record=False)["filesstructure"]
    lab = sio.loadmat(root / f"{sim}GT" / "label.mat", squeeze_me=True)["label"]
    return dict(
        frequency=_nii(root / sim / "Frequency.nii.gz").astype(np.float32),
        magnitude=_nii(root / sim / "Magnitude.nii.gz").astype(np.float32),
        phase=_nii(root / sim / "Phase.nii.gz").astype(np.float32),
        mask=_nii(root / fs.maskEroded).astype(np.float32),
        chi=_nii(root / fs.chi_crop).astype(np.float64),
        seg=np.rint(_nii(root / fs.Segment)).astype(np.int16),
        labels=[str(l[0]) for l in lab],
        calc_moment=float(fs.CalcMoment),
    )


def _dilate(m, n=1):
    """dilatemask.m: n x imdilate with ones(3,3,3)."""
    return ndimage.binary_dilation(m != 0, structure=np.ones((3, 3, 3)), iterations=n).astype(np.float32)


def rmse_detrend(recon, true, msk):
    """compute_rmse_detrend_v1.m -> (rmse, rmse_detrend), both demeaned in msk."""
    r = recon[msk != 0].astype(np.float64)
    t = true[msk != 0].astype(np.float64)
    r = r - r.mean()
    t = t - t.mean()
    nt = np.linalg.norm(t)
    rmse = 100 * np.linalg.norm(r - t) / nt
    a, b = np.polyfit(t, r, 1)               # recon ~ a*true + b
    res = (r - b) / a                         # polyval([1/a, -b/a], recon)
    return rmse, 100 * np.linalg.norm(res - t) / nt


def linearity_dgm(recon, true, seg):
    """compute_linearityDeepGM.m: slope of ROI means over labels 1..6."""
    mt = [true[seg == k].mean() for k in range(1, 7)]
    mr = [recon[seg == k].mean() for k in range(1, 7)]
    return np.polyfit(mt, mr, 1)[0]


def _box(mask3, n):
    """Bounding box of mask3 grown by n voxels: list of (lo, hi) inclusive (0-based)."""
    out = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.flatnonzero(mask3.max(axis=other))
        out.append((idx[0] - n, idx[-1] + n))
    return out


def _grow(box, n):
    return [(lo - n, hi + n) for lo, hi in box]


def _fill(shape, box):
    m = np.zeros(shape, bool)
    m[tuple(slice(lo, hi + 1) for lo, hi in box)] = True
    return m


def calcification(recon, true, seg, variant="v0"):
    """compute_calcification_metrics_v0/_v2.m -> (moment, streak)."""
    calc = seg == 16
    if variant == "v0":
        n_cube, n_rim, n_close = 3, 4, 4
    else:
        n_cube, n_rim, n_close = 2, 6, 4
    b_cube = _box(calc, n_cube)
    cube = _fill(seg.shape, b_cube)
    b_rim = _grow(b_cube, n_rim)
    rim = _fill(seg.shape, b_rim) & ~cube
    close = _fill(seg.shape, _grow(b_rim, n_close))
    q_cube = recon[cube]
    q_out = recon[~cube & close]
    thresholds = np.round(np.arange(0, 351) * -0.01, 10)          # 0:-0.01:-3.5
    use = thresholds[-1]
    for th in thresholds:
        if np.count_nonzero(q_out < th) == 0:
            use = th
            break
    seg_c = q_cube < use
    moment = np.count_nonzero(seg_c) * q_cube[seg_c].mean() if seg_c.any() else 0.0
    p = np.polyfit(true[rim], recon[rim], 1)
    resid = recon[rim] - np.polyval(p, true[rim])
    streak = np.std(resid, ddof=1) / abs(q_cube[seg_c].mean()) if seg_c.any() else np.nan
    return float(moment), float(streak)


def evaluate(recon, sim, variant="stage1"):
    """EvaluateRecon_ChallengeFinalMetrics(.m / _vs2.m) for one simulation."""
    recon = np.nan_to_num(np.asarray(recon, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    chi, msk, seg = sim["chi"], sim["mask"], sim["seg"]
    m = {}
    m["rmse"], m["rmse_detrend"] = rmse_detrend(recon, chi, msk)

    tissue = msk.copy()
    tissue[(seg < 7) | (seg > 9)] = 0
    if variant == "vs2":
        tissue = 1 - _dilate(1 - tissue)
    m["rmse_detrend_Tissue"] = rmse_detrend(recon, chi, tissue)[1]

    vein = msk.copy()
    vein[seg != 11] = 0
    m["rmse_detrend_Blood"] = rmse_detrend(recon, chi, _dilate(vein))[1]

    dgm = msk.copy()
    dgm[seg >= 7] = 0
    if variant == "vs2":
        dgm = 1 - _dilate(1 - dgm)
    m["rmse_detrend_DGM"] = rmse_detrend(recon, chi, dgm)[1]
    m["DeviationFromLinearSlope"] = abs(1 - linearity_dgm(recon, chi, seg))

    moment, m["CalcStreak"] = calcification(recon, chi, seg, "v0" if variant == "stage1" else "v2")
    m["DeviationFromCalcMoment"] = abs(sim["calc_moment"] - moment)
    return m


def combine(m1, m2):
    """Combine_ChallengeSim1Sim2Metrics.m: mean of Sim1/Sim2, calcification metrics from Sim2."""
    return {k: (m2[k] if k in ("CalcStreak", "DeviationFromCalcMoment") else (m1[k] + m2[k]) / 2) for k in m1}


def leaderboard(path=ROOT / "metrics" / "master_stage1_anonymized.csv"):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in METRICS:
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = np.nan
    return rows
