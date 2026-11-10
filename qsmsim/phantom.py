"""Numerical head phantom for QSM simulation.

Geometry is defined analytically in mm (origin at the FOV center), so the
same phantom can be voxelized at any resolution - including a finer grid for
forward simulation followed by complex averaging (see pipeline.py), which
avoids the "inverse crime" of simulating and reconstructing on one grid.

Contents: scalp and skull, a facial block with paranasal sinuses (the main
source of background field), CSF, a folded cortical ribbon over white matter,
ventricles, deep grey nuclei (caudate, putamen, globus pallidus, thalamus, red
nucleus, substantia nigra), a cerebellum with dentate nuclei, brainstem, veins
at several orientations to B0 (superior sagittal and straight sinus, internal
cerebral veins, medullary veins), a calcification and a microbleed.

Tissue values (chi relative to water in ppm, proton density M0, R2* in 1/s at
3T) are representative literature values, not a specific atlas. Edit TISSUES
to change them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy import ndimage


class L(IntEnum):
    AIR = 0
    SOFT = 1
    BONE = 2
    SINUS = 3
    CSF = 4
    GM = 5
    WM = 6
    CAUDATE = 7
    PUTAMEN = 8
    GP = 9
    THALAMUS = 10
    RN = 11
    SN = 12
    DENTATE = 13
    VEIN = 14
    CALC = 15
    BLEED = 16
    DURAL = 17       # dural venous sinus (SSS, straight sinus): venous blood, outside the brain mask


# label: (chi ppm rel. water, M0, R2* [1/s] at 3 T)
TISSUES = {
    L.AIR:      (9.40, 0.00, 0.0),
    L.SOFT:     (0.00, 0.65, 25.0),
    L.BONE:     (-2.00, 0.05, 300.0),
    L.SINUS:    (9.40, 0.00, 0.0),
    L.CSF:      (0.00, 1.00, 4.0),
    L.GM:       (0.02, 0.85, 17.0),
    L.WM:       (-0.03, 0.70, 22.0),
    L.CAUDATE:  (0.07, 0.80, 25.0),
    L.PUTAMEN:  (0.06, 0.80, 28.0),
    L.GP:       (0.18, 0.72, 45.0),
    L.THALAMUS: (0.01, 0.78, 21.0),
    L.RN:       (0.13, 0.72, 38.0),
    L.SN:       (0.16, 0.72, 42.0),
    L.DENTATE:  (0.10, 0.75, 35.0),
    L.VEIN:     (0.45, 0.80, 45.0),
    L.CALC:     (-0.60, 0.40, 150.0),
    L.BLEED:    (1.00, 0.50, 150.0),
    L.DURAL:    (0.45, 0.80, 45.0),
}

# Sourced tissue table (Config.tissues = "sourced"; Experiments_v2.md 2.1-2.3, agreed 2026-10-24): T2* in ms per
# field (R2* = 1000 / T2*) and chi (ppm) of the deep-GM nuclei. Ranges in the source table are entered at their
# midpoints; [n] are the table's reference numbers (the list: Experiments_v2.md 2.4). Labels and values not listed
# keep TISSUES; at 7 T a label in FIELD_RATIO_7T takes its 3 T R2* (sourced, else TISSUES) x the ratio.
#   GM, WM: 66 / 33 and 53 / 27 ms [1] (3 T value not confirmed in full text), [2, 3]
#   venous blood (Y ~ 0.6, Hct 0.4): 21 / 7 ms [9] (assumed modelling value), [2, 3]; chi +0.45 [13, 14] (unchanged);
#       the DMVs (VeinParams.r2s), L.VEIN and the dural sinuses
#   caudate 40-55 / 20-28 ms, chi +0.024 to +0.054; putamen 34-45 / 16-24 ms, +0.033 to +0.071; GP 25-34 / ~11 ms,
#       +0.085 to +0.130; thalamus 48-50 / 25-28 ms, +0.003; SN 26-33 / 13-16 ms, +0.121 to +0.130; RN ~28 / ~17 ms,
#       +0.095 to +0.106; dentate 31 ms at 3 T (Sethi; range 25-50), +0.093. T2*: [4-8, 15-17, 20]; chi: [15, 20]
#   dentate at 7 T: not found; 3 T x DEEP_GM_RATIO (the deep-GM 7 T / 3 T R2* slope of Spincemaille 2020)
#   CSF: no intrinsic T2* (its measured T2* is macroscopic dephasing, which the simulation produces itself), so
#       R2* = R2 = 1 / T2, T2 ~ 2 s at 3 T and 0.9 s at 7 T [10-12]
#   cortical GM and WM chi (+0.02, -0.03) lie in the table's ranges (0 to +0.02, -0.02 to -0.03) and are kept
#   calcification, microbleed: synthetic lesions; v1's 150 1/s at 3 T, x 7/3 at 7 T
#   soft tissue, bone: outside the brain mask; v1 at 3 T, x 2.0 at 7 T
SOURCED_T2S_MS = {
    3.0: {L.GM: 66.0, L.WM: 53.0, L.VEIN: 21.0, L.DURAL: 21.0, L.CAUDATE: 47.5, L.PUTAMEN: 39.5, L.GP: 29.5,
          L.THALAMUS: 49.0, L.SN: 29.5, L.RN: 28.0, L.DENTATE: 31.0, L.CSF: 2000.0},
    7.0: {L.GM: 33.0, L.WM: 27.0, L.VEIN: 7.0, L.DURAL: 7.0, L.CAUDATE: 24.0, L.PUTAMEN: 20.0, L.GP: 11.0,
          L.THALAMUS: 26.5, L.SN: 14.5, L.RN: 17.0, L.CSF: 900.0},
}
SOURCED_CHI = {L.CAUDATE: 0.039, L.PUTAMEN: 0.052, L.GP: 0.1075, L.THALAMUS: 0.003, L.SN: 0.1255, L.RN: 0.1005,
               L.DENTATE: 0.093}
DEEP_GM_RATIO = 2.28
FIELD_RATIO_7T = {L.DENTATE: DEEP_GM_RATIO, L.CALC: 7.0 / 3.0, L.BLEED: 7.0 / 3.0, L.SOFT: 2.0, L.BONE: 2.0}


def tissue_lut(table="v1", b0=3.0, scale=1.0):
    """(chi ppm, M0, R2* 1/s) per label, indexed by label value. table "v1": TISSUES with R2* x scale (Experiment 2
    v1 scaled every tissue by B0 / 3 T); "sourced": SOURCED_T2S_MS at b0 (3 or 7 T) and SOURCED_CHI, scale must
    be 1."""
    lut = np.zeros((max(TISSUES) + 1, 3), np.float32)
    for k, v in TISSUES.items():
        lut[k] = v
    if table == "v1":
        if scale != 1.0:
            lut[:, 2] = (lut[:, 2] * np.float32(scale)).astype(np.float32)
        return lut
    if table != "sourced":
        raise ValueError(f"unknown tissue table '{table}'")
    if scale != 1.0:
        raise ValueError("the sourced tissue table is per field: r2s_scale must be 1")
    b0 = float(b0)
    if b0 not in SOURCED_T2S_MS:
        raise ValueError(f"no sourced relaxation at {b0:g} T (have {sorted(SOURCED_T2S_MS)})")
    if b0 == 7.0:                                   # labels without a 7 T value: their 3 T value x the ratio
        for k, f in FIELD_RATIO_7T.items():
            t3 = SOURCED_T2S_MS[3.0].get(k)
            lut[k, 2] = np.float32((lut[k, 2] if t3 is None else 1000.0 / t3) * f)
    for k, t in SOURCED_T2S_MS[b0].items():
        lut[k, 2] = np.float32(1000.0 / t)
    for k, c in SOURCED_CHI.items():
        lut[k, 0] = np.float32(c)
    return lut


def r2s_table(table="v1", b0=3.0, scale=1.0):
    """The R2* column of tissue_lut."""
    return np.ascontiguousarray(tissue_lut(table, b0, scale)[:, 2])


def blood_r2s(b0):
    """Sourced venous-blood R2* (1/s) at b0: the DMVs' VeinParams.r2s under tissues = "sourced"."""
    return float(np.float32(1000.0 / SOURCED_T2S_MS[float(b0)][L.VEIN]))


INTRACRANIAL = [L.CSF, L.GM, L.WM, L.CAUDATE, L.PUTAMEN, L.GP, L.THALAMUS, L.RN, L.SN,
                L.DENTATE, L.VEIN, L.CALC, L.BLEED]
TEXTURED = [L.GM, L.WM, L.CAUDATE, L.PUTAMEN, L.GP, L.THALAMUS, L.RN, L.SN, L.DENTATE]

# ROIs reported in the metrics (label groups)
ROIS = {
    "caudate": [L.CAUDATE], "putamen": [L.PUTAMEN], "globus_pallidus": [L.GP],
    "thalamus": [L.THALAMUS], "red_nucleus": [L.RN], "substantia_nigra": [L.SN],
    "dentate": [L.DENTATE], "white_matter": [L.WM], "cortical_gm": [L.GM],
    "csf": [L.CSF], "veins": [L.VEIN], "calcification": [L.CALC], "microbleed": [L.BLEED],
}


@dataclass
class Grid:
    shape: tuple
    voxel_size: tuple

    @property
    def axes(self):
        """1-D voxel-center coordinates (mm) per axis, origin at FOV center."""
        return [(np.arange(n) - (n - 1) / 2.0) * d for n, d in zip(self.shape, self.voxel_size)]


@dataclass
class Phantom:
    grid: Grid
    labels: np.ndarray            # uint8 label map
    chi: np.ndarray               # ppm (relative to water), incl. air/bone
    m0: np.ndarray
    r2s: np.ndarray
    meta: dict = field(default_factory=dict)


def _rot(yaw=0.0, pitch=0.0, roll=0.0):
    a, b, c = np.deg2rad([yaw, pitch, roll])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
    return rz @ rx @ ry


class _Painter:
    """Paints analytic shapes into a label volume, evaluating each shape only
    inside its bounding box (keeps memory flat on fine grids).

    The volume is the grid's, or any axis-aligned point set given by its 1-D
    coordinate axes (mm; ascending), e.g. one tile of a finer grid. Each point's
    label depends only on its coordinates, so a point gets the same label in
    every point set that contains it. xp: numpy, or cupy to paint on the GPU."""

    def __init__(self, grid: Grid | None, axes=None, xp=np):
        self.grid = grid
        self.ax = [np.asarray(a, float) for a in (grid.axes if axes is None else axes)]
        self.xp = xp
        # labels go to xp as np.uint8: CuPy takes only exact int/float as Python scalars (not the IntEnum)
        self.labels = xp.full(tuple(len(a) for a in self.ax), np.uint8(L.AIR), dtype=np.uint8)

    def _box(self, lo, hi):
        sl, sub = [], []
        for a, l, h in zip(self.ax, lo, hi):
            i0 = int(np.searchsorted(a, l, side="left"))
            i1 = int(np.searchsorted(a, h, side="right"))
            sl.append(slice(i0, max(i0, i1)))
            sub.append(self.xp.asarray(a[i0:max(i0, i1)]))
        X = sub[0][:, None, None]
        Y = sub[1][None, :, None]
        Z = sub[2][None, None, :]
        return tuple(sl), X, Y, Z

    def ellipsoid_mask(self, center, semi, rot=(0, 0, 0), grow=0.0):
        semi = np.asarray(semi, float) + grow
        R = _rot(*rot)
        ext = np.abs(R) @ semi
        c = np.asarray(center, float)
        sl, X, Y, Z = self._box(c - ext, c + ext)
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        # local coordinates u = R^T (p - c)
        u = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
        v = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
        w = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
        rho2 = (u / semi[0]) ** 2 + (v / semi[1]) ** 2 + (w / semi[2]) ** 2
        return sl, rho2 <= 1.0

    def capsule_mask(self, p0, p1, r):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        sl, X, Y, Z = self._box(np.minimum(p0, p1) - r, np.maximum(p0, p1) + r)
        d = p1 - p0
        dd = float(d @ d)
        t = ((X - p0[0]) * d[0] + (Y - p0[1]) * d[1] + (Z - p0[2]) * d[2]) / dd
        t = np.clip(t, 0.0, 1.0)
        dist2 = (X - p0[0] - t * d[0]) ** 2 + (Y - p0[1] - t * d[1]) ** 2 + (Z - p0[2] - t * d[2]) ** 2
        return sl, dist2 <= r * r

    def paint(self, sl_mask, label, where=None):
        sl, m = sl_mask
        view = self.labels[sl]
        if where is not None:
            m = m & self.xp.isin(view, self.xp.asarray(where, dtype=np.uint8))
        view[m] = np.uint8(label)

    def ellipsoid(self, label, center, semi, rot=(0, 0, 0), where=None, grow=0.0):
        self.paint(self.ellipsoid_mask(center, semi, rot, grow), label, where)

    def capsule(self, label, p0, p1, r, where=None):
        self.paint(self.capsule_mask(p0, p1, r), label, where)

    def polyline(self, label, pts, r, where=None):
        for a, b in zip(pts[:-1], pts[1:]):
            self.capsule(label, a, b, r, where)


def _folding(X, Y, Z, n=14, freq=9.0, seed=7):
    """Smooth pseudo-random function of direction, roughly in [-1, 1]; used to
    fold the cortical ribbon and carve sulci."""
    from .backend import xp_of
    xp = xp_of(X, Y, Z)
    rng = np.random.default_rng(seed)
    r = xp.sqrt(X**2 + Y**2 + Z**2) + 1e-6
    ux, uy, uz = X / r, Y / r, Z / r
    g = xp.zeros(np.broadcast_shapes(X.shape, Y.shape, Z.shape), np.float32)
    for _ in range(n):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        f = freq * rng.uniform(0.6, 1.4)
        g += xp.sin(f * (w[0] * ux + w[1] * uy + w[2] * uz) + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    return g * np.float32(np.sqrt(2.0 / n))


# lateral ventricles (bodies + atria): (center, semi-axes, rotation) per hemisphere
LATERAL_VENTRICLES = [shape for s in (-1, 1) for shape in (
    ((s * 11, 2, 14), (5.5, 24, 8), (s * 8, -10, 0)),
    ((s * 22, -26, 6), (4, 10, 6), (s * 20, 0, 0)))]                       # atria


def ventricle_mask(grid: Grid) -> np.ndarray:
    """Lateral ventricles of the analytic head as painted by build_labels (before
    later structures override parts of them), for seeding the DMV network."""
    P = _Painter(grid)
    for c, sm, r in LATERAL_VENTRICLES:
        P.ellipsoid(L.CSF, c, sm, rot=r)
    return P.labels == L.CSF


def build_labels(grid: Grid | None, axes=None, xp=np):
    """Label map of the analytic head on the grid, or on the point set with these coordinate axes (mm)."""
    P = _Painter(grid, axes, xp)

    head_c, head_s = (0, 0, -6), (76, 96, 92)
    face_c, face_s = (0, 60, -64), (46, 36, 32)
    icr_c, icr_s = (0, 0, -6), (65, 85, 81)          # intracranial (inner table)
    brain_c, brain_s = (0, 0, -6), (62.5, 82.5, 78.5)

    # scalp, skull, intracranial CSF
    P.ellipsoid(L.SOFT, head_c, head_s)
    P.ellipsoid(L.BONE, head_c, (70, 90, 86))
    P.ellipsoid(L.CSF, icr_c, icr_s)

    # cortex: folded GM ribbon over WM, with sulcal CSF
    sl, inside = P.ellipsoid_mask(brain_c, brain_s)
    ax = [xp.asarray(a[s]) for a, s in zip(P.ax, sl)]
    X, Y, Z = ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :] - brain_c[2]
    rho = xp.sqrt((X / brain_s[0]) ** 2 + (Y / brain_s[1]) ** 2 + (Z / brain_s[2]) ** 2)
    g1 = _folding(X, Y, Z, seed=7)
    g2 = _folding(X, Y, Z, freq=14.0, seed=11)
    view = P.labels[sl]
    view[inside] = np.uint8(L.GM)
    view[inside & (rho < 0.90 + 0.05 * g1)] = np.uint8(L.WM)
    view[inside & (rho > 0.92) & (g2 > 0.8)] = np.uint8(L.CSF)       # sulci
    view[inside & (abs(X) < 1.2) & (Z > 5)] = np.uint8(L.CSF)     # interhemispheric fissure
    del X, Y, Z, rho, g1, g2, inside

    # cerebellum (GM with WM core) and brainstem
    P.ellipsoid(L.CSF, (0, -52, -50), (49, 29, 25), where=[L.CSF, L.GM, L.WM])
    P.ellipsoid(L.GM, (0, -52, -50), (46, 26, 22), where=[L.CSF, L.GM, L.WM])
    P.ellipsoid(L.WM, (0, -48, -47), (26, 13, 11))
    for s in (-1, 1):
        P.ellipsoid(L.DENTATE, (s * 14, -54, -44), (6, 8, 5), rot=(s * 25, 0, 0))
    P.capsule(L.WM, (0, -18, -22), (0, -30, -96), 11.0, where=[L.CSF, L.GM, L.WM])

    # ventricles
    for c, sm, r in LATERAL_VENTRICLES:
        P.ellipsoid(L.CSF, c, sm, rot=r)
    P.ellipsoid(L.CSF, (0, -6, 0), (2.0, 11, 9))                               # third ventricle

    # deep grey nuclei
    for s in (-1, 1):
        P.ellipsoid(L.THALAMUS, (s * 10, -16, 5), (7.5, 13, 8), rot=(s * 10, 0, 0))
        P.ellipsoid(L.CAUDATE, (s * 15.5, 14, 13), (5, 10, 7), rot=(s * -12, -15, 0))
        P.ellipsoid(L.PUTAMEN, (s * 25, 2, 2), (5.5, 15, 10), rot=(s * 6, 0, 0))
        P.ellipsoid(L.GP, (s * 18.5, -1, 0), (3.5, 8, 6), rot=(s * 6, 0, 0))
        P.ellipsoid(L.RN, (s * 4.5, -20, -11), (3.5, 3.5, 3.5))
        P.ellipsoid(L.SN, (s * 10, -16, -16), (3, 9, 3), rot=(s * -30, 0, 0))

    # veins: SSS along the inner table (midline), straight sinus, internal
    # cerebral veins (perpendicular to B0), medullary veins (various angles).
    # The dural sinuses get their own label so the brain mask excludes them.
    t = np.deg2rad(np.linspace(25, 175, 60))
    sss = [(0, 80 * np.cos(a), -6 + 76 * np.sin(a)) for a in t]
    P.polyline(L.DURAL, sss, 4.0)
    P.capsule(L.DURAL, (0, -76, 2), (0, -28, 28), 2.5)                     # straight sinus
    for s in (-1, 1):
        P.capsule(L.VEIN, (s * 3, 8, 16), (s * 2, -30, 20), 1.5)            # ICVs
    P.capsule(L.VEIN, (34, 18, 18), (34, 18, 50), 1.2)                      # || B0
    P.capsule(L.VEIN, (-34, 26, 22), (-34, 26 + 24, 22 + 24), 1.2)          # 45 deg
    P.capsule(L.VEIN, (-38, -10, 30), (-38, -40, 30), 1.2)                  # perp B0

    # pathology
    P.ellipsoid(L.CALC, (36, -30, 32), (2.5, 2.5, 2.5))
    P.ellipsoid(L.BLEED, (-30, -46, 26), (2.0, 2.0, 2.0))

    # anterior skull base + face with paranasal sinuses (overrides brain)
    P.ellipsoid(L.BONE, face_c, face_s, grow=6.0, where=[L.CSF, L.GM, L.WM, L.SOFT, L.VEIN, L.DURAL])
    P.ellipsoid(L.SOFT, face_c, face_s)
    sinuses = [((0, 30, -50), (11, 10, 9), (0, 0, 0)),                     # sphenoid
               ((0, 50, -36), (8, 12, 7), (0, 0, 0)),                      # ethmoid
               ((0, 74, 8), (13, 5, 10), (0, -20, 0))]                     # frontal
    sinuses += [((s * 18, 58, -72), (12, 15, 13), (0, 0, 0)) for s in (-1, 1)]  # maxillary
    for c, sm, r in sinuses:
        P.ellipsoid(L.BONE, c, sm, r, grow=2.5)
    for c, sm, r in sinuses:
        P.ellipsoid(L.SINUS, c, sm, r)
    return P.labels


def build_phantom(fov_mm=(240, 240, 204), voxel_mm=1.5, oversample=1, texture_ppm=0.005,
                  texture_sigma_mm=3.0, seed=1, lut=None) -> Phantom:
    """Voxelize the phantom on a grid of voxel_mm/oversample over fov_mm. lut: (chi, M0, R2*) per label
    (tissue_lut) in place of TISSUES."""
    vox = voxel_mm / oversample
    shape = tuple(int(round(f / voxel_mm)) * oversample for f in fov_mm)
    grid = Grid(shape, (vox, vox, vox))
    labels = build_labels(grid)

    lut = tissue_lut() if lut is None else np.asarray(lut, np.float32)
    chi, m0, r2s = (lut[labels, i] for i in range(3))

    if texture_ppm > 0:
        tex = texture_field(shape, vox, texture_ppm, texture_sigma_mm, seed)
        sel = np.isin(labels, TEXTURED)
        chi[sel] += tex[sel]

    return Phantom(grid, labels, chi, m0, r2s,
                   meta=dict(fov_mm=fov_mm, voxel_mm=voxel_mm, oversample=oversample,
                             texture_ppm=texture_ppm))


def texture_field(shape, voxel_mm, texture_ppm=0.005, texture_sigma_mm=3.0, seed=1):
    """The smooth random susceptibility texture (ppm, SD texture_ppm) added to TEXTURED tissue."""
    rng = np.random.default_rng(seed)
    tex = rng.standard_normal(shape, dtype=np.float32)
    tex = ndimage.gaussian_filter(tex, texture_sigma_mm / voxel_mm)
    tex *= np.float32(texture_ppm / (tex.std() + 1e-12))
    return tex


def block_mean(a, f):
    """Average non-overlapping f x f x f blocks (f=1 returns a)."""
    if f == 1:
        return a
    s = a.shape
    return a.reshape(s[0] // f, f, s[1] // f, f, s[2] // f, f).mean(axis=(1, 3, 5))


def brain_mask(labels, voxel_mm, erode_mm=3.0):
    """Processing mask: intracranial compartment (parenchyma + CSF), holes
    filled, eroded by erode_mm - a stand-in for a conservative skull strip.
    `labels` may be a label map or an already-computed intracranial bool mask."""
    m = labels if labels.dtype == bool else np.isin(labels, INTRACRANIAL)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_fill_holes(m)
    r = int(round(erode_mm / voxel_mm))
    if r > 0:
        m = ndimage.binary_erosion(m, structure=ball(r))
    return m


def ball(r):
    """Spherical structuring element of radius r voxels."""
    z = np.arange(-r, r + 1)
    return (z[:, None, None] ** 2 + z[None, :, None] ** 2 + z[None, None, :] ** 2) <= r * r + 1e-9


def roi_masks(labels_fine, oversample, min_fraction=0.9):
    """Coarse-grid ROI masks: voxels at least min_fraction covered by the ROI."""
    out = {}
    for name, labs in ROIS.items():
        frac = block_mean(np.isin(labels_fine, labs).astype(np.float32), oversample)
        out[name] = frac >= min_fraction
    return out
