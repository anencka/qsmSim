"""Forward model v2 (GPU_Plan.md, phase 2): every voxel sampled at the fine spacing, in tiles over the master mesh.

Grids. The FOV is a box centred on the origin. The master grid (spacing h, e.g. 0.5 mm) carries the mesh; the
fine grid (h_f = h / F, F odd, e.g. 0.1 mm) subdivides every master cell, and the fine point at the centre of a
master cell is the master sample point, so v2 differs from v1 only below the master spacing. Acquisition voxels
(e.g. 0.8-2.0 mm) are blocks of fine cells, aligned to the FOV corner.

Field. The total field at the fine points is split in two:
  mesh   the field of the block-averaged (master) susceptibility from one padded FFT over the FOV (far field,
         air/bone background), plus the shim;
  tiles  the FOV is cut into tiles (core + halo, aligned to the FOV corner). In each, the fine susceptibility q
         and its master block average B q get their fields from one periodic FFT each, and
             field = ff(q) + trilinear(mesh - fc(B q))
         (the particle-mesh split of finegrid.py, now for all tissue). Tile truncation and wrap-around affect ff
         and fc alike and cancel to the extent that the perturbation q - B q, mean-zero in every master cell, has
         a short-range field (acceptance G.2).
Veins. The intravascular fraction is rasterized once, sparse, at h_f / R (R = vein_refine); the fine level uses
its R^3 block means. In refine sub-tiles (core + halo) that contain vein, every fine cell is split R^3 times and
the vein perturbation gets the same split one level down, tissue being constant within a fine cell:
             field_R = ff_R(dchi_R) + trilinear_R(field - fc(B_R dchi_R)).
Signal. Per point S = [(1 - f) M0 e^{-R2* TE} + f M0_v e^{-R2*_v TE}] exp(i s [phi0 + 2 pi gamma B0 field TE]);
a refined fine cell takes the mean over its R^3 points; an acquisition voxel the mean over its fine cells. The
truth is accumulated the same way (block-mean chi and field, tissue fractions). Nothing at the fine spacing is
kept: tiles are computed on the device (xp = numpy or cupy, backend.py), the mesh and outputs stay on the host.

tissue_sub=False (acceptance G.3): MasterSource, tissue constant per master cell, and the tile FFTs see only the
vein perturbation: v1's model (finegrid.py) with every master voxel sub-sampled.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import backend as be
from . import dipole as dp
from . import phantom as ph
from .signal import hz_per_ppm
from .veins import fraction_on_grid


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@dataclass
class Geom:
    shape_m: tuple              # master grid shape
    h: float                    # master spacing (mm)
    F: int                      # fine points per master cell edge (odd in production: config.check_v2)
    core: int                   # tile core edge (fine cells)
    halo: int                   # tile halo (fine cells, a multiple of F)
    offset_mm: tuple = (0.0, 0.0, 0.0)  # coordinates of the FOV centre in the phantom's frame (a cube cut from a head)

    def __post_init__(self):
        self.shape_m = tuple(int(n) for n in self.shape_m)
        self.offset_mm = tuple(float(v) for v in self.offset_mm)
        if self.core % self.F or self.halo % self.F:          # odd F puts a fine point on every master centre
            raise ValueError(f"F={self.F} must divide the tile core ({self.core}) and halo ({self.halo})")

    @property
    def hf(self):
        return self.h / self.F

    @property
    def shape_f(self):
        return tuple(n * self.F for n in self.shape_m)

    def coords(self, d, j, R=1):
        """Coordinates (mm) along axis d of fine indices j (R = 1) or of indices j on the grid refined R times.
        Indices may lie outside the FOV. The centre fine point of a master cell has exactly the master grid's
        coordinate (phantom.Grid.axes)."""
        j = np.asarray(j)
        if R > 1:
            jf = j // R
            return self.coords(d, jf) + (j - jf * R - (R - 1) / 2.0) * (self.hf / R)
        m = j // self.F
        n = self.shape_m[d]
        x = (m - (n - 1) / 2.0) * self.h + (j - m * self.F - (self.F - 1) / 2.0) * self.hf
        return x + self.offset_mm[d] if self.offset_mm[d] else x

    def axes(self, box, R=1):
        return [self.coords(d, np.arange(b0, b1), R) for d, (b0, b1) in enumerate(box)]

    def tiles(self):
        """Tile cores [(j0, j1)] * 3 (fine indices), aligned to the FOV corner; the last one along an axis may be
        partial."""
        rng = [[(s, min(s + self.core, n)) for s in range(0, n, self.core)] for n in self.shape_f]
        return [(a, b, c) for a in rng[0] for b in rng[1] for c in rng[2]]


def _grow(box, n):
    return tuple((b0 - n, b1 + n) for b0, b1 in box)


def _shape(box):
    return tuple(b1 - b0 for b0, b1 in box)


def _rel(box, ref):
    """box as slices relative to the box ref."""
    return tuple(slice(b0 - r0, b1 - r0) for (b0, b1), (r0, _) in zip(box, ref))


# ---------------------------------------------------------------------------
# master grid -> fine points
# ---------------------------------------------------------------------------

def master_box(arr, m0, m1):
    """arr[m0:m1] per axis, indices clamped to the array (edge values repeated beyond the FOV)."""
    idx = [np.clip(np.arange(a, b), 0, n - 1) for a, b, n in zip(m0, m1, arr.shape)]
    return arr[np.ix_(*idx)]


def _master_range(box, F):
    """Master cells a cell-centred trilinear interpolation needs for the fine points in box."""
    m0 = [int(math.floor((b0 + 0.5) / F - 0.5)) for b0, _ in box]
    m1 = [int(math.floor((b1 - 0.5) / F - 0.5)) + 2 for _, b1 in box]
    return m0, m1


def upsample(a, F, box, m0, mode="linear"):
    """Values at the fine indices of box from the master-grid array a, whose first cell is master cell m0 (per
    axis). mode "linear": trilinear in cell-centred coordinates, edge values held (map_coordinates order=1,
    mode='nearest'); "constant": the value of the containing cell."""
    xp = be.xp_of(a)
    for d, ((j0, j1), o) in enumerate(zip(box, m0)):
        n = a.shape[d]
        j = np.arange(j0, j1)
        if mode == "constant":
            a = xp.take(a, xp.asarray(np.clip(j // F - o, 0, n - 1)), axis=d)
            continue
        c = (j + 0.5) / F - 0.5
        i = np.floor(c).astype(np.int64)
        sh = [1, 1, 1]
        sh[d] = -1
        w = xp.asarray((c - i).astype(np.float32)).reshape(sh)
        a0 = xp.take(a, xp.asarray(np.clip(i - o, 0, n - 1)), axis=d)
        a1 = xp.take(a, xp.asarray(np.clip(i + 1 - o, 0, n - 1)), axis=d)
        a = a0 + w * (a1 - a0)
    return a


def _taps(c, n, mode):
    """Per-axis gather indices (clamped to 0..n-1) and weights for fractional indices c (numpy)."""
    i = np.floor(c).astype(np.int64)
    t = (c - i).astype(np.float32)
    if mode == "linear":
        return [np.clip(i, 0, n - 1), np.clip(i + 1, 0, n - 1)], [1 - t, t]
    if mode == "cubic":                                   # cubic B-spline on prefiltered coefficients
        w = [(1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6, (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6]
        return [np.clip(i + k, 0, n - 1) for k in (-1, 0, 1, 2)], w
    raise ValueError(mode)


def sample_grid(a, cs, mode="linear"):
    """a sampled at the tensor grid of fractional array indices cs = [c_x, c_y, c_z] (separable; edge values held).
    mode "cubic" expects B-spline coefficients (scipy.ndimage.spline_filter of the data)."""
    xp = be.xp_of(a)
    for d, c in enumerate(cs):
        sh = [1, 1, 1]
        sh[d] = -1
        idx, w = _taps(np.asarray(c, float), a.shape[d], mode)
        out = None
        for i, wk in zip(idx, w):
            term = xp.take(a, xp.asarray(i), axis=d) * xp.asarray(wk).reshape(sh)
            out = term if out is None else out + term
        a = out
    return a


def from_master(arr, geom, box, xp, mode="linear"):
    """Master-grid host array -> its values at the fine points of box, on the device of xp."""
    m0, m1 = _master_range(box, geom.F)
    return upsample(xp.asarray(master_box(arr, m0, m1)), geom.F, box, m0, mode)


def _outside(geom, box, xp):
    """Boolean (broadcastable) mask of the fine points of box outside the FOV."""
    out = None
    for d, (b0, b1) in enumerate(box):
        j = np.arange(b0, b1)
        o = (j < 0) | (j >= geom.shape_f[d])
        if o.any():
            sh = [1, 1, 1]
            sh[d] = -1
            o = xp.asarray(o).reshape(sh)
            out = o if out is None else (out | o)
    return out


def _blank_outside(t, geom, box, xp, chi_outside):
    o = _outside(geom, box, xp)
    if o is None:
        return t
    o = xp.broadcast_to(o, t["chi"].shape)
    t["labels"] = xp.where(o, 0, t["labels"]).astype(np.uint8)
    t["chi"] = xp.where(o, np.float32(chi_outside), t["chi"])
    for k in ("m0", "r2s"):
        t[k] = xp.where(o, np.float32(0), t[k])
    t["support"] = t["support"] & ~o
    return t


# ---------------------------------------------------------------------------
# phantom sources: tissue properties at fine points
# ---------------------------------------------------------------------------

class MasterSource:
    """Tissue constant over every master cell: the v1 master arrays (finegrid.Master). With it the tile FFTs see
    only the veins (v1's tissue model; acceptance G.3)."""
    tile_tissue = False

    def __init__(self, M, groups):
        self.M, self.groups, self.chi_outside = M, groups, float(M.chi_outside)

    def paint(self, geom, box, xp):
        M = self.M
        g = lambda a: from_master(a, geom, box, xp, "constant")
        t = dict(labels=g(M.labels), chi=g(M.chi), m0=g(M.m0), r2s=g(M.r2s), phi0=g(M.phi0), support=g(M.support))
        return _blank_outside(t, geom, box, xp, self.chi_outside)


class AnalyticSource:
    """The analytic head (P1/P2) at every fine point: labels painted at the point (phantom.build_labels), tissue
    values from the label, plus the master-grid texture and transceive phase interpolated trilinearly. At master
    cell centres this reproduces the v1 master arrays exactly."""
    tile_tissue = True

    def __init__(self, texture, phi0, r2s_scale=1.0, lut=None):
        """lut: (chi, M0, R2*) per label (phantom.tissue_lut) in place of TISSUES; r2s_scale must then be 1."""
        self.tex, self.phi0, self.r2s_scale = texture, phi0, float(r2s_scale)
        self.lut = ph.tissue_lut()
        if lut is not None:
            if self.r2s_scale != 1.0:
                raise ValueError("an explicit tissue table takes no r2s_scale")
            self.lut = np.array(lut, np.float32)
        self.chi_outside = float(ph.TISSUES[ph.L.AIR][0])
        self.groups = {**{k: list(map(int, v)) for k, v in ph.ROIS.items()},
                       "intracranial": list(map(int, ph.INTRACRANIAL))}

    def paint(self, geom, box, xp):
        lab = ph.build_labels(None, axes=geom.axes(box), xp=xp)
        lut = xp.asarray(self.lut)
        chi, m0, r2s = lut[lab, 0], lut[lab, 1], lut[lab, 2]
        if self.tex is not None:
            tex = from_master(self.tex, geom, box, xp)
            chi = xp.where(xp.isin(lab, xp.asarray(ph.TEXTURED, dtype=np.uint8)), chi + tex, chi)
        if self.r2s_scale != 1.0:
            r2s = (r2s * np.float32(self.r2s_scale)).astype(np.float32)
        t = dict(labels=lab, chi=chi, m0=m0, r2s=r2s, phi0=from_master(self.phi0, geom, box, xp),
                 support=xp.isin(lab, xp.asarray(ph.INTRACRANIAL, dtype=np.uint8)))
        return _blank_outside(t, geom, box, xp, self.chi_outside)


class SharpenedSource:
    """A labelled anatomical volume (e.g. P3 from the RC2 head at 1 mm) with sharpened tissue surfaces at the fine
    points (GPU_Plan.md 2.2): each label's indicator is smoothed (Gaussian, sigma_mm) and evaluated at the point
    by cubic B-spline interpolation; the point takes the label with the largest value, after per-label offsets
    calibrated so that every label keeps its volume (sigma 0.5 mm eroded the one-voxel vessels by a third). chi, M0 and R2* come from
    per-label value fields (the label's own voxels, extended outward by normalized convolution), interpolated
    trilinearly from coefficients chosen so that a voxel's mean over its fine points is the voxel value
    (mean-preserving); values do not bleed across the new surfaces. The support (brain mask) is sharpened the same way;
    chi is zero outside it. The sub-voxel surfaces are synthetic: they follow the coarse segmentation smoothly.

    vols: dict(seg, mask, chi, m0, r2s) on the coarse grid (spacing `voxel`, centred like the master grid);
    groups: label groups for the tissue fractions (key 'intracranial' = the support)."""
    tile_tissue = True

    def __init__(self, vols, voxel, phi0, groups, sigma_mm=0.25, ext_sigma_mm=1.5, r2s_max=None, calibrate=5,
                 log=print):
        from scipy import ndimage
        t0 = time.time()
        self.voxel, self.phi0, self.groups, self.chi_outside = float(voxel), phi0, groups, 0.0
        self.r2s_max = r2s_max
        seg, mask = vols["seg"], vols["mask"]
        self.shape1 = seg.shape
        self.o1 = np.array([-(n - 1) / 2.0 * self.voxel for n in seg.shape])
        sig = sigma_mm / self.voxel

        def boxed(ind, thr=1e-4, grow=3):
            nz = np.argwhere(ind > thr)
            lo = np.maximum(nz.min(0) - grow, 0)
            hi = np.minimum(nz.max(0) + 1 + grow, seg.shape)
            return lo, hi

        def coef(ind):
            sm = ndimage.gaussian_filter(ind.astype(np.float32), sig, mode="nearest")
            lo, hi = boxed(sm)
            c = ndimage.spline_filter(sm[tuple(slice(a, b) for a, b in zip(lo, hi))], order=3, mode="nearest")
            return lo, c.astype(np.float32)

        self.labels, self.ind, self.val = [], {}, {}
        es = ext_sigma_mm / self.voxel
        for L in np.unique(seg):
            m = seg == L
            self.labels.append(int(L))
            self.ind[int(L)] = coef(m)
            w = ndimage.gaussian_filter(m.astype(np.float32), es, mode="nearest")
            lo, hi = boxed(w, thr=1e-3)
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            fields = {}
            for k in ("chi", "m0", "r2s"):
                mm = m[sl] & mask[sl] if k == "chi" else m[sl]       # chi: only the label's voxels inside the mask
                if not mm.any():
                    fields[k] = np.zeros(mm.shape, np.float32)
                    continue
                ww = ndimage.gaussian_filter(mm.astype(np.float32), es, mode="nearest")
                v = vols[k][sl].astype(np.float32)
                num = ndimage.gaussian_filter(np.where(mm, v, 0).astype(np.float32), es, mode="nearest")
                ext = np.where(ww > 1e-6, num / np.maximum(ww, 1e-6), float(v[mm].mean()))
                # coefficients whose trilinear interpolant has the voxel values as its voxel means: the box mean
                # of a cell-centred linear interpolant is [1/8, 3/4, 1/8] per axis, the order-2 B-spline prefilter
                fields[k] = ndimage.spline_filter(np.where(mm, v, ext).astype(np.float32), order=2,
                                                  mode="nearest").astype(np.float32)
            self.val[int(L)] = (lo, fields)
        self.mask_coef = coef(mask)
        self.bias = {L: 0.0 for L in self.labels}
        log(f"sharpened source: {len(self.labels)} labels, sigma {sigma_mm} mm ({time.time() - t0:.0f}s)")
        if calibrate:
            self.calibrate(seg, calibrate, log=log)

    def calibrate(self, seg, F=5, iters=4, xp=None, log=print):
        """Per-label offsets added to the interpolated indicators so that every label keeps its volume (thin
        structures - vessels one voxel wide - otherwise lose volume to the argmax). Volumes are counted at F points
        per coarse voxel edge over the whole volume; offsets move by the relative volume error each round."""
        xp = xp if xp is not None else (be.cupy() if be.use_gpu() else np)
        t0 = time.time()
        ref = {L: float((seg == L).sum()) for L in self.labels}
        g = Geom(seg.shape, self.voxel, F, core=24 * F, halo=0)
        for it in range(iters):
            vol = dict.fromkeys(self.labels, 0.0)
            for core in g.tiles():
                lab = self._labels(g, core, xp)
                cnt = be.to_host(xp.bincount(lab.ravel(), minlength=256))
                for L in self.labels:
                    vol[L] += float(cnt[L]) / F**3
            dev = {L: vol[L] / ref[L] - 1 for L in self.labels if ref[L] >= 100}
            worst = max(dev, key=lambda L: abs(dev[L]))
            log(f"  sharpening calibration round {it + 1}: worst label {worst} volume {dev[worst]:+.2%} "
                f"({time.time() - t0:.0f}s)")
            if abs(dev[worst]) < 0.005 or it == iters - 1:
                break
            for L, e in dev.items():
                self.bias[L] -= float(np.clip(e, -0.1, 0.1))

    def _cs(self, geom, box, lo, shape):
        return [(geom.coords(d, np.arange(b0, b1)) - self.o1[d]) / self.voxel - lo[d] for d, (b0, b1) in enumerate(box)]

    def _hits(self, cs, shape):
        return all(c[-1] >= -2 and c[0] <= n + 1 for c, n in zip(cs, shape))

    def _labels(self, geom, box, xp):
        shp = _shape(box)
        best = xp.full(shp, -np.inf, np.float32)
        lab = xp.zeros(shp, np.uint8)
        for L in self.labels:
            lo, c = self.ind[L]
            cs = self._cs(geom, box, lo, c.shape)
            if not self._hits(cs, c.shape):
                continue
            # outside the label's box the smoothed indicator is ~0: held edge values there are tiny as well
            v = sample_grid(xp.asarray(c), cs, "cubic") + np.float32(self.bias[L])
            upd = v > best
            best = xp.where(upd, v, best)
            lab = xp.where(upd, np.uint8(L), lab)
        return lab

    def _block(self, geom, box):
        """k fine points per voxel edge if every voxel of the volume in box is whole (k points per axis), else None."""
        k = int(round(self.voxel / geom.hf))
        if k < 2 or abs(k * geom.hf - self.voxel) > 1e-9 * self.voxel:
            return None
        for d, (b0, b1) in enumerate(box):
            if (b1 - b0) % k:
                return None
            i = np.floor((geom.coords(d, np.arange(b0, b1)) - self.o1[d]) / self.voxel + 0.5).reshape(-1, k)
            if not (i == i[:, :1]).all():
                return None
        return k

    @staticmethod
    def _nonneg(a, k, xp):
        """M0 / R2* >= 0. The mean-preserving coefficients overshoot below 0 next to zero-valued voxels (CSF R2* is a
        noisy fit floored at 0); clipping alone would raise those voxels' means, so the clipped points of each voxel
        are scaled back to the voxel's unclipped mean (factor 1 where nothing was clipped). k None: clip only."""
        c = xp.maximum(a, np.float32(0))
        if k is None:
            return c
        n = [s // k for s in a.shape]
        b = (n[0], k, n[1], k, n[2], k)
        want = xp.maximum(a.reshape(b).mean((1, 3, 5)), np.float32(0))
        got = c.reshape(b).mean((1, 3, 5))
        fac = xp.where(got > 0, want / xp.maximum(got, np.float32(1e-30)), np.float32(0)).astype(np.float32)
        return (c.reshape(b) * fac[:, None, :, None, :, None]).reshape(a.shape)

    def paint(self, geom, box, xp):
        shp = _shape(box)
        lab = self._labels(geom, box, xp)
        lo, c = self.mask_coef
        sup = sample_grid(xp.asarray(c), self._cs(geom, box, lo, c.shape), "cubic") > 0.5
        out = {k: xp.zeros(shp, np.float32) for k in ("chi", "m0", "r2s")}
        for L in self.labels:
            lo, fields = self.val[L]
            n = fields["chi"].shape
            cs = self._cs(geom, box, lo, n)
            if not self._hits(cs, n):
                continue
            sel = lab == L
            if not bool(sel.any()):
                continue
            for k in out:
                out[k] = xp.where(sel, sample_grid(xp.asarray(fields[k]), cs, "linear"), out[k])
        chi = xp.where(sup, out["chi"], np.float32(0))
        k = self._block(geom, box)
        m0 = self._nonneg(out["m0"], k, xp)
        r2s = self._nonneg(out["r2s"], k, xp)
        if self.r2s_max is not None:
            r2s = xp.minimum(r2s, np.float32(self.r2s_max))
        t = dict(labels=lab, chi=chi, m0=m0, r2s=r2s, phi0=from_master(self.phi0, geom, box, xp), support=sup)
        return _blank_outside(t, geom, box, xp, self.chi_outside)


# ---------------------------------------------------------------------------
# veins: sparse intravascular fraction
# ---------------------------------------------------------------------------

class VeinStore:
    """Intravascular fraction of a vein network, rasterized once (veins.fraction_on_grid) on the fine grid refined
    R times and kept sparse; fine-grid values are its R^3 block means. Only the FOV is stored."""

    def __init__(self, net, geom, R=1, slab_mm=1.0, log=print):
        t0 = time.time()
        self.geom, self.R = geom, int(R)
        hr = geom.hf / self.R
        self.shape_r = tuple(n * self.R for n in geom.shape_f)
        parts = []
        if len(net):
            seg = net.segments()
            p0, p1, r0, r1, _ = seg
            pad = float(np.maximum(r0, r1).max()) + 2 * hr
            lo = np.minimum(p0, p1).min(0) - pad
            hi = np.maximum(p0, p1).max(0) + pad
            o = np.array([geom.coords(d, 0, self.R) for d in range(3)])
            i0 = np.maximum(np.floor((lo - o) / hr).astype(int), 0)
            i1 = np.minimum(np.ceil((hi - o) / hr).astype(int) + 1, self.shape_r)
            step = max(1, int(round(slab_mm / hr)))
            for x0 in range(i0[0], i1[0], step):
                x1 = min(x0 + step, i1[0])
                org = [geom.coords(0, x0, self.R), geom.coords(1, i0[1], self.R), geom.coords(2, i0[2], self.R)]
                f = fraction_on_grid(net, org, (x1 - x0, i1[1] - i0[1], i1[2] - i0[2]), hr, seg=seg)
                nz = np.nonzero(f)
                if len(nz[0]):
                    parts.append((nz[0] + x0, nz[1] + i0[1], nz[2] + i0[2], f[nz]))
        cat = lambda k, dt: (np.concatenate([p[k] for p in parts]).astype(dt) if parts else np.zeros(0, dt))
        self.r = (cat(0, np.int32), cat(1, np.int32), cat(2, np.int32), cat(3, np.float32))
        if self.R == 1:
            self.f = self.r
        else:
            jx, jy, jz = (a // self.R for a in self.r[:3])
            nf = geom.shape_f
            key = (jx.astype(np.int64) * nf[1] + jy) * nf[2] + jz
            u, inv = np.unique(key, return_inverse=True)
            v = (np.bincount(inv, weights=self.r[3].astype(np.float64)) / self.R**3).astype(np.float32)
            kx, rem = np.divmod(u, nf[1] * nf[2])
            ky, kz = np.divmod(rem, nf[2])
            self.f = (kx.astype(np.int32), ky.astype(np.int32), kz.astype(np.int32), v)
        self.volume_mm3 = float(self.r[3].astype(np.float64).sum() * hr**3)
        log(f"veins: {len(self.r[0])} refined / {len(self.f[0])} fine cells with vein (R={self.R}), "
            f"intravascular volume {self.volume_mm3:.1f} mm^3 ({time.time() - t0:.0f}s)")

    @staticmethod
    def _select(S, box):
        ix, iy, iz, v = S
        a, b = np.searchsorted(ix, [box[0][0], box[0][1]])
        sel = slice(a, b)
        m = (iy[sel] >= box[1][0]) & (iy[sel] < box[1][1]) & (iz[sel] >= box[2][0]) & (iz[sel] < box[2][1])
        return ix[sel][m], iy[sel][m], iz[sel][m], v[sel][m]

    def _dense(self, S, box):
        ix, iy, iz, v = self._select(S, box)
        out = np.zeros(_shape(box), np.float32)
        out[ix - box[0][0], iy - box[1][0], iz - box[2][0]] = v
        return out

    def fine(self, box):
        """Fine-grid fraction over box (fine indices), dense, host."""
        return self._dense(self.f, box)

    def refined(self, box):
        """Fraction on the refined grid over box (refined indices), dense, host."""
        return self._dense(self.r, box)

    def count(self, box):
        return len(self._select(self.f, box)[0])


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

_KERNELS = {}


def _kernel(shape, vox, b0, xp):
    key = (tuple(shape), float(vox), tuple(float(v) for v in b0), xp is np)
    D = _KERNELS.get(key)
    if D is None:
        if len(_KERNELS) >= 8:
            _KERNELS.pop(next(iter(_KERNELS)))
        D = dp.dipole_kernel(shape, (vox,) * 3, b0, real=True)
        _KERNELS[key] = D = D if xp is np else xp.asarray(D)
    return D


def periodic_field(q, vox, b0=(0, 0, 1)):
    """Field (ppm) of q (ppm) periodic over its own box: dipole.forward_field(q, pad_factor=1) for an FFT-friendly
    shape (the same scipy calls on the CPU), on q's device."""
    shape = tuple(q.shape)
    if dp.padded_shape(shape, 1.0) != shape:
        raise ValueError(f"tile shape {shape} is not FFT-friendly; choose tile core/halo sizes accordingly")
    xp = be.xp_of(q)
    spec = dp.rfftn(q)
    spec *= _kernel(shape, vox, b0, xp)
    return xp.ascontiguousarray(dp.irfftn(spec, shape), dtype=np.float32)


def mesh_field(chi_master, h, b0=(0, 0, 1), pad_factor=2.0, chi_outside=0.0, shim_ppm=None):
    """Total field on the master grid (ppm): finegrid.mesh_field for a given master susceptibility."""
    f = dp.forward_field(chi_master, (h,) * 3, b0, pad_factor, background=chi_outside)
    if shim_ppm is not None:
        f += shim_ppm
    return f


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

@dataclass
class Model:
    geom: Geom
    source: object                      # MasterSource | AnalyticSource | ...
    veins: VeinStore
    vein: object                        # veins.VeinParams (chi_ppm, m0, r2s)
    acq: list                           # fine cells per acquisition voxel edge, one per output grid
    b0_dir: tuple = (0.0, 0.0, 1.0)
    refine_core: int = 0                # refine sub-tile core (fine cells); 0 = no refinement
    refine_halo: int = 0
    tile_list: list | None = None        # restrict both passes to these tile cores (acceptance checks); None = all
    stats: dict = dc_field(default_factory=dict)

    def tiles(self):
        return self.geom.tiles() if self.tile_list is None else list(self.tile_list)

    @property
    def R(self):
        return self.veins.R if self.refine_core else 1

    def _xp(self, xp):
        return xp if xp is not None else (be.cupy() if be.use_gpu() else np)

    def acq_shape(self, n):
        return tuple(s // n for s in self.geom.shape_f)

    def _paint(self, box, xp):
        t = self.source.paint(self.geom, box, xp)
        f = xp.asarray(self.veins.fine(box))
        t["f"] = f
        t["dchi"] = f * (np.float32(self.vein.chi_ppm) - t["chi"])
        return t

    # ---- pass 1: master susceptibility and the field-independent truth ------------------------------------------
    def mesh_pass(self, xp=None, log=print):
        """Paint every tile core: master susceptibility (block mean of the fine chi, veins included), and per
        output grid the block-mean chi, tissue fractions (unweighted label fractions; 'dmv' the intravascular
        fraction), the centre label. Returns (chi_master, truth{n: dict})."""
        xp = self._xp(xp)
        g = self.geom
        F = g.F
        chi_m = np.zeros(g.shape_m, np.float32)
        names = list(self.source.groups) + ["dmv"]
        truth = {n: dict(chi=np.zeros(self.acq_shape(n), np.float32), labels=np.zeros(self.acq_shape(n), np.uint8),
                         frac={k: np.zeros(self.acq_shape(n), np.float16) for k in names}) for n in self.acq}
        t0 = time.time()
        tiles = self.tiles()
        for it, core in enumerate(tiles):
            t = self._paint(core, xp)
            chi = t["chi"] + t["dchi"]
            chi_m[tuple(slice(b0 // F, b1 // F) for b0, b1 in core)] = be.to_host(ph.block_mean(chi, F))
            for n in self.acq:
                sl = tuple(slice(b0 // n, b1 // n) for b0, b1 in core)
                T = truth[n]
                T["chi"][sl] = be.to_host(ph.block_mean(chi, n))
                c = n // 2
                T["labels"][sl] = be.to_host(t["labels"][c::n, c::n, c::n])
                for k, labs in self.source.groups.items():
                    m = xp.isin(t["labels"], xp.asarray(labs, dtype=np.uint8)) if k != "intracranial" else t["support"]
                    T["frac"][k][sl] = be.to_host(ph.block_mean(m.astype(np.float32), n))
                T["frac"]["dmv"][sl] = be.to_host(ph.block_mean(t["f"], n))
            if it % 50 == 0 or it == len(tiles) - 1:
                log(f"  mesh pass: tile {it + 1}/{len(tiles)} ({time.time() - t0:.0f}s)")
        be.sync()
        self.stats["mesh_pass_s"] = time.time() - t0
        return chi_m, truth

    # ---- pass 2: fields and signals -------------------------------------------------------------------------------
    def tile_field(self, core, mesh, xp):
        """Tissue, vein fraction and total field over the tile region (core + halo); the region box."""
        g = self.geom
        reg = _grow(core, g.halo)
        t = self._paint(reg, xp)
        if self.source.tile_tissue:
            q = t["chi"] + t["dchi"]
            q = q - np.float32(float(q.mean(dtype=np.float64)))     # D(0) = 0: smaller numbers, same field
        else:
            q = t["dchi"]
        ff = periodic_field(q, g.hf, self.b0_dir)
        fc = periodic_field(ph.block_mean(q, g.F), g.h, self.b0_dir)
        del q
        m0 = [b0 // g.F for b0, _ in reg]
        mreg = xp.asarray(master_box(mesh, m0, [b1 // g.F for _, b1 in reg]))
        rbox = tuple((0, b1 - b0) for b0, b1 in reg)
        t["field"] = ff + upsample(mreg - fc, g.F, rbox, (0, 0, 0))
        return t, reg

    def refine_tile(self, core, t, reg, xp):
        """Refined fields of the vein-bearing refine sub-tiles of a tile core: [(sub-core box, field_R, f_R)]."""
        if not self.refine_core:
            return []
        g, R, rc, rh = self.geom, self.R, self.refine_core, self.refine_halo
        hf = g.hf
        out = []
        subs = [[(s, min(s + rc, b1)) for s in range(b0, b1, rc)] for b0, b1 in core]
        for sx in subs[0]:
            for sy in subs[1]:
                for sz in subs[2]:
                    sc = (sx, sy, sz)
                    if not self.veins.count(sc):
                        continue
                    sr = _grow(sc, rh)
                    fR = xp.asarray(self.veins.refined(tuple((R * b0, R * b1) for b0, b1 in sr)))
                    chi_t = t["chi"][_rel(sr, reg)]
                    rbox = tuple((0, R * (b1 - b0)) for b0, b1 in sr)
                    dchi = fR * (np.float32(self.vein.chi_ppm) - upsample(chi_t, R, rbox, (0, 0, 0), "constant"))
                    ffR = periodic_field(dchi, hf / R, self.b0_dir)
                    fc = periodic_field(ph.block_mean(dchi, R), hf, self.b0_dir)
                    del dchi
                    cb = tuple((R * rh, R * (rh + b1 - b0)) for b0, b1 in sc)     # sub-core in the refined box
                    fld = ffR[_rel(cb, rbox)] + upsample(t["field"][_rel(sr, reg)] - fc, R, cb, (0, 0, 0))
                    out.append((sc, fld, fR[_rel(cb, rbox)]))
        return out

    def signal_pass(self, mesh, tes, b0_tesla, phase_sign=1.0, sinks=None, xp=None, log=print):
        """Noiseless signals {n: (acq shape + (len(tes),)) complex64} (written into sinks[n] if given, e.g. memmaps)
        and the block-mean total field {n: acq shape}."""
        xp = self._xp(xp)
        R = self.R
        ne = len(tes)
        sig = sinks or {n: np.zeros(self.acq_shape(n) + (ne,), np.complex64) for n in self.acq}
        fld_t = {n: np.zeros(self.acq_shape(n), np.float32) for n in self.acq}
        ws = [np.float32(2 * np.pi * hz_per_ppm(b0_tesla) * te) for te in tes]
        vamp = [np.float32(self.vein.m0 * np.exp(-self.vein.r2s * te)) for te in tes]
        sgn = np.float32(phase_sign)
        t0 = time.time()
        tiles = self.tiles()
        n_ref = 0
        for it, core in enumerate(tiles):
            t, reg = self.tile_field(core, mesh, xp)
            refined = self.refine_tile(core, t, reg, xp)
            n_ref += len(refined)
            cs = _rel(core, reg)
            m0, r2s, phi0, fld, f = (t[k][cs] for k in ("m0", "r2s", "phi0", "field", "f"))
            subs = []
            for sc, fld_r, f_r in refined:                  # fine cell -> its R^3 points along axes 1, 3, 5
                s = _rel(sc, core)
                k = tuple(b1 - b0 for b0, b1 in sc)
                ex = lambda a: a[s].reshape(k[0], 1, k[1], 1, k[2], 1)
                subs.append((s, ex(m0), ex(r2s), ex(phi0), fld_r.reshape(k[0], R, k[1], R, k[2], R),
                             f_r.reshape(k[0], R, k[1], R, k[2], R)))
            del t
            for n in self.acq:
                fld_t[n][tuple(slice(b0 // n, b1 // n) for b0, b1 in core)] = be.to_host(ph.block_mean(fld, n))
            for e, te in enumerate(tes):
                S = _signal(m0, r2s, phi0, fld, f, te, ws[e], vamp[e], sgn)
                for s, sm0, sr2, sp0, sfld, sf in subs:
                    S[s] = _signal(sm0, sr2, sp0, sfld, sf, te, ws[e], vamp[e], sgn).mean(axis=(1, 3, 5))
                for n in self.acq:
                    sig[n][tuple(slice(b0 // n, b1 // n) for b0, b1 in core) + (e,)] = be.to_host(ph.block_mean(S, n))
            if it % 50 == 0 or it == len(tiles) - 1:
                log(f"  signal pass: tile {it + 1}/{len(tiles)}, {n_ref} refined sub-tiles ({time.time() - t0:.0f}s)")
        be.sync()
        self.stats.update(signal_pass_s=time.time() - t0, refined_subtiles=n_ref)
        return sig, fld_t


def _signal(m0, r2s, phi0, fld, f, te, w, vamp, sgn):
    xp = be.xp_of(fld)
    a = m0 * xp.exp(-r2s * np.float32(te))
    a = a + f * (vamp - a)
    return (a * xp.exp(1j * (sgn * (phi0 + w * fld)))).astype(np.complex64)
