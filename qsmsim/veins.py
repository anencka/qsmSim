"""Deep medullary vein (DMV) network for Experiment 1.

Geometry: veins start on the lateral-ventricle wall and run outward through
white matter (along the gradient of the distance from the ventricles, with a
persistent random wobble) until they leave WM or reach their drawn length.
Diameter tapers from the ventricle end toward the cortex. A few larger
"collector" veins run along the ventricle wall (anterior-posterior).

Rasterization: `fraction_on_grid` paints the intravascular volume fraction on a
(fine) grid using an area-conserving linear ramp across one cell, so even
0.1 mm veins sampled at 0.1 mm keep their true cross-sectional area.

`cylinder_field` is the analytic Lorentz-corrected field of an infinite
cylinder, used to validate the two-scale forward model (finegrid.py).
Coordinates are mm, origin at the FOV center, arrays (x, y, z) as elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

DMV_CHI = 0.45        # ppm vs water (Hct 0.4, Y ~ 0.6)
DMV_R2S = 45.0        # 1/s at 3 T, intravascular
DMV_M0 = 0.80


@dataclass
class VeinParams:
    n_per_hemi: int = 150
    n_collector_per_hemi: int = 10
    diam_start_mm: tuple = (0.2, 0.5)       # at the ventricle end; tapers to max(0.1, taper * start)
    taper: float = 0.4
    diam_min_mm: float = 0.1
    length_mm: tuple = (10.0, 35.0)
    min_length_mm: float = 10.0
    collector_diam_mm: tuple = (0.6, 1.0)
    collector_length_mm: tuple = (8.0, 15.0)
    step_mm: float = 0.5
    wobble: float = 0.15                    # random direction change per step (radians, ~)
    persistence: float = 0.7                # weight of the previous direction vs the outward gradient
    seed_band_mm: tuple = (1.0, 4.0)        # seed distance from the ventricle wall
    seed_spacing_mm: float = 2.0
    clearance_mm: float = 0.3               # minimum wall-to-wall gap between veins
    perp_bias: float = 1.5                  # seed acceptance ~ (1 - |n.b0|)^perp_bias: favour walls whose
                                            # outward normal is perpendicular to B0 (veins mostly in-plane)
    chi_ppm: float = DMV_CHI
    r2s: float = DMV_R2S
    m0: float = DMV_M0


@dataclass
class VeinNetwork:
    points: list                 # per vein (K, 3) mm
    radii: list                  # per vein (K,) mm
    kinds: list                  # "dmv" | "collector"
    params: VeinParams = field(default_factory=VeinParams)

    def __len__(self):
        return len(self.points)

    def segments(self):
        """p0, p1 (S, 3); r0, r1 (S,); vein index (S,)."""
        p0, p1, r0, r1, vid = [], [], [], [], []
        for i, (p, r) in enumerate(zip(self.points, self.radii)):
            p0.append(p[:-1]); p1.append(p[1:]); r0.append(r[:-1]); r1.append(r[1:])
            vid.append(np.full(len(p) - 1, i))
        cat = lambda a: np.concatenate(a) if a else np.zeros((0,))
        return cat(p0).reshape(-1, 3), cat(p1).reshape(-1, 3), cat(r0), cat(r1), cat(vid).astype(np.int32)

    def samples(self, step_mm=0.1):
        """Points along every centerline every step_mm: (P,3) xyz, radius, unit tangent, vein id."""
        xs, rs, ts, ids = [], [], [], []
        for i, (p, r) in enumerate(zip(self.points, self.radii)):
            seg = np.diff(p, axis=0)
            ln = np.linalg.norm(seg, axis=1)
            for k in range(len(seg)):
                m = max(1, int(np.ceil(ln[k] / step_mm)))
                t = (np.arange(m) + 0.5) / m
                xs.append(p[k] + t[:, None] * seg[k])
                rs.append(r[k] + t * (r[k + 1] - r[k]))
                ts.append(np.repeat((seg[k] / ln[k])[None], m, 0))
                ids.append(np.full(m, i))
        if not xs:
            z = np.zeros((0, 3))
            return z, np.zeros(0), z, np.zeros(0, np.int32)
        return (np.concatenate(xs), np.concatenate(rs), np.concatenate(ts),
                np.concatenate(ids).astype(np.int32))

    def table(self, b0=(0, 0, 1)):
        """Per-vein summary: length, mean/start/end diameter, mean angle to B0 (deg), kind."""
        b = np.asarray(b0, float) / np.linalg.norm(b0)
        rows = dict(length_mm=[], diam_mean_mm=[], diam_start_mm=[], diam_end_mm=[], angle_b0_deg=[], kind=[])
        for p, r, k in zip(self.points, self.radii, self.kinds):
            seg = np.diff(p, axis=0)
            ln = np.linalg.norm(seg, axis=1)
            cosang = np.abs(seg @ b) / ln
            rows["length_mm"].append(float(ln.sum()))
            rows["diam_mean_mm"].append(float(2 * np.average((r[:-1] + r[1:]) / 2, weights=ln)))
            rows["diam_start_mm"].append(float(2 * r[0]))
            rows["diam_end_mm"].append(float(2 * r[-1]))
            rows["angle_b0_deg"].append(float(np.degrees(np.arccos(np.clip(np.average(cosang, weights=ln), 0, 1)))))
            rows["kind"].append(k)
        return rows

    def to_npz(self, path):
        lens = np.array([len(p) for p in self.points], np.int64)
        np.savez_compressed(path, points=np.concatenate(self.points) if self.points else np.zeros((0, 3)),
                            radii=np.concatenate(self.radii) if self.radii else np.zeros(0),
                            lengths=lens, kinds=np.array(self.kinds), chi_ppm=self.params.chi_ppm,
                            r2s=self.params.r2s, m0=self.params.m0)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        idx = np.cumsum(d["lengths"])[:-1]
        P = VeinParams(chi_ppm=float(d["chi_ppm"]), r2s=float(d["r2s"]), m0=float(d["m0"]))
        return cls(list(np.split(d["points"], idx)), list(np.split(d["radii"], idx)),
                   [str(k) for k in d["kinds"]], P)


# ---------------------------------------------------------------------------
# geometry generation
# ---------------------------------------------------------------------------

class _Volume:
    """Nearest / trilinear lookups on a working grid (voxel centers at axes)."""

    def __init__(self, axes, voxel_mm):
        self.o = np.array([a[0] for a in axes])
        self.h = float(voxel_mm)
        self.shape = tuple(len(a) for a in axes)

    def idx(self, p):
        return (np.asarray(p) - self.o) / self.h

    def nearest(self, vol, p):
        i = np.rint(self.idx(p)).astype(int)
        if np.any(i < 0) or np.any(i >= self.shape):
            return 0
        return vol[tuple(i)]

    def linear(self, vols, p):
        c = self.idx(p).reshape(3, 1)
        return np.array([ndimage.map_coordinates(v, c, order=1, mode="nearest")[0] for v in vols])


def generate(wm, ventricles, axes, voxel_mm, params: VeinParams, rng, brain=None, b0=(0, 0, 1)):
    """Grow a DMV network on a working grid.

    wm, ventricles (and optionally brain) are boolean volumes on the grid whose
    voxel-center coordinates are `axes` (mm). Hemispheres are split at x = 0."""
    P = params
    V = _Volume(axes, voxel_mm)
    b0 = np.asarray(b0, float) / np.linalg.norm(b0)
    dv = ndimage.distance_transform_edt(~ventricles).astype(np.float32) * voxel_mm
    grad = [g.astype(np.float32) for g in np.gradient(dv, voxel_mm)]
    brain = wm if brain is None else brain
    X = axes[0]

    band = wm & (dv >= P.seed_band_mm[0]) & (dv <= P.seed_band_mm[1])
    cand = np.argwhere(band)
    cand_xyz = np.stack([axes[d][cand[:, d]] for d in range(3)], 1)

    pts_all, net_pts, net_r, kinds = [], [], [], []
    tree = None

    def clear_of_network(p, r):
        if tree is None:
            return True
        d, j = tree.query(p)
        return np.all(d >= r + tree_r[j] + P.clearance_mm)

    def outward(p):
        g = V.linear(grad, p)
        n = np.linalg.norm(g)
        return g / n if n > 1e-6 else None

    def trace_dmv(seed, length):
        p = seed.copy()
        d = outward(p)
        if d is None:
            return None
        path = [p.copy()]
        travelled = 0.0
        while travelled < length:
            g = outward(p)
            if g is None:
                break
            d = P.persistence * d + (1 - P.persistence) * g + rng.normal(0, P.wobble, 3)
            d /= np.linalg.norm(d)
            q = p + P.step_mm * d
            if not V.nearest(wm, q):
                break
            p = q
            path.append(p.copy())
            travelled += P.step_mm
        return np.array(path) if travelled >= P.min_length_mm else None

    def trace_collector(seed, length):
        n = outward(seed)
        if n is None:
            return None
        ey = np.array([0.0, 1.0, 0.0])
        d = ey - (ey @ n) * n
        if np.linalg.norm(d) < 0.3:
            return None
        d = d / np.linalg.norm(d) * (1 if rng.random() < 0.5 else -1)
        p, path, travelled = seed.copy(), [seed.copy()], 0.0
        while travelled < length:
            n = outward(p)
            if n is None:
                break
            t = d - (d @ n) * n
            t = t / (np.linalg.norm(t) + 1e-12)
            d = 0.8 * d + 0.2 * t + rng.normal(0, P.wobble / 2, 3)
            d /= np.linalg.norm(d)
            q = p + P.step_mm * d
            dq = V.nearest(dv, q)
            if not V.nearest(brain, q) or V.nearest(ventricles, q) or dq > 5.0:
                break
            p = q
            path.append(p.copy())
            travelled += P.step_mm
        return np.array(path) if travelled >= 0.6 * length else None

    def accept(path, radii, kind):
        nonlocal tree, tree_r
        dense, dr = _densify(path, radii, 0.25)
        if not clear_of_network(dense, dr):
            return False
        net_pts.append(path); net_r.append(radii); kinds.append(kind)
        pts_all.append((dense, dr))
        allp = np.concatenate([a for a, _ in pts_all]); allr = np.concatenate([b for _, b in pts_all])
        tree, tree_r = cKDTree(allp), allr
        return True

    tree_r = None
    for hemi in (-1, 1):
        sel = np.flatnonzero(np.sign(cand_xyz[:, 0]) == hemi)
        order = rng.permutation(sel)
        seeds_used = []
        # collectors first (larger, along the wall)
        n_col, tries = 0, 0
        for k in order:
            if n_col >= P.n_collector_per_hemi or tries > 50 * max(P.n_collector_per_hemi, 1):
                break
            tries += 1
            s = cand_xyz[k]
            L = rng.uniform(*P.collector_length_mm)
            path = trace_collector(s, L)
            if path is None:
                continue
            r = np.full(len(path), rng.uniform(*P.collector_diam_mm) / 2)
            if accept(path, r, "collector"):
                n_col += 1
                seeds_used.append(s)
        n_dmv, tries = 0, 0
        for k in rng.permutation(sel):
            if n_dmv >= P.n_per_hemi or tries > 200 * P.n_per_hemi:
                break
            tries += 1
            s = cand_xyz[k]
            if seeds_used and np.min(np.linalg.norm(np.array(seeds_used) - s, axis=1)) < P.seed_spacing_mm:
                continue
            n = outward(s)
            if n is None or rng.random() > (1 - abs(n @ b0)) ** P.perp_bias:
                continue
            path = trace_dmv(s, rng.uniform(*P.length_mm))
            if path is None:
                continue
            d0 = rng.uniform(*P.diam_start_mm)
            d1 = max(P.diam_min_mm, P.taper * d0)
            t = np.linspace(0, 1, len(path))
            r = (d0 + t * (d1 - d0)) / 2
            if accept(path, r, "dmv"):
                n_dmv += 1
                seeds_used.append(s)
    return VeinNetwork(net_pts, net_r, kinds, P)


def _densify(path, radii, step):
    xs, rs = [path[:1]], [radii[:1]]
    for k in range(len(path) - 1):
        ln = np.linalg.norm(path[k + 1] - path[k])
        m = max(1, int(np.ceil(ln / step)))
        t = np.arange(1, m + 1) / m
        xs.append(path[k] + t[:, None] * (path[k + 1] - path[k]))
        rs.append(radii[k] + t * (radii[k + 1] - radii[k]))
    return np.concatenate(xs), np.concatenate(rs)


# ---------------------------------------------------------------------------
# rasterization (area-conserving partial volume)
# ---------------------------------------------------------------------------

def _ramp_area(u, h):
    """Cross-sectional integral of clip((u - rho)/h + 1/2, 0, 1) over the plane."""
    rho = np.linspace(0, u + h / 2, 2001)
    f = np.clip((u - rho) / h + 0.5, 0, 1)
    return float(np.trapezoid(f * 2 * np.pi * rho, rho))


_EFF_TABLE = {}


def effective_radius(a, h):
    """Ramp center u such that the ramp profile has the area of a disc of radius a
    (vectorized over a; table cached per h)."""
    key = round(float(h), 9)
    if key not in _EFF_TABLE:
        us = np.linspace(0, 30 * h, 3001)
        areas = np.array([_ramp_area(u, h) for u in us])
        _EFF_TABLE[key] = (np.sqrt(areas / np.pi), us)
    a_tab, u_tab = _EFF_TABLE[key]
    return np.interp(np.asarray(a, float), a_tab, u_tab)


def fraction_on_grid(net: VeinNetwork, origin_mm, shape, h, seg=None):
    """Intravascular volume fraction on a grid of `shape` cells of size h whose
    first cell center is at origin_mm. Overlapping capsules combine by max."""
    out = np.zeros(shape, np.float32)
    p0, p1, r0, r1, _ = seg if seg is not None else net.segments()
    if len(p0) == 0:
        return out
    o = np.asarray(origin_mm, float)
    hi = o + (np.asarray(shape) - 1) * h
    rmax = np.maximum(r0, r1)
    lo_b = np.minimum(p0, p1) - rmax[:, None] - h
    hi_b = np.maximum(p0, p1) + rmax[:, None] + h
    hit = np.all((hi_b >= o) & (lo_b <= hi), axis=1)
    for k in np.flatnonzero(hit):
        i0 = np.maximum(np.floor((lo_b[k] - o) / h).astype(int), 0)
        i1 = np.minimum(np.ceil((hi_b[k] - o) / h).astype(int) + 1, shape)
        if np.any(i1 <= i0):
            continue
        ax = [o[d] + h * np.arange(i0[d], i1[d]) for d in range(3)]
        X, Y, Z = ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]
        d = p1[k] - p0[k]
        dd = float(d @ d) + 1e-12
        t = np.clip(((X - p0[k][0]) * d[0] + (Y - p0[k][1]) * d[1] + (Z - p0[k][2]) * d[2]) / dd, 0, 1)
        rho = np.sqrt((X - p0[k][0] - t * d[0]) ** 2 + (Y - p0[k][1] - t * d[1]) ** 2 + (Z - p0[k][2] - t * d[2]) ** 2)
        a = r0[k] + t * (r1[k] - r0[k])
        u = effective_radius(a, h)
        f = np.clip((u - rho) / h + 0.5, 0, 1).astype(np.float32)
        sl = tuple(slice(i0[d], i1[d]) for d in range(3))
        np.maximum(out[sl], f, out=out[sl])
    return out


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------

def cylinder_field(points, p0, axis, radius, dchi, b0=(0, 0, 1)):
    """Lorentz-corrected relative field (ppm) of an infinite cylinder (susceptibility
    difference dchi to its surroundings) at points (N, 3) mm."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    b = np.asarray(b0, float) / np.linalg.norm(b0)
    cos_t = float(u @ b)
    sin2 = 1 - cos_t**2
    r = np.asarray(points, float) - np.asarray(p0, float)
    r_perp = r - (r @ u)[:, None] * u
    rho = np.linalg.norm(r_perp, axis=1)
    bp = b - cos_t * u                                    # B0 projected onto the cross-section
    nb = np.linalg.norm(bp)
    if nb < 1e-9:                                         # parallel to B0: no exterior field
        cos2phi = np.zeros_like(rho)
    else:
        cosphi = (r_perp @ (bp / nb)) / np.maximum(rho, 1e-12)
        cos2phi = 2 * cosphi**2 - 1
    inside = dchi / 6.0 * (3 * cos_t**2 - 1)
    outside = dchi / 2.0 * sin2 * (radius / np.maximum(rho, 1e-12)) ** 2 * cos2phi
    return np.where(rho <= radius, inside, outside)
