"""Two-scale forward model for sub-voxel veins (Experiment 1).

Master grid (spacing h, e.g. 0.5 mm): bulk tissue chi / M0 / R2*, plus the DMV
susceptibility as volume fractions. The total field is the padded FFT dipole
model on this grid ("mesh"), plus harmonic shim terms.

Near-field correction (particle-mesh style): the DMV territory is cut into
tiles (core + halo). In each tile the DMV chi is rasterized at h/F (e.g. 0.1
mm) and its field computed twice with the same periodic tile FFT: on the fine
grid, and on the master grid after block-averaging. Their difference, at the
sub-sample points, is the field detail below the master grid spacing; the far
field (and the tile truncation / wrap-around error, identical in both) comes
from the mesh. Only master voxels near a vein (distance < clip(8 r, 1, 2.5) mm)
are sub-sampled; every voxel that contains vein is among them.

Signal: S = M0 exp(-R2* TE) exp(i sign (phi0 + 2 pi gamma B0 dB TE)); near-vein
master voxels are the mean over F^3 sub-samples of a two-compartment mixture
(intravascular fraction f with the vein's M0/R2*, extravascular tissue with
the voxel's bulk M0/R2*), each sub-sample with its own field. Acquisition
voxels are block averages of the master grid (default) or a k-space
truncation of it (option, gives the sinc PSF / Gibbs ringing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import dipole as dp
from .phantom import Grid, block_mean
from .signal import hz_per_ppm
from .veins import VeinNetwork, fraction_on_grid


@dataclass
class Master:
    grid: Grid
    labels: np.ndarray          # uint8 tissue labels (phantom-specific label set)
    chi: np.ndarray             # bulk chi (ppm), no DMVs
    m0: np.ndarray
    r2s: np.ndarray
    support: np.ndarray         # bool: intracranial / brain support used to derive masks
    phi0: np.ndarray            # transceive phase (rad)
    shim_ppm: np.ndarray | None = None
    chi_outside: float = 0.0    # chi of the medium beyond the FOV (padding)

    @property
    def h(self):
        return float(self.grid.voxel_size[0])


@dataclass
class NearZone:
    idx: np.ndarray             # flat master indices, sorted (N,)
    frac: np.ndarray            # intravascular fraction per sub-sample (N, F^3)
    field: np.ndarray           # (N, F^3): fine-minus-coarse field until finalize(), total field after
    F: int

    def __len__(self):
        return len(self.idx)


def sub_offsets(F):
    """Sub-sample center offsets in master-index units, (F^3, 3), C order."""
    a = (np.arange(F) + 0.5) / F - 0.5
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)


def near_radius(r, lo=1.0, hi=2.5):
    return np.clip(8.0 * r, lo, hi)


def build_dmv(master: Master, net: VeinNetwork, F=5, core_mm=16.0, halo_mm=4.0, b0=(0, 0, 1),
              near_mm=(1.0, 2.5), log=print):
    """Rasterize the DMVs: returns (dchi_master, NearZone, info). dchi_master is the
    DMV susceptibility change on the master grid (f * (chi_vein - chi_tissue))."""
    g, h = master.grid, master.h
    shape = g.shape
    hf = h / F
    nc = int(round(core_mm / h))
    nh = int(round(halo_mm / h))
    o = np.array([a[0] for a in g.axes])
    dchi_master = np.zeros(shape, np.float32)
    if len(net) == 0:
        return dchi_master, NearZone(np.zeros(0, np.int64), np.zeros((0, F**3), np.float32),
                                     np.zeros((0, F**3), np.float32), F), dict(tiles=0)

    segs = net.segments()
    xs, rs, _, _ = net.samples(min(0.1, hf))
    tree = cKDTree(xs)
    lo_i = np.floor((xs.min(0) - near_mm[1] - o) / h).astype(int)
    hi_i = np.ceil((xs.max(0) + near_mm[1] - o) / h).astype(int)
    starts = [range(max(0, lo_i[d] - lo_i[d] % nc), min(shape[d], hi_i[d] + 1), nc) for d in range(3)]
    offs = sub_offsets(F)
    chi_v = net.params.chi_ppm
    idx_l, frac_l, delta_l = [], [], []
    n_tiles = 0
    for cx in starts[0]:
        for cy in starts[1]:
            for cz in starts[2]:
                c0 = np.array([cx, cy, cz])
                c1 = np.minimum(c0 + nc, shape)
                # master voxels of the core near any vein
                ii = [np.arange(c0[d], c1[d]) for d in range(3)]
                cen = np.stack(np.meshgrid(*[o[d] + h * ii[d] for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
                dist, j = tree.query(cen, distance_upper_bound=near_mm[1])
                ok = np.isfinite(dist)
                if not ok.any():
                    continue
                near = np.zeros(len(cen), bool)
                near[ok] = dist[ok] < near_radius(rs[j[ok]], *near_mm)
                if not near.any():
                    continue
                n_tiles += 1
                r0 = c0 - nh                                   # tile region start (master index)
                n_reg = nc + 2 * nh
                n_f = n_reg * F
                origin = o + h * r0 - h / 2 + hf / 2
                f_fine = fraction_on_grid(net, origin, (n_f,) * 3, hf, seg=segs)
                # tissue chi under the vein (master value, edge-clamped outside the grid)
                reg = [np.clip(np.arange(r0[d], r0[d] + n_reg), 0, shape[d] - 1) for d in range(3)]
                chi_t = master.chi[np.ix_(*reg)]
                chi_t_f = np.repeat(np.repeat(np.repeat(chi_t, F, 0), F, 1), F, 2)
                dchi_f = f_fine * (np.float32(chi_v) - chi_t_f)
                del chi_t_f
                ff = dp.forward_field(dchi_f, (hf,) * 3, b0, pad_factor=1.0)
                dchi_c = block_mean(dchi_f, F).astype(np.float32)
                fc = dp.forward_field(dchi_c, (h,) * 3, b0, pad_factor=1.0)
                core = tuple(slice(nh, nh + (c1[d] - c0[d])) for d in range(3))
                dchi_master[tuple(slice(c0[d], c1[d]) for d in range(3))] = dchi_c[core]
                # sub-samples of the near voxels
                loc = np.argwhere(near.reshape([len(a) for a in ii])) + nh      # region-local master index
                fi = (loc[:, None, :] * F + np.rint((offs + 0.5) * F - 0.5).astype(int)[None]).reshape(-1, 3)
                fr = f_fine[fi[:, 0], fi[:, 1], fi[:, 2]].reshape(len(loc), -1)
                fs = ff[fi[:, 0], fi[:, 1], fi[:, 2]].reshape(len(loc), -1)
                cc = (loc[:, None, :] + offs[None]).reshape(-1, 3).T
                fci = ndimage.map_coordinates(fc, cc, order=1, mode="nearest").reshape(len(loc), -1)
                gl = loc - nh + c0
                idx_l.append(np.ravel_multi_index(gl.T, shape))
                frac_l.append(fr.astype(np.float32))
                delta_l.append((fs - fci).astype(np.float32))
    if idx_l:
        idx = np.concatenate(idx_l)
        order = np.argsort(idx)
        nz = NearZone(idx[order], np.concatenate(frac_l)[order], np.concatenate(delta_l)[order], F)
    else:
        nz = NearZone(np.zeros(0, np.int64), np.zeros((0, F**3), np.float32), np.zeros((0, F**3), np.float32), F)
    info = dict(tiles=n_tiles, near_voxels=len(nz), dmv_volume_mm3=float(nz.frac.sum() * hf**3))
    log(f"DMV tiles: {n_tiles}, near-vein master voxels {len(nz)}, intravascular volume {info['dmv_volume_mm3']:.1f} mm^3")
    return dchi_master, nz, info


def mesh_field(master: Master, dchi_master, b0=(0, 0, 1), pad_factor=2.0):
    """Total field on the master grid (ppm) and the total chi (bulk + DMV fractions)."""
    chi_tot = master.chi + dchi_master
    f = dp.forward_field(chi_tot, master.grid.voxel_size, b0, pad_factor, background=master.chi_outside)
    if master.shim_ppm is not None:
        f += master.shim_ppm
    return f, chi_tot


def finalize(nz: NearZone, mesh, chunk=50_000):
    """Sub-sample total field = trilinear(mesh) + fine-minus-coarse correction (in place)."""
    offs = sub_offsets(nz.F)
    shape = mesh.shape
    for s in range(0, len(nz), chunk):
        ijk = np.array(np.unravel_index(nz.idx[s:s + chunk], shape)).T
        cc = (ijk[:, None, :] + offs[None]).reshape(-1, 3).T
        nz.field[s:s + chunk] += ndimage.map_coordinates(mesh, cc, order=1, mode="nearest").reshape(len(ijk), -1)
    return nz


def _echo_slab(master, mesh, nz, vein, x0, x1, te, b0_tesla, phase_sign):
    """Complex signal of master planes x0:x1 for one echo (complex64)."""
    sl = slice(x0, x1)
    w = np.float32(2 * np.pi * hz_per_ppm(b0_tesla) * te)
    ph = np.float32(phase_sign) * (master.phi0[sl] + w * mesh[sl])
    S = (master.m0[sl] * np.exp(-master.r2s[sl] * np.float32(te))) * np.exp(1j * ph)
    S = S.astype(np.complex64)
    if len(nz):
        ny_nz = mesh.shape[1] * mesh.shape[2]
        a, b = np.searchsorted(nz.idx, [x0 * ny_nz, x1 * ny_nz])
        if b > a:
            idx = nz.idx[a:b]
            fr = nz.frac[a:b]
            m0t = master.m0.reshape(-1)[idx][:, None]
            r2t = master.r2s.reshape(-1)[idx][:, None]
            p0 = master.phi0.reshape(-1)[idx][:, None]
            amp = (1 - fr) * m0t * np.exp(-r2t * np.float32(te)) + fr * np.float32(vein.m0 * np.exp(-vein.r2s * te))
            sig = (amp * np.exp(1j * np.float32(phase_sign) * (p0 + w * nz.field[a:b]))).mean(1)
            S.reshape(-1)[idx - x0 * ny_nz] = sig.astype(np.complex64)
    return S


def synthesize(master: Master, mesh, nz: NearZone, vein_params, tes, factors, b0_tesla=3.0,
               phase_sign=1.0, resample="block", log=print):
    """Noiseless multi-echo signals at each acquisition resolution: {factor: (nx, ny, nz, ne) complex64}."""
    shape = mesh.shape
    for f in factors:
        if any(n % f for n in shape):
            raise ValueError(f"master shape {shape} not divisible by {f}")
    out = {f: np.zeros(tuple(n // f for n in shape) + (len(tes),), np.complex64) for f in factors}
    if resample == "block":
        L = math.lcm(*factors)
        for x0 in range(0, shape[0], L):
            for e, te in enumerate(tes):
                S = _echo_slab(master, mesh, nz, vein_params, x0, x0 + L, te, b0_tesla, phase_sign)
                for f in factors:
                    out[f][x0 // f:(x0 + L) // f, ..., e] = block_mean(S, f)
    elif resample == "kspace":
        import scipy.fft as sfft
        for e, te in enumerate(tes):
            full = np.empty(shape, np.complex64)
            for x0 in range(0, shape[0], 16):
                x1 = min(shape[0], x0 + 16)
                full[x0:x1] = _echo_slab(master, mesh, nz, vein_params, x0, x1, te, b0_tesla, phase_sign)
            K = sfft.fftshift(sfft.fftn(full, workers=dp.WORKERS))
            del full
            for f in factors:
                tgt = tuple(n // f for n in shape)
                sl = tuple(slice((n - t) // 2, (n - t) // 2 + t) for n, t in zip(shape, tgt))
                Kc = K[sl].copy()
                # a DFT-decimated sample sits on the first fine voxel of its block; shift by
                # (f - 1)/2 fine voxels so voxel centers match the block-average grid
                s = (f - 1) / 2.0
                for d, (n, t) in enumerate(zip(shape, tgt)):
                    k = sfft.fftshift(sfft.fftfreq(n))[sl[d]]
                    ramp = np.exp(2j * np.pi * k * s).astype(np.complex64)
                    Kc *= ramp.reshape([-1 if i == d else 1 for i in range(3)])
                img = sfft.ifftn(sfft.ifftshift(Kc), workers=dp.WORKERS) / np.float32(f**3)
                out[f][..., e] = img.astype(np.complex64)
            del K
    else:
        raise ValueError(resample)
    log(f"signal: {len(tes)} echoes -> factors {list(factors)} ({resample})")
    return out
