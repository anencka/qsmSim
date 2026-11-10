"""Dipole inversion (local field ppm -> susceptibility ppm).

tkd   Truncated k-space division (Shmueli 2009): 1/D, with |D| < thr
      replaced by sign(D)/thr.
cfl2  Closed-form L2 gradient regularization (Bilgic 2014):
      argmin ||D chi - f||^2 + lam ||grad chi||^2, solved exactly in k-space.
tv    Weighted-TV with a linear, noise-weighted data term, solved with ADMM
      in the FANSI formulation (Milovic 2018); all sub-problems closed-form:
      argmin 1/2 ||W (D chi - f)||^2 + alpha ||grad chi||_1   (anisotropic TV)
medi  Same solver with MEDI's morphology prior (Liu 2011): gradients at the
      edge_fraction largest magnitude-image gradients are not penalized.
      Linear data term, no CSF/zero-reference term - "MEDI-like", not MEDI.

Gradients are unitless per-voxel forward differences with periodic
boundaries, so regularization weights are resolution-dependent. Inversions
run on the mask bounding box plus a margin of PAD_FRACTION x the largest
bounding-box extent per side (at least 10 voxels), padded to FFT-friendly
sizes; TKD and CF-L2 use DIRECT_PAD_FRACTION. The margin matters: a
10-voxel margin cost CF-L2 ~5 and TV ~3 RMSE points (periodic images of the
solution sit too close, and the unweighted direct methods treat the zero
field outside the mask as data). Edge weights are padded with 1 (penalized):
padding them with 0 lets chi outside the brain absorb background residuals,
which flattered MEDI-like on real background removal by ~20 RMSE points.

Every function runs on the device of its input arrays (backend.py): numpy on the CPU (unchanged), CuPy on
the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import percentile, to_host, xp_of
from .dipole import fftn, ifftn, kernel_for, pad_center, padded_shape

PAD_FRACTION = 0.25          # iterative methods (W = 0 outside the mask)
DIRECT_PAD_FRACTION = 0.5    # TKD / CF-L2 treat zeros outside the mask as data, so padding matters more


@dataclass
class Box:
    """Mask bounding box (+margin) embedded in an FFT-friendly padded grid."""
    crop: tuple
    inner: tuple
    shape: tuple

    @classmethod
    def from_mask(cls, mask, pad_fraction=None, min_margin=10):
        idx = np.argwhere(to_host(mask))
        lo, hi = idx.min(0), idx.max(0) + 1
        frac = PAD_FRACTION if pad_fraction is None else pad_fraction
        margin = max(min_margin, int(np.ceil(frac * (hi - lo).max())))
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        shape = padded_shape(tuple(hi - lo), 1.0, margin)
        inner = tuple(slice((S - (b - a)) // 2, (S - (b - a)) // 2 + (b - a)) for a, b, S in zip(lo, hi, shape))
        return cls(crop, inner, shape)

    def put(self, a, fill=0.0):
        return pad_center(xp_of(a).ascontiguousarray(a[self.crop], dtype=np.float32), self.shape, fill)[0]

    def take(self, big, full_shape):
        out = xp_of(big).zeros(full_shape, np.float32)
        out[self.crop] = big[self.inner]
        return out


def _diff_eig(shape, xp=np):
    """k-space eigenvalues of periodic forward differences, one per axis."""
    out = []
    for i, n in enumerate(shape):
        m = np.fft.fftfreq(n)
        sh = [1, 1, 1]
        sh[i] = n
        out.append(xp.asarray((np.exp(2j * np.pi * m) - 1).reshape(sh).astype(np.complex64)))
    return out


def grad(x):
    xp = xp_of(x)
    return [xp.roll(x, -1, axis=i) - x for i in range(3)]


def grad_adj(g):
    """Adjoint of `grad` (negative backward-difference divergence)."""
    xp = xp_of(g[0])
    return sum(xp.roll(gi, 1, axis=i) - gi for i, gi in enumerate(g))


def tkd(field, mask, voxel_size, b0_dir=(0, 0, 1), thr=0.19):
    xp = xp_of(field)
    box = Box.from_mask(mask, DIRECT_PAD_FRACTION)
    f = box.put(xp.where(mask, field, 0))
    D = kernel_for(f, box.shape, voxel_size, b0_dir)
    inv = xp.where(xp.abs(D) > thr, 1.0 / xp.where(D == 0, 1, D), xp.sign(D) / thr).astype(np.float32)
    chi = ifftn(inv * fftn(f)).real
    return box.take(chi, mask.shape) * mask


def cfl2(field, mask, voxel_size, b0_dir=(0, 0, 1), lam=0.05):
    xp = xp_of(field)
    box = Box.from_mask(mask, DIRECT_PAD_FRACTION)
    f = box.put(xp.where(mask, field, 0))
    D = kernel_for(f, box.shape, voxel_size, b0_dir)
    E2 = sum(xp.abs(e) ** 2 for e in _diff_eig(box.shape, xp)).astype(np.float32)
    den = D**2 + lam * E2
    den[0, 0, 0] = 1.0
    chi = ifftn(D / den * fftn(f)).real
    return box.take(chi, mask.shape) * mask


def edge_weights(magnitude, mask, edge_fraction=0.1):
    """MEDI morphology prior: per-component gradient weights, 0 where the
    normalized magnitude gradient is among the largest edge_fraction."""
    xp = xp_of(magnitude)
    mag = magnitude / (percentile(magnitude[mask], 99) + 1e-12)
    g = [xp.abs(gi) for gi in grad(xp.where(mask, mag, 0).astype(np.float32))]
    pooled = xp.concatenate([gi[mask] for gi in g])
    t = percentile(pooled, 100 * (1 - edge_fraction))
    return [(gi <= t).astype(np.float32) for gi in g]


def tv_admm(field, mask, voxel_size, b0_dir=(0, 0, 1), weights=None, alpha=3e-4,
            mu1_factor=100.0, mu2=1.0, maxiter=200, tol=1e-3, grad_weights=None,
            verbose=False):
    xp = xp_of(field)
    box = Box.from_mask(mask)
    f = box.put(xp.where(mask, field, 0))
    W = box.put(xp.where(mask, 1.0 if weights is None else weights, 0))
    W2 = W * W
    Gw = [xp.ones(box.shape, np.float32)] * 3 if grad_weights is None else [box.put(g, fill=1.0) for g in grad_weights]
    D = kernel_for(f, box.shape, voxel_size, b0_dir)
    E = _diff_eig(box.shape, xp)
    mu1 = mu1_factor * alpha
    den = (mu1 * sum(xp.abs(e) ** 2 for e in E) + mu2 * D**2).astype(np.float32)
    den[0, 0, 0] = 1.0
    thr = [np.float32(alpha / mu1) * g for g in Gw]

    chi = xp.zeros(box.shape, np.float32)
    z = [xp.zeros(box.shape, np.float32) for _ in range(3)]
    s = [xp.zeros(box.shape, np.float32) for _ in range(3)]
    z2 = f.copy()
    s2 = xp.zeros(box.shape, np.float32)
    it = 0
    for it in range(1, maxiter + 1):
        rhs = mu1 * grad_adj([zi - si for zi, si in zip(z, s)])
        X = (fftn(rhs) + mu2 * D * fftn(z2 - s2)) / den
        X[0, 0, 0] = 0
        chi_new = ifftn(X).real.astype(np.float32)
        Dchi = ifftn(D * X).real.astype(np.float32)
        change = float(xp.linalg.norm(chi_new - chi) / (xp.linalg.norm(chi_new) + 1e-12))
        chi = chi_new
        g = grad(chi)
        for i in range(3):
            v = g[i] + s[i]
            z[i] = xp.sign(v) * xp.maximum(xp.abs(v) - thr[i], 0)
            s[i] = v - z[i]
        z2 = (W2 * f + mu2 * (Dchi + s2)) / (W2 + mu2)
        s2 = s2 + Dchi - z2
        if verbose and it % 10 == 0:
            print(f"    iter {it:4d}  dchi {change:.2e}")
        if change < tol:
            break
    out = box.take(chi, mask.shape) * mask
    return out, it


def medi_like(field, mask, voxel_size, magnitude, b0_dir=(0, 0, 1), weights=None,
              alpha=1e-3, edge_fraction=0.1, **kw):
    gw = edge_weights(magnitude, mask, edge_fraction)
    return tv_admm(field, mask, voxel_size, b0_dir, weights, alpha, grad_weights=gw, **kw)
