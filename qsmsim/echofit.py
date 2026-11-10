"""Multi-echo field mapping: complex nonlinear fit of (phi0, omega).

1. omega_0 from the magnitude-weighted average consecutive-echo phase
   difference angle(sum_n S_{n+1} S_n*), spatially unwrapped (this is the only
   spatial unwrap in the pipeline - it absorbs aliasing whenever
   |df| > 1/(2 dTE)).
2. phi0_0 = angle(sum_n S_n exp(-i omega_0 TE_n)).
3. Gauss-Newton on  sum_n | |S_n| exp(i(phi0 + omega TE_n)) - S_n |^2  per voxel
   (MEDI's Fit_ppm_complex model with measured magnitudes as amplitudes).
   Because omega is a real parameter, the fit never re-wraps.

Limitation: with uniform echo spacing, omega and omega + 2 pi/dTE fit every
voxel equally well (the difference is absorbed by phi0), so the aliasing cycle
comes only from the spatial unwrap of the echo-difference phase. Where the
true field changes by more than half a cycle, 1/(2 gamma B0 dTE), between
neighbours (e.g. next to sinuses at 7 T with coarse voxels), single voxels can
land one cycle off. The pipeline reports how many (truth-based diagnostic).

Returns the total field in ppm, its noise standard deviation (from the
Gauss-Newton covariance with a globally estimated noise level) and an RSS
magnitude image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backend
from .signal import hz_per_ppm
from .unwrap import unwrap


@dataclass
class FieldFit:
    field_ppm: np.ndarray
    field_std_ppm: np.ndarray
    phi0: np.ndarray
    magnitude: np.ndarray
    weights: np.ndarray     # 1/std normalized to mean 1 inside the mask


def fit_field(S, tes, mask, voxel_size, b0_tesla, phase_sign=1.0, gn_iters=10):
    """Runs on the device of S (backend.py); on the GPU in float64 as on the CPU."""
    xp = backend.xp_of(S)
    tes = np.asarray(tes, np.float64)
    ne = len(tes)
    if ne < 2:
        raise ValueError("need >= 2 echoes")
    dte = np.diff(tes)
    if not np.allclose(dte, dte[0], rtol=1e-3):
        raise ValueError("echo spacing must be uniform")
    dte = dte[0]
    tes = xp.asarray(tes)

    Sm = S[mask].astype(np.complex128)          # (nvox, ne)
    mag = xp.abs(Sm)

    # 1-2. initial estimates
    dphi = xp.zeros(mask.shape, np.float32)
    dphi[mask] = xp.angle(xp.sum(Sm[:, 1:] * xp.conj(Sm[:, :-1]), axis=1))
    dphi_u = unwrap(dphi, mask, voxel_size)
    omega = dphi_u[mask].astype(np.float64) / dte
    phi0 = xp.angle(xp.sum(Sm * xp.exp(-1j * xp.outer(omega, tes)), axis=1))

    # 3. Gauss-Newton
    w = mag**2
    a11 = w.sum(1)
    a12 = (w * tes).sum(1)
    a22 = (w * tes**2).sum(1)
    det = a11 * a22 - a12**2
    det = xp.where(det > 0, det, np.inf)
    for _ in range(gn_iters):
        model = mag * xp.exp(1j * (phi0[:, None] + omega[:, None] * tes))
        q = xp.imag(xp.conj(model) * Sm)
        g1, g2 = q.sum(1), (q * tes).sum(1)
        d_phi0 = (a22 * g1 - a12 * g2) / det
        d_omega = (-a12 * g1 + a11 * g2) / det
        phi0 += d_phi0
        omega += d_omega

    model = mag * xp.exp(1j * (phi0[:, None] + omega[:, None] * tes))
    res2 = xp.sum(xp.abs(model - Sm) ** 2, axis=1)
    # With measured magnitudes as amplitudes only the tangential noise component
    # (variance sigma^2 per echo) remains: E[res2] = (ne - 2) sigma^2.
    keep = res2 <= backend.percentile(res2, 99)
    sigma2 = res2[keep].mean() / max(ne - 2, 1)
    var_omega = sigma2 * a11 / det                        # sigma^2 [A^-1]_22

    scale = phase_sign / (2 * np.pi * hz_per_ppm(b0_tesla))
    field = xp.zeros(mask.shape, np.float32)
    std = xp.zeros(mask.shape, np.float32)
    ph0 = xp.zeros(mask.shape, np.float32)
    field[mask] = omega * scale
    std[mask] = xp.sqrt(var_omega) * abs(scale)
    ph0[mask] = xp.angle(xp.exp(1j * phi0))

    magnitude = xp.sqrt(xp.sum(xp.abs(S) ** 2, axis=-1)).astype(np.float32)

    return FieldFit(field, std, ph0, magnitude, noise_weights(std, mask))


def noise_weights(std, mask):
    """Data weights 1/std, normalized to mean 1 inside the mask."""
    xp = backend.xp_of(std)
    wts = xp.zeros(mask.shape, np.float32)
    s = std[mask]
    s = xp.maximum(s, backend.percentile(s[xp.isfinite(s)], 1))  # guard against tiny std
    inv = 1.0 / s
    inv[~xp.isfinite(inv)] = 0
    wts[mask] = inv / inv.mean()
    return wts
