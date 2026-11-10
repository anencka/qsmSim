"""Spatial phase unwrapping.

unwrap (default, method="lsq"): masked least-squares unwrapping (Ghiglia &
    Romero 1994). Minimizes sum over voxel pairs *inside the mask* of
    (grad u - wrap(grad phi))^2, i.e. a Poisson problem with natural (Neumann)
    boundaries on the mask, solved by CG preconditioned with the FFT Laplacian.
    For residue-free data (|true gradient| < pi per voxel) this returns the
    true phase up to a constant per connected component. A congruence step
    then snaps the result onto the data (u = phi_wrapped + 2 pi k), making it
    exact rather than merely smooth.

method="laplacian": Schofield & Zhu (2003) FFT Laplacian unwrapping +
    congruence. Fast, but it recovers only the non-harmonic part of the phase
    under periodic boundaries, so wherever the lost harmonic part exceeds pi
    the congruence step picks the wrong cycle. Kept for comparison.

unwrap / lsq_unwrap run on the device of their inputs (backend.py); laplacian_unwrap is CPU only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg

from . import backend
from .dipole import fftn, ifftn, kgrid, pad_center, padded_shape


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---- Laplacian (Schofield & Zhu) ---------------------------------------------

def _lap_eig(shape, voxel_size):
    """Eigenvalues of the periodic 7-point Laplacian on the FFT grid."""
    ks = kgrid(shape, voxel_size)
    return sum((2 * np.cos(2 * np.pi * k * d) - 2) / d**2 for k, d in zip(ks, voxel_size))


def laplacian_unwrap(phase, voxel_size=(1, 1, 1), mask=None, pad_factor=1.25):
    phase = np.asarray(phase, np.float32)
    if mask is not None:
        phase = np.where(mask, phase, 0).astype(np.float32)
    shape = padded_shape(phase.shape, pad_factor)
    big, sl = pad_center(phase, shape)
    lam = _lap_eig(shape, voxel_size)
    lap = lambda a: ifftn(lam * fftn(a)).real
    c, s = np.cos(big), np.sin(big)
    lap_true = c * lap(s) - s * lap(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(lam != 0, 1.0 / lam, 0.0)
    return ifftn(inv * fftn(lap_true)).real[sl].astype(np.float32)


# ---- masked least squares -----------------------------------------------------

def lsq_unwrap(phase, mask, maxiter=300, tol=1e-6):
    """Masked least-squares unwrapped estimate (smooth, not yet congruent)."""
    xp = backend.xp_of(phase)
    idx = np.argwhere(backend.to_host(mask))
    lo, hi = idx.min(0), idx.max(0) + 1
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    shape = padded_shape(tuple(hi - lo), 1.0, 1)          # >= 1 voxel of zero padding per side
    p, inner = pad_center(xp.asarray(phase[crop], np.float32), shape)
    m, _ = pad_center(xp.asarray(mask[crop], bool), shape)

    w = [(m & xp.roll(m, -1, axis=i)).astype(np.float32) for i in range(3)]
    G = lambda u: [xp.roll(u, -1, axis=i) - u for i in range(3)]
    Gt = lambda g: sum(xp.roll(gi, 1, axis=i) - gi for i, gi in enumerate(g))
    b_full = Gt([wi * wrap(gi) for wi, gi in zip(w, G(p))])

    def embed(v):
        u = xp.zeros(shape, np.float32)
        u[m] = v
        return u

    def A(v):
        return Gt([wi * gi for wi, gi in zip(w, G(embed(v)))])[m]

    lam = xp.asarray(-_lap_eig(shape, (1, 1, 1)).astype(np.float32))   # eigenvalues of G^T G (>= 0)
    if xp is np:
        lam[0, 0, 0] = np.inf

        def Minv(v):
            return ifftn(fftn(embed(v)) / lam).real[m].astype(np.float32)
    else:
        # CuPy's complex / real division gives NaN for an infinite divisor (numpy gives 0), so the GPU
        # multiplies by the reciprocal with the DC term set to 0 instead
        inv_lam = xp.zeros_like(lam)
        nz = lam != 0
        inv_lam[nz] = 1.0 / lam[nz]
        inv_lam[0, 0, 0] = 0.0

        def Minv(v):
            return ifftn(fftn(embed(v)) * inv_lam).real[m].astype(np.float32)

    if xp is np:
        n = int(m.sum())
        sol, _ = cg(LinearOperator((n, n), matvec=A, dtype=np.float32), b_full[m],
                    M=LinearOperator((n, n), matvec=Minv, dtype=np.float32), rtol=tol, maxiter=maxiter)
    else:
        sol, _ = backend.cg(A, b_full[m], rtol=tol, maxiter=maxiter, psolve=Minv)
    out = xp.zeros(mask.shape, np.float32)
    out[crop] = embed(sol)[inner]
    return out


def unwrap(phase, mask, voxel_size=(1, 1, 1), method="lsq"):
    """Unwrap inside mask (zero outside); exact up to 2 pi * (integer) per
    connected component for residue-free data."""
    xp = backend.xp_of(phase)
    phase = xp.asarray(phase, np.float32)
    if method == "lsq":
        est = lsq_unwrap(phase, mask)
    elif method == "laplacian":
        if xp is not np:
            raise NotImplementedError("laplacian unwrapping is CPU only")
        est = laplacian_unwrap(phase, voxel_size, mask)
    else:
        raise ValueError(method)
    out = xp.zeros_like(phase)
    lab, n = ndimage.label(backend.to_host(mask))         # component labelling on the CPU (cheap)
    lab = backend.like(lab, phase)
    for c in range(1, n + 1):
        sel = lab == c
        # remove the (arbitrary) component offset modulo 2 pi before rounding
        off = xp.angle(xp.mean(xp.exp(1j * (phase[sel] - est[sel]))))
        k = xp.round((est[sel] + off - phase[sel]) / (2 * np.pi))
        out[sel] = phase[sel] + 2 * np.pi * k
    return out
