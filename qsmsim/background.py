"""Background field removal.

sharp   Schweser et al. 2011 - single spherical-mean-value (SMV) kernel:
        high-pass with (delta - SMV), deconvolve with TSVD threshold.
vsharp  Wu/Li et al. 2012 - variable kernel: each voxel takes the largest
        sphere that fits inside the mask, deconvolution with the largest
        kernel (TSVD). Final mask = mask eroded by the smallest radius.
        The TSVD deconvolution is non-local, so the result depends on the FFT
        grid (~15 % of the local-field norm in the default phantom). domain=
        "crop" (default) reproduces QSM.jl's grid (mask bounding box + r_max
        voxels - 1, 7-smooth sizes) and matches QSM.jl to 0.001 %; "fov" uses
        the full field of view.
pdf     Liu et al. 2011 - projection onto dipole fields: fit sources outside
        the mask whose field best explains the (noise-weighted) measured field
        inside it; the remainder is the local field. Solved with CG.
lbv     Zhou et al. 2014 - Laplacian boundary value: solve lap(f_l) = lap(f)
        on the mask interior (mask eroded by one voxel, so every 7-point
        stencil uses measured field) with f_l = 0 on the boundary ring.
        Solved with CG preconditioned by the FFT Laplacian. (QSM.jl 0.5.4's
        lbv() zeroes the field outside the mask before taking the Laplacian,
        so it returns its input unchanged; this is an independent version.)

All return (local_field_ppm, mask_used). vsharp, sharp and pdf run on the device of their inputs
(backend.py); lbv runs on the CPU and returns arrays on the input's device.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg

from . import backend
from .dipole import fftn, ifftn, kernel_for, pad_center, padded_shape


def smv_kernel(shape, voxel_size, radius_mm, xp=np):
    """k-space transform of a normalized sphere of radius_mm (real, FFT order)."""
    ax = []
    for i, (n, d) in enumerate(zip(shape, voxel_size)):
        a = np.fft.fftfreq(n, d=1.0 / n) * d          # signed distance in mm (wrapped)
        sh = [1, 1, 1]
        sh[i] = n
        ax.append(xp.asarray(a.reshape(sh)))
    sphere = (ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2 <= radius_mm**2).astype(np.float32)
    sphere /= sphere.sum()
    return fftn(sphere).real.astype(np.float32)


def _next_smooth(n, primes=(2, 3, 5, 7)):
    while True:
        k = n
        for p in primes:
            while k % p == 0:
                k //= p
        if k == 1:
            return n
        n += 1


def _domain(field, mask, max_r_mm, voxel_size, domain):
    """Embed the (masked) field in the FFT grid; returns field, mask and the
    slices that map the grid back onto the input array."""
    xp = backend.xp_of(field)
    pad = int(np.ceil(max_r_mm / min(voxel_size)))
    if domain == "fov":
        crop = tuple(slice(0, n) for n in field.shape)
        shape = padded_shape(field.shape, 1.0, pad + 2)
    elif domain == "crop":
        idx = np.argwhere(backend.to_host(mask))
        lo, hi = idx.min(0), idx.max(0) + 1
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))
        shape = [_next_smooth(int(b - a) + pad - 1) for a, b in zip(lo, hi)]
        s0 = s = shape[0]                    # QSM.jl/FFTW rfft rule: prefer an even first dim
        for _ in range(3):
            if s % 2 == 0:
                break
            s = _next_smooth(s + 1)
        shape[0] = s0 + 1 if s % 2 else s
    else:
        raise ValueError(domain)
    sub = [c.stop - c.start for c in crop]
    off = [(S - n + 1) >> 1 for S, n in zip(shape, sub)]
    inner = tuple(slice(o, o + n) for o, n in zip(off, sub))
    f = xp.zeros(tuple(shape), np.float32)
    m = xp.zeros(tuple(shape), bool)
    f[inner] = xp.where(mask, field, 0)[crop]
    m[inner] = mask[crop]
    return f, m, crop, inner


def vsharp(field, mask, voxel_size, radii_mm=None, thr=0.05, domain="crop"):
    if radii_mm is None:
        step = min(voxel_size)
        radii_mm = np.arange(12.0, step - 1e-6, -step)       # 12 mm -> 1 voxel
    radii_mm = sorted(radii_mm, reverse=True)
    xp = backend.xp_of(field)
    f, m, crop, inner = _domain(field, mask, radii_mm[0], voxel_size, domain)
    F, Mk = fftn(f), fftn(m.astype(np.float32))

    filt = xp.zeros(f.shape, np.float32)
    prev = None
    for i, r in enumerate(radii_mm):
        S = smv_kernel(f.shape, voxel_size, r, xp)
        m_r = ifftn(S * Mk).real > 1 - 3.5e-4
        hp = ifftn((1 - S) * F).real.astype(np.float32)
        if i == 0:
            DS0 = 1 - S
            filt[m_r] = hp[m_r]
        else:
            new = m_r & ~prev
            filt[new] = hp[new]
        prev = m_r
    inv = xp.zeros_like(DS0)
    keep = xp.abs(DS0) > thr
    inv[keep] = 1.0 / DS0[keep]
    local = ifftn(inv * fftn(filt)).real.astype(np.float32)
    local[~prev] = 0
    out = xp.zeros(field.shape, np.float32)
    out_m = xp.zeros(field.shape, bool)
    out[crop] = local[inner]
    out_m[crop] = prev[inner]
    return out, out_m


def sharp(field, mask, voxel_size, radius_mm=6.0, thr=0.05, domain="crop"):
    return vsharp(field, mask, voxel_size, radii_mm=[radius_mm], thr=thr, domain=domain)


def pdf(field, mask, voxel_size, b0_dir=(0, 0, 1), weights=None, pad_factor=1.25,
        maxiter=150, tol=1e-3, info=None):
    """info: a dict that receives the CG iteration count and the final relative residual |b - A x| / |b|
    (one extra operator application after the solve; the result is unchanged)."""
    xp = backend.xp_of(field)
    shape = padded_shape(field.shape, pad_factor)
    f, sl = pad_center(xp.where(mask, field, 0).astype(np.float32), shape)
    m, _ = pad_center(mask.astype(bool), shape)
    w = xp.ones(field.shape, np.float32) if weights is None else weights.astype(np.float32)
    w, _ = pad_center(xp.where(mask, w, 0).astype(np.float32), shape)
    w2 = w * w
    D = kernel_for(f, shape, voxel_size, b0_dir)
    out = ~m

    def conv(x):
        return ifftn(D * fftn(x)).real.astype(np.float32)

    def normal(xv):
        x = xp.zeros(shape, np.float32)
        x[out] = xv
        y = w2 * conv(x)                 # W^2 M D O x   (w2 is 0 outside M)
        return conv(y)[out]              # O D^T (.)

    b = conv(w2 * f)[out]
    count = [0]

    def tick(_):
        count[0] += 1
    cb = tick if info is not None else None
    if xp is np:
        n = int(out.sum())
        A = LinearOperator((n, n), matvec=normal, dtype=np.float32)
        xv, flag = cg(A, b, rtol=tol, maxiter=maxiter, callback=cb)
    else:
        xv, flag = backend.cg(normal, b, rtol=tol, maxiter=maxiter, callback=cb)
    if info is not None:
        bn = float(xp.linalg.norm(b.astype(np.float64)))
        rn = float(xp.linalg.norm((b - normal(xv)).astype(np.float64)))
        info.update(iterations=count[0], converged=int(flag) == 0, residual=rn / bn if bn > 0 else 0.0,
                    maxiter=int(maxiter), tol=float(tol))
    x = xp.zeros(shape, np.float32)
    x[out] = xv
    bkg = conv(x)
    local = xp.where(m, f - bkg, 0).astype(np.float32)
    return xp.ascontiguousarray(local[sl]), mask.copy()


def lbv(field, mask, voxel_size, maxiter=1000, tol=1e-6):
    if backend.is_gpu(field):                         # not ported (not used in Experiments 1-2): CPU round trip
        local, out_m = lbv(backend.to_host(field), backend.to_host(mask), voxel_size, maxiter, tol)
        return backend.like(local, field), backend.like(out_m, field)
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    shape = padded_shape(tuple(hi - lo), 1.0, 2)
    f, inner = pad_center(np.where(mask, field, 0)[crop].astype(np.float64), shape)
    m, _ = pad_center(mask[crop].astype(bool), shape)
    interior = ndimage.binary_erosion(m, ndimage.generate_binary_structure(3, 1))
    h2 = [1.0 / d**2 for d in voxel_size]

    def neg_lap(u):                                   # -Laplacian, 7-point stencil
        out = np.zeros_like(u)
        for ax, w in enumerate(h2):
            out += w * (2 * u - np.roll(u, 1, ax) - np.roll(u, -1, ax))
        return out

    def embed(v):
        u = np.zeros(shape)
        u[interior] = v
        return u

    lam = sum(w * (2 - 2 * np.cos(2 * np.pi * np.fft.fftfreq(n)))[tuple(slice(None) if i == ax else None
              for i in range(3))] for ax, (w, n) in enumerate(zip(h2, shape)))
    lam = np.broadcast_to(lam, shape).copy()
    lam[0, 0, 0] = np.inf

    n = int(interior.sum())
    A = LinearOperator((n, n), matvec=lambda v: neg_lap(embed(v))[interior], dtype=np.float64)
    M = LinearOperator((n, n), matvec=lambda v: ifftn(fftn(embed(v)) / lam).real[interior], dtype=np.float64)
    sol, _ = cg(A, neg_lap(f)[interior], M=M, rtol=tol, maxiter=maxiter)
    local = np.zeros(field.shape, np.float32)
    out_m = np.zeros(field.shape, bool)
    local[crop] = embed(sol)[inner]
    out_m[crop] = interior[inner]
    return local, out_m
