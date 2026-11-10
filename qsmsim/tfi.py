"""Total field inversion (single-step QSM): background removal and dipole
inversion solved jointly. After Liu Z et al., "Preconditioned total field
inversion (TFI) method for quantitative susceptibility mapping", MRM 2017.

    chi* = argmin_chi  1/2 ||W (D chi - f_total)||^2 + lam ||G grad chi||_1

chi is unknown over the whole (padded) field of view, so sources outside the
brain (air, bone, sinuses) are part of the solution instead of being removed
first. W is the noise weight inside `data_mask` and 0 elsewhere. By default
data_mask = the brain mask (the same data PDF sees); passing a head mask uses
the field measured in scalp and other soft tissue too, as the original TFI
does - information two-step methods throw away. G are MEDI morphology
weights inside the brain; outside it G = `bg_weight` (1 = TV-penalized,
0 = free).

Preconditioning (the key ingredient of TFI): chi = P y with P = 1 inside the
brain and P = pb outside. Background susceptibilities are ~100x larger than
tissue contrast, so without P the CG solver spends its iterations on the
background and under-resolves tissue.

Solver: iteratively reweighted least squares for the L1 term (as MEDI),
each step solved with CG and warm-started from the last:
    (P D W^2 D P + lam P grad^T V grad P) y = P D W^2 f,
    V_i = G_i^2 / sqrt((G_i grad_i chi)^2 + eps)
Differences from the MEDI toolbox TFI: linear data term (as everywhere in
qsmSim), anisotropic L1, no CSF zero-reference term.

Runs on the device of its inputs (backend.py). On the GPU the CG is backend.cg, the same algorithm as scipy's
with float64 inner products.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy.sparse.linalg import LinearOperator, cg

from . import backend
from .dipole import WORKERS, irfftn, kernel_for, pad_center, padded_shape, rfftn
from .inversion import edge_weights, grad, grad_adj


def tfi(total_field, mask, voxel_size, magnitude, b0_dir=(0, 0, 1), weights=None, lam=1e-4,
        pb=30.0, edge_fraction=0.1, bg_weight=1.0, pad_factor=1.0, outer=12, cg_iters=40,
        eps=1e-6, tol=1e-2, data_mask=None, verbose=False):
    """mask: brain mask (P = 1, morphology prior, output region).
    data_mask: where the field is used (default: mask); weights must cover it.
    Returns (chi inside the mask, chi over the full FOV, info dict)."""
    xp = backend.xp_of(total_field)
    gpu = xp is not np
    dm = mask if data_mask is None else data_mask
    shape = padded_shape(total_field.shape, pad_factor)
    put = lambda a, fill=0.0: pad_center(xp.asarray(a, np.float32), shape, fill)
    f, sl = put(xp.where(dm, total_field, 0))
    m, _ = put(mask.astype(np.float32))
    m = m > 0
    w = xp.ones(mask.shape, np.float32) if weights is None else weights
    W2 = put(xp.where(dm, w, 0))[0] ** 2
    P = xp.where(m, 1.0, pb).astype(np.float32)
    G = [xp.where(mask, g, bg_weight) for g in edge_weights(magnitude, mask, edge_fraction)]
    G2 = [put(g, bg_weight)[0] ** 2 for g in G]

    D = kernel_for(f, shape, voxel_size, b0_dir, real=True)
    if gpu:
        Dop = lambda x: irfftn(D * rfftn(x), shape).astype(np.float32)
    else:
        Dop = lambda x: sfft.irfftn(D * sfft.rfftn(x, workers=WORKERS), s=shape, workers=WORKERS).astype(np.float32)
    b = (P * Dop(W2 * f)).ravel()
    n = b.size

    y = xp.zeros(n, np.float32)
    info = dict(outer=0, cg=[])
    for it in range(1, outer + 1):
        chi = P * y.reshape(shape)
        V = [g2 / xp.sqrt(g2 * gi * gi + eps) for g2, gi in zip(G2, grad(chi))]

        def A(v, V=V):
            x = P * v.reshape(shape)
            data = Dop(W2 * Dop(x))
            reg = grad_adj([vi * gi for vi, gi in zip(V, grad(x))])
            return (P * (data + np.float32(lam) * reg)).ravel()

        counter = [0]
        count = lambda _: counter.__setitem__(0, counter[0] + 1)
        if gpu:
            y_new, _ = backend.cg(A, b, x0=y, maxiter=cg_iters, rtol=1e-4, callback=count)
        else:
            y_new, _ = cg(LinearOperator((n, n), matvec=A, dtype=np.float32), b, x0=y, maxiter=cg_iters,
                          rtol=1e-4, callback=count)
        mv = m.ravel()                                   # converge on brain chi, not the (large) background
        change = float(xp.linalg.norm((y_new - y)[mv]) / (xp.linalg.norm(y_new[mv]) + 1e-12))
        y = y_new.astype(np.float32)
        info["outer"] = it
        info["cg"].append(counter[0])
        if verbose:
            print(f"    TFI outer {it:2d}: {counter[0]:3d} CG iters, rel. change {change:.2e}", flush=True)
        if change < tol:
            break
    chi = (P * y.reshape(shape))[sl]
    return xp.where(mask, chi, 0).astype(np.float32), xp.ascontiguousarray(chi), info
