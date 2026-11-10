"""Dipole kernel, k-space grids and the susceptibility -> field forward model.

Conventions (used everywhere in the package):
  * Arrays are indexed (x, y, z) = (L->R, P->A, I->S); voxel sizes in mm.
  * Susceptibility chi and fields are in ppm. The field is the relative
    z-component of the induced field, dB/B0, Lorentz-sphere corrected:
        F{dB/B0}(k) = D(k) F{chi}(k),   D(k) = 1/3 - (k.b)^2 / |k|^2
    with b the unit B0 direction in image coordinates. D(0) is set to 0, so a
    uniform susceptibility produces no field (the k=0 term is unobservable).
  * Positive chi parallel to B0 gives a positive field (a vertical vein is
    bright on the field map); the phase sign convention lives in signal.py.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .backend import fft as _fft_of
from .backend import xp_of

WORKERS = -1  # scipy.fft worker threads (-1 = all cores)


# FFTs on whichever device the array is on (backend.py): numpy -> scipy.fft with WORKERS, CuPy -> cuFFT
def fftn(a, s=None):
    xp = xp_of(a)
    if xp is np:
        return sfft.fftn(a, s=s, workers=WORKERS)
    return _fft_of(xp).fftn(a, s=s)


def ifftn(a, s=None):
    xp = xp_of(a)
    if xp is np:
        return sfft.ifftn(a, s=s, workers=WORKERS)
    return _fft_of(xp).ifftn(a, s=s)


def rfftn(a):
    xp = xp_of(a)
    if xp is np:
        return sfft.rfftn(a, workers=WORKERS)
    return _fft_of(xp).rfftn(a)


def irfftn(a, s):
    xp = xp_of(a)
    if xp is np:
        return sfft.irfftn(a, s=s, workers=WORKERS)
    return _fft_of(xp).irfftn(a, s=s)


def kgrid(shape, voxel_size, real=False):
    """Spatial-frequency grids (cycles/mm) in FFT order, broadcastable.
    real=True gives the half-spectrum grid of rfftn (last axis)."""
    ks = []
    for i, (n, d) in enumerate(zip(shape, voxel_size)):
        k = sfft.rfftfreq(n, d=d) if (real and i == len(shape) - 1) else sfft.fftfreq(n, d=d)
        n = k.size
        sh = [1, 1, 1]
        sh[i] = n
        ks.append(k.reshape(sh))
    return ks


def b0_unit(b0_dir=(0.0, 0.0, 1.0)):
    b = np.asarray(b0_dir, dtype=float)
    return b / np.linalg.norm(b)


def b0_from_tilt(tilt_deg=(0.0, 0.0)):
    """B0 direction for a head pitched by tilt_deg[0] (about x) and rolled by
    tilt_deg[1] (about y), expressed in image coordinates."""
    ax, ay = np.deg2rad(tilt_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    return ry @ rx @ np.array([0.0, 0.0, 1.0])


def dipole_kernel(shape, voxel_size, b0_dir=(0, 0, 1), kind="continuous", real=False):
    """k-space dipole kernel D(k) on an FFT grid (float32, FFT order).

    kind="continuous": analytic 1/3 - (k.b)^2/k^2 sampled on the grid.
    kind="discrete":   k_i^2 replaced by the finite-difference Laplacian
                       eigenvalues (2-2cos(2 pi k_i d_i))/d_i^2 (Milovic 2017),
                       useful to avoid an inverse crime when the forward
                       model used the continuous kernel.
    real=True returns the rfftn half-spectrum.
    """
    b = b0_unit(b0_dir)
    kx, ky, kz = kgrid(shape, voxel_size, real)
    if kind == "continuous":
        kb = kx * b[0] + ky * b[1] + kz * b[2]
        k2 = kx**2 + ky**2 + kz**2
        with np.errstate(divide="ignore", invalid="ignore"):
            D = 1.0 / 3.0 - kb**2 / k2
    elif kind == "discrete":
        if np.count_nonzero(b) != 1:
            raise ValueError("discrete kernel implemented for axis-aligned B0 only")
        lap = [(2 - 2 * np.cos(2 * np.pi * k * d)) / d**2
               for k, d in zip((kx, ky, kz), voxel_size)]
        ax = int(np.flatnonzero(b)[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            D = 1.0 / 3.0 - lap[ax] / (lap[0] + lap[1] + lap[2])
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")
    full = tuple(shape[:-1]) + ((shape[-1] // 2 + 1,) if real else (shape[-1],))
    D = np.broadcast_to(D, full).astype(np.float32)
    D[0, 0, 0] = 0.0
    return D


_device_kernels = {}


def kernel_for(ref, shape, voxel_size, b0_dir=(0, 0, 1), kind="continuous", real=False):
    """dipole_kernel on the device of `ref`. On the GPU the last few kernels are kept (lambda sweeps reuse them);
    on the CPU it is dipole_kernel itself."""
    xp = xp_of(ref)
    if xp is np:
        return dipole_kernel(shape, voxel_size, b0_dir, kind, real)
    key = (tuple(shape), tuple(float(v) for v in voxel_size), tuple(float(v) for v in np.ravel(b0_dir)), kind, real)
    D = _device_kernels.get(key)
    if D is None:
        if len(_device_kernels) >= 4:
            _device_kernels.pop(next(iter(_device_kernels)))
        D = _device_kernels[key] = xp.asarray(dipole_kernel(shape, voxel_size, b0_dir, kind, real))
    return D


def padded_shape(shape, pad_factor=1.0, min_pad=0):
    """Shape padded by pad_factor (and at least min_pad voxels per side),
    rounded up to FFT-friendly sizes."""
    out = []
    for n in shape:
        m = max(int(np.ceil(n * pad_factor)), n + 2 * min_pad)
        out.append(sfft.next_fast_len(m, real=False))
    return tuple(out)


def pad_center(a, shape, value=0.0):
    """Embed a in the center of a new array of the given shape."""
    out = xp_of(a).full(shape, value, dtype=a.dtype)
    sl = tuple(slice((S - s) // 2, (S - s) // 2 + s) for s, S in zip(a.shape, shape))
    out[sl] = a
    return out, sl


def forward_field(chi, voxel_size, b0_dir=(0, 0, 1), pad_factor=2.0, kind="continuous",
                  background=0.0):
    """Field (ppm) produced by susceptibility distribution chi (ppm).

    The volume is zero-padded (with chi=background, i.e. the medium beyond the
    FOV) by pad_factor to suppress periodic wrap-around of the dipole field.
    """
    chi = np.asarray(chi, dtype=np.float32) - np.float32(background)
    shape = padded_shape(chi.shape, pad_factor)
    big, sl = pad_center(chi, shape)
    D = dipole_kernel(shape, voxel_size, b0_dir, kind, real=True)
    spec = sfft.rfftn(big, workers=WORKERS)
    del big
    spec *= D
    del D
    field = sfft.irfftn(spec, s=shape, workers=WORKERS)[sl]
    return np.ascontiguousarray(field, dtype=np.float32)


def convolve_kernel(a, K):
    """Circular convolution of real array a with a k-space kernel K (same shape)."""
    return ifftn(K * fftn(a)).real.astype(np.float32)
