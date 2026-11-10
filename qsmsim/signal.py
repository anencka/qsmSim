"""Multi-echo GRE signal simulation.

    S_n(r) = M0(r) exp(-R2*(r) TE_n) exp(i [phi0(r) + 2 pi gamma B0 dB(r) TE_n]) + noise

with dB the total relative field (ppm) and phi0 a smooth transceive phase.
Positive field -> positive phase evolution (sign flips between vendors; flip
`phase_sign` to mimic a scanner that reports the opposite convention).
Noise is complex Gaussian with sigma = 1/SNR, i.e. SNR is defined for a
voxel with M0 = 1 (CSF) at TE = 0.
"""
from __future__ import annotations

import numpy as np

GAMMA_HZ_PER_T = 42.577478e6


def hz_per_ppm(b0_tesla):
    return GAMMA_HZ_PER_T * b0_tesla * 1e-6


def echo_times(te1_ms=4.0, dte_ms=4.0, n_echoes=6):
    return (te1_ms + dte_ms * np.arange(n_echoes)) * 1e-3


def smooth_poly_field(grid, coeffs, extent_mm=100.0):
    """Low-order 3-D polynomial (orders 0..2) on the grid. coeffs is a dict
    {'x':..,'y':..,'z':..,'xy':..,'xz':..,'yz':..,'z2':..,'x2-y2':..,'c':..}
    with coordinates normalized by extent_mm. All terms listed are harmonic
    (zero Laplacian), so as a field offset they model shim imperfections."""
    x, y, z = [a / extent_mm for a in grid.axes]
    X, Y, Z = x[:, None, None], y[None, :, None], z[None, None, :]
    terms = {"c": 1.0, "x": X, "y": Y, "z": Z, "xy": X * Y, "xz": X * Z, "yz": Y * Z,
             "z2": Z**2 - 0.5 * (X**2 + Y**2), "x2-y2": X**2 - Y**2}
    out = np.zeros(grid.shape, np.float32)
    for k, c in coeffs.items():
        out += np.float32(c) * terms[k]
    return out


def transceive_phase(grid, rng, amplitude=np.pi):
    """Random smooth (quadratic) phase offset phi0, roughly within +/- amplitude."""
    keys = ["x", "y", "z", "xy", "xz", "yz", "z2", "x2-y2"]
    coeffs = {k: rng.uniform(-1, 1) for k in keys}
    coeffs["c"] = rng.uniform(-np.pi, np.pi)
    phi = smooth_poly_field(grid, coeffs)
    return (amplitude * phi / (np.abs(phi).max() + 1e-12)).astype(np.float32)


def gre_echo(m0, r2s, field_ppm, phi0, te, b0_tesla, phase_sign=1.0):
    """Noiseless complex signal for one echo (complex64)."""
    ph = phase_sign * (phi0 + 2 * np.pi * hz_per_ppm(b0_tesla) * field_ppm * te)
    return (m0 * np.exp(-r2s * te)).astype(np.float32) * np.exp(1j * ph.astype(np.float32))


def add_noise(sig, snr, rng):
    if not snr or np.isinf(snr):
        return sig
    sigma = np.float32(1.0 / snr)
    n = rng.standard_normal(sig.shape, dtype=np.float32) + 1j * rng.standard_normal(sig.shape, dtype=np.float32)
    return (sig + sigma * n).astype(np.complex64)
