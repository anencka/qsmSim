import numpy as np
import pytest

from qsmsim import background as bg
from qsmsim import dipole as dp
from qsmsim import echofit, inversion, metrics, phantom as ph, signal as sg
from qsmsim.unwrap import unwrap, wrap


def sphere(shape, center, r):
    g = np.indices(shape) - (np.array(shape)[:, None, None, None] - 1) / 2.0
    c = np.asarray(center, float)[:, None, None, None]
    return np.sqrt(((g - c) ** 2).sum(0)) <= r


# ---- dipole kernel / forward model ------------------------------------------

def test_kernel_range_and_dc():
    D = dp.dipole_kernel((32, 32, 32), (1, 1, 1))
    assert D[0, 0, 0] == 0
    assert D.min() >= -2 / 3 - 1e-6 and D.max() <= 1 / 3 + 1e-6


def test_rfft_kernel_matches_full():
    D = dp.dipole_kernel((16, 18, 20), (1, 1.2, 1.5), (0.2, 0.1, 1))
    Dr = dp.dipole_kernel((16, 18, 20), (1, 1.2, 1.5), (0.2, 0.1, 1), real=True)
    assert np.allclose(D[:, :, :10], Dr[:, :, :10])     # Nyquist plane differs for oblique B0 (sign of kz)


def test_sphere_field_matches_analytic():
    n, R, chi = 96, 8, 1.0
    s = sphere((n, n, n), (0, 0, 0), R).astype(np.float32) * chi
    f = dp.forward_field(s, (1, 1, 1), pad_factor=2.0)
    c = (n - 1) / 2.0
    # Lorentz-corrected interior field of a sphere is 0
    assert abs(f[sphere((n, n, n), (0, 0, 0), R - 3)].mean()) < 0.02 * chi
    # exterior on the B0 axis: 2 chi R^3 / (3 r^3)
    vol_r = (3 * s.sum() / (4 * np.pi)) ** (1 / 3)        # effective voxelized radius
    for r in (14.5, 20.5, 26.5):
        k = int(c + r)
        expect = 2 * chi * vol_r**3 / (3 * (k - c) ** 3)
        assert f[n // 2, n // 2, k] == pytest.approx(expect, rel=0.1)


def test_tilt_unit_vector():
    b = dp.b0_from_tilt((30, 0))
    assert np.linalg.norm(b) == pytest.approx(1)
    assert b[2] == pytest.approx(np.cos(np.deg2rad(30)))


# ---- unwrapping and field fitting --------------------------------------------

@pytest.mark.parametrize("method", ["lsq"])
def test_unwrap_recovers_smooth_phase(method):
    shape = (48, 48, 48)
    g = np.indices(shape).astype(np.float32)
    true = 0.02 * ((g[0] - 24) ** 2 + (g[1] - 20) ** 2) + 0.3 * g[2]   # incl. a harmonic ramp
    mask = sphere(shape, (0, 0, 0), 20)
    u = unwrap(wrap(true).astype(np.float32), mask, method=method)
    d = (u - true)[mask]
    assert np.allclose(d - d[0], 0, atol=1e-4)                         # exact up to 2*pi*k
    assert np.allclose(np.round(d[0] / (2 * np.pi)) * 2 * np.pi, d[0], atol=1e-4)


def test_field_fit_noiseless_with_aliasing():
    shape = (40, 40, 40)
    g = np.indices(shape).astype(np.float32)
    mask = sphere(shape, (0, 0, 0), 17)
    field = 0.08 * (g[0] - 20) + 0.5     # ppm; up to ~2 ppm -> aliases between echoes at 3 T
    tes = sg.echo_times(4, 4, 5)
    phi0 = np.float32(0.7) * np.ones(shape, np.float32)
    S = np.stack([sg.gre_echo(np.ones(shape, np.float32), np.full(shape, 20, np.float32),
                              field, phi0, t, 3.0) for t in tes], -1)
    fit = echofit.fit_field(S, tes, mask, (1, 1, 1), 3.0)
    err = fit.field_ppm[mask] - field[mask]
    assert np.allclose(err - np.median(err), 0, atol=1e-4)
    # global offset only by a whole number of cycles over dTE (unobservable)
    cyc = 1 / (sg.hz_per_ppm(3.0) * 4e-3)
    assert np.median(err) / cyc == pytest.approx(round(np.median(err) / cyc), abs=1e-3)


def test_field_std_matches_empirical_noise():
    shape = (24, 24, 24)
    mask = np.ones(shape, bool)
    tes = sg.echo_times(4, 4, 6)
    rng = np.random.default_rng(3)
    S = np.stack([sg.add_noise(sg.gre_echo(np.ones(shape, np.float32), np.full(shape, 20, np.float32),
                                           np.zeros(shape, np.float32), np.zeros(shape, np.float32), t, 3.0),
                               50, rng) for t in tes], -1)
    fit = echofit.fit_field(S, tes, mask, (1, 1, 1), 3.0)
    emp = fit.field_ppm[mask].std()
    assert np.median(fit.field_std_ppm[mask]) == pytest.approx(emp, rel=0.15)


# ---- background removal -------------------------------------------------------

@pytest.fixture(scope="module")
def bg_case():
    n = 80
    shape = (n, n, n)
    mask = sphere(shape, (0, 0, 0), 26)
    inner = sphere(shape, (5, -4, 3), 4).astype(np.float32) * 0.2
    outer = sphere(shape, (0, 0, -34), 6).astype(np.float32) * 9.0    # "sinus" below the mask
    loc = dp.forward_field(inner, (1, 1, 1)) * mask
    tot = dp.forward_field(inner + outer, (1, 1, 1)) * mask
    return tot, loc, mask


@pytest.mark.parametrize("method", ["vsharp", "vsharp_fov", "pdf", "sharp", "lbv"])
def test_background_removal(bg_case, method):
    tot, loc, mask = bg_case
    fn = {"vsharp": lambda: bg.vsharp(tot, mask, (1, 1, 1)),
          "vsharp_fov": lambda: bg.vsharp(tot, mask, (1, 1, 1), domain="fov"),
          "sharp": lambda: bg.sharp(tot, mask, (1, 1, 1), radius_mm=5),
          "pdf": lambda: bg.pdf(tot, mask, (1, 1, 1)),
          "lbv": lambda: bg.lbv(tot, mask, (1, 1, 1))}[method]
    est, m = fn()
    before = metrics.nrmse(metrics.reference(tot, m), metrics.reference(loc, m), m)
    after = metrics.nrmse(metrics.reference(est, m), metrics.reference(loc, m), m)
    assert after < 25 and after < before / 10


# ---- inversion ----------------------------------------------------------------

@pytest.fixture(scope="module")
def inv_case():
    n = 64
    shape = (n, n, n)
    mask = sphere(shape, (0, 0, 0), 26)
    chi = np.zeros(shape, np.float32)
    chi[sphere(shape, (8, 0, 0), 6)] = 0.1
    chi[sphere(shape, (-9, 5, 4), 4)] = -0.05
    chi[sphere(shape, (0, -10, -8), 3)] = 0.3
    chi *= mask
    f = dp.forward_field(chi, (1, 1, 1)) * mask
    return f, chi, mask


@pytest.mark.parametrize("method,limit", [("tkd", 45), ("cfl2", 60), ("tv", 25), ("medi", 25)])
def test_inversions_noiseless(inv_case, method, limit):
    f, chi, mask = inv_case
    vox = (1, 1, 1)
    if method == "tkd":
        x = inversion.tkd(f, mask, vox)
    elif method == "cfl2":
        x = inversion.cfl2(f, mask, vox, lam=1e-3)
    elif method == "tv":
        x, _ = inversion.tv_admm(f, mask, vox, alpha=1e-5, maxiter=100)
    else:
        x, _ = inversion.medi_like(f, mask, vox, magnitude=1 + chi * 5, alpha=1e-5, maxiter=100)
    err = metrics.nrmse(metrics.reference(x, mask), metrics.reference(chi, mask), mask)
    assert err < limit


def test_grad_adjoint():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 9, 10)).astype(np.float64)
    g = [rng.standard_normal((8, 9, 10)) for _ in range(3)]
    lhs = sum((a * b).sum() for a, b in zip(inversion.grad(x), g))
    rhs = (x * inversion.grad_adj(g)).sum()
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ---- metrics / phantom --------------------------------------------------------

def test_metrics_identity():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((20, 20, 20)).astype(np.float32)
    m = np.ones_like(t, bool)
    assert metrics.nrmse(t, t, m) == 0
    assert metrics.ssim(t, t, m) == pytest.approx(1.0)
    assert metrics.hfen(t, t, m) == 0


def test_phantom_coarse():
    P = ph.build_phantom(voxel_mm=3.0, texture_ppm=0)
    labs = set(np.unique(P.labels))
    for want in (ph.L.GP, ph.L.RN, ph.L.SN, ph.L.SINUS, ph.L.VEIN, ph.L.DURAL):
        assert want in labs
    m = ph.brain_mask(P.labels, 3.0)
    assert not np.any(m & np.isin(P.labels, [ph.L.AIR, ph.L.SINUS, ph.L.BONE, ph.L.DURAL]))


# ---- total field inversion ------------------------------------------------------

def test_tfi_handles_background_comparably_to_pdf():
    n = 64
    shape = (n, n, n)
    mask = sphere(shape, (0, 0, 0), 20)
    chi_in = np.zeros(shape, np.float32)
    chi_in[sphere(shape, (5, -3, 2), 5)] = 0.1
    chi_in[sphere(shape, (-6, 4, -4), 3)] = 0.3
    chi_in *= mask
    outer = sphere(shape, (0, 0, -27), 5).astype(np.float32) * 9.0          # "sinus" below the mask
    total = dp.forward_field(chi_in + outer, (1, 1, 1)) * mask
    err = lambda x: metrics.nrmse(metrics.reference(x, mask), metrics.reference(chi_in, mask), mask)
    from qsmsim.tfi import tfi
    x, full, info = tfi(total, mask, (1, 1, 1), magnitude=1 + 5 * chi_in, lam=3e-5, pb=30, bg_weight=0.0,
                        pad_factor=1.25, outer=15, cg_iters=60)
    naive, _ = inversion.tv_admm(total, mask, (1, 1, 1), alpha=1e-5, maxiter=100)     # no background removal
    loc, _ = bg.pdf(total, mask, (1, 1, 1))
    two_step, _ = inversion.tv_admm(loc, mask, (1, 1, 1), alpha=1e-5, maxiter=100)
    assert err(x) < err(naive) / 10            # ~35 % vs ~860 %
    assert err(x) < 2 * err(two_step)          # PDF + TV ~25 %
    assert full.shape == x.shape and info["outer"] >= 1


# ---- RC2 metric port (synthetic checks; the leaderboard check needs the downloaded data) ----

def test_rc2_metrics_identity_and_scaling():
    from qsmsim import rc2
    rng = np.random.default_rng(2)
    shape = (40, 40, 40)
    seg = np.full(shape, 8, np.int16)
    for k in range(1, 7):                                   # six deep-GM labels
        seg[5 + 5 * k:8 + 5 * k, 10:16, 10:16] = k
    seg[18:21, 25:28, 25:28] = 16                           # calcification
    seg[30:33, 5:35, 20] = 11                               # vein
    chi = rng.normal(0, 0.02, shape) + np.where(seg <= 6, 0.1, 0) - np.where(seg == 16, 1.0, 0)
    mask = np.ones(shape, np.float32)
    sim = dict(chi=chi, mask=mask, seg=seg, calc_moment=0.0)
    m = rc2.evaluate(chi, sim)
    assert m["rmse"] < 1e-9 and m["rmse_detrend"] < 1e-9 and m["DeviationFromLinearSlope"] < 1e-9
    m2 = rc2.evaluate(0.5 * chi, sim)                       # uniform underestimation
    assert m2["rmse"] == pytest.approx(50, rel=1e-6)        # demeaned NRMSE of a 0.5x map
    assert m2["rmse_detrend"] < 1e-6                         # detrending removes the scale
    assert m2["DeviationFromLinearSlope"] == pytest.approx(0.5, rel=1e-6)
