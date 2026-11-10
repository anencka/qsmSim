"""CPU/GPU backend (GPU_Plan.md, phase 1).

The first group runs everywhere: the GPU path's CG (backend.cg) is checked against scipy's on numpy arrays,
and numpy inputs must stay on numpy. The second group (marker `gpu`) compares every ported step on the GPU
with the CPU result; it is skipped where no CUDA device is available and runs in the GPU image with a device
attached (docker/qsmsim.sh test --gpu, docker/gpu_check.sh). QSMSIM_TEST_FAKE_GPU=1 runs that group CPU against CPU, to
check the tests themselves on a machine without a GPU.
"""
import os

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, cg

from qsmsim import background as bg
from qsmsim import backend as be
from qsmsim import dipole as dp
from qsmsim import echofit, inversion, phantom as ph, tfi, unwrap


def _spd(n=400, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, n)).astype(np.float32)
    return (q @ q.T / n + np.eye(n, dtype=np.float32) * 0.05).astype(np.float32), rng.normal(size=n).astype(np.float32)


@pytest.mark.parametrize("dots64", [False, True])
def test_backend_cg_follows_scipy(dots64):
    A, b = _spd()
    ref, _ = cg(LinearOperator(A.shape, matvec=lambda v: A @ v, dtype=np.float32), b, rtol=1e-6, maxiter=500)
    got, info = be.cg(lambda v: A @ v, b, rtol=1e-6, maxiter=500, dots64=dots64)
    assert info == 0
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4
    # fixed iteration count, warm start and a preconditioner, as TFI / the LSQ unwrap use it
    d = np.diag(A).copy()
    x0 = np.full_like(b, 0.1)
    n1, n2 = [0], [0]
    ref, _ = cg(LinearOperator(A.shape, matvec=lambda v: A @ v, dtype=np.float32), b, x0=x0.copy(), rtol=1e-12,
                maxiter=7, M=LinearOperator(A.shape, matvec=lambda v: v / d, dtype=np.float32),
                callback=lambda _: n1.__setitem__(0, n1[0] + 1))
    got, _ = be.cg(lambda v: A @ v, b, x0=x0, rtol=1e-12, maxiter=7, psolve=lambda v: v / d,
                   callback=lambda _: n2.__setitem__(0, n2[0] + 1), dots64=dots64)
    assert n1 == n2 == [7]
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4


def test_backend_cg_history():
    """history: one |r|/|b| per stopping test, so a converged run stopped at len - 1 iterations (scipy's count);
    rtol=0 runs exactly maxiter iterations and leaves the solution unchanged."""
    A, b = _spd()
    n = [0]
    cg(LinearOperator(A.shape, matvec=lambda v: A @ v, dtype=np.float32), b, rtol=1e-3, maxiter=500,
       callback=lambda _: n.__setitem__(0, n[0] + 1))
    h = []
    x, info = be.cg(lambda v: A @ v, b, rtol=1e-3, maxiter=500, dots64=False, history=h)
    assert info == 0 and len(h) - 1 == n[0] and h[-1] < 1e-3 <= h[-2]
    h2 = []
    x2, _ = be.cg(lambda v: A @ v, b, rtol=0.0, maxiter=len(h) - 1, dots64=False, history=h2)
    assert np.array_equal(x, x2) and len(h2) == len(h) and np.allclose(h2, h)


def test_numpy_inputs_stay_on_numpy():
    a = np.zeros(3)
    assert be.xp_of(a) is np and be.to_host(a) is a and not be.is_gpu(a)


# ------------------------------------------------------------------ GPU vs CPU

def _has_gpu():
    cp = be.cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


FAKE = os.environ.get("QSMSIM_TEST_FAKE_GPU") == "1"
gpu = pytest.mark.skipif(not (_has_gpu() or FAKE), reason="no CUDA device")


@pytest.fixture(scope="module")
def data():
    """P1 at 4 mm: chi, mask, local and total field, magnitude, and a 6-echo noisy signal."""
    res = 4.0
    P = ph.build_phantom((240.0, 240.0, 204.0), res, 1, 0.005, 3.0, seed=1)
    mask = ph.brain_mask(P.labels, res, 3.0)
    vox = (res,) * 3
    rng = np.random.default_rng(3)
    total = dp.forward_field(P.chi, vox, (0, 0, 1), 2.0, background=float(ph.TISSUES[ph.L.AIR][0])).astype(np.float32)
    local = (dp.forward_field(np.where(mask, P.chi, 0), vox, (0, 0, 1), 2.0) * mask).astype(np.float32)
    tes = (4.0 + 5.4 * np.arange(6)) * 1e-3
    w = 2 * np.pi * 42.577478 * 3.0
    S = (P.m0[..., None] * np.exp(-P.r2s[..., None] * tes) * np.exp(1j * w * total[..., None] * tes)).astype(np.complex64)
    S += (rng.normal(0, 0.01, S.shape) + 1j * rng.normal(0, 0.01, S.shape)).astype(np.complex64)
    return dict(mask=mask, vox=vox, total=total, local=local, m0=P.m0.astype(np.float32), S=S, tes=tes)


def _rel(a, b, m):
    a, b = be.to_host(a)[m], np.asarray(b)[m]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _d(a):
    return a if FAKE else be.to_device(a)


def _finite(a, m, what):
    h = be.to_host(a)[m]
    bad = int((~np.isfinite(h)).sum())
    assert bad == 0, f"{what}: {bad} of {h.size} values not finite"


@gpu
def test_gpu_unwrap(data):
    m = data["mask"]
    ph_w = unwrap.wrap(data["total"] * 2.5).astype(np.float32)
    est = unwrap.lsq_unwrap(_d(ph_w), _d(m))
    _finite(est, m, "lsq_unwrap estimate")
    assert _rel(est, unwrap.lsq_unwrap(ph_w, m), m) < 1e-4
    assert _rel(unwrap.unwrap(_d(ph_w), _d(m)), unwrap.unwrap(ph_w, m), m) < 1e-5


@gpu
def test_gpu_field_fit(data):
    m = data["mask"]
    c = echofit.fit_field(data["S"], data["tes"], m, data["vox"], 3.0)
    g = echofit.fit_field(_d(data["S"]), data["tes"], _d(m), data["vox"], 3.0)
    assert FAKE or be.is_gpu(g.field_ppm)
    for k in ("field_ppm", "field_std_ppm", "phi0", "weights"):
        _finite(getattr(g, k), m, f"fit_field {k}")
    assert _rel(g.field_ppm, c.field_ppm, m) < 1e-5
    assert _rel(g.weights, c.weights, m) < 1e-4


@gpu
def test_gpu_background_removal(data):
    m, f = data["mask"], data["total"]
    radii = np.arange(12.0, 4.0 - 1e-6, -4.0)
    lc, mc = bg.vsharp(f, m, data["vox"], radii_mm=radii)
    lg, mg = bg.vsharp(_d(f), _d(m), data["vox"], radii_mm=radii)
    assert np.array_equal(be.to_host(mg), mc)
    assert _rel(lg, lc, mc) < 1e-4
    lc, _ = bg.pdf(f, m, data["vox"])
    lg, _ = bg.pdf(_d(f), _d(m), data["vox"])
    assert _rel(lg, lc, m) < 1e-3


@gpu
@pytest.mark.parametrize("method", ["tkd", "cfl2", "tv", "medi"])
def test_gpu_inversions(data, method):
    m, f, vox = data["mask"], data["local"], data["vox"]
    if method == "tkd":
        run = lambda F, M: inversion.tkd(F, M, vox)
    elif method == "cfl2":
        run = lambda F, M: inversion.cfl2(F, M, vox)
    elif method == "tv":
        run = lambda F, M: inversion.tv_admm(F, M, vox, maxiter=30, tol=0.0)[0]
    else:
        mag = data["m0"]
        run = lambda F, M: inversion.medi_like(F, M, vox, be.like(mag, F), maxiter=30, tol=0.0)[0]
    assert _rel(run(_d(f), _d(m)), run(f, m), m) < 1e-4


@gpu
def test_gpu_tfi(data):
    m = data["mask"]
    kw = dict(outer=2, cg_iters=10, tol=0.0, pad_factor=1.5, bg_weight=0.0, lam=3e-3)
    c, _, ic = tfi.tfi(data["total"], m, data["vox"], data["m0"], **kw)
    g, _, ig = tfi.tfi(_d(data["total"]), _d(m), data["vox"], _d(data["m0"]), **kw)
    rel = _rel(g, c, m)
    print(f"TFI GPU vs CPU rel. diff {rel:.2e}, CG {ic['cg']} vs {ig['cg']}")
    assert rel < 1e-2


# ---- forward model v2 (GPU_Plan.md phase 2, G.5 in small) ----------------------------------------------------------

def _xp():
    return np if FAKE else be.cupy()


@gpu
def test_gpu_painter_labels():
    from qsmsim import forward2 as f2
    g = f2.Geom((120, 120, 102), 2.0, 5, core=60, halo=10)
    src = f2.AnalyticSource(ph.texture_field(g.shape_m, 2.0, 0.005, 3.0, 1), np.zeros(g.shape_m, np.float32))
    box = f2._grow(((240, 300), (300, 360), (240, 300)), 10)
    c, d = src.paint(g, box, np), src.paint(g, box, _xp())
    assert int((be.to_host(d["labels"]) != c["labels"]).sum()) <= 2           # a coordinate exactly on a surface
    for k in ("chi", "m0", "r2s", "phi0"):
        assert np.abs(be.to_host(d[k]) - c[k])[be.to_host(d["labels"]) == c["labels"]].max() < 1e-6, k


@gpu
def test_gpu_forward2_tiles():
    """Tiles with tissue, veins and the 2x vein refinement on both backends."""
    from qsmsim import finegrid as fg, forward2 as f2, signal as sg, veins as vn
    n, h = 32, 0.5
    g = ph.Grid((n,) * 3, (h,) * 3)
    X, Y, Z = np.meshgrid(*g.axes, indexing="ij")
    lab = np.where(X + 0.3 * Y > 2, ph.L.GM, ph.L.WM).astype(np.uint8)
    lab[X + 0.3 * Y > 5] = ph.L.CSF
    lut = np.zeros((max(ph.TISSUES) + 1, 3), np.float32)
    for k, v in ph.TISSUES.items():
        lut[k] = v
    M = fg.Master(g, lab, lut[lab, 0], lut[lab, 1], lut[lab, 2], np.isin(lab, ph.INTRACRANIAL),
                  sg.transceive_phase(g, np.random.default_rng(1), 1.0), None, 0.0)
    u = np.array([0.8, 0.3, 0.52]) / np.linalg.norm([0.8, 0.3, 0.52])
    net = vn.VeinNetwork([np.linspace(-4 * u, 4 * u, 17) + 0.13], [np.full(17, 0.15)], ["dmv"], vn.VeinParams())
    tes = (4.0 + 5.4 * np.arange(4)) * 1e-3
    geom = f2.Geom(g.shape, h, 5, core=80, halo=20)
    store = f2.VeinStore(net, geom, R=2, log=lambda m: None)
    out = []
    for xp in (np, _xp()):
        mod = f2.Model(geom, f2.MasterSource(M, {"wm": [int(ph.L.WM)], "intracranial": []}), store, net.params,
                       acq=[10], refine_core=30, refine_halo=10)
        chi_m, truth = mod.mesh_pass(xp, log=lambda m: None)
        mesh = f2.mesh_field(chi_m, h, pad_factor=2.0)
        S, fld = mod.signal_pass(mesh, tes, 7.0, xp=xp, log=lambda m: None)
        out.append((chi_m, S[10], fld[10], mod.stats["refined_subtiles"]))
    (c0, s0, f0, r0), (c1, s1, f1, r1) = out
    assert r0 == r1 > 0
    assert np.abs(c1 - c0).max() < 1e-6 and np.abs(f1 - f0).max() < 1e-5
    assert np.abs(s1 - s0).max() < 1e-5 * np.abs(s0).max()
