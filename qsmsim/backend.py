"""Array backend: numpy/scipy on the CPU (default) or CuPy on the GPU (GPU_Plan.md, phase 1).

The reconstruction modules (unwrap, echofit, background, inversion, tfi) pick their array module from
their inputs, `xp = xp_of(array)`. numpy inputs run exactly the calls they always ran, in the same order,
so CPU results are unchanged bit for bit. CuPy inputs run the same algorithm on the GPU. The caller
decides where the data lives: with QSMSIM_BACKEND=gpu, exp1's Dataset moves its arrays to the device
and returns every chi map to the host for evaluation.

  QSMSIM_BACKEND=cpu|gpu    (default cpu)
  QSMSIM_GPU_MEM_GB=<GB>    cap on CuPy's memory pool (default: no cap)
"""
from __future__ import annotations

import os

import numpy as np

_cp = None


def cupy():
    """The cupy module, or None when it is not installed."""
    global _cp
    if _cp is None:
        try:
            import cupy as c
            _cp = c
        except Exception:                                # not installed / no CUDA libraries
            _cp = False
    return _cp or None


def use_gpu():
    return os.environ.get("QSMSIM_BACKEND", "cpu").strip().lower() == "gpu"


def is_gpu(a):
    cp = cupy() if _cp is not False else None
    return cp is not None and isinstance(a, cp.ndarray)


def xp_of(*arrays):
    """cupy when any argument is a CuPy array, else numpy."""
    for a in arrays:
        if is_gpu(a):
            return _cp
    return np


def to_device(a):
    """numpy array -> CuPy array (sets up the memory pool on first use); None passes through."""
    if a is None:
        return None
    setup()
    return cupy().asarray(a)


def to_host(a):
    if a is None:
        return None
    return a.get() if is_gpu(a) else a


def like(a, ref):
    """a on the device of ref."""
    return cupy().asarray(a) if is_gpu(ref) else np.asarray(a)


def percentile(a, q):
    """np.percentile; for a CuPy array computed on the host. Percentiles are not hot here (a few vectors per
    dataset), the values then equal the CPU's exactly, and the GPU path needs no device sort (CuPy's sort
    hit an illegal-address fault on an RTX A5000, CuPy 14.2)."""
    if is_gpu(a):
        h = a.get()
        if h.size == 0:
            raise ValueError("percentile of an empty selection (non-finite values upstream?)")
        return np.percentile(h, q)
    return np.percentile(a, q)


def fft(xp):
    if xp is np:
        import scipy.fft as m
    else:
        import cupyx.scipy.fft as m
    return m


def ndimage(xp):
    if xp is np:
        from scipy import ndimage as m
    else:
        import cupyx.scipy.ndimage as m
    return m


_ready = False


def sync():
    """Wait for the GPU, if anything has been put on it (a no-op otherwise)."""
    if _cp and _ready:
        _cp.cuda.Device().synchronize()


def setup():
    global _ready
    if _ready:
        return
    cp = cupy()
    if cp is None:
        raise RuntimeError("QSMSIM_BACKEND=gpu but CuPy is not available")
    gb = os.environ.get("QSMSIM_GPU_MEM_GB")
    if gb:
        cp.get_default_memory_pool().set_limit(size=int(float(gb) * 1e9))
    _ready = True


def pool_held_gb():
    """GB CuPy's default memory pool holds (used + cached blocks; blocks are not returned to the device while a job
    runs, so at its end this is about the job's peak), or None without the GPU."""
    if not (_cp and _ready):
        return None
    return _cp.get_default_memory_pool().total_bytes() / 1e9


def device_info():
    cp = cupy()
    if cp is None:
        return {}
    try:
        rt = cp.cuda.runtime
        p = rt.getDeviceProperties(cp.cuda.Device().id)
        name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
        return dict(gpu=name, gpu_mem_gb=p["totalGlobalMem"] / 1e9, cupy=cp.__version__,
                    cuda_runtime=rt.runtimeGetVersion(), cuda_driver=rt.driverGetVersion())
    except Exception as e:                               # no device / no driver: report it, do not fail
        return dict(gpu=f"unavailable ({type(e).__name__})", cupy=cp.__version__)


def cg(matvec, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, psolve=None, callback=None, dots64=None,
       history=None):
    """Conjugate gradients, step for step as scipy.sparse.linalg.cg (scipy 1.17), for any array module.

    The CPU path of every caller keeps using scipy's cg itself. This is the GPU path. With dots64 the inner
    products accumulate in float64 (the iterates stay float32): numpy's BLAS and CuPy sum float32 products
    in different orders, and in the ill-conditioned TFI system that difference grows over the iterations.
    dots64=None reads QSMSIM_CG_DOTS64 (default 1); 0 gives float32 inner products, the arithmetic variant the
    G.1 acceptance uses to measure how much an early-stopped CG moves under a change of arithmetic alone.
    history: a list that receives |r| / |b| at every stopping test (diagnostics; the iteration count at which a
    converged run stopped is len(history) - 1). rtol=0 runs exactly maxiter iterations.
    Returns (x, info) as scipy does: info = 0 when converged, else maxiter."""
    if dots64 is None:
        dots64 = os.environ.get("QSMSIM_CG_DOTS64", "1") != "0"
    xp = xp_of(b)
    dt = b.dtype

    def dot(u, v):
        return (u * v).sum(dtype=xp.float64) if dots64 else xp.dot(u, v)

    x = xp.zeros_like(b) if x0 is None else x0.astype(dt, copy=True)
    bnrm2 = float(xp.linalg.norm(b.astype(xp.float64) if dots64 else b))
    atol = max(float(atol), float(rtol) * bnrm2)
    if bnrm2 == 0:
        return b.copy(), 0
    if maxiter is None:
        maxiter = b.size * 10
    r = b - matvec(x) if bool(x.any()) else b.copy()
    rho_prev, p = None, None
    for it in range(maxiter):
        rn = float(xp.sqrt(dot(r, r)))
        if history is not None:
            history.append(rn / bnrm2)
        if rn < atol:
            return x, 0
        z = psolve(r) if psolve is not None else r
        rho = dot(r, z)
        if it > 0:
            p *= (rho / rho_prev).astype(dt)
            p += z
        else:
            p = z.copy()
        q = matvec(p)
        alpha = (rho / dot(p, q)).astype(dt)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho
        if callback is not None:
            callback(x)
    if history is not None:
        history.append(float(xp.sqrt(dot(r, r))) / bnrm2)
    return x, maxiter
