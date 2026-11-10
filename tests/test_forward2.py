"""Forward model v2 unit tests (GPU_Plan.md, phase 2): geometry, painter, interpolation, vein store, reduction to v1."""
import numpy as np
import pytest
from scipy import ndimage

from qsmsim import finegrid as fg
from qsmsim import forward2 as f2
from qsmsim import phantom as ph
from qsmsim import signal as sg
from qsmsim import veins as vn
from qsmsim.exp1 import config


def test_v2_config_geometry():
    c = config.production()
    c.forward, c.resolutions = "v2", [0.8, 1.0, 1.5, 2.0]
    c.check_v2()
    assert c.fine_per_master() == 5 and [c.fine_per_voxel(r) for r in c.resolutions] == [8, 10, 15, 20]
    for bad in ({"resolutions": [0.75]}, {"fine_mm": 0.25}, {"v2_core_mm": 20.0}, {"refine_halo_mm": 13.0},
                {"refine_halo_mm": 0.9}):                   # vein sub-tile 2 x 78 = 156 points: not FFT-friendly
        b = config.production()
        b.forward, b.resolutions = "v2", [1.0]
        for k, v in bad.items():
            setattr(b, k, v)
        with pytest.raises(ValueError):
            b.check_v2()
    assert config.Config.from_json(config.production().to_json()).forward == "v1"
    g = f2.Geom((480, 480, 408), 0.5, 5, core=240, halo=40)
    t = g.tiles()
    assert len(t) == 10 * 10 * 9 and t[-1][2] == (1920, 2040)         # 204 mm: 8 full tiles + one of 12 mm
    ax = ph.Grid(g.shape_m, (0.5,) * 3).axes
    for d in range(3):
        j = np.arange(-7, g.shape_f[d] + 7)
        x = g.coords(d, j)
        assert np.allclose(np.diff(x), 0.1)
        assert np.array_equal(g.coords(d, 5 * np.arange(g.shape_m[d]) + 2), ax[d])   # centre point = master point
        xr = g.coords(d, np.arange(40), R=2)
        assert np.allclose(xr.reshape(-1, 2).mean(1), x[7:27])


@pytest.mark.parametrize("mode", ["linear", "constant"])
def test_upsample_matches_map_coordinates(mode):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((9, 8, 7)).astype(np.float32)
    F = 5
    box = ((-6, 41), (3, 37), (10, 36))                        # starts / ends beyond the array on purpose
    got = f2.upsample(a, F, box, (0, 0, 0), mode)
    c = [(np.arange(b0, b1) + 0.5) / F - 0.5 for b0, b1 in box]
    if mode == "constant":
        c = [np.clip(np.floor((np.arange(b0, b1)) / F), 0, n - 1) for (b0, b1), n in zip(box, a.shape)]
    C = np.stack(np.meshgrid(*c, indexing="ij")).reshape(3, -1)
    ref = ndimage.map_coordinates(a, C, order=1 if mode == "linear" else 0, mode="nearest").reshape(got.shape)
    assert np.abs(got - ref).max() < 1e-5
    # from_master on a sub-box agrees with the whole-array result
    g = f2.Geom(a.shape, 1.0, F, core=10, halo=5)
    sub = f2.from_master(a, g, ((7, 23), (0, 40), (12, 30)), np, mode)
    full = f2.upsample(a, F, ((0, 45), (0, 40), (0, 35)), (0, 0, 0), mode)
    assert np.abs(sub - full[7:23, 0:40, 12:30]).max() < 1e-6


def test_v2_painter_centre_samples_equal_v1():
    h, F = 2.0, 5
    P = ph.build_phantom((240, 240, 204), h, 1, 0.005, 3.0, seed=11)
    phi0 = sg.transceive_phase(P.grid, np.random.default_rng(3), np.pi)
    src = f2.AnalyticSource(ph.texture_field(P.grid.shape, h, 0.005, 3.0, 11), phi0)
    g = f2.Geom(P.grid.shape, h, F, core=60, halo=10)
    for core in [((240, 300), (300, 360), (240, 300)), g.tiles()[-1]]:  # cortex/deep nuclei; a partial edge tile
        box = f2._grow(core, g.halo)
        t = src.paint(g, box, np)
        m = [(max(b0, 0) // F + (max(b0, 0) % F > 2), min(b1, g.shape_f[d]) // F) for d, (b0, b1) in enumerate(box)]
        msl = tuple(slice(a, b) for a, b in m)
        fsl = tuple(slice(a * F + 2 - box[d][0], b * F - box[d][0], F) for d, (a, b) in enumerate(m))
        for k in ("labels", "chi", "m0", "r2s"):
            assert np.array_equal(t[k][fsl], getattr(P, k)[msl]), k
        assert np.array_equal(t["phi0"][fsl], phi0[msl])
    assert len(np.unique(t["labels"])) >= 1


def _cube(n=32, h=0.5):
    g = ph.Grid((n,) * 3, (h,) * 3)
    X, Y, Z = np.meshgrid(*g.axes, indexing="ij")
    lab = np.full(g.shape, ph.L.WM, np.uint8)
    lab[X + 0.3 * Y > 2] = ph.L.GM
    lab[X + 0.3 * Y > 5] = ph.L.CSF
    lut = np.zeros((max(ph.TISSUES) + 1, 3), np.float32)
    for k, v in ph.TISSUES.items():
        lut[k] = v
    chi = lut[lab, 0] + ph.texture_field(g.shape, h, 0.005, 3.0, 1) * np.isin(lab, ph.TEXTURED)
    M = fg.Master(g, lab, chi.astype(np.float32), lut[lab, 1], lut[lab, 2], np.isin(lab, ph.INTRACRANIAL),
                  sg.transceive_phase(g, np.random.default_rng(1), 1.0), None, 0.0)
    pts, radii = [], []
    for k, (deg, r) in enumerate([(90, 0.08), (60, 0.2)]):
        a = np.radians(deg)
        u = np.array([np.sin(a) * np.cos(0.9 * k), np.sin(a) * np.sin(0.9 * k), np.cos(a)])
        c = np.array([(k - 0.5) * 4 + 0.13, 0.07, 0.03])
        pts.append(np.linspace(c - 4 * u, c + 4 * u, 17))
        radii.append(np.full(17, r))
    return M, vn.VeinNetwork(pts, radii, ["dmv"] * 2, vn.VeinParams())


def test_vein_store_refined_block_mean():
    M, net = _cube()
    g = f2.Geom(M.grid.shape, 0.5, 5, core=80, halo=20)
    s1 = f2.VeinStore(net, g, R=1, log=lambda m: None)
    s2 = f2.VeinStore(net, g, R=2, log=lambda m: None)
    box = ((40, 120), (40, 120), (40, 120))
    fr = s2.refined(tuple((2 * a, 2 * b) for a, b in box))
    assert np.abs(ph.block_mean(fr, 2) - s2.fine(box)).max() < 1e-6
    assert abs(s2.volume_mm3 / s1.volume_mm3 - 1) < 0.02                   # both rasterizations conserve area
    o = np.array([g.coords(d, 0) for d in range(3)])
    ref = vn.fraction_on_grid(net, o, g.shape_f, g.hf)
    assert np.array_equal(s1.fine(tuple((0, n) for n in g.shape_f)), ref)


def test_v2_reduces_to_v1():
    """G.3 in small: tissue constant per master cell, veins at 0.1 mm, every voxel sub-sampled (v1 with an unlimited
    near zone) -> the v1 noiseless signal to float32 tolerance."""
    M, net = _cube()
    tes = (4.0 + 5.4 * np.arange(5)) * 1e-3
    dchi, nz, _ = fg.build_dmv(M, net, F=5, core_mm=8.0, halo_mm=2.0, near_mm=(1e9, 1e9), log=lambda m: None)
    mesh1, chi1 = fg.mesh_field(M, dchi, pad_factor=2.0)
    fg.finalize(nz, mesh1)
    S1 = fg.synthesize(M, mesh1, nz, net.params, tes, [2, 4], 3.0, log=lambda m: None)
    g = f2.Geom(M.grid.shape, 0.5, 5, core=80, halo=20)
    src = f2.MasterSource(M, {"wm": [int(ph.L.WM)], "intracranial": []})
    mod = f2.Model(g, src, f2.VeinStore(net, g, R=1, log=lambda m: None), net.params, acq=[10, 20])
    chi_m, truth = mod.mesh_pass(np, log=lambda m: None)
    assert np.abs(chi_m - chi1).max() < 1e-6
    mesh2 = f2.mesh_field(chi_m, 0.5, pad_factor=2.0)
    S2, fld = mod.signal_pass(mesh2, tes, 3.0, xp=np, log=lambda m: None)
    for f, n in ((2, 10), (4, 20)):
        assert np.abs(S2[n] - S1[f]).max() < 1e-5 * np.abs(S1[f]).max()
        assert np.abs(truth[n]["chi"] - ph.block_mean(chi1, f)).max() < 1e-6
    assert np.abs(fld[20] - ph.block_mean(mesh1, 4)).max() < 0.01        # sub-voxel field detail only near veins


class _StrictArray(np.ndarray):
    """CuPy's scalar rule on a numpy array: a Python scalar must be exactly int/float/bool/complex (CuPy rejects an
    IntEnum label under NumPy 2; numpy scalars are fine)."""
    def __setitem__(self, k, v):
        assert not (isinstance(v, int) and type(v) not in (int, bool)), f"{type(v).__name__} scalar reaches xp"
        super().__setitem__(k, v)


class _StrictXp:
    def __getattr__(self, name):
        return getattr(np, name)

    def full(self, shape, fill, dtype=None):
        assert not (isinstance(fill, int) and type(fill) not in (int, bool)), f"{type(fill).__name__} fill reaches xp"
        return np.full(shape, fill, dtype).view(_StrictArray)


def test_painter_scalars_cupy_safe():
    g = ph.Grid((48, 48, 40), (4.0,) * 3)
    assert np.array_equal(np.asarray(ph.build_labels(g, xp=_StrictXp())), ph.build_labels(g))
