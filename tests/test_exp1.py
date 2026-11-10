"""Experiment 1 unit tests (Experiment_1.md, plan section 5)."""
import json

import numpy as np
import pytest
from scipy import ndimage

from qsmsim import dipole as dp
from qsmsim import finegrid as fg
from qsmsim import phantom as ph
from qsmsim import rc2_3t
from qsmsim import veins as vn
from qsmsim.exp1 import config, jobs, prepare, seeds
from qsmsim.exp1.evaluate import Evaluator
from qsmsim.exp1.runner import Runner, build_jobs


# ---- two-scale forward model -------------------------------------------------

def _empty_master(n=32, h=0.5):
    g = ph.Grid((n, n, n), (h,) * 3)
    z = np.zeros(g.shape, np.float32)
    return fg.Master(g, z.astype(np.uint8), z.copy(), np.ones_like(z), z.copy(), np.ones(g.shape, bool), z.copy())


@pytest.mark.parametrize("angle", [90, 45])
def test_two_scale_field_matches_brute_force_and_cylinder(angle):
    n, h, F, r = 32, 0.5, 5, 0.2
    M = _empty_master(n, h)
    a = np.radians(angle)
    u = np.array([np.sin(a), 0, np.cos(a)])
    c = np.array([0.07, 0.13, 0.03])
    pts = np.linspace(c - 7.5 * u, c + 7.5 * u, 31)
    net = vn.VeinNetwork([pts], [np.full(len(pts), r)], ["dmv"])
    dchi, nz, _ = fg.build_dmv(M, net, F=F, core_mm=4.0, halo_mm=3.0, log=lambda m: None)
    mesh, _ = fg.mesh_field(M, dchi, pad_factor=2.0)
    fg.finalize(nz, mesh)
    hf = h / F
    o = np.array([ax[0] for ax in M.grid.axes]) - h / 2 + hf / 2
    f_fine = vn.fraction_on_grid(net, o, (n * F,) * 3, hf)
    bf = dp.forward_field(f_fine * np.float32(vn.DMV_CHI), (hf,) * 3, (0, 0, 1), 2.0)
    ijk = np.array(np.unravel_index(nz.idx, M.grid.shape)).T
    offs = np.rint((fg.sub_offsets(F) + 0.5) * F - 0.5).astype(int)
    fi = (ijk[:, None, :] * F + offs[None]).reshape(-1, 3)
    ref = bf[fi[:, 0], fi[:, 1], fi[:, 2]].reshape(len(ijk), -1)
    assert np.sqrt(np.mean((nz.field - ref) ** 2)) / np.sqrt(np.mean(ref**2)) < 0.02
    # brute force vs analytic infinite cylinder near the middle of the vein
    X = [o[d] + hf * np.arange(n * F) for d in range(3)]
    P = np.stack(np.meshgrid(*X, indexing="ij"), -1).reshape(-1, 3)
    rr = P - c
    along = rr @ u
    rho = np.linalg.norm(rr - along[:, None] * u, axis=1)
    out = (np.abs(along) < 1.5) & (rho > 0.6) & (rho < 1.5)
    an = vn.cylinder_field(P[out], c, u, r, vn.DMV_CHI)
    assert np.sqrt(np.mean((bf.reshape(-1)[out] - an) ** 2)) / np.sqrt(np.mean(an**2)) < 0.05
    ins = (np.abs(along) < 1.5) & (rho < 0.08)
    expect = vn.cylinder_field(c[None], c, u, r, vn.DMV_CHI)[0]
    assert abs(bf.reshape(-1)[ins].mean() - expect) < 0.03 * vn.DMV_CHI


@pytest.mark.parametrize("radius", [0.05, 0.15, 0.25])
def test_partial_volume_conserves_vein_volume(radius):
    h, L = 0.1, 6.0
    pts = np.array([[-L / 2, 0.013, -0.021], [L / 2, 0.013, -0.021]])
    net = vn.VeinNetwork([pts], [np.full(2, radius)], ["dmv"])
    n = 80
    f = vn.fraction_on_grid(net, (-(n - 1) / 2 * h,) * 3, (n,) * 3, h)
    vol = f.sum() * h**3
    expect = np.pi * radius**2 * L + 4 / 3 * np.pi * radius**3       # capsule
    assert abs(vol / expect - 1) < 0.03


def test_kspace_resampling_aligned_with_block_average():
    n = 24
    g = ph.Grid((n, n, n), (0.5,) * 3)
    X, Y, Z = np.meshgrid(*g.axes, indexing="ij")
    m0 = np.exp(-((X - 0.9) ** 2 + (Y + 0.4) ** 2 + Z**2) / (2 * 2.0**2)).astype(np.float32)
    z = np.zeros(g.shape, np.float32)
    M = fg.Master(g, z.astype(np.uint8), z, m0, z, np.ones(g.shape, bool), z)
    nz = fg.NearZone(np.zeros(0, np.int64), np.zeros((0, 8), np.float32), np.zeros((0, 8), np.float32), 2)
    vp = vn.VeinParams()
    b = fg.synthesize(M, z, nz, vp, [0.004], (2, 3), resample="block", log=lambda m: None)
    k = fg.synthesize(M, z, nz, vp, [0.004], (2, 3), resample="kspace", log=lambda m: None)
    for f in (2, 3):
        bb, kk = b[f][..., 0].real, k[f][..., 0].real
        grids = np.meshgrid(*[np.arange(s) for s in bb.shape], indexing="ij")
        com = lambda a: np.array([np.sum(a * i) / a.sum() for i in grids])
        assert np.all(np.abs(com(kk) - com(bb)) < 0.01)
        assert np.linalg.norm(bb - kk) / np.linalg.norm(bb) < 0.05


# ---- vein network --------------------------------------------------------------

def test_vein_network_counts_orientation_and_containment():
    g = ph.Grid((240, 240, 204), (1.0,) * 3)
    lab = ph.build_labels(g)
    vp = vn.VeinParams(n_per_hemi=40, n_collector_per_hemi=3)
    net = vn.generate(lab == ph.L.WM, ph.ventricle_mask(g), g.axes, 1.0, vp, np.random.default_rng(5),
                      brain=np.isin(lab, ph.INTRACRANIAL))
    T = net.table()
    kinds = np.array(T["kind"])
    assert (kinds == "dmv").sum() == 80 and (kinds == "collector").sum() >= 4
    d = np.array(T["diam_mean_mm"])[kinds == "dmv"]
    assert d.min() >= 0.1 - 1e-9 and d.max() <= 0.5 + 1e-9
    assert np.median(np.array(T["angle_b0_deg"])[kinds == "dmv"]) > 60          # mostly perpendicular to B0
    brain = np.isin(lab, ph.INTRACRANIAL)
    o = np.array([a[0] for a in g.axes])
    allp = np.concatenate(net.points)
    idx = np.rint(allp - o).astype(int)
    assert brain[tuple(idx.T)].mean() > 0.99
    # save / load round trip
    net2 = vn.VeinNetwork.from_npz(_tmp_npz(net))
    assert all(np.array_equal(a, b) for a, b in zip(net.points, net2.points))


def _tmp_npz(net, _d={}):
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "v.npz"
    net.to_npz(p)
    return p


# ---- seeds / noise -------------------------------------------------------------

def test_noise_is_deterministic_and_keyed():
    cfg = config.toy()
    a = seeds.unit_noise(cfg, 1, 2.0, 0, 3, (4, 5, 6))
    assert np.array_equal(a, seeds.unit_noise(cfg, 1, 2.0, 0, 3, (4, 5, 6)))
    assert not np.array_equal(a, seeds.unit_noise(cfg, 1, 2.0, 1, 3, (4, 5, 6)))     # replicate
    assert not np.array_equal(a, seeds.unit_noise(cfg, 1, 2.0, 0, 4, (4, 5, 6)))     # echo
    assert not np.array_equal(a, seeds.unit_noise(cfg, 3, 2.0, 0, 3, (4, 5, 6)))     # phantom seed key
    assert seeds.sha256(a) == seeds.sha256(a.copy())


# ---- P3 -------------------------------------------------------------------------

def _have_rc2():
    from qsmsim.exp1.inputs import check_rc2
    return check_rc2()


@pytest.mark.skipif(not _have_rc2(), reason="QSM Challenge 2.0 files not present (qsmsim-fetch-rc2 --p3)")
def test_rc2_3t_tissue_values():
    d = rc2_3t.build(master_mm=1.0)
    seg, mask = d["seg1"], d["mask1"]
    wm = np.median(d["r2s"][(seg == rc2_3t.WM) & mask])
    gp = np.median(d["r2s"][(seg == rc2_3t.LABELS["globus_pallidus"]) & mask])
    assert 10 < wm < 30 and gp > 1.5 * wm
    assert np.count_nonzero(d["chi"][~d["mask"]]) == 0                              # no background sources
    assert abs(np.median(d["m0"][(seg == rc2_3t.WM) & mask]) - 0.70) < 0.01
    vent = rc2_3t.lateral_ventricles(seg, mask)
    assert 10e3 < vent.sum() < 60e3                                                 # 10-60 ml


# ---- a tiny prepared phantom: SNR definition, evaluator, job graph ---------------

@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    cfg = config.toy()
    cfg.name = "mini"
    cfg.master_mm, cfg.resolutions = 2.0, [4.0]
    cfg.veins = vn.VeinParams(n_per_hemi=8, n_collector_per_hemi=1)
    cfg.only_phantoms = ["P1"]
    out = tmp_path_factory.mktemp("exp1mini")
    prepare.prepare(cfg, "P1", out, log=lambda m: None)
    return cfg, out


def test_first_echo_snr_definition(mini):
    cfg, out = mini
    prep = jobs.Prep(cfg, out, "P1", 4.0)
    S, sha, sigma = jobs.noisy_signal(cfg, prep, 5.4, 20, 0)
    S0 = np.asarray(prep.S0(5.4))
    wm = prep.rois["white_matter"] & prep.truth["mask"]
    noise = (S - S0)[..., 0]
    snr = np.abs(S0[..., 0])[wm].mean() / (noise.real.std())
    assert abs(snr / 20 - 1) < 0.02
    # echo noise variance is constant over echoes (SNR falls with T2* decay)
    sd = [(S - S0)[..., e].real.std() for e in range(S.shape[-1])]
    assert np.ptp(sd) / np.mean(sd) < 0.02
    # regeneration reproduces the source hash
    assert jobs.noisy_signal(cfg, prep, 5.4, 20, 0)[1] == sha


def test_evaluator_identity_and_effective_fwhm(mini):
    cfg, out = mini
    prep = jobs.Prep(cfg, out, "P1", 4.0)
    ev = Evaluator(prep.truth, prep.truth["mask"], prep.rois, None, prep.net)
    m, per = ev(prep.truth["chi"])
    assert m["rmse"] < 1e-6 and m["region"]["brain"]["eff_fwhm_mm"] == 0.0
    if "veins" in m and m["veins"]["n_cross_sections"]:
        assert abs(m["veins"]["moment_ratio_median"] - 1) < 1e-6
    mask = prep.truth["mask"]
    blurred = ndimage.gaussian_filter(np.where(mask, prep.truth["chi"], 0), 6.0 / 2.3548 / 4.0)  # tissue only
    m, _ = ev(blurred)
    assert abs(m["region"]["brain"]["eff_fwhm_mm"] - 5.89) < 0.6        # nearest grid sigma to 6 mm FWHM


def test_evaluator_partial_volume_metrics(mini):
    """v2 tissue fractions -> pure / boundary classes, cortical ribbon, deep-GM edge bins, veins by diameter/voxel."""
    cfg, out = mini
    prep = jobs.Prep(cfg, out, "P1", 4.0)
    assert prep.fractions is None                                     # v1 prep: no fractions, no "pv" block
    lab = prep.truth["labels"]
    frac = {k: ndimage.uniform_filter(np.isin(lab, v).astype(np.float32), 3) for k, v in ph.ROIS.items()}
    frac["intracranial"] = np.isin(lab, ph.INTRACRANIAL).astype(np.float32)
    frac["dmv"] = np.zeros(lab.shape, np.float32)
    mask = prep.truth["mask"]
    ev = Evaluator(prep.truth, mask, prep.rois, None, prep.net, fractions=frac)
    m, _ = ev(prep.truth["chi"])
    pv = m["pv"]
    assert pv["voxels"]["pure"] + pv["voxels"]["boundary"] == int(mask.sum()) and pv["voxels"]["boundary"] > 0
    assert pv["region"]["brain"]["boundary"]["rms_ppb"] < 1e-6 and pv["ribbon"]
    assert all(abs(b["bias_ppb"]) < 1e-6 for b in pv["ribbon"].values())
    assert pv["deep_gm"] and all(abs(d["edge"]["bias_ppb"]) < 1e-6 for d in pv["deep_gm"].values())
    off = np.where(mask, prep.truth["chi"] + 0.01 * (frac["cortical_gm"] > 0.5), 0).astype(np.float32)
    pv2 = ev(off)[0]["pv"]
    assert pv2["ribbon"]["gm90-100"]["bias_ppb"] > pv2["ribbon"]["gm0-10"]["bias_ppb"]   # bias follows the GM fraction
    assert Evaluator(prep.truth, mask, prep.rois, None, prep.net)(prep.truth["chi"])[0].keys() == m.keys() - {"pv"}


def test_lam_scheme_grid_and_edge_rule():
    from qsmsim.exp1 import lamgrid as LG
    sc = LG.scheme_v2()
    g = LG.grid(sc, "medi", 3e-4)
    assert len(g) == 17 and 3e-4 in g and g[0] == 3e-7 and g[-1] == 3e-3          # 1e-3 .. 10 x, quarter decades
    assert LG.grid(sc, "tkd", None) == [round(0.05 + 0.025 * i, 4) for i in range(23)]
    lim = LG.limits(sc, "medi", 1e-3)
    # monotone metric: the optimum walks to the cap two decades below the range, two steps at a time, on the lattice
    rows, e = LG.sweep(lambda vs: [dict(value=v, metrics=dict(rmse=v)) for v in vs], sc, "medi", 1e-3,
                       LG.grid(sc, "medi", 1e-3), lim, ["rmse"])
    assert e["extended_low"] == 8 and e["capped_low"] and rows[0]["value"] == 1e-8 and e["at_low"]["rmse"]
    # interior optimum: nothing extended; the vein AUC (highest) can extend on its own
    f = lambda vs: [dict(value=v, metrics=dict(rmse=(np.log10(v) + 4.5) ** 2, veins=dict(auc_core_vs_shell=v)))
                    for v in vs]
    _, e = LG.sweep(f, sc, "medi", 1e-3, LG.grid(sc, "medi", 1e-3), lim, ["rmse"])
    assert e["extended_low"] == e["extended_high"] == 0 and not e["capped_high"]
    _, e = LG.sweep(f, sc, "medi", 1e-3, LG.grid(sc, "medi", 1e-3), lim, ["rmse", "vein_auc"])
    assert e["extended_high"] == 8 and e["capped_high"] and e["optima"]["rmse"] == 3.162e-5
    # a cap that lands on a rounded lattice value (0.01 / 10^0.5 -> 0.003162) is still reached
    toy = dict(step=10 ** 0.5, factors={"cfl2": [1.0, 10.0]}, edge_steps=1, edge_cap_decades=0.5)
    _, e = LG.sweep(lambda vs: [dict(value=v, metrics=dict(rmse=v)) for v in vs], toy, "cfl2", 0.01, [0.01, 0.1],
                    (0.01 / 10 ** 0.5 * (1 - 1e-3), 1.0), ["rmse"])
    assert e["extended_low"] == 1 and e["range"][0] == 0.003162 and e["capped_low"]
    c = config.with_lam_scheme(config.production())
    assert c.full_grids["medi"][0] == 1e-5 and c.full_grids["tkd"] == LG.grid(sc, "tkd", None)
    assert config.Config.from_json(c.to_json()).lam_scheme == c.lam_scheme
    from qsmsim.exp1 import resweep
    r = resweep.make_config(config.with_lam_scheme(config.production()), "/tmp/a/rs", "/tmp/a/prod")
    assert resweep.grid(r, "tv", 1e-3) == LG.grid(sc, "tv", 1e-3)


def test_stage1_edge_rule_runs(mini, tmp_path):
    """Stage-1 tuning under a (small) λ scheme records the edge rule per method."""
    import shutil
    cfg, out = mini
    c = config.Config.from_json(cfg.to_json())
    c.inversions, c.tfi = ["tkd", "medi"], False
    c.full_grids = {"tkd": [0.15, 0.2], "medi": [1e-4, 1e-3]}
    c.lam_scheme = dict(step=10 ** 0.5, factors={"medi": [0.1, 1.0]}, tkd=[0.15, 0.2, 0.05], tkd_limits=[0.1, 0.3],
                        edge_steps=1, edge_cap_decades=0.5, criteria=["rmse"])
    o = tmp_path / "run"
    shutil.copytree(out / "prep", o / "prep")
    shutil.copytree(out / "cache", o / "cache")
    r = jobs.run_tune(c, o, "P1", 4.0, 20, 1, log=lambda m: None)
    for inv in ("tkd", "medi"):
        e = r["methods"][inv]["edge"]
        vals = [x["value"] for x in r["methods"][inv]["rows"]]
        assert e["n_values"] == len(vals) >= 2 and e["range"] == [vals[0], vals[-1]]
        assert r["methods"][inv]["best"] == e["optima"]["rmse"]
        assert not (e["at_low"]["rmse"] or e["at_high"]["rmse"]) or e["capped_low"] or e["capped_high"]
    assert 0.1 - 1e-9 <= min(x["value"] for x in r["methods"]["tkd"]["rows"])


def test_job_graph_dependencies():
    cfg = config.production()
    jobs_ = build_jobs(cfg)
    kinds = {}
    for j in jobs_:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    assert kinds == {"prep": 3, "tune1": 27, "tune2": 27 * 9, "run": 3 * 3 * 3 * 2 * 3}
    ids = {j.id for j in jobs_}
    assert all(d in ids for j in jobs_ for d in j.deps)
    run = next(j for j in jobs_ if j.kind == "run" and j.args["dte"] == 2.4)
    assert len(run.deps) == 1 + 5                       # prep + one tuning job per echo count


def test_resweep_job_graph_and_grid(tmp_path):
    from qsmsim.exp1 import resweep
    cfg = resweep.make_config(config.production(), tmp_path / "rs", tmp_path / "production")
    assert cfg.resweep["src"] == "../production"
    js = build_jobs(cfg)
    kinds = {}
    for j in js:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    assert kinds == {"cache": 3, "resweep": 3 * 3 * 3 * 2 * 3}
    ids = {j.id for j in js}
    assert all(d in ids for j in js for d in j.deps)
    assert all(j.deps == [f"cache:{j.args['ph']}"] for j in js if j.kind == "resweep")
    # the grid contains the stage-1 value and the stage-2 half-decade value exactly (rep-0 cross-check)
    g = resweep.grid(cfg, "medi", 1e-3)
    assert g[0] == 1e-3 and g[1] == float(f"{1e-3 / 10 ** 0.5:.4g}") and min(g) == 1e-5
    assert resweep.grid(cfg, "tkd", 0.19) == cfg.full_grids["tkd"]
    # the config survives the JSON round trip (frozen config of the resweep output)
    assert config.Config.from_json(cfg.to_json()).to_json() == cfg.to_json()
    # a TFI-only pass on chosen replicates
    t = resweep.make_config(config.production(), tmp_path / "rst", tmp_path / "production", methods=[],
                            tfi_reps=[0, 1], reps=[0, 1])
    assert t.resweep["methods"] == [] and t.active_reps() == [0, 1]
    assert sum(j.kind == "resweep" for j in build_jobs(t)) == 3 * 3 * 3 * 2 * 2
    with pytest.raises(ValueError):
        resweep.make_config(config.production(), tmp_path / "x", tmp_path / "production", methods=["tfi"])
    with pytest.raises(ValueError):
        resweep.make_config(config.production(), tmp_path / "x", tmp_path / "production", methods=[])


def test_runner_skips_done_jobs(tmp_path):
    cfg = config.toy()
    r = Runner(cfg, tmp_path, 1, 1.0)
    first = next(iter(r.jobs))
    (tmp_path / "state" / f"{first.replace(':', '__')}.done").write_text(json.dumps({}))
    assert r.status_of(first) == "done"
    assert sum(v.get("done", 0) for v in r.summary().values()) == 1


def test_runner_gpu_lanes(tmp_path):
    """GPU lanes: reconstruction jobs are GPU jobs (sized for their evaluation pool), at most gpu_lanes run at once,
    CPU jobs run alongside, and each job gets its backend in the environment."""
    cfg = config.toy()
    cfg.gpu_lanes, cfg.eval_workers = 1, 3
    js = build_jobs(cfg)
    cpu = {j.id: j for j in build_jobs(config.toy())}
    for j in js:
        assert j.gpu == (j.kind in ("tune1", "tune2", "run"))            # v1 prep stays on the CPU
        if j.gpu:
            assert j.threads == 4 and abs(j.mem_gb - cpu[j.id].mem_gb * 2.5) < 1e-9
    v2 = config.v2_toy()
    v2.gpu_lanes = 1
    assert all(j.gpu for j in build_jobs(v2) if j.kind == "prep")

    class FakeProc:
        def __init__(self, r, job):
            self.r, self.job, self.n = r, job, 0
        def poll(self):
            self.n += 1
            if self.n < 3:
                return None
            (self.r.state / f"{self.r._safe(self.job.id)}.done").write_text("{}")
            return 0

    r = Runner(cfg, tmp_path, 8, 1e6, poll_s=0)
    seen, envs = [], {}
    def launch(job):
        running = [r.jobs[k] for k in r.running]
        seen.append((sum(j.gpu for j in running) + job.gpu, sum(not j.gpu for j in running) + (not job.gpu)))
        r.running[job.id] = (FakeProc(r, job), open(tmp_path / "x.log", "w"), 0.0)
    r._launch = launch
    assert r.run(log=lambda m: None) == 0
    assert max(g for g, _ in seen) == 1 and max(c for _, c in seen) >= 2       # one GPU job at a time, preps in parallel
    import os
    os.environ.pop("QSMSIM_BACKEND", None)
    r2 = Runner(cfg, tmp_path / "b", 8, 1e6)
    calls = {}
    import subprocess
    orig = subprocess.Popen
    subprocess.Popen = lambda *a, **k: calls.setdefault(a[0][4], k["env"]) or FakeProc(r2, r2.jobs[a[0][4]])
    try:
        g = next(j for j in r2.jobs.values() if j.gpu)
        c = next(j for j in r2.jobs.values() if not j.gpu)
        r2._launch(g)
        r2._launch(c)
    finally:
        subprocess.Popen = orig
    assert calls[g.id]["QSMSIM_BACKEND"] == "gpu" and calls[g.id]["QSMSIM_EVAL_WORKERS"] == "3"
    assert calls[g.id]["QSMSIM_THREADS"] == "1" and calls[c.id]["QSMSIM_BACKEND"] == "cpu"
    assert calls[c.id]["QSMSIM_EVAL_WORKERS"] == "0"


# ---- Experiment 2 (7 T anchored to a 3T reference run) ----------------------------

def _fake_ref(path, cfg):
    path.mkdir(parents=True)
    (path / "config.json").write_text(cfg.to_json())
    for p in cfg.active_phantoms():
        (path / "prep" / p).mkdir(parents=True)
        (path / "prep" / p / "prep.json").write_text("{}")
    return path


def test_exp2_config_and_job_graph(tmp_path):
    from qsmsim.exp1 import exp2
    ref = _fake_ref(tmp_path / "production", config.production())
    cfg = exp2.make_config(config.exp2_7t(), tmp_path / "exp2", ref)
    assert cfg.exp2["ref"] == "../production" and cfg.b0_tesla == 7.0
    assert cfg.snrs == [5, 10, 15.5, 20, 40, 62, 100, 200, 310]
    assert exp2.level(cfg, 15.5) == (5.0, 3.1) and exp2.level(cfg, 200) == (100.0, 2.0)
    assert abs(cfg.veins.r2s - 105.0) < 1e-9 and abs(cfg.r2s_scale - 7 / 3) < 1e-12 and cfg.shim_b0_tesla == 3.0
    assert cfg.protocols() == [(5.4, n) for n in (5, 7, 9, 11, 13)]
    js = build_jobs(cfg)
    kinds = {}
    for j in js:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    assert kinds == {"prep": 2, "tune1": 2 * 3 * 9, "resweep": 2 * 3 * 9 * 3}
    ids = {j.id for j in js}
    assert all(d in ids for j in js for d in j.deps)
    assert config.Config.from_json(cfg.to_json()).to_json() == cfg.to_json()
    # options: 7T with the high-performance gradient (2.4 ms), TFI on chosen replicates
    h = exp2.make_config(config.exp2_7t(), tmp_path / "h", ref, hpg=True, tfi_reps=[0])
    assert sum(j.kind == "resweep" for j in build_jobs(h)) == 2 * kinds["resweep"]
    assert h.tfi and h.resweep["tfi_reps"] == [0] and (2.4, 11) in h.protocols()
    # a reference that is not the same phantom set-up is refused
    other = config.production()
    other.root_seed += 1
    with pytest.raises(ValueError, match="root_seed"):
        exp2.make_config(config.exp2_7t(), tmp_path / "y", _fake_ref(tmp_path / "other", other))
    with pytest.raises(ValueError):
        exp2.make_config(config.production(), tmp_path / "z", ref)


def test_exp2_same_phantom_and_anchored_noise(mini, tmp_path):
    from qsmsim.exp1 import exp2
    cfg3, ref = mini
    (ref / "config.json").write_text(cfg3.to_json())
    c = config._exp2(config.Config.from_json(cfg3.to_json()), s3t=(5, 20), gains=(1.0, 3.1))
    c.name, c.only_phantoms = "mini7t", ["P1"]
    out = tmp_path / "x7"
    c = exp2.make_config(c, out, ref)
    info = prepare.prepare(c, "P1", out, log=lambda m: None)     # raises if the truth volumes differ
    assert info["exp2"]["truth_match"] and abs(info["exp2"]["dmv_r2s"] - 105.0) < 1e-9
    v3, v7 = vn.VeinNetwork.from_npz(ref / "prep/P1/veins.npz"), vn.VeinNetwork.from_npz(out / "prep/P1/veins.npz")
    assert all(np.array_equal(a, b) for a, b in zip(v3.points, v7.points)) and v7.params.r2s > v3.params.r2s
    p3, p7 = jobs.Prep(cfg3, ref, "P1", 4.0), jobs.Prep(c, out, "P1", 4.0)
    assert p7.wm_first_echo(5.4) < p3.wm_first_echo(5.4)         # faster decay by TE1 at 7 T
    s3 = jobs.noisy_signal(cfg3, p3, 5.4, 20, 1)[2]
    assert jobs.noisy_signal(c, p7, 5.4, 20, 1)[2] == s3          # k = 1: exactly the 3T sigma
    assert abs(jobs.noisy_signal(c, p7, 5.4, 62, 1)[2] * 3.1 / s3 - 1) < 1e-6
    # same unit noise as the paired 3T replicate (common random numbers)
    S7, _, sg7 = jobs.noisy_signal(c, p7, 5.4, 20, 1)
    S3, _, _ = jobs.noisy_signal(cfg3, p3, 5.4, 20, 1)
    assert np.allclose(S7 - np.asarray(p7.S0(5.4)), S3 - np.asarray(p3.S0(5.4)), atol=1e-5)
    # the phase evolves 7/3 as fast: the fitted noiseless field equals the 3T one in ppm
    ds7 = jobs.Dataset(c, p7, np.asarray(p7.S0(5.4)), 5.4, 5)
    ds3 = jobs.Dataset(cfg3, p3, np.asarray(p3.S0(5.4)), 5.4, 5)
    m = p3.truth["mask"] & (p3.truth["dmv_dist"] > 3)
    assert np.median(np.abs(ds7.fit.field_ppm - ds3.fit.field_ppm)[m]) < 2e-4
    assert ds3.field_metrics["voxels_cycle_error"] == 0 and "global_cycles_removed" not in ds3.field_metrics


def test_exp2_resolution_the_reference_lacks(mini, tmp_path):
    """Tier B's 0.8 mm arm in miniature: a 7 T resolution with no 3T run takes the noise of its noise_res anchor, is
    checked through the vein network, and has no 3T partner; without an anchor it is refused."""
    from qsmsim.exp1 import exp2
    cfg3, ref = mini
    (ref / "config.json").write_text(cfg3.to_json())
    c = config._exp2(config.Config.from_json(cfg3.to_json()), s3t=(5, 20), gains=(1.0, 3.1))
    c.name, c.only_phantoms, c.resolutions = "mini7t_fine", ["P1"], [6.0]
    with pytest.raises(ValueError, match="resolution 6 mm"):
        exp2.make_config(config.Config.from_json(c.to_json()), tmp_path / "no_anchor", ref)
    c.exp2["noise_res"] = {"6": 4.0}
    out = tmp_path / "x7f"
    c = exp2.make_config(c, out, ref)
    info = prepare.prepare(c, "P1", out, log=lambda m: None)
    assert info["exp2"]["checked"] == [] and info["exp2"]["unanchored"] == ["res6.0/truth"]
    assert info["exp2"]["veins_table_match"]
    p3, p7 = jobs.Prep(cfg3, ref, "P1", 4.0), jobs.Prep(c, out, "P1", 6.0)
    assert p7.snr_reference(5.4) == p3.wm_first_echo(5.4)              # option (a): sigma of the anchor cell
    assert jobs.noisy_signal(c, p7, 5.4, 20, 0)[2] == jobs.noisy_signal(cfg3, p3, 5.4, 20, 0)[2]
    assert exp2.partner(c, out, "P1", 6.0, 20, 5.4, 5, 0) is None
    # a different vein network is caught
    vt = out / "prep" / "P1" / "veins_table.json"
    vt.write_text(vt.read_text().replace("1", "2", 1))
    with pytest.raises(RuntimeError, match="vein network"):
        exp2.check_truth(c, out, "P1", {"res6.0/truth": "x"})


def test_tierb_presets_and_widening(tmp_path, capsys):
    """Tier B: the 0.8 mm arm and the 3 T SNR sweep validate against exp1_v2; frozen runs grow in place by
    replicates / TFI replicates only."""
    from qsmsim.exp1 import exp2 as X
    from qsmsim.exp1 import __main__ as M
    e1 = config.exp1_v2()
    ref = _fake_ref(tmp_path / "exp1_v2", e1)
    c8 = X.make_config(config.exp2_v2_08(), tmp_path / "exp2_v2_08", ref, tfi_reps=[0, 1, 2], reps=range(6))
    c8.check_v2()
    assert c8.active_resolutions() == [0.8] and X.anchor_res(c8, 0.8) == 1.0 and c8.active_reps() == list(range(6))
    assert c8.snrs == config.exp2_v2().snrs
    js = build_jobs(c8)
    assert sum(j.kind == "resweep" for j in js) == 2 * 9 * 6 and sum(j.kind == "tune1" for j in js) == 2 * 9
    s3 = X.make_config(config.snr3t_v2(), tmp_path / "snr3t_v2", ref, reps=range(6))
    assert s3.b0_tesla == 3.0 and s3.snrs == config.exp2_v2().snrs and not s3.tfi
    assert s3.protocols() == [(5.4, 5), (5.4, 13), (2.4, 11)] and s3.active_resolutions() == [1.0]
    assert X.level(s3, 62) == (20.0, 3.1) and X.level(s3, 5) == (20.0, 0.25)
    # widening rules
    have = X.make_config(config.exp2_v2(), tmp_path / "e2", ref, tfi_reps=[0, 1, 2], reps=[0, 1, 2])
    grow = lambda reps=None, tfi=None, **kw: X.make_config(config.exp2_v2(), tmp_path / "e2", ref,
                                                          tfi_reps=tfi or [0, 1, 2], reps=reps or [0, 1, 2])
    assert M.widens(have, grow(reps=range(6)))
    assert M.widens(have, grow(reps=range(6), tfi=range(6)))
    assert not M.widens(grow(reps=range(6)), have)                      # never drops replicates
    assert not M.widens(have, have)
    other = grow(reps=range(6))
    other.snrs = [5.0]
    assert not M.widens(have, other)
    # through the command line: exp2 --reps widens the frozen config in place; a narrowing is refused
    out = tmp_path / "e2run"
    argv = ["exp2", "--ref", str(ref), "--out", str(out), "--config", "exp2_v2", "--plan"]
    for extra in (["--reps", "0,1,2", "--tfi-reps", "0,1,2"], ["--reps", "0,1,2,3,4,5", "--tfi-reps", "0,1,2"]):
        assert M.main(argv + extra) == 0
    frozen = config.Config.from_json((out / "config.json").read_text())
    assert frozen.active_reps() == list(range(6)) and frozen.resweep["tfi_reps"] == [0, 1, 2]
    assert "widening in place" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="only a widening"):
        M.main(argv + ["--reps", "0,1", "--tfi-reps", "0,1,2"])


def test_control_config_and_job_graph(tmp_path):
    """Phase 4 control: the v1 run's design on forward model v2; 7 T needs the 3T control; jobs per phantom / dataset."""
    from qsmsim.exp1 import control as CT, exp2
    prod = _fake_ref(tmp_path / "production", config.production())
    c3 = CT.make_config(config.production(), tmp_path / "control_3t", prod)
    assert c3.forward == "v2" and c3.control["src"] == "../production" and c3.control["lam"] == "runs"
    assert not c3.tuning and c3.resweep is None and c3.snrs == [5, 20, 100] and c3.protocols() == config.production().protocols()
    js = build_jobs(c3)
    kinds = {}
    for j in js:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    assert c3.active_phantoms() == ["P1", "P3"]                          # P2 left out (plan 2.4)
    assert kinds == {"cache": 2, "prep": 2, "cmp": 2 * 3, "ctl": 2 * 3 * 3 * 2 * 3}
    ids = {j.id for j in js}
    assert all(d in ids for j in js for d in j.deps)
    assert config.Config.from_json(c3.to_json()).to_json() == c3.to_json()
    c3.gpu_lanes = 1
    assert {j.kind for j in build_jobs(c3) if j.gpu} == {"prep", "ctl"}
    # 7 T: against Experiment 2, λ from its stage 1, v2 truth checked against the 3T control
    (tmp_path / "control_3t").mkdir()
    (tmp_path / "control_3t" / "config.json").write_text(c3.to_json())
    e2 = exp2.make_config(config.exp2_7t(), tmp_path / "exp2", prod)
    (tmp_path / "exp2").mkdir()
    (tmp_path / "exp2" / "config.json").write_text(e2.to_json())
    with pytest.raises(ValueError, match="v2-ref"):
        CT.make_config(e2, tmp_path / "control_7t", tmp_path / "exp2")
    c7 = CT.make_config(e2, tmp_path / "control_7t", tmp_path / "exp2", v2_ref=tmp_path / "control_3t")
    assert c7.control["lam"] == "stage1" and c7.exp2["ref"] == "../control_3t" and c7.b0_tesla == 7.0
    assert sum(j.kind == "ctl" for j in build_jobs(c7)) == 2 * 3 * 9 * 3
    v1 = CT.v1_config(c7, tmp_path / "control_7t")
    assert v1.forward == "v1" and v1.exp2["ref"] == str((tmp_path / "production").resolve())
    with pytest.raises(ValueError, match="control run"):
        CT.make_config(e2, tmp_path / "x", tmp_path / "exp2", v2_ref=prod)
    with pytest.raises(ValueError, match="v1"):
        CT.make_config(c3, tmp_path / "y", tmp_path / "control_3t")


def test_control_classes_and_change_stats():
    """Disjoint classes covering the mask; the change of the error map is located where the arms differ."""
    from types import SimpleNamespace
    from qsmsim.exp1 import control as CT
    n = 40
    z = np.indices((n, n, n))
    ball = ((z - n / 2 + 0.5) ** 2).sum(0) < 17 ** 2
    gm = np.where(ball & (z[0] < 20), 1.0, 0.0).astype(np.float32)
    gm[19] = 0.5 * ball[19]                                              # one plane of mixed voxels
    wm = (ball * 1.0 - gm).astype(np.float32)
    dist = np.where(z[1] < 4, 1.0, 50.0).astype(np.float32)             # "veins" along one face
    prep2 = SimpleNamespace(fractions={"cortical_gm": gm, "white_matter": wm, "intracranial": ball.astype(np.float32)},
                            truth={"dmv_dist": dist}, res=1.0)
    cm = CT.class_maps(prep2)
    mask = ndimage.binary_erosion(ball, iterations=2)
    C = CT.classes(cm, mask)
    total = sum(m.astype(int) for m in C.values())
    assert np.array_equal(total, mask.astype(int))                       # disjoint and covering
    assert C["boundary"][18:21].any() and not C["agree"][18:21].any()   # the mixed plane and its neighbours
    assert not C["agree"][:, :4].any() and C["agree"].sum() > 1000
    # a labelled vein whose partial volumes are all on the pure side (94 % core, 9 % neighbours) is vein, not agree
    veins = np.zeros_like(wm)
    veins[28:30, 20:22, 10:30] = 0.94
    veins[27, 20:22, 10:30] = veins[30, 20:22, 10:30] = veins[28:30, 19, 10:30] = veins[28:30, 22, 10:30] = 0.09
    veins *= ball
    prep2.fractions.update(veins=veins, white_matter=(wm - veins).astype(np.float32))
    Cv = CT.classes(CT.class_maps(prep2), mask)
    assert not C["vein"][28:30, 20:22, 15:25].any() and C["agree"][28:30, 20:22, 15:25].all()
    assert Cv["vein"][27:31, 19:23, 15:25].all() and not Cv["agree"][27:31, 19:23, 15:25].any()
    assert Cv["agree"][24:26, 20:22, 15:25].all()                       # two voxels off is agree again
    assert np.array_equal(sum(m.astype(int) for m in Cv.values()), mask.astype(int))
    rng = np.random.default_rng(0)
    t1 = rng.normal(size=ball.shape).astype(np.float32)
    t2 = t1.copy()
    x1 = t1 + 0.01 * rng.normal(size=ball.shape).astype(np.float32)
    x2 = x1.copy()
    x2[C["boundary"]] += 0.5                                             # v2 changes only at the boundary
    ch = CT.change_stats(x1, x2, t1, t2, mask, C)
    assert ch["boundary"]["error_change_share"] > 0.6 and ch["boundary"]["voxel_share"] < 0.2
    assert abs(sum(ch[k]["voxel_share"] for k in C if k in ch) - 1) < 1e-9


def test_control_pass_table_relative_tolerance():
    """A deterministic change passes within 5 % of the v1 agree error even when the v1 CI is tiny (high SNR), fails
    above it, and a change hidden by the noise passes on the CI."""
    from qsmsim.exp1 import control as CT

    def unit(chain, v1_rms, d_rms, spread, d_bias=0.0):
        return [{"phantom": "P1", "res": 1.0, "snr": 100, "dte": 5.4, "n": 5, "chain": chain, "rep": i,
                 "v1.region.agree.rms_ppb": v1_rms + spread * (i - 1), "d.region.agree.rms_ppb": d_rms,
                 "v1.region.agree.bias_ppb": spread * (i - 1), "d.region.agree.bias_ppb": d_bias} for i in range(3)]
    rows = (unit("small", 10.0, 0.4, 0.001) + unit("large", 10.0, 0.6, 0.001)
            + unit("noisy", 10.0, 0.6, 1.0) + unit("biased", 10.0, 0.1, 0.001, d_bias=-0.7))
    P = {p["chain"]: p for p in CT.pass_table(rows)}
    assert P["small"]["pass"] is True and P["large"]["pass"] is False
    assert P["noisy"]["pass"] is True                      # v1 CI half-width 4.3 x 1.0 / sqrt(3) > 0.5 tolerance
    assert P["biased"]["pass"] is False                    # the bias is held to the same 0.5 ppb (5 % of 10 ppb)
    assert P["small"]["tol"] == pytest.approx(0.5)


# ---- phase 5: sourced relaxation and the v2 rerun presets (Experiments_v2.md) ------------------------------------

def test_sourced_tissue_tables():
    v1 = ph.tissue_lut("v1", 3.0)
    assert all(np.array_equal(v1[k], np.float32(v)) for k, v in ph.TISSUES.items())
    np.testing.assert_allclose(ph.r2s_table("v1", 7.0, 7 / 3), v1[:, 2] * np.float32(7 / 3), rtol=1e-6)
    s3, s7 = ph.tissue_lut("sourced", 3.0), ph.tissue_lut("sourced", 7.0)
    t3, t7 = s3[:, 2], s7[:, 2]
    L = ph.L
    for lab, ms3, ms7 in ((L.GM, 66, 33), (L.WM, 53, 27), (L.VEIN, 21, 7), (L.DURAL, 21, 7), (L.GP, 29.5, 11),
                          (L.CAUDATE, 47.5, 24), (L.PUTAMEN, 39.5, 20), (L.THALAMUS, 49, 26.5), (L.SN, 29.5, 14.5),
                          (L.RN, 28, 17), (L.CSF, 2000, 900)):
        assert t3[lab] == pytest.approx(1000 / ms3, rel=1e-6) and t7[lab] == pytest.approx(1000 / ms7, rel=1e-6)
    assert t3[L.DENTATE] == pytest.approx(1000 / 31, rel=1e-6)
    assert t7[L.DENTATE] == pytest.approx(1000 / 31 * ph.DEEP_GM_RATIO, rel=1e-6)    # no 7 T source
    for lab, f in ((L.CALC, 7 / 3), (L.BLEED, 7 / 3), (L.SOFT, 2.0), (L.BONE, 2.0)):
        assert t3[lab] == v1[lab, 2] and t7[lab] == pytest.approx(v1[lab, 2] * f, rel=1e-6)
    for lab, chi in ((L.CAUDATE, 0.039), (L.PUTAMEN, 0.052), (L.GP, 0.1075), (L.THALAMUS, 0.003), (L.SN, 0.1255),
                     (L.RN, 0.1005), (L.DENTATE, 0.093)):
        assert s3[lab, 0] == s7[lab, 0] == np.float32(chi)
    other = [k for k in ph.TISSUES if k not in ph.SOURCED_CHI]
    assert np.array_equal(s3[other, :2], v1[other, :2]) and np.array_equal(s7[:, :2], s3[:, :2])  # chi, M0 field-free
    assert t3[L.AIR] == t3[L.SINUS] == 0 and ph.blood_r2s(7.0) == pytest.approx(1000 / 7, rel=1e-6)
    with pytest.raises(ValueError):
        ph.tissue_lut("sourced", 3.0, 7 / 3)
    with pytest.raises(ValueError):
        ph.tissue_lut("sourced", 1.5)
    # build_phantom with the sourced table: chi from the table plus texture, R2* from the table
    P0 = ph.build_phantom(voxel_mm=4.0, texture_ppm=0.0)
    P1 = ph.build_phantom(voxel_mm=4.0, texture_ppm=0.0, lut=s3)
    gp = P0.labels == L.GP
    assert gp.any() and np.all(P1.chi[gp] == np.float32(0.1075)) and np.all(P0.chi[gp] == np.float32(0.18))
    assert np.array_equal(P1.r2s, t3[P1.labels])


def test_v2_rerun_presets(tmp_path):
    from qsmsim.exp1 import exp2 as X
    e1, e2 = config.exp1_v2(), config.exp2_v2()
    for c, b0, blood in ((e1, 3.0, 1000 / 21), (e2, 7.0, 1000 / 7)):
        c.check_tissues()
        assert (c.tissues, c.b0_tesla, c.r2s_scale, c.forward) == ("sourced", b0, 1.0, "v2")
        assert c.veins.r2s == pytest.approx(blood, rel=1e-6)
        assert c.active_phantoms() == ["P1", "P3"] and c.active_reps() == [0, 1, 2] and c.reps == 6
        assert c.lam_scheme and c.eval_workers == 12
        c.check_v2()
        assert config.Config.from_json(c.to_json()).to_json() == c.to_json()
    assert prepare.rc2_r2s(e1) == dict(r2s_scale=0.5, r2s_max=175.0)
    assert prepare.rc2_r2s(e2) == dict(r2s_scale=1.0, r2s_max=350.0)
    assert prepare.rc2_r2s(config.production()) == {}                   # v1 untouched: 3/7, clip 150 (build's defaults)
    assert e1.snrs == [5, 20, 100] and len(e1.protocols()) == 10
    assert len(e2.snrs) == 9 and e2.protocols() == [(5.4, n) for n in (5, 7, 9, 11, 13)]
    # exp2_v2 against an exp1_v2 reference passes the compatibility check; a v1 reference does not (tissues differ)
    ref = tmp_path / "ref"
    for p in ("P1", "P3"):
        (ref / "prep" / p).mkdir(parents=True)
        (ref / "prep" / p / "prep.json").write_text("{}")
    X.check_compatible(e2, e1, ref)
    v1 = config.production()
    v1.forward = "v2"
    with pytest.raises(ValueError, match="tissues"):
        X.check_compatible(e2, v1, ref)
    bad = config.exp1_v2()
    bad.veins = vn.VeinParams(**{**bad.veins.__dict__, "r2s": 45.0})
    with pytest.raises(ValueError, match="DMV blood"):
        bad.check_tissues()


def test_analytic_source_takes_sourced_table():
    from qsmsim import forward2 as f2
    t = ph.tissue_lut("sourced", 7.0)
    src = f2.AnalyticSource(None, np.zeros((2, 2, 2), np.float32), 1.0, lut=t)
    np.testing.assert_array_equal(src.lut, t)
    np.testing.assert_array_equal(f2.AnalyticSource(None, np.zeros((2, 2, 2), np.float32)).lut, ph.tissue_lut())
    with pytest.raises(ValueError):
        f2.AnalyticSource(None, np.zeros((2, 2, 2), np.float32), 7 / 3, lut=t)


def test_blood3t_presets(tmp_path):
    """The blood ablation: four 3 T runs against exp1_v2 that differ only in the DMV blood R2*; the sourced one has
    exp1_v2's physics; labels 20 / 100 take exp1_v2's noise, 10000 is effectively noiseless."""
    from dataclasses import asdict
    from qsmsim.exp1 import exp2 as X
    from qsmsim.phantom import blood_r2s
    e1 = config.exp1_v2()
    ref = _fake_ref(tmp_path / "exp1_v2", e1)
    want = {"sourced": blood_r2s(3.0), "nodecay": 0.0, "fast": blood_r2s(7.0), "void": 2000.0}
    cs = {v: X.make_config(config.load(f"blood3t_{v}"), tmp_path / f"b_{v}", ref) for v in want}
    strip = lambda c: {k: v for k, v in asdict(c).items() if k not in ("name", "veins", "exp2")}
    for v, c in cs.items():
        c.check_tissues()
        assert abs(c.veins.r2s - want[v]) < 1e-3 and c.exp2["blood_variant"] == v
        assert {k: x for k, x in asdict(c.veins).items() if k != "r2s"} == \
            {k: x for k, x in asdict(e1.veins).items() if k != "r2s"}
        assert strip(c) == strip(cs["sourced"])                 # nothing else differs between the variants
        assert c.b0_tesla == 3.0 and c.active_resolutions() == [1.0] and c.active_reps() == [0, 1, 2]
        assert c.protocols() == [(5.4, 5), (5.4, 13), (2.4, 5), (2.4, 11)] and not c.tfi
        assert X.level(c, 20) == (20.0, 1.0) and X.level(c, 100) == (20.0, 5.0) and X.level(c, 10000) == (20.0, 500.0)
        js = build_jobs(c)
        assert sum(j.kind == "resweep" for j in js) == 2 * 3 * 2 * 3 and sum(j.kind == "tune1" for j in js) == 2 * 3
    assert cs["sourced"].veins == e1.veins and cs["sourced"].tissues == e1.tissues
    bad = config.load("blood3t_void")
    bad.exp2.pop("dmv_blood_r2s")
    with pytest.raises(ValueError, match="DMV blood"):
        bad.check_tissues()
