"""Experiment 1 configuration (see Experiment_1.md, updated plan v2).

Presets:
  production - the approved design (0.5 mm master grid, 1.0/1.5/2.0 mm, ...)
  pilot      - production restricted to P1, 2.0 mm, SNR 20, replicate 0
  toy        - every code path on a coarse grid (1 mm master, 2/3/4 mm), minutes
  realsize   - one production-configuration 1.5 mm job from a 0.75 mm master
               (memory/time check on a small workstation)
  exp2_7t    - Experiment 2 (Experiment_2.md): the production phantoms at 7 T, noise anchored to a
               3T reference run (`exp2` command); exp2_toy is its toy-scale counterpart
  v2_toy     - the toy run on forward model v2 (GPU_Plan.md phase 2: fine tiles over the mesh)
  g6_toy     - v2_toy on P1 and P3 at SNR 20 with a small λ scheme: the GPU runner acceptance (G.6, phase 3)
  exp1_v2, exp2_v2 - the phase 5 reruns (Experiments_v2.md): forward model v2, the sourced tissue table (R2* per field, deep-GM χ), the v2
               λ scheme, P1 and P3, replicates 0-2 (tier A); exp1_v2_toy / exp2_v2_toy rehearse them
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

from ..veins import VeinParams

FULL_GRIDS = {                              # stage 1 (wide): defaults' grids, extended at the high end for SNR 5
    "tkd": [0.1, 0.15, 0.19, 0.25, 0.3],
    "cfl2": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "tv": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "medi": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "tfi": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}
# Experiment 2 (7 T): up to ~7x less field noise than 3T, so two more half-decades at the light end
EXP2_GRIDS = {
    "tkd": [0.1, 0.15, 0.19, 0.25, 0.3],
    "cfl2": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "tv": [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "medi": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "tfi": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}
EXP2_FACTORS = [10 ** 0.5, 1.0, 10 ** -0.5, 0.1, 10 ** -1.5, 0.01]
# P3 under tissues = "sourced": the RC2 7 T R2* fit x scale (3 T: 0.5, the sourced GM / WM 7 T -> 3 T ratio; v1 used
# 3/7), clipped (v1: 150 at 3 T, 350 at 7 T)
RC2_SOURCED_SCALE = {3.0: 0.5, 7.0: 1.0}
RC2_SOURCED_MAX = {3.0: 175.0, 7.0: 350.0}
PARAM_KEY = {"tkd": "thr", "cfl2": "lam", "tv": "alpha", "medi": "alpha", "tfi": "lam"}


@dataclass
class PhantomSpec:
    name: str                   # P1 | P2 | P3
    kind: str                   # "analytic" | "rc2"
    fov_mm: tuple
    texture_ppm: float = 0.005
    texture_sigma_mm: float = 3.0
    background_field: bool = True
    seed_key: int = 1           # geometry seed key; P1 and P2 share it (identical veins)


@dataclass
class Config:
    name: str = "production"
    phantoms: list = field(default_factory=lambda: [
        PhantomSpec("P1", "analytic", (240.0, 240.0, 204.0)),
        PhantomSpec("P2", "analytic", (240.0, 240.0, 204.0), texture_ppm=0.02, texture_sigma_mm=10.0),
        PhantomSpec("P3", "rc2", (168.0, 216.0, 216.0), background_field=False, seed_key=3)])
    master_mm: float = 0.5
    fine_factor: int = 5                    # sub-samples per master voxel edge (0.1 mm)
    tile_core_mm: float = 16.0
    tile_halo_mm: float = 4.0
    near_mm: tuple = (1.0, 2.5)
    mesh_pad: float = 2.0
    # forward model (GPU_Plan.md, phase 2). "v1": the two-scale model of Experiments 1-2 (tissue on the master
    # grid, sub-samples only near veins; default). "v2": every voxel sampled at fine_mm in tiles over the master
    # mesh (qsmsim/forward2.py); tiles of v2_core_mm + v2_halo_mm, aligned to v2_align_mm (a multiple of every
    # acquisition size), veins refined vein_refine x per axis in refine_core_mm sub-tiles. Halo 12 mm = 50 % of the core
    # per side (agreed 2026-10-20): the tiling error is truncation of the fine chi beyond the halo (G.2: brain max
    # 1.4 / 0.5 sigma_min at 4 / 6 mm halos)
    forward: str = "v1"
    fine_mm: float = 0.1
    v2_core_mm: float = 24.0
    v2_halo_mm: float = 12.0
    v2_align_mm: float = 12.0
    vein_refine: int = 2
    refine_core_mm: float = 6.0
    refine_halo_mm: float = 1.0
    tissue_sub: bool = True                 # False: tissue constant per master cell (acceptance G.3, v2 -> v1)
    resolutions: list = field(default_factory=lambda: [1.0, 1.5, 2.0])
    resample: str = "block"                 # "block" | "kspace"
    b0_tesla: float = 3.0
    r2s_scale: float = 1.0                  # bulk tissue R2* relative to the 3T model (P1 TISSUES; P3 vs its 3T scaling)
    # tissue table (P1): "v1" = TISSUES, R2* x r2s_scale (Experiments 1-2 v1); "sourced" = phantom.tissue_lut at
    # b0_tesla (the v2 reruns, Experiments_v2.md 2.1-2.3: per-field R2*, deep-GM chi): r2s_scale 1, DMV blood =
    # phantom.blood_r2s(b0), P3 = its 7 T R2* fit x RC2_SOURCED_SCALE[b0], clipped at RC2_SOURCED_MAX[b0] (P3's chi is
    # RC2's own)
    tissues: str = "v1"
    shim_b0_tesla: float | None = None      # field at which shim_hz is defined (None: b0_tesla, i.e. fixed in Hz)
    b0_dir: tuple = (0.0, 0.0, 1.0)
    phase_sign: float = 1.0
    te1_ms: float = 4.0
    dtes_ms: list = field(default_factory=lambda: [5.4, 2.4])
    n_echo_sim: int = 13
    echo_counts: dict = field(default_factory=lambda: {"5.4": [5, 7, 9, 11, 13], "2.4": [5, 7, 9, 11, 13]})
    snrs: list = field(default_factory=lambda: [5, 20, 100])      # first-echo SNR in WM
    bandwidth_penalty: bool = False          # sigma * sqrt(dte_ref / dte) if True (not used in Exp. 1)
    bandwidth_ref_dte_ms: float = 5.4
    reps: int = 3
    root_seed: int = 20261001
    transceive_amp: float = 3.141592653589793
    shim_hz: dict = field(default_factory=lambda: {"x": 6.0, "y": -10.0, "z": 12.0, "z2": 15.0, "xy": 5.0})
    mask_erode_mm: float = 3.0
    veins: VeinParams = field(default_factory=VeinParams)
    # reconstruction
    backgrounds: list = field(default_factory=lambda: ["oracle", "vsharp", "pdf"])   # P1/P2; P3 uses "direct"
    inversions: list = field(default_factory=lambda: ["tkd", "cfl2", "tv", "medi"])
    tfi: bool = True
    recon_params: dict = field(default_factory=dict)                  # overrides of pipeline.DEFAULT_PARAMS
    p3_tfi: dict = field(default_factory=lambda: dict(pad_factor=1.1, outer=10, cg_iters=40))
    # tuning (two-stage, Experiment_1.md section 7)
    tuning: bool = True
    tune_ref: tuple = (5.4, 5)
    full_grids: dict = field(default_factory=lambda: copy.deepcopy(FULL_GRIDS))
    narrow_step: float = 10 ** 0.5           # stage-2 grid: best/step, best, best*step (TKD: full grid)
    sweep_tfi_outer: int = 15
    keep_sweep_maps: bool = True
    # outputs
    save_float: str = "float32"
    # scheduling (overridden by environment variables in the container)
    workers: int = 24
    mem_gb: float = 88.0
    prep_threads: int = 8
    job_threads: int = 1
    # GPU (GPU_Plan.md phase 3): gpu_lanes > 0 runs every reconstruction job (and a v2 preparation) on the GPU, at most
    # gpu_lanes at a time (one process saturates an A5000; two only hide host-side gaps), each with a pool of
    # eval_workers CPU processes scoring its chi maps; host memory per GPU job = the CPU estimate x
    # (1 + eval_mem_factor x eval_workers); gpu_mem_gb caps CuPy's pool, shared by the lanes (0 = no cap).
    # One backend per experiment (Q7): with lanes, no reconstruction runs on the CPU.
    gpu_lanes: int = 0
    eval_workers: int = 5
    eval_mem_factor: float = 0.5
    gpu_mem_gb: float = 0.0
    # filters (pilot / partial runs)
    only_phantoms: list | None = None
    only_resolutions: list | None = None
    only_snrs: list | None = None
    only_reps: list | None = None
    # extended λ sweep on a finished run (qsmsim/exp1/resweep.py); None for every other preset
    resweep: dict | None = None
    # λ scheme with the automatic edge rule (qsmsim/exp1/lamgrid.py, GPU_Plan.md 2.5): None = the v1 grids (the
    # resweep factors, stage 1 without extension); with_lam_scheme() sets it and the matching stage-1 grids
    lam_scheme: dict | None = None
    # Experiment 2: a run at another field with noise anchored to a 3T reference run (exp2 command);
    # dict(ref=<3T run dir, relative to out>, ref_b0, levels={snr label: [s3T, k]}); None otherwise
    exp2: dict | None = None
    # phase 4 control run (GPU_Plan.md 2.4, qsmsim/exp1/control.py): forward model v2 against the finished v1 run `src`
    # (relative to out) on the same seeds and noise; dict(src, lam, sweep, maps_rep); None otherwise
    control: dict | None = None
    # remove whole aliasing cycles of global field offset after the field fit, as a scanner's reference frequency
    # does (always on for Experiment 2; qsmsim.explore turns it on at every field)
    remove_global_cycles: bool = False

    # ---- derived -----------------------------------------------------------
    def phantom(self, name):
        return next(p for p in self.phantoms if p.name == name)

    def active_phantoms(self):
        return [p.name for p in self.phantoms if not self.only_phantoms or p.name in self.only_phantoms]

    def active_resolutions(self):
        return [r for r in self.resolutions if not self.only_resolutions or r in self.only_resolutions]

    def active_snrs(self):
        return [s for s in self.snrs if not self.only_snrs or s in self.only_snrs]

    def active_reps(self):
        return [r for r in range(self.reps) if self.only_reps is None or r in self.only_reps]

    def counts(self, dte):
        return list(self.echo_counts[_k(dte)])

    def protocols(self):
        return [(d, n) for d in self.dtes_ms for n in self.counts(d)]

    def factor(self, res):
        f = res / self.master_mm
        if abs(f - round(f)) > 1e-6:
            raise ValueError(f"resolution {res} is not a multiple of the master spacing {self.master_mm}")
        return int(round(f))

    def fine_per_master(self):
        """v2: fine points per master cell edge (odd, so a fine point sits at each master cell centre)."""
        return _ratio(self.master_mm, self.fine_mm, "master spacing", "fine spacing", odd=True)

    def fine_per_voxel(self, res):
        """v2: fine points per acquisition voxel edge."""
        return _ratio(res, self.fine_mm, f"resolution {res}", "fine spacing")

    def check_v2(self):
        """Raise if the v2 geometry is inconsistent (tiles, acquisition sizes, FOVs, refinement)."""
        self.fine_per_master()
        for r in self.active_resolutions():
            self.fine_per_voxel(r)
            _ratio(self.v2_align_mm, r, "v2_align_mm", f"resolution {r}")
        _ratio(self.v2_align_mm, self.master_mm, "v2_align_mm", "master spacing")
        _ratio(self.v2_core_mm, self.v2_align_mm, "v2_core_mm", "v2_align_mm")
        _ratio(self.v2_halo_mm, self.master_mm, "v2_halo_mm", "master spacing")
        if self.vein_refine > 1:
            _ratio(self.v2_core_mm, self.refine_core_mm, "v2_core_mm", "refine_core_mm")
            _ratio(self.refine_halo_mm, self.fine_mm, "refine_halo_mm", "fine spacing")
            if self.refine_halo_mm > self.v2_halo_mm:
                raise ValueError("refine_halo_mm must not exceed v2_halo_mm")
        # the tile FFTs run at their own size (forward2.periodic_field): it must be FFT-friendly
        from scipy.fft import next_fast_len
        sizes = {"tile (v2_core_mm + 2 v2_halo_mm)": round((self.v2_core_mm + 2 * self.v2_halo_mm) / self.fine_mm)}
        if self.vein_refine > 1:
            sizes["vein sub-tile (refine_core_mm + 2 refine_halo_mm)"] = self.vein_refine * round(
                (self.refine_core_mm + 2 * self.refine_halo_mm) / self.fine_mm)
        for what, n in sizes.items():
            if next_fast_len(n, real=False) != n:
                raise ValueError(f"{what}: {n} points per axis is not an FFT-friendly size")
        for p in self.phantoms:
            if p.name in self.active_phantoms():
                for f in p.fov_mm:
                    _ratio(f, self.v2_align_mm, f"{p.name} FOV {f}", "v2_align_mm")

    def check_tissues(self):
        """Raise if the tissue settings are inconsistent: under "sourced" no r2s_scale, a sourced field, and the
        DMV blood R2* equal to the table's venous blood."""
        from .. import phantom as ph
        ph.tissue_lut(self.tissues, self.b0_tesla, self.r2s_scale)     # unknown table / field, or scale != 1
        # the blood ablation (blood3t_*) sets the DMV blood on purpose and records it in exp2["dmv_blood_r2s"]
        if self.tissues != "sourced":
            return
        want = (self.exp2 or {}).get("dmv_blood_r2s")
        want = ph.blood_r2s(self.b0_tesla) if want is None else float(want)
        if abs(self.veins.r2s - want) > 1e-3:
            raise ValueError(f"DMV blood R2* {self.veins.r2s:g} 1/s is not the expected venous blood value "
                             f"{want:.4g} at {self.b0_tesla:g} T")

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=list)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        d["phantoms"] = [PhantomSpec(**{**p, "fov_mm": tuple(p["fov_mm"])}) for p in d["phantoms"]]
        d["veins"] = VeinParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["veins"].items()})
        for k in ("near_mm", "b0_dir", "tune_ref"):
            d[k] = tuple(d[k])
        return cls(**d)


def _ratio(a, b, na, nb, odd=False):
    q = a / b
    if abs(q - round(q)) > 1e-6 or round(q) < 1 or (odd and round(q) % 2 == 0):
        raise ValueError(f"{na} ({a}) must be {'an odd' if odd else 'a'} multiple of {nb} ({b})")
    return int(round(q))


def _k(dte):
    return f"{float(dte):g}"


def production():
    return Config()


def pilot():
    c = Config(name="pilot")
    c.only_phantoms, c.only_resolutions, c.only_snrs, c.only_reps = ["P1"], [2.0], [20], [0]
    return c


def toy():
    """Coarse but complete: every phantom, both spacings, all SNRs, all chains, tuning."""
    c = Config(name="toy")
    c.master_mm = 1.0
    c.resolutions = [2.0, 3.0, 4.0]
    c.fine_factor = 5                       # 0.2 mm sub-samples
    c.mesh_pad = 1.5
    c.echo_counts = {"5.4": [5, 9], "2.4": [5, 11]}
    c.reps = 2
    c.veins = VeinParams(n_per_hemi=10, n_collector_per_hemi=2)
    c.recon_params = {"tv": {"maxiter": 30}, "medi": {"maxiter": 30}, "pdf": {"maxiter": 30},
                      "tfi": {"outer": 3, "cg_iters": 15}}
    c.p3_tfi = dict(pad_factor=1.1, outer=3, cg_iters=15)
    c.full_grids = {"tkd": [0.15, 0.25], "cfl2": [1e-2, 1e-1], "tv": [1e-4, 1e-3], "medi": [1e-4, 1e-3],
                    "tfi": [1e-3, 1e-2]}
    c.sweep_tfi_outer = 3
    c.workers, c.mem_gb, c.prep_threads = 4, 3.5, 4
    return c


def realsize():
    """One production-configuration reconstruction job at 1.5 mm (P1, SNR 20, rep 0,
    5.4 ms) with a 0.75 mm master grid so the preparation fits a small machine."""
    c = Config(name="realsize")
    c.master_mm = 0.75
    c.resolutions = [1.5]
    c.mesh_pad = 1.5
    c.only_phantoms, c.only_snrs, c.only_reps = ["P1"], [20], [0]
    c.dtes_ms = [5.4]
    c.echo_counts = {"5.4": [5, 13]}          # the two ends of the echo sweep (time per dataset ~ independent of n)
    c.tuning = False
    c.workers, c.mem_gb, c.prep_threads = 1, 5.0, 8
    return c


def _exp2(c, s3t=(5, 20, 100), gains=(1.0, 2.0, 3.1), b0=7.0, ref_b0=3.0):
    """Turn an Experiment 1 config into its Experiment 2 counterpart (Experiment_2.md, plan v2).
    Noise levels are labeled by snr = s3T * k: sigma = (3T reference WM first-echo magnitude) / snr."""
    scale = b0 / ref_b0
    c.b0_tesla = b0
    c.r2s_scale = scale                                  # all tissue R2* linear in B0
    c.veins = VeinParams(**{**asdict(c.veins), "r2s": c.veins.r2s * scale})   # DMV blood 45 -> 105 1/s
    c.shim_b0_tesla = ref_b0                             # shim residual identical in ppm
    c.only_phantoms = ["P1", "P3"]
    c.dtes_ms = [5.4]                                    # standard gradients: the Experiment 1 standard protocol
    c.echo_counts = {k: v for k, v in c.echo_counts.items() if k == "5.4"}
    levels = {}
    for s in s3t:
        for k in gains:
            lab = float(f"{s * k:.6g}")
            if lab in levels:
                raise ValueError(f"noise label {lab} is not unique")
            levels[lab] = [float(s), float(k)]
    c.snrs = sorted(levels)
    c.inversions = ["tkd", "cfl2", "tv", "medi"]
    c.tfi = False
    c.full_grids = copy.deepcopy(EXP2_GRIDS)
    c.keep_sweep_maps = False                            # tables only; maps are regenerable from seeds
    c.exp2 = dict(ref=None, ref_b0=ref_b0, s3t=[float(s) for s in s3t], gains=[float(k) for k in gains],
                  levels={f"{k:g}": v for k, v in sorted(levels.items())})
    c.resweep = dict(src=".", factors=[float(f) for f in EXP2_FACTORS], methods=["medi", "tv", "cfl2", "tkd"],
                     tfi_reps=[], keep_maps=False)
    return c


def exp2_7t():
    c = _exp2(Config(name="exp2_7t"))
    c.mem_gb = 128.0
    return c


def exp2_toy():
    """Experiment 2 on the toy grid (needs a toy 3T reference run)."""
    c = _exp2(toy(), s3t=(5, 100), gains=(1.0, 3.1))
    c.name = "exp2_toy"
    c.full_grids = {"tkd": [0.15, 0.25], "cfl2": [1e-3, 1e-2, 1e-1], "tv": [1e-5, 1e-4, 1e-3],
                    "medi": [1e-5, 1e-4, 1e-3], "tfi": [1e-3, 1e-2]}
    c.resweep["factors"] = [10 ** 0.5, 1.0, 0.1]
    return c


def v2_toy():
    """The toy run on forward model v2: 0.2 mm fine points in 24 mm tiles over the 1 mm mesh, veins at 0.1 mm."""
    c = toy()
    c.name = "v2_toy"
    c.forward = "v2"
    c.fine_mm = 0.2
    return c


def quarter_decade_grids(grids, tkd):
    """Stage-1 grids at quarter-decade steps over each method's range (decade-aligned: 1, 1.778, 3.162, 5.623 x 10^k);
    TKD = the scheme's thresholds."""
    import math
    out = {}
    for inv, g in grids.items():
        if inv == "tkd":
            lo, hi, st = tkd
            out[inv] = [float(f"{lo + i * st:.4g}") for i in range(int(round((hi - lo) / st)) + 1)]
            continue
        k0, k1 = math.floor(4 * math.log10(min(g)) + 1e-6), math.ceil(4 * math.log10(max(g)) - 1e-6)
        out[inv] = [float(f"{10 ** (k / 4):.4g}") for k in range(k0, k1 + 1)]
    return out


def with_lam_scheme(c, scheme=None):
    """The v2 λ scheme (GPU_Plan.md 2.5): quarter-decade stage-1 grids over the preset's ranges with the edge rule,
    and sweeps around each cell's stage-1 λ over the scheme's factor ranges (TKD 0.05-0.6 in 0.025)."""
    from .lamgrid import scheme_v2
    c.lam_scheme = scheme or scheme_v2()
    c.full_grids = quarter_decade_grids(c.full_grids, c.lam_scheme["tkd"])
    return c


def g6_toy():
    """Acceptance G.6 (GPU_Plan.md phase 3): v2_toy on P1 and P3 at SNR 20, two replicates, with a small λ scheme so
    the edge rule runs too; minutes per run on the GPU."""
    c = v2_toy()
    c.name = "g6_toy"
    c.only_phantoms, c.only_snrs = ["P1", "P3"], [20]
    c.lam_scheme = dict(step=10 ** 0.5, factors={k: [0.1, 10 ** 0.5] for k in ("cfl2", "tv", "medi", "tfi")},
                        tkd=[0.15, 0.25, 0.05], tkd_limits=[0.1, 0.35], edge_steps=1, edge_cap_decades=0.5,
                        criteria=["rmse", "vein_auc"])
    return c


def sourced(c):
    """The sourced tissue table (Experiments_v2.md 2.1-2.3) at c.b0_tesla: no R2* scaling, DMV blood sourced."""
    from ..phantom import blood_r2s
    c.tissues, c.r2s_scale = "sourced", 1.0
    c.veins = VeinParams(**{**asdict(c.veins), "r2s": blood_r2s(c.b0_tesla)})
    c.check_tissues()
    return c


V2_REPS = 6                     # the design's replicates (Experiments_v2.md 3); tier A runs ABSTRACT_REPS of them
ABSTRACT_REPS = [0, 1, 2]       # tier A (the abstract, 2026-10-24): replicates 0-2, paired with v1's unit noise


def _v2(c):
    """The v2 rerun settings shared by both experiments: forward model v2 on P1 and P3, the v2 λ scheme, six seeded
    replicates of which tier A runs 0-2, an evaluation pool of 12 per GPU lane (the lanes come from the host,
    docker/run_v2.sh)."""
    c.forward = "v2"
    c.only_phantoms = ["P1", "P3"]
    c.reps, c.only_reps = V2_REPS, list(ABSTRACT_REPS)
    c.eval_workers = 12
    c.mem_gb = 120.0
    return c


def exp1_v2():
    """Experiment 1 v2 (Experiments_v2.md 3.1): 3 T, 1.0 / 1.5 / 2.0 mm, SNR 5 / 20 / 100 locked per label at every
    resolution, 10 protocols, two-stage tuning on the v2 λ scheme, production chains; the λ sweeps (TFI included)
    are a resweep of this run (docker/run_v2.sh)."""
    return _v2(with_lam_scheme(sourced(Config(name="exp1_v2"))))


def exp2_v2():
    """Experiment 2 v2 (Experiments_v2.md 3.2): exp1_v2 at 7 T with the sourced 7 T relaxation; noise anchored to
    the exp1_v2 run (sigma = its WM first-echo magnitude / (s3T x k), the same label at every resolution); stage 1 on
    Experiment 2's extended light-end ranges, then the λ sweeps (TFI via --tfi-reps)."""
    c = _exp2(Config(name="exp2_v2"))
    return _v2(with_lam_scheme(sourced(c)))


def exp2_v2_08():
    """Tier B's 7 T 0.8 mm arm (Experiments_v2.md 4, option (a)): exp2_v2 at 0.8 mm only, each noise label with the
    sigma of the 3T reference's 1.0 mm cell (sigma_0.8 = sigma_1.0: same label, not physically scaled). There is no
    3T 0.8 mm run: the truth is checked through the vein network instead of the 3T hashes, and the datasets have no
    3T partner. Its own run (exp2 --ref exp1_v2 --config exp2_v2_08), all six replicates."""
    c = exp2_v2()
    c.name = "exp2_v2_08"
    c.resolutions = [0.8]
    c.exp2["noise_res"] = {"0.8": 1.0}
    return c


# The 3 T SNR sweep (the plateau, 2026-10-29): the nine 7 T noise labels of exp2_v2 (s3T 5 / 20 / 100 x k 1 / 2 / 3.1),
# reached at 3 T as one anchor (s3T = 20) times a pure noise multiplier, so both fields share one noise axis.
SNR3T_GAINS = (0.25, 0.5, 0.775, 1.0, 2.0, 3.1, 5.0, 10.0, 15.5)   # labels 5 10 15.5 20 40 62 100 200 310


def snr3t_v2():
    """3 T noise sweep for the SNR plateau of small-vein CNR: Experiment 2's noise anchoring with B0 = the reference's
    3 T, so k is a pure noise multiplier: sigma = (exp1_v2 WM first-echo magnitude) / (20 k), the labels of exp2_v2.
    Same seeds and truth as exp1_v2 (checked); labels 5 / 20 / 100 reproduce exp1_v2's noise exactly. 1 mm only;
    3T standard 5 and 13 @ 5.4 ms and 3T HPG 11 @ 2.4 ms; no TFI (two thirds of P1's GPU time); stage 1 on 5 @ 5.4,
    then the v2 λ sweeps. Run as exp2: --ref exp1_v2 --config snr3t_v2 (tier B runs replicates 0-5)."""
    c = _exp2(Config(name="snr3t_v2"), s3t=(20,), gains=SNR3T_GAINS, b0=3.0, ref_b0=3.0)
    c = _v2(with_lam_scheme(sourced(c)))
    c.dtes_ms = [5.4, 2.4]
    c.echo_counts = {"5.4": [5, 13], "2.4": [11]}
    c.only_resolutions = [1.0]
    return c


def exp1_v2_toy():
    """exp1_v2 on the toy grid (v2_toy geometry, small scheme of g6_toy): the rehearsal of docker/run_v2.sh."""
    c = g6_toy()
    c.name = "exp1_v2_toy"
    c.only_snrs = None
    c.snrs = [5, 100]
    sourced(c)
    c.reps, c.only_reps = 3, [0, 1]
    return c


def exp2_v2_toy():
    """exp2_v2 on the toy grid (needs an exp1_v2_toy reference run)."""
    c = _exp2(v2_toy(), s3t=(5, 100), gains=(1.0, 3.1))
    c.name = "exp2_v2_toy"
    c.full_grids = {"tkd": [0.15, 0.25], "cfl2": [1e-3, 1e-2, 1e-1], "tv": [1e-5, 1e-4, 1e-3],
                    "medi": [1e-5, 1e-4, 1e-3], "tfi": [1e-3, 1e-2]}
    c.lam_scheme = g6_toy().lam_scheme
    sourced(c)
    c.reps, c.only_reps = 3, [0, 1]
    return c


def exp2_v2_toy_fine():
    """exp2_v2_08 on the toy grid: a resolution the toy reference lacks (1.2 mm, the 0.8 mm stand-in), its noise
    anchored on the toy reference's finest cell (2.0 mm)."""
    c = exp2_v2_toy()
    c.name = "exp2_v2_toy_fine"
    c.resolutions = [1.2]
    c.exp2["noise_res"] = {"1.2": 2.0}
    return c


def snr3t_v2_toy():
    """snr3t_v2 on the toy grid (needs an exp1_v2_toy reference run): anchor s3T = 5, three gains, 2 mm only."""
    c = _exp2(v2_toy(), s3t=(5,), gains=(0.5, 1.0, 4.0), b0=3.0, ref_b0=3.0)
    c.name = "snr3t_v2_toy"
    c.full_grids = exp2_v2_toy().full_grids
    c.lam_scheme = g6_toy().lam_scheme
    sourced(c)
    c.dtes_ms = [5.4, 2.4]
    c.echo_counts = {"5.4": [5, 9], "2.4": [11]}
    c.only_resolutions = [2.0]
    c.reps, c.only_reps = 3, [0, 1]
    return c


# The blood ablation (2026-11-07): what drives the 3T HPG gain for small veins - a better field map (more echoes in
# the same window), or more vein signal surviving the decay? The same 3T exp2-against-exp1_v2 run four times, only
# the DMV network's blood R2* changed (DMVs + collectors; L.VEIN capsules, dural sinuses and P3's own veins keep the
# sourced values). Noise labels s3T 20 x k: 20 and 100 reproduce exp1_v2's noise exactly; 10000 is effectively
# noiseless (sigma = WM / 1e4, same unit noise), so any HPG-standard difference there is signal-model bias.
BLOOD3T_R2S = {
    "sourced": None,            # 47.6 1/s (T2* 21 ms): the v2 value - this variant reproduces exp1_v2's 1 mm signals
    "nodecay": 0.0,             # blood never decays: no advantage from sampling early
    "fast": 7.0,                # the sourced 7 T blood, 142.9 1/s (T2* 7 ms), at 3 T: decay alone, without 7 T phase
    "void": 2000.0,             # T2* 0.5 ms: 0.03 % of the blood signal left at TE1 = 4 ms; extravascular signal only
}
BLOOD3T_GAINS = (1.0, 5.0, 500.0)                   # labels 20, 100, 10000


def _blood3t_r2s(variant):
    from ..phantom import blood_r2s
    v = BLOOD3T_R2S[variant]
    return blood_r2s(3.0) if v is None else blood_r2s(7.0) if v == 7.0 else float(v)


def _set_dmv_blood(c, variant):
    r2s = _blood3t_r2s(variant)
    c.veins = VeinParams(**{**asdict(c.veins), "r2s": r2s})
    c.exp2["dmv_blood_r2s"] = r2s
    c.exp2["blood_variant"] = variant
    c.check_tissues()
    return c


def _blood3t(variant):
    """3 T, 1 mm, P1 + P3, replicates 0-2; 5 and 13 @ 5.4 ms, 5 and 11 @ 2.4 ms (HPG = 11 @ 2.4); no TFI; stage 1 on
    5 @ 5.4 per noise label, then the v2 λ sweeps. Run as exp2: --ref exp1_v2 --config blood3t_<variant>."""
    c = _exp2(Config(name=f"blood3t_{variant}"), s3t=(20,), gains=BLOOD3T_GAINS, b0=3.0, ref_b0=3.0)
    c = _v2(with_lam_scheme(sourced(c)))
    c.dtes_ms = [5.4, 2.4]
    c.echo_counts = {"5.4": [5, 13], "2.4": [5, 11]}
    c.only_resolutions = [1.0]
    return _set_dmv_blood(c, variant)


def _blood3t_toy(variant):
    """The blood ablation on the toy grid (needs an exp1_v2_toy reference run): 2 mm, s3T 5 x (1, 500)."""
    c = _exp2(v2_toy(), s3t=(5,), gains=(1.0, 500.0), b0=3.0, ref_b0=3.0)
    c.name = f"blood3t_toy_{variant}"
    c.full_grids = exp2_v2_toy().full_grids
    c.lam_scheme = g6_toy().lam_scheme
    sourced(c)
    c.dtes_ms = [5.4, 2.4]
    c.echo_counts = {"5.4": [5, 9], "2.4": [5, 11]}
    c.only_resolutions = [2.0]
    c.reps, c.only_reps = 3, [0, 1]
    return _set_dmv_blood(c, variant)


def explore():
    """Base geometry for qsmsim-explore: the production phantoms and vein network on the 0.5 mm master grid, forward
    model v1 (voxel sizes that are multiples of 0.5 mm)."""
    c = Config(name="explore")
    c.tuning = False
    return c


PRESETS = {"production": production, "explore": explore, "pilot": pilot, "toy": toy, "realsize": realsize,
           "exp2_7t": exp2_7t, "exp2_toy": exp2_toy, "v2_toy": v2_toy, "g6_toy": g6_toy,
           "exp1_v2": exp1_v2, "exp2_v2": exp2_v2, "exp1_v2_toy": exp1_v2_toy, "exp2_v2_toy": exp2_v2_toy,
           "exp2_v2_08": exp2_v2_08, "snr3t_v2": snr3t_v2,
           "exp2_v2_toy_fine": exp2_v2_toy_fine, "snr3t_v2_toy": snr3t_v2_toy}
for _v in BLOOD3T_R2S:
    PRESETS[f"blood3t_{_v}"] = (lambda v: lambda: _blood3t(v))(_v)
    PRESETS[f"blood3t_toy_{_v}"] = (lambda v: lambda: _blood3t_toy(v))(_v)


def load(name_or_path):
    if name_or_path in PRESETS:
        return PRESETS[name_or_path]()
    with open(name_or_path) as fh:
        return Config.from_json(fh.read())
