"""Explore QSM performance as acquisition and reconstruction parameters change.

One spec file (TOML or JSON) names a grid: phantom, field strength, voxel size, echo trains (spacing x count),
first echo time, SNR, noise replicates, background removal, dipole inversion and regularization strength, plus
named variants of any other model setting (vein sizes, field direction, forward model, tissue table, ...).
Every combination is simulated with the Experiment 1/2 machinery: the sub-voxel deep-medullary-vein phantom, the
multi-echo GRE forward model, complex Gaussian noise, the field fit and the reconstruction chain. Each chi map is
scored against the truth (whole brain, regions and per-vein metrics).

    qsmsim-explore example > my.toml          # a commented starting spec
    qsmsim-explore run my.toml --out runs/my   # simulate (resumable: finished datasets are kept)
    qsmsim-explore summarize runs/my           # best regularization per condition -> summary.csv
    qsmsim-explore plot runs/my --x snr --y vein_auc --group protocol --facet b0_tesla
    qsmsim-explore plot runs/my --results --x lam --y rmse --group chain   # every regularization value

Outputs under --out:
    spec.toml / spec.json     the spec as run
    prep/<key>/               phantom truth and noiseless signals (shared by every SNR, replicate and method)
    rows/<dataset>.json       one dataset's metrics for every chain and regularization value
    results.csv               every (dataset, chain, value) row
    summary.csv               per condition and chain: the value chosen by each selection metric, metrics
                              averaged over replicates at that value
    maps/<dataset>/*.nii.gz   chi maps (replicate 0) when save_maps = true

The backend is the CPU (numpy/scipy) unless backend = "gpu" or QSMSIM_BACKEND=gpu (needs CuPy and a CUDA device).
The two give the same results to floating-point tolerance, not bit for bit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

EXAMPLE = '''\
# qsmsim-explore spec. Every list is a sweep dimension; the run covers every combination.
name = "quick"

# Simulation geometry and model: a preset from qsmsim.exp1.config.
#   "toy"      1 mm master grid, 20 veins, coarse voxels (2-4 mm): minutes on a laptop
#   "explore"  0.5 mm master grid, full vein network, v1 forward model: 1-2 mm voxels, ~16-32 GB RAM
#   "exp1_v2"  forward model v2 (every voxel sampled at 0.1 mm) with the sourced tissue table: use a GPU
base = "toy"
backend = "cpu"            # "cpu" | "gpu"

phantoms = ["P1"]          # "P1" analytic head with background field; "P3" QSM Challenge 2.0 head (needs the RC2 data)
b0_tesla = [3.0]           # field strength: tissue R2* and vein phase scale with it
resolution_mm = [2.0]      # isotropic voxel; a multiple of the preset's master grid
te1_ms = [4.0]             # first echo time
snr = [10, 40]             # first-echo SNR in white matter (see [noise])
replicates = 1             # independent noise draws per condition

# Echo trains: spacing and number of echoes (up to 13).
protocols = [
  { dte_ms = 5.4, echoes = 5 },
  { dte_ms = 2.4, echoes = 11 },
]

backgrounds = ["oracle"]   # P1 only: "oracle" (true local field), "vsharp", "pdf", "sharp", "lbv". P3 has none.
inversions = ["tv", "medi"]   # "tkd", "cfl2", "tv", "medi", "tfi" (total field inversion, no background step)

# Regularization values per inversion (threshold for TKD). "grid" = the preset's wide grid
# (extended at the light end above 3 T). Or give lists: lambda = { medi = [1e-4, 3e-4], tv = [1e-4] }
lambda = "grid"

# How summary.csv picks a value per condition: rmse, hfen (lowest) or vein_auc, ssim, cnr_0.2-0.3mm (highest), ...
select = ["rmse", "vein_auc"]

save_maps = false          # NIfTI chi maps of replicate 0 for every chain and value

[noise]
# SNR is defined against the white-matter first-echo magnitude. By default each field uses its own; set
# reference_b0 to define every field's noise against one field (as the 3T-vs-7T comparison did), and gain to
# give a field an SNR multiplier (e.g. 7 T coil and field SNR gain over 3 T).
# reference_b0 = 3.0
# gain = { "7" = 2.0 }
bandwidth_penalty = false  # true: sigma x sqrt(5.4 ms / spacing) (shorter spacing = wider readout bandwidth)

[config]
# Any qsmsim.exp1.config.Config field, e.g. fewer veins or iterations:
# veins = { n_per_hemi = 10 }
# recon_params = { medi = { maxiter = 50 } }

# Variants: named sets of [config] overrides, run as one more sweep dimension (column "variant"), e.g. thinner
# veins, a tilted head or the other forward model. Each variant gets its own phantom preparation.
# [[variants]]
# name = "thin"
# config = { veins = { diam_start_mm = [0.1, 0.3] } }
# [[variants]]
# name = "tilt20"
# config = { b0_dir = [0.0, 0.342, 0.940] }
'''

LOWER_BETTER = ("rmse", "hfen", "nrmse", "rms_ppb", "error", "lsf_fwhm")
SCHED_FIELDS = ("name", "workers", "mem_gb", "prep_threads", "job_threads", "gpu_lanes", "eval_workers",
                "eval_mem_factor", "gpu_mem_gb", "keep_sweep_maps", "tuning")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- spec
def load_spec(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        spec = json.loads(text)
    else:
        import tomllib
        spec = tomllib.loads(text)
    return normalize(spec)


def _list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def normalize(spec):
    s = dict(spec)
    s.setdefault("name", "explore")
    s.setdefault("base", "toy")
    s.setdefault("backend", os.environ.get("QSMSIM_BACKEND", "cpu"))
    for k, d in (("phantoms", ["P1"]), ("b0_tesla", [3.0]), ("resolution_mm", [2.0]), ("te1_ms", [4.0]),
                 ("snr", [20]), ("backgrounds", ["oracle"]), ("inversions", ["medi"]),
                 ("select", ["rmse", "vein_auc"])):
        s[k] = _list(s.get(k, d))
    s["b0_tesla"] = [float(b) for b in s["b0_tesla"]]
    s["resolution_mm"] = [float(r) for r in s["resolution_mm"]]
    s["te1_ms"] = [float(t) for t in s["te1_ms"]]
    s["snr"] = [float(x) for x in s["snr"]]
    s["replicates"] = int(s.get("replicates", 1))
    s.setdefault("protocols", [{"dte_ms": 5.4, "echoes": 5}])
    prots = []
    for p in s["protocols"]:
        d, n = float(p["dte_ms"]), int(p["echoes"])
        if not 1 <= n <= 13:
            raise ValueError(f"protocol {p}: echoes must be 1-13")
        prots.append(dict(dte_ms=d, echoes=n, name=p.get("name", f"{n}x{d:g}ms")))
    s["protocols"] = prots
    s.setdefault("lambda", "grid")
    s.setdefault("save_maps", False)
    s.setdefault("noise", {})
    s.setdefault("config", {})
    variants = []
    for v in s.get("variants") or []:
        name = str(v.get("name", ""))
        if not name or not all(ch.isalnum() or ch in "._-" for ch in name):
            raise ValueError(f"variant {v}: needs a name of letters, digits, '.', '_' or '-'")
        if name in (x["name"] for x in variants):
            raise ValueError(f"variant {name}: duplicate name")
        variants.append(dict(name=name, config=dict(v.get("config") or {})))
    s["variants"] = variants
    bad = set(s["inversions"]) - {"tkd", "cfl2", "tv", "medi", "tfi"}
    if bad:
        raise ValueError(f"unknown inversions {sorted(bad)}")
    bad = set(s["backgrounds"]) - {"oracle", "vsharp", "pdf", "sharp", "lbv"}
    if bad:
        raise ValueError(f"unknown backgrounds {sorted(bad)}")
    if s["backend"] not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return s


# --------------------------------------------------------------------------- configs
def _apply_overrides(cfg, over):
    from .exp1.config import PhantomSpec
    from .veins import VeinParams
    for k, v in over.items():
        if not hasattr(cfg, k):
            raise ValueError(f"[config] {k}: not a Config field")
        if k == "veins":
            v = VeinParams(**{**asdict(cfg.veins), **{a: tuple(b) if isinstance(b, list) else b
                                                       for a, b in v.items()}})
        elif k == "phantoms":
            v = [PhantomSpec(**{**p, "fov_mm": tuple(p["fov_mm"])}) for p in v]
        elif isinstance(getattr(cfg, k), tuple):
            v = tuple(v)
        elif isinstance(getattr(cfg, k), dict) and isinstance(v, dict) and k == "recon_params":
            merged = copy.deepcopy(cfg.recon_params)
            for m, p in v.items():
                merged[m] = {**merged.get(m, {}), **p}
            v = merged
        setattr(cfg, k, v)
    return cfg


def _variant_config(spec, variant):
    if not variant:
        return {}
    for v in spec.get("variants") or []:
        if v["name"] == variant:
            return v["config"]
    raise ValueError(f"no variant {variant!r} in the spec")


def field_config(spec, b0, te1, variant=None):
    """The experiment Config for one field strength, first echo time and variant (all protocols, voxel sizes,
    chains). A variant's [config] is applied after the spec's own."""
    from .exp1 import config as C
    from .veins import VeinParams
    cfg = C.load(spec["base"])
    cfg = _apply_overrides(cfg, spec["config"])
    vconf = _variant_config(spec, variant)
    cfg = _apply_overrides(cfg, vconf)
    ref_b0 = float(cfg.b0_tesla)
    cfg.name = f"explore_{spec['name']}"
    cfg.exp2, cfg.resweep, cfg.control, cfg.lam_scheme = None, None, None, cfg.lam_scheme
    cfg.only_phantoms = cfg.only_resolutions = cfg.only_snrs = cfg.only_reps = None
    if cfg.tissues == "sourced":
        cfg.b0_tesla = b0
        C.sourced(cfg)                                   # per-field relaxation + venous blood from the table
    else:                                                # v1 tissue table: R2* linear in B0 (Experiment 2)
        scale = b0 / ref_b0
        cfg.r2s_scale = cfg.r2s_scale * scale
        cfg.veins = VeinParams(**{**asdict(cfg.veins), "r2s": cfg.veins.r2s * scale})
        cfg.b0_tesla = b0
    cfg.shim_b0_tesla = ref_b0 if cfg.shim_b0_tesla is None else cfg.shim_b0_tesla   # shim residual fixed in ppm
    cfg.te1_ms = te1
    cfg.resolutions = list(spec["resolution_mm"])
    cfg.dtes_ms = sorted({p["dte_ms"] for p in spec["protocols"]})
    cfg.echo_counts = {f"{d:g}": sorted({p["echoes"] for p in spec["protocols"] if p["dte_ms"] == d})
                       for d in cfg.dtes_ms}
    cfg.n_echo_sim = max(p["echoes"] for p in spec["protocols"])
    cfg.backgrounds = list(spec["backgrounds"])
    cfg.inversions = [i for i in spec["inversions"] if i != "tfi"]
    cfg.tfi = "tfi" in spec["inversions"]
    cfg.bandwidth_penalty = bool(spec["noise"].get("bandwidth_penalty", False))
    cfg.remove_global_cycles = True
    if b0 > 3.0 + 1e-6 and spec["lambda"] == "grid" and "full_grids" not in {**spec["config"], **vconf}:
        cfg.full_grids = copy.deepcopy(C.EXP2_GRIDS)
    return cfg


def prep_key(cfg, phantom):
    """Hash of everything that changes the phantom truth or the noiseless signals."""
    d = json.loads(cfg.to_json())
    for k in SCHED_FIELDS + ("backgrounds", "inversions", "tfi", "recon_params", "p3_tfi", "full_grids",
                             "narrow_step", "sweep_tfi_outer", "snrs", "reps", "bandwidth_penalty",
                             "bandwidth_ref_dte_ms", "echo_counts", "lam_scheme", "tune_ref", "mask_erode_mm",
                             "remove_global_cycles"):
        d.pop(k, None)
    d["phantoms"] = [p for p in d["phantoms"] if p["name"] == phantom]
    h = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{phantom}_{cfg.b0_tesla:g}T_te{cfg.te1_ms:g}_{h}"


def lam_values(spec, cfg, inv):
    lam = spec["lambda"]
    if isinstance(lam, dict):
        if inv not in lam:
            raise ValueError(f"lambda has no values for {inv}")
        return [float(v) for v in _list(lam[inv])]
    if lam != "grid":
        raise ValueError("lambda must be 'grid' or a table of lists per inversion")
    return [float(v) for v in cfg.full_grids[inv]]


# ------------------------------------------------------------------------------ plan
def datasets(spec):
    """Every (variant, phantom, field, first echo, voxel, protocol, SNR, replicate) to simulate."""
    out = []
    variants = [v["name"] for v in spec.get("variants") or []] or [None]
    for var, ph, b0, te1, res, p, snr, rep in product(variants, spec["phantoms"], spec["b0_tesla"], spec["te1_ms"],
                                                       spec["resolution_mm"], spec["protocols"], spec["snr"],
                                                       range(spec["replicates"])):
        name = (f"{ph}_{b0:g}T_te{te1:g}_res{res:g}_{p['echoes']}x{p['dte_ms']:g}_snr{snr:g}_rep{rep}")
        task = dict(id=name, phantom=ph, b0=b0, te1=te1, res=res, dte=p["dte_ms"], n=p["echoes"],
                    protocol=p["name"], snr=snr, rep=rep)
        if var is not None:
            task = dict(id=f"{var}_{name}", variant=var, **{k: v for k, v in task.items() if k != "id"})
        out.append(task)
    return out


def ensure_prep(spec, root, phantom, b0, te1, variant=None):
    from .exp1.prepare import prepare
    from .exp1.io import res_tag
    cfg = field_config(spec, b0, te1, variant)
    key = prep_key(cfg, phantom)
    d = Path(root) / "prep" / key
    done = d / "prep.done"
    if not done.exists():
        if phantom == "P3":
            from .exp1.inputs import check_rc2
            from .rc2 import ROOT
            if not check_rc2():
                raise SystemExit(f"P3 needs the QSM Challenge 2.0 Sim1 files in {ROOT} "
                                 "(qsmsim-fetch-rc2, or set QSMSIM_RC2_DIR)")
        if cfg.forward == "v2":
            cfg.check_v2()
        log(f"prep {key}: {phantom} at {b0:g} T, TE1 {te1:g} ms, voxels {cfg.resolutions} mm, "
            f"spacings {cfg.dtes_ms} ms ({cfg.forward} forward model)" + (f", variant {variant}" if variant else ""))
        t = time.time()
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(cfg.to_json())
        prepare(cfg, phantom, d, log=lambda m: log(f"  {m}"))
        missing = [f"{res_tag(r)}/S0_dte{x:g}" for r in cfg.resolutions for x in cfg.dtes_ms
                   if not (d / "cache" / phantom / res_tag(r) / f"S0_dte{x:g}.npy").exists()]
        if missing:
            raise RuntimeError(f"prep {key}: missing {missing}")
        done.write_text(json.dumps(dict(seconds=time.time() - t)))
        log(f"prep {key}: done ({time.time() - t:.0f} s)")
    return cfg, d


# ------------------------------------------------------------------------------ run
def _dprime(auc):
    from scipy.stats import norm
    if auc is None or not np.isfinite(auc):
        return float("nan")
    return float(math.sqrt(2.0) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))


def flatten(m):
    """The metrics a sweep is usually read by, as flat columns."""
    r = dict(rmse=m.get("rmse"), hfen=m.get("hfen"), ssim=m.get("ssim"), deep_gm_slope=m.get("deep_gm_slope"))
    for name, d in (m.get("region") or {}).items():
        for k in ("nrmse", "rms_ppb", "ssim", "hfen"):
            if k in d:
                r[f"{name}.{k}"] = d[k]
    v = m.get("veins") or {}
    auc = v.get("auc_core_vs_shell")
    r.update(vein_auc=auc, vein_dprime=_dprime(auc), vein_moment_ratio=v.get("moment_ratio_median"),
             vein_peak_ratio=v.get("peak_ratio_median"))
    for key, b in (v.get("bins") or {}).items():
        if key.endswith("_a0-90deg"):
            dia = key[1:].split("_a")[0]
            for k in ("cnr", "moment_ratio", "peak_ratio", "lsf_fwhm_mm"):
                if k in b:
                    r[f"{k}_{dia}"] = b[k]
    return r


def _noise_sigma(spec, cfg, prep, root, task):
    """sigma for this dataset: the reference magnitude over (SNR x gain); bandwidth penalty as configured."""
    from .exp1 import jobs as J
    nz = spec["noise"]
    gain = float({f"{float(k):g}": v for k, v in (nz.get("gain") or {}).items()}.get(f"{task['b0']:g}", 1.0))
    ref_b0 = nz.get("reference_b0")
    if ref_b0 is None or float(ref_b0) == task["b0"]:
        wm = prep.wm_first_echo(task["dte"])
    else:
        rcfg, rdir = ensure_prep(spec, root, task["phantom"], float(ref_b0), task["te1"], task.get("variant"))
        wm = J.Prep(rcfg, rdir, task["phantom"], task["res"]).wm_first_echo(task["dte"])
    s = wm / (task["snr"] * gain)
    if cfg.bandwidth_penalty:
        s *= math.sqrt(cfg.bandwidth_ref_dte_ms / task["dte"])
    return float(s), gain


def run_dataset(spec, root, task, threads):
    """Simulate one dataset and score every chain x value. Returns the rows (also written to rows/<id>.json)."""
    if spec["backend"] == "gpu":
        os.environ["QSMSIM_BACKEND"] = "gpu"
    from . import backend as be
    from . import dipole, tfi
    from .exp1 import jobs as J
    from .exp1.io import save_nifti_cropped, write_json
    dipole.WORKERS = tfi.WORKERS = threads
    root = Path(root)
    final = root / "rows" / f"{task['id']}.json"
    if final.exists():
        return json.loads(final.read_text())
    t0 = time.time()
    cfg, pdir = ensure_prep(spec, root, task["phantom"], task["b0"], task["te1"], task.get("variant"))
    prep = J.Prep(cfg, pdir, task["phantom"], task["res"])
    sigma, gain = _noise_sigma(spec, cfg, prep, root, task)
    S, sha, _ = J.noisy_signal(cfg, prep, task["dte"], task["snr"], task["rep"], sigma=sigma)
    ds = J.Dataset(cfg, prep, S, task["dte"], task["n"], gpu=be.use_gpu())
    del S
    base = dict(task, sigma=sigma, gain=gain, te_last_ms=task["te1"] + task["dte"] * (task["n"] - 1),
                source_sha256=sha, field_err_median_ppb=ds.field_metrics["median_abs_err_ppb"],
                field_std_pred_ppb=ds.field_metrics["median_pred_std_ppb"],
                field_cycle_error_voxels=ds.field_metrics["voxels_cycle_error"])
    base.pop("id")
    rows = []
    maps = {}
    for bg, inv in ds.chains():
        for lam in lam_values(spec, cfg, inv):
            t = time.time()
            chi, _ = ds.invert(bg, inv, lam)
            sec = time.time() - t
            m, _ = ds.evaluator(chi)
            rows.append(dict(base, dataset=task["id"], background="none" if inv == "tfi" else bg, inversion=inv,
                             chain=J.chain_name(bg, inv), lam=lam, recon_seconds=sec, **flatten(m)))
            if spec["save_maps"] and task["rep"] == 0:
                maps[f"chi_{J.chain_name(bg, inv)}_lam{lam:.4g}"] = chi
            del chi
    if maps:
        save_nifti_cropped(root / "maps" / task["id"], maps, prep.box, prep.res)
    final.parent.mkdir(parents=True, exist_ok=True)
    write_json(final, rows)
    log(f"{task['id']}: {len(rows)} reconstructions ({time.time() - t0:.0f} s)")
    return rows


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_rows(root):
    rows = []
    for f in sorted((Path(root) / "rows").glob("*.json")):
        rows.extend(json.loads(f.read_text()))
    return rows


def cmd_run(a):
    spec = load_spec(a.spec)
    if a.backend:
        spec["backend"] = a.backend
    root = Path(a.out or f"runs/{spec['name']}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec, indent=1))
    if Path(a.spec).suffix.lower() != ".json":
        (root / "spec.toml").write_text(Path(a.spec).read_text())
    tasks = datasets(spec)
    workers = 1 if spec["backend"] == "gpu" else max(1, a.workers)
    nvar = len(spec["variants"])
    log(f"{spec['name']}: {len(tasks)} datasets" + (f" ({nvar} variants)" if nvar else "")
        + f" x {len(spec['inversions'])} inversions on the "
        f"{spec['backend'].upper()}, {workers} worker(s) x {a.threads} thread(s) -> {root}")
    from .exp1 import runner as R
    cfg0 = field_config(spec, spec["b0_tesla"][0], spec["te1_ms"][0], tasks[0].get("variant"))
    for ph in spec["phantoms"]:
        ps = cfg0.phantom(ph)
        log(f"  memory estimate {ph}: preparation {R.prep_mem_gb(cfg0, ps):.1f} GB, then per dataset "
            + ", ".join(f"{r:g} mm {R.job_mem_gb(cfg0, ps, r):.1f} GB" for r in spec["resolution_mm"])
            + (f" (x {workers} workers)" if workers > 1 else ""))
    # phantoms first, in this process (each is shared by many datasets; also builds the noise reference field)
    for var, ph, b0, te1 in sorted({(t.get("variant") or "", t["phantom"], t["b0"], t["te1"]) for t in tasks}):
        if spec["backend"] == "gpu":
            os.environ["QSMSIM_BACKEND"] = "gpu"
        ensure_prep(spec, root, ph, b0, te1, var or None)
        ref = spec["noise"].get("reference_b0")
        if ref is not None:
            ensure_prep(spec, root, ph, float(ref), te1, var or None)
    failed = 0
    if workers == 1:
        for t in tasks:
            try:
                run_dataset(spec, root, t, a.threads)
            except Exception as e:                                       # noqa: BLE001
                failed += 1
                log(f"{t['id']}: FAILED ({type(e).__name__}: {e})")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(run_dataset, spec, root, t, a.threads): t for t in tasks}
            for f in as_completed(fut):
                try:
                    f.result()
                except Exception as e:                                   # noqa: BLE001
                    failed += 1
                    log(f"{fut[f]['id']}: FAILED ({type(e).__name__}: {e})")
    rows = read_rows(root)
    write_csv(root / "results.csv", rows)
    summary = summarize(rows, spec["select"])
    write_csv(root / "summary.csv", summary)
    log(f"done: {len(rows)} rows -> {root / 'results.csv'}, {len(summary)} -> {root / 'summary.csv'}"
        + (f"; {failed} dataset(s) FAILED" if failed else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------- summary
COND = ("variant", "phantom", "b0_tesla", "te1", "res", "protocol", "dte", "n", "snr", "gain", "chain", "background",
        "inversion")


def _better(metric):
    return min if any(metric.startswith(p) or f".{p}" in metric for p in LOWER_BETTER) else max


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def summarize(rows, select):
    """Per condition x chain: the value each selection metric picks (mean over replicates), with every metric's
    replicate mean, min and max at that value."""
    rows = [dict(r, b0_tesla=r.get("b0_tesla", r.get("b0"))) for r in rows]
    skip = set(COND) | {"b0", "rep", "dataset", "lam", "sigma", "source_sha256", "te_last_ms"}
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in COND), {}).setdefault(float(r["lam"]), []).append(r)
    out = []
    for cond, by_lam in groups.items():
        metrics = [k for k in by_lam[next(iter(by_lam))][0] if k not in skip]
        means = {lam: {k: np.mean([v for v in (_num(r.get(k)) for r in rs) if v is not None] or [np.nan])
                       for k in metrics} for lam, rs in by_lam.items()}
        for sel in select:
            cand = [lam for lam in by_lam if np.isfinite(means[lam].get(sel, np.nan))]
            if not cand:
                continue
            best = _better(sel)(cand, key=lambda lam: means[lam][sel])
            lams = sorted(by_lam)
            rs = by_lam[best]
            row = dict(zip(COND, cond), selected_by=sel, lam=best, n_rep=len(rs),
                       lam_at_grid_edge=best in (lams[0], lams[-1]) and len(lams) > 1,
                       te_last_ms=rs[0].get("te_last_ms"))
            for k in metrics:
                vals = [v for v in (_num(r.get(k)) for r in rs) if v is not None]
                row[k] = float(np.mean(vals)) if vals else None
                if len(vals) > 1 and k in (sel, "vein_auc", "rmse"):
                    row[f"{k}_min"], row[f"{k}_max"] = float(min(vals)), float(max(vals))
            out.append(row)
    return out


def cmd_summarize(a):
    root = Path(a.out)
    spec = json.loads((root / "spec.json").read_text())
    rows = read_rows(root)
    write_csv(root / "results.csv", rows)
    s = summarize(rows, a.select or spec["select"])
    write_csv(root / "summary.csv", s)
    log(f"{len(rows)} rows -> {root / 'summary.csv'} ({len(s)} rows)")
    return 0


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(a.out)
    if a.results:                                    # every value of every run (replicates averaged)
        rows, sel = _read_csv(root / "results.csv"), None
    else:
        rows = _read_csv(root / "summary.csv")
        sel = a.selected_by or rows[0]["selected_by"]
        rows = [r for r in rows if r["selected_by"] == sel]
    for f in a.where or []:
        k, v = f.split("=", 1)
        rows = [r for r in rows if r.get(k) == v]
    if not rows:
        raise SystemExit("no rows to plot (check --selected-by / --where)")
    if a.y not in rows[0]:
        raise SystemExit(f"{a.y} is not a column; columns: {', '.join(rows[0])}")
    groups = a.group.split(",") if a.group else ["chain"]
    facets = sorted({r[a.facet] for r in rows}, key=lambda v: _num(v) if _num(v) is not None else v) \
        if a.facet else [None]
    fig, axes = plt.subplots(1, len(facets), figsize=(4.6 * len(facets), 3.8), squeeze=False, sharey=True)
    for ax, fv in zip(axes[0], facets):
        sub = [r for r in rows if fv is None or r[a.facet] == fv]
        acc = {}
        for r in sub:
            x, y = _num(r[a.x]), _num(r[a.y])
            if x is not None and y is not None:
                acc.setdefault(" ".join(r.get(g, "") for g in groups), {}).setdefault(x, []).append(y)
        lines = {lab: sorted((x, float(np.mean(ys))) for x, ys in d.items()) for lab, d in acc.items()}
        for lab, pts in sorted(lines.items()):
            ax.plot(*zip(*pts), marker="o", lw=2, ms=6, label=lab)
        if a.logx or a.x in ("snr", "lam"):
            ax.set_xscale("log")
        xs = sorted({x for pts in lines.values() for x, _ in pts})
        if 0 < len(xs) <= 12:                                  # tick at the simulated values, plain numbers
            ax.set_xticks(xs, [f"{x:g}" for x in xs])
            ax.minorticks_off()
        ax.set_xlabel(a.x)
        ax.set_title(f"{a.facet} = {fv}" if fv is not None else "")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"{a.y} (value chosen by {sel})" if sel else f"{a.y} (mean over replicates)")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(a.png or root / f"plot_{a.y}_vs_{a.x}.png")
    fig.savefig(out, dpi=150)
    log(f"-> {out}")
    return 0


def cmd_compare(a):
    """Two runs of the same spec (e.g. CPU and GPU): the largest difference per metric over matching rows."""
    A, B = (_read_csv(Path(d) / "results.csv") for d in (a.a, a.b))
    key = lambda r: (r["dataset"], r["chain"], f"{float(r['lam']):.6g}")        # noqa: E731
    bmap = {key(r): r for r in B}
    pairs = [(r, bmap[key(r)]) for r in A if key(r) in bmap]
    if not pairs:
        raise SystemExit("no matching rows (dataset, chain, value)")
    if any(ra["source_sha256"] != rb["source_sha256"] for ra, rb in pairs):
        raise SystemExit("the noisy source data differ: not the same spec or seeds")
    skip = {"dataset", "chain", "lam", "source_sha256", "recon_seconds", "background", "inversion", "protocol",
            "phantom"}
    # each difference is relative to the metric's largest magnitude in the run (near-zero values stay meaningful)
    scale = {}
    for r in A:
        for k, v in r.items():
            v = _num(v)
            if v is not None:
                scale[k] = max(scale.get(k, 0.0), abs(v))
    worst = {}
    for ra, rb in pairs:
        for k in ra:
            if k in skip or k not in rb:
                continue
            x, y = _num(ra[k]), _num(rb[k])
            if x is None or y is None:
                continue
            d = abs(x - y) / max(scale.get(k, 0.0), 1e-12)
            if d > worst.get(k, (0.0,))[0]:
                worst[k] = (d, ra["dataset"], ra["chain"], ra["lam"])
    bad = {k: v for k, v in worst.items() if v[0] > a.tol}
    print(f"{len(pairs)} matching rows of {len(A)} / {len(B)}; largest differences (fraction of the metric's "
          "largest magnitude in the run):")
    for k, (d, ds, ch, lam) in sorted(worst.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {k:32s} {d:9.2e}  {'OVER' if d > a.tol else 'ok  '}  ({ds}, {ch}, {lam})")
    print(f"{'PASS' if not bad else 'FAIL'}: {len(bad)} metric(s) over the tolerance {a.tol:g}")
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="qsmsim-explore", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("example", help="print a commented example spec")
    r = sp.add_parser("run", help="simulate every combination in a spec")
    r.add_argument("spec")
    r.add_argument("--out", help="output directory (default runs/<name>)")
    r.add_argument("--workers", type=int, default=1, help="datasets in parallel (CPU only; each needs its own memory)")
    r.add_argument("--threads", type=int, default=int(os.environ.get("QSMSIM_THREADS", "-1")),
                   help="FFT threads per dataset (-1 = all cores)")
    r.add_argument("--backend", choices=["cpu", "gpu"], help="override the spec's backend")
    s = sp.add_parser("summarize", help="rebuild results.csv / summary.csv from rows/")
    s.add_argument("out")
    s.add_argument("--select", nargs="+", help="selection metrics (default: the spec's)")
    q = sp.add_parser("plot", help="plot a summary metric against a parameter")
    q.add_argument("out")
    q.add_argument("--x", default="snr")
    q.add_argument("--y", default="vein_auc")
    q.add_argument("--group", default="protocol", help="comma-separated columns that make one line")
    q.add_argument("--facet", help="column that makes one panel per value")
    q.add_argument("--selected-by", help="which selection rule's values to plot (default: the first)")
    q.add_argument("--where", nargs="*", help="filters col=value")
    q.add_argument("--results", action="store_true",
                   help="plot results.csv (every regularization value) instead of the selected values, e.g. --x lam")
    q.add_argument("--logx", action="store_true")
    q.add_argument("--png")
    c = sp.add_parser("compare", help="compare two runs of one spec (e.g. CPU against GPU)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=0.02, help="relative tolerance per metric (default 0.02)")
    c.add_argument("--top", type=int, default=15)
    a = p.parse_args(argv)
    if a.cmd == "example":
        sys.stdout.write(EXAMPLE)
        return 0
    return {"run": cmd_run, "summarize": cmd_summarize, "plot": cmd_plot, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
