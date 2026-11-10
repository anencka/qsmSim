"""Experiment 1 command line (Experiment_1.md).

  python -m qsmsim.exp1 plan       --config production --out OUT
  python -m qsmsim.exp1 run        --config production --out OUT [--stages prep,tune1,tune2,run] [--reps 0,1,2,3,4,5]
  python -m qsmsim.exp1 status     --out OUT
  python -m qsmsim.exp1 verify     --out OUT [--full] [--sample 6]
  python -m qsmsim.exp1 aggregate  --out OUT
  python -m qsmsim.exp1 resweep    --src PROD_OUT --out OUT [--methods none] [--tfi-reps 0,1,2] [--reps 0] [--keep-maps]
  python -m qsmsim.exp1 exp2       --ref EXP1_OUT --out OUT [--config exp2_7t] [--hpg] [--tfi-reps 0] [--plan]
  python -m qsmsim.exp1 control    --src V1_OUT --out OUT [--v2-ref CONTROL_3T_OUT] [--reps 0] [--snrs 20] [--plan]
  python -m qsmsim.exp1 check-inputs
  python -m qsmsim.exp1 job ID     --out OUT          (internal: one job, used by the runner)

--config is a preset (production, pilot, toy, realsize) or a JSON file. The
config is frozen into OUT/config.json on the first run; later commands on the
same OUT use it (a different --config is refused unless --force). The one change accepted in place is a widening
(tier B, Experiments_v2.md 6): more replicates (run / resweep / exp2 --reps) or more TFI replicates (--tfi-reps),
everything else equal; finished jobs stay done and the new replicates run as new jobs.
`resweep` reconstructs a finished production run (--src) again on an extended λ grid
(qsmsim/exp1/resweep.py) into its own OUT; status / aggregate / run --out OUT work on it as well.
`exp2` simulates the Experiment 1 phantoms at 7 T with noise anchored to a finished 3T run (--ref) and
sweeps λ like the resweep (Experiment_2.md, qsmsim/exp1/exp2.py).
`control` is the phase 4 control run (GPU_Plan.md 2.4, qsmsim/exp1/control.py): forward model v2 against the finished
v1 run --src (Experiment 1 production at 3 T; Experiment 2 at 7 T, with --v2-ref the 3T control run) on the same seeds
and noise, both on one backend.

Environment: QSMSIM_WORKERS, QSMSIM_MEM_GB, QSMSIM_PREP_THREADS override the
scheduling fields of the config; on a GPU host QSMSIM_GPU_LANES (reconstruction on the GPU, that many jobs at a
time), QSMSIM_EVAL_WORKERS_PER_LANE (CPU evaluation pool per GPU job) and QSMSIM_GPU_POOL_GB (CuPy pool cap shared
by the lanes) do the same for the GPU fields (GPU_Plan.md phase 3).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

from . import config as C


def _cfg_for(out, name, force=False, create=True):
    out = Path(out)
    frozen = out / "config.json"
    if frozen.exists():
        cfg = C.Config.from_json(frozen.read_text())
        if name and name != cfg.name and not force:
            other = C.load(name)
            if other.to_json() != cfg.to_json():
                sys.exit(f"{out} was created with config '{cfg.name}'; refusing '{name}' (use --force or a new --out)")
    else:
        if not create:
            sys.exit(f"{frozen} not found")
        cfg = C.load(name or "production")
        out.mkdir(parents=True, exist_ok=True)
        frozen.write_text(cfg.to_json())
    env = os.environ
    cfg.workers = int(env.get("QSMSIM_WORKERS", cfg.workers))
    cfg.mem_gb = float(env.get("QSMSIM_MEM_GB", cfg.mem_gb))
    cfg.prep_threads = int(env.get("QSMSIM_PREP_THREADS", cfg.prep_threads))
    cfg.gpu_lanes = int(env.get("QSMSIM_GPU_LANES", cfg.gpu_lanes))
    cfg.eval_workers = int(env.get("QSMSIM_EVAL_WORKERS_PER_LANE", cfg.eval_workers))
    cfg.gpu_mem_gb = float(env.get("QSMSIM_GPU_POOL_GB", cfg.gpu_mem_gb))
    return cfg


def widens(have, want):
    """Tier B (Experiments_v2.md 6): a frozen run may grow in place, by replicates (only_reps) and TFI replicates
    (resweep.tfi_reps), as long as nothing else changes and nothing is dropped. True when `want` only widens `have`;
    the finished jobs stay done and the new replicates become new jobs."""
    a, b = json.loads(have.to_json()), json.loads(want.to_json())

    def grab(d):
        return d.pop("only_reps"), (d["resweep"].pop("tfi_reps") if d.get("resweep") else None)
    (ra, ta), (rb, tb) = grab(a), grab(b)
    if a != b or (ra, ta) == (rb, tb):
        return False
    every = set(range(have.reps))
    sa, sb = every if ra is None else set(ra), every if rb is None else set(rb)
    if not sa <= sb or not sb <= every or not set(ta or []) <= set(tb or []):
        return False
    return True


def _describe_widening(have, want):
    t = lambda c: (c.resweep or {}).get("tfi_reps")
    return (f"replicates {have.active_reps()} -> {want.active_reps()}"
            + (f", TFI replicates {t(have)} -> {t(want)}" if t(have) != t(want) else ""))


def _freeze(out, want, force, what):
    """Write `want` as out/config.json: new runs, --force, or a widening of the frozen one (tier B)."""
    frozen = Path(out) / "config.json"
    if frozen.exists():
        have = C.Config.from_json(frozen.read_text())
        if have.to_json() == want.to_json():
            return
        if not force and not widens(have, want):
            sys.exit(f"{out} holds a different {what} (use --force or a new --out; only a widening of the "
                     "replicates or TFI replicates is accepted in place)")
        if not force:
            print(f"{out}: widening in place: {_describe_widening(have, want)}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = frozen.with_suffix(".json.tmp")
    tmp.write_text(want.to_json())
    os.replace(tmp, frozen)


def _set_threads(n):
    from .. import dipole, tfi
    dipole.WORKERS = tfi.WORKERS = n


def cmd_job(a):
    from .runner import build_jobs, execute
    out = Path(a.out)
    cfg = _cfg_for(out, None, create=False)
    job = {j.id: j for j in build_jobs(cfg)}[a.id]
    _set_threads(int(os.environ.get("QSMSIM_THREADS", job.threads)))
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:8.1f}s] {m}", flush=True)
    safe = a.id.replace(":", "__")
    state = out / "state"
    state.mkdir(parents=True, exist_ok=True)
    from . import evalpool
    try:
        res = execute(cfg, out, job, log)
    except BaseException:
        traceback.print_exc()
        (state / f"{safe}.failed").write_text(json.dumps(dict(error=traceback.format_exc()[-4000:])))
        return 1
    finally:
        evalpool.close()
    rec = dict(id=a.id, seconds=time.time() - t0, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               peak_rss_children_gb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6,
               threads=job.threads, mem_budget_gb=job.mem_gb, gpu=job.gpu, result=res, host=_host())
    from .. import backend as be
    if job.gpu and be.use_gpu():
        rec["device"] = be.device_info()
        rec["gpu_pool_held_gb"] = be.pool_held_gb()
    tmp = state / f"{safe}.done.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, state / f"{safe}.done")
    log(f"done in {rec['seconds']:.0f}s, peak RSS {rec['peak_rss_gb']:.2f} GB")
    return 0


def _host():
    import platform
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    return dict(node=platform.node(), cpu=cpu, git_commit=os.environ.get("QSMSIM_GIT_COMMIT", ""),
                image=os.environ.get("QSMSIM_IMAGE", ""))


def cmd_plan(a):
    from .runner import build_jobs
    cfg = _cfg_for(a.out, a.config, a.force)
    jobs = build_jobs(cfg)
    by = {}
    for j in jobs:
        by.setdefault(j.kind, []).append(j)
    plan = dict(config=cfg.name, jobs=len(jobs), by_kind={k: len(v) for k, v in by.items()},
                datasets_production=sum(len(cfg.counts(j.args["dte"])) for j in by.get("run", [])),
                chi_maps_production=None, mem_gb={k: sorted({round(j.mem_gb, 1) for j in v}) for k, v in by.items()},
                deps_ok=all(d in {x.id for x in jobs} for j in jobs for d in j.deps))
    nch = len(cfg.backgrounds) * len(cfg.inversions) + int(cfg.tfi)
    nch3 = len(cfg.inversions) + int(cfg.tfi)
    plan["chi_maps_production"] = sum(len(cfg.counts(j.args["dte"])) * (nch3 if j.args["ph"] == "P3" else nch)
                                      for j in by.get("run", []))
    Path(a.out, "plan.json").write_text(json.dumps(dict(summary=plan, jobs=[j.__dict__ for j in jobs]), indent=1,
                                                   default=list))
    print(json.dumps(plan, indent=1))
    return 0


def cmd_run(a):
    from .runner import Runner
    cfg = _cfg_for(a.out, a.config, a.force)
    if a.reps:                                       # tier B: more replicates on the same (frozen) run
        frozen = Path(a.out) / "config.json"
        want = C.Config.from_json(frozen.read_text())
        want.only_reps = [int(r) for r in a.reps.split(",")]
        _freeze(a.out, want, False, "configuration")
        cfg = _cfg_for(a.out, None, create=False)
    stages = a.stages.split(",") if a.stages else None
    r = Runner(cfg, a.out, a.workers or cfg.workers, a.mem_gb or cfg.mem_gb, stages, a.retry_failed)
    return r.run(log=lambda m: print(m, flush=True))


def cmd_resweep(a):
    from . import resweep as RS
    from .runner import Runner
    out, src = Path(a.out), Path(a.src)
    if not (src / "config.json").exists():
        sys.exit(f"{src}/config.json not found: --src must be a finished production output directory")
    src_cfg = C.Config.from_json((src / "config.json").read_text())
    if src_cfg.resweep:
        sys.exit(f"{src} is itself a resweep output")
    factors = [float(f) for f in a.factors.split(",")] if a.factors else None
    methods = None if not a.methods else [] if a.methods == "none" else a.methods.split(",")
    tfi_reps = [int(r) for r in a.tfi_reps.split(",")] if a.tfi_reps else []
    reps = [int(r) for r in a.reps.split(",")] if a.reps else None
    try:
        want = RS.make_config(src_cfg, out, src, factors, methods, tfi_reps, a.keep_maps, reps)
    except ValueError as e:
        sys.exit(str(e))
    _freeze(out, want, a.force, "resweep configuration")
    cfg = _cfg_for(out, None, create=False)
    if a.plan:
        from .runner import build_jobs
        js = build_jobs(cfg)
        n_ds = sum(len(cfg.counts(j.args["dte"])) for j in js if j.kind == "resweep")
        print(json.dumps(dict(jobs=len(js), by_kind={k: sum(1 for j in js if j.kind == k) for k in ("cache", "resweep")},
                              datasets=n_ds, resweep=cfg.resweep,
                              mem_gb=sorted({round(j.mem_gb, 1) for j in js})), indent=1))
        return 0
    r = Runner(cfg, out, a.workers or cfg.workers, a.mem_gb or cfg.mem_gb, None, a.retry_failed)
    return r.run(log=lambda m: print(m, flush=True))


def cmd_exp2(a):
    from . import exp2 as X
    from .runner import Runner, build_jobs
    out = Path(a.out)
    frozen = out / "config.json"
    tfi_reps = [int(r) for r in a.tfi_reps.split(",")] if a.tfi_reps else []
    reps = [int(r) for r in a.reps.split(",")] if a.reps else None
    try:
        want = X.make_config(C.load(a.config), out, a.ref, a.hpg, tfi_reps, reps)
    except ValueError as e:
        sys.exit(str(e))
    _freeze(out, want, a.force, "Experiment 2 configuration")
    cfg = _cfg_for(out, None, create=False)
    if a.plan:
        js = build_jobs(cfg)
        n_ds = sum(len(cfg.counts(j.args["dte"])) for j in js if j.kind == "resweep")
        print(json.dumps(dict(jobs=len(js), by_kind={k: sum(1 for j in js if j.kind == k) for k in ("prep", "tune1", "resweep")},
                              datasets=n_ds, b0_tesla=cfg.b0_tesla, ref=cfg.exp2["ref"], levels=cfg.exp2["levels"],
                              protocols=cfg.protocols(), factors=cfg.resweep["factors"],
                              mem_gb=sorted({round(j.mem_gb, 1) for j in js})), indent=1))
        return 0
    r = Runner(cfg, out, a.workers or cfg.workers, a.mem_gb or cfg.mem_gb, None, a.retry_failed)
    return r.run(log=lambda m: print(m, flush=True))


def cmd_control(a):
    from . import control as CT
    from .runner import Runner, build_jobs
    out, src = Path(a.out), Path(a.src)
    split = lambda v, f: [f(x) for x in v.split(",")] if v else None
    try:
        want = CT.make_config(CT._load_cfg(src), out, src, a.v2_ref, split(a.reps, int), split(a.snrs, float),
                              split(a.phantoms, str), split(a.resolutions, float), a.fine_mm)
    except ValueError as e:
        sys.exit(str(e))
    frozen = out / "config.json"
    if frozen.exists():
        have = C.Config.from_json(frozen.read_text())
        if have.to_json() != want.to_json() and not a.force:
            sys.exit(f"{out} holds a different control configuration (use --force or a new --out)")
    out.mkdir(parents=True, exist_ok=True)
    if not frozen.exists() or a.force:
        frozen.write_text(want.to_json())
    cfg = _cfg_for(out, None, create=False)
    if a.plan:
        js = build_jobs(cfg)
        n_ds = sum(len(cfg.counts(j.args["dte"])) for j in js if j.kind == "ctl")
        print(json.dumps(dict(jobs=len(js), by_kind={k: sum(1 for j in js if j.kind == k) for k in ("cache", "prep", "cmp", "ctl")},
                              datasets=n_ds, b0_tesla=cfg.b0_tesla, control=cfg.control, snrs=cfg.active_snrs(),
                              resolutions=cfg.active_resolutions(), protocols=cfg.protocols(), reps=cfg.active_reps(),
                              mem_gb=sorted({round(j.mem_gb, 1) for j in js})), indent=1))
        return 0
    r = Runner(cfg, out, a.workers or cfg.workers, a.mem_gb or cfg.mem_gb, None, a.retry_failed)
    return r.run(log=lambda m: print(m, flush=True))


def cmd_status(a):
    from .runner import Runner
    cfg = _cfg_for(a.out, None, create=False)
    r = Runner(cfg, a.out, 1, 1)
    print("\n".join(r.eta()) if a.eta else json.dumps(r.summary(), indent=1))
    return 0


def cmd_verify(a):
    from .verify import verify
    cfg = _cfg_for(a.out, None, create=False)
    if cfg.control:
        print("verify: a control run checks itself while it runs: each v1 dataset's source SHA-256 against the v1 run "
              "(aggregate/summary.md) and, at 7 T, the v2 truth volumes against the 3T control's")
        return 0
    if cfg.exp2:
        print("verify: an Experiment 2 output is checked while it runs: each prep job asserts that the truth "
              "volumes equal the 3T reference's, and each dataset's evaluation mask is compared with its 3T "
              "partner (aggregate/summary.md)")
        return 0
    if cfg.resweep:
        print("verify: a resweep output has no data of its own to verify; every dataset's SHA-256 is "
              "compared with production while it runs (aggregate/summary.md lists any mismatch)")
        return 0
    ok = verify(cfg, a.out, full=a.full, sample=a.sample, phantoms=a.phantoms)
    return 0 if ok else 1


def cmd_aggregate(a):
    from .aggregate import aggregate
    cfg = _cfg_for(a.out, None, create=False)
    aggregate(cfg, a.out)
    return 0


def cmd_check_inputs(a):
    from .inputs import check_rc2
    return 0 if check_rc2(verbose=True) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m qsmsim.exp1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--config"); p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_plan)
    p = sub.add_parser("run"); p.add_argument("--config"); p.add_argument("--out", required=True)
    p.add_argument("--stages"); p.add_argument("--workers", type=int); p.add_argument("--mem-gb", type=float)
    p.add_argument("--reps", help="replicates to run, e.g. 0,1,2,3,4,5: widens a frozen run in place (tier B)")
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("status"); p.add_argument("--out", required=True)
    p.add_argument("--eta", action="store_true", help="time estimate from the finished jobs instead of the counts")
    p.set_defaults(f=cmd_status)
    p = sub.add_parser("verify"); p.add_argument("--out", required=True); p.add_argument("--full", action="store_true")
    p.add_argument("--sample", type=int, default=6); p.add_argument("--phantoms", nargs="*")
    p.set_defaults(f=cmd_verify)
    p = sub.add_parser("aggregate"); p.add_argument("--out", required=True); p.set_defaults(f=cmd_aggregate)
    p = sub.add_parser("resweep"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--factors", help="comma-separated multiples of the stage-1 λ (default 1,0.316,0.1,0.0316,0.01)")
    p.add_argument("--methods", help="comma-separated inversions (default medi,tv,cfl2,tkd); 'none' for TFI only")
    p.add_argument("--reps", help="replicates to sweep, e.g. 0 (default all)")
    p.add_argument("--tfi-reps", help="replicates that also sweep TFI, e.g. 0 or 0,1,2 (default none)")
    p.add_argument("--keep-maps", action="store_true", help="also save every χ map and per-vein table")
    p.add_argument("--plan", action="store_true", help="write the config and print the job plan only")
    p.add_argument("--workers", type=int); p.add_argument("--mem-gb", type=float)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_resweep)
    p = sub.add_parser("exp2"); p.add_argument("--ref", required=True, help="finished Experiment 1 (3T) run directory")
    p.add_argument("--out", required=True); p.add_argument("--config", default="exp2_7t")
    p.add_argument("--hpg", action="store_true", help="also simulate 2.4 ms (7T with high-performance gradients)")
    p.add_argument("--tfi-reps", help="replicates that also sweep TFI, e.g. 0 or 0,1,2 (default none)")
    p.add_argument("--reps", help="replicates to sweep (default all)")
    p.add_argument("--plan", action="store_true", help="write the config and print the job plan only")
    p.add_argument("--workers", type=int); p.add_argument("--mem-gb", type=float)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_exp2)
    p = sub.add_parser("control"); p.add_argument("--src", required=True, help="finished v1 run (production or exp2)")
    p.add_argument("--out", required=True)
    p.add_argument("--v2-ref", help="7 T only: the 3T control run (its v2 truth volumes are checked)")
    p.add_argument("--reps", help="replicates (default all)"); p.add_argument("--snrs", help="noise levels (default all)")
    p.add_argument("--phantoms", help="phantoms (default the v1 run's)"); p.add_argument("--resolutions")
    p.add_argument("--fine-mm", type=float, help="v2 fine spacing (default the config's 0.1 mm; toy runs 0.2)")
    p.add_argument("--plan", action="store_true", help="write the config and print the job plan only")
    p.add_argument("--workers", type=int); p.add_argument("--mem-gb", type=float)
    p.add_argument("--retry-failed", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_control)
    p = sub.add_parser("check-inputs"); p.set_defaults(f=cmd_check_inputs)
    p = sub.add_parser("job"); p.add_argument("id"); p.add_argument("--out", required=True); p.set_defaults(f=cmd_job)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
