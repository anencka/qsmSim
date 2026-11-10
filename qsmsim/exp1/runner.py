"""Job graph and a resumable, memory-aware scheduler for Experiment 1.

Jobs run as subprocesses (`python -m qsmsim.exp1 job <id>`), one thread each by
default, so memory is released per job and a crash or out-of-memory kill only
loses that job. Completion is recorded as out/state/<id>.done (JSON) after the
job's outputs were written atomically; a restart skips done jobs and re-runs
everything else. Ready jobs start in priority order (coarse resolutions first,
then tuning before production) whenever cores and the memory budget allow.

GPU lanes (GPU_Plan.md phase 3; Config.gpu_lanes > 0): every reconstruction job (tune1, tune2, run, resweep) and a
v2 preparation is a GPU job. At most gpu_lanes of them run at once, each launched with QSMSIM_BACKEND=gpu and a CPU
evaluation pool of eval_workers processes (QSMSIM_EVAL_WORKERS); their cores and host memory count against the same
budgets as CPU jobs, which keep running alongside (v1 preparation, cache checks).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import proto_tag, res_tag, snr_tag

STAGE_ORDER = {"prep": 0, "tune1": 1, "tune2": 2, "run": 3, "cache": 0, "resweep": 4, "cmp": 2, "ctl": 4}


@dataclass
class Job:
    id: str
    kind: str
    args: dict
    deps: list = field(default_factory=list)
    mem_gb: float = 1.0
    threads: int = 1
    priority: tuple = ()
    gpu: bool = False


GPU_KINDS = ("tune1", "tune2", "run", "resweep", "ctl")


def gpu_jobs(cfg, jobs):
    """With GPU lanes: mark the GPU jobs and size them (cores: 1 + the evaluation pool; host memory: the CPU estimate
    plus the pool's evaluator copies)."""
    if cfg.gpu_lanes <= 0:
        return jobs
    for j in jobs:
        if j.kind in GPU_KINDS:
            j.gpu = True
            j.threads = 1 + cfg.eval_workers
            j.mem_gb = j.mem_gb * (1 + cfg.eval_mem_factor * cfg.eval_workers)
        elif j.kind == "prep" and cfg.forward == "v2":
            j.gpu = True
    return jobs


def _voxels(cfg, spec, res):
    return int(np.prod([round(f / res) for f in spec.fov_mm]))


def prep_mem_gb(cfg, spec):
    # measured on the toy run: peak = 0.98 x (0.5 + n * (60 + 22 pad^3) B); 1.25x margin on top
    nm = _voxels(cfg, spec, cfg.master_mm)
    return 1.25 * (0.5 + nm * (60 + 22 * cfg.mesh_pad**3) / 1e9)


def job_mem_gb(cfg, spec, res):
    # realsize acceptance job (1.5 mm, production settings): peak 0.96 x (0.4 + n * 650 B); 1.25x margin
    return 1.25 * (0.4 + _voxels(cfg, spec, res) * 650 / 1e9)


def job_threads(cfg, mem_gb):
    """FFT threads per job: when memory, not cores, limits concurrency, the idle cores
    go to the running jobs (up to 4 each). BLAS stays single-threaded (bit-reproducible)."""
    if cfg.job_threads > 1:
        return cfg.job_threads
    conc = max(1, int(cfg.mem_gb // mem_gb))
    return int(max(1, min(4, cfg.workers // conc))) if conc < cfg.workers else 1


def job_id(kind, **a):
    if kind == "prep":
        return f"prep:{a['ph']}"
    if kind == "tune1":
        return f"tune1:{a['ph']}:{res_tag(a['res'])}:{snr_tag(a['snr'])}"
    if kind == "tune2":
        return f"tune2:{a['ph']}:{res_tag(a['res'])}:{snr_tag(a['snr'])}:{proto_tag(a['dte'], a['n'])}"
    if kind == "cache":
        return f"cache:{a['ph']}"
    if kind == "cmp":
        return f"cmp:{a['ph']}:{res_tag(a['res'])}"
    tail = f"{a['ph']}:{res_tag(a['res'])}:{snr_tag(a['snr'])}:dte{a['dte']:g}:rep{a['rep']}"
    return f"{kind}:{tail}"


def build_jobs(cfg):
    if cfg.control:
        return gpu_jobs(cfg, build_control_jobs(cfg))
    if cfg.exp2:
        return gpu_jobs(cfg, build_exp2_jobs(cfg))
    if cfg.resweep:
        return gpu_jobs(cfg, build_resweep_jobs(cfg))
    return gpu_jobs(cfg, _build_exp1_jobs(cfg))


def _build_exp1_jobs(cfg):
    jobs = []
    ref = (float(cfg.tune_ref[0]), int(cfg.tune_ref[1]))
    for ph in cfg.active_phantoms():
        spec = cfg.phantom(ph)
        pid = job_id("prep", ph=ph)
        jobs.append(Job(pid, "prep", dict(ph=ph), [], prep_mem_gb(cfg, spec), cfg.prep_threads, (0, 0, ph)))
        for res in cfg.active_resolutions():
            mem = job_mem_gb(cfg, spec, res)
            thr = job_threads(cfg, mem)
            for snr in cfg.active_snrs():
                base = (-res, ph, snr)
                t1 = job_id("tune1", ph=ph, res=res, snr=snr)
                if cfg.tuning:
                    jobs.append(Job(t1, "tune1", dict(ph=ph, res=res, snr=snr), [pid], mem, thr,
                                    (1, *base, STAGE_ORDER["tune1"])))
                    for dte, n in cfg.protocols():
                        if (float(dte), int(n)) == ref:
                            continue
                        jobs.append(Job(job_id("tune2", ph=ph, res=res, snr=snr, dte=dte, n=n), "tune2",
                                        dict(ph=ph, res=res, snr=snr, dte=dte, n=n), [t1], mem, thr,
                                        (1, *base, STAGE_ORDER["tune2"])))
                for dte in cfg.dtes_ms:
                    for rep in cfg.active_reps():
                        deps = [pid]
                        if cfg.tuning:
                            for n in cfg.counts(dte):
                                deps.append(t1 if (float(dte), int(n)) == ref else
                                            job_id("tune2", ph=ph, res=res, snr=snr, dte=dte, n=n))
                        jobs.append(Job(job_id("run", ph=ph, res=res, snr=snr, dte=dte, rep=rep), "run",
                                        dict(ph=ph, res=res, snr=snr, dte=dte, rep=rep), deps, mem, thr,
                                        (1, *base, STAGE_ORDER["run"], dte, rep)))
    ids = {j.id for j in jobs}
    for j in jobs:                                   # filtered runs: dependencies outside the filter must be done
        j.deps = [d for d in j.deps if d in ids]
    return jobs


def build_resweep_jobs(cfg):
    """Extended λ sweep on a finished run: per phantom a cache check (or regeneration), then one
    job per phantom x resolution x SNR x dTE x replicate, coarse resolutions first."""
    jobs = []
    for ph in cfg.active_phantoms():
        spec = cfg.phantom(ph)
        cid = job_id("cache", ph=ph)
        jobs.append(Job(cid, "cache", dict(ph=ph), [], prep_mem_gb(cfg, spec), cfg.prep_threads, (0, 0, ph)))
        for res in cfg.active_resolutions():
            mem = job_mem_gb(cfg, spec, res)
            thr = job_threads(cfg, mem)
            for snr in cfg.active_snrs():
                for dte in cfg.dtes_ms:
                    for rep in cfg.active_reps():
                        jobs.append(Job(job_id("resweep", ph=ph, res=res, snr=snr, dte=dte, rep=rep), "resweep",
                                        dict(ph=ph, res=res, snr=snr, dte=dte, rep=rep), [cid], mem, thr,
                                        (1, -res, ph, snr, STAGE_ORDER["resweep"], dte, rep)))
    return jobs


def build_exp2_jobs(cfg):
    """Experiment 2: per phantom a prep (truth checked against the 3T reference), per phantom x resolution x
    noise level a stage-1 tuning on the reference protocol, then the λ sweep per dTE x replicate."""
    jobs = []
    for ph in cfg.active_phantoms():
        spec = cfg.phantom(ph)
        pid = job_id("prep", ph=ph)
        jobs.append(Job(pid, "prep", dict(ph=ph), [], prep_mem_gb(cfg, spec), cfg.prep_threads, (0, 0, ph)))
        for res in cfg.active_resolutions():
            mem = job_mem_gb(cfg, spec, res)
            thr = job_threads(cfg, mem)
            for snr in cfg.active_snrs():
                base = (-res, ph, snr)
                t1 = job_id("tune1", ph=ph, res=res, snr=snr)
                jobs.append(Job(t1, "tune1", dict(ph=ph, res=res, snr=snr), [pid], mem, thr,
                                (1, *base, STAGE_ORDER["tune1"])))
                for dte in cfg.dtes_ms:
                    for rep in cfg.active_reps():
                        jobs.append(Job(job_id("resweep", ph=ph, res=res, snr=snr, dte=dte, rep=rep), "resweep",
                                        dict(ph=ph, res=res, snr=snr, dte=dte, rep=rep), [t1], mem, thr,
                                        (1, *base, STAGE_ORDER["resweep"], dte, rep)))
    return jobs


def build_control_jobs(cfg):
    """Phase 4 control run (control.py): per phantom the v1 cache check (CPU) and the v2 preparation (GPU); per
    phantom x resolution the noiseless comparison (CPU); per phantom x resolution x SNR x dTE x replicate the paired
    reconstruction of both forward models (GPU), coarse resolutions first."""
    jobs = []
    for ph in cfg.active_phantoms():
        spec = cfg.phantom(ph)
        cid, pid = job_id("cache", ph=ph), job_id("prep", ph=ph)
        jobs.append(Job(cid, "cache", dict(ph=ph), [], prep_mem_gb(cfg, spec), cfg.prep_threads, (0, 0, ph)))
        jobs.append(Job(pid, "prep", dict(ph=ph), [], prep_mem_gb(cfg, spec), cfg.prep_threads, (0, 1, ph)))
        for res in cfg.active_resolutions():
            mem = 2 * job_mem_gb(cfg, spec, res)               # both forward models' datasets at once
            thr = job_threads(cfg, mem)
            jobs.append(Job(job_id("cmp", ph=ph, res=res), "cmp", dict(ph=ph, res=res), [cid, pid], mem, 1,
                            (1, -res, ph, 0, STAGE_ORDER["cmp"])))
            for snr in cfg.active_snrs():
                for dte in cfg.dtes_ms:
                    for rep in cfg.active_reps():
                        jobs.append(Job(job_id("ctl", ph=ph, res=res, snr=snr, dte=dte, rep=rep), "ctl",
                                        dict(ph=ph, res=res, snr=snr, dte=dte, rep=rep), [cid, pid], mem, thr,
                                        (1, -res, ph, snr, STAGE_ORDER["ctl"], dte, rep)))
    return jobs


def execute(cfg, out, job: Job, log=print):
    """Run one job in this process (called by `python -m qsmsim.exp1 job`)."""
    from . import jobs as J
    from . import prepare as P
    a = job.args
    if cfg.control:
        from . import control as CT
        if job.kind == "cache":
            return CT.ensure_v1_cache(cfg, out, a["ph"], log)
        if job.kind == "prep":
            info = CT.prepare_v2(cfg, out, a["ph"], log)
            return dict(total_s=info["total_s"], hashes=info["hashes"])
        if job.kind == "cmp":
            return CT.run_compare(cfg, out, a["ph"], a["res"], log)
        if job.kind == "ctl":
            return CT.run_control(cfg, out, a["ph"], a["res"], a["snr"], a["dte"], a["rep"], log)
        raise ValueError(job.kind)
    if job.kind == "prep":
        info = P.prepare(cfg, a["ph"], out, log)
        return dict(total_s=info["total_s"], hashes=info["hashes"])
    if job.kind == "tune1":
        r = J.run_tune(cfg, out, a["ph"], a["res"], a["snr"], 1, log=log)
        return dict(best={k: v["best"] for k, v in r["methods"].items()}, source_sha256=r["source_sha256"])
    if job.kind == "tune2":
        r = J.run_tune(cfg, out, a["ph"], a["res"], a["snr"], 2, a["dte"], a["n"], log=log)
        return dict(best={k: v["best"] for k, v in r["methods"].items()}, source_sha256=r["source_sha256"])
    if job.kind == "run":
        return J.run_production(cfg, out, a["ph"], a["res"], a["snr"], a["dte"], a["rep"], log=log)
    if job.kind in ("cache", "resweep"):
        from . import resweep as RS
        if job.kind == "cache":
            return RS.ensure_cache(cfg, out, a["ph"], log)
        return RS.run_resweep(cfg, out, a["ph"], a["res"], a["snr"], a["dte"], a["rep"], log=log)
    raise ValueError(job.kind)


class Runner:
    def __init__(self, cfg, out, workers, mem_gb, stages=None, retry_failed=False, poll_s=0.5):
        self.cfg, self.out = cfg, Path(out)
        self.workers, self.mem_gb = workers, mem_gb
        self.state = self.out / "state"
        self.logs = self.out / "logs"
        self.state.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.jobs = {j.id: j for j in build_jobs(cfg)}
        self.stages = set(stages or STAGE_ORDER)
        self.poll_s = poll_s
        if retry_failed:
            for f in self.state.glob("*.failed"):
                f.unlink()
        self.running = {}
        self._stop = False

    def _safe(self, jid):
        return jid.replace(":", "__")

    def status_of(self, jid):
        s = self._safe(jid)
        if (self.state / f"{s}.done").exists():
            return "done"
        if (self.state / f"{s}.failed").exists():
            return "failed"
        return "pending"

    def summary(self):
        counts = {}
        for j in self.jobs.values():
            st = self.status_of(j.id)
            counts.setdefault(j.kind, {}).setdefault(st, 0)
            counts[j.kind][st] += 1
        return counts

    def eta(self):
        """Time estimate from the finished jobs (Experiments_v2.md 6.3): each pending job takes the mean time of the
        finished jobs of its kind, phantom and resolution, or of its kind and phantom at another resolution scaled by
        voxel count. Hours are serial (one GPU lane runs the reconstruction jobs one at a time)."""
        done, pend = {}, []
        for j in self.jobs.values():
            st = self.status_of(j.id)
            if st == "done":
                try:
                    s = float(json.loads((self.state / f"{self._safe(j.id)}.done").read_text())["seconds"])
                except (OSError, ValueError, KeyError):
                    continue
                done.setdefault((j.kind, j.args.get("ph"), j.args.get("res")), []).append(s)
            elif st == "pending":
                pend.append(j)
        mean = {k: sum(v) / len(v) for k, v in done.items()}
        left, unknown = 0.0, {}
        for j in pend:
            k = (j.kind, j.args.get("ph"), j.args.get("res"))
            if k in mean:
                left += mean[k]
                continue
            other = [(r, m) for (kd, p, r), m in mean.items() if kd == k[0] and p == k[1] and r and k[2]]
            if other:
                r, m = min(other, key=lambda t: abs(t[0] - k[2]))
                left += m * (r / k[2]) ** 3
            else:
                unknown[j.kind] = unknown.get(j.kind, 0) + 1
        spent = sum(sum(v) for v in done.values())
        lines = [f"finished {sum(len(v) for v in done.values())} jobs, {spent / 3600:.1f} h of job time; "
                 f"pending {len(pend)}: about {left / 3600:.1f} h more"
                 + (f" + {sum(unknown.values())} not yet measured ({', '.join(f'{k} {n}' for k, n in unknown.items())})"
                    if unknown else "")]
        for (kd, p, r), m in sorted(mean.items(), key=lambda t: tuple(str(x) for x in t[0])):
            lines.append(f"  {kd} {p or ''} {f'{r:g} mm' if r else ''}: {m:.0f} s mean over {len(done[(kd, p, r)])}")
        return lines

    def _launch(self, job):
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[k] = "1"                                 # BLAS single-threaded: reproducible reductions
        env["QSMSIM_THREADS"] = str(job.threads)         # scipy.fft workers
        # QSMSIM_LANE_BACKEND=cpu: a rehearsal of the lanes and pools on a machine without a GPU (tests, workstation)
        env["QSMSIM_BACKEND"] = os.environ.get("QSMSIM_LANE_BACKEND", "gpu") if job.gpu else "cpu"
        env["QSMSIM_EVAL_WORKERS"] = str(self.cfg.eval_workers if job.gpu and job.kind in GPU_KINDS else 0)
        if job.gpu:
            if job.kind in GPU_KINDS:
                env["QSMSIM_THREADS"] = "1"              # the job's own host FFTs; its other cores are the pool's
            if self.cfg.gpu_mem_gb > 0:
                env["QSMSIM_GPU_MEM_GB"] = f"{self.cfg.gpu_mem_gb / max(1, self.cfg.gpu_lanes):g}"
            env.setdefault("TMPDIR", str(self.out / ".tmp"))       # evaluator scratch (evalpool.py)
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        logf = open(self.logs / f"{self._safe(job.id)}.log", "w")
        p = subprocess.Popen([sys.executable, "-m", "qsmsim.exp1", "job", job.id, "--out", str(self.out)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.running[job.id] = (p, logf, time.time())

    def _handle_signal(self, signum, frame):
        self._stop = True

    def run(self, log=print):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        todo = [j for j in self.jobs.values() if j.kind in self.stages and self.status_of(j.id) == "pending"]
        todo.sort(key=lambda j: j.priority)
        n_total, n_done0 = len(todo), 0
        t0 = time.time()
        lanes = self.cfg.gpu_lanes
        log(f"runner: {n_total} jobs to run ({', '.join(f'{k}: {sum(1 for j in todo if j.kind == k)}' for k in STAGE_ORDER if any(j.kind == k for j in self.jobs.values()))}); "
            f"workers {self.workers}, memory budget {self.mem_gb:g} GB"
            + (f"; GPU lanes {lanes} ({sum(j.gpu for j in todo)} GPU jobs), evaluation pool {self.cfg.eval_workers} "
               f"per lane" if lanes > 0 else ""))
        finished = failed = skipped = 0
        while (todo or self.running) and not self._stop:
            # reap
            for jid, (p, logf, ts) in list(self.running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                logf.close()
                del self.running[jid]
                ok = rc == 0 and self.status_of(jid) == "done"
                if ok:
                    finished += 1
                else:
                    failed += 1
                    if self.status_of(jid) != "failed":
                        (self.state / f"{self._safe(jid)}.failed").write_text(json.dumps(dict(returncode=rc)))
                el = time.time() - t0
                left = len(todo) + len(self.running)
                log(f"[{el / 3600:6.2f} h] {'done  ' if ok else 'FAILED'} {jid} ({time.time() - ts:.0f}s)  "
                    f"finished {finished}, failed {failed}, skipped {skipped}, left {left}")
            # launch
            used_mem = sum(self.jobs[j].mem_gb for j in self.running)
            used_cpu = sum(self.jobs[j].threads for j in self.running)
            used_gpu = sum(self.jobs[j].gpu for j in self.running)
            todo_ids = {j.id for j in todo}
            for job in list(todo):
                st = [self.status_of(d) for d in job.deps]
                if any(s == "failed" for s in st):
                    todo.remove(job)
                    skipped += 1
                    log(f"skip {job.id}: a dependency failed")
                    continue
                unsched = [d for d, s in zip(job.deps, st)
                           if s == "pending" and d not in todo_ids and d not in self.running]
                if unsched:
                    todo.remove(job)
                    skipped += 1
                    log(f"skip {job.id}: dependency {unsched[0]} is not done and not scheduled (--stages?)")
                    continue
                if not all(s == "done" for s in st) or any(d in self.running for d in job.deps):
                    continue
                if job.gpu and used_gpu >= lanes:
                    continue
                fits = used_cpu + job.threads <= self.workers and used_mem + job.mem_gb <= self.mem_gb
                if fits or not self.running:
                    self._launch(job)
                    todo.remove(job)
                    used_mem += job.mem_gb
                    used_cpu += job.threads
                    used_gpu += job.gpu
            time.sleep(self.poll_s)
        if self._stop:
            log("runner: stopping - terminating running jobs (they will be re-run on restart)")
            for jid, (p, logf, ts) in self.running.items():
                os.killpg(p.pid, signal.SIGTERM)
            for jid, (p, logf, ts) in self.running.items():
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                logf.close()
            return 130
        log(f"runner: finished {finished}, failed {failed}, skipped {skipped} in {(time.time() - t0) / 3600:.2f} h")
        return 1 if failed or skipped else 0
