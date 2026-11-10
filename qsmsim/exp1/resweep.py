"""Extended λ sweep on a finished production run (Experiment_1_results.md, layer 2v, next step 1).

The production tuning grids stop about ½ decade below the whole-brain optimum, but the vein-optimal
λ is about 1 decade lighter. This stage reconstructs every production dataset (all protocols, all
replicates) again on a fixed grid around the stage-1 λ of its phantom x resolution x SNR:

  MEDI / TV / CF-L2 (and TFI, optionally): λ_stage1 x factors (default 1, 10^-0.5, 0.1, 10^-1.5, 0.01)
  TKD:                                     the full stage-1 threshold grid

on the reference field (oracle on P1/P2, the fitted field on P3), so a vein-optimal λ can be chosen per
protocol and the contrasts of Aims 1-3 get confidence intervals over replicates.

It writes to its own output directory (config, state, logs, sweeps/, aggregate/) and only reads the
production directory (`src`): prep/, cache/, tune/stage1/ and runs/**/metrics.json. Each dataset is
regenerated from its seeds and its SHA-256 compared with the one production recorded (`sha_match`).
The background-removal steps run as in production, so the evaluation mask is identical.
If src/cache/ was deleted, a `cache` job regenerates it into <out>/regen/ and checks every hash
against src/prep/<P>/prep.json first.

Experiment 2 (exp2.py) runs the same sweep with the run itself as the source (src = "."): there is no
production record, so the SHA check is replaced by the evaluation-mask check against the paired 3T dataset.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import seeds
from .io import AtomicDir, proto_tag, res_tag, save_nifti_cropped, snr_tag, write_json

DEFAULT_FACTORS = [1.0, 10 ** -0.5, 0.1, 10 ** -1.5, 0.01]
DEFAULT_METHODS = ["medi", "tv", "cfl2", "tkd"]
REL_TOL = 1e-6


def make_config(src_cfg, out, src, factors=None, methods=None, tfi_reps=(), keep_maps=False, reps=None):
    """methods None = DEFAULT_METHODS, [] = none (a TFI-only sweep with tfi_reps); reps restricts the
    replicates (None = all). TFI is chosen per replicate with tfi_reps, never through methods."""
    methods = list(DEFAULT_METHODS if methods is None else methods)
    if "tfi" in methods:
        raise ValueError("select TFI with tfi_reps (--tfi-reps), not in methods")
    if not methods and not tfi_reps:
        raise ValueError("nothing to sweep: no methods and no TFI replicates")
    cfg = src_cfg
    cfg.name = "resweep"
    if reps is not None:
        cfg.only_reps = [int(r) for r in reps]
    cfg.resweep = dict(src=os.path.relpath(Path(src).resolve(), Path(out).resolve()),
                       factors=[float(f) for f in (factors or DEFAULT_FACTORS)],
                       methods=methods, tfi_reps=[int(r) for r in tfi_reps], keep_maps=bool(keep_maps))
    return cfg


def src_dir(cfg, out):
    return (Path(out) / cfg.resweep["src"]).resolve()


def regen_dir(out):
    return Path(out) / "regen"


# ---------------------------------------------------------------------- cache
def _cache_ok(src, ph):
    """True when every noiseless signal of src/cache/<ph> exists and matches prep.json."""
    want = json.loads((src / "prep" / ph / "prep.json").read_text())["hashes"]
    for key, h in want.items():
        if "/S0_" not in key:
            continue
        f = src / "cache" / ph / (key + ".npy")
        if not f.exists() or seeds.sha256(np.load(f)) != h:
            return False
    return True


def ensure_cache(cfg, out, ph, log=print, src=None, regen=None):
    """Use src/cache when intact; otherwise regenerate it into <out>/regen and check it. src / regen: the run and
    regeneration directory when they are not this resweep's (the control run's v1 arm, control.py)."""
    src = Path(src) if src is not None else src_dir(cfg, out)
    regen = Path(regen) if regen is not None else regen_dir(out)
    if _cache_ok(src, ph):
        log(f"cache {ph}: production cache intact")
        return dict(cache_root=str(src))
    from .prepare import prepare
    log(f"cache {ph}: production cache missing or changed - regenerating into {regen}")
    scratch = regen / f"tmp_{ph}"
    if scratch.exists():
        shutil.rmtree(scratch)
    info = prepare(cfg, ph, scratch, log=log)
    want = json.loads((src / "prep" / ph / "prep.json").read_text())["hashes"]
    worst, inexact = 0.0, []
    for key, h in want.items():
        if info["hashes"].get(key) == h:
            continue
        if "/S0_" not in key:
            raise RuntimeError(f"regenerated {ph} {key} differs from production (truth/vein data must be exact)")
        inexact.append(key)
        ref = src / "cache" / ph / (key + ".npy")
        if ref.exists():
            a, b = np.load(ref), np.load(scratch / "cache" / ph / (key + ".npy"))
            worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)))
        else:
            # nothing to compare against: each dataset's sha_match (vs production) then shows the difference
            log(f"cache {ph}: {key} is not bit-identical to production and the production array is gone")
    if worst > REL_TOL:
        raise RuntimeError(f"regenerated {ph} noiseless signals differ from production (max rel {worst:.2e})")
    final = regen / "cache" / ph
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(scratch / "cache" / ph, final)
    shutil.rmtree(scratch)
    return dict(cache_root=str(regen), max_rel_diff=worst, not_bit_identical=inexact)


def cache_root(cfg, out, ph):
    r = regen_dir(out)
    return r if (r / "cache" / ph).exists() else src_dir(cfg, out)


# ---------------------------------------------------------------------- sweep
def sweep_dir(out, ph, res, snr, dte, n, rep):
    return Path(out) / "sweeps" / ph / res_tag(res) / snr_tag(snr) / proto_tag(dte, n) / f"rep{rep}"


def grid(cfg, inv, stage1_value):
    """Initial sweep values: the λ scheme's range around the stage-1 value (lamgrid.py), or the v1 factors."""
    if cfg.lam_scheme:
        from . import lamgrid as LG
        return LG.grid(cfg.lam_scheme, inv, None if inv == "tkd" else stage1_value)
    if inv == "tkd":
        return list(cfg.full_grids["tkd"])
    return [float(f"{stage1_value * f:.4g}") for f in cfg.resweep["factors"]]


def run_resweep(cfg, out, ph, res, snr, dte, rep, log=print):
    """All echo counts of one (phantom, res, SNR, dTE, replicate): the extended grid per inversion."""
    from . import jobs as J
    src = src_dir(cfg, out)
    rs = cfg.resweep
    prep = J.Prep(cfg, src, ph, res, cache_root=cache_root(cfg, out, ph))
    s1 = json.loads(J.stage1_path(src, ph, res, snr).read_text())["methods"]
    methods = list(rs["methods"]) + (["tfi"] if rep in rs["tfi_reps"] else [])
    t0 = time.time()
    S, sha, sigma = J.noisy_signal(cfg, prep, dte, snr, rep)
    summary = []
    for n in cfg.counts(dte):
        if cfg.exp2:                                   # no production record: pair with the 3T reference
            from . import exp2 as X
            prod = X.partner(cfg, out, ph, res, snr, dte, n, rep)
            s3t, k = X.level(cfg, snr)
        else:
            prod = json.loads((J.run_dir(src, ph, res, snr, dte, n, rep) / "metrics.json").read_text())
        t = time.time()
        ds = J.Dataset(cfg, prep, S, dte, n)
        t_ds = time.time() - t
        bg = ds.ref_background()
        rec = dict(phantom=ph, res=res, snr=snr, dte=dte, n=n, rep=rep, source_sha256=sha,
                   sha_match=None if cfg.exp2 else sha == prod["source_sha256"], sigma=sigma, field=ds.field_metrics,
                   eval_mask_voxels=int(ds.eval_mask.sum()),
                   eval_mask_match=None if prod is None else int(ds.eval_mask.sum()) == int(prod["eval_mask_voxels"]),
                   background=bg, tuned={} if cfg.exp2 else prod["lambdas"], methods={}, dataset_seconds=t_ds,
                   bg_info=ds.bg_info)
        if cfg.exp2:
            rec.update(b0=cfg.b0_tesla, s3t=s3t, k=k, ref_sigma=None if prod is None else prod["sigma"])
        with AtomicDir(sweep_dir(out, ph, res, snr, dte, n, rep)) as tmp:
            per_vein = {}
            for inv in methods:
                b1 = s1[inv]["best"]
                extra = ({"outer": min(cfg.sweep_tfi_outer, ds.simcfg.method_params("tfi")["outer"])}
                         if inv == "tfi" else None)
                def many(values):
                    # inversions one after another; each map scored as it comes (CPU pool on a GPU job)
                    pend = []
                    for v in values:
                        t = time.time()
                        chi, info = ds.invert(bg, inv, v, extra)
                        tr = time.time() - t
                        tag = f"{J.chain_name(bg, inv)}_{J.PARAM_KEY[inv]}{v:g}"
                        if rs["keep_maps"]:
                            save_nifti_cropped(tmp, {f"chi_{tag}": chi}, prep.box, prep.res)
                        pend.append((v, tag, info, tr, ds.score(chi)))
                    rows = []
                    for v, tag, info, tr, fut in pend:
                        m, per, te = fut.result()
                        if rs["keep_maps"] and per is not None:
                            per_vein.update({f"{tag}/{k}": a for k, a in per.items()})
                        rows.append(dict(value=v, factor=v / b1, metrics=m, info=J._jsonable_info(info),
                                         seconds=tr, eval_seconds=te))
                    return rows
                edge = None
                if cfg.lam_scheme:                     # the edge rule (lamgrid.py)
                    from . import lamgrid as LG
                    anchor = None if inv == "tkd" else b1
                    rows, edge = LG.sweep(many, cfg.lam_scheme, inv, anchor, grid(cfg, inv, b1),
                                          LG.limits(cfg.lam_scheme, inv, anchor), cfg.lam_scheme["criteria"])
                else:
                    rows = many(grid(cfg, inv, b1))
                rec["methods"][inv] = dict(param=J.PARAM_KEY[inv], stage1_value=b1,
                                           tuned_value=None if cfg.exp2 else prod["lambdas"].get(inv), rows=rows,
                                           **({"edge": edge} if edge else {}))
            ds.release()
            if per_vein:
                np.savez_compressed(tmp / "veins.npz", **per_vein)
            rec["seconds"] = time.time() - t0
            write_json(tmp / "metrics.json", rec)
        best = {inv: min(m["rows"], key=lambda r: r["metrics"]["region"]["dmv_territory"]["nrmse"])["value"]
                for inv, m in rec["methods"].items()}
        summary.append(dict(n=n, sha_match=rec["sha_match"], eval_mask_match=rec["eval_mask_match"]))
        check = ({None: "no 3T partner", True: "eval mask = 3T", False: "EVAL MASK DIFFERS FROM 3T"}[rec["eval_mask_match"]]
                 if cfg.exp2 else f"sha {'ok' if rec['sha_match'] else 'MISMATCH'}")
        log(f"resweep {ph} {res_tag(res)} {snr_tag(snr)} {proto_tag(dte, n)} rep{rep}: {check}; best DMV-nRMSE λ "
            + " ".join(f"{k}={v:g}" for k, v in best.items()))
    return dict(source_sha256=sha, summary=summary, seconds=time.time() - t0)


# ---------------------------------------------------------------------- aggregate
CHECK_KEYS = ("rmse", "hfen", "ssim")


def crosscheck(cfg, out, d):
    """Compare one dataset's sweep with what the source run computed at the same λ: the production
    chain at its tuned λ (every replicate) and the stage-1/2 tuning sweep (replicate 0).
    Returns (points compared, max abs difference over CHECK_KEYS) or None if the source is absent."""
    from . import jobs as J
    src = src_dir(cfg, out)
    args = (d["phantom"], d["res"], d["snr"], d["dte"], d["n"])
    pairs = []
    prod = J.run_dir(src, *args, d["rep"]) / "metrics.json"
    if prod.exists():
        chains = json.loads(prod.read_text())["chains"]
        for inv, md in d["methods"].items():
            c = chains.get(J.chain_name(d["background"], inv))
            for r in md["rows"]:
                if c is not None and md["tuned_value"] is not None and np.isclose(r["value"], md["tuned_value"], rtol=1e-9, atol=0):
                    # TFI sweeps cap the outer iterations: comparable only where both stopped at the same count
                    if inv == "tfi" and c.get("info", {}).get("outer") != r["info"].get("outer"):
                        continue
                    pairs.append((r["metrics"], c))
    if d["rep"] == 0:
        ref = (float(cfg.tune_ref[0]), int(cfg.tune_ref[1]))
        tp = (J.stage1_path(src, *args[:3]) if (float(d["dte"]), int(d["n"])) == ref
              else J.stage2_path(src, *args))
        if tp.exists():
            tm = json.loads(tp.read_text())["methods"]
            for inv, md in d["methods"].items():
                if inv == "tfi" or inv not in tm:            # TFI sweeps may differ in outer iterations
                    continue
                by_v = {float(r["value"]): r["metrics"] for r in tm[inv]["rows"]}
                for r in md["rows"]:
                    m = next((v for k, v in by_v.items() if np.isclose(k, r["value"], rtol=1e-9, atol=0)), None)
                    if m is not None:
                        pairs.append((r["metrics"], m))
    if not pairs:
        return None
    diff = max(abs(a[k] - b[k]) for a, b in pairs for k in CHECK_KEYS)
    return len(pairs), float(diff)


def aggregate(cfg, out, flatten, write, log=print):
    out = Path(out)
    adir = out / "aggregate"
    adir.mkdir(exist_ok=True)
    rows, manifest, edges = [], [], []
    n_cmp, worst = 0, 0.0
    for f in sorted((out / "sweeps").glob("*/*/*/*/rep*/metrics.json")):
        d = json.loads(f.read_text())
        key = {k: d[k] for k in ("phantom", "res", "snr", "dte", "n", "rep")}
        if cfg.exp2:
            key.update(b0=d["b0"], s3t=d["s3t"], k=d["k"])
        cc = crosscheck(cfg, out, d)
        if cc:
            n_cmp, worst = n_cmp + cc[0], max(worst, cc[1])
        manifest.append(dict(**key, source_sha256=d["source_sha256"], sha_match=d["sha_match"],
                             eval_mask_match=d["eval_mask_match"], sigma=d["sigma"],
                             **({"ref_sigma": d["ref_sigma"]} if cfg.exp2 else {}),
                             tuned=json.dumps(d["tuned"]), **flatten(d["field"], "field."),
                             **flatten(d.get("bg_info", {}), "bg."),
                             seconds=d["seconds"], crosscheck_points=cc[0] if cc else 0,
                             crosscheck_max_diff=cc[1] if cc else None, path=str(f.parent.relative_to(out))))
        for inv, md in d["methods"].items():
            if md.get("edge"):
                from .lamgrid import edge_row
                edges.append(edge_row(key, inv, md["edge"]))
            for r in md["rows"]:
                row = dict(**key, method=inv, background=d["background"], param=md["param"], value=r["value"],
                           factor=r["factor"], stage1_value=md["stage1_value"], tuned_value=md["tuned_value"],
                           is_stage1=bool(np.isclose(r["value"], md["stage1_value"], rtol=1e-3, atol=0)),
                           is_tuned=md["tuned_value"] is not None
                           and bool(np.isclose(r["value"], md["tuned_value"], rtol=1e-3, atol=0)),
                           seconds=r["seconds"], eval_seconds=r["eval_seconds"])
                row.update(flatten(r["metrics"]))
                rows.append(row)
    write(adir / "resweep.csv", rows)
    write(adir / "resweep_manifest.csv", manifest)
    if edges:
        write(adir / "edges.csv", edges)
    from .lamgrid import summary_lines
    edge_lines = summary_lines(edges, "sweeps")
    states = {}
    for s in (out / "state").glob("*"):
        states[s.suffix] = states.get(s.suffix, 0) + 1
    bad = [m for m in manifest if m["sha_match"] is False or m["eval_mask_match"] is False]
    if cfg.exp2:
        from .aggregate import tuning_rows
        write(adir / "tuning.csv", tuning_rows(out))
        return _exp2_summary(cfg, out, adir, rows, manifest, bad, n_cmp, worst, states, log, edge_lines)
    L = ["# Experiment 1 extended λ sweep (resweep)", "",
         f"- source run: `{cfg.resweep['src']}`; factors {cfg.resweep['factors']}; methods {cfg.resweep['methods']}; "
         f"TFI on replicates {cfg.resweep['tfi_reps']}",
         f"- datasets: {len(manifest)}; sweep points: {len(rows)}; job states: {states}",
         f"- datasets whose source SHA-256 or evaluation mask differ from production: {len(bad)}",
         f"- cross-check against the source run at shared λ (production tuned λ, all replicates; tuning sweeps, "
         f"replicate 0): {n_cmp} points, max |difference| in {'/'.join(CHECK_KEYS)} = {worst:.3g}"
         + ("" if n_cmp else " (nothing comparable: TFI-only sweep, or the source run is not mounted)")]
    L += [f"  - {m['path']}" for m in bad[:20]]
    L += edge_lines
    (adir / "summary.md").write_text("\n".join(L) + "\n")
    log(f"aggregate (resweep): {len(manifest)} datasets, {len(rows)} sweep points, "
        f"{len(bad)} source mismatches, cross-check {n_cmp} points max |diff| {worst:.3g} -> {adir}")
    return adir


def _exp2_summary(cfg, out, adir, rows, manifest, bad, n_cmp, worst, states, log, edge_lines=()):
    X = cfg.exp2
    unpaired = sum(1 for m in manifest if m["eval_mask_match"] is None)
    sig = [abs(m["sigma"] * m["k"] / m["ref_sigma"] - 1) for m in manifest if m.get("ref_sigma")]
    L = [f"# Experiment 2: {cfg.b0_tesla:g} T against the {X['ref_b0']:g} T reference", "",
         f"- reference run: `{X['ref']}`; 3T SNR anchors {X['s3t']}; SNR gains k {X['gains']}; "
         f"R2* x {cfg.r2s_scale:.4g}; DMV blood R2* {cfg.veins.r2s:g} 1/s",
         f"- λ factors {cfg.resweep['factors']} around each stage-1 λ; methods {cfg.resweep['methods']}; "
         f"TFI on replicates {cfg.resweep['tfi_reps']}",
         f"- datasets: {len(manifest)}; sweep points: {len(rows)}; job states: {states}",
         f"- evaluation mask differs from the paired 3T dataset: {len(bad)}; no 3T partner: {unpaired}",
         f"- sigma x k / 3T sigma - 1: max {max(sig):.2g} over {len(sig)} pairs" if sig else "- sigma: no 3T partners",
         f"- cross-check against stage-1 tuning at shared λ (replicate 0, 5 @ 5.4 ms): {n_cmp} points, "
         f"max |difference| in {'/'.join(CHECK_KEYS)} = {worst:.3g}"]
    L += [f"  - {m['path']}" for m in bad[:20]]
    L += list(edge_lines)
    (adir / "summary.md").write_text("\n".join(L) + "\n")
    log(f"aggregate (exp2): {len(manifest)} datasets, {len(rows)} sweep points, {len(bad)} eval-mask mismatches, "
        f"{unpaired} unpaired, cross-check {n_cmp} points max |diff| {worst:.3g} -> {adir}")
    return adir
