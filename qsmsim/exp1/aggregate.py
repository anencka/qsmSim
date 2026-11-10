"""Collect every metrics.json / tuning result into flat CSV tables:

  aggregate/runs.csv     one row per dataset x chain (phantom, res, snr, dte, n, rep, chain, metrics...)
  aggregate/tuning.csv   one row per sweep point (stage, protocol, method, value, metrics...)
  aggregate/manifest.csv one row per dataset: source SHA-256, sigma, seeds, lambdas, timings
  aggregate/summary.md   short overview (counts, failures, median nRMSE per chain and resolution)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            continue
        else:
            out[key] = v
    return out


def _write(path, rows):
    cols = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def tuning_rows(out):
    """One row per tuning sweep point (stage 1 and 2)."""
    tune = []
    for f in sorted((Path(out) / "tune").glob("stage*/**/*.json")):
        d = json.loads(f.read_text())
        for meth, md in d["methods"].items():
            for r in md["rows"]:
                row = dict(stage=d["stage"], phantom=d["phantom"], res=d["res"], snr=d["snr"], dte=d["dte"], n=d["n"],
                           method=meth, param=md["param"], value=r["value"], best=md["best"],
                           stage1_value=md.get("stage1_value"), is_best=r["value"] == md["best"],
                           at_grid_edge=md["at_grid_edge"], seconds=r["seconds"],
                           edge_extended=(md["edge"]["extended_low"] + md["edge"]["extended_high"]) if md.get("edge") else None,
                           edge_capped=(md["edge"]["capped_low"] or md["edge"]["capped_high"]) if md.get("edge") else None)
                row.update(flatten(r["metrics"]))
                tune.append(row)
    return tune


def aggregate(cfg, out, log=print):
    out = Path(out)
    if cfg.control:
        from .control import aggregate as control_aggregate
        return control_aggregate(cfg, out, flatten, _write, log)
    if cfg.resweep:
        from .resweep import aggregate as resweep_aggregate
        return resweep_aggregate(cfg, out, flatten, _write, log)
    adir = out / "aggregate"
    adir.mkdir(exist_ok=True)
    runs, manifest = [], []
    for m in sorted((out / "runs").glob("*/*/*/*/rep*/metrics.json")):
        d = json.loads(m.read_text())
        key = {k: d[k] for k in ("phantom", "res", "snr", "dte", "n", "rep")}
        manifest.append(dict(**key, source_sha256=d["source_sha256"], sigma=d["sigma"],
                             noise_seed_keys=json.dumps(d.get("seed_keys", {})), lambdas=json.dumps(d["lambdas"]),
                             eval_mask_voxels=d["eval_mask_voxels"], **flatten(d["field"], "field."),
                             **flatten(d.get("bg_info", {}), "bg."),
                             **flatten(d.get("seconds", {}), "seconds."), path=str(m.parent.relative_to(out))))
        for chain, c in d["chains"].items():
            row = dict(**key, chain=chain)
            row.update(flatten({k: v for k, v in c.items() if k != "info"}))
            runs.append(row)
    tune = tuning_rows(out)
    from .lamgrid import edge_row, summary_lines
    edges = []
    for f in sorted((out / "tune" / "stage1").glob("**/*.json")):
        d = json.loads(f.read_text())
        for meth, md in d["methods"].items():
            if md.get("edge"):
                edges.append(edge_row({k: d[k] for k in ("phantom", "res", "snr")}, meth, md["edge"]))
    _write(adir / "runs.csv", runs)
    _write(adir / "manifest.csv", manifest)
    _write(adir / "tuning.csv", tune)
    # summary
    med = defaultdict(list)
    for r in runs:
        med[(r["phantom"], r["res"], r["chain"])].append(r["rmse"])
    states = defaultdict(int)
    for f in (out / "state").glob("*"):
        states[f.suffix] += 1
    L = ["# Experiment 1 aggregate", "",
         f"- datasets: {len(manifest)}; chain results: {len(runs)}; tuning points: {len(tune)}",
         f"- job states: {dict(states)}", *summary_lines(edges, "stage 1"), "",
         "## Median whole-brain nRMSE (%) by phantom, resolution, chain", "",
         "| phantom | res | chain | n | median nRMSE |", "|---|---|---|---:|---:|"]
    for (ph, res, ch), v in sorted(med.items()):
        L.append(f"| {ph} | {res} | {ch} | {len(v)} | {np.median(v):.1f} |")
    (adir / "summary.md").write_text("\n".join(L) + "\n")
    log(f"aggregate: {len(manifest)} datasets, {len(runs)} chain rows, {len(tune)} tuning rows -> {adir}")
    return adir
