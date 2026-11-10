"""λ grids with an automatic edge rule (GPU_Plan.md, updated plan 2.5).

A scheme (Config.lam_scheme) sets, per method, the sweep range around the stage-1 λ of the cell (MEDI, TV, CF-L2,
TFI: factors of it) or in absolute terms (TKD thresholds), the step (quarter decades; TKD 0.025), and the edge rule:
when the optimum of a criterion (whole-brain nRMSE, lowest; vein detection AUC, highest) lands on an end of the swept
values, the sweep is extended by `edge_steps` steps in that direction and the criteria are checked again, up to
`edge_cap_decades` decades beyond the configured range (TKD: within `tkd_limits`). Each dataset applies the rule on
its own; every extension, and every edge still hit at the cap, is recorded with the sweep (`edge`) and listed by the
aggregate, so a capped edge is never silent.
"""
from __future__ import annotations

import math

CRITERIA = {"rmse": (lambda m: m.get("rmse"), min),
            "vein_auc": (lambda m: (m.get("veins") or {}).get("auc_core_vs_shell"), max)}


def scheme_v2():
    """The scheme agreed in the updated plan (2.5): ranges = the union of the stage-1 and resweep ranges, one decade
    beyond every edge the v1 runs hit; quarter-decade steps (TKD 0.025)."""
    return dict(step=10 ** 0.25,
                factors={"medi": [1e-3, 10.0], "tv": [1e-2, 10.0], "cfl2": [1e-3, 10 ** 0.5], "tfi": [1e-3, 10 ** 0.5]},
                tkd=[0.05, 0.6, 0.025], tkd_limits=[0.025, 1.0],
                edge_steps=2, edge_cap_decades=2.0, criteria=["rmse", "vein_auc"])


def _r(v):
    return float(f"{v:.4g}")


def _k_range(lo, hi, step):
    return range(int(math.floor(math.log(lo) / math.log(step) + 1e-6)), int(math.ceil(math.log(hi) / math.log(step) - 1e-6)) + 1)


def grid(scheme, inv, anchor):
    """Initial values: anchor x step^k over the factor range (the anchor itself exactly), or the TKD thresholds."""
    if inv == "tkd":
        lo, hi, st = scheme["tkd"]
        return [_r(lo + i * st) for i in range(int(round((hi - lo) / st)) + 1)]
    lo, hi = scheme["factors"][inv]
    return sorted({_r(anchor * scheme["step"] ** k) for k in _k_range(lo, hi, scheme["step"])})


def limits(scheme, inv, anchor):
    """Hard ends of the edge rule."""
    if inv == "tkd":
        lo, hi = scheme["tkd_limits"]
        return lo - 1e-9, hi + 1e-9
    lo, hi = scheme["factors"][inv]
    c = 10 ** scheme["edge_cap_decades"]
    return anchor * lo / c * (1 - 1e-3), anchor * hi * c * (1 + 1e-3)        # values carry 4 significant figures


def beyond(scheme, inv, anchor, values, direction, lim):
    """Up to edge_steps new values past the low (-1) or high (+1) end, inside the limits (on the anchor's lattice:
    anchor x step^k, TKD lo + j x 0.025, so values never drift by rounding)."""
    end = min(values) if direction < 0 else max(values)
    out = []
    for i in range(1, scheme["edge_steps"] + 1):
        if inv == "tkd":
            lo, _, st = scheme["tkd"]
            v = _r(lo + (round((end - lo) / st) + direction * i) * st)
        else:
            st = scheme["step"]
            v = _r(anchor * st ** (round(math.log(end / anchor) / math.log(st)) + direction * i))
        if lim[0] <= v <= lim[1]:
            out.append(v)
    return out


def optima(rows, criteria):
    """{criterion: optimal value} over rows (dicts with 'value' and 'metrics'); criteria without data are left out."""
    out = {}
    for c in criteria:
        get, pick = CRITERIA[c]
        have = [(get(r["metrics"]), r["value"]) for r in rows]
        have = [(m, v) for m, v in have if m is not None and math.isfinite(m)]
        if have:
            out[c] = pick(have, key=lambda t: t[0])[1]
    return out


def sweep(evaluate_many, scheme, inv, anchor, values, lim, criteria, rows=None):
    """Evaluate every value (evaluate_many([v, ...]) -> row dicts with 'value' and 'metrics', in order; a batch, so
    a GPU job can overlap the inversions with a CPU evaluation pool), then apply the edge rule.
    Returns (rows sorted by value, edge record)."""
    rows = list(rows or [])
    have = {r["value"] for r in rows}
    ext = {-1: 0, 1: 0}
    todo = [v for v in values if v not in have]
    while True:
        if todo:
            rows += evaluate_many(todo)
            have.update(todo)
        vals = sorted(have)
        opt = optima(rows, criteria)
        todo = []
        capped = {-1: False, 1: False}
        for d, end in ((-1, vals[0]), (1, vals[-1])):
            if not any(o == end for o in opt.values()):
                continue
            new = [v for v in beyond(scheme, inv, anchor, vals, d, lim) if v not in have]
            if new:
                todo += new
                ext[d] += len(new)
            else:
                capped[d] = True
        if not todo:
            break
    rows.sort(key=lambda r: r["value"])
    vals = [r["value"] for r in rows]
    rec = dict(optima=opt, at_low={c: o == vals[0] for c, o in opt.items()},
               at_high={c: o == vals[-1] for c, o in opt.items()}, extended_low=ext[-1], extended_high=ext[1],
               capped_low=capped[-1], capped_high=capped[1], range=[vals[0], vals[-1]], n_values=len(vals))
    return rows, rec


def edge_row(key, method, edge):
    """One aggregate row per dataset x method with its edge record."""
    row = dict(**key, method=method, range_lo=edge["range"][0], range_hi=edge["range"][1], n_values=edge["n_values"],
               extended_low=edge["extended_low"], extended_high=edge["extended_high"],
               capped_low=edge["capped_low"], capped_high=edge["capped_high"])
    for c, v in edge["optima"].items():
        row[f"opt_{c}"] = v
        row[f"{c}_at_edge"] = bool(edge["at_low"].get(c) or edge["at_high"].get(c))
    return row


def summary_lines(rows, what):
    """Markdown lines: extensions and edges still hit (at the cap), per method."""
    if not rows:
        return []
    L = [f"- λ edge rule ({what}): {len(rows)} sweeps; per method: extended / still at an edge (capped)"]
    for m in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == m]
        ext = sum(1 for r in rr if r["extended_low"] or r["extended_high"])
        cap = [r for r in rr if r["capped_low"] or r["capped_high"]]
        L.append(f"  - {m}: {ext} / {len(cap)} of {len(rr)}")
        L += ["    - capped: " + ", ".join(f"{k}={r[k]}" for k in ("phantom", "res", "snr", "dte", "n", "rep") if k in r)
              + f" ({'low' if r['capped_low'] else 'high'}, range {r['range_lo']:g}-{r['range_hi']:g})" for r in cap[:20]]
    return L
