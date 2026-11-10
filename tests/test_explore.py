"""qsmsim.explore: spec handling, per-field configs, selection and a small end-to-end run."""
import csv
import json
import math
from itertools import product
from pathlib import Path

import pytest

from qsmsim import explore as E

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_specs_parse_and_build_configs():
    files = sorted(EXAMPLES.glob("*.toml"))
    assert files
    for f in files:
        s = E.load_spec(f)
        assert E.datasets(s)
        for b0 in s["b0_tesla"]:
            c = E.field_config(s, b0, s["te1_ms"][0])
            assert c.b0_tesla == b0 and c.remove_global_cycles
            if c.forward == "v2":
                c.check_v2()


def test_field_config_scales_relaxation_and_keys_prep():
    s = E.normalize(dict(base="toy", b0_tesla=[3, 7], protocols=[{"dte_ms": 2.4, "echoes": 11}]))
    c3, c7 = E.field_config(s, 3.0, 4.0), E.field_config(s, 7.0, 4.0)
    assert c7.r2s_scale == pytest.approx(7 / 3 * c3.r2s_scale)
    assert c7.veins.r2s == pytest.approx(7 / 3 * c3.veins.r2s)
    assert c3.n_echo_sim == 11 and c3.dtes_ms == [2.4]
    assert E.prep_key(c3, "P1") != E.prep_key(c7, "P1")
    # recon-only settings do not change the phantom
    s2 = E.normalize(dict(s, inversions=["tv"], snr=[5], config={"recon_params": {"tv": {"maxiter": 3}}}))
    assert E.prep_key(E.field_config(s2, 3.0, 4.0), "P1") == E.prep_key(c3, "P1")


def test_sourced_tissues_follow_the_field():
    from qsmsim.phantom import blood_r2s
    s = E.normalize(dict(base="exp1_v2", phantoms=["P3"], b0_tesla=[7]))
    c = E.field_config(s, 7.0, 4.0)
    assert c.tissues == "sourced" and c.veins.r2s == pytest.approx(blood_r2s(7.0))


def test_variants_are_a_sweep_dimension():
    s = E.normalize(dict(base="toy", snr=[20, 40], variants=[
        {"name": "thin", "config": {"veins": {"diam_start_mm": [0.1, 0.25]}}},
        {"name": "tilt", "config": {"b0_dir": [0.0, 0.342, 0.94]}},
        {"name": "iter10", "config": {"recon_params": {"medi": {"maxiter": 10}}}}]))
    ds = E.datasets(s)
    assert len(ds) == 3 * 2 and {d["variant"] for d in ds} == {"thin", "tilt", "iter10"}
    assert len({d["id"] for d in ds}) == len(ds) and ds[0]["id"].startswith("thin_P1_")
    base = E.field_config(s, 3.0, 4.0)
    thin, tilt, it = (E.field_config(s, 3.0, 4.0, v) for v in ("thin", "tilt", "iter10"))
    assert thin.veins.diam_start_mm == (0.1, 0.25) and base.veins.diam_start_mm != (0.1, 0.25)
    assert tilt.b0_dir == (0.0, 0.342, 0.94)
    assert it.recon_params["medi"]["maxiter"] == 10 and it.recon_params["tv"] == base.recon_params["tv"]
    keys = {E.prep_key(c, "P1") for c in (base, thin, tilt)}
    assert len(keys) == 3                               # phantom settings: their own preparation
    assert E.prep_key(it, "P1") == E.prep_key(base, "P1")   # recon settings: shared
    assert "variant" not in E.datasets(E.normalize(dict(base="toy")))[0]
    with pytest.raises(ValueError):
        E.normalize(dict(variants=[{"name": "a b"}]))
    with pytest.raises(ValueError):
        E.normalize(dict(variants=[{"name": "a"}, {"name": "a"}]))


def test_knob_demos_parse():
    files = sorted((EXAMPLES / "knobs").glob("*.toml"))
    assert len(files) >= 15
    for f in files:
        assert f.read_text().startswith("# Knob: ") and "# plot: " in f.read_text()
        s = E.load_spec(f)
        assert E.datasets(s)
        for v, b0 in product([x["name"] for x in s["variants"]] or [None], s["b0_tesla"]):
            c = E.field_config(s, b0, s["te1_ms"][0], v)
            c.check_tissues()
            if c.forward == "v2":
                c.check_v2()


def test_bad_specs_are_rejected():
    with pytest.raises(ValueError):
        E.normalize(dict(inversions=["nope"]))
    with pytest.raises(ValueError):
        E.normalize(dict(protocols=[{"dte_ms": 2.4, "echoes": 14}]))
    with pytest.raises(ValueError):
        E.field_config(E.normalize(dict(config={"not_a_field": 1})), 3.0, 4.0)


def test_selection_direction_and_dprime():
    assert E._better("rmse") is min and E._better("dmv_territory.nrmse") is min
    assert E._better("vein_auc") is max and E._better("cnr_0.2-0.3mm") is max
    assert E._dprime(0.5) == pytest.approx(0.0, abs=1e-12)
    assert E._dprime(0.76) == pytest.approx(math.sqrt(2) * 0.7063, abs=1e-3)
    base = dict(phantom="P1", b0=3.0, te1=4.0, res=2.0, protocol="p", dte=5.4, n=5, snr=20.0, gain=1.0,
                chain="oracle_medi", background="oracle", inversion="medi")
    rows = [dict(base, rep=r, lam=lam, rmse=rm, vein_auc=auc)
            for r in (0, 1) for lam, rm, auc in ((1e-4, 50.0 + r, 0.70), (1e-3, 30.0 + r, 0.60))]
    out = {r["selected_by"]: r for r in E.summarize(rows, ["rmse", "vein_auc"])}
    assert out["rmse"]["lam"] == 1e-3 and out["rmse"]["rmse"] == pytest.approx(30.5)
    assert out["vein_auc"]["lam"] == 1e-4 and out["vein_auc"]["n_rep"] == 2
    assert out["rmse"]["lam_at_grid_edge"]


def test_end_to_end_small_run(tmp_path):
    spec = dict(name="t", base="toy", phantoms=["P1"], resolution_mm=[4.0], snr=[20],
                protocols=[{"dte_ms": 5.4, "echoes": 5}, {"dte_ms": 2.4, "echoes": 11}],
                backgrounds=["oracle"], inversions=["tkd", "medi"],
                config={"veins": {"n_per_hemi": 6, "n_collector_per_hemi": 1},
                        "recon_params": {"medi": {"maxiter": 10}}})
    spec["lambda"] = {"tkd": [0.19], "medi": [1e-4, 1e-3]}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(spec))
    out = tmp_path / "run"
    assert E.main(["run", str(f), "--out", str(out), "--threads", "2"]) == 0
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert len(rows) == 2 * 3                          # 2 protocols x (TKD + 2 MEDI values)
    assert all(r["source_sha256"] for r in rows)
    fe = {r["protocol"]: float(r["field_err_median_ppb"]) for r in rows}
    assert fe["11x2.4ms"] < fe["5x5.4ms"]              # more echoes in the window: a better field map
    # rerunning keeps the finished datasets and gives the same numbers
    before = (out / "results.csv").read_text()
    assert E.main(["run", str(f), "--out", str(out)]) == 0
    assert (out / "results.csv").read_text() == before
    assert E.main(["plot", str(out), "--x", "snr", "--y", "rmse"]) == 0
    assert list(out.glob("plot_*.png"))
    assert E.main(["plot", str(out), "--results", "--x", "lam", "--y", "rmse", "--group", "chain",
                   "--where", "inversion=medi", "--png", str(out / "lam.png")]) == 0
    assert (out / "lam.png").exists()
