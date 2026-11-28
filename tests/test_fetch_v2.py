"""Real-network tests for scripts/fetch_data_v2.py (spec 2.11). No mocks.

Run: pytest tests/test_fetch_v2.py -v
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fetch_data_v2 as f  # noqa: E402
import validate_forecasts_v2 as v  # noqa: E402

SUBSET = ["Alta", "Snowbird", "Zermatt", "Davos", "St. Anton", "Val Thorens", "Niseko United"]


@pytest.fixture(scope="module")
def run_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("fetch") / "forecasts_v2.json"
    rc = f.main(["--only", ",".join(SUBSET), "--out", str(out)])
    assert rc == 0, "fetcher failed (validation or network)"
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def by_id(run_output):
    return {r["id"]: r for r in run_output["resorts"]}


def test_subset_passes_validation(run_output):
    assert v.validate(run_output, expected_count=len(SUBSET)) == []
    assert run_output["schema_version"] == 3


def test_alta_has_no_ben_lomond(by_id):
    ids = [s["id"] for s in by_id["alta"]["stations"]]
    assert "332:UT:SNTL" not in ids
    for s in by_id["alta"]["stations"]:
        assert s["distance_km"] <= 60


def test_no_fake_tenki_100(run_output):
    tenki = [r for r in run_output["resorts"] if r["depth"]["provider"] == "tenki.jp"]
    vals = {r["depth"]["value_cm"] for r in tenki}
    if any(r["depth"]["value_cm"] == 100.0 for r in tenki):
        assert len(vals) > 1


def test_snowfall_height_null_in_us(by_id):
    for d in by_id["alta"]["days"]:
        if d["index"] in (0, 1):
            assert d["derived"]["snowfall_height_precip_min_m"] is None
            assert d["derived"]["snowfall_height_precip_max_m"] is None


def test_summit_null_iff_no_summit(run_output):
    for r in run_output["resorts"]:
        if r["summit_elevation_m"] is None:
            assert r["next_72h"]["summit_cm"] is None
        elif r["quality"]["summit_forecast_ok"]:
            assert r["next_72h"]["summit_cm"] is not None


def test_summit_is_a_real_call(run_output):
    for r in run_output["resorts"]:
        b, s = r["now"]["base"], r["now"]["summit"]
        if b and s and r["summit_elevation_m"] - r["elevation_m"] > 100:
            assert b["temp_c"] != s["temp_c"] or b["temp_c"] is None


# ---- pure functions on real responses ------------------------------------


@pytest.fixture(scope="module")
def real_alta():
    """Fetch real Open-Meteo responses for Alta and assemble days."""
    r = [x for x in f.build_resorts() if x["name"] == "Alta"][0]
    failed = []
    om = {}
    for kind in ("daily", "hourly"):
        res = f.openmeteo_fetch(kind, [r], failed, f._om_points)
        for lab, resp in res[r["id"]].items():
            om[f"{kind}_{lab}"] = resp
    run = datetime.now(timezone.utc)
    H = f.local_hour(run, om["hourly_base"]["utc_offset_seconds"])
    days = f.build_days(om, {"base": r["base_m"], "mid": r["mid_m"], "summit": r["summit_m"]}, run, H)
    return days


def test_zero_snow_day_has_no_powder_band(real_alta):
    for d in real_alta:
        s = d["summit"]
        if s and s["snowfall_cm"] == 0.0:
            assert d["derived"]["powder_band"] is None


def test_extended_wind_hold_never_false(real_alta):
    for d in real_alta:
        if d["horizon"] == "extended":
            assert d["derived"]["wind_hold"] is not False


def test_ski_score_null_on_wind_hold(real_alta):
    for d in real_alta:
        if d["derived"]["wind_hold"] is True:
            assert d["derived"]["ski_score_cm"] is None


def test_past_days_have_no_derived(real_alta):
    for d in real_alta:
        if d["horizon"] == "past_model":
            der = d["derived"]
            assert all(val is None for k, val in der.items() if k != "rain_risk")
            assert all(val is None for val in der["rain_risk"].values())


def test_three_valued_helpers():
    assert f.tri_or(None, False) is None
    assert f.tri_or(True, None) is True
    assert f.tri_or(False, False) is False
    assert f.tri_and(False, None) is False
    assert f.tri_and(True, None) is None
    assert f.tri_and(True, True) is True
