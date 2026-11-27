"""The validator must reject each real-shaped broken file.

Base file: a 7-resort file written by the current fetcher from recorded payloads
(valid). Each case breaks exactly one thing in a copy of it and checks that the
validator names that thing. The CLI is exercised as a subprocess on files
written to disk, since that is how the refresh workflow runs it.
"""

import copy
import json
import math
import subprocess
import sys

import pytest

from conftest import SCRIPTS


def V(v, doc, n=7):
    return v.validate(doc, expected_count=n)


def test_base_file_is_valid(v, smoke):
    assert V(v, smoke) == []


def test_validator_default_count_tracks_the_roster(f, v):
    """The expansion plans add resorts in phases; the validator's default must move with build_resorts()."""
    assert v.EXPECTED_RESORTS == len(f.build_resorts())


def test_validator_knows_every_country_and_provider(f, v):
    assert {r["country"] for r in f.build_resorts()} <= v.COUNTRIES
    assert set(f.STATION_PROVIDERS) == set(v.STATION_PROVIDERS)


def test_real_run_file_rejected_for_missing_observed_utc(v, real_run):
    """The 12:15 UTC file predates recent.measured.observed_utc; the current validator refuses it.

    This is the committed data/forecasts_v2.json as of this suite's creation. The next fetcher
    run writes the key; until then the workflow's post-write validate step would fail on it."""
    errs = v.validate(real_run)
    assert errs
    assert all("recent.measured: missing keys ['observed_utc']" in e for e in errs)


def _r(doc, rid):
    return next(r for r in doc["resorts"] if r["id"] == rid)


def _day(doc, rid, idx):
    return next(d for d in _r(doc, rid)["days"] if d["index"] == idx)


def m_missing_top(d):
    del d["thresholds"]


def m_schema(d):
    d["schema_version"] = 2


def m_dup_id(d):
    d["resorts"][1]["id"] = d["resorts"][0]["id"]


def m_drop_resort(d):
    d["resorts"].pop()


def m_country(d):
    _r(d, "alta")["country"] = "Canada"


def m_station_far(d):
    _r(d, "alta")["stations"][0]["distance_km"] = 15.1


def m_station_order(d):
    st = _r(d, "alta")["stations"]
    assert len(st) >= 2
    st.reverse()


def m_station_provider(d):
    _r(d, "alta")["stations"][0]["provider"] = "NWS"


def m_station_key(d):
    del _r(d, "zermatt")["stations"][0]["new_24h_cm"]


def m_nan(d):
    _day(d, "zermatt", 2)["summit"]["temp_max_c"] = math.nan


def m_inf(d):
    _r(d, "vail")["next_72h"]["summit_cm"] = math.inf


def m_negative_snow(d):
    _day(d, "furano", 3)["mid"]["snowfall_cm"] = -0.4


def m_negative_new_snow(d):
    _r(d, "alta")["forecast"][2]["new_snow_cm"] = -1.0


def m_past_derived(d):
    _day(d, "st-anton", -1)["derived"]["powder_band"] = "average"


def m_extended_hourly(d):
    _day(d, "val-thorens", 9)["derived"]["flat_light_hours"] = 0


def m_extended_wind_false(d):
    _day(d, "val-thorens", 9)["derived"]["wind_hold"] = False


def m_horizon(d):
    _day(d, "alta", 7)["horizon"] = "short"


def m_wind_hold_str(d):
    _day(d, "alta", 1)["derived"]["wind_hold"] = "yes"


def m_powder_band(d):
    _day(d, "alta", 1)["derived"]["powder_band"] = "champagne"


def m_days_order(d):
    days = _r(d, "zermatt")["days"]
    days[3], days[4] = days[4], days[3]


def m_days_missing(d):
    _r(d, "zermatt")["days"].pop()


def m_base_depth(d):
    r = _r(d, "alta")
    r["base_depth_cm"] = (r["depth"]["value_cm"] or 0) + 10


def m_depth_source(d):
    _r(d, "alta")["base_depth_source"] = "estimated"


def m_depth_modelled(d):
    _r(d, "zermatt")["depth"]["source_label"] = "Modelled snow depth (ERA5)"


def m_depth_provider(d):
    _r(d, "zermatt")["depth"]["provider"] = "Open-Meteo"


def m_depth_type(d):
    r = _r(d, "zermatt")
    r["depth"]["value_cm"] = "112"
    r["base_depth_cm"] = "112"


def m_depth_ok(d):
    r = _r(d, "zermatt")
    r["quality"]["depth_ok"] = not r["quality"]["depth_ok"]


def m_temp(d):
    _r(d, "alta")["temp_c"] = 99.0


def m_summit_temp(d):
    _r(d, "alta")["summit_temp_c"] = 99.0


def m_seven_day(d):
    r = _r(d, "alta")
    r["seven_day_total_cm"] = round((r["seven_day_total_cm"] or 0) + 5, 1)


def m_seven_day_null_day(d):
    r = _r(d, "alta")
    r["forecast"][4]["new_snow_cm"] = None


def m_summit_week(d):
    r = _r(d, "alta")
    r["summit_seven_day_total_cm"] = 0.123


def m_measured_kind(d):
    _r(d, "alta")["recent"]["measured"]["kind"] = "model"


def m_measured_key(d):
    del _r(d, "alta")["recent"]["measured"]["observed_utc"]


def m_summit_without_elev(d):
    r = _r(d, "alta")
    r["summit_elevation_m"] = None


def m_elev_type(d):
    _r(d, "alta")["elevation_m"] = 2600.0


def m_summit_note(d):
    _r(d, "alta")["elevation_note"] = ""


def m_forecast_ok_low(d):
    for r in d["resorts"][:2]:
        r["quality"]["forecast_ok"] = False


def m_resort_key(d):
    del _r(d, "furano")["flags"]


def m_week_key(d):
    del _r(d, "furano")["week"]["best_ski_day"]


def m_run_key(d):
    del d["run"]["warnings"]


CASES = [
    (m_missing_top, "top: missing keys ['thresholds']"),
    (m_schema, "schema_version is 2"),
    (m_dup_id, "resort ids are not unique"),
    (m_drop_resort, "resort count 6 != 7"),
    (m_country, "bad country Canada"),
    (m_station_far, "is 15.1 km away"),
    (m_station_order, "stations not sorted by distance"),
    (m_station_provider, "bad station provider NWS"),
    (m_station_key, "station: missing keys ['new_24h_cm']"),
    (m_nan, "NaN or infinite"),
    (m_inf, "NaN or infinite"),
    (m_negative_snow, "negative value -0.4"),
    (m_negative_new_snow, "negative value -1.0"),
    (m_past_derived, "past_model day -1 has derived values ['powder_band']"),
    (m_extended_hourly, "extended day 9 has hourly-derived values ['flat_light_hours']"),
    (m_extended_wind_false, "wind_hold=false"),
    (m_horizon, "day 7 horizon short"),
    (m_wind_hold_str, "wind_hold not bool/null"),
    (m_powder_band, "bad powder_band"),
    (m_days_order, "days not ascending"),
    (m_days_missing, "forecast_ok but days cover"),
    (m_base_depth, "base_depth_cm != depth.value_cm"),
    (m_depth_source, "base_depth_source != depth.source_label"),
    (m_depth_modelled, "depth mentions modelled data"),
    (m_depth_provider, "depth.provider Open-Meteo not allowed"),
    (m_depth_type, "depth.value_cm wrong type"),
    (m_depth_ok, "quality.depth_ok inconsistent"),
    (m_temp, "temp_c != now.base.temp_c"),
    (m_summit_temp, "summit_temp_c != now.summit.temp_c"),
    (m_seven_day, "seven_day_total_cm"),
    (m_seven_day_null_day, "seven_day_total_cm set although a forecast day is null"),
    (m_summit_week, "summit_seven_day_total_cm != week.summit_cm"),
    (m_measured_kind, "bad recent.measured kind"),
    (m_measured_key, "recent.measured: missing keys ['observed_utc']"),
    (m_summit_without_elev, "summit forecast without summit elevation"),
    (m_elev_type, "elevation_m not int"),
    (m_summit_note, "summit without elevation_note"),
    (m_forecast_ok_low, "of 7 resorts have forecast_ok"),
    (m_resort_key, "missing keys ['flags']"),
    (m_week_key, "week: missing keys ['best_ski_day']"),
    (m_run_key, "run: missing keys ['warnings']"),
]


@pytest.mark.parametrize("mutate,expect", CASES, ids=[c[0].__name__ for c in CASES])
def test_broken_file_rejected(v, smoke, mutate, expect):
    doc = copy.deepcopy(smoke)
    mutate(doc)
    errs = V(v, doc)
    assert errs, f"{mutate.__name__} was accepted"
    assert any(expect in e for e in errs), errs[:5]


def test_resorts_not_a_list(v, smoke):
    smoke["resorts"] = {"alta": {}}
    assert V(v, smoke) == ["resorts is not a list"]


def test_not_an_object(v):
    assert v.validate([], expected_count=None) == ["top: not an object"]


@pytest.mark.xfail(strict=True, reason=(
    "GAP: the validator checks only the 15 km distance rule; a station outside the elevation window "
    "(300 m below base to 300 m above summit, docs/depth_coverage_plan.md 0.1) passes, although the "
    "file carries station elevation_m, resort elevation_m and summit_elevation_m."))
def test_station_outside_elevation_window_rejected(v, smoke):
    r = _r(smoke, "alta")
    r["stations"][0]["elevation_m"] = r["elevation_m"] - 301
    assert V(v, smoke)


# ----------------------------------------------------------------------------- CLI


def _cli(path, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / "validate_forecasts_v2.py"), str(path), *args],
                          capture_output=True, text=True, timeout=60)


def test_cli_accepts_valid_file(smoke, tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")
    res = _cli(p, "--expected-count", "7")
    assert res.returncode == 0
    assert res.stdout.strip() == "OK: 7 resorts, schema_version 3"


def test_cli_rejects_nan_literal_on_disk(smoke, tmp_path):
    """json.dumps writes NaN as a bare literal; the CLI must refuse such a file."""
    smoke["resorts"][0]["days"][4]["base"]["rain_mm"] = math.nan
    p = tmp_path / "nan.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")
    assert "NaN" in p.read_text(encoding="utf-8")
    res = _cli(p, "--expected-count", "7")
    assert res.returncode == 1
    assert "INVALID:" in res.stdout and "NaN or infinite" in res.stdout


def test_cli_count_defaults_to_the_files_own_resort_count(smoke, tmp_path):
    """The workflow's --health step passes no count; after resorts are added it must still pass.
    The fetcher checks the count against its own resort list before writing."""
    p = tmp_path / "seven.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")
    assert smoke["run"]["resort_count"] == 7
    res = _cli(p)
    assert res.returncode == 0, res.stdout
    res = _cli(p, "--expected-count", "86")
    assert res.returncode == 1
    assert "resort count 7 != 86" in res.stdout


def test_cli_rejects_resort_count_that_disagrees_with_the_list(smoke, tmp_path):
    smoke["run"]["resort_count"] = 86
    p = tmp_path / "lying.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")
    res = _cli(p)
    assert res.returncode == 1
    assert "resort count 7 != 86" in res.stdout
    assert "run.resort_count 86 != 7 resorts in the file" in res.stdout


def test_cli_truncated_file_fails_loudly(smoke, tmp_path):
    text = json.dumps(smoke)
    p = tmp_path / "cut.json"
    p.write_text(text[: len(text) // 2], encoding="utf-8")
    res = _cli(p, "--expected-count", "7")
    assert res.returncode != 0
    assert "JSONDecodeError" in res.stderr


def test_cli_health_flags_run_warnings(smoke, tmp_path):
    smoke["run"]["warnings"] = ["SNOTEL (AWDB) request failed; no SNOTEL station data this run"]
    p = tmp_path / "warn.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")
    res = _cli(p, "--expected-count", "7", "--health")
    assert res.returncode == 1
    assert "HEALTH: run warning: SNOTEL (AWDB) request failed" in res.stdout


def test_health_flags_provider_with_no_readings(v, real_run):
    for r in real_run["resorts"]:
        for s in r["stations"]:
            if s["provider"] == "SNOTEL":
                s["depth_cm"] = None
    probs = v.health(real_run)
    assert any(p.startswith("SNOTEL: 0 of ") for p in probs)
