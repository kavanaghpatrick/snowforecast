"""Powder upgrades in the fetcher: Kuchera snow-to-liquid ratio (per band, underground
pressure levels masked), storm profile, post-snowfall gust, and past-day fields.

No mocks. Pure functions get hand-computed inputs built as real Open-Meteo hourly blocks;
the pipeline tests run the fetcher's own build_days / assemble_resort / validator on
Open-Meteo payloads recorded live on 28 Nov 2026 12:46 UTC (Myoko Kogen 731 / 1,293 /
1,855 m and Jackson Hole 1,924 / 2,555 / 3,185 m; daily, hourly and the summit-only
pressure-level call): fixtures/om_kuchera_myoko_jackson.json.gz.
"""

import copy
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta

import pytest

from conftest import REPO, load_fixture
from _replay import InterceptProxy

PLEV = ("temperature_850hPa", "temperature_700hPa", "temperature_600hPa", "temperature_500hPa")
DAY = date(2026, 12, 1)
TIMES = [f"{DAY.isoformat()}T{h:02d}:00" for h in range(24)]


def hourly(f, **series):
    """A real Hourly index over one day; each series is a 24-value list or a constant."""
    h = {"time": TIMES}
    for k, v in series.items():
        h[k] = list(v) if isinstance(v, (list, tuple)) else [v] * 24
    return f.Hourly({"hourly": h})


# ----------------------------------------------------------------------------- Kuchera formula


@pytest.mark.parametrize("tmax_k, exp", [
    (271.16, 12.0),   # the pivot
    (270.16, 13.0),   # cold side: +1 per K
    (266.16, 17.0),
    (256.16, 27.0),
    (272.16, 10.0),   # warm side: -2 per K
    (273.16, 8.0),
    (276.16, 2.0),
    (277.16, 0.0),    # 12 - 12
    (280.16, 0.0),    # 12 - 18 floored at 0
])
def test_kuchera_formula_both_sides_of_the_pivot(f, tmax_k, exp):
    assert f.kuchera_slr(tmax_k) == pytest.approx(exp, abs=1e-9)


def test_kuchera_none_in_none_out(f):
    assert f.kuchera_slr(None) is None


# ----------------------------------------------------------------------------- underground mask

# A warm, low 850 hPa reading under a cold column: the classic trap.
PLEV_C = {"temperature_850hPa": 6.5, "temperature_700hPa": -5.0,
          "temperature_600hPa": -12.0, "temperature_500hPa": -20.0}


def test_mask_at_a_3000_m_band(f):
    # 850 hPa (1,457 m) is underground at 3,000 m and must not count; 700 hPa (3,012 m) counts.
    t = f.column_tmax_k(-8.0, PLEV_C, 3000)
    assert t == pytest.approx(-5.0 + 273.15)
    assert f.kuchera_slr(t) == pytest.approx(12 + (271.16 - 268.15))       # 15.01
    # Unmasked the same column says almost no snow: the failure the mask exists for.
    unmasked = max([-8.0] + list(PLEV_C.values())) + 273.15
    assert f.kuchera_slr(unmasked) == 0.0


def test_same_column_counts_850_for_a_low_band_and_masks_it_for_the_summit(f):
    """Per-band mask: the same summit-call column, a 1,200 m mid band and a 3,000 m summit."""
    mid = f.column_tmax_k(-1.0, PLEV_C, 1200)
    summit = f.column_tmax_k(-8.0, PLEV_C, 3000)
    assert mid == pytest.approx(6.5 + 273.15)         # 850 hPa is above 1,200 m: it counts
    assert summit == pytest.approx(-5.0 + 273.15)      # masked at 3,000 m
    assert f.kuchera_slr(mid) == 0.0                   # 12 + 2 x (271.16 - 279.65) < 0
    assert f.kuchera_slr(summit) == pytest.approx(15.01)


def test_band_2m_temperature_enters_the_column_max(f):
    # a 2 m air warmer than every counted level sets Tmax
    assert f.column_tmax_k(-1.0, PLEV_C, 3000) == pytest.approx(-1.0 + 273.15)


def test_mask_boundary_is_inclusive_at_the_level_height(f):
    # a band exactly at 1,457 m still has 850 hPa at or above it
    assert f.column_tmax_k(-10.0, PLEV_C, 1457) == pytest.approx(6.5 + 273.15)
    assert f.column_tmax_k(-10.0, PLEV_C, 1458) == pytest.approx(-5.0 + 273.15)


def test_missing_counted_level_is_unknown_but_missing_masked_level_is_not(f):
    p = dict(PLEV_C, temperature_700hPa=None)
    assert f.column_tmax_k(-8.0, p, 3000) is None
    p = dict(PLEV_C, temperature_850hPa=None)
    assert f.column_tmax_k(-8.0, p, 3000) == pytest.approx(268.15)
    assert f.column_tmax_k(None, PLEV_C, 3000) is None


# ----------------------------------------------------------------------------- storm mean, amount, profile


def _storm(f, t2m_by_hour, elev=2000, rain=0.0, precip=1.0, fl=1000.0):
    """Snow at hours 6..6+n-1 (1 mm each); freezing level 1,000 m so a 2,000 m band is above
    the snow line (750 m) and every hour is snow by the phase split. The column above 700 hPa
    is colder than the 2 m air, so Tmax = the 2 m temperature."""
    n = len(t2m_by_hour)
    T = [0.0] * 24
    P = [0.0] * 24
    for i, t in enumerate(t2m_by_hour):
        T[6 + i] = t
        P[6 + i] = precip
    h = hourly(f, precipitation=P, rain=[min(rain, p) for p in P], freezing_level_height=fl,
               snowfall_height=[None] * 24, temperature_2m=T)
    hp = hourly(f, temperature_850hPa=10.0, temperature_700hPa=-30.0,
                temperature_600hPa=-35.0, temperature_500hPa=-40.0)
    return f.kuchera_band(h, hp, TIMES, elev), n


def _slr_c(t):
    k = t + 273.15
    return 12 + (271.16 - k) if k <= 271.16 else 12 + 2 * (271.16 - k)


def test_rising_ratio_is_right_side_up(f):
    temps = [-2.0, -3.0, -4.0, -6.0, -8.0, -10.0]
    (slr, snow, prof), n = _storm(f, temps)
    s = [_slr_c(t) for t in temps]          # 12.01, 13.01, 14.01, 16.01, 18.01, 20.01
    assert slr == round(sum(s) / n, 1) == 15.5
    assert snow == round(sum(s) / 10, 1) == 9.3   # 6 mm of frozen water at 15.5:1
    assert prof == "rsu"                    # last third 19.01 vs first third 12.51


def test_falling_ratio_is_upside_down(f):
    (slr, snow, prof), _ = _storm(f, [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0])
    assert prof == "usd"
    assert slr == 15.5 and snow == 9.3


def test_flat_or_short_storm_has_no_profile(f):
    assert _storm(f, [-6.0, -6.5, -7.0, -7.0, -6.5, -6.0])[0][2] is None   # within 2
    assert _storm(f, [-2.0, -12.0])[0][2] is None                           # fewer than 3 hours
    assert _storm(f, [-2.0, -6.0, -12.0])[0][2] == "rsu"                    # thirds of one hour


def test_profile_step_edge(f):
    assert f.storm_profile([12.0, 12.0, 14.0]) == "rsu"     # exactly +2
    assert f.storm_profile([12.0, 12.0, 13.99]) is None
    assert f.storm_profile([14.0, 12.0, 12.0]) == "usd"


def test_rain_hours_never_count_and_rain_only_is_unknown(f):
    # freezing level 3,000 m: line 2,750 m is above a 2,000 m band, so liquid is rain and
    # frozen water is "mixed": no snow-phase water, no ratio, no amount
    (slr, snow, prof), _ = _storm(f, [-2.0, -3.0, -4.0], fl=3000.0)
    assert (slr, snow, prof) == (None, None, None)


def test_liquid_part_of_a_snow_hour_is_not_scaled(f):
    # 1 mm per hour of which 0.4 mm liquid: only the 0.6 mm frozen part is snow water
    (slr, snow, _), _ = _storm(f, [-5.0, -5.0, -5.0], rain=0.4)
    exp = _slr_c(-5.0)
    assert slr == round(exp, 1)
    assert snow == round(3 * 0.6 * exp / 10, 1)


def test_no_pressure_levels_means_no_ratio(f):
    h = hourly(f, precipitation=1.0, rain=0.0, freezing_level_height=1000.0,
               snowfall_height=[None] * 24, temperature_2m=-5.0)
    assert f.kuchera_band(h, None, TIMES, 2000) == (None, None, None)
    hp = hourly(f, temperature_850hPa=0.0, temperature_700hPa=[None] * 24,
                temperature_600hPa=-20.0, temperature_500hPa=-30.0)
    assert f.kuchera_band(h, hp, TIMES, 2000) == (None, None, None)


def test_missing_precip_hour_means_no_ratio(f):
    P = [1.0] * 24
    P[3] = None
    h = hourly(f, precipitation=P, rain=0.0, freezing_level_height=1000.0,
               snowfall_height=[None] * 24, temperature_2m=-5.0)
    hp = hourly(f, **{k: -30.0 for k in PLEV})
    assert f.kuchera_band(h, hp, TIMES, 2000) == (None, None, None)


# ----------------------------------------------------------------------------- post-snowfall gust

SUNRISE = datetime(2026, 12, 1, 7, 30)
SUNSET = datetime(2026, 12, 1, 16, 30)       # lift hours 08:00 .. 16:00


def _gusts(f, snow_hours, gust_map=None, base=10.0):
    S = [0.0] * 24
    for hh in snow_hours:
        S[hh] = 0.5
    G = [base] * 24
    for hh, g in (gust_map or {}).items():
        G[hh] = g
    return hourly(f, snowfall=S, wind_gusts_10m=G)


def test_post_gust_window_starts_after_last_snow_hour_and_ends_at_last_lift_hour(f):
    # snow 06..10; a 90 km/h gust at 10:00 (still snowing) and 95 at 17:00 (after sunset)
    # are outside; the window 11:00..16:00 holds 30 and 44.4 -> 44
    h = _gusts(f, range(6, 11), {10: 90.0, 12: 30.0, 15: 44.4, 17: 95.0})
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) == 44


def test_post_gust_null_when_snow_falls_through_last_lift_hour(f):
    h = _gusts(f, range(8, 17), {12: 80.0})
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) is None
    h = _gusts(f, [9, 20])            # last snow after sunset: no lift hour after it
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) is None


def test_post_gust_null_when_no_snow_fell(f):
    assert f.post_snow_gust(_gusts(f, []), TIMES, SUNRISE, SUNSET) is None


def test_post_gust_snow_before_lifts_open_uses_all_lift_hours(f):
    h = _gusts(f, [2, 3], {8: 61.0, 16: 20.0, 7: 99.0})   # 07:00 is before sunrise
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) == 61


def test_post_gust_null_when_a_window_gust_is_missing(f):
    S = [0.0] * 24
    S[9] = 1.0
    G = [10.0] * 24
    G[13] = None
    h = hourly(f, snowfall=S, wind_gusts_10m=G)
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) is None
    G[13] = 10.0
    G[20] = None                       # outside the window: does not matter
    h = hourly(f, snowfall=S, wind_gusts_10m=G)
    assert f.post_snow_gust(h, TIMES, SUNRISE, SUNSET) == 10


# ----------------------------------------------------------------------------- recorded payloads


@pytest.fixture(scope="module")
def rec():
    return load_fixture("om_kuchera_myoko_jackson.json.gz")


def _labels(rec, name):
    lab = {f"{k}_{b}": rec[k][f"{name}|{b}"] for k in ("daily", "hourly") for b in ("base", "mid", "summit")}
    lab["plev_summit"] = rec["plev"][name]
    return lab


def _run_utc(rec):
    return datetime.fromisoformat(rec["recorded_utc"].replace("Z", "+00:00"))


@pytest.fixture(scope="module")
def built(f, rec):
    out = {}
    rs = {r["name"]: r for r in f.build_resorts()}
    for name in rec["resorts"]:
        r = rs[name]
        lab = _labels(rec, name)
        elev = {"base": r["base_m"], "mid": r["mid_m"], "summit": r["summit_m"]}
        H = f.local_hour(_run_utc(rec), lab["daily_base"]["utc_offset_seconds"])
        out[name] = (f.build_days(lab, elev, _run_utc(rec), H), lab, elev)
    return out


def test_recorded_plev_call_is_summit_only_and_complete(rec):
    for name in rec["resorts"]:
        h = rec["plev"][name]["hourly"]
        assert set(h) == {"time", *PLEV}
        assert len(h["time"]) == 240          # past_days=3 + forecast_days=7
        assert all(v is not None for k in PLEV for v in h[k])


def test_recorded_past_days_keep_temps_gust_phase(built):
    for name, (days, _, elev) in built.items():
        past = [d for d in days if d["horizon"] == "past_model"]
        assert [d["index"] for d in past] == [-3, -2, -1]
        for d in past:
            for b in ("base", "mid", "summit"):
                for k in ("temp_max_c", "temp_min_c", "gust_max_kmh"):
                    assert d[b][k] is not None, (name, d["index"], b, k)
                assert isinstance(d["derived"]["rain_risk"][b], bool)
                assert isinstance(d["derived"]["mixed_risk"][b], bool)
            for k in ("slr_kuchera", "snow_kuchera_cm", "storm_profile", "post_gust_max_kmh"):
                assert set(d["derived"][k]) == {"base", "mid", "summit"}
                assert set(d["derived"][k].values()) == {None}
        # day 0's "rained the day before" input now exists at the scored (mid) band
        assert isinstance(next(d for d in days if d["index"] == -1)["derived"]["rain_risk"]["mid"], bool)


def test_recorded_snowy_past_day_has_powder_temp(f, built):
    # Myoko day -3: 11.6 cm at mid; the summit snowfall-weighted temperature exists
    days, lab, _ = built["Myoko Kogen"]
    d = next(x for x in days if x["index"] == -3)
    h = f.Hourly(lab["hourly_summit"])
    ts = h.day_times(date.fromisoformat(d["date"]))
    S, T = h.series("snowfall", ts), h.series("temperature_2m", ts)
    assert sum(S) > 0
    assert d["derived"]["powder_temp_c"] == round(sum(a * b for a, b in zip(T, S)) / sum(S), 1)


def _independent_band(lab, band, elev, d, mask=True):
    """Kuchera storm mean recomputed from the raw arrays, independently of the fetcher."""
    hh = lab[f"hourly_{band}"]["hourly"]
    pp = lab["plev_summit"]["hourly"]
    heights = {"temperature_850hPa": 1457, "temperature_700hPa": 3012,
               "temperature_600hPa": 4206, "temperature_500hPa": 5574}
    num = den = 0.0
    for i, t in enumerate(hh["time"]):
        if t[:10] != d:
            continue
        p = hh["precipitation"][i]
        if p <= 0:
            continue
        sh, fl = hh["snowfall_height"][i], hh["freezing_level_height"][i]
        line = sh if sh is not None else fl - 250
        frozen = max(p - min(hh["rain"][i], p), 0.0)
        if elev < line or frozen <= 0:
            continue
        j = pp["time"].index(t)
        col = [hh["temperature_2m"][i]] + [pp[k][j] for k, z in heights.items() if z >= elev or not mask]
        tk = max(col) + 273.15
        slr = max(12 + (271.16 - tk) if tk <= 271.16 else 12 + 2 * (271.16 - tk), 0.0)
        num += frozen * slr
        den += frozen
    return (round(num / den, 1), round(num / 10, 1)) if den else (None, None)


def test_recorded_kuchera_per_band_matches_independent_recompute(built):
    checked = 0
    for name, (days, lab, elev) in built.items():
        for d in days:
            if d["horizon"] != "short":
                continue
            for b in ("base", "mid", "summit"):
                exp = _independent_band(lab, b, elev[b], d["date"])
                assert (d["derived"]["slr_kuchera"][b], d["derived"]["snow_kuchera_cm"][b]) == exp, \
                    (name, d["index"], b)
                checked += exp[0] is not None
    assert checked >= 15


def test_recorded_mask_differs_by_band(built):
    """Myoko mid (1,293 m) keeps 850 hPa, summit (1,855 m) masks it; Jackson Hole masks 850 at
    every band (base 1,924 m). Unmasked, Jackson summit's big day-0 ratio collapses."""
    days, lab, elev = built["Jackson Hole"]
    d0 = next(d for d in days if d["index"] == 0)
    slr = d0["derived"]["slr_kuchera"]
    assert slr == {"base": 9.2, "mid": 10.3, "summit": 15.1}
    unmasked = _independent_band(lab, "summit", elev["summit"], d0["date"], mask=False)
    assert unmasked[0] < slr["summit"] - 5
    days, lab, elev = built["Myoko Kogen"]
    d1 = next(d for d in days if d["index"] == 1)
    assert d1["derived"]["slr_kuchera"]["mid"] == _independent_band(lab, "mid", 1293, d1["date"])[0] == 11.5
    assert _independent_band(lab, "mid", 1293, d1["date"], mask=False)[0] == 11.5   # 850 counts here anyway
    assert d1["derived"]["slr_kuchera"]["summit"] == 15.7


def test_recorded_kuchera_amount_is_extra_and_7_to_1_untouched(built):
    days, lab, _ = built["Jackson Hole"]
    d0 = next(d for d in days if d["index"] == 0)
    i = lab["daily_summit"]["daily"]["time"].index(d0["date"])
    assert d0["summit"]["snowfall_cm"] == round(lab["daily_summit"]["daily"]["snowfall_sum"][i], 1)
    assert d0["derived"]["snow_kuchera_cm"]["summit"] == 43.1


def test_recorded_extended_days_have_no_kuchera(built):
    for days, _, _ in built.values():
        for d in days:
            if d["horizon"] == "extended":
                for k in ("slr_kuchera", "snow_kuchera_cm", "storm_profile", "post_gust_max_kmh"):
                    assert set(d["derived"][k].values()) == {None}


def test_recorded_post_gust_per_band(built):
    days, lab, _ = built["Jackson Hole"]
    d0 = next(d for d in days if d["index"] == 0)
    sr, ss = datetime.fromisoformat(d0["sunrise_local"]), datetime.fromisoformat(d0["sunset_local"])
    for b in ("base", "mid", "summit"):
        hh = lab[f"hourly_{b}"]["hourly"]
        idx = [i for i, t in enumerate(hh["time"]) if t[:10] == d0["date"]]
        snowy = [i for i in idx if hh["snowfall"][i] > 0]
        win = [i for i in idx if i > snowy[-1] and sr < datetime.fromisoformat(hh["time"][i]) <= ss]
        exp = int(max(hh["wind_gusts_10m"][i] for i in win) + 0.5) if win else None
        assert d0["derived"]["post_gust_max_kmh"][b] == exp, b
    assert d0["derived"]["post_gust_max_kmh"]["summit"] == 55


# ----------------------------------------------------------------------------- end to end


def _assemble_doc(f, rec, smoke):
    rs = {r["name"]: r for r in f.build_resorts()}
    om = {}
    out = []
    for name in rec["resorts"]:
        r = dict(rs[name], notes=[], stations=[])
        om[r["id"]] = _labels(rec, name)
        out.append(f.assemble_resort(r, om, _run_utc(rec), False, lambda r_: []))
    doc = copy.deepcopy(smoke)
    doc["updated"] = rec["recorded_utc"]
    doc["run"]["resort_count"] = len(out)
    doc["resorts"] = out
    return doc


def test_new_fields_survive_assembly_validation_and_write(f, v, rec, smoke, tmp_path):
    doc = _assemble_doc(f, rec, smoke)
    assert v.validate(doc, expected_count=2) == []
    p = tmp_path / "out.json"
    f.write_atomic(doc, p)
    back = json.loads(p.read_text())
    jh = next(r for r in back["resorts"] if r["name"] == "Jackson Hole")
    by = {d["index"]: d for d in jh["days"]}
    assert by[-1]["mid"]["temp_max_c"] is not None and by[-1]["mid"]["gust_max_kmh"] is not None
    assert isinstance(by[-1]["derived"]["rain_risk"]["mid"], bool)
    assert by[0]["derived"]["slr_kuchera"] == {"base": 9.2, "mid": 10.3, "summit": 15.1}
    assert by[0]["derived"]["storm_profile"]["summit"] == "rsu"


@pytest.mark.parametrize("change, needle", [
    (lambda d: d.update(slr_kuchera={"mid": 75.0}), "slr_kuchera.mid = 75.0 outside"),
    (lambda d: d.update(slr_kuchera={"mid": -1.0}), "outside"),
    (lambda d: d.update(storm_profile={"summit": "sideways"}), "not one of"),
    (lambda d: d.update(slr_kuchera={"mid": None}, snow_kuchera_cm={"mid": 5.0}),
     "snow_kuchera_cm.mid set without slr_kuchera.mid"),
    (lambda d: d.update(post_gust_max_kmh={"summit": "40"}), "not a number"),
    (lambda d: d.update(slr_kuchera={"top": 12.0}), "not an object keyed by band"),
    (lambda d: d.update(slr_kuchera=12.0), "not an object keyed by band"),
])
def test_validator_bounds_new_fields(f, v, rec, smoke, change, needle):
    doc = _assemble_doc(f, rec, smoke)
    jh = next(r for r in doc["resorts"] if r["name"] == "Jackson Hole")
    change(next(d for d in jh["days"] if d["index"] == 2)["derived"])
    errs = v.validate(doc, expected_count=2)
    assert any(needle in e for e in errs), errs


def test_validator_rejects_kuchera_on_past_and_extended_days(f, v, rec, smoke):
    doc = _assemble_doc(f, rec, smoke)
    jh = next(r for r in doc["resorts"] if r["name"] == "Jackson Hole")
    by = {d["index"]: d for d in jh["days"]}
    by[-2]["derived"]["slr_kuchera"]["mid"] = 12.0
    by[9]["derived"]["post_gust_max_kmh"]["summit"] = 30
    errs = v.validate(doc, expected_count=2)
    assert any("past_model day -2" in e and "slr_kuchera" in e for e in errs), errs
    assert any("extended day 9" in e and "post_gust_max_kmh" in e for e in errs), errs


def test_validator_still_rejects_other_derived_on_past_days(f, v, rec, smoke):
    doc = _assemble_doc(f, rec, smoke)
    jh = next(r for r in doc["resorts"] if r["name"] == "Jackson Hole")
    next(d for d in jh["days"] if d["index"] == -1)["derived"]["ski_score_cm"] = 4.0
    assert any("past_model day -1 has derived values ['ski_score_cm']" in e
               for e in v.validate(doc, expected_count=2))


def test_files_without_the_new_fields_stay_valid(v, smoke):
    assert v.validate(smoke, expected_count=smoke["run"]["resort_count"]) == []


def test_fetcher_subprocess_asks_plev_on_summit_only_and_writes_fields(rec, tmp_path):
    """The unmodified fetcher, through the intercept proxy, replaying the recorded payloads.
    The plev call must carry one point per resort (the summit) and the written file must hold
    the past-day and Kuchera fields. Day indices follow the wall clock, so the forecast window
    only lines up on the recording date; on other dates the run is skipped, not faked."""
    name = "Jackson Hole"
    lab = _labels(rec, name)
    off = timedelta(seconds=lab["daily_base"]["utc_offset_seconds"])
    local_today = (datetime.now(tz=_run_utc(rec).tzinfo) + off).date()
    rec_today = (_run_utc(rec) + off).date()
    if local_today != rec_today:
        pytest.skip(f"recorded payloads are for {rec_today}, today is {local_today}")
    pts = lambda k: [lab[f"{k}_{b}"] for b in ("base", "mid", "summit")]   # noqa: E731
    rules = [
        {"host": "open-meteo", "path": "temperature_850hPa", "mode": "replay", "om": [lab["plev_summit"]]},
        {"host": "open-meteo", "path": "models=", "mode": "replay", "om": load_fixture("om_models_probe.json")},
        {"host": "open-meteo", "path": "hourly=", "mode": "replay", "om": pts("hourly")},
        {"host": "open-meteo", "path": "", "mode": "replay", "om": pts("daily")},
    ]
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    out = tmp_path / "out.json"
    with InterceptProxy(rules, str(cert_dir)) as proxy:
        env = proxy.env(dict(os.environ))
        env.update({"FETCH_RUN_BUDGET_S": "90", "FETCH_OM_RESERVE_S": "60"})
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "fetch_data_v2.py"), "--only", name,
                            "--out", str(out)], env=env, capture_output=True, text=True, timeout=180,
                           cwd=str(REPO))
        log = list(proxy.log)
    plev = [e for e in log if "temperature_850hPa" in e["query"]]
    assert len(plev) == 1, p.stderr[-2000:]
    assert "elevation=3185&" in plev[0]["query"]          # one point: the summit
    others = [e for e in log if "open-meteo" in e["host"] and "hourly=" in e["query"] and e not in plev]
    assert others and all("850hPa" not in e["query"] for e in others)
    assert p.returncode == 0, p.stderr[-3000:]
    doc = json.loads(out.read_text())
    by = {d["index"]: d for d in doc["resorts"][0]["days"]}
    assert by[-1]["summit"]["temp_max_c"] is not None
    assert isinstance(by[-1]["derived"]["rain_risk"]["mid"], bool)
    assert by[0]["derived"]["slr_kuchera"]["summit"] == 15.1
