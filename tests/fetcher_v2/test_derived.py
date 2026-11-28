"""Derived metrics against the spec formulas (spec 1.9.1, 1.10, 1.11).

Thresholds always come from the "thresholds" block of the real JSON, never from
numbers typed into the test, so the test reads the same rules the page reads.
Formulas are re-implemented here from the spec text, independently of the
fetcher, and compared with what the fetcher computed.
"""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from conftest import load_fixture

# Open-Meteo "current" in the probe payload is 2026-11-27T01:45 America/Denver = 08:45 UTC.
PROBE_RUN_UTC = datetime(2026, 11, 27, 8, 50, tzinfo=timezone.utc)


def half_up(x):
    return int(math.floor(x + 0.5))


def tri_and(vals):
    if any(v is False for v in vals):
        return False
    return True if all(v is True for v in vals) else None


def tri_or(vals):
    if any(v is True for v in vals):
        return True
    return False if all(v is False for v in vals) else None


@pytest.fixture(scope="module")
def T(real_run_doc):
    return real_run_doc["thresholds"]


def test_thresholds_block_is_the_fetchers(f, real_run_doc, smoke_doc):
    assert real_run_doc["thresholds"] == f.THRESHOLDS
    assert smoke_doc["thresholds"] == f.THRESHOLDS


def test_thresholds_values_match_spec(T):
    assert T["wind_gust_hold_kmh"] == 60
    assert T["wind_700hpa_hold_kmh"] == 70
    assert T["wind_700hpa_min_summit_m"] == 2500
    assert T["storm_3day_cm"] == 15
    assert T["storm_codes"] == [73, 75, 85, 86]
    assert T["storm_prob_pct"] == 50
    assert T["storm_day_min_cm"] == 10
    assert T["powder_cold_below_c"] == -8
    assert T["powder_warm_above_c"] == -3
    assert T["bluebird_prev24_cm"] == 10
    assert T["bluebird_codes"] == [0, 1, 2]
    assert T["bluebird_cloud_mean_pct"] == 40
    assert T["bluebird_flat_light_max_h"] == 2
    assert T["snowy_day_min_cm"] == 1.0


# ----------------------------------------------------------------------------- powder


def spec_powder_band(t, T):
    if t is None:
        return None
    if t < T["powder_cold_below_c"]:
        return "cold_dry"
    if t <= T["powder_warm_above_c"]:
        return "average"
    return "warm_heavy"


@pytest.mark.parametrize("t", [None, -20.0, -8.1, -8.0, -5.0, -3.0, -2.9, 0.0, 3.0])
def test_powder_band_boundaries(f, T, t):
    assert f.powder_band(t) == spec_powder_band(t, T)


def test_powder_band_exact_edges(f):
    assert f.powder_band(-8.0) == "average"     # -8 <= t is average
    assert f.powder_band(-3.0) == "average"     # t <= -3 is average
    assert f.powder_band(-8.01) == "cold_dry"
    assert f.powder_band(-2.99) == "warm_heavy"


@pytest.fixture(scope="module")
def snowbird_days(f):
    """build_days on recorded Snowbird summit daily (probe a[1]) and hourly (probe b[0]) payloads."""
    a = load_fixture("om_daily_probe.json")
    b = load_fixture("om_hourly_probe.json")
    assert a[1]["elevation"] == 3215.0 and b[0]["elevation"] == 3215.0
    labels = {"daily_base": a[1], "daily_summit": a[1], "hourly_base": b[0], "hourly_summit": b[0]}
    elev = {"base": 3215, "mid": None, "summit": 3215}
    H = f.local_hour(PROBE_RUN_UTC, a[1]["utc_offset_seconds"])
    days = f.build_days(labels, elev, PROBE_RUN_UTC, H)
    return days, f.Hourly(b[0]), a[1]


def test_powder_temp_is_snow_weighted_mean(f, T, snowbird_days):
    days, h, _ = snowbird_days
    checked = 0
    for d in days:
        if d["horizon"] != "short":
            continue
        der = d["derived"]
        sday = d["summit"]["snowfall_cm"]
        ts = h.day_times(date.fromisoformat(d["date"]))
        S = h.series("snowfall", ts)
        Tt = h.series("temperature_2m", ts)
        if sday is None or sday < T["snowy_day_min_cm"] or sum(S) == 0:
            assert der["powder_temp_c"] is None, d["date"]
            assert der["powder_band"] is None, d["date"]
            continue
        exp = round(sum(a * b for a, b in zip(Tt, S)) / sum(S), 1)
        assert der["powder_temp_c"] == pytest.approx(exp, abs=0.05), d["date"]
        assert der["powder_band"] == spec_powder_band(der["powder_temp_c"], T)
        checked += 1
    assert checked >= 2


def _lift_times(h, d):
    sr = datetime.fromisoformat(d["sunrise_local"])
    ss = datetime.fromisoformat(d["sunset_local"])
    return [t for t in h.day_times(date.fromisoformat(d["date"])) if sr < datetime.fromisoformat(t) <= ss]


def test_wind_hold_from_lift_hours(f, T, snowbird_days):
    """Short days: gust > 60 during lift hours OR (summit >= 2,500 m) 700 hPa wind > 70."""
    days, h, _ = snowbird_days
    seen_true = seen_false = 0
    for d in days:
        if d["horizon"] != "short":
            continue
        lt = _lift_times(h, d)
        g = max(h.series("wind_gusts_10m", lt))
        w7 = max(h.series("wind_speed_700hPa", lt))
        der = d["derived"]
        # whole km/h, half up (the fetcher's rint), then compared with the thresholds
        assert der["wind_gust_lift_max_kmh"] == half_up(g)
        assert der["wind_700hpa_lift_max_kmh"] == half_up(w7)
        exp = (half_up(g) > T["wind_gust_hold_kmh"]) or (half_up(w7) > T["wind_700hpa_hold_kmh"])
        assert der["wind_hold"] is exp, d["date"]
        seen_true += exp
        seen_false += not exp
    assert seen_true and seen_false, "the recorded week has both held and open days"


def test_lift_hour_fields(f, snowbird_days):
    days, h, _ = snowbird_days
    for d in days:
        if d["horizon"] != "short":
            continue
        lt = _lift_times(h, d)
        der = d["derived"]
        assert der["lift_hours_snowfall_cm"] == round(sum(h.series("snowfall", lt)), 1)
        assert der["wind_chill_min_c"] == round(min(h.series("apparent_temperature", lt)), 1)
        assert der["cloud_cover_lift_mean_pct"] == half_up(sum(h.series("cloud_cover", lt)) / len(lt))
        # the recorded probe predates cloud_cover_low/mid in HOURLY_VARS: flat light is unknown, not 0
        assert "cloud_cover_low" not in h.h
        assert der["flat_light_hours"] is None
        assert der["bluebird"] in (None, False)


PAST_DERIVED_SET = {"rain_risk", "mixed_risk", "phase", "rain_phase_mm", "powder_temp_c"}


def test_past_model_days_carry_only_phase_and_powder_temp(f, T, snowbird_days):
    """Past days get the phase fields and powder_temp_c by the forecast-day rules (the
    hourly call reaches back 3 days); every other derived value stays null."""
    days, h, _ = snowbird_days
    past = [d for d in days if d["horizon"] == "past_model"]
    assert [d["index"] for d in past] == [-3, -2, -1]
    for d in past:
        for k, v in d["derived"].items():
            if k in PAST_DERIVED_SET:
                continue
            assert v is None or (isinstance(v, dict) and set(v.values()) <= {None}), (d["index"], k)
        ts = h.day_times(date.fromisoformat(d["date"]))
        assert len(ts) == 24
        ps = f.phase_summary(h, ts, 3215)
        assert d["derived"]["rain_risk"]["summit"] is ps["rain_risk"]
        assert isinstance(d["derived"]["rain_risk"]["summit"], bool)
        assert d["derived"]["phase"]["summit"] == ps["phase"]
        S, Tt = h.series("snowfall", ts), h.series("temperature_2m", ts)
        if d["summit"]["snowfall_cm"] >= T["snowy_day_min_cm"] and sum(S) > 0:
            exp = round(sum(a * b for a, b in zip(Tt, S)) / sum(S), 1)
            assert d["derived"]["powder_temp_c"] == pytest.approx(exp, abs=0.05)
        else:
            assert d["derived"]["powder_temp_c"] is None


def test_past_model_day_points_keep_amounts_temps_and_gust(f, snowbird_days):
    days, _, daily = snowbird_days
    dd = daily["daily"]
    seen = 0
    for d in days:
        if d["horizon"] == "past_model":
            i = dd["time"].index(d["date"])
            kept = {k for k, v in d["summit"].items() if v is not None}
            assert kept <= set(f.PAST_DAY_POINT_KEYS)
            assert d["summit"]["temp_max_c"] == round(dd["temperature_2m_max"][i], 1)
            assert d["summit"]["temp_min_c"] == round(dd["temperature_2m_min"][i], 1)
            assert d["summit"]["gust_max_kmh"] == half_up(dd["wind_gusts_10m_max"][i])
            assert d["summit"]["weather_code"] is None and d["summit"]["sunshine_s"] is None
            seen += 1
    assert seen == 3


def test_daily_day16_dropped(f, snowbird_days):
    days, _, daily = snowbird_days
    assert daily["daily"]["snowfall_sum"][-1] is None
    assert len(daily["daily"]["time"]) == 19
    assert max(d["index"] for d in days) == 14
    assert all(d["date"] != daily["daily"]["time"][-1] for d in days)


# ----------------------------------------------------------------------------- the real file


def _short_days(doc):
    for r in doc["resorts"]:
        by = {d["index"]: d for d in r["days"]}
        for d in r["days"]:
            yield r, by, d


def test_real_powder_band_consistent(real_run_doc, T):
    n = 0
    for r, by, d in _short_days(real_run_doc):
        der = d["derived"]
        assert der["powder_band"] == spec_powder_band(der["powder_temp_c"], T), (r["id"], d["date"])
        s = d["summit"]
        if d["horizon"] == "short" and s and s["snowfall_cm"] is not None and s["snowfall_cm"] < T["snowy_day_min_cm"]:
            assert der["powder_band"] is None        # spec test 6: no snow, no powder word
            n += 1
    assert n > 50


def test_real_wind_hold_formula(real_run_doc, T):
    n = 0
    for r, by, d in _short_days(real_run_doc):
        der = d["derived"]
        if d["horizon"] == "short":
            sm = r["summit_elevation_m"]
            g = der["wind_gust_lift_max_kmh"]
            w7 = der["wind_700hpa_lift_max_kmh"]
            c1 = None if g is None else g > T["wind_gust_hold_kmh"]
            if sm is None:
                c2 = None
            elif sm >= T["wind_700hpa_min_summit_m"]:
                c2 = None if w7 is None else w7 > T["wind_700hpa_hold_kmh"]
            else:
                c2 = False
            assert der["wind_hold"] == tri_or([c1, c2]), (r["id"], d["date"])
            n += 1
        elif d["horizon"] == "extended":
            gust = (d["summit"] or {}).get("gust_max_kmh")
            assert der["wind_hold"] is not False, (r["id"], d["date"])
            assert der["wind_hold"] == (True if gust is not None and gust > T["wind_gust_hold_kmh"] else None)
    assert n == 86 * 7


def test_real_wind_700hpa_ignored_below_2500(real_run_doc, T):
    """A low summit with a strong 700 hPa wind but calm lift gusts is not on hold."""
    hits = 0
    for r, by, d in _short_days(real_run_doc):
        der = d["derived"]
        sm = r["summit_elevation_m"]
        if (d["horizon"] == "short" and sm is not None and sm < T["wind_700hpa_min_summit_m"]
                and (der["wind_700hpa_lift_max_kmh"] or 0) > T["wind_700hpa_hold_kmh"]
                and der["wind_gust_lift_max_kmh"] is not None
                and der["wind_gust_lift_max_kmh"] <= T["wind_gust_hold_kmh"]):
            assert der["wind_hold"] is False, (r["id"], d["date"])
            hits += 1
    assert hits > 0, "the real run contains this case"


def test_real_bluebird_formula(real_run_doc, T):
    n = 0
    for r, by, d in _short_days(real_run_doc):
        if not 0 <= d["index"] <= 6:
            continue
        der = d["derived"]
        prev = by.get(d["index"] - 1)
        ps = ((prev or {}).get("summit") or {}).get("snowfall_cm")
        s = d["summit"] or {}
        exp = tri_and([
            None if ps is None else ps >= T["bluebird_prev24_cm"],
            None if s.get("snowfall_cm") is None else s["snowfall_cm"] < T["snowy_day_min_cm"],
            None if s.get("weather_code") is None else s["weather_code"] in T["bluebird_codes"],
            None if der["cloud_cover_lift_mean_pct"] is None else der["cloud_cover_lift_mean_pct"] <= T["bluebird_cloud_mean_pct"],
            None if der["flat_light_hours"] is None else der["flat_light_hours"] <= T["bluebird_flat_light_max_h"],
        ])
        assert der["bluebird"] == exp, (r["id"], d["date"])
        n += 1
    assert n == 86 * 7


def test_real_bluebird_days_flag(real_run_doc):
    for r in real_run_doc["resorts"]:
        assert r["flags"]["bluebird_days"] == [d["index"] for d in r["days"] if d["derived"]["bluebird"] is True]


def test_real_ski_score_formula(f, real_run_doc):
    started = datetime.fromisoformat(real_run_doc["run"]["started_utc"].replace("Z", "+00:00"))
    for r in real_run_doc["resorts"]:
        by = {d["index"]: d for d in r["days"]}
        key = "mid" if r["mid_elevation_m"] is not None else "summit"
        H = (started + timedelta(seconds=r["utc_offset_s"])).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        for i in range(7):
            d = by[i]
            der = d["derived"]
            prev = by.get(i - 1)
            ps = ((prev or {}).get("summit") or {}).get("snowfall_cm")
            lh = der["lift_hours_snowfall_cm"]
            exp = None if ps is None or lh is None else round(ps + lh, 1)
            prev_rr = (prev or {}).get("derived", {}).get("rain_risk", {}).get(key)
            if der["wind_hold"] is True or der["rain_risk"][key] is True or prev_rr is True:
                exp = None
            if i == 0 and (d["sunset_local"] is None or H > datetime.fromisoformat(d["sunset_local"])):
                exp = None
            assert der["ski_score_cm"] == exp, (r["id"], i)
            if der["wind_hold"] is True:
                assert der["ski_score_cm"] is None   # spec test 6


def spec_storm(days, use_summit, T):
    e = "summit" if use_summit else "base"
    by = {d["index"]: d for d in days}

    def snow(i):
        return ((by.get(i) or {}).get(e) or {}).get("snowfall_cm")

    s = [snow(1), snow(2), snow(3)]
    conds = [None if any(x is None for x in s) else sum(s) >= T["storm_3day_cm"]]
    reasons = [1] if conds[0] is True else []
    for i in range(7):
        dp = (by.get(i) or {}).get(e)
        if not dp:
            conds.append(None)
            continue
        c = tri_and([None if dp["weather_code"] is None else dp["weather_code"] in T["storm_codes"],
                     None if dp["precip_prob_pct"] is None else dp["precip_prob_pct"] >= T["storm_prob_pct"],
                     None if dp["snowfall_cm"] is None else dp["snowfall_cm"] >= T["storm_day_min_cm"]])
        conds.append(c)
        if c is True:
            reasons.append(i)
    return tri_or(conds), reasons


def test_real_storm_flags(real_run_doc, T):
    n_true = 0
    for r in real_run_doc["resorts"]:
        exp, reasons = spec_storm(r["days"], r["summit_elevation_m"] is not None, T)
        assert r["flags"]["storm_incoming"] == exp, r["id"]
        assert [x["index"] for x in r["flags"]["storm_reasons"]] == reasons, r["id"]
        n_true += exp is True
    assert 0 < n_true < 86


def test_storm_code_alone_is_not_a_storm(f, T, real_run_doc):
    """A heavy-snow code with 0 cm must not flag a storm (spec 1.11 QA note)."""
    import copy
    r = copy.deepcopy(next(x for x in real_run_doc["resorts"] if x["summit_elevation_m"] is not None))
    for d in r["days"]:
        if d["summit"] is not None:
            d["summit"].update({"weather_code": 75, "precip_prob_pct": 90, "snowfall_cm": 0.0})
    flags = f.storm_flags(r["days"], use_summit=True)
    assert flags["storm_incoming"] is False
    assert flags["storm_reasons"] == []


def test_storm_unknown_when_snow_missing(f, real_run_doc):
    import copy
    r = copy.deepcopy(next(x for x in real_run_doc["resorts"] if x["summit_elevation_m"] is not None))
    for d in r["days"]:
        if d["summit"] is not None:
            d["summit"].update({"weather_code": 3, "precip_prob_pct": 10, "snowfall_cm": None})
    flags = f.storm_flags(r["days"], use_summit=True)
    assert flags["storm_incoming"] is None


def test_real_week_and_extended(f, real_run_doc):
    for r in real_run_doc["resorts"]:
        week, ext = f.week_summary(r["days"])
        assert week == r["week"], r["id"]
        assert ext == r["extended"], r["id"]
        by = {d["index"]: d for d in r["days"]}
        svals = [by[i]["summit"]["snowfall_cm"] if by[i]["summit"] else None for i in range(7)]
        if all(v is not None for v in svals) and any(v > 0 for v in svals):
            assert r["week"]["best_snow_day"] == svals.index(max(svals))
        else:
            assert r["week"]["best_snow_day"] is None


def test_real_rain_and_mixed_days(real_run_doc):
    for r in real_run_doc["resorts"]:
        key = "mid" if r["mid_elevation_m"] is not None else "summit"
        by = {d["index"]: d for d in r["days"]}
        assert r["flags"]["rain_days"] == [i for i in range(7) if by[i]["derived"]["rain_risk"][key] is True]
        assert r["flags"]["mixed_days"] == [i for i in range(7) if by[i]["derived"]["mixed_risk"][key] is True]


# ----------------------------------------------------------------------------- model agreement


def spec_agree_through(days):
    k = None
    for i, d in enumerate(sorted(days, key=lambda x: x["index"])):
        vs = [x for x in d["snowfall_cm"].values() if x is not None]
        if d["index"] != i or len(vs) < 2 or max(vs) - min(vs) > max(3.0, 0.3 * max(vs)):
            break
        k = i
    return k


def test_model_agreement_on_recorded_models_call(f):
    c = load_fixture("om_models_probe.json")
    today = date(2026, 11, 27)
    snowbird = f.model_agreement(c[0], ["best_match", "ecmwf_ifs025"], today)
    zermatt = f.model_agreement(c[1], ["best_match", "ecmwf_ifs025", "icon_d2"], today)
    # Snowbird: 7.63 against 2.24 cm on day 1 differs by more than max(3, 2.3) -> agree through day 0
    assert snowbird["agree_through"] == spec_agree_through(snowbird["days"]) == 0
    # Zermatt: icon_d2 runs out after day 1 but two models remain, all within 3 cm
    assert zermatt["agree_through"] == spec_agree_through(zermatt["days"]) == 6
    assert zermatt["days"][3]["snowfall_cm"] == {"best_match": 2.0, "ecmwf_ifs025": 1.8, "icon_d2": None}
    # a model not compared for this resort is null, never borrowed from another
    assert all(d["snowfall_cm"]["icon_d2"] is None for d in snowbird["days"])
