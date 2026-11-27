"""Precipitation phase (spec 1.9.2) on recorded Open-Meteo hourly payloads.

Regression cases come from docs/snowline_decision.md and spec 1.9.2:
- Crystal Mountain, Sunday 29 Nov: summit above the line (mixed, no rain risk),
  mid with rain risk, base rain; the summit cell's "rain" is not counted as rain.
- Timberline, Monday 30 Nov: the freezing level never rises above 1,970 m, yet the
  cell calls most of its water "rain". No band at or above the line may be
  classified as raining.
- Timberline, Friday 27 Nov: the "8,100 ft" figure was the peak freezing level
  minus 250 m. rain_limit_m stays that classification threshold (it is kept for
  compatibility by docs/snowline_fetcher_fixes.md item 7) and is never written
  into snowfall_height_*.
"""

from datetime import date, datetime

import pytest

from conftest import RUN_UTC, load_fixture

CRYSTAL = {"base": 1341, "mid": 1736, "summit": 2131}
TIMBERLINE = {"base": 1829, "mid": 2216, "summit": 2603}


@pytest.fixture(scope="module")
def crystal(f):
    pts = load_fixture("om_hourly_crystal_bands.json")
    assert [p["elevation"] for p in pts] == [1341.0, 1736.0, 2131.0]
    return {"base": f.Hourly(pts[0]), "mid": f.Hourly(pts[1]), "summit": f.Hourly(pts[2]), "raw": pts}


@pytest.fixture(scope="module")
def timberline(f):
    pts = load_fixture("om_hourly_timberline_bands.json")
    # The recorded cells sit at 1,219 / 1,911 / 2,603 m. Phase uses only the cell's water and the
    # column freezing level plus the band height passed in, so the resort's band heights apply.
    return {"base": f.Hourly(pts[0]), "mid": f.Hourly(pts[1]), "summit": f.Hourly(pts[2]), "raw": pts}


def test_resort_band_heights_match_fixtures(resorts_by_name):
    c = resorts_by_name["Crystal Mountain"]
    t = resorts_by_name["Timberline"]
    assert (c["base_m"], c["mid_m"], c["summit_m"]) == tuple(CRYSTAL.values())
    assert (t["base_m"], t["mid_m"], t["summit_m"]) == tuple(TIMBERLINE.values())


# ----------------------------------------------------------------------------- hour_split


def test_threshold_snowline_offset_is_250(f, real_run_doc):
    assert f.THRESHOLDS["snowline_below_fl_m"] == 250
    assert real_run_doc["thresholds"]["snowline_below_fl_m"] == 250


def test_hour_split_real_hour_above_line(f, crystal):
    # Crystal mid, Sun 29 Nov 15:00: 4.8 mm, 1.6 mm of it "rain", freezing level 1,920 m.
    h = crystal["mid"]
    t = "2026-11-29T15:00"
    p, rain, fl, sh = (h.get(k, t) for k in ("precipitation", "rain", "freezing_level_height", "snowfall_height"))
    assert (p, rain, fl, sh) == (4.8, 1.6, 1920.0, None)
    rain_mm, mixed, snow = f.hour_split(1736, p, rain, fl, sh)
    # line = 1,920 - 250 = 1,670 m <= 1,736 m: liquid is mixed, frozen is snow
    assert rain_mm == 0.0
    assert mixed == pytest.approx(1.6)
    assert snow == pytest.approx(3.2)


def test_hour_split_real_hour_below_line(f, crystal):
    # Same hour at the base (1,341 m < 1,670 m line): the liquid is rain, the frozen part is mixed.
    h = crystal["base"]
    t = "2026-11-29T15:00"
    p, rain, fl = (h.get(k, t) for k in ("precipitation", "rain", "freezing_level_height"))
    rain_mm, mixed, snow = f.hour_split(1341, p, rain, fl, None)
    assert rain_mm == pytest.approx(min(rain, p))
    assert mixed == pytest.approx(p - min(rain, p))
    assert snow == 0.0


def test_hour_split_line_boundary_counts_as_snow_side(f):
    # elev == line: "otherwise" branch of the spec (elev >= line)
    assert f.hour_split(1750, 2.0, 0.5, 2000, None) == (0.0, 0.5, 1.5)
    assert f.hour_split(1749, 2.0, 0.5, 2000, None) == (0.5, 1.5, 0.0)


def test_hour_split_prefers_snowfall_height(f):
    # Europe: snowfall_height given, freezing level ignored even if it would say otherwise.
    assert f.hour_split(2000, 1.0, 1.0, 1500, 2100) == (1.0, 0.0, 0.0)
    assert f.hour_split(2000, 1.0, 1.0, 3000, 1900) == (0.0, 1.0, 0.0)


def test_hour_split_rain_above_precip_is_clamped(f):
    rain_mm, mixed, snow = f.hour_split(1000, 1.0, 1.3, 3000, None)
    assert (rain_mm, mixed, snow) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("args", [
    (1000, None, 0.5, 2000, None),
    (1000, 1.0, None, 2000, None),
    (1000, 1.0, 0.5, None, None),
])
def test_hour_split_missing_input_is_none(f, args):
    assert f.hour_split(*args) is None


def test_phase_summary_null_hour_makes_period_null(f, crystal):
    """A wet hour with no freezing level and no snowfall height nulls the whole period (no filling)."""
    import copy
    pt = copy.deepcopy(crystal["raw"][2])
    hrs = pt["hourly"]
    i = hrs["time"].index("2026-11-29T15:00")
    assert hrs["precipitation"][i] > 0
    hrs["freezing_level_height"][i] = None
    hrs["snowfall_height"][i] = None
    h = f.Hourly(pt)
    ps = f.phase_summary(h, h.day_times(date(2026, 11, 29)), 2131)
    assert set(ps.values()) == {None}


# ----------------------------------------------------------------------------- Crystal regression


def _day(f, h, ds, elev):
    return f.phase_summary(h, h.day_times(date.fromisoformat(ds)), elev)


def test_crystal_sunday_summit_mixed_without_rain_risk(f, crystal):
    ps = _day(f, crystal["summit"], "2026-11-29", CRYSTAL["summit"])
    assert ps["phase"] == "mixed"
    assert ps["rain_risk"] is False
    assert ps["mixed_risk"] is True
    assert ps["rain_mm"] == 0.0


def test_crystal_sunday_summit_cell_rain_is_not_counted_as_rain(f, crystal):
    """Spec 1.9.2: the 2,131 m summit got 11.8 mm of cell "rain"; the column says it is above the line."""
    h = crystal["summit"]
    ts = h.day_times(date(2026, 11, 29))
    cell_rain = round(sum(h.series("rain", ts)), 1)
    assert cell_rain == 11.8
    ps = f.phase_summary(h, ts, CRYSTAL["summit"])
    assert ps["rain_mm"] == 0.0
    assert ps["mixed_mm"] == 11.8


def test_crystal_sunday_mid_has_rain_risk(f, crystal):
    ps = _day(f, crystal["mid"], "2026-11-29", CRYSTAL["mid"])
    assert ps["rain_risk"] is True
    assert ps["phase"] == "mixed"


def test_crystal_sunday_base_is_rain(f, crystal):
    ps = _day(f, crystal["base"], "2026-11-29", CRYSTAL["base"])
    assert ps["phase"] == "rain"
    assert ps["rain_risk"] is True


def test_crystal_sunday_matches_the_real_run(f, crystal, real_run_doc):
    """The 12:15 UTC run's file says the same phase words for Sunday as the recording gives."""
    r = next(x for x in real_run_doc["resorts"] if x["id"] == "crystal-mountain")
    day = next(d for d in r["days"] if d["date"] == "2026-11-29")
    for band in ("base", "mid", "summit"):
        ps = _day(f, crystal[band], "2026-11-29", CRYSTAL[band])
        assert day["derived"]["phase"][band] == ps["phase"], band
        assert day["derived"]["rain_risk"][band] == ps["rain_risk"], band


def test_phase_totals_conserve_water(f, crystal):
    for band, elev in CRYSTAL.items():
        h = crystal[band]
        for ds in ("2026-11-27", "2026-11-29", "2026-11-30"):
            ts = h.day_times(date.fromisoformat(ds))
            ps = f.phase_summary(h, ts, elev)
            total = sum(h.series("precipitation", ts))
            assert ps["rain_mm"] + ps["mixed_mm"] + ps["snow_mm"] == pytest.approx(total, abs=0.16), (band, ds)


def test_dry_day_phase_none(f, crystal):
    ps = _day(f, crystal["summit"], "2026-11-28", CRYSTAL["summit"])
    assert ps["phase"] == "none"
    assert ps["rain_risk"] is False and ps["mixed_risk"] is False


def test_significance_needs_both_amount_and_share(f):
    """phase_min_mm (1.0) AND phase_share_min (0.2): 1.2 mm of rain in 10 mm is 12%, not significant."""
    hours = [f"2026-11-29T{h:02d}:00" for h in range(24)]
    precip = [0.0] * 24
    rain = [0.0] * 24
    precip[10], rain[10] = 1.2, 1.2      # rain at the base: line 1,750 > 1,341
    precip[11], rain[11] = 8.8, 0.0      # all frozen: mixed at the base
    fl = [2000.0] * 24
    resp = {"hourly": {"time": hours, "precipitation": precip, "rain": rain,
                       "freezing_level_height": fl, "snowfall_height": [None] * 24}}
    ps = f.phase_summary(f.Hourly(resp), hours, 1341)
    assert ps["rain_mm"] == 1.2
    assert ps["rain_risk"] is False
    assert ps["phase"] == "mixed"
    # now make the rain 2.0 mm of 10 (20%): significant
    precip[10], rain[10], precip[11] = 2.0, 2.0, 8.0
    ps = f.phase_summary(f.Hourly(resp), hours, 1341)
    assert ps["rain_risk"] is True and ps["phase"] == "mixed"


# ----------------------------------------------------------------------------- Timberline regression


def test_timberline_monday_freezing_level_stays_below_bands(timberline):
    for band in ("base", "mid", "summit"):
        h = timberline[band]
        ts = h.day_times(date(2026, 11, 30))
        wet = [t for t in ts if h.get("precipitation", t) > 0]
        assert max(h.get("freezing_level_height", t) for t in wet) == 1970.0
        # the cell still calls most of the water rain
        assert sum(h.series("rain", ts)) > 0.8 * sum(h.series("precipitation", ts))


@pytest.mark.parametrize("band", ["base", "mid", "summit"])
def test_timberline_monday_no_rain_at_or_above_line(f, timberline, band):
    ps = _day(f, timberline[band], "2026-11-30", TIMBERLINE[band])
    assert ps["rain_mm"] == 0.0
    assert ps["rain_risk"] is False
    assert ps["phase"] in ("mixed", "snow")


def test_timberline_monday_real_run_has_no_rain_flag(real_run_doc):
    r = next(x for x in real_run_doc["resorts"] if x["id"] == "timberline")
    day = next(d for d in r["days"] if d["date"] == "2026-11-30")
    assert day["derived"]["freezing_level_precip_max_m"] == 1970
    assert day["derived"]["rain_risk"] == {"base": False, "mid": False, "summit": False}
    assert day["derived"]["rain_limit_m"] is None
    assert day["index"] == 3
    assert 3 not in r["flags"]["rain_days"]
    assert 2 in r["flags"]["rain_days"]   # Sunday: freezing level to 2,670 m, rain at mid is real


@pytest.fixture(scope="module")
def timberline_days(f):
    pts = load_fixture("om_hourly_timberline_bands.json")
    labels = {"daily_base": pts[0], "daily_mid": pts[1], "daily_summit": pts[2],
              "hourly_base": pts[0], "hourly_mid": pts[1], "hourly_summit": pts[2]}
    H = f.local_hour(RUN_UTC, pts[0]["utc_offset_seconds"])
    return {d["date"]: d for d in f.build_days(labels, dict(TIMBERLINE), RUN_UTC, H)}


def test_timberline_friday_rain_limit_is_fl_minus_250(f, timberline, timberline_days):
    """rain_limit_m = max(line) over base hours with rain >= 0.1 where the base is below the line."""
    d = timberline_days["2026-11-27"]["derived"]
    h = timberline["base"]
    lines = []
    for t in h.day_times(date(2026, 11, 27)):
        if h.get("rain", t) >= 0.1:
            line = h.get("freezing_level_height", t) - 250
            if TIMBERLINE["base"] < line:
                lines.append(line)
    assert lines, "the recorded Friday has rain hours below the line at the base"
    assert d["rain_limit_m"] == round(max(lines))
    assert d["rain_limit_kind"] == "freezing_level_minus_250"
    assert d["rain_limit_m"] <= d["freezing_level_max_m"] - 250


def test_us_snowfall_height_never_filled_from_freezing_level(timberline_days):
    """The US models have no snowfall_height: the fields stay null; the line is never stored there."""
    for ds, day in timberline_days.items():
        der = day["derived"]
        if day["horizon"] != "short":
            continue
        assert der["snowfall_height_precip_min_m"] is None, ds
        assert der["snowfall_height_precip_max_m"] is None, ds


def test_real_run_rain_limit_below_precip_freezing_level(real_run_doc):
    """Across all 86 resorts: a fallback rain limit is at most the wet-hour freezing level minus 250 m."""
    checked = 0
    for r in real_run_doc["resorts"]:
        for day in r["days"]:
            der = day["derived"]
            if der.get("rain_limit_kind") == "freezing_level_minus_250":
                assert der["rain_limit_m"] <= der["freezing_level_precip_max_m"] - 250, (r["id"], day["date"])
                checked += 1
    assert checked > 20


def test_extended_days_rain_risk_false_only_when_dry(real_run_doc):
    """Spec: extended rain_risk.E is false when rain_mm == 0, else null (never true)."""
    for r in real_run_doc["resorts"]:
        for day in r["days"]:
            if day["horizon"] != "extended":
                continue
            for e in ("base", "mid", "summit"):
                rr = day["derived"]["rain_risk"][e]
                dp = day[e]
                if dp is None:
                    continue
                if dp["rain_mm"] is None:
                    assert rr is None
                elif dp["rain_mm"] > 0:
                    assert rr is None, (r["id"], day["date"], e)
                else:
                    assert rr is False, (r["id"], day["date"], e)


def test_local_hour_floors_to_hour(f):
    H = f.local_hour(RUN_UTC, -28800)
    assert H == datetime(2026, 11, 27, 4, 0)
