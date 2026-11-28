"""The powder upgrades (docs/powder_validation_report.md, upgrades 1 and 2,
defects 1 and 2), on real dashboard code with synthetic resort files.

A synthetic file is a real fixture here: plain JSON shaped exactly like
data/forecasts_v2.json (schema 3), carrying the new optional fields the
fetcher adds. No mocks, no monkeypatching. Every test runs the page's own
gate functions and its own renderer.

Field contract under test:
  derived.slr_kuchera / snow_kuchera_cm / storm_profile  {band: value|null}
  derived.post_gust_max_kmh                              number|null (or {band: ...})
  past days keep temp_max_c, gust_max_kmh, derived.powder_temp_c and rain_risk
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from conftest import D, LEAK, P, inline_text, real_data, usable_resorts, visible

NEW_KEYS = ("slr_kuchera", "snow_kuchera_cm", "storm_profile", "post_gust_max_kmh")
BANDS = ("base", "mid", "summit")
NOW = datetime.now(timezone.utc)


# ----------------------------------------------------------------------------
# A synthetic resort: cold, calm, all-snow, every input present
# ----------------------------------------------------------------------------

def _band(snow=0.0, precip=None, tmax=-9.0, gust=20, rain=0.0):
    return {"snowfall_cm": snow, "rain_mm": rain,
            "precip_mm": round(snow / 0.7, 2) if precip is None else precip,
            "temp_max_c": tmax, "temp_min_c": tmax - 6, "gust_max_kmh": gust}


def _day(i, snow=0.0, horizon="short", pt=-10.0, **kw):
    today = NOW.date()
    b = _band(snow, **kw)
    return {
        "date": (today + timedelta(days=i)).isoformat(), "index": i, "horizon": horizon,
        "base": dict(b), "mid": dict(b), "summit": dict(b),
        "derived": {
            "rain_risk": {k: False for k in BANDS}, "mixed_risk": {k: False for k in BANDS},
            "phase": {k: "snow" if snow else None for k in BANDS},
            "powder_temp_c": pt if snow else None, "wind_hold": False,
        },
    }


def _resort(rid="synthetic", name="Synth Peak"):
    days = [_day(i, horizon="past_model" if i < 0 else "short") for i in range(-3, 7)]
    return {
        "id": rid, "name": name, "country": "USA", "region": "Test",
        "lat": 40.0, "lon": -111.0, "timezone": "UTC", "utc_offset_s": 0,
        "elevation_m": 2000, "mid_elevation_m": 2500, "summit_elevation_m": 3000,
        "days": days,
        "models": {"compared": ["best_match", "ecmwf_ifs025"],
                   "days": [{"index": i, "snowfall_cm": {"best_match": 0.0, "ecmwf_ifs025": 0.0}}
                            for i in range(7)]},
    }


def day(r, i):
    return D.day_by_index(r, i)


def set_models(r, i, cm):
    row = next(x for x in r["models"]["days"] if x["index"] == i)
    row["snowfall_cm"] = {"best_match": cm, "ecmwf_ifs025": cm}


def snow_day(r, i, cm, **kw):
    """Make day i of r a cold, calm snow day of cm at every band (7:1)."""
    new = _day(i, cm, horizon=day(r, i)["horizon"], **kw)
    r["days"][[d["index"] for d in r["days"]].index(i)] = new
    if 0 <= i < 7:
        set_models(r, i, cm)
    return new


def kuchera(d, slr, bands=BANDS):
    """Add the fetcher's Kuchera fields for the named bands, from the day's
    own 7:1 snow: frozen water = snow / 0.7, Kuchera cm = water x slr / 10."""
    dv = d["derived"]
    dv["slr_kuchera"] = {b: slr for b in bands}
    dv["snow_kuchera_cm"] = {b: round(d[b]["snowfall_cm"] / 0.7 * slr / 10, 1) for b in bands}
    dv["storm_profile"] = {b: "rsu" for b in bands}


# ----------------------------------------------------------------------------
# Upgrade 1: the Kuchera amount gate
# ----------------------------------------------------------------------------

def test_kuchera_passes_amount_gate_where_7to1_fails():
    """Big Sky shape: 10.7 mm at -10 C, 7.5 cm at 7:1, ~16 cm at 15:1."""
    r = _resort()
    d = snow_day(r, 1, 7.5)
    s = D.powder_score(r, d)
    assert s["ok"] is False and s["gate"] == "amount"
    assert P.forecast_row(r) is None

    kuchera(d, 15.0)
    s = D.powder_score(r, d)
    assert s["ok"] is True and s["kuchera_cm"] == pytest.approx(16.1)
    assert s["amount_cm"] == s["kuchera_cm"]
    assert s["snow_cm"] == 7.5            # the 7:1 figure is untouched
    assert s["ff"] == pytest.approx(1.0, abs=0.01)  # frozen fraction stays on 7:1
    row = P.forecast_row(r)
    assert row is not None and row["index"] == 1
    assert row["cm"] == 7.5               # the daily-table / resort-page figure stays 7:1
    assert row["page_cm"] == 7.5          # displayed amount is the same 7:1 figure (one number per day)


def test_kuchera_can_fail_a_storm_7to1_passes():
    """A warm storm under 7:1 must clear more, not be waved through."""
    r = _resort()
    d = snow_day(r, 2, 11.0)
    assert D.powder_score(r, d)["ok"] is True
    kuchera(d, 5.0)   # 15.7 mm x 5 / 10 = 7.9 cm
    s = D.powder_score(r, d)
    assert s["ok"] is False and s["gate"] == "amount"


def test_kuchera_never_crosses_bands():
    """The powder band is mid; a summit-only estimate is not read for mid."""
    r = _resort()
    d = snow_day(r, 1, 7.5)
    kuchera(d, 15.0, bands=("summit",))
    assert D.powder_band_key(r) == "mid"
    assert D.kuchera_cm(d, "mid") is None
    assert D.powder_score(r, d)["gate"] == "amount"


@pytest.mark.parametrize("shape", ["absent", "null", "bare_number", "null_slr", "negative_slr"])
def test_kuchera_missing_falls_back_to_7to1(shape):
    """Absent key, null, a non-dict, or an amount without its ratio: no estimate."""
    r = _resort()
    d = snow_day(r, 1, 7.5)
    kuchera(d, 15.0)
    dv = d["derived"]
    if shape == "absent":
        del dv["snow_kuchera_cm"]
    elif shape == "null":
        dv["snow_kuchera_cm"] = {"base": None, "mid": None, "summit": None}
    elif shape == "bare_number":
        dv["snow_kuchera_cm"], dv["slr_kuchera"] = 16.1, 15.0
    elif shape == "null_slr":
        dv["slr_kuchera"]["mid"] = None
    else:
        dv["slr_kuchera"]["mid"] = -1.0   # out of range: not a ratio
    s = D.powder_score(r, d)
    assert s["kuchera_cm"] is None and s["amount_cm"] == 7.5
    assert s["ok"] is False and s["gate"] == "amount"


def test_zero_ratio_is_an_estimate_not_a_fallback():
    """The fetcher floors a warm column (Tmax >= 277.16 K, a melting layer
    aloft) at SLR 0, so slr 0 / snow_kuchera_cm 0 is its real answer. Read
    as missing, the gate fell back to the larger 7:1 figure and listed the
    warmest storm of all as powder (adversarial review, 28 Nov)."""
    r = _resort()
    d = snow_day(r, 1, 12.0)            # 7:1 alone passes
    assert P.forecast_row(r) is not None
    kuchera(d, 0.0)                     # 0.0 cm at every band
    s = D.powder_score(r, d)
    assert s["kuchera_cm"] == 0.0 and s["amount_cm"] == 0.0
    assert s["ok"] is False and s["gate"] == "amount"
    assert P.forecast_row(r) is None
    assert P.resort_line(r, "cm") == ""
    # the agreement bar never divides by a zero ratio
    assert P._agree_min_cm(d, {"band": "mid", "kuchera_cm": 0.0}) == float("inf")


def test_stale_day0_is_never_listed_as_a_forecast():
    """Between a resort's midnight and the next fetch (4 a day) the file's
    day 0 is yesterday there. With day 0 now scoreable it listed as
    "powder snow on Yesterday" in the forecast group."""
    r = _resort()
    # shift every date back one day: the file was written yesterday
    for d in r["days"]:
        d["date"] = (datetime.fromisoformat(d["date"]) - timedelta(days=1)).date().isoformat()
    snow_day(r, 0, 12.0)
    day(r, 0)["date"] = (NOW.date() - timedelta(days=1)).isoformat()
    assert D.powder_score(r, day(r, 0))["ok"] is True   # the score itself still passes
    assert P.forecast_verdict(r, 0) == (None, "past day")
    assert P.forecast_row(r) is None
    h = P.page([r], "cm", None, now=NOW)
    assert "Powder forecast" not in visible(h)
    assert "esterday" not in visible(h)


def test_stale_day0_blocks_fallen_rows():
    """Fallen gates check through day 0. If day 0 is older than the
    reading's date, the reading day's hours before it went unchecked."""
    r = fallen_resort()
    r["stations"][0]["observed_utc"] = NOW.isoformat()   # read today, whatever the hour
    assert P.fallen_verdict(r, NOW)[0] is not None
    for d in r["days"]:
        d["date"] = (datetime.fromisoformat(d["date"]) - timedelta(days=1)).date().isoformat()
    row, why = P.fallen_verdict(r, NOW)
    assert row is None and why == "no data for today"


def test_model_agreement_scales_with_the_same_band_ratio():
    r = _resort()
    d = snow_day(r, 1, 7.5)
    kuchera(d, 15.0)
    # threshold = 10 x 7 / 15 = 4.67 cm of 7:1 model snow
    set_models(r, 1, 4.7)
    assert P.forecast_row(r) is not None
    set_models(r, 1, 4.6)
    assert P.forecast_row(r) is None
    assert P.forecast_verdict(r, 1) == (None, "models disagree")
    # Without Kuchera the bar is the plain 10 cm
    r2 = _resort()
    snow_day(r2, 1, 12.0)
    set_models(r2, 1, 9.9)
    assert P.forecast_row(r2) is None


def test_model_agreement_rises_above_10cm_under_7to1():
    """No floor: a 6:1 day needs 10 x 7 / 6 = 11.7 cm from every model."""
    r = _resort()
    d = snow_day(r, 1, 14.0)
    kuchera(d, 6.0)   # 20 mm x 6 / 10 = 12.0 cm passes the amount gate
    assert D.powder_score(r, d)["ok"] is True
    set_models(r, 1, 11.6)
    assert P.forecast_row(r) is None
    set_models(r, 1, 11.7)
    assert P.forecast_row(r) is not None


def test_mid_band_ratio_drives_agreement_not_summit():
    r = _resort()
    d = snow_day(r, 1, 7.5)
    kuchera(d, 15.0)
    d["derived"]["slr_kuchera"]["summit"] = 30.0   # would drop the bar to 2.3 cm
    set_models(r, 1, 3.0)
    assert P.forecast_row(r) is None


# ----------------------------------------------------------------------------
# Upgrade 2: wind after the snow stops
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("value,lists", [(41, False), (40, True), (None, True),
                                         ({"mid": 55, "summit": 10}, False),
                                         ({"mid": 10, "summit": 55}, True)])
def test_post_storm_wind_excludes_row(value, lists):
    r = _resort()
    d = snow_day(r, 1, 12.0)
    d["derived"]["post_gust_max_kmh"] = value
    row, why = P.forecast_verdict(r, 1)
    assert (row is not None) is lists
    assert why == ("" if lists else "wind after snowfall")


def test_post_storm_wind_absent_behaves_as_today():
    r = _resort()
    snow_day(r, 1, 12.0)
    assert P.forecast_verdict(r, 1)[0] is not None


# ----------------------------------------------------------------------------
# Defects 1 and 2: past days keep their fields
# ----------------------------------------------------------------------------

def fallen_resort(h72=30.0):
    """Measured new snow on days -2 and -1, all retroactive inputs present."""
    r = _resort("fallen", "Cold Bowl")
    snow_day(r, -2, 14.0)
    snow_day(r, -1, 16.0)
    r["stations"] = [{"name": "Cold Bowl", "provider": "SNOTEL",
                      "observed_utc": (NOW - timedelta(hours=3)).isoformat()}]
    r["recent"] = {"measured": {"kind": "measured_new_snow", "provider": "SNOTEL",
                                "label": "New snow at Cold Bowl station", "h72_cm": h72}}
    r["depth"] = {"value_cm": 120.0, "kind": "station", "provider": "SNOTEL"}
    return r


def test_restored_past_days_pass_retroactive_gates():
    r = fallen_resort()
    assert P.fallen_gates(r) == ([], [])
    row, why = P.fallen_verdict(r, NOW)
    assert why == "" and row["group"] == "measured" and row["cm"] == 30.0


def test_today_nulls_still_block_fallen_rows():
    """The file as it is today: past days nulled, every gate missing."""
    r = fallen_resort()
    for i in (-2, -1):
        d = day(r, i)
        for b in BANDS:
            d[b]["temp_max_c"] = d[b]["gust_max_kmh"] = None
        d["derived"] = {k: None for k in d["derived"]}
    failed, missing = P.fallen_gates(r)
    assert failed == [] and set(missing) == {"cold", "thaw", "wind"}
    assert P.fallen_verdict(r, NOW) == (None, "no data for cold, thaw, wind")


@pytest.mark.parametrize("gate,edit", [
    ("cold", lambda r: day(r, -2)["derived"].__setitem__("powder_temp_c", -1.0)),
    ("thaw", lambda r: day(r, 0)["mid"].__setitem__("temp_max_c", 2.0)),
    ("wind", lambda r: day(r, -1)["mid"].__setitem__("gust_max_kmh", 55)),
    ("rain", lambda r: day(r, 0)["mid"].__setitem__("rain_mm", 3.0)),
    ("phase", lambda r: day(r, -1)["mid"].__setitem__("precip_mm", 60.0)),
])
def test_each_retroactive_gate_blocks_its_own_violation(gate, edit):
    r = fallen_resort()
    edit(r)
    failed, missing = P.fallen_gates(r)
    assert failed == [gate] and missing == []
    assert P.fallen_verdict(r, NOW) == (None, f"failed {gate}")


def test_fallen_group_renders_per_copy_law():
    r = fallen_resort()
    for units in ("cm", "in"):
        h = P.page([r], units, None, now=NOW)
        t = visible(h)
        assert "Snow measured, 3 days" in t
        assert P.FALLEN_CAPTION in t
        lead = inline_text(re.search(r'<p class="leaders[^>]*>(.*?)</p>', h).group(1))
        assert lead == ("Cold Bowl measured 30 cm new snow." if units == "cm"
                        else "Cold Bowl measured 12″ new snow.")
        assert len(P._words(lead)) <= P.MAX_WORDS and "snow" in lead
        # a measured figure: no "~" anywhere in its row
        body = h.split("Snow measured, 3 days", 1)[1].split("</table>", 1)[0]
        assert "~" not in inline_text(body)
        assert not LEAK.search(t)


def test_day0_scoreable_when_previous_day_rain_risk_exists():
    r = _resort()
    snow_day(r, 0, 12.0)
    snow_day(r, 1, 12.0)   # a rival day, so today has to win on its own
    for b in BANDS:
        day(r, 1)["mid"]["gust_max_kmh"] = 60     # day 1 is windy: not powder
    prev = day(r, -1)
    s = D.powder_score(r, day(r, 0))
    assert s["ok"] is True
    row = P.forecast_row(r)
    assert row["index"] == 0
    h = P.page([r], "cm", None, now=NOW)
    lead = inline_text(re.search(r'<p class="leaders[^>]*>(.*?)</p>', h).group(1))
    assert lead == "Synth Peak: dry powder snow today, ~12 cm."
    assert len(P._words(lead)) <= P.MAX_WORDS

    # Today's file: day -1 derived nulled, so day 0 makes no claim
    prev["derived"]["rain_risk"] = None
    assert D.powder_score(r, day(r, 0)) is None
    assert P.forecast_verdict(r, 0)[0] is None

    # And a rained-on day -1 blocks it
    prev["derived"]["rain_risk"] = {b: True for b in BANDS}
    assert D.powder_score(r, day(r, 0))["gate"] == "rained_before"


# ----------------------------------------------------------------------------
# The page with the new fields, and the real file without them
# ----------------------------------------------------------------------------

def test_kuchera_row_renders_with_model_mark():
    r = _resort()
    d = snow_day(r, 2, 7.5)
    kuchera(d, 15.0)
    for units in ("cm", "in"):
        h = P.page([r], units, None, now=NOW)
        t = visible(h)
        assert "Powder forecast, 4 days" in t
        body = h.split("Powder forecast, 4 days", 1)[1]
        cells = re.findall(r'<td class="r">(.*?)</td>', body)
        assert cells and all(inline_text(c).startswith("~") for c in cells)
        assert not LEAK.search(t)
    assert P.resort_line(r, "cm").startswith("Dry powder snow ")


def test_kuchera_passed_row_displays_the_shared_7to1_figure():
    """Display ruling (final): one estimate, one number per day, product-wide.
    A row the Kuchera gate rescued still displays the 7:1 figure every other
    page shows, so the same day can never carry two different "~" amounts."""
    r = _resort()
    d = snow_day(r, 1, 7.5)   # 7:1 gives 7.5 cm; the internal bar is 10
    kuchera(d, 15.0)          # Kuchera estimate 16.1 cm qualifies the day
    row = P.forecast_row(r)
    assert row is not None                               # Kuchera decided the gate...
    assert row["page_cm"] == row["cm"] == 7.5            # ...but the display stays 7:1
    h = P.page([r], "cm", None, now=NOW)
    lead = inline_text(re.search(r'<p class="leaders[^>]*>(.*?)</p>', h).group(1))
    assert "~8" in lead and "16" not in lead             # 7.5 rounds to the shown ~8
    assert len(P._words(lead)) <= P.MAX_WORDS and "snow" in lead
    body = h.split("Powder forecast, 4 days", 1)[1].split("</table>", 1)[0]
    cells = [inline_text(c) for c in re.findall(r'<td class="r">(.*?)</td>', body)]
    assert cells and all(c.startswith("~") and "16" not in c for c in cells)
    # and the resort page agrees with the powder page on the same figure
    line = inline_text(P.resort_line(r, "cm"))
    assert line.startswith("Dry powder snow ") and "~8" in line and "16" not in line


def test_7to1_passed_row_is_unchanged():
    r = _resort()
    snow_day(r, 1, 12.0)
    row = P.forecast_row(r)
    assert row["page_cm"] == row["cm"] == 12.0
    h = P.page([r], "cm", None, now=NOW)
    lead = inline_text(re.search(r'<p class="leaders[^>]*>(.*?)</p>', h).group(1))
    assert lead == "Synth Peak: dry powder snow tomorrow, ~12 cm."


def test_one_day_one_number_across_pages():
    """No build may ever show two different amounts for the same powder day
    on different pages — the Crystal Mountain class of defect."""
    builds = [
        lambda r: kuchera(snow_day(r, 1, 7.5), 15.0),   # Kuchera-rescued
        lambda r: kuchera(snow_day(r, 1, 12.0), 6.0),   # warm ratio still passing
        lambda r: snow_day(r, 1, 12.0),                 # plain 7:1
    ]
    for build in builds:
        r = _resort()
        build(r)
        row = P.forecast_row(r)
        assert row is not None and row["page_cm"] == row["cm"]
        h = P.page([r], "cm", None, now=NOW)
        body = h.split("Powder forecast, 4 days", 1)[1].split("</table>", 1)[0]
        cell_nums = {float(re.search(r"[\d.]+", inline_text(c)).group())
                     for c in re.findall(r'<td class="r">(.*?)</td>', body)}
        line_num = float(re.search(r"[\d.]+", P.resort_line(r, "cm")).group())
        assert cell_nums == {line_num}


def _strip_new(data):
    for r in data["resorts"]:
        for d in r.get("days") or []:
            dv = d.get("derived") if isinstance(d, dict) else None
            if isinstance(dv, dict):
                for k in NEW_KEYS:
                    dv.pop(k, None)
    return data


@pytest.mark.parametrize("units", ["cm", "in"])
def test_real_file_unchanged_without_new_fields(units):
    """The site cannot break between deploys: with the new keys stripped the
    page is byte-identical to the file as shipped, and every powder score
    keeps its 7:1 amount."""
    real = usable_resorts(real_data())
    stripped = usable_resorts(_strip_new(real_data()))
    assert P.page(real, units, None, now=NOW) == P.page(stripped, units, None, now=NOW)
    for r in stripped:
        for i in P.FORECAST_DAYS:
            s = D.powder_score(r, D.day_by_index(r, i))
            if s:
                assert s["kuchera_cm"] is None and s["amount_cm"] == s["snow_cm"]


def test_real_resort_with_kuchera_injected_renders():
    """A real resort, every short day given Kuchera fields: no crash, no leaks."""
    data = real_data()
    for r in usable_resorts(data):
        for d in r.get("days") or []:
            if isinstance(d, dict) and d.get("horizon") == "short" and isinstance(d.get("derived"), dict):
                bands = [b for b in BANDS if isinstance(d.get(b), dict)
                         and D.is_num(d[b].get("snowfall_cm"))]
                kuchera(d, 14.0, bands)
                d["derived"]["post_gust_max_kmh"] = None
    rs = usable_resorts(data)
    for units in ("cm", "in"):
        t = visible(P.page(rs, units, None, now=NOW))
        assert not LEAK.search(t)
    json.dumps(copy.deepcopy(data))  # still plain JSON
