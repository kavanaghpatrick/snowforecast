"""Data loading and pure helpers for the v2 dashboard.

The forecast file (schema version 3) is the only input. Nothing here invents a
value: every helper returns None when its input is None, and zero stays zero.
"""

from __future__ import annotations

import json
import math
import os
import time
import unicodedata
import urllib.request
import zlib
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/kavanaghpatrick/snowforecast/main/data/forecasts_v2.json"
)
LOCAL_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "forecasts_v2.json"

SCOPES = ("nearby", "country", "all")
SORTS = ("next3", "week", "best", "depth", "name")
UNITS = ("cm", "in")
COUNTRY_ORDER = ("USA", "Austria", "Switzerland", "France", "Spain", "Japan")


def name_key(s):
    """A-Z key that ignores accents and case, so Méribel sorts before Morzine
    and Val d'Isère before Val Thorens."""
    s = unicodedata.normalize("NFKD", s if isinstance(s, str) else "")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _parse_utc(value):
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _usable(obj) -> bool:
    if not isinstance(obj, dict) or obj.get("schema_version") != 3:
        return False
    if obj.get("fixture") and os.environ.get("SNOW_FIXTURE") != "1":
        return False
    resorts = obj.get("resorts")
    return isinstance(resorts, list) and any(
        isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"] for r in resorts)


GITHUB_SOCKET_TIMEOUT_S = 4   # the bundled local copy is always there as a fallback
GITHUB_DEADLINE_S = 15        # whole read, so a slow trickle cannot hold a visitor
GITHUB_MAX_BYTES = 8 * 1024 * 1024


def _read_github(url=GITHUB_DATA_URL):
    try:
        # gzip: the 3 MB file is about 280 KB on the wire.
        req = urllib.request.Request(url, headers={"User-Agent": "snowforecast-dashboard/3",
                                                   "Accept-Encoding": "gzip"})
        deadline = time.monotonic() + GITHUB_DEADLINE_S
        chunks, size = [], 0
        with urllib.request.urlopen(req, timeout=GITHUB_SOCKET_TIMEOUT_S) as resp:
            encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
            while True:
                if time.monotonic() > deadline:
                    return None
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > GITHUB_MAX_BYTES:
                    return None
                chunks.append(chunk)
        body = b"".join(chunks)
        if encoding == "gzip":
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            body = d.decompress(body, GITHUB_MAX_BYTES + 1)
            if not d.eof or len(body) > GITHUB_MAX_BYTES:   # truncated, or bigger than allowed
                return None
        elif encoding not in ("", "identity"):
            return None
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _read_local():
    try:
        with open(LOCAL_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load():
    """Return (data, origin, error).

    origin is "github" or "local"; error is None, "missing" or "old".
    """
    candidates = []
    for origin, obj in (("github", _read_github()), ("local", _read_local())):
        if obj is not None:
            candidates.append((origin, obj))
    if not candidates:
        return None, None, "missing"
    usable = [(o, d) for o, d in candidates if _usable(d)]
    if not usable:
        return None, None, "old"
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    origin, obj = max(usable, key=lambda od: _parse_utc(od[1].get("updated")) or epoch)
    return obj, origin, None


# --------------------------------------------------------------------------
# Safe access
# --------------------------------------------------------------------------

def g(obj, *path):
    """Walk dicts/lists; return None as soon as anything is missing."""
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(key, int):
            if not isinstance(cur, list) or not -len(cur) <= key < len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
    return cur


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def day_by_index(resort, index):
    days = resort.get("days")
    for d in days if isinstance(days, list) else []:
        if isinstance(d, dict) and d.get("index") == index:
            return d
    return None


def has_summit(resort) -> bool:
    return resort.get("summit_elevation_m") is not None


BAND_ELEV_KEY = {"base": "elevation_m", "mid": "mid_elevation_m", "summit": "summit_elevation_m"}
BAND_WORD = {"base": "base", "mid": "mid-mountain", "summit": "summit"}


def hero_elevation(resort):
    """The band every headline and ranking number speaks for: mid, else
    summit, else base (docs/snowline_decision.md). Summit flatters wet
    resorts, where the top is the only band above the snow line."""
    for band in ("mid", "summit", "base"):
        if resort.get(BAND_ELEV_KEY[band]) is not None and is_num(g(resort, "next_72h", f"{band}_cm")):
            return band
    return "base"


# --------------------------------------------------------------------------
# Rain the model's own column calls snow (docs/snowline_decision.md)
#
# Open-Meteo splits rain from snow at the grid cell's own terrain height, not at
# the height we ask for. When a band sits at or above the freezing level in every
# wet hour of a day, and its own hour-by-hour phase has no rain, the cell's
# "rain" there is snow by the same model's temperatures. We count that water as
# snow at Open-Meteo's own 0.7 cm per mm. Nothing is estimated: the water and
# the freezing level are both the model's. Any other day is left as the model
# gave it.
# --------------------------------------------------------------------------

SNOW_CM_PER_MM = 0.7   # Open-Meteo's own snowfall factor (median 0.70 in this file)
RECOUNT_MIN_MM = 1.0   # less liquid than this is not worth recounting


def _recount_mm(resort, day, band, seg=None):
    """mm of the cell's liquid that is snow by the model's own freezing level, else 0.

    seg is a next_72h by_day segment of the same date; its own water is used
    (the day's freezing-level maximum still bounds it, as its hours are a subset).
    """
    if day is None or day.get("horizon") != "short":
        return 0.0
    dv = g(day, "derived") or {}
    el = resort.get(BAND_ELEV_KEY.get(band, ""))
    flx = dv.get("freezing_level_precip_max_m")
    if not is_num(el) or not is_num(flx) or el < flx:
        return 0.0
    src = seg if seg is not None else g(day, band)
    if not isinstance(src, dict):
        return 0.0
    ph = src.get("phase") if seg is not None else g(dv, "phase", band)
    rp = src.get("rain_phase_mm") if seg is not None else g(dv, "rain_phase_mm", band)
    if ph not in ("snow", "mixed") or not is_num(rp) or rp >= RECOUNT_MIN_MM:
        return 0.0
    rain, pr = src.get("rain_mm"), src.get("precip_mm")
    if not is_num(rain) or not is_num(pr):
        return 0.0
    liquid = min(rain, pr)
    return liquid if liquid >= RECOUNT_MIN_MM else 0.0


def _day_by_date(resort, ds):
    days = resort.get("days")
    for d in days if isinstance(days, list) else []:
        if isinstance(d, dict) and d.get("date") == ds:
            return d
    return None


def recounted(resort, day, band):
    """True when this day's snow at this band includes recounted model rain."""
    return _recount_mm(resort, day, band) > 0


def seg_snow(resort, seg, band):
    """Snow in one next_72h by_day segment, with the model's own snow-rain recounted."""
    v = seg.get("snowfall_cm") if isinstance(seg, dict) else None
    if not is_num(v):
        return None
    extra = _recount_mm(resort, _day_by_date(resort, seg.get("date")), band, seg)
    return round(v + extra * SNOW_CM_PER_MM, 1)


def seg_recounted(resort, seg, band):
    return isinstance(seg, dict) and _recount_mm(resort, _day_by_date(resort, seg.get("date")), band, seg) > 0


def next3_band(resort, band):
    """next_72h snow at one band, with the model's own snow-rain recounted."""
    v = g(resort, "next_72h", f"{band}_cm")
    if not is_num(v):
        return None
    extra = 0.0
    for seg in g(resort, "next_72h", "by_day", band) or []:
        extra += _recount_mm(resort, _day_by_date(resort, seg.get("date")), band, seg)
    return round(v + extra * SNOW_CM_PER_MM, 1)


def next3(resort):
    """Next 3 days of snow at the hero band."""
    return next3_band(resort, hero_elevation(resort))


def week_total(resort):
    band = hero_elevation(resort)
    v = g(resort, "week", f"{band}_cm")
    if not is_num(v):
        return None
    extra = sum(_recount_mm(resort, day_by_index(resort, i), band) for i in range(7))
    return round(v + extra * SNOW_CM_PER_MM, 1)


def extended_total(resort):
    """Days 8-15 at the hero band: the fetcher's total, else the strict sum of
    the eight days' own values (None if any is missing)."""
    band = hero_elevation(resort)
    v = g(resort, "extended", f"{band}_cm")
    if is_num(v):
        return v
    vals = [g(day_by_index(resort, i), band, "snowfall_cm") for i in range(7, 15)]
    if all(is_num(x) for x in vals):
        return round(sum(vals), 1)
    return None


def day_snow(resort, day, elev=None):
    """Snowfall for one day at the hero band (or elev), with the model's own
    snow-rain recounted."""
    if day is None:
        return None
    elev = elev or hero_elevation(resort)
    v = g(day, elev, "snowfall_cm")
    if not is_num(v):
        return None
    extra = _recount_mm(resort, day, elev)
    return round(v + extra * SNOW_CM_PER_MM, 1) if extra else v


# Kinds of recent.measured that really are new snow. A depth change is not:
# it mixes settling and melt into the number and can be negative.
MEASURED_NEW_SNOW_KINDS = ("resort_reported_new_snow", "measured_new_snow")


def recent_fallen(resort):
    """Snow that has actually fallen, preferring the resort's own measurement.

    Returns (cm, source, hours). source is "reported" (the resort's report),
    "measured" (a station's new-snow measurement) or "model" (the forecast
    model's hindcast at the hero elevation). (None, None, None) when nothing
    is known. Measured and modelled figures are never averaged: whichever one
    is returned is quoted alone.
    """
    m = g(resort, "recent", "measured")
    if isinstance(m, dict) and m.get("kind") in MEASURED_NEW_SNOW_KINDS:
        src = "reported" if m.get("kind") == "resort_reported_new_snow" else "measured"
        for key, hours in (("h72_cm", 72), ("h48_cm", 48), ("h24_cm", 24)):
            v = m.get(key)
            if is_num(v):
                return v, src, hours
    v = g(resort, "recent", "modelled", hero_elevation(resort), "h72_cm")
    if is_num(v):
        return v, "model", 72
    return None, None, None


def flag_band(resort):
    """The band rain flags speak about: mid, else summit, else base."""
    if resort.get("mid_elevation_m") is not None:
        return "mid"
    if resort.get("summit_elevation_m") is not None:
        return "summit"
    return "base"


def rain_threat_72h(resort):
    """Snow-vs-rain sequence in the headline window at the flag band.

    Rain only matters as a threat to snow: rain before the snow gets buried,
    rain during or after it wrecks it. Returns None when nothing falls wet,
    else {kind, date, phase, snow_date, snow_cm} where kind is "threat"
    (wet during/after snow, or rain with no snow coming) or "buried" (all
    the wet falls before the snow). snow_date/snow_cm name the biggest snow
    day before the threat, or the burying day.
    """
    segs = g(resort, "next_72h", "by_day", flag_band(resort))
    if not isinstance(segs, list):
        return None
    band = flag_band(resort)
    # a segment whose model rain is recounted as snow (seg_recounted) is snow, not wet
    wet = [s for s in segs if s.get("phase") in ("rain", "mixed") and local_date(s.get("date"))
           and not seg_recounted(resort, s, band)]
    if not wet:
        return None
    snow = [dict(s, snowfall_cm=seg_snow(resort, s, band)) for s in segs
            if s.get("phase") in ("snow", "mixed") and is_num(seg_snow(resort, s, band))
            and seg_snow(resort, s, band) >= 1 and local_date(s.get("date"))]
    first_snow = min((local_date(s["date"]) for s in snow), default=None)
    after = [s for s in wet if first_snow is not None and local_date(s["date"]) >= first_snow]
    if snow and not after:
        burying = max(snow, key=lambda s: s["snowfall_cm"])
        last_wet = max(wet, key=lambda s: local_date(s["date"]))
        return {"kind": "buried", "date": local_date(last_wet["date"]), "phase": last_wet["phase"],
                "snow_date": local_date(burying["date"]), "snow_cm": burying["snowfall_cm"],
                "wet_mm": _wet_mm(last_wet)}
    t = min(after, key=lambda s: local_date(s["date"])) if after else wet[0]
    t_date = local_date(t["date"])
    prior = [s for s in snow if local_date(s["date"]) < t_date]
    big = max(prior, key=lambda s: s["snowfall_cm"]) if prior else None
    return {"kind": "threat", "date": t_date, "phase": t["phase"],
            "snow_date": local_date(big["date"]) if big else None,
            "snow_cm": big["snowfall_cm"] if big else None,
            "wet_mm": _wet_mm(t)}


def _wet_mm(seg):
    """Water that falls as rain or wet snow in one segment; never the band's
    raw rain_mm, which is the grid cell's."""
    a, b = seg.get("rain_phase_mm"), seg.get("mixed_phase_mm")
    if not is_num(a) and not is_num(b):
        return None
    return (a if is_num(a) else 0) + (b if is_num(b) else 0)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_km(a, b):
    try:
        d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return d if is_num(d) else None


def default_resort(resorts):
    def key(r):
        n = next3(r)
        w = week_total(r)
        return (
            0 if n is None else 1,
            n if n is not None else 0,
            w if w is not None else -1,
        )

    best = sorted(resorts, key=lambda r: name_key(r.get("name")))
    best = max(best, key=key)  # max keeps the first of equals, i.e. A-Z on name
    return best


def scope_resorts(resorts, selected, scope):
    if scope == "all":
        return list(resorts)
    if scope == "country":
        return [r for r in resorts if r.get("country") == selected.get("country")]
    with_d = []
    for r in resorts:
        d = distance_km(selected, r)
        with_d.append((d if d is not None else float("inf"), name_key(r.get("name")), r))
    with_d.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in with_d[:10]]


def _desc_nulls_last(value_fn):
    def key(r):
        v = value_fn(r)
        return (v is None, -(v or 0), name_key(r.get("name")))
    return key


def sort_resorts(resorts, sort, units="cm"):
    if sort == "name":
        def key(r):
            c = r.get("country")
            ci = COUNTRY_ORDER.index(c) if c in COUNTRY_ORDER else len(COUNTRY_ORDER)
            return (ci, name_key(c), name_key(r.get("region")), name_key(r.get("name")))
        return sorted(resorts, key=key)
    if sort == "best":
        # Chronological (ux spec 5a): soonest best day first, amount descending
        # inside a day, reason-word rows last.
        def bkey(r):
            i = best_ski_index(r)
            return (i is None, i if i is not None else 0,
                    -(best_display_value(r, units) or 0), name_key(r.get("name")))
        return sorted(resorts, key=bkey)
    fn = {
        "next3": next3,
        "week": week_total,
        "depth": lambda r: g(r, "depth", "value_cm") if is_num(g(r, "depth", "value_cm")) else None,
    }[sort]
    return sorted(resorts, key=_desc_nulls_last(fn))


POWDER_MIN_CM = 5.0  # a "best ski day" needs at least this much fresh snow (spec 0.1 item 7)


def best_ski_index(resort):
    """week.best_ski_day when its fresh-snow score is at least POWDER_MIN_CM and
    the snow to ski at the hero band is too, else None. The fetcher scores the
    summit; the page quotes the hero band, so both must clear the bar.

    When the best day is today on already-fallen snow and a measurement exists,
    the MEASURED figure must also clear the bar: the model promising a powder
    morning the resort's own sensors refute (Whitefish: model said a dump,
    the resort reported 0) is not a best ski day."""
    i = g(resort, "week", "best_ski_day")
    score = g(resort, "week", "best_ski_day_score_cm")
    if isinstance(i, int) and is_num(score) and score >= POWDER_MIN_CM:
        amt = _best_amount_at(resort, i)
        if is_num(amt) and amt >= POWDER_MIN_CM:
            if i == 0:
                # Mirror best_ski_display's fallen case exactly: yesterday had
                # snow and today's own row doesn't beat it.
                own = day_snow(resort, day_by_index(resort, 0))
                prev = day_snow(resort, day_by_index(resort, -1))
                own_v = own if is_num(own) else 0
                prev_v = prev if is_num(prev) else 0
                if prev_v >= 0.5 and own_v <= prev_v:
                    v, src, _hours = recent_fallen(resort)
                    if src in ("reported", "measured") and is_num(v) and v < POWDER_MIN_CM:
                        return None
            return i
    return None


def best_ski_amount(resort):
    """Fresh snow to ski on the best ski day, in cm, from the days' own values at the
    headline elevation so it matches the daily table: the day before's snow when
    none falls on the day, the day's own when none fell the day before, else both.
    None when no day qualifies (sorts last)."""
    i = best_ski_index(resort)
    if i is None:
        return None
    return _best_amount_at(resort, i)


def _best_amount_at(resort, i):
    own = day_snow(resort, day_by_index(resort, i))
    prev = day_snow(resort, day_by_index(resort, i - 1))
    own_v = own if is_num(own) else 0
    prev_v = prev if is_num(prev) else 0
    if own_v < 0.5:
        return prev if is_num(prev) else None
    if prev_v < 0.5:
        return own
    return own_v + prev_v


def best_ski_display(resort):
    """(amount_cm, tag) shown for the best ski day, everywhere it is quoted.

    tag None: forecast snow, findable in the daily table's own rows.
    tag "reported"/"measured"/"model": the best day is today and its snow has
    mostly already fallen. Yesterday's model row is not on the page, so the
    quoted amount is recent_fallen() — the same figure as the ground line —
    with the measured report beating the model wherever one exists.
    """
    i = best_ski_index(resort)
    if i is None:
        return None, None
    own = day_snow(resort, day_by_index(resort, i))
    prev = day_snow(resort, day_by_index(resort, i - 1))
    own_v = own if is_num(own) else 0
    prev_v = prev if is_num(prev) else 0
    if i == 0 and prev_v >= 0.5:
        if own_v > prev_v:
            # Mostly still to come: today's own table row carries the number.
            return own, None
        v, src, _hours = recent_fallen(resort)
        if v is not None:
            return v, src
        return best_ski_amount(resort), "model"
    return best_ski_amount(resort), None


def _disp_snow(v, units):
    """A snow amount as the page displays it: converted once, then rounded."""
    if not is_num(v):
        return None
    x = v / 2.54 if units == "in" else v
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def best_display_value(resort, units="cm"):
    """The number the Best ski day cell shows, in display units.

    Forecast amounts are composed from the already-rounded daily row values so
    the reader can add them up on the page; fallen amounts come from
    recent_fallen() via best_ski_display. This is also the sort key for the
    best-day columns, so sorted order is monotonic with the visible numbers.
    """
    i = best_ski_index(resort)
    if i is None:
        return None
    amt, tag = best_ski_display(resort)
    if tag is not None:
        return _disp_snow(amt, units)
    own = day_snow(resort, day_by_index(resort, i))
    prev = day_snow(resort, day_by_index(resort, i - 1)) if i > 0 else None
    if not is_num(prev) or prev < 0.5:
        n = _disp_snow(own, units)
        return n if n is not None else _disp_snow(amt, units)
    if not is_num(own) or own < 0.5:
        return _disp_snow(prev, units)
    return (_disp_snow(own, units) or 0) + (_disp_snow(prev, units) or 0)


def updated_utc(data):
    return _parse_utc(data.get("updated"))


def local_date(value):
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Powder score (docs/powder_spec: science section 2)
#
# A forecast, never a fact: every input is the model's. Open-Meteo counts all
# frozen water at a fixed 7:1 (0.7 cm per mm), so snowfall / (0.7 x precip)
# is the frozen fraction of the day's water, not the snow's density.
# --------------------------------------------------------------------------

POWDER_AMOUNT_MIN_CM = 10.0   # = thresholds.storm_day_min_cm
POWDER_FF_MIN = 0.9           # frozen fraction, 1.0 = all snow
POWDER_COLD_C = -8.0          # = thresholds.powder_cold_below_c
POWDER_WARM_C = -3.0          # = thresholds.powder_warm_above_c
POWDER_DENDRITE_MIN_C = -18.0  # colder than this grows small, denser crystals
POWDER_BAND_MAX_C = -3.0      # band max temp for the full cold credit
POWDER_CALM_GUST_KMH = 40.0   # gusts at or under this leave fresh snow in place
POWDER_LABELS = {3: "dry powder", 2: "powder", 1: "heavy snow", 0: "heavy snow"}


def powder_band_key(resort):
    """The band the score speaks for: mid, else summit (the ski_score band)."""
    if resort.get("mid_elevation_m") is not None:
        return "mid"
    if resort.get("summit_elevation_m") is not None:
        return "summit"
    return None


def powder_score(resort, day):
    """The powder score for one short-horizon day, or None for no claim.

    None when the day is not "short" or any input is missing (the page shows
    "—"). Otherwise a dict: ok False with the first failed gate, or ok True
    with q (0..3), label, tp, wp, band, snow_cm, ff and crust (warmth or rain
    the next day may crust the snow).

    Gates: band snow >= 10 cm; no rain or mixed risk at the band and frozen
    fraction >= 0.9; the day before not rained on at the band; no wind hold.
    Tp (0..2) from powder_temp_c and the band's max temperature; Wp (0..1)
    from the band's daily max gust. Q = Tp + Wp.
    """
    if not isinstance(day, dict) or day.get("horizon") != "short":
        return None
    band = powder_band_key(resort)
    if band is None:
        return None
    b = g(day, band) or {}
    dv = g(day, "derived") or {}
    prev = day_by_index(resort, day.get("index", 0) - 1) if isinstance(day.get("index"), int) else None
    snow, pr = b.get("snowfall_cm"), b.get("precip_mm")
    rr, mx = g(dv, "rain_risk", band), g(dv, "mixed_risk", band)
    prr = g(prev, "derived", "rain_risk", band)
    hold = dv.get("wind_hold")
    pt, tmax, gust = dv.get("powder_temp_c"), b.get("temp_max_c"), b.get("gust_max_kmh")
    if not all(is_num(x) for x in (snow, pr, pt, tmax, gust)):
        return None
    if not all(isinstance(x, bool) for x in (rr, mx, prr, hold)):
        return None
    out = {"band": band, "snow_cm": snow, "index": day.get("index")}
    if snow < POWDER_AMOUNT_MIN_CM:
        return {**out, "ok": False, "gate": "amount"}
    ff = snow / (SNOW_CM_PER_MM * pr) if pr > 0 else None
    out["ff"] = ff
    if rr or mx or ff is None or ff < POWDER_FF_MIN:
        return {**out, "ok": False, "gate": "phase"}
    if prr:
        return {**out, "ok": False, "gate": "rained_before"}
    if hold:
        return {**out, "ok": False, "gate": "wind"}
    # Temperature points
    if pt > POWDER_WARM_C or tmax > 0:
        tp = 0
    elif POWDER_DENDRITE_MIN_C <= pt <= POWDER_COLD_C and tmax <= POWDER_BAND_MAX_C:
        tp = 2
    else:
        tp = 1  # colder than -18, -8 < pt <= -3, or a cold summit over a warmer band
    wp = 1 if gust <= POWDER_CALM_GUST_KMH else 0
    q = tp + wp
    nxt = day_by_index(resort, day.get("index") + 1)
    n_tmax = g(nxt, "summit", "temp_max_c")
    n_rr = g(nxt, "derived", "rain_risk", band)
    crust = (is_num(n_tmax) and n_tmax > 0) or n_rr is True
    return {**out, "ok": True, "q": q, "tp": tp, "wp": wp, "label": POWDER_LABELS[q], "crust": crust}


def models_agree_cm(resort, index, min_cm=POWDER_AMOUNT_MIN_CM):
    """True when every compared model gives at least min_cm that day; a
    missing value fails. None when the file compares no models."""
    m = resort.get("models")
    if not isinstance(m, dict) or not m.get("compared"):
        return None
    row = next((d for d in m.get("days") or [] if d.get("index") == index), None)
    if row is None:
        return False
    vals = [g(row, "snowfall_cm", k) for k in m["compared"]]
    return all(is_num(v) and v >= min_cm for v in vals)
