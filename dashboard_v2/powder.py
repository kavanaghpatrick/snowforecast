"""The Powder page: fallen snow that passed quality checks, and powder forecast.

?view=powder&resort=<id>&units=cm|in

The page lists only snow that passes quality checks; any other new snow is
still on the resort and everywhere pages, just not here. Fallen snow is
measured as an amount only, so its quality comes from the forecast model's
fields for the past days, and the caption says "Model:" to keep that apart
from the measurement.

  Snow measured, 3 days   recent.measured kind measured_new_snow, h72 >= 20 cm,
                          reading < 36 h old, and every retroactive gate passes
  Snow reported, 3 days   kind resort_reported_new_snow, same gates, and the
                          3-day new snow is at most twice the resort's own depth
  Powder forecast, 4 days days 0-3 that pass data.powder_score with a
                          temperature point and the wind point (so "powder"
                          or "dry powder", never "dense snow"), with no gust
                          over 40 km/h after the snow stops (when the file
                          says), and on which every compared model gives
                          >= 10 cm (in Kuchera terms when the amount gate
                          read the Kuchera estimate). Displayed amounts are
                          always the 7:1 figure every other page shows, so
                          one day carries one number product-wide; Kuchera
                          decides which days qualify but is never displayed
                          (the page states no bar, so a shown figure under
                          10 cm contradicts nothing the user can see).

Retroactive gates, at the powder band, over the model's past days and today
(day 0, whole day, so stricter than the hours before the reading), from the
first day with >= 1 cm snow there (fallen_gates): fell as snow (frozen
fraction >= 0.9), fell cold (powder_temp_c <= -3 and band max <= 0 C), no
rain since (< 0.5 mm a day), no thaw since (band max <= 0 C), no gust over
40 km/h since. A gate whose input is missing makes no claim and drops the row.

Depth rises are dropped from this page: a rise mixes settling, melt and drift
into the number, so it cannot say snow fell, let alone how.

An empty group does not render. All groups share one bar scale.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from html import escape as _html_escape


def escape(s, quote=True):
    """html.escape, with every run of whitespace folded to one space.

    A newline pair inside a data string would end Streamlit's raw HTML block and
    turn the rest of the string into live markdown (remote images, links).
    """
    return _html_escape(" ".join(str(s).split()), quote)
from urllib.parse import urlencode

from data import (
    day_by_index,
    day_snow,
    g,
    is_num,
    kuchera_slr,
    local_date,
    models_agree_cm,
    name_key,
    post_gust_kmh,
    powder_band_key,
    powder_score,
    POWDER_AMOUNT_MIN_CM,
    POWDER_CALM_GUST_KMH,
    POWDER_FF_MIN,
    POWDER_WARM_C,
    SNOW_CM_PER_MM,
)
from render import (DASH, THIN, _tz, bar, disp_cell, prose_num, resort_today, scale_max,
                    short_day_word, snow_cover_str, snow_str, snow_val)

FALLEN_MIN_CM = 20.0      # a fallen-snow row needs this much in 72 hours
READING_MAX_AGE_H = 36    # and a reading no older than this
FORECAST_DAYS = range(0, 4)
MAX_WORDS = 8

MEASURED_KINDS = ("measured_new_snow", "depth_change")   # how reading_time finds the station
LISTED_KINDS = ("measured_new_snow", "resort_reported_new_snow")  # depth rises never list here
PAST_DAYS = (-3, -2, -1, 0)  # the model's past days and today, oldest first.
# Today is the whole day, so its hours after the reading count too: stricter,
# never looser, and no hour before the reading goes unchecked.
SNOWY_DAY_CM = 1.0          # = thresholds.snowy_day_min_cm: the snow fell from the first such day
RAIN_SINCE_MM = 0.5         # = thresholds.phase_day_min_mm: less rain than this is no rain
WIND_PACK_GUST_KMH = POWDER_CALM_GUST_KMH  # gusts above this move fresh snow
FALLEN_CAPTION = "Model: snow fell cold, no rain since."
IMPLAUSIBLE_X = 2.0         # 3-day new snow over twice the resort's own depth
REPORTED_KINDS = ("resort_reported_new_snow",)
BANDS_LOW_TO_HIGH = (("base", "elevation_m"), ("mid", "mid_elevation_m"), ("summit", "summit_elevation_m"))


def _href(**p):
    return "?" + urlencode({k: v for k, v in p.items() if v is not None})


# --------------------------------------------------------------------------
# Fallen snow: what a station or the resort says came down
# --------------------------------------------------------------------------

def _as_utc(value, resort):
    """A reading time as an aware UTC datetime. A bare date is taken as the
    start of that day at the resort, the oldest the reading could be."""
    if not isinstance(value, str) or not value:
        return None
    if len(value) == 10:
        d = local_date(value)
        if d is None:
            return None
        tz = _tz(resort)
        if tz is None:
            off = resort.get("utc_offset_s")
            tz = timezone(timedelta(seconds=off)) if is_num(off) and abs(off) < 86400 else timezone.utc
        return datetime.combine(d, time(0, 0), tzinfo=tz).astimezone(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _local_day(dt, resort):
    """The resort's calendar date at the aware datetime dt."""
    tz = _tz(resort)
    off = resort.get("utc_offset_s")
    if tz is None:
        tz = timezone(timedelta(seconds=off)) if is_num(off) and abs(off) < 86400 else timezone.utc
    return dt.astimezone(tz).date()


def reading_time(resort):
    """When recent.measured was read, or None when the file can't say.

    A station figure: the station its label names. A resort report: the
    depth report from the same provider. Nothing else is borrowed.
    """
    m = g(resort, "recent", "measured") or {}
    kind = m.get("kind")
    if kind in MEASURED_KINDS:
        hit = re.search(r"\bat (.+?) station\b", m.get("label") or "")
        name = hit.group(1).casefold() if hit else None
        for s in resort.get("stations") or []:
            if name and (s.get("name") or "").casefold() == name and s.get("provider") == m.get("provider"):
                return _as_utc(s.get("observed_utc") or s.get("observed_date"), resort)
        return None
    if kind in REPORTED_KINDS:
        dep = resort.get("depth") or {}
        if dep.get("kind") == "resort_report" and dep.get("provider") == m.get("provider"):
            return _as_utc(dep.get("observed_utc"), resort)
    return None


def fallen_gates(resort):
    """Retroactive quality gates for snow that fell in the model's past days.

    Returns (failed, missing): the gates the snow fails and the gates with
    missing inputs. Snow passes only when both are empty. Gates, at the
    powder band, from the first day with snow there through today:
    phase (frozen fraction), cold (fell cold), rain, thaw and wind since.
    """
    band = powder_band_key(resort)
    days = [day_by_index(resort, i) for i in PAST_DAYS]
    if band is None or any(d is None for d in days):
        return [], ["band"]
    bands = [g(d, band) or {} for d in days]
    snows = [b.get("snowfall_cm") for b in bands]
    if not all(is_num(x) for x in snows):
        return [], ["phase"]
    first = next((i for i, x in enumerate(snows) if x >= SNOWY_DAY_CM), None)
    if first is None:
        return [], ["phase"]  # the model has no snow here to judge
    since = list(range(first, len(days)))
    snowy = [i for i in since if snows[i] >= SNOWY_DAY_CM]
    failed, missing = [], []

    pr = [bands[i].get("precip_mm") for i in snowy]
    if not all(is_num(x) for x in pr) or sum(pr) <= 0:
        missing.append("phase")
    elif sum(snows[i] for i in snowy) / (SNOW_CM_PER_MM * sum(pr)) < POWDER_FF_MIN:
        failed.append("phase")

    rain = [bands[i].get("rain_mm") for i in since]
    if not all(is_num(x) for x in rain):
        missing.append("rain")
    elif any(x >= RAIN_SINCE_MM for x in rain):
        failed.append("rain")

    pt = [g(days[i], "derived", "powder_temp_c") for i in snowy]
    tmax_fall = [bands[i].get("temp_max_c") for i in snowy]
    if not all(is_num(x) for x in pt + tmax_fall):
        missing.append("cold")
    elif any(x > POWDER_WARM_C for x in pt) or any(x > 0 for x in tmax_fall):
        failed.append("cold")

    tmax = [bands[i].get("temp_max_c") for i in since]
    if not all(is_num(x) for x in tmax):
        missing.append("thaw")
    elif any(x > 0 for x in tmax):
        failed.append("thaw")

    gust = [bands[i].get("gust_max_kmh") for i in since]
    if not all(is_num(x) for x in gust):
        missing.append("wind")
    elif any(x > WIND_PACK_GUST_KMH for x in gust):
        failed.append("wind")
    return failed, missing


def implausible(resort):
    """True when a resort report's 3-day new snow is over twice its own
    reported depth, or the depth is not there to check against."""
    m = g(resort, "recent", "measured") or {}
    dep = resort.get("depth") or {}
    dv = dep.get("value_cm")
    if dep.get("kind") != "resort_report" or dep.get("provider") != m.get("provider") or not is_num(dv):
        return True
    return is_num(m.get("h72_cm")) and m["h72_cm"] > IMPLAUSIBLE_X * dv


def fallen_verdict(resort, now=None):
    """(row, why): the row when the resort's fallen snow lists on this page,
    else None and the reason it does not ("" when there is no candidate)."""
    m = g(resort, "recent", "measured")
    if not isinstance(m, dict):
        return None, ""
    kind = m.get("kind")
    cm = m.get("h72_cm")
    if kind not in MEASURED_KINDS + REPORTED_KINDS or not is_num(cm) or cm < FALLEN_MIN_CM:
        return None, ""
    t = reading_time(resort)
    now = now or datetime.now(timezone.utc)
    if t is None or (now - t).total_seconds() / 3600 >= READING_MAX_AGE_H:
        return None, "stale reading"
    if kind not in LISTED_KINDS:
        return None, "depth rise"
    if kind in REPORTED_KINDS and implausible(resort):
        return None, "implausible report"
    failed, missing = fallen_gates(resort)
    # The gates run through day 0. When the file's day 0 is older than the
    # reading's own date (a resort past midnight before the next fetch), the
    # hours that day before the reading went unchecked: no claim.
    d0 = local_date(g(day_by_index(resort, 0), "date"))
    if d0 is None or d0 < _local_day(t, resort):
        missing = missing + ["today"]
    if failed or missing:
        return None, "; ".join([f"failed {', '.join(failed)}"] * bool(failed)
                               + [f"no data for {', '.join(missing)}"] * bool(missing))
    dv = g(resort, "depth", "value_cm")
    return {"resort": resort, "group": "measured" if kind in MEASURED_KINDS else "reported",
            "kind": kind, "cm": cm, "depth_cm": dv if is_num(dv) else None}, ""


def fallen_row(resort, now=None):
    """{group, kind, cm, depth_cm} when the resort's fallen snow passes every gate, else None."""
    return fallen_verdict(resort, now)[0]


# --------------------------------------------------------------------------
# Powder forecast
# --------------------------------------------------------------------------

def snow_down_to(resort, day):
    """The lowest band that is all snow with at least 10 cm, or None."""
    for band, key in BANDS_LOW_TO_HIGH:
        if resort.get(key) is None:
            continue
        sn = g(day, band, "snowfall_cm")
        if g(day, "derived", "phase", band) == "snow" and is_num(sn) and sn >= 10:
            return band
    return None


def _agree_min_cm(day, s):
    """The amount every compared model must give, in its own 7:1 cm.

    10 cm, unless the day's amount gate read the Kuchera estimate: then the
    same 10 cm in Kuchera terms, at the scored band's own ratio (the one
    the amount gate used, so the two cannot disagree). 7:1 cm x ratio / 7
    >= 10 cm. A colder ratio lowers the bar; a ratio under 7 raises it
    above 10 cm, with no floor: a warm storm must clear more.
    """
    slr = kuchera_slr(day, s["band"])
    if s.get("kuchera_cm") is None or slr is None:
        return POWDER_AMOUNT_MIN_CM
    if slr <= 0:
        return float("inf")   # no model amount clears a ratio of 0 (the amount gate already failed)
    return POWDER_AMOUNT_MIN_CM * SNOW_CM_PER_MM * 10 / slr


def forecast_verdict(resort, i):
    """(row, why) for forecast day i: the row when it lists on this page,
    else None and the reason it does not ("" when no score or a failed
    score gate, which powder_score already names)."""
    day = day_by_index(resort, i)
    # Between a resort's midnight and the next fetch the file's day 0 is
    # yesterday there. A past day is not a forecast: it never lists here.
    d = local_date(g(day, "date"))
    if d is None or d < resort_today(resort):
        return None, "past day" if d is not None else ""
    s = powder_score(resort, day)
    if not s or not s.get("ok"):
        return None, ""
    # Powder only: fell cold (tp >= 1) and calm (wp 1), the same bar as
    # fallen snow. "dense snow" is not powder and does not list here.
    if s.get("tp", 0) < 1 or s.get("wp") != 1:
        return None, "not powder"
    # Wind after the snow stops packs it as surely as wind during it
    pg = post_gust_kmh(day, s["band"])
    if pg is not None and pg > WIND_PACK_GUST_KMH:
        return None, "wind after snowfall"
    if models_agree_cm(resort, i, _agree_min_cm(day, s)) is not True:
        return None, "models disagree"
    shown = day_snow(resort, day, s["band"])
    cm = shown if is_num(shown) else s["snow_cm"]
    # One estimate on every page: all displayed amounts are the 7:1 figure,
    # the same number the resort page and daily table carry, so the same day
    # can never show two different "~" amounts (ruling, Nov 28). The Kuchera
    # estimate decides WHICH days qualify (the gates) but is not displayed;
    # the page states no 10 cm bar, so a smaller shown figure contradicts
    # nothing. page_cm is kept equal to cm for sort/scale stability.
    return {"resort": resort, "group": "forecast", "index": i, "day": day, "score": s,
            "cm": cm, "page_cm": cm}, ""


def forecast_row(resort):
    """The resort's best qualifying powder day in days 0-3, or None.

    Best = highest score, then most snow, then the earlier day.
    """
    best = None
    for i in FORECAST_DAYS:
        row, _why = forecast_verdict(resort, i)
        if row is None:
            continue
        key = (row["score"]["q"], row["cm"], -i)
        if best is None or key > best[0]:
            best = (key, row)
    return best[1] if best else None


def _watch(resort, row):
    """'rain next day' / 'thaw next day' when the next day may crust the snow."""
    s = row["score"]
    if not s.get("crust"):
        return ""
    nxt = day_by_index(resort, row["index"] + 1)
    if g(nxt, "derived", "rain_risk", s["band"]) is True:
        return "rain next day"
    return "thaw next day"


# --------------------------------------------------------------------------
# Groups and the lead sentence
# --------------------------------------------------------------------------

def groups(resorts, now=None):
    """(measured, reported, forecast) rows, each ranked."""
    measured, reported, forecast = [], [], []
    for r in resorts:
        f = fallen_row(r, now)
        if f:
            (measured if f["group"] == "measured" else reported).append(f)
        p = forecast_row(r)
        if p:
            forecast.append(p)
    for rows in (measured, reported):
        rows.sort(key=lambda x: (-x["cm"], name_key(x["resort"].get("name"))))
    forecast.sort(key=lambda x: (-x["score"]["q"], -x["page_cm"], x["index"], name_key(x["resort"].get("name"))))
    return measured, reported, forecast


def any_powder(resorts, now=None):
    """True when any resort clears a fallen-snow or powder-forecast gate."""
    return any(groups(resorts, now))


def _words(sentence):
    """Words as the page counts them: a number with its unit is one word."""
    s = re.sub(r"<[^>]+>", "", sentence).replace("&#8239;", " ").rstrip(".")
    s = re.sub(r"~?\b[\d,]+\s?(cm|in)\b|~?[\d,]+″", "N", s)
    return [w for w in s.split() if w]


def _amount(v, units, model=False):
    """'16 cm' / '6″' in prose; model=True prefixes "~", the model's mark."""
    n = snow_str(v, units)
    if n is None:
        return None
    a = f"{prose_num(n)}″" if units == "in" else f"{prose_num(n)}{THIN}cm"
    return f"~{a}" if model and n != "0" else a


def _day_words(resort, index):
    """'today' / 'tomorrow' / 'Sun' for a forecast day."""
    w = short_day_word(resort, index)
    if w is None:
        return None
    if w == "Today":
        return "today"
    d = local_date(g(day_by_index(resort, index), "date"))
    if d is not None and d == resort_today(resort) + timedelta(days=1):
        return "tomorrow"
    return w


def forecast_sentence(name_html, row, units, resort):
    """'Myoko Kogen: powder snow on Sun, ~16 cm.' Drops the quality word,
    then the colon-led name, to stay within eight words. The amount is the
    7:1 figure every other page shows for the day (one estimate, one number)."""
    amt = _amount(row["page_cm"], units, model=True)
    day = _day_words(resort, row["index"])
    if amt is None or day is None:
        return None
    on = day if day in ("today", "tomorrow") else f"on {day}"
    label = row["score"]["label"]
    kind = label if label.endswith("snow") else f"{label} snow"
    for k in (kind, "snow"):
        s = f"{name_html}: {k} {on}, {amt}."
        if len(_words(s)) <= MAX_WORDS:
            return s
    return f"{kind[0].upper()}{kind[1:]} {on}, {amt}."


def fallen_sentence(name_html, row, units):
    amt = _amount(row["cm"], units)
    if row["kind"] == "measured_new_snow":
        s = f"{name_html} measured {amt} new snow."
    else:
        s = f"{name_html} reports {amt} new snow."
    if len(_words(s)) > MAX_WORDS:
        s = f"{amt} new snow measured." if row["kind"] == "measured_new_snow" else f"{amt} new snow reported."
    return s


def lead(measured, reported, forecast, units, link, resorts=None):
    """The top row of the first non-empty group, as one sentence."""
    for rows in (measured, reported):
        if rows:
            r = rows[0]["resort"]
            return fallen_sentence(link(r), rows[0], units)
    if forecast:
        r = forecast[0]["resort"]
        s = forecast_sentence(link(r), forecast[0], units, r)
        if s:
            return s
    # No data at all is a different fact from data that qualifies nothing.
    if resorts is not None and not any(
            day_by_index(r, i) is not None for r in resorts for i in FORECAST_DAYS):
        return "No forecast for any resort."
    # Not "no 10 cm snow day": snowy days that fail a quality gate exist
    return "Models show no powder snow next 4 days."


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def _name_cell(r, href, show_country=True):
    name = f'<a class="plain" href="{escape(href)}" target="_self">{escape(r.get("name") or "?")}</a>'
    if show_country and r.get("country"):
        name += f' <span class="label">{escape(r["country"])}</span>'
    return f'<td class="first">{name}</td>'


def _down_to_str(resort, day, units):
    """How much of the mountain the powder covers: "all" / "top ~60%" / "none".

    A raw elevation meant nothing without knowing the resort's heights (user
    directive, 27 Nov), so this shares snow_cover_str with the daily table.
    Falls back to the share implied by the qualifying band's own elevation;
    "\u2014" when nothing is computable.
    """
    cover = snow_cover_str(resort, day)
    if cover is not None:
        return cover
    band = snow_down_to(resort, day)
    key = {"base": "elevation_m", "mid": "mid_elevation_m", "summit": "summit_elevation_m"}.get(band)
    v = resort.get(key) if key else None
    bot, top = resort.get("elevation_m"), resort.get("summit_elevation_m")
    if v is None or bot is None or top is None or top <= bot:
        return DASH
    if v <= bot:
        return "~100%"
    pct = int(round((top - v) / (top - bot) * 20)) * 5
    return "~100%" if pct >= 100 else ("~0%" if pct <= 0 else f"upper ~{pct}%")


def page(resorts, units, resort_id, debug=False, now=None):
    extra = {"debug": "1"} if debug else {}
    measured, reported, forecast = groups(resorts, now)

    def href(r):
        return _href(resort=r.get("id", ""), units=units, **extra)

    def link(r):
        return f'<a class="plain" href="{escape(href(r))}" target="_self">{escape(r.get("name") or "?")}</a>'

    vmax = scale_max([snow_val(x["cm"], units) for x in measured + reported]
                     + [snow_val(x["page_cm"], units) for x in forecast], units)

    def tr(row, cells):
        cls = ' class="is-selected"' if row["resort"].get("id") == resort_id else ""
        return f"<tr{cls}>{''.join(cells)}</tr>"

    out = ['<div class="page every powder">', '<h1 class="title">Powder</h1>',
           f'<p class="leaders prose gap1">{lead(measured, reported, forecast, units, link, resorts)}</p>']

    for title, rows in (("Snow measured, 3 days", measured), ("Snow reported, 3 days", reported)):
        if not rows:
            continue
        # Headers name the measure; the unit sits in each cell. These rows are
        # measured or reported figures, so the cells stay bare of "~".
        head = ('<tr><th>Resort</th><th class="barcol"></th><th class="r">New snow</th>'
                '<th class="r">Snow depth</th></tr>')
        body = []
        for x in rows:
            v = snow_val(x["cm"], units)
            dv = snow_val(x["depth_cm"], units)
            cells = [_name_cell(x["resort"], href(x["resort"])),
                     f'<td class="barcol">{bar(v, vmax)}</td>',
                     f'<td class="r">{disp_cell(v, units)}</td>',
                     f'<td class="r">{disp_cell(dv, units)}</td>']
            body.append(tr(x, cells))
        # Literally true of every row: fallen_gates passed phase, cold and rain
        out.append(f'<div class="label gap2">{title}</div>'
                   f'<p class="caption gap0">{FALLEN_CAPTION}</p>'
                   f'<div class="tablewrap gap0"><table class="t every-t pow-t">{head}{"".join(body)}</table></div>')

    if forecast:
        # Watch for shows only when some row has something to watch for
        watch = any(_watch(x["resort"], x) for x in forecast)
        # No "Snow type" column: every row on this page is powder (skim spec 1.6)
        head = ('<tr><th>Resort</th><th class="barcol"></th><th class="r">New snow</th><th>Day</th>'
                f'<th>Snow reaches</th>{"<th>Watch for</th>" if watch else ""}</tr>')
        body = []
        for x in forecast:
            r = x["resort"]
            v = snow_val(x["page_cm"], units)
            cells = [_name_cell(r, href(r)),
                     f'<td class="barcol">{bar(v, vmax)}</td>',
                     f'<td class="r">{disp_cell(v, units, model=True)}</td>',
                     f'<td>{escape(short_day_word(r, x["index"]) or DASH)}</td>',
                     f'<td>{escape(_down_to_str(r, x["day"], units))}</td>']
            if watch:
                cells.append(f'<td>{escape(_watch(r, x))}</td>')
            body.append(tr(x, cells))
        out.append('<div class="label gap2">Powder forecast, 4 days</div>'
                   f'<div class="tablewrap gap0"><table class="t every-t pow-t">{head}{"".join(body)}</table></div>')

    out.append("</div>")
    return "".join(out)


def resort_line(resort, units):
    """The resort page's powder sentence, or "" when the resort has no
    qualifying powder forecast in days 0-3. Always worded as a forecast."""
    row = forecast_row(resort)
    if row is None:
        return ""
    amt = _amount(row["cm"], units, model=True)
    day = _day_words(resort, row["index"])
    if amt is None or day is None:
        return ""
    on = day if day in ("today", "tomorrow") else f"on {day}"
    label = row["score"]["label"]
    kind = label if label.endswith("snow") else f"{label} snow"
    s = f"{kind[0].upper()}{kind[1:]} {on}: {amt}."
    if row["score"].get("crust"):
        s += " Warmth may crust the snow."
    return s


CSS = """
/* the three tables line up: same resort and bar columns, numbers on one edge */
[data-testid="stMarkdownContainer"] table.pow-t td.first, [data-testid="stMarkdownContainer"] table.pow-t th:first-child{
  width:17em; min-width:17em; white-space:nowrap; }
[data-testid="stMarkdownContainer"] table.pow-t td.r, [data-testid="stMarkdownContainer"] table.pow-t th.r{ min-width:5.5em; }
@media (max-width:700px){
  [data-testid="stMarkdownContainer"] table.pow-t td.first, [data-testid="stMarkdownContainer"] table.pow-t th:first-child{
    width:auto; min-width:8.5em; white-space:normal; }
  [data-testid="stMarkdownContainer"] table.pow-t td.r, [data-testid="stMarkdownContainer"] table.pow-t th.r{ min-width:4.5em; }
}
"""
