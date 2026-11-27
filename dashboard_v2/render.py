"""HTML for the v2 dashboard, in the order of docs/ux_hierarchy_spec.md.

Every function returns a string of HTML with no leading indentation and no
blank lines (so Streamlit's markdown pass keeps it as one raw HTML block).
Every data string goes through html.escape. The page runs in the order a skier
decides: verdict, hero, mountain strip, nearby, ground, days, compare,
stations, footer. Every sentence that counts snow says "snow". Missing values
are words in prose and "—" in table cells; zero is always "0".
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from html import escape as _html_escape


def escape(s, quote=True):
    """html.escape, with every run of whitespace folded to one space.

    A newline pair inside a data string would end Streamlit's raw HTML block and
    turn the rest of the string into live markdown (remote images, links).
    """
    return _html_escape(" ".join(str(s).split()), quote)
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from data import (
    BAND_ELEV_KEY,
    BAND_WORD,
    best_display_value,
    best_ski_amount,
    best_ski_display,
    best_ski_index,
    day_by_index,
    day_snow,
    distance_km,
    extended_total,
    g,
    has_summit,
    hero_elevation,
    is_num,
    local_date,
    next3,
    next3_band,
    rain_threat_72h,
    recent_fallen,
    recounted,
    seg_recounted,
    seg_snow,
    week_total,
)
from skill import ROUGH_FROM_INDEX, likely_range

DASH = "—"
MINUS = "−"
THIN = "&#8239;"  # narrow no-break space, so a number never wraps away from its unit
DOT = " · "

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


# --------------------------------------------------------------------------
# Context: current params, used to build links
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, resort_id, scope, sort, units, debug=False):
        self.resort_id = resort_id
        self.scope = scope
        self.sort = sort
        self.units = units
        self.debug = debug

    def href(self, **over):
        p = {"resort": self.resort_id, "scope": self.scope, "sort": self.sort, "units": self.units}
        p.update(over)
        if self.debug:
            p["debug"] = "1"
        return "?" + urlencode(p)

    def link(self, text, cls="", **over):
        c = f' class="{cls}"' if cls else ""
        return f'<a href="{escape(self.href(**over))}" target="_self"{c}>{escape(text)}</a>'

    @property
    def snow_unit(self):
        return "in" if self.units == "in" else "cm"

    @property
    def elev_unit(self):
        return "ft" if self.units == "in" else "m"


# --------------------------------------------------------------------------
# Number formatting (spec 4.8)
# --------------------------------------------------------------------------

def _round(v):
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def _minus(s):
    return s.replace("-", MINUS)


def snow_val(v, units):
    """Snow in display units as an int, or None."""
    if not is_num(v):
        return None
    return _round(v / 2.54) if units == "in" else _round(v)


def snow_str(v, units):
    n = snow_val(v, units)
    return None if n is None else f"{n:,}"


def signed_str(v, units):
    n = snow_val(v, units)
    if n is None:
        return None
    if n > 0:
        return f"+{n}"
    if n < 0:
        return f"{MINUS}{abs(n)}"
    return "0"


def elev_str(v, units):
    if not is_num(v):
        return None
    if units == "in":
        return f"{_round(v * 3.28084 / 10) * 10:,}"
    return f"{_round(v / 10) * 10:,}"


def temp_str(v, units):
    if not is_num(v):
        return None
    if units == "in":
        return _minus(f"{_round(v * 9 / 5 + 32)}") + "°F"
    return _minus(f"{_round(v)}") + "°C"


def wind_str(v, units):
    if not is_num(v):
        return None
    if units == "in":
        return f"{_round(v / 1.609344)} mph"
    return f"{_round(v)} km/h"


def pct_str(v):
    return None if not is_num(v) else f"{_round(v)}%"


def num(v, ctx, cls="", unit=True):
    """A snow numeral with its unit, or the dash for a table/row cell."""
    s = snow_str(v, ctx.units)
    if s is None:
        return DASH
    u = f'<span class="unit">{THIN}{ctx.snow_unit}</span>' if unit else ""
    c = f' class="{cls}"' if cls else ""
    return f"<span{c}>{s}{u}</span>"


def prose_num(text):
    return f'<span class="num">{escape(text)}</span>'


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def short_day(d):
    return f"{WEEKDAYS[d.weekday()][:3]} {MONTHS[d.month - 1][:3]} {d.day}"


def short_date(d):
    return f"{MONTHS[d.month - 1][:3]} {d.day}"


def resort_today(resort):
    """The current date at the resort, from its timezone (or UTC offset)."""
    tz = _tz(resort)
    off = resort.get("utc_offset_s")
    if tz is None and is_num(off) and abs(off) < 86400:
        tz = timezone(timedelta(seconds=off))
    return datetime.now(tz or timezone.utc).date()


def rel_word(resort, d):
    """today / tomorrow / Saturday for a real date, judged against the resort's current date."""
    today = resort_today(resort)
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    if d == today - timedelta(days=1):
        return "yesterday"
    if d < today:
        # A bare weekday name always means a future day; past days carry their date.
        return short_day(d)
    return WEEKDAYS[d.weekday()]


def day_word(resort, index, capital=False):
    """today / tomorrow / Saturday for a day index."""
    d = local_date(g(day_by_index(resort, index), "date"))
    if d is None:
        return None
    w = rel_word(resort, d)
    return w[0].upper() + w[1:] if capital else w


def short_day_word(resort, index):
    d = local_date(g(day_by_index(resort, index), "date"))
    if d is None:
        return None
    w = rel_word(resort, d)
    if w == "today":
        return "Today"
    if w == "tomorrow":
        return WEEKDAYS[d.weekday()][:3]
    return w[:3] if w in WEEKDAYS else w[0].upper() + w[1:]


def _tz(resort):
    name = resort.get("timezone")
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def update_line(updated, resort):
    if updated is None:
        return "Update time unknown"
    tz = _tz(resort)
    if tz is None:
        utc = updated.astimezone(timezone.utc)
        return f"Updated {short_day(utc.date())}, {utc:%H:%M} UTC"
    loc = updated.astimezone(tz)
    return f"Updated {short_day(loc.date())}, {loc:%H:%M} {loc.tzname()}"


def age_hours(updated):
    if updated is None:
        return None
    return (datetime.now(timezone.utc) - updated).total_seconds() / 3600


def freshness_html(updated, resort):
    parts = [escape(update_line(updated, resort))]
    age = age_hours(updated)
    if age is not None and age > 13:
        # Past 36 hours the hero sentence carries the "out of date" warning (spec 3.3);
        # this line keeps the plain age so the two never repeat each other.
        parts.append(f"{_round(age)}{THIN}h old")
    return DOT.join(parts)


def _obs_date(value, resort):
    """Date words for an observed_utc string, in the resort's timezone."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    tz = _tz(resort)
    if tz is not None and dt.tzinfo is not None:
        dt = dt.astimezone(tz)
    return short_date(dt.date())


# --------------------------------------------------------------------------
# Marks: bars and sparklines (brief 6.2, 6.3)
# --------------------------------------------------------------------------

def scale_max(values, units):
    vals = [v for v in values if v is not None]
    top = max(vals) if vals else 0
    if units == "in":
        return max(20, math.ceil(top / 10) * 10)
    return max(50, math.ceil(top / 25) * 25)


def bar(v, vmax, best=False):
    """v in display units (int) or None; zero and None draw nothing."""
    if v is None or v <= 0 or not vmax:
        return ""
    frac = min(v / vmax, 1.0)
    cls = "bar is-best" if best else "bar"
    return f'<span class="{cls}" style="--v:{frac:.4f};min-width:2px"></span>'


def spark_bars(values, best_index, vmax, height=12, label=""):
    """Vertical day bars, 3px wide with 2px gaps, viewBox (5n-2) x height.

    values are display-unit numbers (rounded) or None. Zero and None draw no
    rect; any other value is at least 1px. Returns "" when no rect would be
    drawn, so an empty frame never renders (skim spec 6.3).
    """
    rects = []
    for i, v in enumerate(values):
        if v is None or v <= 0 or not vmax:
            continue
        h = max(1, int(round(min(v / vmax, 1.0) * height)))
        cls = ' class="is-best"' if i == best_index else ""
        rects.append(f'<rect x="{i * 5}" y="{height - h}" width="3" height="{h}"{cls}/>')
    if not rects:
        return ""
    w = 5 * len(values) - 2
    lab = escape(label)
    return (f'<svg class="sb" width="{w}" height="{height}" viewBox="0 0 {w} {height}" '
            f'shape-rendering="crispEdges" role="img" aria-label="{lab}"><title>{lab}</title>'
            + "".join(rects) + "</svg>")


SL_W, SL_X0, SL_X1 = 120, 6, 114  # snow-line mark: 120px wide, axis from x=6 (base) to x=114 (summit)


def _sl_svg(title, shapes):
    t = escape(title)
    return (f'<svg class="sl" width="{SL_W}" height="12" viewBox="0 0 {SL_W} 12" shape-rendering="crispEdges" '
            f'role="img" aria-label="{t}"><title>{t}</title>{"".join(shapes)}</svg>')


def snow_cover_str(resort, day):
    """How much of the mountain gets snow, from the day's snow line: "all",
    "top ~60%", or "none". Raw elevations meant nothing to a reader who
    doesn't know the resort's heights, and the axis mark needed explaining
    (user directive, 27 Nov) — a share of the mountain needs neither.
    Rounded to 5%: the line is model-derived, finer would be false precision.
    None when the snow line or the resort's heights are unknown.
    """
    rng = snow_line_range(day)
    ax = _sl_axis(resort)
    if rng is None or ax is None:
        return None
    base, summit = ax

    def pct(line):
        p = (summit - line) / (summit - base) * 100
        return max(0, min(100, int(round(p / 5)) * 5))

    # The snow line moves during a mixed day. Collapsing its range to the
    # worst hour said "~0%" beside a 4-inch snow row (user, Crystal Mountain,
    # 27 Nov) — the snow fell in the other, colder hours. A split day shows
    # the honest range; a steady day keeps the single figure (conservative).
    worst, best = pct(rng[1]), pct(rng[0])
    if best - worst <= 5:
        if worst >= 100:
            return "~100%"
        if worst <= 0:
            return "~0%"
        return f"upper ~{worst}%"
    lo_w = "~0" if worst <= 0 else f"~{worst}"
    hi_w = "100" if best >= 100 else f"{best}"
    return f"{lo_w}\u2013{hi_w}%"


def snow_line_mark(lo, hi, base, summit, text):
    """One day's snow-line range on the resort's base-to-summit axis (skim spec 1.2).

    A 1px hairline from lo to hi with a tick at each end that sits inside the
    axis; an end beyond the axis is clipped with no tick ("keeps going"). A
    range wholly below base or above summit is a 4px stub outside that end.
    """
    if hi <= base:
        return _sl_svg("Snow line below base", ['<line x1="0" y1="6.5" x2="4" y2="6.5"/>'])
    if lo >= summit:
        return _sl_svg("Snow line above summit", [f'<line x1="{SL_W - 4}" y1="6.5" x2="{SL_W}" y2="6.5"/>'])

    def x(v):
        return int(round(SL_X0 + (v - base) / (summit - base) * (SL_X1 - SL_X0)))

    lo_in, hi_in = lo >= base, hi <= summit
    xa = x(lo) if lo_in else SL_X0
    xb = x(hi) if hi_in else SL_X1
    shapes = []
    if xb > xa or not (lo_in and hi_in):
        shapes.append(f'<line x1="{xa}" y1="6.5" x2="{xb + (1 if hi_in else 0)}" y2="6.5"/>')
    if lo_in:
        shapes.append(f'<line x1="{xa + .5}" y1="3" x2="{xa + .5}" y2="10"/>')
    if hi_in and xb != xa:
        shapes.append(f'<line x1="{xb + .5}" y1="3" x2="{xb + .5}" y2="10"/>')
    return _sl_svg(f"Snow line {text}", shapes)


def spark_line(values):
    """Depth series; None breaks the line. Accent dot on the latest value."""
    n = len(values)
    known = [v for v in values if v is not None]
    if n < 2 or not known:
        return ""
    lo, hi = min(known), max(known)
    span = hi - lo if hi > lo else 1.0
    w, h = 70.0, 12.0

    def xy(i, v):
        x = i * w / (n - 1)
        y = h - (v - lo) / span * (h - 2) - 1 if hi > lo else h / 2
        return x, y

    segs, cur = [], []
    for i, v in enumerate(values):
        if v is None:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(xy(i, v))
    if cur:
        segs.append(cur)
    lines = "".join(
        '<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in s) + '"/>'
        for s in segs if len(s) >= 2
    )
    last_i = max(i for i, v in enumerate(values) if v is not None)
    lx, ly = xy(last_i, values[last_i])
    return (
        f'<svg class="spark" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none" aria-hidden="true">'
        f'{lines}<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2" vector-effect="non-scaling-stroke"/></svg>'
    )


# --------------------------------------------------------------------------
# Shared bits
# --------------------------------------------------------------------------

POWDER_MIN_CM = 5.0   # a "best ski day" needs at least this much fresh snow (spec 0.1 item 7)
WET_MIN_MM = 2.0      # rain or wet snow under this much water is not worth a clause (spec 2.3)
RAIN_AMOUNT_MM = 5.0  # a rain amount is printed only from this much up (spec 2.3)
SNOWLINE_BELOW_FL_M = 250  # the fetcher's snow line proxy (thresholds.snowline_below_fl_m)
# The verdict may tell someone to go ("Ski Mon ...") only on an established
# base: a measured or resort-reported depth of at least this much (ux spec 5c).
# 20 cm is not a ski-quality line. It is the point below which "go ski" is not
# supportable: under it, or with no depth at all, fresh snow may be falling on
# rocks or on a resort that has not opened, so the verdict states the fact
# ("Fresh snow Mon: 26 cm.") without the recommendation. A thin station reading
# does not prove thin slopes (stations sit off-piste at their own elevations),
# so depth changes only this claim: never which day is best, the sort order,
# the cells' numbers, or Powder qualification.
GO_SKI_BASE_MIN_CM = 20.0


def best_ski_day(resort):
    """week.best_ski_day, but only when its fresh snow is at least 5 cm.

    The fetcher names a best ski day for any score above 0. Below 5 cm calling it
    "the day to ski" oversells it, so the page says there is no powder day.
    """
    return best_ski_index(resort)


def _spoil_reasons(resort):
    """Why snow days 0..6 lost their ski score: a set of "wind" and "rain"."""
    out = set()
    key = "mid" if resort.get("mid_elevation_m") is not None else "summit"
    for i in range(7):
        d = day_by_index(resort, i)
        dv = g(d, "derived") or {}
        prev = day_by_index(resort, i - 1)
        sn = day_snow(resort, d)
        ps = day_snow(resort, prev)
        if not ((is_num(sn) and sn >= 1) or (is_num(ps) and ps >= 1)):
            continue
        if dv.get("ski_score_cm") is not None:
            continue
        if dv.get("wind_hold") is True:
            out.add("wind")
        rr = dv.get("rain_risk") or {}
        prr = g(prev, "derived", "rain_risk") or {}
        if rr.get(key) is True or rr.get("summit") is True or prr.get(key) is True:
            out.add("rain")
    return out


def _snow_spoiled(resort):
    """True when a day 0..6 with new snow lost its ski score to wind or rain."""
    for i in range(7):
        d = day_by_index(resort, i)
        dv = g(d, "derived") or {}
        if not is_num(day_snow(resort, d)) or day_snow(resort, d) < 1:
            continue
        rr = dv.get("rain_risk") or {}
        prev_rr = g(day_by_index(resort, i - 1), "derived", "rain_risk") or {}
        if dv.get("ski_score_cm") is None and (dv.get("wind_hold") is True or rr.get("mid") is True
                                               or rr.get("summit") is True or prev_rr.get("mid") is True):
            return True
    return False


def best_display_num(resort, units):
    """(display value, tag) for the best ski day: the one number the cell and
    the verdict both quote, from data.best_display_value. The tag says where a
    fallen amount came from; only the verdict sentence voices it — table cells
    show the bare number (Never 12: no repeated caption fragments)."""
    if best_ski_day(resort) is None:
        return None, None
    return best_display_value(resort, units), best_ski_display(resort)[1]


def best_cell(resort, ctx_units):
    """'~12 cm on Sun' for the qualifying best ski day, else why not.

    The number is the one the page stands behind: the measured figure when
    today's best-day snow has already fallen (then bare — measured figures
    carry no "~"), else the model's, marked "~". Never blank (spec 0.1
    item 5): "No ski day, wind", "No ski day, rain" or the dash.
    """
    i = best_ski_day(resort)
    if i is not None:
        w = short_day_word(resort, i)
        v, tag = best_display_num(resort, ctx_units)
        if w and v is not None:
            amt = disp_cell(v, ctx_units, model=tag not in ("measured", "reported"))
            if w == "Today":
                return f"{amt} today"
            if w == "Yesterday":
                return f"{amt} yesterday"
            return f"{amt} on {w}"
    reasons = _spoil_reasons(resort)
    if "wind" in reasons:
        return "No ski day, wind"
    if "rain" in reasons:
        return "No ski day, rain"
    return DASH


def watch_resort(r):
    """Watch for, next 3 days, for the resort tables: at most 2 items, rain first."""
    out = []
    t = rain_threat_72h(r)
    if t is not None and t["kind"] == "threat":
        out.append("rain" if t["phase"] == "rain" else "wet snow")

    def within(key):
        return any(isinstance(i, int) and 0 <= i <= 2 for i in g(r, "flags", key) or [])

    if within("wind_hold_days"):
        out.append("strong wind")
    return out[:2]


def option_label(resort):
    region = resort.get("region")
    name = resort.get("name") or "?"
    if region == name:
        # "Sierra Nevada, Sierra Nevada" says nothing twice; the country does.
        region = resort.get("country") or None
    return f"{name}, {region}" if region else name


def where_line(resort, ctx):
    """'Japan · 730–1,860 m'"""
    place = resort.get("country") or ""
    base = elev_str(resort.get("elevation_m"), ctx.units)
    summit = elev_str(resort.get("summit_elevation_m"), ctx.units)
    u = ctx.elev_unit
    if base is None:
        el = ""
    elif summit is None:
        el = f"{base}{THIN}{u} elevation"
    else:
        el = f"{base}–{summit}{THIN}{u} elevation"
    return DOT.join(x for x in (escape(place), el) if x)


def _period_phrase(resort, block):
    d = local_date(block.get("start_local"))
    today = resort_today(resort)
    label = block.get("label")
    if d is None or today is None or label not in ("night", "morning", "afternoon", "evening"):
        return None
    if label == "night":
        prev = d - timedelta(days=1)
        if prev == today:
            return "tonight"
        if prev == today - timedelta(days=1):
            return "early this morning"
        return f"{WEEKDAYS[prev.weekday()]} night"
    if d == today:
        return f"this {label}"
    if d == today + timedelta(days=1):
        return f"tomorrow {label}"
    return f"{WEEKDAYS[d.weekday()]} {label}"


def _date_word(resort, d):
    """Short day word for mid-sentence use: today / tomorrow / yesterday / Sun / Wed 25."""
    if d is None:
        return None
    w = rel_word(resort, d)
    return w[:3] if w in WEEKDAYS else w


def _possessive_day(resort, d):
    """"Sun's" / "today's" / "yesterday's" for a date (short weekday forms
    throughout: a long form next to a short one breaks the date rule)."""
    w = rel_word(resort, d)
    w = w[:3] if w in WEEKDAYS else w
    return f"{w}&#8217;s"


def _disp_amount(n, ctx, model=False):
    """A display-unit int as a prose amount: '48 cm' / '9″', unit glued on.

    Inches use the double-prime everywhere: "9 in snow" read as the
    preposition (user, 27 Nov). model=True prefixes "~", the page's one mark
    for a model figure (never on zero)."""
    s = f"{n:,}"
    a = f"{prose_num(s)}″" if ctx.units == "in" else f"{prose_num(s)}{THIN}{ctx.snow_unit}"
    return f"~{a}" if model and n else a


def _snow_amount(v, ctx, model=False):
    """A cm value as a prose amount ('48 cm' / '9″'), or None."""
    n = snow_val(v, ctx.units)
    return None if n is None else _disp_amount(n, ctx, model)


def disp_cell(n, units, model=False):
    """A table-cell amount with its unit glued on ('~21 cm', '8″'), or the
    dash. n is a display-unit int; model marks a model figure with "~"."""
    if n is None:
        return DASH
    t = "~" if model and n else ""
    if units == "in":
        return f"{t}{n:,}″"
    return f"{t}{n:,}{THIN}cm"


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _fallen_words(hours):
    """The window a recent_fallen() figure covers, in words."""
    return {72: "last 3 days", 48: "last 2 days", 24: "since yesterday"}.get(hours, "recently")


# --------------------------------------------------------------------------
# Level 0: the top bar
# --------------------------------------------------------------------------

def updated_ago(updated):
    """'Updated 2 hours ago' / 'Updated 40 minutes ago' / 'Updated 3 days ago'."""
    if updated is None:
        return "Update time unknown"
    age = (datetime.now(timezone.utc) - updated).total_seconds()
    if age < 0:
        age = 0
    mins = age / 60
    if mins < 60:
        m = max(1, _round(mins))
        return f"Updated {m} minute{'s' if m != 1 else ''} ago"
    hrs = mins / 60
    if hrs < 48:
        h = _round(hrs)
        return f"Updated {h} hour{'s' if h != 1 else ''} ago"
    return f"Updated {_round(hrs / 24)} days ago"


def topbar(current, resort_href, all_href, cm_href, in_href, units, updated, resort,
           powder_href=None, powder_hot=False):
    """One small line: Resort search · All resorts · Powder · cm / in · Updated 2 h ago.

    "Powder" is in the accent on the other pages only when some resort clears a
    powder threshold (the accent marks fresh snow); on its own page it is the
    current tab in ink, so that page keeps one accent: the selected resort's row.
    """
    tabs = []
    items = [("resort", "Resort search", resort_href), ("all", "All resorts", all_href)]
    if powder_href:
        items.append(("powder", "Powder", powder_href))
    for key, text, href in items:
        cls = "nav-cur" if key == current else "nav-other"
        if key == "powder":
            cls += " nav-ink" if key == current else (" nav-hot" if powder_hot else "")
        tabs.append(f'<a class="{cls}" href="{escape(href)}" target="_self">{text}</a>')
    u = []
    for key, href in (("cm", cm_href), ("in", in_href)):
        cls = "nav-cur" if key == units else "nav-other"
        u.append(f'<a class="{cls}" href="{escape(href)}" target="_self">{key}</a>')
    title = escape(update_line(updated, resort))
    upd = f'<span class="upd" title="{title}">{updated_ago(updated)}</span>'
    return (f'<div class="topbar label">{"".join(tabs)}'
            f'<span class="units">{u[0]} / {u[1]}</span>{upd}</div>')


# --------------------------------------------------------------------------
# Level 2: the verdict sentence (spec 2.2)
# --------------------------------------------------------------------------

def has_base(resort):
    """True when a measured or resort-reported depth is at least
    GO_SKI_BASE_MIN_CM: the condition for the verdict's "Ski {day}" form."""
    dep = resort.get("depth") or {}
    v = dep.get("value_cm")
    return dep.get("kind") in ("measured", "resort_report") and is_num(v) and v >= GO_SKI_BASE_MIN_CM


def _ski_sentence(resort, ctx):
    """The best-day verdict. "Ski {day} ..." only on an established base
    (has_base); otherwise the same day and amount as a fact ("New snow ...").
    A missing depth adds nothing: the ground line already says "No snow depth report."
    """
    i = best_ski_day(resort)
    if i is None:
        return None
    go = has_base(resort)
    score = g(resort, "week", "best_ski_day_score_cm")
    d = local_date(g(day_by_index(resort, i), "date"))
    if d is None:
        return None
    dw = _date_word(resort, d)
    # The same amount the Best ski day cell shows, from the days' own values.
    amt_cm = best_ski_amount(resort)
    amt = _snow_amount(amt_cm if is_num(amt_cm) else score, ctx, model=True)
    # Wet weather the day after the ski day changes the advice: go before it.
    t = rain_threat_72h(resort)
    if t is not None and t["kind"] == "threat" and t["date"] == d + timedelta(days=1):
        word = "rain" if t["phase"] == "rain" else "wet snow"
        tw = _date_word(resort, t["date"])
        if go:
            return f"Ski {dw}; {word} arrives {tw}."
        wet = f"Rain on {tw}." if t["phase"] == "rain" else f"Wet snow on {tw}."
        return f"New snow through {dw}: {amt}. {wet}" if amt else f"New snow {dw}; {word} arrives {tw}."
    # The score is the previous day's snow plus what falls in lift hours on the day.
    own = day_snow(resort, day_by_index(resort, i))
    ps = day_snow(resort, day_by_index(resort, i - 1))
    pd = d - timedelta(days=1)
    if i == 0 and is_num(ps) and ps >= 0.5:
        # Yesterday is not a row on the page, so a sum that includes it can't be
        # checked (spec 0.1.3). Quote what has actually fallen instead: the
        # measured report when one exists (bare — measured figures carry no
        # "~"), else the model marked "~" — the same figure the ground line
        # shows. Never an average of the two.
        if is_num(own) and own > ps:
            n = _snow_amount(own, ctx, model=True)
            return f"Ski {dw}: {n} new snow." if go else f"New snow {dw}: {n}."
        v, src, hours = recent_fallen(resort)
        if v is not None and src != "model":
            n = _snow_amount(v, ctx)
            return (f"Ski {dw}: {n} new snow, {_fallen_words(hours)}." if go
                    else f"{_cap(_fallen_words(hours))}: {n} new snow.")
        n = _snow_amount(v, ctx, model=True) if v is not None else amt
        return (f"Ski {dw}: {n} new snow." if go
                else f"{_cap(_fallen_words(hours))}: {n} new snow.")
    # Future ski days quote the same number as the Best ski day cell, composed
    # from the visible daily rows (one row, or two adjacent rows added up).
    if not is_num(ps) or ps < 0.5:
        n = _snow_amount(own if is_num(own) and own >= 0.5 else score, ctx, model=True)
        return f"Ski {dw}: {n} new snow." if go else f"New snow {dw}: {n}."
    if not is_num(own) or own < 0.5:
        n = _snow_amount(ps, ctx, model=True)
        return (f"Ski {dw} after {_possessive_day(resort, pd)} {n} snow." if go
                else f"New snow through {dw}: {n}.")
    dn, tag = best_display_num(resort, ctx.units)
    n = amt if dn is None else _disp_amount(dn, ctx, model=tag not in ("measured", "reported"))
    pw = _date_word(resort, pd)
    # The window is the two adjacent snow days, named in full: "since {day}"
    # claimed an open-ended window.
    return f"Ski {dw}: {n} snow {pw} and {dw}." if go else f"New snow through {dw}: {n}."


def _first_ext_snow_day(resort, ctx):
    """First day 7..14 with 5 cm or more, as 'Next snow on Dec 8: ~6 cm.'

    Month-form date: a bare 'Tue 8' reads as day + centimetres everywhere
    else on the page, so as a date it was incomprehensible (user, 27 Nov)."""
    for i in range(7, 15):
        d = day_by_index(resort, i)
        v = day_snow(resort, d)
        if is_num(v) and v >= POWDER_MIN_CM:
            dd = local_date(d.get("date"))
            if dd:
                return (f"Next snow on {MONTHS[dd.month - 1][:3]} {dd.day}: "
                        f"{_snow_amount(v, ctx, model=True)}.")
    return None


def verdict(resort, ctx, updated):
    """The first sentence on the page: which day to ski, or that there isn't one."""
    age = age_hours(updated)
    if age is not None and age > 36:
        if age < 48:
            return f"Warning: forecast is {_round(age)} hours old."
        return f"Warning: forecast is {max(2, _round(age / 24))} days old."
    n3 = next3(resort)
    if n3 is None:
        return "No forecast for this resort."
    s = _ski_sentence(resort, ctx)
    if s:
        return s
    wk = week_total(resort)
    if is_num(wk) and wk < POWDER_MIN_CM:
        nxt = _first_ext_snow_day(resort, ctx)
        # Quote the same total the daily table's total row shows.
        wn = week_display_val(resort, ctx.units)
        if wn == 0 or wn is None:
            if nxt:
                return f"Models show no snow next 7 days. {nxt}"
            et = extended_total(resort)
            if is_num(et) and snow_val(et, ctx.units) == 0:
                return "Models show no snow next 15 days."
            return "Models show no snow next 7 days."
        out = f"Only {_disp_amount(wn, ctx, model=True)} snow next 7 days."
        return f"{out} {nxt}" if nxt else out
    if _snow_spoiled(resort):
        # The code only proves some snowy day lost its ski score, not every one.
        return "Wind or rain on some snow days."
    if n3 < 1:
        best = g(resort, "week", "best_snow_day")
        if isinstance(best, int) and best >= 3:
            d = local_date(g(day_by_index(resort, best), "date"))
            if d:
                return f"Models show no snow until {_date_word(resort, d)}."
    blocks = [b for b in (g(resort, "next_72h", "summit_by_period") or []) if is_num(b.get("snowfall_cm"))]
    if n3 >= 1 and blocks and has_summit(resort):
        top = max(blocks, key=lambda b: b["snowfall_cm"])
        phrase = _period_phrase(resort, top) if top["snowfall_cm"] > 0 else None
        if phrase:
            return f"Models put most snow {phrase}."
    return "No powder day next 7 days."


# --------------------------------------------------------------------------
# Level 3: hero number, label and caption
# --------------------------------------------------------------------------

def window_words(resort):
    """'Fri to Mon' / 'Fri to Sun night' from the headline window."""
    a = g(resort, "next_72h", "window_start_local")
    b = g(resort, "next_72h", "window_end_local")
    try:
        s = datetime.fromisoformat(a)
        e = datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return "next 72 hours"
    if e.hour < 6:
        end = f"{WEEKDAYS[(e - timedelta(days=1)).weekday()][:3]} night"
    else:
        end = WEEKDAYS[e.weekday()][:3]
    return f"{WEEKDAYS[s.weekday()][:3]} to {end}"


def _day0_disagrees(models):
    """True only when day 0 has two or more model values that differ by the
    fetcher's own tolerance (fetch_data_v2: hi - lo > max(3, 0.3 * hi)).
    A null agree_through also comes from a missing model value; that is not
    disagreement."""
    for day in models.get("days") if isinstance(models.get("days"), list) else []:
        if isinstance(day, dict) and day.get("index") == 0:
            vals = day.get("snowfall_cm")
            vs = [v for v in (vals.values() if isinstance(vals, dict) else []) if is_num(v)]
            if len(vs) < 2:
                return False
            hi, lo = max(vs), min(vs)
            return hi - lo > max(3.0, 0.3 * hi)
    return False


def confidence_tag(resort):
    models = resort.get("models")
    if not isinstance(models, dict):
        return None  # no model data: no claim ("firmer than later" had none behind it)
    at = models.get("agree_through")
    if isinstance(at, int):
        if at >= 6:
            return "Models agree on snow all week."
        if at == 0:
            w = day_word(resort, 0) or "today"
            # Day 0 can be yesterday in a stale file; agreement on a past
            # day says nothing about the week ahead, so say nothing.
            if w == "yesterday":
                return None
            return f"Models agree on snow only {w}."
        w = day_word(resort, at)
        return f"Models agree on snow through {w}." if w else None
    if at is None and _day0_disagrees(models):
        # Only when two or more real day-0 values differ by the fetcher's own
        # tolerance: a missing value must not become a claim.
        w = day_word(resort, 0) or "today"
        if w == "yesterday":
            return None
        return f"Models disagree on snow from {w}."
    return None  # too few model values to say either way: say nothing


def hero_range_words(resort, v, ctx):
    """'Likely 9–24 cm': the disclosed spread beside the hero (docs/model_improvements.md).

    The hero stays the model's number. The range is how far real 3-day totals
    landed from forecasts of this size in this country last winter (middle two
    thirds of measured / forecast, scripts/validation/hero_ranges.py). None when
    the country was never backtested, the forecast is under 5 cm, or the window
    holds recounted rain (the table was built from the raw model, not the recount).
    """
    band = hero_elevation(resort)
    if any(seg_recounted(resort, s, band) for s in g(resort, "next_72h", "by_day", band) or []):
        return None
    r = likely_range(resort.get("country"), v)
    if r is None:
        return None
    lo, hi = snow_val(r[0], ctx.units), snow_val(r[1], ctx.units)
    if lo is None or hi is None or hi <= lo:
        return None
    # A literal narrow no-break space: the caption is escaped as a whole.
    if ctx.units == "in":
        return f"Likely {lo:,}–{hi:,}″"
    return f"Likely {lo:,}–{hi:,}\u202f{ctx.snow_unit}"


def hero_block(resort, ctx):
    """The two numbers a skier needs in big lettering (user directive, 27 Nov):
    new snow next 3 days (accent, dominant) and current snow depth (ink, one
    scale step down). Depth unknown stays an honest big dash."""
    v = next3(resort)
    if v is None:
        return ""
    elev = _cap(BAND_WORD.get(hero_elevation(resort), "base"))
    cap = [f"{elev}: {window_words(resort)}."]
    rng = hero_range_words(resort, v, ctx)
    if rng:
        cap.append(f"{rng}.")
    if snow_val(v, ctx.units) and snow_val(v, ctx.units) > 0:
        tag = confidence_tag(resort)
        if tag:
            cap.append(tag)
    unit = "″" if ctx.units == "in" else f'{THIN}{ctx.snow_unit}'
    dv = g(resort, "depth", "value_cm")
    if is_num(dv):
        # The depth numeral is measured: it stays bare.
        dnum = f'{snow_str(dv, ctx.units)}<span class="unit">{unit}</span>'
        dlab = f"Snow depth{_depth_where(resort)}" or "Snow depth"
    else:
        dnum = DASH
        dlab = "Snow depth"
    mark = "~" if snow_val(v, ctx.units) else ""  # the 3-day number is the model's
    return (f'<div class="herorow gap1">'
            f'<div class="herostat"><div class="hero">{mark}{snow_str(v, ctx.units)}'
            f'<span class="unit">{unit}</span></div>'
            f'<div class="herolab body">Snow next 3 days</div></div>'
            f'<div class="herostat herodepth"><div class="big">{dnum}</div>'
            f'<div class="herolab body">{escape(dlab)}</div></div>'
            f'</div>'
            f'<p class="caption gap0">{escape(" ".join(cap))}</p>')


# --------------------------------------------------------------------------
# Level 4: the mountain strip (spec 2.3)
# --------------------------------------------------------------------------

def strip_bands(resort):
    """[(data band, label, low m, high m)], merging bands the model can't separate."""
    spec = [("summit", "Summit", "summit_elevation_m"), ("mid", "Mid", "mid_elevation_m"),
            ("base", "Base", "elevation_m")]
    bands = g(resort, "quality", "bands") or {}
    have = {e for e, _, key in spec if resort.get(key) is not None}
    if bands.get("mid_summit_same_72h") is True and {"summit", "mid"} <= have:
        if bands.get("base_mid_same_72h") is True and "base" in have:
            return [("summit", "Base–summit", resort.get("elevation_m"), resort.get("summit_elevation_m"))]
        out = [("summit", "Mid–summit", resort.get("mid_elevation_m"), resort.get("summit_elevation_m"))]
        if "base" in have:
            out.append(("base", "Base", resort.get("elevation_m"), None))
        return out
    return [(e, name, resort.get(key), None) for e, name, key in spec if e in have]


def _wet(seg):
    a, b = seg.get("rain_phase_mm"), seg.get("mixed_phase_mm")
    return (a if is_num(a) else 0) + (b if is_num(b) else 0)


def _rain_amount(mm, ctx):
    if not is_num(mm) or mm < RAIN_AMOUNT_MM:
        return ""
    if ctx.units == "in":
        return f" ({mm / 25.4:.1f}″)"
    return f" ({_round(mm)}{THIN}mm)"


def _strip_day(resort, d):
    """'today' or the weekday as the daily table shows it ('Sat'), never 'tomorrow'."""
    w = _date_word(resort, d)
    return WEEKDAYS[d.weekday()][:3] if w == "tomorrow" else w


def _day_span(resort, dates):
    if not dates:
        return None
    a, b = min(dates), max(dates)
    if a == b:
        return _strip_day(resort, a)
    return f"{WEEKDAYS[a.weekday()][:3]}–{WEEKDAYS[b.weekday()][:3]}"


def strip_text(resort, band, ctx):
    """Snow and rain story for one band over the headline window, or None."""
    segs = g(resort, "next_72h", "by_day", band)
    total = next3_band(resort, band)
    if not isinstance(segs, list) or not segs or not is_num(total):
        return None
    if any(s.get("phase") is None for s in segs):
        return None
    rows = [(local_date(s.get("date")), s) for s in segs]
    rows = [(d, s) for d, s in rows if d is not None]
    rain = [(d, s) for d, s in rows if s.get("phase") == "rain" and _wet(s) >= WET_MIN_MM]
    wet = [(d, s) for d, s in rows if s.get("phase") == "mixed" and _wet(s) >= WET_MIN_MM
           and not seg_recounted(resort, s, band)]
    snowy = [d for d, s in rows if s.get("phase") in ("snow", "mixed")
             and is_num(seg_snow(resort, s, band)) and seg_snow(resort, s, band) >= 0.5]
    pure = [d for d, s in rows if (s.get("phase") == "snow" or seg_recounted(resort, s, band))
            and is_num(seg_snow(resort, s, band)) and seg_snow(resort, s, band) >= 0.5]
    n = snow_val(total, ctx.units)
    su = "″" if ctx.units == "in" else f"{THIN}{ctx.snow_unit}"
    snow = f"{n}{su} snow" if n and n >= 1 else None
    if rain:
        rd = [d for d, _ in rain]
        span = _day_span(resort, rd)
        on = span if span in ("today", "yesterday") else f"on {span}"
        rtxt = f"rain {on}{_rain_amount(sum(_wet(s) for _, s in rain), ctx)}"
        if snow:
            if snowy and min(snowy) < min(rd):
                return f"{snow}, then {rtxt}"
            return f"{rtxt}, then {snow}"
        after = [d for d, _ in wet if d >= min(rd)]
        if after:
            return f"{rtxt}, wet snow {_on_day(resort, min(after))}"
        return f"{rtxt}, no snow"
    if wet:
        fw = min(d for d, _ in wet)
        if snow:
            if pure and min(pure) > fw:
                return f"wet snow {_on_day(resort, fw)}, then {snow}"
            return f"{snow}, wet snow {_on_day(resort, fw)}"
        return f"wet snow {_on_day(resort, fw)}"
    return snow or "dry"


def strip_lines(resort, ctx):
    """[(label, text)] for the mountain strip; one 'Whole mountain' line when all dry."""
    out = []
    u = ctx.elev_unit
    for band, name, lo, hi in strip_bands(resort):
        t = strip_text(resort, band, ctx)
        if t is None:
            continue
        lo_s = elev_str(lo, ctx.units)
        if hi is not None:
            h = f"{lo_s}–{elev_str(hi, ctx.units)}{THIN}{u}"
        else:
            h = f"{lo_s}{THIN}{u}" if lo_s else ""
        out.append((f"{name} {h}".strip(), t))
    if out and all(t == "dry" for _, t in out) and len(out) > 1:
        return [("Whole mountain", "dry")]
    return out


def _on_day(resort, d):
    """'today' / 'on Sat' for a condition phrase: a day name takes "on"."""
    w = _strip_day(resort, d)
    return w if w in ("today", "yesterday") else f"on {w}"


def _band_spoiler(resort, band):
    """'rain on Sun' / 'wet snow today' / '' for one band over the headline window.

    The strip's own tests: rain is a rain-phase segment with at least
    WET_MIN_MM of water (first day it happens); wet is a mixed-phase segment
    with that much water that is not recounted as snow, and only when no rain.
    """
    segs = g(resort, "next_72h", "by_day", band)
    if not isinstance(segs, list):
        return ""
    rows = [(local_date(s.get("date")), s) for s in segs if isinstance(s, dict)]
    rows = [(d, s) for d, s in rows if d is not None]
    rain = [d for d, s in rows if s.get("phase") == "rain" and _wet(s) >= WET_MIN_MM]
    wet = [d for d, s in rows if s.get("phase") == "mixed" and _wet(s) >= WET_MIN_MM
           and not seg_recounted(resort, s, band)]
    for word, days in (("rain", rain), ("wet snow", wet)):
        if days:
            return f"{word} {_on_day(resort, min(days))}"
    return ""


def band_rows(resort, ctx):
    """[(wide label, narrow label, value, spoiler)], summit on top (skim spec 1.1)."""
    out = []
    u = ctx.elev_unit
    for band, name, lo, hi in strip_bands(resort):
        lo_s = elev_str(lo, ctx.units)
        if lo_s is None:
            continue
        hi_s = elev_str(hi, ctx.units) if hi is not None else None
        wide = f"{name} {lo_s}–{hi_s}{THIN}{u}" if hi_s else f"{name} {lo_s}{THIN}{u}"
        # The narrow label carries the unit too: a bare height is not an elevation.
        out.append((wide, f"{name} {lo_s}{THIN}{u}", snow_val(next3_band(resort, band), ctx.units),
                    _band_spoiler(resort, band)))
    return out


def bands_block(resort, ctx):
    """Three rows on the line grid: label, ink bar, numeral, at most two words."""
    rows = band_rows(resort, ctx)
    if not rows:
        return ""
    vmax = scale_max([v for _, _, v, _ in rows], ctx.units)
    cells = []
    for wide, narrow, v, word in rows:
        # The numeral is a model figure: "~" and the unit travel with it.
        cells.append(f'<span class="bl"><span class="lw">{wide}</span><span class="ln">{narrow}</span></span>'
                     f'<span class="bc">&#8203;{bar(v, vmax)}</span>'
                     f'<span class="n">{disp_cell(v, ctx.units, model=True)}</span>'
                     f'<span class="bw">{escape(word)}</span>')
    return '<div class="bands gap1">' + "".join(cells) + "</div>"


def strip_block(resort, ctx):
    """Kept for callers of the old name: the band rows."""
    return bands_block(resort, ctx)


def _bands_low_to_high(resort):
    return [b for b in ("base", "mid", "summit") if is_num(resort.get(BAND_ELEV_KEY[b]))]


def split_line(resort, ctx):
    """'Today: snow above 2,220 m; wet snow below.' or None.

    For the headline window's snowiest day (at least POWDER_MIN_CM at the top) on which the snow line cuts through
    the resort: the top band gets snow, a lower band gets rain or wet snow
    (at least WET_MIN_MM of water). The height is the lowest band that is all
    snow, by the model's own hour-by-hour phase, so the sentence is true at
    the heights we have and never quotes the day's peak freezing level.
    """
    bands = _bands_low_to_high(resort)
    if len(bands) < 2:
        return None
    by = {}
    for b in bands:
        segs = g(resort, "next_72h", "by_day", b)
        if not isinstance(segs, list):
            return None
        by[b] = {s.get("date"): s for s in segs if isinstance(s, dict)}
    dates = sorted(set.intersection(*[set(v) for v in by.values()]) - {None})
    best = None
    for ds in dates:
        ph = {}
        for b in bands:
            s = by[b][ds]
            p = "snow" if seg_recounted(resort, s, b) else s.get("phase")
            if p is None:
                break
            ph[b] = p
        else:
            top = bands[-1]
            ts = seg_snow(resort, by[top][ds], top)
            if ph[top] != "snow" or not is_num(ts) or ts < POWDER_MIN_CM:
                continue
            k = len(bands) - 1
            while k > 0 and ph[bands[k - 1]] == "snow":
                k -= 1
            if k == 0:
                continue
            low = bands[k - 1]
            if ph[low] not in ("rain", "mixed") or _wet(by[low][ds]) < WET_MIN_MM:
                continue
            cand = (ts, ds, bands[k], "rain" if ph[low] == "rain" else "wet snow")
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return None
    d = local_date(best[1])
    if d is None:
        return None
    h = elev_str(resort.get(BAND_ELEV_KEY[best[2]]), ctx.units)
    return f"{_cap(_strip_day(resort, d))}: snow above {h}{THIN}{ctx.elev_unit}; {best[3]} below."


def low_threat(resort, ctx):
    """'Wet snow below 2,220 m Sunday.' / 'Snow turns rain below 2,220 m Sunday.' or None.

    Only for the verdict's own day (the best ski day), only when the top band
    is all snow that day and a lower band gets rain or wet snow (at least
    WET_MIN_MM of water). The height is the lowest band that is all snow, by
    the model's own phase. Six words or fewer (skim spec 1.2, Decision).
    """
    i = best_ski_day(resort)
    if i is None:
        return None
    d = local_date(g(day_by_index(resort, i), "date"))
    bands = _bands_low_to_high(resort)
    if d is None or len(bands) < 2:
        return None
    ph = {}
    for b in bands:
        segs = g(resort, "next_72h", "by_day", b)
        s = next((x for x in segs or [] if isinstance(x, dict) and local_date(x.get("date")) == d), None)
        if s is None:
            return None
        p = "snow" if seg_recounted(resort, s, b) else s.get("phase")
        if p is None:
            return None
        ph[b] = (p, s)
    if ph[bands[-1]][0] != "snow":
        return None
    k = len(bands) - 1
    while k > 0 and ph[bands[k - 1]][0] == "snow":
        k -= 1
    if k == 0:
        return None
    p, s = ph[bands[k - 1]]
    if p not in ("rain", "mixed") or _wet(s) < WET_MIN_MM:
        return None
    h = elev_str(resort.get(BAND_ELEV_KEY[bands[k]]), ctx.units)
    if h is None:
        return None
    lead = "Wet snow" if p == "mixed" else "Snow turns rain"
    return f"{lead} below {h}{THIN}{ctx.elev_unit} {rel_word(resort, d)}."


def recount_note(resort):
    """'We count Monday's model rain as snow. It's below freezing up there.'

    Said once, under the daily table, whenever a number on the page includes
    model rain recounted as snow (data._recount_mm): a day row at the hero
    band, or a strip band inside the headline window.
    """
    hero = hero_elevation(resort)
    dates = set()
    for i in range(7):
        d = day_by_index(resort, i)
        if d is not None and recounted(resort, d, hero):
            dd = local_date(d.get("date"))
            if dd:
                dates.add(dd)
    for band, _name, _lo, _hi in strip_bands(resort):
        for s in g(resort, "next_72h", "by_day", band) or []:
            if seg_recounted(resort, s, band):
                dd = local_date(s.get("date"))
                if dd:
                    dates.add(dd)
    if not dates:
        return ""
    # The WHY must travel with the claim: "rain counted as snow" alone read
    # as malpractice (user, 27 Nov). It is the snow-line correction — the
    # band is below freezing in every wet hour, the model's low grid cell
    # mislabelled the precipitation (docs/snowline_decision.md, NWS-validated).
    ds = [_cap(_strip_day(resort, d)) for d in sorted(dates)]
    if len(ds) == 1:
        return f"Freezing air {ds[0]}: rain falls as snow."
    if len(ds) == 2:
        return f"Freezing air {ds[0]}, {ds[1]}: rain falls as snow."
    return "Freezing air: the model's rain falls as snow."


# --------------------------------------------------------------------------
# Level 6: the ground line
# --------------------------------------------------------------------------

def _depth_source_words(resort):
    """'Morse Lake station, yesterday' / 'resort report, today', or ''."""
    kind = g(resort, "depth", "kind")
    lab = g(resort, "depth", "source_label") or ""
    when = None
    if kind == "measured":
        m = re.match(r"^\S+\s+(.+?)\s+station", lab)
        src = f"{m.group(1)} station" if m else "station"
        od = g(resort, "depth", "observed_utc")
        if not od:
            st = next((s for s in resort.get("stations") or [] if is_num(s.get("depth_cm"))), None)
            od = g(st, "observed_date")
        when = local_date(od) if isinstance(od, str) else None
    elif kind == "resort_report":
        src = "resort report"
        when = local_date(g(resort, "depth", "observed_utc"))
    else:
        return ""
    w = _date_word(resort, when) if when else None
    return f"{src}, {w}" if w else src


def _depth_where(resort):
    """Where on the mountain the depth is measured: ' at base', ' mid-mountain', ' near summit', or ''.

    A valley station reading 0 cm next to a summit snowfall reads as a contradiction
    unless the line says the depth is at the base (brief, Never 11: location in the
    stat's own label, never the source).
    """
    lab = g(resort, "depth", "source_label") or ""
    if "mid-mountain" in lab:
        return " at mid-mountain"
    if "(base)" in lab:
        return " at base"
    if g(resort, "depth", "kind") != "measured":
        return ""
    m = re.search(r"(\d[\d,]*) m\b", lab)
    if not m:
        return ""
    h = int(m.group(1).replace(",", ""))
    bands = [(" at base", resort.get("elevation_m")), (" at mid-mountain", resort.get("mid_elevation_m")),
             (" near summit", resort.get("summit_elevation_m"))]
    bands = [(w, e) for w, e in bands if is_num(e)]
    if not bands:
        return ""
    if h < bands[0][1] - 150:
        return " below base"
    return min(bands, key=lambda b: abs(b[1] - h))[0]


def _fallen_series(resort, src, ctx):
    """Display-unit snow for each of the last three 24 h windows, oldest first,
    from the same cumulative 24/48/72 h figures the total comes from; None
    unless all three exist and never fall (no made-up split)."""
    if src == "model":
        m = g(resort, "recent", "modelled", hero_elevation(resort)) or {}
    else:
        m = g(resort, "recent", "measured") or {}
    a, b, c = m.get("h24_cm"), m.get("h48_cm"), m.get("h72_cm")
    if not all(is_num(x) for x in (a, b, c)) or not (0 <= a <= b <= c):
        return None
    return [snow_val(c - b, ctx.units), snow_val(b - a, ctx.units), snow_val(a, ctx.units)]


def _ground_num(v, ctx, approx=False):
    """approx marks a model hindcast with "~", the page's mark for a model
    figure (verdict, snow-cover column). The backtest found these run low:
    measured was 1.2 times them in France and 1.8 times in the US."""
    s = snow_str(v, ctx.units)
    if s is None:
        return DASH
    if approx and s != "0":
        s = "~" + s
    u = "″" if ctx.units == "in" else f"{THIN}{ctx.snow_unit}"
    return f'<span class="gn">{s}</span><span class="gu">{u}</span>'


def ground_line(resort, ctx, vmax=None):
    """'Snow depth, mid-mountain 48 cm    Last 3 days ▁█▃ 23 cm' (skim spec 1.4).

    No provenance tag after the number (brief, Never 11): measured beats
    modelled where recent_fallen chooses the value, not in a caption.
    """
    # Depth itself now sits in big lettering beside the hero (user directive,
    # 27 Nov) — the ground line keeps only the recent-snow story, plus the
    # honest sentence when there is no reading at all.
    parts = []
    dv = g(resort, "depth", "value_cm")
    if not is_num(dv):
        parts.append('<span class="gi">No snow depth report.</span>')
    # The one figure for fallen snow, measured before modelled — the same
    # number the verdict and the best-day cell quote when the snow is down.
    v, src, hours = recent_fallen(resort)
    win = _fallen_words(hours if v is not None else 72)
    lab = f"New snow {win}"  # the figure is fallen new snow, not a depth change
    spark = ""
    if v is not None and hours == 72:
        series = _fallen_series(resort, src, ctx)
        if series is not None:
            vm = vmax or scale_max(series, ctx.units)
            # Spoken/assistive text spells the unit: "″" is not reliably voiced.
            uw = "inches" if ctx.units == "in" else "cm"
            spark = spark_bars(series, None, vm, height=18,
                               label=f"{_cap(win)}, oldest first: " + ", ".join(f"{x} {uw}" for x in series)) + " "
    parts.append(f'<span class="gi">{lab} {spark}{_ground_num(v, ctx, approx=(src == "model"))}</span>')
    return f'<p class="body ground gap1">{"".join(parts)}</p>'


# --------------------------------------------------------------------------
# The first screenful
# --------------------------------------------------------------------------

def screen_error(kind):
    msg = "Forecast data is out of date." if kind == "old" else "Forecast data failed to load."
    return f'<div class="page screen top"><p class="body sentence">{escape(msg)}</p></div>'


def first_screen(resort, nearby, ctx, updated):
    """Everything under the picker down to the ground line (levels 1 to 6)."""
    v = verdict(resort, ctx, updated)
    # The snow-line clause left the verdict (skim spec 1.2, Decision): the day
    # table's marks show it. Only a threat to the verdict's own day stays.
    th = low_threat(resort, ctx) if not v.startswith(("Warning", "No forecast")) else None
    if th:
        v = f"{v} {th}"
    out = [f'<div class="caption gap0">{where_line(resort, ctx)}</div>',
           f'<p class="verdict prose gap1">{v}</p>']
    if next3(resort) is not None:
        out.append(hero_block(resort, ctx))
        out.append(bands_block(resort, ctx))
        # The nearby line ("More snow: Timberline, 13″.") was removed on the
        # user's ruling, 27 Nov: distracting, and "nearby" can be very far
        # away. The comparison table below already answers where else to go.
    out.append(ground_line(resort, ctx, day_vmax(resort, ctx)))
    return '<div class="page">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------
# Level 7 and 8: snow by day, and days 8-15
# --------------------------------------------------------------------------

POWDER_WORD = {"cold_dry": "dry powder", "warm_heavy": "heavy snow"}


def watch_day(resort, day, extended=False):
    """Watch for, one day: at most 2 items, rain first."""
    dv = g(day, "derived") or {}
    out = []
    if not extended:
        band = "mid" if resort.get("mid_elevation_m") is not None else "summit"
        rr = g(dv, "rain_risk") or {}
        mx = g(dv, "mixed_risk") or {}
        # A wet-weather word needs wet weather that day at that band: "rain"
        # only when rain actually falls, "wet snow" only when snow falls.
        rain_mm = g(day, band, "rain_mm")
        sn_cm = g(day, band, "snowfall_cm")
        rains = is_num(rain_mm) and rain_mm >= 1
        snows = is_num(sn_cm) and sn_cm >= 0.5
        if rr.get(band) is True and rains:
            out.append("rain")
        elif mx.get(band) is True and (snows or rains) and not recounted(resort, day, band):
            out.append("wet snow")
    if dv.get("wind_hold") is True:
        out.append("strong wind")
    if not extended:
        pw = POWDER_WORD.get(dv.get("powder_band"))
        if pw:
            out.append(pw)
    return out[:2]


def snow_line_range(day):
    """(low m, high m) of the snow line over the day's wet hours, or None.

    The model's snowfall height where it gives one, else its freezing level
    minus 250 m, the same line the phase rule uses. A range, never the day's
    peak alone: the peak (often from a dry or brief warm hour) overstated
    Timberline's snow line by about 250 m against the Weather Service.
    """
    dv = g(day, "derived") or {}
    lo, hi = dv.get("snowfall_height_precip_min_m"), dv.get("snowfall_height_precip_max_m")
    if not (is_num(lo) and is_num(hi)):
        a, b = dv.get("freezing_level_precip_min_m"), dv.get("freezing_level_precip_max_m")
        if not (is_num(a) and is_num(b)):
            return None
        lo, hi = a - SNOWLINE_BELOW_FL_M, b - SNOWLINE_BELOW_FL_M
    return lo, hi


def _sl_axis(resort):
    """(base m, summit m) for the snow-line axis, or None: never an invented axis."""
    bot, top = resort.get("elevation_m"), resort.get("summit_elevation_m")
    if is_num(bot) and is_num(top) and top > bot:
        return bot, top
    return None


def _snow_line_text(resort, lo, hi, ctx):
    top, bot = resort.get("summit_elevation_m"), resort.get("elevation_m")
    if is_num(top) and lo >= top:
        return "above summit"
    if is_num(bot) and hi <= bot:
        return "below base"
    a, b = elev_str(lo, ctx.units), elev_str(hi, ctx.units)
    if a == b:
        return f"Snow line ~{a}{THIN}{ctx.elev_unit}"
    return f"Snow line ~{a}–{b}{THIN}{ctx.elev_unit}"


def _snow_line(resort, day, ctx):
    """How much of the mountain gets snow on a wet day: "all", "top ~60%",
    "none" (snow_cover_str). A dry day is empty; a wet day with no snow-line
    data is the dash; when the resort's heights are unknown the share can't
    be computed, so the range falls back to words.
    """
    if day is None:
        return ""
    elev = hero_elevation(resort)
    pr = g(day, elev, "precip_mm")
    sn = g(day, elev, "snowfall_cm")
    wet_day = (is_num(pr) and pr >= 1) or (is_num(sn) and sn >= 0.5)
    if not wet_day:
        return ""
    cover = snow_cover_str(resort, day)
    if cover is not None:
        return f'<span class="slt">{cover}</span>'
    rng = snow_line_range(day)
    if rng is None:
        return f'<span class="sld">{DASH}</span>'
    return f'<span class="slt">{_snow_line_text(resort, rng[0], rng[1], ctx)}</span>'


def _day_label(resort, d):
    dd = local_date(g(d, "date"))
    if dd is None:
        return DASH
    today = resort_today(resort)
    if dd == today:
        return "Today"
    if dd == today - timedelta(days=1):
        return "Yesterday"
    return short_day(dd)


def _day_row(resort, ctx, d, v, vmax, is_best=False, extended=False):
    lab = _day_label(resort, d)
    ncls = "n accent" if is_best else "n"
    lcls = ' class="accent"' if is_best else ""
    line = "" if extended else _snow_line(resort, d, ctx)
    watch = ", ".join(watch_day(resort, d, extended)) if d else ""
    # Watch for sits right after the numeral, then the snow line (skim spec 1.7).
    # Every day row is a model figure, so the cell carries "~" and its unit.
    return (f'<span{lcls}>{escape(lab)}</span><span class="bc">&#8203;{bar(v, vmax, is_best)}</span>'
            f'<span class="{ncls}">{disp_cell(v, ctx.units, model=True)}</span>'
            f'<span class="f">{escape(watch)}</span><span class="slc">{line}</span>')


def _vals(resort, ctx, indices):
    return [(i, day_by_index(resort, i), snow_val(day_snow(resort, day_by_index(resort, i)), ctx.units))
            for i in indices]


def week_display_val(resort, units):
    """The 7-day total as the table's rows add up: the sum of the displayed
    (already-rounded) day values, so the total row and any sentence quoting it
    always match the rows, in both units. Falls back to converting the raw
    total only when a day is missing."""
    vals = [snow_val(day_snow(resort, day_by_index(resort, i)), units) for i in range(7)]
    if all(v is not None for v in vals):
        return sum(vals)
    return snow_val(week_total(resort), units)


def day_vmax(resort, ctx):
    """The day table's bar scale (days 1 to 15), shared by every snow mark on the page."""
    vals = [v for _, _, v in _vals(resort, ctx, range(0, 15))]
    return scale_max(vals, ctx.units)


def _sl_head(resort, ctx):
    """The column header: a share of the mountain needs no axis numerals."""
    return '<span class="hd slh">Snow reaches</span>'


def rough_note(resort, ctx):
    """'Amounts from Wednesday on are a rough guide.'

    Said only when a day 5 or more days ahead shows real snow (5 cm), because
    the backtest found little day-by-day skill that far out (correlation 0.31
    five days ahead, about 0.12 at six and seven; docs/model_validation_report.md
    section 3). Days are counted from the resort's own today, not the file's.
    """
    today = resort_today(resort)
    first = None
    for i in range(7):
        d = day_by_index(resort, i)
        dd = local_date(g(d, "date"))
        if dd is None or (dd - today).days < ROUGH_FROM_INDEX:
            continue
        if first is None:
            first = dd
        v = day_snow(resort, d)
        if is_num(v) and v >= POWDER_MIN_CM:
            return f"Amounts from {rel_word(resort, first)} on are a rough guide."
    return None


def days_section(resort, ctx):
    week = _vals(resort, ctx, range(0, 7))
    if all(d is None for _, d, _ in week):
        return ""
    ext = _vals(resort, ctx, range(7, 15))
    vmax = scale_max([v for _, _, v in week + ext], ctx.units)
    best = best_ski_day(resort)
    u = ctx.snow_unit
    # Units sit in each cell, not the header: the header names the measure only.
    head = ('<span class="hd">Day</span><span class="hd"></span><span class="hd r">New snow</span>'
            f'<span class="hd f">Watch for</span>{_sl_head(resort, ctx)}')
    rows = [head]
    for i, d, v in week:
        rows.append(_day_row(resort, ctx, d, v, vmax, is_best=(i == best)))
    wtv = week_display_val(resort, ctx.units)
    rows.append(f'<span class="tot">Snow next 7 days</span>'
                f'<span class="n">{disp_cell(wtv, ctx.units, model=True)}</span>'
                f'<span class="n2 tu"></span><span class="slc"></span>')
    out = ['<div class="label">Snow by day</div>',
           f'<div class="rows days wk gap0">{"".join(rows)}</div>']
    note = recount_note(resort)
    if note:
        out.append(f'<p class="caption sentence gap0">{note}</p>')
    rough = rough_note(resort, ctx)
    if rough:
        out.append(f'<p class="caption sentence gap0">{rough}</p>')
    et = extended_total(resort)
    if is_num(et) and any(d is not None for _, d, _ in ext):
        open_ = any(is_num(day_snow(resort, d)) and day_snow(resort, d) >= POWDER_MIN_CM for _, d, _ in ext)
        erows = [_day_row(resort, ctx, d, v, vmax, extended=True) for _, d, v in ext if d is not None]
        # The summary amount is the sum its own rows show, never a separately
        # rounded total (displayed sums match displayed totals).
        evals = [v for _, d, v in ext if d is not None]
        etv = sum(evals) if evals and all(v is not None for v in evals) else snow_val(et, ctx.units)
        svals = [v for _, _, v in ext]
        # Spoken/assistive text spells the unit: "″" is not reliably voiced.
        uw = "inches" if ctx.units == "in" else "cm"
        spark = spark_bars(svals, None, vmax, height=18,
                           label="Days 8–15: " + ", ".join(DASH if x is None else f"{x} {uw}" for x in svals))
        eu = "″" if ctx.units == "in" else f"{THIN}{u}"
        amt = DASH if etv is None else (f'<span class="en">{"~" if etv else ""}{etv:,}</span>'
                                        f'<span class="eu">{eu}</span>')
        out.append(f'<details class="ext gap1"{" open" if open_ else ""}>'
                   f'<summary class="body"><span class="u">Days 8–15</span> {spark + " " if spark else ""}{amt}</summary>'
                   f'<div class="rows days wk gap0">{"".join(erows)}</div></details>')
    return '<div class="page sec">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------
# Resort tables (compare, and the All resorts page)
# --------------------------------------------------------------------------

# No short header forms: "N-day snow" and "Depth" are retired (copy audit),
# and phone headers already wrap to two lines (white-space:normal).
NARROW_HEADS = {}


def th_text(text):
    """A column header: full form wide, short form at phone width."""
    short = NARROW_HEADS.get(text)
    if not short:
        return text
    return f'<span class="thw">{text}</span><span class="thn">{short}</span>'


def _depth_title(r):
    bits = []
    kind = g(r, "depth", "kind")
    lab = g(r, "depth", "source_label")
    if isinstance(lab, str) and lab:
        bits.append(lab)
    elif kind:
        bits.append(kind.replace("_", " "))
    od = g(r, "depth", "observed_utc")
    d = local_date(od) if isinstance(od, str) else None
    if d:
        bits.append(short_date(d))
    return ", ".join(bits)


def resort_row(r, units, selected_id, href, vmax, km=None, show_country=True, show_ext=False):
    """One row: Resort (country) · [km] · bar · 3-day · 7-day · Last 3 days · Best ski day · Watch for · Snow depth · [8-15]."""
    sel = r.get("id") == selected_id
    v3 = snow_val(next3(r), units)
    name = f'<a class="plain" href="{escape(href)}" target="_self">{escape(r.get("name") or "?")}</a>'
    if show_country and r.get("country"):
        name += f' <span class="label">{escape(r["country"])}</span>'
    cells = [f'<td class="first">{name}</td>']
    if km is not None:
        cells.append(f'<td class="r">{km}</td>')
    wt = snow_val(week_total(r), units)
    l3v, l3src, _h = recent_fallen(r)
    l3 = snow_val(l3v, units)
    dv = snow_val(g(r, "depth", "value_cm"), units)
    dt = _depth_title(r) if dv is not None else ""
    title = f' title="{escape(dt)}"' if dt else ""
    # Units sit in each cell; "~" marks model figures. Last 3 days is model-or-
    # measured per resort, so its mark is decided per cell; depth is measured.
    cells += [
        f'<td class="barcol">{bar(v3, vmax, best=False)}</td>',
        f'<td class="r">{disp_cell(v3, units, model=True)}</td>',
        f'<td class="r wk7">{disp_cell(wt, units, model=True)}</td>',
        f'<td class="r">{disp_cell(l3, units, model=(l3src == "model"))}</td>',
        f'<td>{best_cell(r, units)}</td>',
        f'<td class="wf">{escape(", ".join(watch_resort(r)))}</td>',
        f'<td class="r"{title}>{disp_cell(dv, units)}</td>',
    ]
    if show_ext:
        et = snow_val(extended_total(r), units)
        cells.append(f'<td class="r">{disp_cell(et, units, model=True)}</td>')
    trc = ' class="is-selected"' if sel else ""
    return f"<tr{trc}>{''.join(cells)}</tr>"


def compare_section(resort, rows, ctx, total_count, all_href):
    """Compare resorts: Show, Sort by and the table, with nothing in between."""
    country = resort.get("country") or "Country"
    all_word = "All 1 resort" if total_count == 1 else f"All {total_count} resorts"
    show = [ctx.link("Nearby", "is-current" if ctx.scope == "nearby" else "", scope="nearby"),
            ctx.link(country, "is-current" if ctx.scope == "country" else "", scope="country"),
            f'<a href="{escape(all_href)}" target="_self">{all_word}</a>']
    sorts = [ctx.link(text, "is-current" if ctx.sort == key else "", sort=key)
             for key, text in (("next3", "Snow next 3 days"), ("week", "Snow next 7 days"),
                               ("best", "Models' best day"), ("depth", "Snow depth"), ("name", "A–Z"))]
    near = ctx.scope == "nearby"
    du = "km" if ctx.units == "cm" else "mi"
    vmax = scale_max([snow_val(next3(r), ctx.units) for r in rows], ctx.units)
    best_head = th_text("Models' best day")
    head = ('<tr><th>Resort</th>' + (f'<th class="r">{du}</th>' if near else "")
            + f'<th class="barcol"></th><th class="r">{th_text("Snow next 3 days")}</th>'
            f'<th class="r wk7">{th_text("Snow next 7 days")}</th>'
            f'<th class="r">{th_text("Last 3 days")}</th>'
            f'<th>{best_head}</th><th>Watch for</th><th class="r">{th_text("Snow depth")}</th></tr>')
    # The country word only when the visible rows are from more than one
    # country: otherwise the Show line already names it (skim spec 1.6).
    mixed = len({r.get("country") for r in rows}) > 1
    body = []
    for r in rows:
        km = None
        if near:
            if r.get("id") == resort.get("id"):
                km = ""
            else:
                dk = distance_km(resort, r)
                km = DASH if dk is None else f"{_round(dk if ctx.units == 'cm' else dk / 1.609344):,}"
        body.append(resort_row(r, ctx.units, resort.get("id"), ctx.href(resort=r.get("id", "")), vmax,
                               km=km, show_country=mixed))
    cap = f"All figures are for {BAND_WORD.get(hero_elevation(resort), 'base')}."
    return ('<div class="page sec"><h2 class="h40">Compare resorts</h2>'
            f'<p class="body words gap0">Show: {DOT.join(show)}</p>'
            f'<p class="body words">Sort by: {DOT.join(sorts)}</p>'
            f'<p class="caption gap0">{cap}</p>'
            f'<div class="tablewrap gap0"><table class="t every-t">{head}{"".join(body)}</table></div></div>')


# --------------------------------------------------------------------------
# Level 10: snow stations (only when one exists)
# --------------------------------------------------------------------------

def _cells(vals):
    return "".join(f'<td class="r">{DASH if v is None else v}</td>' for v in vals)


def stations_section(resort, ctx):
    stations = resort.get("stations")
    stations = [s for s in stations if isinstance(s, dict)] if isinstance(stations, list) else []
    if not stations:
        return ""
    su = "″" if ctx.units == "in" else f"{THIN}cm"
    du = "mi" if ctx.units == "in" else "km"
    # Headers name the measure only; the unit sits in each cell.
    head = (f'<tr><th>Station</th><th class="r">{du}</th><th class="r">Height</th>'
            f'<th class="r">Snow depth</th><th class="r">24-hour depth change</th>'
            f'<th>Snow depth, last 15 days</th><th>Last reading</th></tr>')
    srows = []
    for s in stations:
        dk = s.get("distance_km")
        dist = DASH if not is_num(dk) else f"{_round(dk / 1.609344 if ctx.units == 'in' else dk):,}"
        el = elev_str(s.get("elevation_m"), ctx.units)
        el = DASH if el is None else f"{el}{THIN}{ctx.elev_unit}"
        dep = snow_str(s.get("depth_cm"), ctx.units)
        dep = DASH if dep is None else f"{dep}{su}"  # measured: bare, with its unit
        # 24 h only: the 15-day line beside it shows the longer slope (skim spec 1.5)
        chs = signed_str(s.get("change_24h_cm"), ctx.units)
        ch = [None if chs is None else f"{chs}{su}"]
        raw = s.get("series") if isinstance(s.get("series"), list) else []
        # a non-dict entry keeps its slot as a gap, so the trend's days do not shift
        series = [x.get("depth_cm") if isinstance(x, dict) and is_num(x.get("depth_cm")) else None
                  for x in raw]
        series_disp = [None if v is None else (v / 2.54 if ctx.units == "in" else v) for v in series]
        sp = spark_line(series_disp)
        spark_cell = sp if sp else DASH
        od = s.get("observed_date")
        odd = local_date(od) if od else None
        reading = _date_word(resort, odd) if odd else DASH
        srows.append(
            f'<tr><td class="first">{escape(s.get("name") or "?")}</td>'
            f'<td class="r">{dist}</td><td class="r">{el}</td><td class="r">{dep}</td>'
            f'{_cells(ch)}<td>{spark_cell}</td><td>{escape(reading)}</td></tr>'
        )
    out = ['<div class="label">Measuring stations</div>',
           f'<div class="tablewrap gap0"><table class="t st-t">{head}{"".join(srows)}</table></div>']
    notes = [n for n in (g(resort, "quality", "notes") or []) if isinstance(n, str)]
    if notes and ctx.debug:
        out.append('<p class="caption sentence gap1">' + " ".join(escape(n.rstrip(".") + ".") for n in notes) + "</p>")
    return '<div class="page sec">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------
# Level 11: footer
# --------------------------------------------------------------------------

# Short names for attribution: the provider stays named (open-licence terms),
# the licence words move to one Licences sentence at the end.
SOURCE_SHORT = {"open-meteo": "Open-Meteo", "snotel": "SNOTEL", "geosphere": "GeoSphere", "slf": "SLF",
                "jma": "Japan Meteorological Agency (AMeDAS)", "meteo-france": "Météo-France",
                "meteocat": "Servei Meteorològic de Catalunya (XEMA)",
                "tirol": "Tirol's weather, avalanche and river services",
                "aemet": "AEMET OpenData", "tenki": "tenki.jp", "onthesnow": "OnTheSnow"}
SOURCE_LICENCE = {"meteo-france": "Licence Ouverte 2.0", "tirol": "CC BY 4.0", "aemet": "CC BY 4.0"}
KIND_WORDS = {"model": "Forecast", "measured": "Snow depth", "resort_report": "Resort reports"}


def footer(data, ctx=None):
    debug = bool(getattr(ctx, "debug", False))
    groups = {}
    licences = []
    for s in g(data, "sources", "list") or []:
        name = escape(SOURCE_SHORT.get(s.get("id"), s.get("name") or "?"))
        kind = KIND_WORDS.get(s.get("kind"))
        if kind is None:
            continue
        lic = SOURCE_LICENCE.get(s.get("id"))
        if lic and lic not in licences:
            licences.append(lic)
        url = s.get("url")
        nm = (f'<a href="{escape(url)}" target="_blank" rel="noopener">{name}</a>'
              if isinstance(url, str) and url.startswith("https://") else name)
        groups.setdefault(kind, []).append(nm)
    # Sources sit one click away in a closed disclosure (skim spec 1.6); the
    # "reported at N of M resorts" count is gone: the "—" cells show it.
    out = []
    if groups:
        body = " ".join(f"{k}: {', '.join(v)}." for k, v in groups.items())
        if licences:
            body += f" Licences: {'; '.join(licences)}."
        out.append('<details class="ext src"><summary class="caption"><span class="u">Sources</span></summary>'
                   f'<p class="caption sentence">{body}</p></details>')
    if debug:
        run = data.get("run") or {}
        dbg = "".join(f'<p class="caption">{escape(str(k))}: {escape(str(v))}</p>' for k, v in run.items())
        out.append(f'<div class="gap1">{dbg}</div>')
    return '<div class="page sec">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------
# Names app.py calls
# --------------------------------------------------------------------------

def screen1_main(resort, ctx, updated, nearby):
    """Levels 1 to 6 (app.py wraps this in the page block)."""
    html = first_screen(resort, nearby, ctx, updated)
    return html[len('<div class="page">'):-len("</div>")]


def daily_table(resort, ctx):
    """Levels 7 and 8: snow by day and the collapsed days 8-15."""
    return days_section(resort, ctx)


# --------------------------------------------------------------------------
# Layout CSS for the resort page
# --------------------------------------------------------------------------

CSS = """
.topbar{ padding-top:var(--s1); display:flex; flex-wrap:wrap; column-gap:2em; row-gap:0; align-items:baseline; }
.topbar a{ margin-right:0; }
.topbar .units a{ margin:0; }
.stApp .topbar a, .stApp .topbar a:visited{ text-decoration:none !important; }
.stApp .topbar a.nav-cur{ color:var(--accent) !important; text-decoration:underline !important; }
.stApp .topbar a.nav-cur.nav-ink{ color:var(--ink) !important; }
.stApp .topbar a.nav-hot{ color:var(--accent) !important; }
.topbar + *{ margin-top:var(--s1); }
.sec{ padding-top:var(--s4); }
.verdict, .stApp .verdict, [data-testid="stMarkdownContainer"] p.verdict{ font-size:40px; line-height:1.1;
  letter-spacing:-.01em; max-width:20em; text-wrap:balance; }
.herorow{ display:flex; flex-wrap:wrap; align-items:baseline; column-gap:var(--l); }
.herolab{ white-space:nowrap; }
/* band rows (skim spec 1.1): label, ink bar, numeral, at most two words */
.bands{ display:grid; grid-template-columns:fit-content(12em) 10em 5.5em auto; column-gap:.5em; align-items:baseline; }
.bands > *{ line-height:var(--l); min-height:var(--l); white-space:nowrap; }
.bands .bl{ overflow:hidden; }
.bands .ln{ display:none; }
.bands .n{ text-align:right; font-variant-numeric:lining-nums tabular-nums; font-weight:500; }
.bands .bw{ font-size:14px; }
.bands .bar{ width:calc(var(--v)*10em); }
/* ground line: depth, then the last three days as bars and a total */
.ground .gi{ display:inline-block; white-space:nowrap; margin-right:2em; }
.ground .gn, details.ext .en{ font-variant-numeric:lining-nums tabular-nums; font-weight:500; }
.ground .gu, details.ext .eu{ font-size:14px; }
svg.sb{ vertical-align:baseline; overflow:visible; }
svg.sb rect{ fill:var(--ink); } svg.sb rect.is-best{ fill:var(--accent); }
/* snow-line marks (skim spec 1.2) */
svg.sl{ display:inline-block; vertical-align:calc(.41em - 6px); overflow:visible; }
svg.sl line{ stroke:var(--ink); stroke-width:1; vector-effect:non-scaling-stroke; fill:none; }
.rows .slc{ text-align:left; white-space:nowrap; }
.rows .slc .sld, .rows .slc .slt{ font-size:14px; }
.rows .slc .sld{ padding-left:6px; }
.rows .hd.slh{ line-height:20px; }
.rows .hd .slax{ display:flex; justify-content:space-between; width:120px; box-sizing:border-box; padding:0 6px;
  font-size:14px; font-variant-caps:normal; letter-spacing:0; font-variant-numeric:lining-nums tabular-nums; }
details.ext summary{ cursor:pointer; list-style:none; padding-block:9px; display:inline-block; }
details.ext summary .u{ text-decoration:underline; text-decoration-thickness:1px; text-underline-offset:.2em; }
details.ext summary::-webkit-details-marker{ display:none; }
details.ext summary:focus-visible{ outline:2px solid var(--accent); outline-offset:3px; }
.rows .hd{ font-size:12px; font-variant-caps:all-small-caps; letter-spacing:.12em; white-space:nowrap; }
.rows .hd.r, .rows .n2{ text-align:right; }
.rows .n2{ font-variant-numeric:lining-nums tabular-nums; white-space:nowrap; font-size:14px; }
.rows .n2.tu{ text-align:left; }
.rows .f{ font-size:14px; white-space:nowrap; }
.rows .hd.f{ font-size:12px; }
.rows .tot{ grid-column:1 / 3; }
.thn{ display:none; }
/* inline, not inline-block: the padding still enlarges the tap area, but a
   wrapped sort line keeps normal line height instead of a 46px gap */
.stApp .words a{ display:inline; }
.stApp .rows.days.wk{ grid-template-columns:5.2em 10em 5.5em minmax(7em,max-content) minmax(8.5em,max-content); }
@media (max-width:700px){
  .thw{ display:none; }
  .thn{ display:inline; }
  .verdict, .stApp .verdict, [data-testid="stMarkdownContainer"] p.verdict{ font-size:34px; }
  .bands{ grid-template-columns:fit-content(9em) 4em 4.5em auto; }
  .bands .lw{ display:none; }
  .bands .ln{ display:inline; }
  .bands .bar{ width:calc(var(--v)*4em); }
  svg.sb{ height:1em; width:auto; }
  .ground .gi{ margin-right:1.2em; }
  /* Watch for stays on the day's row, never under the bar. The bar column is
     fixed so bar length stays proportional (a shrinking column clipped long
     bars to the same length); two Watch words that don't fit wrap inside
     their own column. */
  .stApp .rows.days.wk{ grid-template-columns:3.3em 2.4em 3.6em minmax(0,1fr) 120px; column-gap:.35em; }
  .stApp .rows.days.wk > span:first-child, .stApp .rows.days.wk .tot{ white-space:nowrap; }
  .stApp .rows.days.wk > *:nth-child(5n+1){ white-space:nowrap; }
  .stApp .rows.days.wk .bar{ width:calc(var(--v)*2.4em); max-width:none; }
  .stApp .rows.days.wk .bc{ overflow:visible; }
  .stApp .rows.days.wk .f:not(.hd){ font-size:14px; white-space:normal; line-height:16px; }
  /* stations at 390: headers wrap, names wrap, the trend line narrows, so all five columns fit */
  [data-testid="stMarkdownContainer"] table.st-t th{ white-space:normal; line-height:1.2; vertical-align:bottom; }
  [data-testid="stMarkdownContainer"] table.st-t td, [data-testid="stMarkdownContainer"] table.st-t th{ padding-right:.6em; }
  [data-testid="stMarkdownContainer"] table.st-t td.first{ white-space:normal; font-size:17px; padding-right:.6em; line-height:1.2; }
  table.st-t .spark{ width:3.5em; }
  .stApp .rows.days.wk .hd.r, .stApp .rows.days.wk .hd.f{ white-space:normal; line-height:1.2; }
  .topbar{ column-gap:1.2em; }
}
"""
