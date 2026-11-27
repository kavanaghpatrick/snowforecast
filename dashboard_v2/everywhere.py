"""The All resorts page: every resort in one table, sortable by any column.

?view=all&sort=<col>&dir=asc|desc&units=cm|in
Columns: name, next3, week, best, flags (Watch for), depth, ext (Days 8-15).
Default: sort=next3, dir=desc. Missing values always sort last.
"""

from __future__ import annotations

from html import escape as _html_escape


def escape(s, quote=True):
    """html.escape, with every run of whitespace folded to one space.

    A newline pair inside a data string would end Streamlit's raw HTML block and
    turn the rest of the string into live markdown (remote images, links).
    """
    return _html_escape(" ".join(str(s).split()), quote)
from urllib.parse import urlencode

from data import (best_display_value, best_ski_index, extended_total, g, is_num, name_key, next3,
                  recent_fallen, week_total)
from render import (DOT, THIN, prose_num, resort_row, scale_max, snow_str, snow_val, th_text,
                    topbar, watch_resort)

# key, header, numeric (right-aligned), first-click direction
COLS = (
    ("name", "Resort", False, "asc"),
    ("next3", "Snow next 3 days", True, "desc"),
    ("week", "Snow next 7 days", True, "desc"),
    ("last3", "Last 3 days", True, "desc"),
    ("best", "Models' best day", False, "desc"),
    ("flags", "Watch for", False, "desc"),
    ("depth", "Snow depth", True, "desc"),
    ("ext", "Days 8–15", True, "desc"),
)
SORT_KEYS = tuple(c[0] for c in COLS)
DIRS = ("asc", "desc")
SORT_WORDS = (("next3", "Snow next 3 days"), ("week", "Snow next 7 days"), ("last3", "Last 3 days"),
              ("best", "Models' best day"), ("depth", "Snow depth"), ("name", "A–Z"))


def _href(**p):
    return "?" + urlencode({k: v for k, v in p.items() if v is not None})


def nav(current, resort_id, units, updated, debug=False, resort=None, unit_params=None, powder_hot=False):
    """Level 0, the top bar on every page:
    Resort search · All resorts · Powder · cm / in · Updated 2 h ago.

    unit_params are the page's own params, so switching units keeps the page as it is.
    powder_hot puts "Powder" in the accent (some resort clears a powder threshold).
    """
    extra = {"debug": "1"} if debug else {}
    keep = dict(unit_params or {"resort": resort_id})
    keep.pop("units", None)
    bar_html = topbar(
        current,
        _href(resort=resort_id, units=units, **extra),
        _href(view="all", resort=resort_id, units=units, **extra),
        _href(**keep, units="cm", **extra),
        _href(**keep, units="in", **extra),
        units, updated, resort or {},
        powder_href=_href(view="powder", resort=resort_id, units=units, **extra),
        powder_hot=powder_hot,
    )
    return '<div class="page">' + bar_html + "</div>"


def _depth(r):
    v = g(r, "depth", "value_cm")
    return v if is_num(v) else None


WATCH_ORDER = ("rain", "wet snow", "strong wind", "sunny")


def _value(r, col, units="cm"):
    if col == "next3":
        return next3(r)
    if col == "week":
        return week_total(r)
    if col == "last3":
        return recent_fallen(r)[0]
    if col == "ext":
        return extended_total(r)
    if col == "depth":
        return _depth(r)
    if col == "best":
        # The exact number the Best ski day cell displays, so the sorted
        # order is monotonic with the visible column.
        return best_display_value(r, units)
    if col == "flags":
        f = watch_resort(r)
        return sum(2 ** (3 - WATCH_ORDER.index(x)) for x in f if x in WATCH_ORDER) or None
    return name_key(r.get("name")) or None


def sort_all(resorts, col, direction, units="cm"):
    """Sort every resort by one column. Missing values go last in either direction;
    ties break A-Z on name."""
    if col == "best":
        # Chronological (ux spec 5a): a planning sort. Soonest best day first,
        # amount descending inside a day, reason-word rows last. The direction
        # toggle flips the day order only, never back to amount-first.
        present = [r for r in resorts if best_ski_index(r) is not None]
        missing = [r for r in resorts if best_ski_index(r) is None]
        flip = -1 if direction == "asc" else 1
        present.sort(key=lambda r: (flip * best_ski_index(r),
                                    -(best_display_value(r, units) or 0),
                                    name_key(r.get("name"))))
        missing.sort(key=lambda r: name_key(r.get("name")))
        return present + missing
    present = [r for r in resorts if _value(r, col, units) is not None]
    missing = [r for r in resorts if _value(r, col, units) is None]
    present.sort(key=lambda r: name_key(r.get("name")))
    present.sort(key=lambda r: _value(r, col, units), reverse=(direction == "desc"))
    missing.sort(key=lambda r: name_key(r.get("name")))
    return present + missing


# Mountain-range filters (user directive, 27 Nov): Japan / Alps / Pyrenees / US.
# Membership is by country, except the Pyrenees, which are a region (Spain's
# Sierra Nevada honestly matches no range and shows only under All).
RANGES = (("all", "All"), ("japan", "Japan"), ("alps", "Alps"),
          ("pyrenees", "Pyrenees"), ("us", "US"))
RANGE_KEYS = tuple(k for k, _ in RANGES)
_ALPS_COUNTRIES = {"Austria", "Switzerland", "Italy", "Germany", "Slovenia"}


def in_range(resort, key):
    c, reg = resort.get("country"), resort.get("region") or ""
    if key == "japan":
        return c == "Japan"
    if key == "us":
        return c == "USA"
    if key == "pyrenees":
        return "Pyrenees" in reg
    if key == "alps":
        return c in _ALPS_COUNTRIES or (c in ("France", "Spain") and "Alps" in reg)
    return True


def leaders(resorts, units, resort_id, extra):
    """'Most snowfall next 3 days: Myoko Kogen, 36 cm.' The top resort by 3-day
    snow for the current range filter; rows 2 and 3 are right under it in the
    table (skim spec 1.6). Below 5 cm (2 in) nobody is crowned; all-null
    renders nothing."""
    top = sort_all(resorts, "next3", "desc")[:1]
    top = [r for r in top if next3(r) is not None]
    if not top:
        return ""
    r = top[0]
    n = snow_val(next3(r), units)
    if n == 0:
        return "No snowfall next 3 days."
    if n < (2 if units == "in" else 5):
        return "Little snowfall next 3 days."
    unit = "″" if units == "in" else f"{THIN}cm"
    href = _href(resort=r.get("id", ""), units=units, **extra)
    name = f'<a class="plain" href="{escape(href)}" target="_self">{escape(r.get("name") or "?")}</a>'
    return f"Most snowfall next 3 days: {name}, ~{prose_num(snow_str(next3(r), units))}{unit}."


def page(resorts, col, direction, units, resort_id, debug=False, rng="all"):
    if rng != "all":
        resorts = [r for r in resorts if in_range(r, rng)]
    rows = sort_all(resorts, col, direction, units)
    extra = {"debug": "1"} if debug else {}
    u = "in" if units == "in" else "cm"

    head = []
    for key, text, numeric, first in COLS:
        cur = key == col
        nxt = ("asc" if direction == "desc" else "desc") if cur else first
        href = _href(view="all", sort=key, dir=nxt, resort=resort_id, units=units, **({"range": rng} if rng != "all" else {}), **extra)
        arrow = ("&#8239;↓" if direction == "desc" else "&#8239;↑") if cur else ""
        cls = ["r"] if numeric else []
        if key == "week":
            cls.append("wk7")
        if cur:
            cls.append("cur")
        c = f' class="{" ".join(cls)}"' if cls else ""
        aria = f' aria-sort="{"descending" if direction == "desc" else "ascending"}"' if cur else ""
        head.append(f'<th{c}{aria}><a href="{escape(href)}" target="_self">{th_text(text)}{arrow}</a></th>')
        if key == "name":
            head.append('<th class="barcol"></th>')

    vmax = scale_max([snow_val(next3(r), units) for r in rows], units)
    body = [resort_row(r, units, resort_id, _href(resort=r.get("id", ""), units=units, **extra), vmax,
                       show_country=True, show_ext=True) for r in rows]

    shows = []
    for key, text in RANGES:
        href = _href(view="all", sort=col, dir=direction, resort=resort_id, units=units,
                     **({"range": key} if key != "all" else {}), **extra)
        cls = ' class="is-current"' if key == rng else ""
        shows.append(f'<a href="{escape(href)}" target="_self"{cls}>{text}</a>')

    sorts = []
    for key, text in SORT_WORDS:
        first = next(c[3] for c in COLS if c[0] == key)
        href = _href(view="all", sort=key, dir=first, resort=resort_id, units=units, **({"range": rng} if rng != "all" else {}), **extra)
        cls = ' class="is-current"' if key == col else ""
        sorts.append(f'<a href="{escape(href)}" target="_self"{cls}>{text}</a>')

    # Every number speaks for mid-mountain (data.hero_elevation; docs/snowline_decision.md).
    cap = "All figures are for mid-mountain."
    lead = leaders(resorts, units, resort_id, extra)
    lead_html = f'<p class="leaders prose gap1">{lead}</p>' if lead else ""
    return (
        '<div class="page every">'
        '<h1 class="title">All resorts</h1>'
        f'<p class="caption gap0">{escape(cap)}</p>'
        f'{lead_html}'
        f'<p class="body words gap1">Show: {DOT.join(shows)}</p>'
        f'<p class="body words gap0">Sort by: {DOT.join(sorts)}</p>'
        f'<div class="tablewrap gap0"><table class="t every-t"><tr>{"".join(head)}</tr>{"".join(body)}</table></div></div>'
    )


CSS = """
.every-t th a, .every-t th a:visited{ text-decoration:none !important; }
.stApp [data-testid="stMarkdownContainer"] table.t th.cur, .stApp [data-testid="stMarkdownContainer"] table.t th.cur a{ color:var(--accent) !important; }
[data-testid="stMarkdownContainer"] table.every-t td.first{ font-size:16px; padding-right:1.5em; }
[data-testid="stMarkdownContainer"] table.every-t td.first .label{ padding-left:.3em; }
[data-testid="stMarkdownContainer"] table.every-t td, [data-testid="stMarkdownContainer"] table.every-t th{ padding-right:1.4em; }
[data-testid="stMarkdownContainer"] table.every-t td.barcol, [data-testid="stMarkdownContainer"] table.every-t th.barcol{ width:6em; padding-right:.6em; }
table.every-t td.barcol .bar{ width:calc(var(--v)*6em); }
table.every-t td.r{ font-variant-numeric:lining-nums tabular-nums; }
.every{ padding-bottom:0; }
.stApp h1.title, [data-testid="stMarkdownContainer"] h1.title{ font-size:61px; line-height:1.05; letter-spacing:-.01em; font-weight:400;
  margin:0; padding:0; color:var(--ink); }
.leaders, .stApp .leaders{ font-size:40px; line-height:1.1; letter-spacing:-.01em; max-width:24em; text-wrap:balance; }
[data-testid="stMarkdownContainer"] table.every-t td.wf{ white-space:nowrap; }
@media (max-width:700px){
  /* phone: the bar stays (4em, right after the name); the 7-day column goes instead */
  [data-testid="stMarkdownContainer"] table.every-t td.barcol, [data-testid="stMarkdownContainer"] table.every-t th.barcol{ width:4em; min-width:4em; padding-right:.4em; }
  table.every-t td.barcol .bar{ width:calc(var(--v)*4em); }
  [data-testid="stMarkdownContainer"] table.every-t .wk7{ display:none; }
  [data-testid="stMarkdownContainer"] table.every-t td, [data-testid="stMarkdownContainer"] table.every-t th{ padding-right:.8em; }
  /* phone: headers wrap and the country label drops, so Best ski day stays on screen */
  [data-testid="stMarkdownContainer"] table.every-t th{ white-space:normal; line-height:1.2; vertical-align:bottom; min-width:3.5em; }
  [data-testid="stMarkdownContainer"] table.every-t th:first-child{ min-width:0; }
  [data-testid="stMarkdownContainer"] table.every-t td.first .label{ display:none; }
  [data-testid="stMarkdownContainer"] table.every-t td.first{ font-size:15px; white-space:normal; padding-right:.6em; }
  .stApp h1.title, [data-testid="stMarkdownContainer"] h1.title{ font-size:44px; }
  .leaders, .stApp .leaders{ font-size:34px; }
}
"""
