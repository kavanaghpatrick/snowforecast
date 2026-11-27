"""The standing invariants (design brief v2, ux hierarchy spec, skim spec) as tests.

Run over every resort in the real data/forecasts_v2.json, in both units, by
calling the real render functions. Snapshot fixtures pin the real records that
the base-gate and measured-refutation rules need, so those tests stay
meaningful after the next fetcher run.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from conftest import (D, E, P, R, UNITS, inline_text, load_fixture, resort_ids, usable_resorts,
                      visible)

IDS = resort_ids()
WEEKDAY3 = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _by_id(resorts):
    return {r["id"]: r for r in resorts}


@pytest.fixture(scope="module")
def by_id(resorts):
    return _by_id(resorts)


def ctx(r, units, scope="nearby", sort="next3"):
    return R.Ctx(r["id"], scope, sort, units)


# ----------------------------------------------------------------------------
# Band sums = headline
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("rid", IDS)
def test_band_rows_equal_band_totals_and_hero(by_id, rid):
    """Each band row's numeral is that band's 3-day total, and the hero number
    is the hero band's total: one number per band, everywhere on the page."""
    r = by_id[rid]
    for units in UNITS:
        c = ctx(r, units)
        rows = R.band_rows(r, c)
        bands = [b for b, *_ in R.strip_bands(r) if R.elev_str(_[1], units) is not None]
        assert len(rows) == len(bands)
        for (band, *_rest), (_w, _n, v, _sp) in zip([b for b in R.strip_bands(r)
                                                     if R.elev_str(b[2], units) is not None], rows):
            assert v == R.snow_val(D.next3_band(r, band), units), (rid, units, band)
        hero = R.snow_val(D.next3(r), units)
        hb = R.hero_block(r, c)
        if hero is None:
            assert hb == ""
            continue
        m = re.search(r'<div class="hero">~?([\d,]+)<', hb)
        assert m and int(m.group(1).replace(",", "")) == hero
        assert hero == R.snow_val(D.next3_band(r, D.hero_elevation(r)), units)


@pytest.mark.parametrize("rid", IDS)
def test_strip_text_amount_is_band_total(by_id, rid):
    r = by_id[rid]
    for units in UNITS:
        c = ctx(r, units)
        for band, *_ in R.strip_bands(r):
            t = R.strip_text(r, band, c)
            if t is None:
                continue
            tt = inline_text(t)
            m = re.search(r"([\d,]+)(?: (?:cm|in)|″) snow", tt)
            got = int(m.group(1).replace(",", "")) if m else 0
            want = R.snow_val(D.next3_band(r, band), units) or 0
            assert got == want, (rid, units, band, tt)


def _day_table_numbers(sec):
    """(7 day-row numerals, 7-day total, ext row numerals, ext summary) from the rendered table."""
    week_html, _, ext_html = sec.partition('<details class="ext')
    rows = re.findall(r'<span class="n(?: accent)?">([^<]*)</span>', week_html)
    tot = re.search(r'<span class="tot">Snow next 7 days</span><span class="n">([^<]*)</span>', week_html)
    erows = re.findall(r'<span class="n">([^<]*)</span>', ext_html)
    esum = re.search(r'<span class="en">([^<]*)</span>', ext_html)
    return rows, tot.group(1) if tot else None, erows, esum.group(1) if esum else None


def _n(s):
    """The integer inside a cell like '~21&#8239;cm', '8″' or '—', else None."""
    if s in (None, R.DASH):
        return None
    digits = re.sub(r"[^\d]", "", inline_text(s))
    return int(digits) if digits else None


@pytest.mark.parametrize("rid", IDS)
def test_daily_rows_sum_to_shown_totals(by_id, rid):
    """Displayed day rows add up to the displayed 7-day total and Days 8-15 summary."""
    r = by_id[rid]
    for units in UNITS:
        sec = R.days_section(r, ctx(r, units))
        if not sec:
            continue
        rows, tot, erows, esum = _day_table_numbers(sec)
        # the total row's own numeral is captured by the row regex too; drop it
        rows = rows[:7]
        vals = [_n(x) for x in rows]
        if all(v is not None for v in vals) and _n(tot) is not None:
            assert sum(vals) == _n(tot), (rid, units, rows, tot)
        evals = [_n(x) for x in erows]
        if esum is not None and evals and all(v is not None for v in evals):
            assert sum(evals) == _n(esum), (rid, units, erows, esum)


# ----------------------------------------------------------------------------
# Phase words only on matching phases
# ----------------------------------------------------------------------------

def _phases(r, band):
    return {s.get("phase") for s in D.g(r, "next_72h", "by_day", band) or [] if isinstance(s, dict)}


@pytest.mark.parametrize("rid", IDS)
def test_band_words_match_phases(by_id, rid):
    r = by_id[rid]
    for units in UNITS:
        c = ctx(r, units)
        for band, *_ in R.strip_bands(r):
            ph = _phases(r, band)
            sp = R._band_spoiler(r, band)
            if sp.startswith("rain"):
                assert "rain" in ph, (rid, band, sp, ph)
            if sp.startswith("wet"):
                assert "mixed" in ph, (rid, band, sp, ph)
            t = R.strip_text(r, band, c)
            if t is None:
                continue
            if re.search(r"\brain\b", t):
                assert "rain" in ph, (rid, band, t, ph)
            if "wet snow" in t:
                assert "mixed" in ph, (rid, band, t, ph)


@pytest.mark.parametrize("rid", IDS)
def test_watch_words_need_wet_weather(by_id, rid):
    """'rain' only when rain falls (rain_risk and >= 1 mm) at the flag band;
    'wet snow' only on a mixed-phase day with snow or rain falling."""
    r = by_id[rid]
    band = "mid" if r.get("mid_elevation_m") is not None else "summit"
    for i in range(7):
        d = D.day_by_index(r, i)
        if d is None:
            continue
        words = R.watch_day(r, d)
        assert len(words) <= 2
        rain = D.g(d, band, "rain_mm")
        sn = D.g(d, band, "snowfall_cm")
        if "rain" in words:
            assert D.g(d, "derived", "rain_risk", band) is True and D.is_num(rain) and rain >= 1
        if "wet snow" in words:
            assert D.g(d, "derived", "mixed_risk", band) is True
            assert (D.is_num(sn) and sn >= 0.5) or (D.is_num(rain) and rain >= 1)
    t = D.rain_threat_72h(r)
    w = R.watch_resort(r)
    if "rain" in w:
        assert t is not None and t["kind"] == "threat" and t["phase"] == "rain"
    if "wet snow" in w:
        assert t is not None and t["kind"] == "threat" and t["phase"] == "mixed"


# ----------------------------------------------------------------------------
# Merged bands
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("rid", IDS)
def test_merged_bands_render_once(by_id, rid):
    r = by_id[rid]
    qb = D.g(r, "quality", "bands") or {}
    names = [name for _b, name, _lo, _hi in R.strip_bands(r)]
    has_mid = r.get("mid_elevation_m") is not None and r.get("summit_elevation_m") is not None
    if qb.get("mid_summit_same_72h") is True and has_mid:
        assert "Mid" not in names and "Summit" not in names, (rid, names)
        if qb.get("base_mid_same_72h") is True and r.get("elevation_m") is not None:
            assert names == ["Base–summit"]
        else:
            assert names[0] == "Mid–summit"
    for units in UNITS:
        labels = [w for w, *_ in R.band_rows(r, ctx(r, units))]
        assert len(labels) == len(set(labels))


# ----------------------------------------------------------------------------
# Sorts: best is chronological; amount sorts are monotonic in both units
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("direction", ("desc", "asc"))
def test_all_page_best_sort_is_chronological(resorts, units, direction):
    rows = E.sort_all(resorts, "best", direction, units)
    idx = [D.best_ski_index(r) for r in rows]
    present = [i for i in idx if i is not None]
    assert idx[:len(present)] == present, "resorts without a best day must sort last"
    assert present == sorted(present, reverse=(direction == "asc"))
    for i in set(present):
        amts = [D.best_display_value(r, units) for r in rows if D.best_ski_index(r) == i]
        assert amts == sorted(amts, reverse=True), (i, amts)


@pytest.mark.parametrize("units", UNITS)
def test_compare_best_sort_is_chronological(resorts, units):
    rows = D.sort_resorts(resorts, "best", units)
    idx = [D.best_ski_index(r) for r in rows]
    present = [i for i in idx if i is not None]
    assert idx[:len(present)] == present
    assert present == sorted(present)
    for i in set(present):
        amts = [D.best_display_value(r, units) for r in rows if D.best_ski_index(r) == i]
        assert amts == sorted(amts, reverse=True)


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("direction", ("desc", "asc"))
@pytest.mark.parametrize("col", ("next3", "week", "last3", "depth", "ext"))
def test_all_page_amount_sorts_monotonic(resorts, units, direction, col):
    rows = E.sort_all(resorts, col, direction, units)
    vals = [E._value(r, col, units) for r in rows]
    present = [v for v in vals if v is not None]
    assert vals[:len(present)] == present, "missing values must sort last"
    assert present == sorted(present, reverse=(direction == "desc"))
    shown = [R.snow_val(v, units) for v in present]
    assert shown == sorted(shown, reverse=(direction == "desc")), "displayed column not monotonic"


@pytest.mark.parametrize("units", UNITS)
@pytest.mark.parametrize("col", ("next3", "week", "depth"))
def test_compare_amount_sorts_monotonic(resorts, units, col):
    fn = {"next3": D.next3, "week": D.week_total,
          "depth": lambda r: D.g(r, "depth", "value_cm") if D.is_num(D.g(r, "depth", "value_cm")) else None}[col]
    rows = D.sort_resorts(resorts, col, units)
    vals = [fn(r) for r in rows]
    present = [v for v in vals if v is not None]
    assert vals[:len(present)] == present
    shown = [R.snow_val(v, units) for v in present]
    assert shown == sorted(shown, reverse=True)


def test_name_sort_groups_countries_in_order(resorts):
    rows = D.sort_resorts(resorts, "name")
    order = [D.COUNTRY_ORDER.index(r["country"]) if r.get("country") in D.COUNTRY_ORDER else 99 for r in rows]
    assert order == sorted(order)


# ----------------------------------------------------------------------------
# No annotated cells
# ----------------------------------------------------------------------------

BEST_CELL = re.compile(r"^(?:~?[\d,]+(?: cm|″)(?: today| yesterday| on (?:" + "|".join(WEEKDAY3) + r"))"
                       r"|No ski day, (?:wind|rain)|" + R.DASH + r")$")


@pytest.mark.parametrize("rid", IDS)
def test_best_cell_is_day_and_number_or_reason(by_id, rid):
    r = by_id[rid]
    for units in UNITS:
        cell = inline_text(R.best_cell(r, units))
        assert BEST_CELL.match(cell), (rid, units, cell)
        v = D.best_display_value(r, units)
        if D.best_ski_index(r) is not None and v is not None:
            m = re.match(r"~?([\d,]+)", cell)
            assert m and int(m.group(1).replace(",", "")) == v, (rid, units, cell, v)


# "~" is the page's one mark for a model figure and units sit in each cell
# (copy audit glossary), so neither is an annotation; provenance words,
# parentheses and footnote marks still are.
ANNOTATION = re.compile(r"[()*]|\b(model|measured|reported|est\.?|estimate|approx)\b", re.I)


@pytest.mark.parametrize("units", UNITS)
def test_table_cells_carry_no_annotations(resorts, units):
    """Never 12: a table cell is a bare number, day + number, '—' or a short
    word; never a caption fragment, a symbol or a provenance tag."""
    pages = [E.page(resorts, "next3", "desc", units, resorts[0]["id"])]
    for r in resorts[:: max(1, len(resorts) // 12)]:
        c = ctx(r, units)
        pages.append(R.compare_section(r, D.sort_resorts(D.scope_resorts(resorts, r, "nearby"), "next3", units),
                                       c, len(resorts), "?view=all"))
    for h in pages:
        for cell in re.findall(r"<td[^>]*>(.*?)</td>", h, flags=re.S):
            t = visible(cell)
            if 'class="first"' in cell or "<a " in cell:
                continue  # resort name cell
            assert not ANNOTATION.search(t), (units, t)


# ----------------------------------------------------------------------------
# Verdicts: at most 8 words per sentence; a sentence with an amount says "snow"
# ----------------------------------------------------------------------------

def words(sentence):
    """Words as the spec counts them: a number with its unit is one word."""
    s = sentence.strip().rstrip(".")
    s = re.sub(r"~?\d[\d,]*(\.\d+)?\s?(cm|in|mm|m|ft|h|″|%)?(?=\W|$)", "N", s)
    return [w for w in re.split(r"[\s;:,]+", s) if re.search(r"\w", w)]


def sentences(text):
    return [s for s in re.split(r"(?<=\.)\s+", text.strip()) if s]


@pytest.mark.parametrize("rid", IDS)
def test_verdict_short_and_names_snow(by_id, rid, fresh):
    r = by_id[rid]
    for units in UNITS:
        c = ctx(r, units)
        screen = R.first_screen(r, [], c, fresh)
        v = inline_text(re.search(r'<p class="verdict prose gap1">(.*?)</p>', screen, flags=re.S).group(1))
        assert v, rid
        for s in sentences(v):
            assert len(words(s)) <= 8, (rid, units, s, words(s))
            if re.search(r"\d\s?(cm|in|″)", s):
                assert "snow" in s.lower(), (rid, units, s)


def test_verdict_warns_when_file_is_stale(resorts):
    """Past 36 h the first sentence is the staleness warning, nothing else."""
    r = resorts[0]
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    v = R.verdict(r, ctx(r, "cm"), old)
    assert v.startswith("Warning: forecast is ") and v.endswith(" days old.")


# ----------------------------------------------------------------------------
# Snow-cover strings
# ----------------------------------------------------------------------------

COVER = re.compile(r"^(~100%|~0%|upper ~(?:[1-9]?5|[1-9]0)%)$")


@pytest.mark.parametrize("rid", IDS)
def test_snow_cover_strings_well_formed(by_id, rid):
    r = by_id[rid]
    for i in range(-1, 15):
        d = D.day_by_index(r, i)
        if d is None:
            continue
        cov = R.snow_cover_str(r, d)
        if cov is None:
            continue
        assert COVER.match(cov), (rid, i, cov)
        if cov.startswith("upper"):
            pct = int(cov[7:-1])
            assert 5 <= pct <= 95
        lo_hi = R.snow_line_range(d)
        base, summit = R._sl_axis(r)
        if cov == "~100%":
            assert lo_hi[1] <= base or (summit - lo_hi[1]) / (summit - base) >= 0.975
        if cov == "~0%":
            assert lo_hi[1] >= summit or (summit - lo_hi[1]) / (summit - base) < 0.025
    for units in UNITS:
        sec = R.days_section(r, ctx(r, units))
        for cell in re.findall(r'<span class="slc">(.*?)</span>(?=<span|</div>|$)', sec):
            t = inline_text(cell)
            if t in ("", R.DASH):
                continue
            assert COVER.match(t) or re.search(r"\d (m|ft)$|above summit|below base", t), (rid, t)


@pytest.mark.parametrize("units", UNITS)
def test_powder_page_snow_cover_cells(resorts, units):
    _m, _r, forecast = P.groups(resorts)
    for row in forecast:
        s = P._down_to_str(row["resort"], row["day"], units)
        assert s == R.DASH or COVER.match(s), s


# ----------------------------------------------------------------------------
# Base-aware Ski gate, including the measured-refutation rule
# ----------------------------------------------------------------------------

def _gate_cases():
    real = usable_resorts(load_fixture("snapshot_base_gate.json"))
    return real + usable_resorts(load_fixture("snapshot_whitefish.json"))


def _verdict_text(r, units, now):
    return inline_text(R.verdict(r, ctx(r, units), now))


@pytest.mark.parametrize("rid", IDS)
def test_ski_imperative_only_on_established_base(by_id, rid, fresh):
    r = by_id[rid]
    for units in UNITS:
        v = _verdict_text(r, units, fresh)
        if v.startswith("Ski "):
            assert R.has_base(r), (rid, v, r.get("depth"))


def test_base_gate_snapshots(fresh):
    """Recorded real records: a thin measured base (Niseko 5 cm), no depth at
    all (Astun), a deep resort-reported base (Alta) and a 22 cm measured base
    (Solden). Only the last two may say "Ski"; all four keep their best day."""
    rs = _by_id(usable_resorts(load_fixture("snapshot_base_gate.json")))
    for rid, go in (("niseko-united", False), ("astun", False), ("alta", True), ("solden", True)):
        r = rs[rid]
        assert R.has_base(r) is go
        assert D.best_ski_index(r) is not None, "depth never re-ranks or suppresses the best day"
        for units in UNITS:
            v = _verdict_text(r, units, fresh)
            assert v.startswith("Ski ") is go, (rid, v)
            if not go:
                assert "snow" in v.lower()
            cell = inline_text(R.best_cell(r, units))
            assert BEST_CELL.match(cell) and not cell.startswith((R.DASH, "No ski day")), cell


def test_measured_refutation_snapshot(fresh):
    """Whitefish (recorded): the model's best day is today on snow that fell
    yesterday, but the resort reported 0 cm. The measurement wins: no best day,
    no Ski or Fresh snow claim, and the cell gives no day."""
    r = usable_resorts(load_fixture("snapshot_whitefish.json"))[0]
    assert D.g(r, "week", "best_ski_day") == 0
    v_cm, src, _h = D.recent_fallen(r)
    assert src in ("reported", "measured") and v_cm < R.POWDER_MIN_CM
    assert D.best_ski_index(r) is None
    assert D.best_display_value(r, "cm") is None
    for units in UNITS:
        v = _verdict_text(r, units, fresh)
        assert not v.startswith(("Ski ", "Fresh snow", "New snow")), v
        cell = R.best_cell(r, units)
        assert cell == R.DASH or cell.startswith("No ski day"), cell


@pytest.mark.parametrize("rid", IDS)
def test_measured_refutation_rule_holds_everywhere(by_id, rid):
    """No resort keeps a today-best-day on fallen snow its own measurement refutes."""
    r = by_id[rid]
    i = D.best_ski_index(r)
    if i != 0:
        return
    own = D.day_snow(r, D.day_by_index(r, 0)) or 0
    prev = D.day_snow(r, D.day_by_index(r, -1)) or 0
    if prev >= 0.5 and own <= prev:
        v, src, _h = D.recent_fallen(r)
        if src in ("reported", "measured"):
            assert v >= R.POWDER_MIN_CM, (rid, v, src)


# ----------------------------------------------------------------------------
# Honest nulls and banned chrome on whole pages
# ----------------------------------------------------------------------------

# Strings the brief, the copy audit and the user's rulings ban from the visible
# page. Each entry: (regex, why).
BANNED = [
    (r"Made with Streamlit", "Never 9: Streamlit chrome"),
    (r"no report\b", "Never 12: symbol legend"),
    (r"\*\s?estimate|\* ?est\b", "Never 12: symbol legend"),
    (r"\d\*|\*\d", "Never 12: footnote marks on numbers"),
    (r"\bsnow down\b", "user, 27 Nov: invented jargon"),
    (r"\blvl\b|Δ|\bSWE\b", "copy audit: unexplained shorthand"),
    (r"\bFlags\b", "copy audit: column renamed Watch for"),
    (r"\bEverywhere\b", "old page name"),
    (r"\bBundled\b|\bburied\b", "copy audit: removed phrasing"),
    (r"\bUTC\b", "visible UTC belongs in the hover title only"),
    (r"OnTheSnow \(|SNOTEL [A-Z]|GeoSphere [A-Z]", "Never 11: provenance caption under a stat"),
    (r"[\U0001F300-\U0001FAFF☀-⛿❄☔]", "Never 1: emoji or pictograms"),
]


def _resort_page(r, resorts, real, units, updated):
    c = ctx(r, units)
    near = D.scope_resorts(resorts, r, "nearby")
    return (E.nav("resort", r["id"], units, updated, False, r, unit_params={"resort": r["id"]})
            + R.screen1_main(r, c, updated, near) + (P.resort_line(r, units) or "")
            + R.daily_table(r, c)
            + R.compare_section(r, D.sort_resorts(near, "next3", units), c, len(resorts), "?view=all")
            + R.stations_section(r, c) + R.footer(real, c))


@pytest.mark.parametrize("rid", IDS)
def test_pages_have_no_leaks_or_banned_strings(by_id, resorts, real, rid, fresh):
    r = by_id[rid]
    for units in UNITS:
        t = visible(_resort_page(r, resorts, real, units, fresh))
        body = t.split("Sources")[0]
        assert not re.search(r"\b(None|nan|NaN|inf|Infinity|null|undefined)\b", t), (rid, units)
        for pat, why in BANNED:
            m = re.search(pat, body)
            assert not m, (rid, units, why, body[max(0, m.start() - 40):m.end() + 40])
        if units == "in":
            assert "°C" not in t and not re.search(r"\d mm\b", t), rid


@pytest.mark.parametrize("units", UNITS)
def test_list_pages_have_no_leaks_or_banned_strings(resorts, units):
    hs = [E.page(resorts, col, d, units, resorts[0]["id"], rng=rng)
          for col in E.SORT_KEYS for d in E.DIRS for rng in ("all",)]
    hs += [E.page(resorts, "next3", "desc", units, resorts[0]["id"], rng=k) for k in E.RANGE_KEYS]
    hs.append(P.page(resorts, units, resorts[0]["id"]))
    for h in hs:
        t = visible(h)
        assert not re.search(r"\b(None|nan|NaN|inf|Infinity|null|undefined)\b", t)
        for pat, why in BANNED:
            m = re.search(pat, t)
            assert not m, (units, why, t[max(0, m.start() - 40):m.end() + 40])
