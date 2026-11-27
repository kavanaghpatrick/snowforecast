"""Every render function returns a string for every resort in the real file, in both units.

Functions documented as "... or None" may also return None; nothing raises.
The HTML builders also keep the module's own contract (render.py docstring):
no blank lines, so Streamlit's markdown pass keeps each block as raw HTML.
"""

from __future__ import annotations

import pytest

from conftest import D, E, P, R, UNITS, resort_ids

IDS = resort_ids()

# (name, call) for functions that must return str
STR_FUNCS = [
    ("verdict", lambda r, c, rs, data, now: R.verdict(r, c, now)),
    ("hero_block", lambda r, c, rs, data, now: R.hero_block(r, c)),
    ("bands_block", lambda r, c, rs, data, now: R.bands_block(r, c)),
    ("strip_block", lambda r, c, rs, data, now: R.strip_block(r, c)),
    ("ground_line", lambda r, c, rs, data, now: R.ground_line(r, c, R.day_vmax(r, c))),
    ("recount_note", lambda r, c, rs, data, now: R.recount_note(r)),
    ("first_screen", lambda r, c, rs, data, now: R.first_screen(r, [], c, now)),
    ("screen1_main", lambda r, c, rs, data, now: R.screen1_main(r, c, now, D.scope_resorts(rs, r, "nearby"))),
    ("daily_table", lambda r, c, rs, data, now: R.daily_table(r, c)),
    ("days_section", lambda r, c, rs, data, now: R.days_section(r, c)),
    ("stations_section", lambda r, c, rs, data, now: R.stations_section(r, c)),
    ("footer", lambda r, c, rs, data, now: R.footer(data, c)),
    ("best_cell", lambda r, c, rs, data, now: R.best_cell(r, c.units)),
    ("where_line", lambda r, c, rs, data, now: R.where_line(r, c)),
    ("option_label", lambda r, c, rs, data, now: R.option_label(r)),
    ("window_words", lambda r, c, rs, data, now: R.window_words(r)),
    ("update_line", lambda r, c, rs, data, now: R.update_line(D.updated_utc(data), r)),
    ("freshness_html", lambda r, c, rs, data, now: R.freshness_html(D.updated_utc(data), r)),
    ("resort_row", lambda r, c, rs, data, now: R.resort_row(r, c.units, r["id"], "?", 50, km="")),
    ("nav", lambda r, c, rs, data, now: E.nav("resort", r["id"], c.units, now, False, r, {"resort": r["id"]})),
    ("resort_line", lambda r, c, rs, data, now: P.resort_line(r, c.units)),
]
# functions that return str or None
OPT_FUNCS = [
    ("confidence_tag", lambda r, c: R.confidence_tag(r)),
    ("split_line", lambda r, c: R.split_line(r, c)),
    ("low_threat", lambda r, c: R.low_threat(r, c)),
] + [(f"strip_text[{b}]", (lambda b: lambda r, c: R.strip_text(r, b, c))(b)) for b in ("base", "mid", "summit")]


@pytest.mark.parametrize("rid", IDS)
def test_render_functions_return_strings(real, resorts, fresh, rid):
    r = next(x for x in resorts if x["id"] == rid)
    for units in UNITS:
        c = R.Ctx(rid, "nearby", "next3", units)
        for name, fn in STR_FUNCS:
            out = fn(r, c, resorts, real, fresh)
            assert isinstance(out, str), (rid, units, name, type(out))
            if out.startswith("<"):
                assert "\n\n" not in out, (rid, units, name, "blank line breaks the HTML block")
        for name, fn in OPT_FUNCS:
            out = fn(r, c)
            assert out is None or isinstance(out, str), (rid, units, name, type(out))
        assert all(isinstance(t, str) for t in R.watch_resort(r))
        for lab, txt in R.strip_lines(r, c):
            assert isinstance(lab, str) and isinstance(txt, str)
        for i in range(-1, 15):
            d = D.day_by_index(r, i)
            if d is not None:
                assert all(isinstance(w, str) for w in R.watch_day(r, d, extended=i >= 7))
                cov = R.snow_cover_str(r, d)
                assert cov is None or isinstance(cov, str)


@pytest.mark.parametrize("rid", IDS)
def test_compare_section_every_scope_and_sort(resorts, rid):
    r = next(x for x in resorts if x["id"] == rid)
    for units in UNITS:
        for scope in ("nearby", "country"):
            scoped = D.scope_resorts(resorts, r, scope)
            for sort in D.SORTS:
                c = R.Ctx(rid, scope, sort, units)
                out = R.compare_section(r, D.sort_resorts(scoped, sort, units), c, len(resorts), "?view=all")
                assert isinstance(out, str) and out.startswith('<div class="page sec">')
                assert r.get("name") is None or R.escape(r["name"]) in out


@pytest.mark.parametrize("units", UNITS)
def test_list_pages_return_strings(resorts, units):
    for col in E.SORT_KEYS:
        for d in E.DIRS:
            for rng in E.RANGE_KEYS:
                assert isinstance(E.page(resorts, col, d, units, resorts[0]["id"], rng=rng), str)
    assert isinstance(P.page(resorts, units, resorts[0]["id"]), str)
    assert isinstance(P.page(resorts, units, resorts[0]["id"], debug=True), str)
    assert isinstance(R.screen_error("old"), str) and isinstance(R.screen_error("missing"), str)
