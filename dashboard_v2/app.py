"""Snow forecast dashboard v2.

One long page about one resort, in the order a skier decides: go or not,
where, when, will it be good (docs/ux_hierarchy_spec.md).
Reads data/forecasts_v2.json (schema version 3). State lives in the URL:
?resort=<id>&scope=nearby|country&sort=next3|week|best|depth|name&units=cm|in
All resorts page (every resort, one table):
?view=all&sort=name|next3|week|best|flags|depth|ext&dir=asc|desc&units=cm|in
Powder page (measured snow, then the powder forecast):
?view=powder&resort=<id>&units=cm|in

Run: streamlit run dashboard_v2/app.py
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

import art as A  # noqa: E402
import everywhere as E  # noqa: E402
import powder as P  # noqa: E402
import render as R  # noqa: E402
from data import (  # noqa: E402
    COUNTRY_ORDER,
    SORTS,
    UNITS,
    default_resort,
    load,
    name_key,
    scope_resorts,
    sort_resorts,
    updated_utc,
)
from theme import apply_theme  # noqa: E402

PAGE_SCOPES = ("nearby", "country")  # "all" is now the All resorts page


def _param(name, allowed, default):
    v = st.query_params.get(name)
    return v if v in allowed else default


def _html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def _on_pick():
    st.session_state["picked"] = st.session_state["picker"]


def main():
    data, origin, err = load()
    debug = st.query_params.get("debug") == "1"

    if err is not None:
        apply_theme("Snow forecast")
        _html(R.screen_error(err))
        return

    resorts = [r for r in data["resorts"]
               if isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"]]
    by_id = {r["id"]: r for r in resorts}
    picked = st.session_state.pop("picked", None)
    if picked in by_id:
        st.query_params["resort"] = picked  # written during the run, so the browser URL follows
    resort_id = st.query_params.get("resort")
    if resort_id not in by_id:
        resort_id = default_resort(resorts)["id"]
    resort = by_id[resort_id]
    units = _param("units", UNITS, "cm")
    updated = updated_utc(data)
    extra = {"debug": "1"} if debug else {}

    hot = P.any_powder(resorts)

    if st.query_params.get("view") == "powder":
        apply_theme("Snow forecast, powder")
        _html("<style>" + R.CSS + E.CSS + P.CSS + A.CSS + "</style>")
        _html(E.nav("powder", resort_id, units, updated, debug, resort,
                    unit_params={"view": "powder", "resort": resort_id}, powder_hot=hot))
        _html(A.powder_art())
        _html(A.powder_companion(P.page(resorts, units, resort_id, debug), updated))
        ctx = R.Ctx(resort_id, "nearby", "next3", units, debug)
        _html(R.footer(data, ctx))
        if st.button("Refresh forecast"):
            load.clear()
            st.rerun()
        _html('<div class="close-end"></div>')
        return

    if st.query_params.get("view") == "all":
        apply_theme("Snow forecast, all resorts")
        sort = _param("sort", E.SORT_KEYS, "next3")
        direction = _param("dir", E.DIRS, "desc")
        rng = _param("range", E.RANGE_KEYS, "all")
        _html("<style>" + R.CSS + E.CSS + "</style>")
        _html(E.nav("all", resort_id, units, updated, debug, resort,
                    unit_params={"view": "all", "sort": sort, "dir": direction,
                                 "range": rng, "resort": resort_id},
                    powder_hot=hot))
        _html(E.page(resorts, sort, direction, units, resort_id, debug, rng=rng))
        ctx = R.Ctx(resort_id, "nearby", "next3", units, debug)
        _html(R.footer(data, ctx))
        if st.button("Refresh forecast"):
            load.clear()
            st.rerun()
        _html('<div class="close-end"></div>')
        return

    ctx = R.Ctx(resort_id, _param("scope", PAGE_SCOPES, "nearby"), _param("sort", SORTS, "next3"),
                units, debug)
    all_href = "?" + urlencode({"view": "all", "resort": resort_id, "units": units, **extra})

    apply_theme(f"Snow forecast, {resort.get('name', '')}")
    _html("<style>" + E.CSS + R.CSS + A.CSS + "</style>")
    art_state = A.state(resort, updated)

    # Level 0: top bar (corner paintings anchor before it, at the viewport top)
    top_art = A.top_art(art_state, resort_id, resort)
    if top_art:
        _html(top_art)
    _html(E.nav("resort", resort_id, units, updated, debug, resort,
                unit_params={"resort": resort_id, "scope": ctx.scope, "sort": ctx.sort}, powder_hot=hot))
    verdict_art = A.verdict_art(art_state, resort_id, resort)
    if verdict_art:
        _html(verdict_art)

    # Level 1: picker
    def order(r):
        c = r.get("country")
        return (COUNTRY_ORDER.index(c) if c in COUNTRY_ORDER else 99, name_key(r.get("name")))

    ids = [r["id"] for r in sorted(resorts, key=order)]
    st.session_state["picker"] = resort_id
    st.selectbox("Resort", ids, key="picker", label_visibility="collapsed",
                 format_func=lambda i: R.option_label(by_id[i]), on_change=_on_pick)

    # Levels 2 to 6: verdict, hero, mountain strip, nearby line, ground line
    nearby = scope_resorts(resorts, resort, "nearby")
    _html('<div class="page">' + R.screen1_main(resort, ctx, updated, nearby) + "</div>")

    # Powder forecast line, only when this resort has a qualifying powder day
    pl = P.resort_line(resort, units)
    if pl:
        _html(f'<div class="page"><p class="body prose gap1">{pl}</p></div>')

    # Levels 7 and 8: daily table and days 8-15
    table_art = A.table_art(art_state, resort_id)
    if table_art:
        _html(table_art)
    _html(R.daily_table(resort, ctx))

    # Level 9: compare resorts
    scoped = scope_resorts(resorts, resort, ctx.scope)
    _html(R.compare_section(resort, sort_resorts(scoped, ctx.sort, ctx.units), ctx, len(resorts), all_href))

    # Level 10: snow stations, only when one exists
    st_html = R.stations_section(resort, ctx)
    if st_html:
        _html(st_html)

    # Level 11: footer
    footer_art = A.footer_art(art_state, resort_id)
    if footer_art:
        _html(footer_art)
    _html(R.footer(data, ctx))
    if st.button("Refresh forecast"):
        load.clear()
        st.rerun()
    _html('<div class="close-end"></div>')
    foot_art = A.foot_art(art_state, resort_id)
    if foot_art:
        _html(foot_art)


main()
