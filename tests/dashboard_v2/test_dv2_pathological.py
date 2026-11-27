"""Pathological inputs from the dashboard robustness audit, against the real functions.

Each fixture is a real recorded resort record with only the field(s) the
finding names changed (fixtures/make_fixtures.py). A test that fails against
the current code is marked xfail with the finding it reproduces, so the
assertion states the behaviour we want, not the behaviour we have.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import pytest

from conftest import (D, E, P, R, ROOT, UNITS, inline_text, load_fixture, usable_resorts, visible)

sys.path.insert(0, str(ROOT / "scripts"))
import validate_forecasts_v2 as V  # noqa: E402


def ctx(r, units="cm"):
    return R.Ctx(r["id"], "nearby", "next3", units)


def resorts_of(name):
    return usable_resorts(load_fixture(name))


def by_id(name):
    return {r["id"]: r for r in resorts_of(name)}


def render_everything(resorts, data, now, units_list=UNITS):
    """Everything app.main() calls, for every resort in the list."""
    upd = D.updated_utc(data)
    D.default_resort(resorts)
    P.any_powder(resorts)
    for units in units_list:
        for col in E.SORT_KEYS:
            E.page(resorts, col, "desc", units, resorts[0]["id"])
        P.page(resorts, units, resorts[0]["id"])
        for r in resorts:
            c = ctx(r, units)
            near = D.scope_resorts(resorts, r, "nearby")
            E.nav("resort", r["id"], units, upd, False, r, {"resort": r["id"]}, False)
            R.option_label(r)
            R.screen1_main(r, c, now, near)
            P.resort_line(r, units)
            R.daily_table(r, c)
            for scope in ("nearby", "country"):
                for sort in D.SORTS:
                    rows = D.sort_resorts(D.scope_resorts(resorts, r, scope), sort, units)
                    R.compare_section(r, rows, R.Ctx(r["id"], scope, sort, units), len(resorts), "?view=all")
            R.stations_section(r, c)
            R.footer(data, c)


# ----------------------------------------------------------------------------
# data.py:20 deploy-order hazard: the old file is rejected, never half-shown
# ----------------------------------------------------------------------------

def test_old_schema_file_is_unusable():
    assert D._usable(load_fixture("github_main_old_schema.json")) is False
    assert D._usable(load_fixture("schema_v2.json")) is False
    assert D._usable(dict(load_fixture("schema_v2.json"), schema_version="3")) is False


def test_fixture_flag_needs_env():
    d = load_fixture("snapshot_base_gate.json")
    assert D._usable(d) is True
    assert D._usable(dict(d, fixture=True)) is False  # SNOW_FIXTURE is not set in this process


# ----------------------------------------------------------------------------
# render.py:181 and powder.py:97: an offset outside +-24 h
# ----------------------------------------------------------------------------

BAD_TZ = "finding render.py:181: utc_offset_s outside +-24 h reaches timezone(timedelta(...)) unchecked"


@pytest.mark.xfail(strict=False, reason=BAD_TZ)
def test_resort_today_survives_bad_offset():
    r = by_id("bad_tz_offset.json")["alta"]
    assert R.resort_today(r) is not None


@pytest.mark.xfail(strict=False, reason=BAD_TZ.replace("render.py:181", "powder.py:97 (_as_utc)"))
def test_powder_reading_time_survives_bad_offset():
    r = by_id("bad_tz_offset.json")["alta"]
    assert P._as_utc("2026-11-27", r) is not None


@pytest.mark.xfail(strict=False, reason=BAD_TZ + "; vail's page lists alta and crashes too")
def test_bad_offset_in_one_resort_spares_every_page(fresh):
    d = load_fixture("bad_tz_offset.json")
    render_everything(usable_resorts(d), d, fresh, units_list=("cm",))


@pytest.mark.xfail(strict=False, reason="finding render.py:181 (optional fix): the validator accepts "
                                         "utc_offset_s=90000")
def test_validator_rejects_out_of_range_offset():
    d = load_fixture("bad_tz_offset.json")
    errs = V.validate(d, expected_count=None)
    assert any("utc_offset_s" in e for e in errs), errs[:5]


# ----------------------------------------------------------------------------
# data.py:391 and data.py:30: null and non-string names
# ----------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="finding data.py:391: scope_resorts tiebreak compares str with None")
def test_scope_resorts_null_name_no_coords():
    rs = resorts_of("null_name_no_coords.json")
    vail = next(r for r in rs if r["id"] == "vail")
    out = D.scope_resorts(rs, vail, "nearby")
    assert {r["id"] for r in out} == {r["id"] for r in rs}


@pytest.mark.xfail(strict=False, reason="finding data.py:30: name_key() crashes on a non-string name")
def test_name_key_tolerates_non_string():
    assert D.name_key(12345) == "" or isinstance(D.name_key(12345), str)
    d = load_fixture("int_name.json")
    D.sort_resorts(usable_resorts(d), "name")


def test_name_key_folds_accents():
    assert D.name_key("Méribel") < D.name_key("Morzine")
    assert D.name_key(None) == ""


# ----------------------------------------------------------------------------
# app.py:73: a resorts list with nothing usable
# ----------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="finding app.py:73: _usable() accepts a resorts list with no dict "
                                         "that has an id; the app then crashes in default_resort([])")
def test_usable_rejects_list_without_usable_resorts():
    assert D._usable(load_fixture("junk_resorts.json")) is False


@pytest.mark.xfail(strict=False, reason="finding app.py:73 (proof): validate_forecasts_v2.validate itself "
                                         "raises AttributeError on a non-dict resort")
def test_validator_reports_junk_resorts_instead_of_crashing():
    errs = V.validate(load_fixture("junk_resorts.json"), expected_count=None)
    assert errs


def test_empty_resorts_is_unusable():
    assert D._usable(dict(load_fixture("junk_resorts.json"), resorts=[])) is False


# ----------------------------------------------------------------------------
# render.py:805: confidence_tag must not say more than the data
# ----------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="finding render.py:805 (a): agree_through naming a missing day "
                                         "prints 'Forecasts agree through None.'")
def test_confidence_tag_missing_day_says_nothing_false():
    r = resorts_of("agree_through_missing_day.json")[0]
    tag = R.confidence_tag(r)
    assert tag is None or "None" not in tag
    assert "None" not in visible(R.hero_block(r, ctx(r)))


@pytest.mark.xfail(strict=False, reason="finding render.py:805 (b): a missing ECMWF day-0 value is shown "
                                         "as 'Forecasts disagree from today.'")
def test_confidence_tag_missing_model_value_is_not_disagreement():
    r = resorts_of("ecmwf_day0_null.json")[0]
    tag = R.confidence_tag(r) or ""
    assert "disagree" not in tag, tag


def test_confidence_tag_real_file_is_well_formed(resorts):
    ok = re.compile(r"^Models (agree on snow (all week|only \w+|through \w+)|disagree on snow from \w+)\.$")
    for r in resorts:
        tag = R.confidence_tag(r)
        assert tag is None or ok.match(tag), (r["id"], tag)


# ----------------------------------------------------------------------------
# Honest nulls: every missing number is "—", zero stays "0"
# ----------------------------------------------------------------------------

def test_honest_nulls_render_as_dash(fresh):
    d = load_fixture("honest_nulls.json")
    rs = usable_resorts(d)
    r = next(x for x in rs if x["id"] == "alta-nulls")
    for units in UNITS:
        c = ctx(r, units)
        assert D.next3(r) is None and D.week_total(r) is None
        assert R.verdict(r, c, fresh) == "No forecast for this resort."
        assert R.hero_block(r, c) == ""
        assert R.best_cell(r, units) == R.DASH
        row = R.resort_row(r, units, "x", "?", 50)
        cells = [visible(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row)]
        # 3-day, 7-day, last 3 days, best day, depth: dashes, never blank, never 0
        assert cells[2] == R.DASH and cells[3] == R.DASH and cells[4] == R.DASH
        assert cells[5] == R.DASH and cells[7] == R.DASH, cells
        g = visible(R.ground_line(r, c))
        assert "No snow depth report." in g and R.DASH in g
        sec = R.days_section(r, c)
        nums = re.findall(r'<span class="n(?: accent)?">([^<]*)</span>', sec)
        assert nums and all(n == R.DASH for n in nums), nums
        st = R.stations_section(r, c)
        if st:
            assert R.DASH in visible(st)
        for h in (R.screen1_main(r, c, fresh, rs), sec, st, row):
            assert not re.search(r"\b(None|nan|NaN|inf|null)\b", visible(h))
    render_everything(rs, d, fresh)


def test_zero_stays_zero(resorts):
    z = [r for r in resorts if D.next3(r) == 0]
    for r in z:
        row = R.resort_row(r, "cm", "x", "?", 50)
        cells = [visible(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row)]
        assert cells[2] == "0 cm", (r["id"], cells)  # zero stays 0, bare of "~"


def test_bare_resort_record_renders(fresh):
    """A record that is only an id: every function answers in words or dashes."""
    d = load_fixture("snapshot_whitefish.json")
    bare = {"id": "bare"}
    rs = [bare] + usable_resorts(d)
    render_everything(rs, d, fresh)
    c = ctx(bare)
    assert R.verdict(bare, c, fresh) == "No forecast for this resort."
    assert R.best_cell(bare, "cm") == R.DASH


# ----------------------------------------------------------------------------
# Inches in prose: "9 in snow" reads as the preposition (user, 27 Nov)
# ----------------------------------------------------------------------------

def test_inch_prose_uses_double_prime(resorts):
    """Fixed (copy audit): every verdict amount now goes through _snow_amount /
    _disp_amount, which use the double prime in inches mode."""
    now = datetime.now(timezone.utc)
    bad = []
    for r in resorts:
        v = inline_text(R.verdict(r, ctx(r, "in"), now))
        if re.search(r"\d in\b", v):
            bad.append((r["id"], v))
    assert bad == []
