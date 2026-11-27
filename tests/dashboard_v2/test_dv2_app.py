"""The real app, the real GitHub read and the real validator, end to end.

The app runs through Streamlit's own AppTest in a subprocess, from a copy of
dashboard_v2/ made at test time next to a data/forecasts_v2.json chosen per
test (the real file, or a fixture). The subprocess's HTTPS_PROXY is a local
test proxy, so the GitHub read meets a real refusing or stalling peer on
localhost and the network is never used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _local_proxy import LocalProxy
from conftest import D, DASH_DIR, DATA_FILE, FIXTURES, ROOT

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_apptest_runner.py"


@pytest.fixture(scope="module")
def refuse_proxy():
    p = LocalProxy("refuse")
    yield p
    p.close()


@pytest.fixture(scope="module")
def stall_proxy():
    p = LocalProxy("stall", hold_s=30)
    yield p
    p.close()


def _env(proxy):
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("https_proxy", "http_proxy", "all_proxy", "no_proxy", "snow_fixture")}
    env["HTTPS_PROXY"] = env["https_proxy"] = proxy.url
    return env


def _tree(tmp_path, data_src=None, raw=None):
    """dashboard_v2/ copied verbatim, plus data/forecasts_v2.json."""
    shutil.copytree(DASH_DIR, tmp_path / "dashboard_v2",
                    ignore=shutil.ignore_patterns("__pycache__", "*.txt", ".streamlit"))
    (tmp_path / "data").mkdir()
    dst = tmp_path / "data" / "forecasts_v2.json"
    if raw is not None:
        dst.write_text(raw, encoding="utf-8")
    elif data_src is not None:
        shutil.copyfile(data_src, dst)
    return tmp_path / "dashboard_v2" / "app.py"


def run_app(app, cases, proxy, timeout=240):
    r = subprocess.run([sys.executable, str(RUNNER), str(app), json.dumps(cases)],
                       capture_output=True, text=True, env=_env(proxy), timeout=timeout, cwd=str(app.parent))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------
# The real file, every page, GitHub refused
# ----------------------------------------------------------------------------

def test_real_file_every_page_renders(tmp_path, refuse_proxy):
    app = _tree(tmp_path, DATA_FILE)
    cases = [{"params": {"resort": "alta"}}, {"params": {"resort": "whitefish", "units": "in"}},
             {"params": {"view": "all", "sort": "best", "dir": "asc"}},
             {"params": {"view": "all", "units": "in", "range": "alps"}},
             {"params": {"view": "powder"}}, {"params": {}}]
    for res in run_app(app, cases, refuse_proxy):
        assert res["exceptions"] == [], res
        assert not res["error_screen"], res
        assert res["leak"] is None, res
        assert res["md_len"] > 2000, res
        if "view" not in res["params"]:
            assert res["has_verdict"], res
    assert refuse_proxy.connects >= 1, "the app should have tried GitHub first"


# ----------------------------------------------------------------------------
# Bad files: the honest error screen, never a traceback
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("name, raw, want", [
    ("github_main_old_schema.json", None, "Forecast data is out of date."),
    ("schema_v2.json", None, "Forecast data is out of date."),
    ("truncated.json", None, "Forecast data failed to load."),
    (None, "", "Forecast data failed to load."),
])
def test_bad_files_show_error_screen(tmp_path, refuse_proxy, name, raw, want):
    """The recorded GitHub-main file (old schema), a schema-2 file, a truncated
    download and an empty file. data.py:20 (deploy-order hazard): this is the
    screen every visitor would get if the dashboard deployed before a
    schema-3 file reached main."""
    app = _tree(tmp_path, FIXTURES / name if name else None, raw=raw)
    [res] = run_app(app, [{"params": {"resort": "alta"}}], refuse_proxy)
    assert res["exceptions"] == [], res
    assert res["error_text"] == want, res


@pytest.mark.xfail(strict=False, reason="finding app.py:73 / data.py _usable: a resorts list with no usable "
                                         "entry passes _usable(), then default_resort([]) raises from max()")
def test_junk_resorts_show_error_screen(tmp_path, refuse_proxy):
    app = _tree(tmp_path, FIXTURES / "junk_resorts.json")
    [res] = run_app(app, [{"params": {}}], refuse_proxy)
    assert res["exceptions"] == [], res
    assert res["error_screen"], res


@pytest.mark.xfail(strict=False, reason="finding render.py:181: one resort with timezone 'Bogus/Zone' and "
                                         "utc_offset_s=90000 crashes other resorts' pages and All resorts")
def test_bad_offset_in_one_resort_spares_the_others(tmp_path, refuse_proxy):
    app = _tree(tmp_path, FIXTURES / "bad_tz_offset.json")
    res = run_app(app, [{"params": {"resort": "vail"}}, {"params": {"view": "all"}},
                        {"params": {"resort": "alta"}}], refuse_proxy)
    for r in res:
        assert r["exceptions"] == [], r


@pytest.mark.xfail(strict=False, reason="finding data.py:391: null name + two resorts without coordinates "
                                         "makes scope_resorts compare str with None")
def test_null_name_without_coords_page(tmp_path, refuse_proxy):
    app = _tree(tmp_path, FIXTURES / "null_name_no_coords.json")
    [res] = run_app(app, [{"params": {"resort": "vail"}}], refuse_proxy)
    assert res["exceptions"] == [], res


# ----------------------------------------------------------------------------
# The GitHub read against a refusing and a stalling peer
# ----------------------------------------------------------------------------

_READ_GH = ("import sys, time, json; sys.path.insert(0, {d!r}); import data as D; "
            "t = time.time(); x = D._read_github(); print(json.dumps([x is None, time.time() - t]))")


def _read_github(proxy, timeout=60):
    r = subprocess.run([sys.executable, "-c", _READ_GH.format(d=str(DASH_DIR))], capture_output=True,
                       text=True, env=_env(proxy), timeout=timeout)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_github_refused_returns_none_fast(refuse_proxy):
    is_none, secs = _read_github(refuse_proxy)
    assert is_none and secs < 3


@pytest.mark.xfail(strict=False, reason="finding data.py:64: GitHub read has a 10 s per-socket timeout and "
                                         "no total deadline; a stalled GitHub costs a visitor >= 10 s")
def test_github_stall_is_bounded(stall_proxy):
    is_none, secs = _read_github(stall_proxy)
    assert is_none
    assert secs < 6, f"stalled GitHub held the read for {secs:.1f} s"


# ----------------------------------------------------------------------------
# Repository state the deploy depends on (data.py:20, validator baseline)
# ----------------------------------------------------------------------------

def _git_show(path):
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def test_recorded_github_main_file_is_rejected():
    """The recorded old-schema file must never be shown as a forecast."""
    old = json.loads((FIXTURES / "github_main_old_schema.json").read_text(encoding="utf-8"))
    assert old.get("schema_version") is None
    assert D._usable(old) is False


@pytest.mark.xfail(strict=False, reason="finding data.py:20 deploy-order hazard: the committed "
                                         "data/forecasts_v2.json (what GitHub main serves) is old-schema")
def test_committed_data_file_is_schema_3():
    raw = _git_show("data/forecasts_v2.json")
    assert raw is not None
    assert D._usable(json.loads(raw))


@pytest.mark.xfail(strict=False, reason="finding data.py:20 deploy-order hazard: the committed refresh "
                                         "workflow still runs the old fetcher, without the validator")
def test_committed_workflow_runs_new_fetcher_and_validator():
    wf = _git_show(".github/workflows/refresh_v2.yml") or ""
    assert "fetch_data_v2.py" in wf and "validate_forecasts_v2.py" in wf


def _health(path):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_forecasts_v2.py"), str(path),
                           "--health"], capture_output=True, text=True, timeout=120)


@pytest.mark.xfail(strict=False, reason="finding validate_forecasts_v2.py:56: data file predates the "
                                         "validator's observed_utc requirement; --health exits 1")
def test_real_file_passes_validator_health():
    r = _health(DATA_FILE)
    assert r.returncode == 0, r.stdout[-1500:]


def test_validator_errors_on_real_file_are_only_observed_utc():
    """Pins the baseline the finding describes: if the real file fails, it
    fails only for the missing recent.measured.observed_utc key."""
    r = _health(DATA_FILE)
    bad = [ln for ln in (r.stdout + r.stderr).splitlines() if "INVALID" in ln or "missing keys" in ln]
    other = [ln for ln in bad if "observed_utc" not in ln]
    assert other == [], other[:10]


# ----------------------------------------------------------------------------
# Requirements
# ----------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="finding requirements.txt:2: pydeck==0.9.1 is pinned but nothing "
                                         "in dashboard_v2/ imports it")
def test_every_pinned_requirement_is_used():
    req = (DASH_DIR / "requirements.txt").read_text(encoding="utf-8")
    pins = [re.split(r"[=<>~!\[ ]", ln.strip())[0] for ln in req.splitlines()
            if ln.strip() and not ln.startswith("#")]
    src = "\n".join(p.read_text(encoding="utf-8") for p in DASH_DIR.glob("*.py"))
    for pkg in pins:
        if pkg in ("tzdata",):  # data-only package, used by zoneinfo, never imported
            continue
        assert re.search(rf"^\s*(import|from)\s+{re.escape(pkg)}\b", src, flags=re.M), pkg


def test_streamlit_pin_matches_installed():
    import streamlit
    req = (DASH_DIR / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"^streamlit==(\S+)", req, flags=re.M)
    assert m and m.group(1) == streamlit.__version__
