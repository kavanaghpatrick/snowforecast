"""Fixtures for the fetcher_v2 suite (scripts/fetch_data_v2.py pure logic).

No mocks and no monkeypatching. The fetcher and validator are imported from
their files with importlib (the same module the CLI runs). Network paths run
against real local servers (see _replay.py) that serve recorded payloads.

Fixture provenance (all recorded from the live providers on 27 Nov 2026,
copied from the session scratchpad so the suite does not depend on it):

  om_hourly_crystal_bands.json     Open-Meteo hourly, Crystal Mountain base 1,341 / mid 1,736 /
                                   summit 2,131 m, past_days=3, 10 days (scratchpad om/now.json)
  om_hourly_timberline_bands.json  Open-Meteo hourly, Timberline cells at 1,219 / 1,911 / 2,603 m,
                                   7 days from 27 Nov (scratchpad om/timberline.json)
  om_daily_probe.json              Open-Meteo daily, fetcher's DAILY_VARS, Alta base, Snowbird
                                   summit, Zermatt base (scratchpad probe/a.json)
  om_hourly_probe.json             Open-Meteo hourly + current, fetcher's HOURLY_VARS, Snowbird
                                   3,215 m and Zermatt 3,883 m (scratchpad probe/b.json)
  om_models_probe.json             Open-Meteo per-model daily snowfall (scratchpad probe/c.json)
  om_error_daily_limit.json        Open-Meteo's own error answer when this machine's daily quota ran
                                   out (scratchpad om/crystal.json)
  awdb_stations.json / awdb_daily.json
                                   USDA AWDB metadata (6 configured triplets) and daily SNWD/WTEQ
  meteocat_readings.json / meteocat_stations.json
                                   Meteocat XEMA snow depth (variable 38) and station metadata
  jma_amedastable_subset.json      JMA AMeDAS station table entries 16126 and 35426
  forecasts_v2_run_20261127T1215Z.json.gz
                                   data/forecasts_v2.json as written by the 27 Nov 12:15 UTC run
  forecasts_v2_replay_smoke.json.gz
                                   a 7-resort file the current fetcher wrote from recorded payloads
                                   through the robustness harness (valid against the validator)
"""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE))

from _replay import InterceptProxy  # noqa: E402

# started_utc of the recorded 27 Nov run; the recorded payloads were fetched that morning.
RUN_UTC = datetime(2026, 11, 27, 12, 15, 48, tzinfo=timezone.utc)


def _load_module(name, path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def v():
    return _load_module("validate_forecasts_v2", SCRIPTS / "validate_forecasts_v2.py")


@pytest.fixture(scope="session")
def f(v):
    return _load_module("fetch_data_v2", SCRIPTS / "fetch_data_v2.py")


def load_fixture(name):
    p = FIXTURES / name
    if name.endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fixture_bytes(name):
    return (FIXTURES / name).read_bytes()


@pytest.fixture(scope="session")
def real_run_doc():
    return load_fixture("forecasts_v2_run_20261127T1215Z.json.gz")


@pytest.fixture
def real_run(real_run_doc):
    return copy.deepcopy(real_run_doc)


@pytest.fixture(scope="session")
def smoke_doc():
    return load_fixture("forecasts_v2_replay_smoke.json.gz")


@pytest.fixture
def smoke(smoke_doc):
    return copy.deepcopy(smoke_doc)


@pytest.fixture(scope="session")
def resorts_by_name(f):
    return {r["name"]: r for r in f.build_resorts()}


@pytest.fixture
def run_provider(tmp_path):
    """Run one fetcher provider function in a subprocess through the intercept proxy."""

    def _run(func, names, rules, run_utc=RUN_UTC, timeout=120):
        cert_dir = tmp_path / "certs"
        cert_dir.mkdir(exist_ok=True)
        out = tmp_path / f"{func}.json"
        with InterceptProxy(rules, str(cert_dir)) as proxy:
            p = subprocess.run(
                [sys.executable, str(HERE / "_provider_driver.py"), func, ",".join(names),
                 run_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)],
                env=proxy.env(dict(os.environ)), capture_output=True, text=True, timeout=timeout,
                cwd=str(REPO))
            log = list(proxy.log)
        assert p.returncode == 0, p.stderr[-3000:]
        res = json.loads(out.read_text())
        res["_proxy_log"] = log
        res["_stderr"] = p.stderr
        res["by_name"] = {r["name"]: r for r in res["resorts"]}
        return res

    return _run
