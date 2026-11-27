"""The fetcher's concurrency: report pages fetched four at a time with spaced starts, and
the three fetch chains (measured stations, resort reports, Open-Meteo) run in parallel.

Real local servers serving recorded payloads; no mocks, no patching. Concurrency is
switched off through the fetcher's own environment settings (FETCH_PARALLEL_CHAINS,
FETCH_REPORT_WORKERS) to get the serial result to compare against.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from conftest import FIXTURES, REPO, fixture_bytes, load_fixture
from _replay import InterceptProxy, ReplayServer

OTS_PAGE = gzip.decompress((FIXTURES / "ots_tignes.html.gz").read_bytes())   # recorded 27 Nov 2026


def test_report_pages_overlap_but_starts_stay_spaced(f, resorts_by_name):
    """Eight recorded pages, each answered after 1.5 s (production pages take about 1.4 s).
    Starts must stay at least the configured spacing apart, and the run must take about
    one spacing per page instead of latency plus a 1 s sleep per page."""
    names = [n for n, r in resorts_by_name.items() if r["ots_url"]][:8]
    rules = [{"path": "/ots/", "mode": "slow", "delay_s": 1.5, "body": OTS_PAGE, "ctype": "text/html"}]
    with ReplayServer(rules) as srv:
        rs = [dict(resorts_by_name[n], notes=[]) for n in names]
        for r in rs:
            r["ots_url"] = f"{srv.url}/ots/{r['id']}"
        w, ps = [], {}
        t0 = time.monotonic()
        f.onthesnow_all(rs, w, ps)
        el = time.monotonic() - t0
        starts = sorted(e["t"] for e in srv.log)
        order = [e["path"].rsplit("/", 1)[-1] for e in sorted(srv.log, key=lambda e: e["t"])]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) == 8
    assert min(gaps) >= f.REPORT_SPACING_S - 0.05, gaps
    assert order == [r["id"] for r in rs]                       # requested in page order
    assert el < 7 * f.REPORT_SPACING_S + 1.5 + 2, el            # serial took 8 x 1.5 + 7 x 1 = 19 s
    assert ps["OnTheSnow"] == "8/8"
    first = (rs[0].get("ots_report"), rs[0].get("ots_observed"), rs[0]["notes"])
    assert all((r.get("ots_report"), r.get("ots_observed"), r["notes"]) == first for r in rs)


def test_report_pages_skip_by_budget_in_page_order(f, resorts_by_name):
    """With the phase budget nearly spent, the pages not yet started are skipped with the
    budget note, and the skipped ones are always the last pages, as in the serial loop."""
    names = [n for n, r in resorts_by_name.items() if r["ots_url"]][:10]
    rules = [{"path": "/ots/", "mode": "slow", "delay_s": 0.5, "body": OTS_PAGE, "ctype": "text/html"}]
    out = {}

    def run():
        # A thread-local phase deadline, the way the fetch chains set it: 4.5 s from now,
        # less the 1 s the page loop keeps back.
        f._THREAD.phase_deadline = time.monotonic() + 4.5
        with ReplayServer(rules) as srv:
            rs = [dict(resorts_by_name[n], notes=[]) for n in names]
            for r in rs:
                r["ots_url"] = f"{srv.url}/ots/{r['id']}"
            w = []
            f.onthesnow_all(rs, w, {})
            out.update(rs=rs, w=w, n=len(srv.log))

    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    rs = out["rs"]
    skipped = [i for i, r in enumerate(rs) if any("time budget" in n for n in r["notes"])]
    fetched = [i for i, r in enumerate(rs) if r.get("ots_observed") or any("OnTheSnow report" in n for n in r["notes"])]
    assert skipped and fetched
    assert skipped == list(range(len(rs) - len(skipped), len(rs)))   # a tail of the list
    assert out["n"] == len(rs) - len(skipped)
    assert any(f"{len(skipped)} of {len(rs)} pages skipped" in w for w in out["w"])


def test_counter_is_exact_across_threads(f):
    c = f.HttpCounter()

    def bump():
        for i in range(20000):
            c.count(retry=i % 4 == 0)

    ts = [threading.Thread(target=bump) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert (c.requests, c.retries) == (160000, 40000)


def test_hour_key_cache_keeps_aware_times_apart(f):
    naive = datetime(2026, 11, 28, 7, 0)
    assert f._hfmt(naive) == "2026-11-28T07:00"
    a = datetime(2026, 11, 28, 7, 0, tzinfo=timezone.utc)
    b = a.astimezone(timezone(timedelta(hours=9)))
    assert a == b and f._hfmt(a) == "2026-11-28T07:00" and f._hfmt(b) == "2026-11-28T16:00"


def _main_rules():
    return [
        {"host": r"wcc\.sc", "path": "/stations", "mode": "replay", "body": fixture_bytes("awdb_stations.json")},
        {"host": r"wcc\.sc", "path": "/data", "mode": "replay", "body": fixture_bytes("awdb_daily.json")},
        {"host": r"transparenciacatalunya", "path": "nzvn-apee", "mode": "replay",
         "body": fixture_bytes("meteocat_readings.json")},
        {"host": r"transparenciacatalunya", "path": "yqwd-vj5e", "mode": "replay",
         "body": fixture_bytes("meteocat_stations.json")},
        {"host": "onthesnow", "mode": "slow", "delay_s": 0.3, "body": OTS_PAGE, "ctype": "text/html"},
        {"host": "open-meteo", "path": "models=", "mode": "replay", "om": load_fixture("om_models_probe.json")},
        {"host": "open-meteo", "path": "hourly=", "mode": "replay", "om": load_fixture("om_hourly_probe.json")},
        {"host": "open-meteo", "path": "", "mode": "replay", "om": load_fixture("om_daily_probe.json")},
    ]


def _run(tmp_path, tag, names, extra_env):
    cert_dir = tmp_path / f"certs_{tag}"
    cert_dir.mkdir()
    out = tmp_path / f"{tag}.json"
    with InterceptProxy(_main_rules(), str(cert_dir)) as proxy:
        env = proxy.env(dict(os.environ))
        env.update(extra_env)
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "fetch_data_v2.py"), "--only", names,
                            "--out", str(out)],
                           env=env, capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert p.returncode in (0, 2), p.stderr[-2000:]
    return p, (json.loads(out.read_text()) if out.exists() else None)


def _blank_clock(doc):
    doc["updated"] = None
    for k in ("started_utc", "finished_utc", "duration_s"):
        doc["run"][k] = None
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def test_parallel_run_writes_the_same_file_as_a_serial_run(tmp_path):
    """Same recorded payloads, run serially and in parallel at the same time: the files must
    match byte for byte once the run clock fields are blanked (notes, warnings and
    provider_status in the same order)."""
    names = "Alta,Vail,Vallter 2000,Tignes,Niseko United"
    res = {}
    jobs = {"serial": {"FETCH_PARALLEL_CHAINS": "0", "FETCH_REPORT_WORKERS": "1"},
            "parallel": {}}
    ts = [threading.Thread(target=lambda k=k, e=e: res.__setitem__(k, _run(tmp_path, k, names, e)))
          for k, e in jobs.items()]
    for t in ts:
        t.start()
    for t in ts:
        t.join(320)
    (ps, ds), (pp, dp) = res["serial"], res["parallel"]
    assert (ps.returncode, ds is None) == (pp.returncode, dp is None)
    if ds is None:   # validation refused both (recorded forecast window has passed); compare logs
        tail = lambda p: [l.split("] ", 1)[-1] for l in p.stderr.splitlines() if "VALIDATION" in l]
        assert tail(ps) == tail(pp)
        return
    assert _blank_clock(ds) == _blank_clock(dp)
    assert ds["run"]["provider_status"]["OnTheSnow"] == "5/5"
