"""Fetcher robustness fixes, each checked against real local sockets (no mocks).

- redirects: a caller's own headers (AEMET's api_key) never reach another host,
  and only http(s) URLs are fetched
- time: a server that drips bytes is cut at the attempt's wall-clock limit, a
  hanging report site cannot use up the run, and 4xx answers are not retried
- shape: a 200 answer that is empty or the wrong shape is a provider outage, not
  every station "removed"; Open-Meteo points that are not objects are rejected
- Météo-France: a zlib.error in one department file loses only that department

Success bodies are recorded real payloads (fixtures/). mf_H_73_subset.csv.gz and
mf_H_74_subset.csv.gz are the real Météo-France hourly files recorded on 27 Nov,
cut to the configured stations' rows since 1 Nov 2026.
"""

import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import pytest

from _replay import InterceptProxy, ReplayServer
from conftest import HERE, REPO, fixture_bytes, load_fixture

DAILY = fixture_bytes("om_daily_probe.json")


def _get(f, url, **kw):
    c = f.HttpCounter()
    args = dict(attempts=1, backoff_s=[0], timeout_s=5, headers=None, counter=c, want_json=True)
    args.update(kw)
    body, info = f._fetch(url, **args)
    return body, info, c


# ----------------------------------------------------------------------------- redirects


def test_redirect_to_other_host_drops_the_secret_header(f):
    with ReplayServer([{"path": "/steal", "mode": "replay", "body": DAILY}]) as other:
        # 127.0.0.1 -> localhost: same machine, different host name, as a hostile redirect would be
        loc = f"http://localhost:{other.port}/steal"
        with ReplayServer([{"path": "/opendata", "mode": "redirect", "location": loc}]) as aemet:
            body, info, _ = _get(f, aemet.url + "/opendata",
                                 headers={"api_key": "SECRET-AEMET-KEY-123", "Accept": "application/json"})
        seen = other.log
    assert body == json.loads(DAILY)
    assert len(seen) == 1
    assert "api_key" not in seen[0]["headers"]
    assert "SECRET-AEMET-KEY-123" not in json.dumps(seen[0]["headers"])
    assert seen[0]["headers"].get("accept") == "application/json"   # harmless headers still go


def test_redirect_on_same_host_keeps_the_header(f):
    with ReplayServer([{"path": "/opendata", "mode": "redirect", "location": "/datos"},
                       {"path": "/datos", "mode": "replay", "body": DAILY}]) as s:
        body, _, _ = _get(f, s.url + "/opendata", headers={"api_key": "K"})
        log = s.log
    assert body == json.loads(DAILY)
    assert [e["path"] for e in log] == ["/opendata", "/datos"]
    assert log[1]["headers"].get("api_key") == "K"


def test_non_http_urls_are_refused(f, tmp_path):
    p = tmp_path / "local.json"
    p.write_text('[{"local": "file read"}]')
    assert f.http_json(p.as_uri(), attempts=1, backoff_s=[0], timeout_s=5) is None
    body, info, c = _get(f, p.as_uri())
    assert body is None and info == "refused non-HTTP URL" and c.requests == 0


def test_redirect_to_file_url_is_refused(f, tmp_path):
    p = tmp_path / "local.json"
    p.write_text('[{"local": "file read"}]')
    with ReplayServer([{"path": "/opendata", "mode": "redirect", "location": p.as_uri()}]) as s:
        body, info, _ = _get(f, s.url + "/opendata")
    assert body is None and info.startswith("HTTP 302")


def test_aemet_follows_datos_only_on_aemets_own_host(tmp_path):
    """AEMET's index answer names the data URL; one pointing anywhere else is not followed."""
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    out = tmp_path / "aemet.json"
    # A failure-mode index answer (AEMET's documented {estado, datos} shape) whose datos
    # points off AEMET's host. No AEMET payload is recorded (no key), so no success case here.
    idx = json.dumps({"estado": 200, "datos": "https://collector.example.invalid/x"}).encode()
    rules = [{"host": r"opendata\.aemet\.es", "mode": "replay", "body": idx},
             {"host": r"collector", "mode": "replay", "body": b"[]"}]
    with InterceptProxy(rules, str(cert_dir)) as proxy:
        env = proxy.env(dict(os.environ))
        env["AEMET_API_KEY"] = "harness-placeholder-not-a-key"
        p = subprocess.run([sys.executable, str(HERE / "_provider_driver.py"), "aemet_batch", "Formigal",
                            "2026-11-27T12:15:48Z", str(out)], env=env, capture_output=True, text=True,
                           timeout=60, cwd=str(REPO))
        log = list(proxy.log)
    assert p.returncode == 0, p.stderr[-2000:]
    res = json.loads(out.read_text())
    assert [e["host"] for e in log] == ["opendata.aemet.es"]
    assert res["status"]["AEMET"] == "failed"
    assert "not followed" in p.stderr


# ----------------------------------------------------------------------------- time


def test_drip_feed_is_cut_at_the_wall_clock_limit(f):
    # 1 KB every 0.05 s: each read succeeds, so a socket timeout never fires.
    big = fixture_bytes("om_hourly_probe.json")
    assert len(big) > 30_000   # about 2.3 s at this rate, limit 1 s
    with ReplayServer([{"path": "/d", "mode": "drip", "chunk": 1000, "every_s": 0.05, "body": big}]) as s:
        t0 = time.monotonic()
        body, info, _ = _get(f, s.url + "/d", timeout_s=1.0)
        el = time.monotonic() - t0
    assert body is None
    assert "deadline" in info or "timed out" in info
    assert el < 2.0


def test_drip_feed_inside_the_limit_is_read_whole(f):
    with ReplayServer([{"path": "/d", "mode": "drip", "chunk": 4000, "every_s": 0.01, "body": DAILY}]) as s:
        body, _, _ = _get(f, s.url + "/d", timeout_s=10)
    assert body == json.loads(DAILY)


def test_deterministic_4xx_is_not_retried(f):
    with ReplayServer([{"path": "/d", "mode": "status", "code": 400,
                        "body": b'{"error":true,"reason":"Cannot initialize WeatherVariable from invalid String value x"}'}]) as s:
        body, info, c = _get(f, s.url + "/d", attempts=3)
        n = len(s.log)
    assert body is None and info == "HTTP 400" and n == 1 and c.retries == 0


def test_408_is_retried(f):
    with ReplayServer([{"path": "/d", "mode": "fail_first", "n": 1, "fail_code": 408, "body": DAILY}]) as s:
        body, _, c = _get(f, s.url + "/d", attempts=2)
    assert body == json.loads(DAILY) and c.retries == 1


def test_retry_after_is_honoured(f):
    with ReplayServer([{"path": "/d", "mode": "fail_first", "n": 1, "fail_code": 503,
                        "headers": {"Retry-After": "1"}, "body": DAILY}]) as s:
        t0 = time.monotonic()
        body, _, c = _get(f, s.url + "/d", attempts=2)
        el = time.monotonic() - t0
    assert body == json.loads(DAILY) and c.retries == 1
    assert el >= 1.0


def _run_main(tmp_path, names, rules, budget_s, reserve_s, timeout=180):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir(exist_ok=True)
    out = tmp_path / "out.json"
    with InterceptProxy(rules, str(cert_dir)) as proxy:
        env = proxy.env(dict(os.environ))
        env.update({"FETCH_RUN_BUDGET_S": str(budget_s), "FETCH_OM_RESERVE_S": str(reserve_s)})
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, str(REPO / "scripts" / "fetch_data_v2.py"), "--only", names,
                            "--out", str(out)], env=env, capture_output=True, text=True, timeout=timeout,
                           cwd=str(REPO))
        el = time.monotonic() - t0
        log = list(proxy.log)
    doc = json.loads(out.read_text()) if out.exists() else None
    return p, el, log, doc


def _om_rules():
    return [
        {"host": "open-meteo", "path": "models=", "mode": "replay", "om": load_fixture("om_models_probe.json")},
        {"host": "open-meteo", "path": "hourly=", "mode": "replay", "om": load_fixture("om_hourly_probe.json")},
        {"host": "open-meteo", "path": "", "mode": "replay", "om": load_fixture("om_daily_probe.json")},
    ]


def test_hanging_report_site_cannot_use_up_the_run(tmp_path):
    """OnTheSnow accepts the connection and never answers. Before the fix one page took
    30 s x 2 attempts and the full roster ran into the workflow's 25-minute kill with no
    Open-Meteo call made. Now the report phase stops at its budget and Open-Meteo runs."""
    rules = [{"host": "onthesnow", "mode": "slow", "delay_s": 3600, "body": b""},
             {"host": r"wcc\.sc", "path": "/stations", "mode": "replay", "body": fixture_bytes("awdb_stations.json")},
             {"host": r"wcc\.sc", "path": "/data", "mode": "replay", "body": fixture_bytes("awdb_daily.json")},
             ] + _om_rules()
    # Budget 50 s, 35 s kept for Open-Meteo: the report phase must give up by about 15 s.
    p, el, log, doc = _run_main(tmp_path, "Vail,Alta", rules, budget_s=50, reserve_s=35)
    ots = [e for e in log if "onthesnow" in e.get("host", "")]
    om = [e for e in log if "open-meteo" in e.get("host", "")]
    assert ots, "OnTheSnow was never asked: " + p.stderr[-1500:]
    assert om, "Open-Meteo was never called: " + p.stderr[-1500:]
    assert el < 50, el
    assert p.returncode in (0, 2), p.stderr[-2000:]
    if doc is not None:   # written when today's date is inside the recorded forecast window
        notes = [n for r in doc["resorts"] for n in r["quality"]["notes"]]
        # Report pages run up to four at a time, so both pages are in flight when the phase
        # budget ends: each says it timed out with no budget left to retry.
        assert any("budget" in n for n in notes if n.startswith("OnTheSnow"))
        assert doc["run"]["provider_status"]["OnTheSnow"].startswith("0/")


def test_report_site_down_stops_after_four_pages(run_provider, resorts_by_name):
    """Four pages in a row lost to connection errors mark the site down; the rest are
    skipped with a note instead of each waiting out its own timeouts."""
    names = [n for n, r in resorts_by_name.items() if r["ots_url"]][:7]
    res = run_provider("onthesnow_all", names, [{"host": "onthesnow", "mode": "close"}])
    asked = [e for e in res["_proxy_log"] if "onthesnow" in e.get("host", "")]
    assert len(asked) == 4 * 2          # 4 pages, 2 attempts each
    skipped = [n for n in names if any("site not responding" in x for x in res["by_name"][n]["notes"])]
    assert len(skipped) == 3
    assert any("stopped after 4 pages in a row" in w for w in res["warnings"])
    assert res["status"]["OnTheSnow"] == "0/7"


# ----------------------------------------------------------------------------- shape


@pytest.mark.parametrize("func,names,host,bodies", [
    ("snotel_batch", ["Alta"], r"wcc\.sc\.egov\.usda\.gov", [b"[]", b"{}"]),
    ("geosphere_batch", ["St. Anton"], r"geosphere", [b"[]", b"{}", b'{"error":"rate limited"}']),
    ("slf_once", ["Zermatt"], r"slf\.ch", [b"[]", b"{}"]),
    ("meteocat_batch", ["Vallter 2000"], r"transparenciacatalunya", [b"[]"]),
])
def test_empty_or_wrong_shape_answer_is_an_outage(run_provider, func, names, host, bodies):
    for body in bodies:
        res = run_provider(func, names, [{"host": host, "mode": "replay", "body": body}])
        prov = next(iter(res["status"]))
        assert res["status"][prov] == "failed", (body, res["status"])
        notes = res["by_name"][names[0]]["notes"]
        assert any("not available this run" in n and "empty or malformed" in n for n in notes), notes
        assert not any("removed" in n for n in notes), notes
        assert len(res["warnings"]) == 1 and "empty or malformed" in res["warnings"][0]


def test_snotel_real_payload_still_ok(run_provider):
    rules = [{"host": r"wcc\.sc", "path": "/stations", "mode": "replay", "body": fixture_bytes("awdb_stations.json")},
             {"host": r"wcc\.sc", "path": "/data", "mode": "replay", "body": fixture_bytes("awdb_daily.json")}]
    res = run_provider("snotel_batch", ["Alta"], rules)
    assert res["status"]["SNOTEL"] == "ok"
    assert any(s["depth_cm"] is not None for s in res["by_name"]["Alta"]["stations"])


@pytest.mark.parametrize("elem", [None, "x", [], {}])
def test_om_points_that_are_not_objects_are_rejected(f, elem):
    with ReplayServer([{"path": "/v1/forecast", "mode": "replay", "body": json.dumps([elem, elem]).encode()}]) as s:
        assert f._om_request(s.url + "/v1/forecast?latitude=1,2&longitude=1,2", 2, attempts=1) is None


# ----------------------------------------------------------------------------- Météo-France


def _zlib_corrupt(raw):
    """Flip 8 bytes in a real gzip file so that reading it raises zlib.error."""
    import csv
    for off in range(20, len(raw) - 20, 7):
        b = bytearray(raw)
        for i in range(off, off + 8):
            b[i] ^= 0x5A
        try:
            with gzip.open(io.BytesIO(bytes(b)), "rt", encoding="utf-8", errors="replace", newline="") as fh:
                for _ in csv.reader(fh, delimiter=";"):
                    pass
        except zlib.error:
            return bytes(b)
        except Exception:
            continue
    raise AssertionError("no zlib.error corruption found")


def test_meteofrance_zlib_error_loses_only_that_department(run_provider):
    h73 = _zlib_corrupt(fixture_bytes("mf_H_73_subset.csv.gz"))
    rules = [{"host": r"data\.gouv\.fr", "mode": "status", "code": 503},   # dataset lookup down: fixed URLs
             {"host": r"ovh\.net", "path": "H_73_", "mode": "replay", "body": h73, "ctype": "application/gzip"},
             {"host": r"ovh\.net", "path": "H_74_", "mode": "replay",
              "body": fixture_bytes("mf_H_74_subset.csv.gz"), "ctype": "application/gzip"}]
    res = run_provider("meteofrance_batch", ["Val Thorens", "Chamonix"], rules, timeout=240)
    assert res["status"]["Meteo-France"] == "1/2 departments"
    assert any("department 73 file could not be read (error" in w for w in res["warnings"]), res["warnings"]
    vt, ch = res["by_name"]["Val Thorens"], res["by_name"]["Chamonix"]
    assert vt["stations"] == []
    assert any("not available this run" in n for n in vt["notes"])
    assert [s["id"] for s in ch["stations"]] == ["74056001"]


# ----------------------------------------------------------------------------- validator coverage


def _first_station_resort(doc):
    return next(r for r in doc["resorts"] if r["stations"] and r["stations"][0]["depth_cm"] is not None)


def test_validator_clean_on_the_replay_smoke_file(v, smoke):
    assert v.validate(smoke, expected_count=None) == []


@pytest.mark.parametrize("change,needle", [
    (lambda d, r: r["stations"][0].__setitem__("depth_cm", -12), "negative value -12"),
    (lambda d, r: r["stations"][0].__setitem__("depth_cm", 9999), "not below 600 cm"),
    (lambda d, r: r["depth"].__setitem__("value_cm", -12) or r.__setitem__("base_depth_cm", -12), "negative value -12"),
    (lambda d, r: r["stations"][0].__setitem__("observed_utc", "2031-01-01T00:00:00Z"), "after the file's update time"),
    (lambda d, r: r["stations"][0].__setitem__("observed_utc", "yesterday"), "unreadable time"),
    (lambda d, r: r.__setitem__("models", "garbage"), "wrong type for ['models']"),
    (lambda d, r: r.__setitem__("now", None), "wrong type for ['now']"),
    (lambda d, r: r["next_72h"].__setitem__("base_cm", "x"), "next_72h.base_cm wrong type"),
    (lambda d, r: r.__setitem__("timezone", None), "timezone or utc_offset_s missing"),
    (lambda d, r: d.__setitem__("updated", "yesterday"), "updated is not an ISO UTC time"),
    (lambda d, r: d.__setitem__("sources", None), "sources is not an object"),
    (lambda d, r: d.__setitem__("thresholds", {}), "thresholds is not a non-empty object"),
    (lambda d, r: d["run"].__setitem__("coverage", None), "run.coverage is not a dict"),
    (lambda d, r: r["stations"][0].__setitem__("swe_mm", -3.0), "negative value -3.0"),
])
def test_validator_catches_malformed_values(v, smoke, change, needle):
    r = _first_station_resort(smoke)
    change(smoke, r)
    errs = v.validate(smoke, expected_count=None)
    assert any(needle in e for e in errs), errs


def test_validator_allows_negative_depth_change_but_not_negative_new_snow(v, smoke):
    r = next(r for r in smoke["resorts"] if r["recent"]["measured"])
    m = r["recent"]["measured"]
    m["kind"], m["h24_cm"] = "depth_change", -4.0
    assert v.validate(smoke, expected_count=None) == []
    m["kind"] = "measured_new_snow"
    assert any("recent.measured.h24_cm negative" in e for e in v.validate(smoke, expected_count=None))


@pytest.mark.parametrize("change", [
    lambda r: r["quality"].pop("forecast_ok"),
    lambda r: r.__setitem__("stations", None),
    lambda r: r["days"][0].__setitem__("index", "0"),
    lambda r: r.__setitem__("next_72h", None),
    lambda r: r["stations"].append("x"),
])
def test_validator_reports_instead_of_crashing(v, smoke, change):
    change(_first_station_resort(smoke))
    errs = v.validate(smoke, expected_count=None)
    assert errs and all(isinstance(e, str) for e in errs)


def test_health_flags_a_stale_file_only_when_asked(v, smoke):
    from datetime import datetime, timezone
    late = datetime(2026, 11, 29, tzinfo=timezone.utc)
    assert not any("stale" in p for p in v.health(smoke, now=late))
    assert any("stale" in p for p in v.health(smoke, max_age_h=13, now=late))
