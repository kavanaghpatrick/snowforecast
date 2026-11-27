"""The fetcher's HTTP layer against a real local server.

Failures are produced by a real socket server on 127.0.0.1 (see _replay.py), not
by patching: 5xx, 404, 429, a body cut off mid-transfer (with and without a
Content-Length), an HTML error page with status 200, Open-Meteo's own recorded
error object (the "daily limit exceeded" answer this machine got on 27 Nov), and
a server slower than the timeout. Every success body is a recorded payload.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from _replay import ReplayServer
from conftest import HERE, REPO, RUN_UTC, fixture_bytes, load_fixture

DAILY = fixture_bytes("om_daily_probe.json")


def fetch(f, url, attempts=3, want_json=True, timeout_s=5, wait_on_429=None, counter=None):
    c = counter or f.HttpCounter()
    body, info = f._fetch(url, attempts=attempts, backoff_s=[0], timeout_s=timeout_s, headers=None,
                          counter=c, want_json=want_json, wait_on_429=wait_on_429)
    return body, info, c


def test_replay_ok_and_user_agent(f):
    with ReplayServer([{"path": "/d", "mode": "replay", "body": DAILY}]) as s:
        body, final, c = fetch(f, s.url + "/d")
        ua = s.log[0]["ua"]
    assert body == json.loads(DAILY)
    assert final == s.url + "/d"
    assert (c.requests, c.retries) == (1, 0)
    assert ua == f.API_UA


def test_5xx_retried_then_ok(f):
    with ReplayServer([{"path": "/d", "mode": "fail_first", "n": 2, "fail_code": 502, "body": DAILY}]) as s:
        body, _, c = fetch(f, s.url + "/d", attempts=3)
        n = len(s.log)
    assert body == json.loads(DAILY)
    assert (n, c.requests, c.retries) == (3, 3, 2)


def test_5xx_exhausts_attempts(f):
    with ReplayServer([{"path": "/d", "mode": "status", "code": 503}]) as s:
        body, info, c = fetch(f, s.url + "/d", attempts=3)
        n = len(s.log)
    assert body is None and info == "HTTP 503" and n == 3


def test_404_not_retried(f):
    with ReplayServer([{"path": "/d", "mode": "status", "code": 404}]) as s:
        body, info, c = fetch(f, s.url + "/d", attempts=5)
        n = len(s.log)
    assert body is None and info == "HTTP 404" and n == 1


def test_429_retried(f):
    with ReplayServer([{"path": "/d", "mode": "fail_first", "n": 1, "fail_code": 429, "body": DAILY}]) as s:
        body, _, c = fetch(f, s.url + "/d", attempts=2, wait_on_429=0)
    assert body == json.loads(DAILY) and c.retries == 1


@pytest.mark.parametrize("no_length", [False, True])
def test_truncated_body_is_a_failure(f, no_length):
    with ReplayServer([{"path": "/d", "mode": "truncate", "frac": 0.5, "no_length": no_length, "body": DAILY}]) as s:
        body, info, c = fetch(f, s.url + "/d", attempts=2)
        n = len(s.log)
    assert body is None
    assert n == 2
    assert ("IncompleteRead" in info) if not no_length else ("JSONDecodeError" in info or "Expecting" in info)


def test_html_error_page_with_200_is_a_failure(f):
    with ReplayServer([{"path": "/d", "mode": "garbage", "text": "<html><body>Service Unavailable</body></html>"}]) as s:
        body, info, _ = fetch(f, s.url + "/d", attempts=1)
    assert body is None and "JSONDecodeError" in info


def test_open_meteo_error_object_is_a_failure(f):
    err = fixture_bytes("om_error_daily_limit.json")
    with ReplayServer([{"path": "/d", "mode": "replay", "body": err}]) as s:
        body, info, _ = fetch(f, s.url + "/d", attempts=2)
    assert body is None
    assert "Daily API request limit exceeded" in info


def test_slow_server_times_out(f):
    with ReplayServer([{"path": "/d", "mode": "slow", "delay_s": 3, "body": DAILY}]) as s:
        t0 = time.monotonic()
        body, info, _ = fetch(f, s.url + "/d", attempts=1, timeout_s=0.5)
        el = time.monotonic() - t0
    assert body is None
    assert el < 2.5
    assert "timed out" in info.lower() or "Timeout" in info


def test_text_and_bytes_modes(f):
    gz = fixture_bytes("forecasts_v2_replay_smoke.json.gz")
    with ReplayServer([{"path": "/t", "mode": "replay", "body": b"2026-11-27T22:00:00+09:00\n", "ctype": "text/plain"},
                       {"path": "/b", "mode": "replay", "body": gz, "ctype": "application/gzip"}]) as s:
        text, _ = f.http_text(s.url + "/t", attempts=1, backoff_s=[0], timeout_s=5)
        raw, _ = f.http_bytes(s.url + "/b", attempts=1, backoff_s=[0], timeout_s=5)
    assert text == "2026-11-27T22:00:00+09:00\n"
    assert raw == gz


def test_om_request_length_mismatch_is_rejected(f):
    one_point = json.dumps(load_fixture("om_daily_probe.json")[0]).encode()
    with ReplayServer([{"path": "/v1/forecast", "mode": "replay", "body": one_point}]) as s:
        assert f._om_request(s.url + "/v1/forecast?latitude=1,2&longitude=1,2", 2, attempts=1) is None
        # one point asked, one object answered: wrapped into a list
        got = f._om_request(s.url + "/v1/forecast?latitude=1&longitude=1", 1, attempts=1)
    assert isinstance(got, list) and len(got) == 1


# ----------------------------------------------------------------------------- Open-Meteo batching


def _run_om(rules, names, kind="daily"):
    tmpdir = tempfile.TemporaryDirectory()
    out = Path(tmpdir.name) / f"om_{kind}.json"
    with ReplayServer(rules) as s:
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
        env["OPEN_METEO_URL"] = s.url + "/v1/forecast"
        env["NO_PROXY"] = "127.0.0.1"
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, str(HERE / "_provider_driver.py"), f"openmeteo_{kind}",
                            ",".join(names), RUN_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)],
                           env=env, capture_output=True, text=True, timeout=120, cwd=str(REPO))
        el = time.monotonic() - t0
        log = list(s.log)
    try:
        assert p.returncode == 0, p.stderr[-2000:]
        res = json.loads(out.read_text())
    finally:
        tmpdir.cleanup()
    res["_log"] = log
    res["_elapsed"] = el
    return res


def test_openmeteo_batch_maps_points_to_bands(f):
    pts = load_fixture("om_daily_probe.json")
    res = _run_om([{"path": "/v1/forecast", "mode": "replay", "om": pts}], ["Alta", "Snowbird", "Zermatt"])
    assert res["failed_chunks"] == []
    assert len(res["_log"]) == 1
    q = res["_log"][0]["query"]
    # batches follow the fetcher's resort order, not the order asked for
    order = [r for r in f.build_resorts() if r["name"] in ("Alta", "Snowbird", "Zermatt")]
    flat = [(r["id"], lab, e) for r in order for lab, e in f._om_points(r)]
    assert len(flat) == 9
    assert "elevation=" + ",".join(str(e) for _, _, e in flat) + "&" in q
    om = res["om"]
    assert set(om) == {"alta", "snowbird", "zermatt"}
    # the i-th point of the answer goes to the i-th (resort, band) of the request
    for i, (rid, lab, _) in enumerate(flat):
        assert om[rid][lab] == pts[i % len(pts)], (rid, lab)


def test_openmeteo_failed_batch_falls_back_per_resort():
    pts = load_fixture("om_daily_probe.json")
    one_point = json.dumps(pts[0]).encode()
    rules = [
        # the 9-point batch call answers with one object: wrong length, so the batch fails at once
        {"path": r"latitude=[^&]*(,[^&]*){8}&", "mode": "replay", "body": one_point},
        {"path": "/v1/forecast", "mode": "replay", "om": pts},
    ]
    res = _run_om(rules, ["Alta", "Snowbird", "Zermatt"])
    assert res["failed_chunks"] == ["daily:0"]
    assert len(res["_log"]) == 4                     # 1 batch + 3 single-resort calls
    assert set(res["om"]) == {"alta", "snowbird", "zermatt"}
    for r in res["resorts"]:
        assert not any("call failed" in n for n in r["notes"])


def test_openmeteo_resort_failure_is_noted_not_filled():
    pts = load_fixture("om_daily_probe.json")
    err = fixture_bytes("om_error_daily_limit.json")
    rules = [
        {"path": r"latitude=[^&]*(,[^&]*){8}&", "mode": "replay", "body": json.dumps(pts[0]).encode()},
        # Zermatt (lat 45.9872) gets Open-Meteo's recorded daily-limit error on its own call
        {"path": r"latitude=45\.9872", "mode": "replay", "body": err},
        {"path": "/v1/forecast", "mode": "replay", "om": pts},
    ]
    res = _run_om(rules, ["Alta", "Snowbird", "Zermatt"])
    assert "daily:zermatt" in res["failed_chunks"]
    assert "zermatt" not in res["om"]
    z = next(r for r in res["resorts"] if r["name"] == "Zermatt")
    assert "Open-Meteo daily call failed for this resort" in z["notes"]
