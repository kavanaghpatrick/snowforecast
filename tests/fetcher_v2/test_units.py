"""Unit conversions and rounding.

SNOTEL (AWDB) reports depth and snow water in inches and elevation in feet. The
conversion is checked end to end: the unmodified snotel_batch runs in a
subprocess against the recorded AWDB payloads through the local intercept proxy.
"""

import math

import pytest

from conftest import RUN_UTC, fixture_bytes, load_fixture

SNOTEL_RESORTS = ["Stevens Pass", "Crystal Mountain", "Vail", "Breckenridge", "Snowbird", "Alta", "Big Sky"]


def awdb_rules():
    return [
        {"host": r"wcc\.sc\.egov\.usda\.gov", "path": r"/stations", "mode": "replay",
         "body": fixture_bytes("awdb_stations.json")},
        {"host": r"wcc\.sc\.egov\.usda\.gov", "path": r"/data", "mode": "replay",
         "body": fixture_bytes("awdb_daily.json")},
    ]


@pytest.fixture(scope="module")
def snotel_run(tmp_path_factory, request):
    # module-scoped wrapper around the function-scoped run_provider
    import json
    import os
    import subprocess
    import sys

    from conftest import HERE, REPO
    from _replay import InterceptProxy

    tmp = tmp_path_factory.mktemp("snotel")
    out = tmp / "out.json"
    with InterceptProxy(awdb_rules(), str(tmp)) as proxy:
        p = subprocess.run([sys.executable, str(HERE / "_provider_driver.py"), "snotel_batch",
                            ",".join(SNOTEL_RESORTS), RUN_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)],
                           env=proxy.env(dict(os.environ)), capture_output=True, text=True, timeout=120,
                           cwd=str(REPO))
        log = list(proxy.log)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(out.read_text())
    res["_proxy_log"] = log
    res["by_name"] = {r["name"]: r for r in res["resorts"]}
    return res


def _awdb_series():
    out = {}
    for item in load_fixture("awdb_daily.json"):
        el = {}
        for b in item["data"]:
            el[b["stationElement"]["elementCode"]] = {v["date"][:10]: v["value"] for v in b["values"]
                                                      if v.get("value") is not None}
        out[item["stationTriplet"]] = el
    return out


def test_snotel_only_awdb_was_contacted(snotel_run):
    hosts = {e.get("host") for e in snotel_run["_proxy_log"]}
    assert hosts == {"wcc.sc.egov.usda.gov"}
    assert snotel_run["status"] == {"SNOTEL": "ok"}


def test_snotel_depth_inches_to_cm(snotel_run):
    series = _awdb_series()
    rows = [s for r in snotel_run["resorts"] for s in r["stations"]]
    assert rows
    for s in rows:
        snwd = series[s["id"]]["SNWD"]
        latest = max(snwd)
        assert s["observed_date"] == latest
        assert s["depth_cm"] == round(snwd[latest] * 2.54, 1)
        assert s["observed_utc"] is None            # AWDB daily values carry a date only


def test_snotel_swe_inches_to_mm(snotel_run):
    series = _awdb_series()
    for r in snotel_run["resorts"]:
        for s in r["stations"]:
            w = series[s["id"]].get("WTEQ", {})
            latest = s["observed_date"]
            if latest in w:
                assert s["swe_mm"] == round(w[latest] * 25.4, 1)
                prev = sorted(w)[sorted(w).index(latest) - 1]
                assert s["swe_change_24h_mm"] == round((w[latest] - w[prev]) * 25.4, 1)


def test_snotel_elevation_feet_to_m(snotel_run):
    meta = {m["stationTriplet"]: m for m in load_fixture("awdb_stations.json")}
    for r in snotel_run["resorts"]:
        for s in r["stations"]:
            assert s["elevation_m"] == int(math.floor(meta[s["id"]]["elevation"] * 0.3048 + 0.5))


def test_snotel_depth_change_is_a_difference_not_new_snow(snotel_run):
    series = _awdb_series()
    for r in snotel_run["resorts"]:
        for s in r["stations"]:
            snwd = series[s["id"]]["SNWD"]
            days = sorted(snwd)
            latest = days[-1]
            from datetime import date, timedelta
            prev = (date.fromisoformat(latest) - timedelta(days=1)).isoformat()
            exp = round(snwd[latest] * 2.54 - snwd[prev] * 2.54, 1) if prev in snwd else None
            assert s["change_24h_cm"] == exp
            assert s["new_24h_cm"] is None          # SNOTEL has no new-snow sensor


def test_snotel_series_is_15_days_with_nulls_not_filled(snotel_run):
    series = _awdb_series()
    for r in snotel_run["resorts"]:
        for s in r["stations"]:
            assert len(s["series"]) == 15
            assert s["series"][-1]["date"] == RUN_UTC.date().isoformat()
            snwd = series[s["id"]]["SNWD"]
            for row in s["series"]:
                if row["date"] in snwd:
                    assert row["depth_cm"] == round(snwd[row["date"]] * 2.54, 1)
                else:
                    assert row["depth_cm"] is None      # the run day has no daily value yet


# ----------------------------------------------------------------------------- small helpers


@pytest.mark.parametrize("x,exp", [(None, None), (2.25, 2.2), (2.35, 2.4), ("3.14159", 3.1), (-0.04, -0.0)])
def test_r1(f, x, exp):
    assert f.r1(x) == exp


@pytest.mark.parametrize("x,exp", [(None, None), (0.5, 1), (1.5, 2), (2.5, 3), (-0.5, 0), (-1.5, -1), (59.5, 60)])
def test_rint_is_half_up(f, x, exp):
    assert f.rint(x) == exp


def test_haversine_known_distance(f):
    # Alta base to Snowbird base, both from the fetcher's own resort list
    R = {r["name"]: r for r in f.build_resorts()}
    a, s = R["Alta"], R["Snowbird"]
    d = f.haversine_km(a["lat"], a["lon"], s["lat"], s["lon"])
    # independent spherical-law-of-cosines computation, same Earth radius
    p1, p2 = math.radians(a["lat"]), math.radians(s["lat"])
    dl = math.radians(s["lon"] - a["lon"])
    exp = 6371.0 * math.acos(math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl))
    assert d == pytest.approx(exp, rel=1e-6)
    assert 1.0 < d < 5.0


def test_slugify_folds_accents(f):
    assert f.slugify("Candanchú") == "candanchu"
    assert f.slugify("Boí Taüll") == "boi-taull"
    assert f.slugify("St. Anton") == "st-anton"


def test_ids_unique_and_slugged(f):
    R = f.build_resorts()
    assert len({r["id"] for r in R}) == len(R)
    assert all(r["id"] == f.slugify(r["name"]) for r in R)


def test_mid_is_rounded_mean_of_real_heights(f):
    for r in f.build_resorts():
        if r["summit_m"] is None:
            assert r["mid_m"] is None
        else:
            assert r["mid_m"] == int(math.floor((r["base_m"] + r["summit_m"]) / 2 + 0.5))


def test_om_units_requested_explicitly(f):
    """Amounts are never converted in the fetcher, so the request must pin the units."""
    url = f._om_url("hourly", [40.0], [-111.0], [2600])
    for part in ("wind_speed_unit=kmh", "precipitation_unit=mm", "temperature_unit=celsius"):
        assert part in url
    assert "elevation=2600" in url
