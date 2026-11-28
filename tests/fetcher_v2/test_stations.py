"""Station acceptance rules (docs/depth_coverage_plan.md 0.1).

A station counts only within 15 km of the resort point and between 300 m below
the base and 300 m above the summit (base + 1,500 m when the summit is unknown).
A station that fails is dropped with a resort note, never kept with a caveat.

Real cases from the 27 Nov run: Vail's Copper Mountain SNOTEL (24.2 km), Kiroro's
Akaigawa JMA station (148 m against a 520 m base), Zao's Yamagata station, and
Espot's Meteocat Z7 (2,519 m against a 2,110 m summit).
"""

import copy
import math

import pytest

from conftest import fixture_bytes, load_fixture
from test_units import SNOTEL_RESORTS, snotel_run  # noqa: F401  (module-scoped replay run)


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def test_rule_constants(f, v):
    assert f.MAX_STATION_DISTANCE_KM == 15.0
    assert f.STATION_ELEV_BELOW_BASE_M == 300
    assert f.STATION_ELEV_ABOVE_TOP_M == 300
    assert v.MAX_STATION_DISTANCE_KM == f.MAX_STATION_DISTANCE_KM


def _row_at(f, resort, km_north, elev, provider="SNOTEL", sid="x", name="test"):
    lat = resort["lat"] + km_north / 111.195   # 1 degree of latitude on a 6,371 km sphere
    return f._station_row(provider, sid, name, lat, resort["lon"], elev, resort)


def test_distance_boundary(f, resorts_by_name):
    r = copy.deepcopy(resorts_by_name["Alta"])
    ok = _row_at(f, r, 14.96, r["base_m"])
    assert ok["distance_km"] == 15.0
    assert f._station_rule_failure(ok, r) is None
    far = _row_at(f, r, 15.1, r["base_m"])
    assert far["distance_km"] == 15.1
    msg = f._station_rule_failure(far, r)
    assert msg and "over 15 km, removed" in msg


def test_elevation_window_boundaries(f, resorts_by_name):
    r = copy.deepcopy(resorts_by_name["Alta"])    # 2,600 to 3,216 m
    lo, hi = r["base_m"] - 300, r["summit_m"] + 300
    assert f._station_rule_failure(_row_at(f, r, 1, lo), r) is None
    assert f._station_rule_failure(_row_at(f, r, 1, hi), r) is None
    below = f._station_rule_failure(_row_at(f, r, 1, lo - 1), r)
    above = f._station_rule_failure(_row_at(f, r, 1, hi + 1), r)
    assert "below the 2,600 m base" in below and "(limit 300 m), removed" in below
    assert "above the 3,216 m summit" in above


def test_unknown_summit_uses_base_plus_1500(f, resorts_by_name):
    r = copy.deepcopy(resorts_by_name["Alta"])
    r["summit_m"] = None
    top = r["base_m"] + 1500
    assert f._station_rule_failure(_row_at(f, r, 1, top + 300), r) is None
    assert "above the 4,100 m summit" in f._station_rule_failure(_row_at(f, r, 1, top + 301), r)


def test_unknown_station_elevation_checks_distance_only(f, resorts_by_name):
    r = copy.deepcopy(resorts_by_name["Alta"])
    assert f._station_rule_failure(_row_at(f, r, 1, None), r) is None
    assert f._station_rule_failure(_row_at(f, r, 20, None), r) is not None


@pytest.mark.parametrize("resort,sid,reason", [
    ("Kiroro", "16126", "372 m below the 520 m base"),
    ("Zao Onsen", "35426", "627 m below the 780 m base"),
])
def test_real_jma_stations_rejected(f, resorts_by_name, resort, sid, reason):
    meta = load_fixture("jma_amedastable_subset.json")[sid]
    r = copy.deepcopy(resorts_by_name[resort])
    lat = meta["lat"][0] + meta["lat"][1] / 60
    lon = meta["lon"][0] + meta["lon"][1] / 60
    row = f._station_row("JMA", sid, meta["enName"], lat, lon, meta["alt"], r)
    warnings = []
    assert f._accept_station(row, r, warnings) is False
    assert any(reason in n for n in r["notes"])


def test_real_meteocat_espot_rejected_above_summit(f, resorts_by_name):
    m = next(x for x in load_fixture("meteocat_stations.json") if x["codi_estacio"] == "Z7")
    r = copy.deepcopy(resorts_by_name["Espot Esquí"])
    row = f._station_row("Meteocat", "Z7", "Espot", float(m["latitud"]), float(m["longitud"]),
                         float(m["altitud"]), r)
    msg = f._station_rule_failure(row, r)
    assert msg == "Meteocat Z7 (Espot) at 2,519 m is 409 m above the 2,110 m summit (limit 300 m), removed"


def test_snotel_replay_applies_rules(snotel_run):  # noqa: F811
    meta = {m["stationTriplet"]: m for m in load_fixture("awdb_stations.json")}
    for r in snotel_run["resorts"]:
        kept = {s["id"] for s in r["stations"]}
        for s in r["stations"]:
            m = meta[s["id"]]
            d = _hav(r["lat"], r["lon"], m["latitude"], m["longitude"])
            assert s["distance_km"] == round(d, 1)
            assert s["distance_km"] <= 15.0
            assert r["base_m"] - 300 <= s["elevation_m"] <= r["summit_m"] + 300
        for note in r["notes"]:
            if "removed" in note:
                sid = note.split()[1]
                assert sid not in kept
    vail = snotel_run["by_name"]["Vail"]
    assert vail["stations"] == []
    assert "SNOTEL 415:CO:SNTL (Copper Mountain) is 24.2 km away, over 15 km, removed" in vail["notes"]
    assert "Breckenridge" in snotel_run["by_name"]


def test_snotel_name_mismatch_is_removed(run_provider):
    """AWDB metadata naming a different station than the config: the row is dropped, never relabelled."""
    import json
    meta = load_fixture("awdb_stations.json")
    for m in meta:
        if m["stationTriplet"] == "766:UT:SNTL":
            m["name"] = "Ben Lomond Peak"     # a real SNOTEL name, the old Alta misassignment
    rules = [
        {"host": r"wcc\.sc\.egov\.usda\.gov", "path": r"/stations", "mode": "replay",
         "body": json.dumps(meta).encode()},
        {"host": r"wcc\.sc\.egov\.usda\.gov", "path": r"/data", "mode": "replay",
         "body": fixture_bytes("awdb_daily.json")},
    ]
    res = run_provider("snotel_batch", ["Snowbird"], rules)
    sb = res["by_name"]["Snowbird"]
    assert sb["stations"] == []
    assert "SNOTEL 766:UT:SNTL is Ben Lomond Peak, not Snowbird, removed" in sb["notes"]


def test_snotel_provider_failure_leaves_nulls(run_provider):
    """AWDB down (404, which the fetcher does not retry): no rows, a note per station, status failed."""
    rules = [{"host": r"wcc\.sc\.egov\.usda\.gov", "path": r"", "mode": "status", "code": 404}]
    res = run_provider("snotel_batch", ["Alta"], rules)
    alta = res["by_name"]["Alta"]
    assert alta["stations"] == []
    assert res["status"] == {"SNOTEL": "failed"}
    assert res["warnings"] == ["SNOTEL (AWDB) request failed; no SNOTEL station data this run"]
    assert [n for n in alta["notes"] if "AWDB request failed" in n] == [
        "SNOTEL 1308:UT:SNTL not available this run (AWDB request failed)",
        "SNOTEL 766:UT:SNTL not available this run (AWDB request failed)"]
    assert len(res["_proxy_log"]) == 2       # metadata and data, one attempt each
