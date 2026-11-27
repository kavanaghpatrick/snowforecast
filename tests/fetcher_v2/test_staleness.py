"""Staleness (docs/depth_coverage_plan.md 0.4): an old reading stays in the station
row but never becomes the resort's depth or recent snow, and the resort gets a note.

Real cases from the 27 Nov 12:15 UTC run: Les Arcs' manual Météo-France reading
from 20 Nov (7 days old) and Courchevel's from 23 Nov (4 days old).
"""

import copy
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(scope="module")
def run_started(real_run_doc):
    return datetime.fromisoformat(real_run_doc["run"]["started_utc"].replace("Z", "+00:00"))


def _cfg_type(resort, provider, sid):
    for p, s, _label, stype in resort["station_cfg"]:
        if p == provider and s == sid:
            return stype
    return None


def _rows_from_real(f, resort, real_resort):
    """Rebuild the fetcher's in-memory station rows from the real file's rows (same values)."""
    rows = []
    for s in real_resort["stations"]:
        # A station configured at run time but since removed from the config (Baqueira's YT) ranks last.
        rank = next((i for i, c in enumerate(resort["station_cfg"]) if c[0] == s["provider"] and c[1] == s["id"]),
                    99)
        stype = _cfg_type(resort, s["provider"], s["id"])
        if s["provider"] == "SLF":
            stype = stype or "imis"
        key = f"{s['provider']}:{stype}" if stype else None
        row = f._station_row(s["provider"], s["id"], s["name"], s["lat"], s["lon"], s["elevation_m"],
                             resort, rank, key)
        for k, val in s.items():
            if k not in ("lat", "lon", "distance_km"):
                row[k] = val
        rows.append(row)
    return rows


def test_every_provider_has_an_age_limit(f, resorts_by_name):
    keys = set(f.STATION_MAX_AGE_H)
    for r in resorts_by_name.values():
        for p, _sid, _label, stype in r["station_cfg"]:
            if p == "SLF":
                stype = stype or "imis"
            key = f"{p}:{stype}" if stype else p
            assert key in keys or p in keys, (r["name"], key)
    for p in f.STATION_PROVIDERS:
        assert p in keys or any(k.startswith(p + ":") for k in keys), p


def test_les_arcs_manual_reading_dropped_auto_used(f, resorts_by_name, real_run_doc, run_started):
    real = next(x for x in real_run_doc["resorts"] if x["id"] == "les-arcs")
    r = copy.deepcopy(resorts_by_name["Les Arcs"])
    r["notes"] = []
    r["stations"] = _rows_from_real(f, r, real)
    f.mark_freshness(r, run_started)
    by = {s["id"]: s for s in r["stations"]}
    assert by["73054401"]["_fresh"] is False          # manual, 20 Nov 06:00
    assert by["73150004"]["_fresh"] is True           # automatic, 27 Nov 04:00
    assert "Meteo-France Les Arcs reading is 7 days old, not used" in r["notes"]
    best = f._best_station(r)
    assert best["id"] == "73150004"
    assert by["73054401"]["depth_cm"] == 60.0          # the old value stays in the row
    d = f.choose_depth(r)
    assert d["value_cm"] == real["depth"]["value_cm"] == 99.0
    assert d["source_label"] == real["depth"]["source_label"]


def test_courchevel_four_day_old_reading(f, resorts_by_name, real_run_doc, run_started):
    real = next(x for x in real_run_doc["resorts"] if x["id"] == "courchevel")
    r = copy.deepcopy(resorts_by_name["Courchevel"])
    r["notes"] = []
    r["stations"] = _rows_from_real(f, r, real)
    f.mark_freshness(r, run_started)
    assert "Meteo-France Courchevel reading is 4 days old, not used" in r["notes"]
    best = f._best_station(r)
    assert best is not None and best["id"] == "73015400"


def test_real_file_depth_never_from_a_stale_station(f, resorts_by_name, real_run_doc, run_started):
    checked = 0
    for real in real_run_doc["resorts"]:
        if real["depth"]["kind"] != "measured":
            continue
        r = copy.deepcopy(resorts_by_name[real["name"]])
        r["notes"] = []
        r["stations"] = _rows_from_real(f, r, real)
        f.mark_freshness(r, run_started)
        best = f._best_station(r)
        assert best is not None, real["id"]
        assert best["depth_cm"] == real["depth"]["value_cm"], real["id"]
        assert best["observed_utc"] == real["depth"]["observed_utc"], real["id"]
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("key", ["JMA", "Meteo-France:auto", "Meteo-France:manual", "Meteocat", "Tirol",
                                 "GeoSphere", "SLF:imis", "SLF:study_plot", "AEMET"])
def test_age_limit_boundary(f, resorts_by_name, run_started, key):
    limit = f.STATION_MAX_AGE_H[key]
    provider = key.split(":")[0]
    for age_h, fresh in ((limit, True), (limit + 1 / 60, False)):
        r = copy.deepcopy(resorts_by_name["Zermatt"])
        r["notes"] = []
        row = f._station_row(provider, "s1", "Test", r["lat"], r["lon"], r["base_m"], r, 0, key)
        f._set_latest(row, run_started - timedelta(hours=age_h), 50.0)
        r["stations"] = [row]
        f.mark_freshness(r, run_started)
        assert row["_fresh"] is fresh, (key, age_h)
        assert (not r["notes"]) is fresh


def test_snotel_date_only_limit(f, resorts_by_name):
    run = datetime(2026, 11, 27, 23, 59, tzinfo=timezone.utc)
    for days_old, fresh in ((0, True), (2, True), (3, False)):
        r = copy.deepcopy(resorts_by_name["Alta"])
        r["notes"] = []
        row = f._station_row("SNOTEL", "1308:UT:SNTL", "Atwater", r["lat"], r["lon"], r["base_m"], r, 0)
        row["depth_cm"] = 40.6
        row["observed_date"] = (run.date() - timedelta(days=days_old)).isoformat()
        r["stations"] = [row]
        f.mark_freshness(r, run)
        assert row["_fresh"] is fresh, days_old
    assert "SNOTEL Atwater reading is 3 days old, not used" in r["notes"]


def test_row_without_any_reading_is_not_fresh_and_not_noted(f, resorts_by_name, run_started):
    r = copy.deepcopy(resorts_by_name["Zermatt"])
    r["notes"] = []
    row = f._station_row("SLF", "ZER2", "Test", r["lat"], r["lon"], r["base_m"], r, 0, "SLF:imis")
    r["stations"] = [row]
    f.mark_freshness(r, run_started)
    assert row["_fresh"] is False
    assert r["notes"] == []
    assert f.choose_depth(r)["value_cm"] is None
    assert f.recent_measured(r) is None


def test_stale_station_gives_no_recent_snow(f, resorts_by_name, run_started):
    r = copy.deepcopy(resorts_by_name["Zermatt"])
    r["notes"] = []
    row = f._station_row("SLF", "ZER2", "Test", r["lat"], r["lon"], r["base_m"], r, 0, "SLF:imis")
    f._set_latest(row, run_started - timedelta(hours=7), 80.0)
    row["new_24h_cm"] = 12.0
    r["stations"] = [row]
    f.mark_freshness(r, run_started)
    assert f.recent_measured(r) is None
    assert f.choose_depth(r)["value_cm"] is None


def test_depth_change_never_labelled_new_snow(f, resorts_by_name, run_started):
    r = copy.deepcopy(resorts_by_name["Alta"])
    r["notes"] = []
    row = f._station_row("SNOTEL", "1308:UT:SNTL", "Atwater", r["lat"], r["lon"], r["base_m"], r, 0)
    row.update({"depth_cm": 40.6, "observed_date": run_started.date().isoformat(), "change_24h_cm": 10.2})
    r["stations"] = [row]
    f.mark_freshness(r, run_started)
    m = f.recent_measured(r)
    assert m["kind"] == "depth_change"
    assert "New snow" not in m["label"]
    assert m["h24_cm"] == 10.2
