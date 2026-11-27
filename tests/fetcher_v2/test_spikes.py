"""Meteocat spike filtering, run end to end.

Meteocat publishes half-hourly snow depth with no quality flag. The fetcher drops
a reading more than 20 cm from the median of that station's readings within
±60 minutes, and takes the latest depth as the median of the last hour.

The unmodified meteocat_batch runs in a subprocess against the recorded XEMA
payload (27 Nov 2026) through the local intercept proxy. Spike cases change one
recorded reading's value and nothing else.
"""

import json
import statistics
from datetime import datetime, timedelta

import pytest

from conftest import fixture_bytes, load_fixture

RESORTS = ["Baqueira-Beret", "Boí Taüll", "Vallter 2000", "Espot Esquí"]
CODE = {"Baqueira-Beret": "Z1", "Boí Taüll": "Z2", "Vallter 2000": "ZC"}


def rules(readings):
    return [
        {"host": r"transparenciacatalunya", "path": r"nzvn-apee", "mode": "replay",
         "body": json.dumps(readings).encode()},
        {"host": r"transparenciacatalunya", "path": r"yqwd-vj5e", "mode": "replay",
         "body": fixture_bytes("meteocat_stations.json")},
    ]


def spec_filter(readings, code):
    """The rule as documented in the fetcher, re-implemented from its comment."""
    rd = sorted((datetime.fromisoformat(x["data_lectura"][:19]), float(x["valor_lectura"]))
                for x in readings if x["codi_estacio"] == code and x.get("valor_lectura") not in (None, ""))
    rd = [(t, v) for t, v in rd if 0 <= v < 600]
    kept = [(t, v) for t, v in rd
            if abs(v - statistics.median([x for u, x in rd if abs(u - t) <= timedelta(minutes=60)])) <= 20]
    return rd, kept


def latest_depth(kept):
    t1 = kept[-1][0]
    return t1, statistics.median([v for t, v in kept if t >= t1 - timedelta(minutes=60)])


@pytest.fixture(scope="module")
def real_readings():
    return load_fixture("meteocat_readings.json")


@pytest.fixture
def run_meteocat(run_provider):
    def _run(readings):
        return run_provider("meteocat_batch", RESORTS, rules(readings))
    return _run


def test_recorded_payload_depths(run_meteocat, real_readings):
    res = run_meteocat(real_readings)
    assert res["status"] == {"Meteocat": "ok"}
    assert {e["host"] for e in res["_proxy_log"]} == {"analisi.transparenciacatalunya.cat"}
    for name, code in CODE.items():
        r = res["by_name"][name]
        assert len(r["stations"]) == 1, name
        s = r["stations"][0]
        rd, kept = spec_filter(real_readings, code)
        t1, v1 = latest_depth(kept)
        assert s["depth_cm"] == round(v1, 1), name
        assert s["observed_utc"] == t1.strftime("%Y-%m-%dT%H:%M:%SZ"), name
        dropped = len(rd) - len(kept)
        spike_notes = [n for n in r["notes"] if "sensor spikes" in n]
        assert len(spike_notes) == (1 if dropped else 0), name
    # Espot's station is rejected by the elevation window, so it has no row at all
    assert res["by_name"]["Espot Esquí"]["stations"] == []


def test_injected_spike_is_dropped_and_noted(run_meteocat, real_readings):
    base = run_meteocat(real_readings)["by_name"]["Baqueira-Beret"]["stations"][0]
    spiked = json.loads(json.dumps(real_readings))
    target = next(x for x in spiked if x["codi_estacio"] == "Z1" and x["data_lectura"].startswith("2026-11-26T12:00"))
    target["valor_lectura"] = str(float(target["valor_lectura"]) + 85)
    res = run_meteocat(spiked)
    r = res["by_name"]["Baqueira-Beret"]
    s = r["stations"][0]
    assert "Meteocat Bonaigua: 1 snow-depth readings dropped as sensor spikes" in r["notes"]
    assert s["depth_cm"] == base["depth_cm"]
    assert s["change_24h_cm"] == base["change_24h_cm"]
    assert s["change_48h_cm"] == base["change_48h_cm"]


def test_spike_in_newest_reading_does_not_become_depth(run_meteocat, real_readings):
    spiked = json.loads(json.dumps(real_readings))
    newest = max((x for x in spiked if x["codi_estacio"] == "Z2"), key=lambda x: x["data_lectura"])
    orig = float(newest["valor_lectura"])
    newest["valor_lectura"] = str(orig + 60)
    res = run_meteocat(spiked)
    r = res["by_name"]["Boí Taüll"]
    s = r["stations"][0]
    rd0, kept0 = spec_filter(real_readings, "Z2")
    rd, kept = spec_filter(spiked, "Z2")
    assert len(rd) - len(kept) == len(rd0) - len(kept0) + 1     # the recording already has real Z2 spikes
    assert f"Meteocat Boí: {len(rd) - len(kept)} snow-depth readings dropped as sensor spikes" in r["notes"]
    t1, v1 = latest_depth(kept)
    assert s["depth_cm"] == round(v1, 1)
    assert s["depth_cm"] < orig + 20


def test_out_of_range_values_never_used(run_meteocat, real_readings):
    broken = json.loads(json.dumps(real_readings))
    newest = max((x for x in broken if x["codi_estacio"] == "Z1"), key=lambda x: x["data_lectura"])
    newest["valor_lectura"] = "-9999"
    res = run_meteocat(broken)
    s = res["by_name"]["Baqueira-Beret"]["stations"][0]
    rd, kept = spec_filter(broken, "Z1")
    t1, v1 = latest_depth(kept)
    assert s["depth_cm"] == round(v1, 1)
    assert s["depth_cm"] >= 0
    assert s["observed_utc"] != newest["data_lectura"][:19] + "Z"


@pytest.mark.xfail(strict=True, reason=(
    "REAL BUG: a reading with no neighbour within ±60 min is its own median, so it always passes the "
    "spike filter. Vallter's Ulldeter (ZC) really reports like this: its newest recorded reading "
    "(27 Nov 09:00) is 17.5 hours after the one before. An 85 cm spike there is published as measured depth."))
def test_isolated_spike_is_not_published(run_meteocat, real_readings):
    spiked = json.loads(json.dumps(real_readings))
    zc = sorted((x for x in spiked if x["codi_estacio"] == "ZC"), key=lambda x: x["data_lectura"])
    newest, before = zc[-1], zc[-2]
    gap = (datetime.fromisoformat(newest["data_lectura"][:19]) - datetime.fromisoformat(before["data_lectura"][:19]))
    assert gap > timedelta(hours=1)       # the recorded data really is this sparse
    newest["valor_lectura"] = "85"        # every other ZC reading is 0 cm
    res = run_meteocat(spiked)
    s = res["by_name"]["Vallter 2000"]["stations"][0]
    assert s["depth_cm"] != 85.0
