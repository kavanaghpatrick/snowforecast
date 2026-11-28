"""write_atomic: the output file is replaced in one step or not at all.

Real files on disk, a real concurrent reader thread, and the recorded smoke file
as the payload.
"""

import json
import math
import threading

import pytest


def test_round_trip_and_no_temp_left(f, smoke, tmp_path):
    out = tmp_path / "data" / "forecasts_v2.json"
    f.write_atomic(smoke, out)
    assert json.loads(out.read_text(encoding="utf-8")) == smoke
    assert sorted(p.name for p in out.parent.iterdir()) == ["forecasts_v2.json"]


def test_compact_utf8_output(f, smoke, tmp_path):
    out = tmp_path / "forecasts_v2.json"
    f.write_atomic(smoke, out)
    text = out.read_text(encoding="utf-8")
    assert '": ' not in text[:2000] and ", " not in text[:200]     # compact separators
    assert "Vallter 2000" in text
    assert "\\u00" not in text                                     # accents written as UTF-8


def test_nan_refused_and_old_file_kept(f, smoke, tmp_path):
    out = tmp_path / "forecasts_v2.json"
    f.write_atomic(smoke, out)
    before = out.read_bytes()
    bad = json.loads(before)
    bad["resorts"][0]["days"][5]["summit"]["snowfall_cm"] = math.nan
    with pytest.raises(ValueError):
        f.write_atomic(bad, out)
    assert out.read_bytes() == before


@pytest.mark.xfail(strict=True, reason=(
    "REAL BUG (minor): when json.dump fails part-way (NaN, or a full disk), write_atomic leaves a "
    "partial forecasts_v2.json.tmp next to the real file. The workflow only adds data/forecasts_v2.json, "
    "so it is debris, not a corrupt publish."))
def test_failed_write_leaves_no_partial_temp(f, smoke, tmp_path):
    out = tmp_path / "forecasts_v2.json"
    bad = json.loads(json.dumps(smoke))
    bad["resorts"][-1]["days"][5]["summit"]["snowfall_cm"] = math.inf
    with pytest.raises(ValueError):
        f.write_atomic(bad, out)
    assert not (tmp_path / "forecasts_v2.json.tmp").exists()


def test_reader_never_sees_a_partial_file(f, smoke, tmp_path):
    out = tmp_path / "forecasts_v2.json"
    f.write_atomic(smoke, out)
    a = smoke
    b = json.loads(json.dumps(smoke))
    b["updated"] = "2026-11-27T23:59:59Z"
    stop = threading.Event()
    errors, reads = [], [0]

    def reader():
        while not stop.is_set():
            try:
                with open(out, encoding="utf-8") as fh:
                    doc = json.load(fh)
                assert doc["updated"] in (a["updated"], b["updated"])
                reads[0] += 1
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(40):
            f.write_atomic(b if i % 2 else a, out)
    finally:
        stop.set()
        t.join()
    assert errors == []
    assert reads[0] > 0

