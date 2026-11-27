#!/usr/bin/env python3
"""Rebuild the JSON fixtures in this directory from recorded real payloads.

Sources, all recorded, none invented:
  - data/forecasts_v2.json (schema 3, written by scripts/fetch_data_v2.py on
    2026-11-27 12:27 UTC from live Open-Meteo / SNOTEL / GeoSphere / SLF /
    OnTheSnow responses).
  - `git show HEAD:data/forecasts_v2.json`, the old-schema file that is still
    on GitHub main (what data._read_github() returns today).

Each pathological variant copies real resort records and changes only the
field(s) a dashboard audit finding names; the change is listed in the file's
"_fixture_note". Files are small (a few resorts each) so the suite stays fast.
Snapshot files (snapshot_*.json) freeze real records whose behaviour the
invariant tests depend on, so a later fetcher run cannot make a test vacuous.

    python3 tests/dashboard_v2/fixtures/make_fixtures.py
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REAL = json.loads((ROOT / "data" / "forecasts_v2.json").read_text(encoding="utf-8"))
BY_ID = {r["id"]: r for r in REAL["resorts"] if isinstance(r, dict) and r.get("id")}


def wrap(resorts, note, **top):
    """The real file's own top level (schema, updated, run, thresholds,
    sources) around a list of resort records."""
    out = {k: copy.deepcopy(v) for k, v in REAL.items() if k != "resorts"}
    out.update(top)
    out["_fixture_note"] = note
    out["resorts"] = resorts
    return out


def rec(rid):
    return copy.deepcopy(BY_ID[rid])


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {name}")


def main():
    # Recorded: the old-schema file GitHub main still serves (deploy-order hazard).
    old = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:data/forecasts_v2.json"],
                         capture_output=True, text=True, check=True).stdout
    (HERE / "github_main_old_schema.json").write_text(old, encoding="utf-8")
    print("wrote github_main_old_schema.json")

    # Real top level, two real resorts, schema_version 2.
    write("schema_v2.json", wrap([rec("alta"), rec("vail")], "real records, schema_version set to 2",
                                 schema_version=2))

    # Truncated mid-file, as a dropped download would leave it.
    raw = json.dumps(wrap([rec("alta"), rec("vail")], "truncated at 40000 bytes"))
    (HERE / "truncated.json").write_text(raw[:40000], encoding="utf-8")
    print("wrote truncated.json")

    # render.py:181 / powder.py:97: invalid zone and an offset outside +-24 h.
    a = rec("alta")
    a["timezone"], a["utc_offset_s"] = "Bogus/Zone", 90000
    write("bad_tz_offset.json", wrap([a, rec("snowbird"), rec("vail")],
                                     "alta: timezone='Bogus/Zone', utc_offset_s=90000"))

    # data.py:391: null name plus two resorts without coordinates.
    a, v = rec("alta"), rec("vail")
    a["name"], a["lat"], v["lat"] = None, None, None
    write("null_name_no_coords.json", wrap([a, v, rec("snowbird")],
                                           "alta: name=None, lat=None; vail: lat=None"))

    # data.py:30: a non-string name.
    a = rec("alta")
    a["name"] = 12345
    write("int_name.json", wrap([a, rec("vail")], "alta: name=12345"))

    # app.py:73: a resorts list with no usable entry.
    write("junk_resorts.json", wrap([None, 5, {"name": "x"}], "resorts with no dict that has an id"))

    # render.py:805 (a): agree_through names a day missing from days.
    a = rec("alta")
    a["models"]["agree_through"] = 2
    a["days"] = [d for d in a["days"] if d.get("index") != 2]
    write("agree_through_missing_day.json", wrap([a], "alta: agree_through=2, day index 2 removed"))

    # render.py:805 (b): one model has no day-0 value; the fetcher then writes
    # agree_through=null (fetch_data_v2.py:2660-2662).
    a = rec("alta")
    for row in a["models"]["days"]:
        if row.get("index") == 0:
            row["snowfall_cm"]["ecmwf_ifs025"] = None
    a["models"]["agree_through"] = None
    write("ecmwf_day0_null.json", wrap([a], "alta: models day 0 ecmwf_ifs025=None, agree_through=None"))

    # Honest nulls: every number the resort page and tables quote is null.
    a = rec("alta")
    a["name"] = "Alta (nulls)"
    a["id"] = "alta-nulls"
    a["depth"] = {**(a.get("depth") or {}), "value_cm": None}
    for key in ("next_72h", "week", "extended"):
        blk = a.get(key) or {}
        for k in list(blk):
            if k.endswith("_cm"):
                blk[k] = None
    for seglist in (a["next_72h"].get("by_day") or {}).values():
        for s in seglist:
            s["snowfall_cm"] = None
    for d in a["days"]:
        for band in ("base", "mid", "summit"):
            if isinstance(d.get(band), dict):
                d[band]["snowfall_cm"] = None
    a["recent"] = {"measured": None, "modelled": None}
    for s in a.get("stations") or []:
        s["depth_cm"] = None
        s["change_24h_cm"] = None
    write("honest_nulls.json", wrap([a, rec("alta"), rec("vail")],
                                    "alta-nulls: every snow amount and depth set to null"))

    # Snapshots of real records the invariant tests lean on.
    write("snapshot_whitefish.json", wrap([rec("whitefish")],
          "real record: best day 0 by the model, resort reported 0 cm (measured refutes)"))
    write("snapshot_base_gate.json", wrap([rec("niseko-united"), rec("astun"), rec("alta"), rec("solden")],
          "real records: thin measured base, no depth, deep resort-reported base, 22 cm measured base"))


if __name__ == "__main__":
    main()
