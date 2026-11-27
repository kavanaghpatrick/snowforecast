"""Likely range for the "Snow, next 3 days" hero, from the backtest's own misses.

The page's hero is the model's number and stays the model's number. This script
builds the disclosed range printed beside it ("likely 9-24 cm"): for each
forecast size and country, the middle two thirds (17th to 83rd percentile) of
measured / forecast over real 3-day forecasts issued last winter.

Input: data/validation/out/pairs.csv (written by analyze.py from the cached
previous-runs archive). The 3-day pairs are rebuilt exactly as analyze.py's
next3_by_issue: for each station and issue day, the forecast is the day-0, day-1
and day-2 forecasts at their own lead, summed; measured is the station's new snow
over the same three local days.

Outputs (deterministic, no timestamps, so a rerun is byte-identical):
  dashboard_v2/skill_ranges.json        the table the page reads
  data/validation/out/hero_ranges.json  the same plus leave-one-station-out coverage

A cell needs 30 blocks and 5 stations, else the bucket's pooled value is used.
Countries the backtest never covered (Switzerland, Japan, Austria for leads)
get no range: the page shows none rather than borrow another country's misses.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from common import OUT, ROOT

PAIRS = OUT / "pairs.csv"
PAGE_TABLE = ROOT / "dashboard_v2" / "skill_ranges.json"
FULL = OUT / "hero_ranges.json"

LO_Q, HI_Q = 17.0, 83.0           # "likely": about two in three
MIN_FORECAST_CM = 5.0             # below this the ratio is unstable (a quarter of blocks measure 0)
BUCKETS = [(5.0, 10.0), (10.0, 20.0), (20.0, None)]
TESTED = ("USA", "France", "Spain")
MIN_BLOCKS, MIN_STATIONS = 30, 5


def blocks():
    rows = list(csv.DictReader(PAIRS.open()))
    L = [r for r in rows if r["kind"] == "local_day" and all(r[f"lead{n}"] != "" for n in range(8))]
    by = defaultdict(dict)
    for r in L:
        by[r["station"]][r["t0"][:10]] = r
    out = []
    for st, dm in by.items():
        for iss in sorted(dm):
            d0 = date.fromisoformat(iss)
            wk = [dm.get((d0 + timedelta(days=k)).isoformat()) for k in range(3)]
            if all(wk):
                out.append({"station": st, "country": wk[0]["country"], "issue": iss,
                            "pred": sum(float(wk[k][f"lead{k}"]) for k in range(3)),
                            "meas": sum(float(wk[k]["meas"]) for k in range(3))})
    return out


def bucket_of(v):
    for i, (lo, hi) in enumerate(BUCKETS):
        if v >= lo and (hi is None or v < hi):
            return i
    return None


def cell(ps):
    r = np.array([p["meas"] / p["pred"] for p in ps])
    return {"n_blocks": len(ps), "n_stations": len({p["station"] for p in ps}),
            "lo_ratio": round(float(np.percentile(r, LO_Q)), 3),
            "median_ratio": round(float(np.percentile(r, 50)), 3),
            "hi_ratio": round(float(np.percentile(r, HI_Q)), 3)}


def build(bs):
    ev = [b for b in bs if b["pred"] >= MIN_FORECAST_CM and b["country"] in TESTED]
    table = {}
    for i, (lo, hi) in enumerate(BUCKETS):
        inb = [b for b in ev if bucket_of(b["pred"]) == i]
        pooled = cell(inb)
        for c in TESTED:
            own = [b for b in inb if b["country"] == c]
            ok = len(own) >= MIN_BLOCKS and len({b["station"] for b in own}) >= MIN_STATIONS
            entry = dict(cell(own), basis="country") if ok else dict(pooled, basis="pooled USA+France+Spain")
            table.setdefault(c, []).append({"min_cm": lo, "max_cm": hi, **entry})
    return table, ev


def lookup(table, country, v):
    i = bucket_of(v)
    if country not in table or i is None:
        return None
    return table[country][i]


def cold_storm_day1():
    """Section 1.2 of docs/model_improvements.md: is the cold-storm excess a typical day
    or a few big days? Day-1 forecast >= 3 cm, binned by the model's snowfall-weighted
    temperature at the station (tw), as analyze.py's ratio_by_tw_bin_lead1."""
    rows = list(csv.DictReader(PAIRS.open()))
    L = [r for r in rows if r["kind"] == "local_day" and all(r[f"lead{n}"] != "" for n in range(8))
         and r["tw"] != "" and float(r["lead1"]) >= 3]
    out = {}
    for lo, hi in [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]:
        s = [(float(r["lead1"]), float(r["meas"])) for r in L if lo <= float(r["tw"]) < hi]
        if not s:
            continue
        rr = np.array([m / f for f, m in s])
        out[f"{lo}..{hi}"] = {"n": len(s), "ratio_of_totals": round(sum(m for _, m in s) / sum(f for f, _ in s), 3),
                              "median_day_ratio": round(float(np.median(rr)), 3),
                              "share_measured_above_forecast": round(float(np.mean([m > f for f, m in s])), 3)}
    return out


def main():
    bs = blocks()
    table, ev = build(bs)
    # Leave-one-station-out: would the range have held a station it never saw?
    cover = defaultdict(lambda: [0, 0])
    for st in sorted({b["station"] for b in ev}):
        t, _ = build([b for b in bs if b["station"] != st])
        for b in ev:
            if b["station"] != st:
                continue
            c = lookup(t, b["country"], b["pred"])
            inside = c["lo_ratio"] * b["pred"] <= b["meas"] <= c["hi_ratio"] * b["pred"]
            for key in (b["country"], "all"):
                cover[key][0] += inside
                cover[key][1] += 1
    coverage = {k: {"inside": v[0], "n": v[1], "share": round(v[0] / v[1], 3)} for k, v in sorted(cover.items())}
    page = {
        "what": "Likely range for the 3-day snow hero: the 17th to 83rd percentile of measured / forecast "
                "over real 3-day forecasts (day 0-2 at their own lead), by forecast size and country.",
        "built_by": "scripts/validation/hero_ranges.py from data/validation/out/pairs.csv",
        "backtest": "1 Dec 2025 - 31 Mar 2026, Open-Meteo previous-runs archive, best_match, station height; "
                    "38 stations (SNOTEL, Meteo-France automatic, Meteocat). 3-day blocks overlap by issue day.",
        "quantiles": [LO_Q, HI_Q],
        "min_forecast_cm": MIN_FORECAST_CM,
        "countries": table,
    }
    PAGE_TABLE.write_text(json.dumps(page, indent=1, sort_keys=True) + "\n")
    cold = cold_storm_day1()
    FULL.write_text(json.dumps({**page, "loso_coverage": coverage, "cold_storm_day1": cold},
                               indent=1, sort_keys=True) + "\n")
    print(f"blocks with forecast >= {MIN_FORECAST_CM} cm in tested countries: {len(ev)}")
    for c, rows in table.items():
        for r in rows:
            hi = f"{r['max_cm']:.0f}" if r["max_cm"] else "+"
            print(f"  {c:7s} {r['min_cm']:.0f}-{hi:>3s} cm  n={r['n_blocks']:4d} st={r['n_stations']:2d}  "
                  f"x{r['lo_ratio']:.2f} .. x{r['hi_ratio']:.2f} (median x{r['median_ratio']:.2f})  {r['basis']}")
    print("leave-one-station-out coverage (target about 0.66):")
    for k, v in coverage.items():
        print(f"  {k:7s} {v['inside']}/{v['n']} = {v['share']:.2f}")
    print("cold-storm check, day-1 forecast >= 3 cm, by model storm temperature:")
    for k, v in cold.items():
        print(f"  {k:8s} n={v['n']:3d} totals x{v['ratio_of_totals']:.2f}  median day x{v['median_day_ratio']:.2f}  "
              f"measured above forecast {v['share_measured_above_forecast']:.0%}")


if __name__ == "__main__":
    main()
