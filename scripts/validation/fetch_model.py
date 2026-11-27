"""Fetch the model hindcast the backtest compares against.

Two Open-Meteo endpoints with their own quotas (api.open-meteo.com is not used):
  historical-forecast-api  the archived best_match forecast, stitched from the
                           first hours of each run (lead ~0-24 h). Same model mix
                           and the same `elevation=` downscaling the product uses.
  previous-runs-api        snowfall/precipitation as forecast 1..7 days before
                           (variables *_previous_dayN), for the lead-time test.

Points:
  S:<provider>:<id>   each history station, at the station's own lat/lon/elevation
  B:<resort>:<band>   each resort with a history station, at the resort's lat/lon and
                      the product's band elevation (exactly what the page shows)
All requests use timezone=GMT; hourly stamp t holds the hour (t-1h, t].
Cached via common.get, so a rerun costs no quota. Stops at the first refusal.
Usage: python3 scripts/validation/fetch_model.py [hf|prev|all]
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, WINDOWS, Refused, get, load_product, write_json  # noqa: E402

HIST = ("snowfall,rain,precipitation,temperature_2m,freezing_level_height,snowfall_height")
PREV = ",".join([f"snowfall_previous_day{n}" for n in range(1, 8)] +
                [f"precipitation_previous_day{n}" for n in range(1, 8)] + ["snowfall", "precipitation"])
HISTORY_PROVIDERS = {"SNOTEL", "Meteo-France", "GeoSphere", "Meteocat"}
BATCH = 6


def points():
    d = load_product()
    pts = {}
    for r in d["resorts"]:
        sts = [s for s in r.get("stations") or [] if s["provider"] in HISTORY_PROVIDERS]
        if not sts:
            continue
        for s in sts:
            pts[f"S:{s['provider']}:{s['id']}"] = (s["lat"], s["lon"], round(s["elevation_m"]))
        for band, key in (("base", "elevation_m"), ("mid", "mid_elevation_m"), ("summit", "summit_elevation_m")):
            if r.get(key) is not None:
                pts[f"B:{r['id']}:{band}"] = (r["lat"], r["lon"], int(r[key]))
    return pts


def fetch(kind, pts, windows):
    host = ("https://historical-forecast-api.open-meteo.com/v1/forecast" if kind == "hf"
            else "https://previous-runs-api.open-meteo.com/v1/forecast")
    vars_ = HIST if kind == "hf" else PREV
    keys = sorted(pts)
    out, log = {}, []
    for wname in windows:
        a, b = WINDOWS[wname]
        a, b = a - timedelta(days=1), b + timedelta(days=1)
        for i in range(0, len(keys), BATCH):
            chunk = keys[i:i + BATCH]
            lat = ",".join(f"{pts[k][0]:.4f}" for k in chunk)
            lon = ",".join(f"{pts[k][1]:.4f}" for k in chunk)
            el = ",".join(str(pts[k][2]) for k in chunk)
            url = (f"{host}?latitude={lat}&longitude={lon}&elevation={el}&hourly={vars_}"
                   f"&start_date={a.isoformat()}&end_date={b.isoformat()}&timezone=GMT"
                   "&precipitation_unit=mm&temperature_unit=celsius")
            try:
                body = get(url, timeout=180, pause=2.0)
            except Refused as e:
                log.append(f"REFUSED at {wname} chunk {i // BATCH}: {e}; stopping")
                return out, log
            if isinstance(body, dict):
                body = [body]
            for k, resp in zip(chunk, body):
                h = resp["hourly"]
                o = out.setdefault(k, {"lat": pts[k][0], "lon": pts[k][1], "elev": pts[k][2],
                                       "cell_lat": resp.get("latitude"), "cell_lon": resp.get("longitude"),
                                       "hourly": {}})
                for j, t in enumerate(h["time"]):
                    o["hourly"][t] = {v: h[v][j] for v in h if v != "time"}
            log.append(f"{kind} {wname} chunk {i // BATCH}: {len(chunk)} points ok")
    return out, log


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pts = points()
    print(f"{len(pts)} points ({sum(k.startswith('S:') for k in pts)} stations)")
    if which in ("hf", "all"):
        out, log = fetch("hf", pts, list(WINDOWS))
        write_json(OUT / "model_hf.json", out)
        (OUT / "model_hf_log.txt").write_text("\n".join(log) + "\n")
        print(log[-1] if log else "no hf log")
    if which in ("prev", "all"):
        spts = {k: v for k, v in pts.items() if k.startswith("S:")}
        out, log = fetch("prev", spts, ["W1_2025_26_winter"])
        write_json(OUT / "model_prev.json", out)
        (OUT / "model_prev_log.txt").write_text("\n".join(log) + "\n")
        print(log[-1] if log else "no prev log")


if __name__ == "__main__":
    main()
