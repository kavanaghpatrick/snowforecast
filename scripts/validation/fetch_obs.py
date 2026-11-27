"""Fetch station observations for the backtest and write one row per station-window.

Stations = every station the product itself accepted (data/forecasts_v2.json
resorts[].stations[]), for the providers that publish free history:
  SNOTEL (AWDB daily), Meteo-France (hourly BASE/HOR files, 2025-2026),
  GeoSphere (TAWES 10-min), Meteocat (XEMA half-hourly).
SLF, JMA and Tirol only publish the last day or so through the endpoints the
product uses; they are covered by the snapshot analysis instead (see report).

Output: data/validation/out/obs_windows.csv with one row per station and window:
  provider, station_id, resort_ids, elev_m, lat, lon, window_kind, t0_utc, t1_utc,
  hs0_cm, hs1_cm, dhs_cm, hn_cm, swe_gain_mm, prec_mm, tmean_c, n_depth_reads, flags
Windows are (t0, t1] in UTC. A value that cannot be computed is left empty;
nothing is filled or interpolated.
Usage: python3 scripts/validation/fetch_obs.py
"""
from __future__ import annotations

import csv
import gzip
import io
import statistics
import sys
import urllib.parse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OUT, RAW, WINDOWS, Refused, daterange, get, load_product  # noqa: E402

UTC = timezone.utc
SPIKE_CM = 20.0          # reading > 20 cm from its +-60 min median is a spike (fetcher rule)
MAX_DEPTH_CM = 1500.0


def stations_from_product():
    d = load_product()
    st = {}
    for r in d["resorts"]:
        for s in r.get("stations") or []:
            key = (s["provider"], s["id"])
            row = st.setdefault(key, {"provider": s["provider"], "id": s["id"], "name": s.get("name"),
                                      "lat": s.get("lat"), "lon": s.get("lon"), "elev": s.get("elevation_m"),
                                      "resorts": [], "utc_offset_s": r["utc_offset_s"]})
            row["resorts"].append(r["id"])
    return st


def all_days():
    for a, b in WINDOWS.values():
        yield from daterange(a, b)


def despike(readings):
    """readings: sorted [(dt, v)]. Drop values > SPIKE_CM from the +-60 min median."""
    kept, dropped = [], 0
    for t, v in readings:
        near = [x for u, x in readings if abs((u - t).total_seconds()) <= 3600]
        if abs(v - statistics.median(near)) <= SPIKE_CM:
            kept.append((t, v))
        else:
            dropped += 1
    return kept, dropped


def depth_at(readings, t, tol_h=1.0):
    """Median of readings within +-tol_h of t, or None."""
    vals = [v for u, v in readings if abs((u - t).total_seconds()) <= tol_h * 3600]
    return statistics.median(vals) if vals else None


def local_day_window(d: date, offset_h: int):
    t0 = datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(hours=offset_h)
    return t0, t0 + timedelta(days=1)


def row(st, kind, t0, t1, **kw):
    out = {"provider": st["provider"], "station_id": st["id"], "station_name": st["name"],
           "resort_ids": "|".join(st["resorts"]), "elev_m": st["elev"], "lat": st["lat"], "lon": st["lon"],
           "window_kind": kind, "t0_utc": t0.strftime("%Y-%m-%dT%H:%M"), "t1_utc": t1.strftime("%Y-%m-%dT%H:%M"),
           "hs0_cm": None, "hs1_cm": None, "dhs_cm": None, "hn_cm": None, "swe_gain_mm": None,
           "prec_mm": None, "tmean_c": None, "n_depth_reads": None, "flags": ""}
    out.update(kw)
    if out["hs0_cm"] is not None and out["hs1_cm"] is not None:
        out["dhs_cm"] = round(out["hs1_cm"] - out["hs0_cm"], 1)
    return out


# ---------------------------------------------------------------- SNOTEL
def snotel(stations, rows, log):
    sts = [s for s in stations.values() if s["provider"] == "SNOTEL"]
    base = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
    trip = ",".join(s["id"] for s in sts)
    series = defaultdict(dict)
    for name, (a, b) in WINDOWS.items():
        url = (f"{base}?stationTriplets={urllib.parse.quote(trip, safe=',:')}&elements=SNWD,WTEQ,PREC,TAVG"
               f"&duration=DAILY&beginDate={(a - timedelta(days=1)).isoformat()}&endDate={b.isoformat()}")
        data = get(url, timeout=180)
        for item in data:
            for blk in item.get("data") or []:
                code = blk["stationElement"]["elementCode"]
                for v in blk.get("values") or []:
                    if v.get("value") is not None:
                        series[(item["stationTriplet"], code)][v["date"][:10]] = float(v["value"])
    for s in sts:
        # AWDB daily SNWD/WTEQ/PREC dated D are the END-of-day (midnight closing D,
        # local standard time) readings: a lag test against the model and against the
        # other providers' sub-daily data puts the peak correlation there (r 0.88 for
        # precipitation vs 0.42 if read as start-of-day). So day D's change is
        # value(D) - value(D-1), and TAVG(D) is that day's mean.
        # The product file's utc_offset_s is the winter (standard) offset.
        off_h = s["utc_offset_s"] // 3600
        SN, WT, PR, TA = (series[(s["id"], k)] for k in ("SNWD", "WTEQ", "PREC", "TAVG"))
        for d in all_days():
            k0, k1 = (d - timedelta(days=1)).isoformat(), d.isoformat()
            t0, t1 = local_day_window(d, off_h)
            hs0 = SN[k0] * 2.54 if k0 in SN else None
            hs1 = SN[k1] * 2.54 if k1 in SN else None
            swe = (WT[k1] - WT[k0]) * 25.4 if k0 in WT and k1 in WT else None
            pr = (PR[k1] - PR[k0]) * 25.4 if k0 in PR and k1 in PR else None
            flags = []
            if pr is not None and pr < 0:
                flags.append("prec_reset_or_negative")
                pr = None
            ta = (TA[k1] - 32) * 5 / 9 if k1 in TA else None
            rows.append(row(s, "local_day", t0, t1, hs0_cm=hs0, hs1_cm=hs1,
                            swe_gain_mm=None if swe is None else round(swe, 1),
                            prec_mm=None if pr is None else round(pr, 1),
                            tmean_c=None if ta is None else round(ta, 1), flags=";".join(flags)))
    log.append(f"SNOTEL: {len(sts)} stations, daily SNWD/WTEQ/PREC/TAVG")


# ---------------------------------------------------------------- Meteo-France
def meteofrance(stations, rows, log):
    sts = {s["id"]: s for s in stations.values() if s["provider"] == "Meteo-France"}
    depts = sorted({sid[:2] for sid in sts})
    recs = defaultdict(list)
    mf_dir = RAW / "mf"
    mf_dir.mkdir(parents=True, exist_ok=True)
    for dep in depts:
        fn = mf_dir / f"H_{dep}.csv.gz"
        if not fn.exists():
            url = ("https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR/"
                   f"H_{dep}_latest-2025-2026.csv.gz")
            fn.write_bytes(get(url, want="bin", timeout=600))
        with gzip.open(fn, "rt", encoding="utf-8", errors="replace", newline="") as f:
            rd = csv.reader(f, delimiter=";")
            hdr = [h.strip() for h in next(rd)]
            ix = {c: hdr.index(c) for c in ("NUM_POSTE", "AAAAMMJJHH", "RR1", "T", "NEIGETOT", "HNEIGEF")}
            for rec in rd:
                sid = rec[ix["NUM_POSTE"]].strip()
                if sid in sts:
                    hh = rec[ix["AAAAMMJJHH"]].strip()
                    if hh >= "2025113000":   # the 2025-2026 file runs to the present day
                        recs[sid].append(tuple(rec[ix[c]].strip() for c in ("AAAAMMJJHH", "RR1", "T", "NEIGETOT", "HNEIGEF")))

    def f(x):
        try:
            return float(x)
        except ValueError:
            return None

    for sid, s in sts.items():
        R = sorted(recs.get(sid, []))
        if not R:
            log.append(f"Meteo-France {sid} ({s['name']}): no rows in the department file, excluded")
            continue
        ts = lambda hh: datetime.strptime(hh, "%Y%m%d%H").replace(tzinfo=UTC)  # noqa: E731
        depth = [(ts(r[0]), f(r[3])) for r in R if f(r[3]) is not None and 0 <= f(r[3]) < MAX_DEPTH_CM]
        n_hours_depth = len({t.hour for t, _ in depth})
        manual = n_hours_depth <= 3
        depth, nd = despike(sorted(depth)) if not manual else (sorted(depth), 0)
        rr = {ts(r[0]): f(r[1]) for r in R if f(r[1]) is not None}
        tt = {ts(r[0]): f(r[2]) for r in R if f(r[2]) is not None}
        hn = {ts(r[0]): f(r[4]) for r in R if f(r[4]) is not None}
        log.append(f"Meteo-France {sid} ({s['name']}): {'manual 06 UTC' if manual else 'hourly'} depth, "
                   f"{len(depth)} depth reads, {nd} spikes dropped, {len(hn)} HNEIGEF values")
        for d in all_days():
            if manual:
                t1 = datetime(d.year, d.month, d.day, 6, tzinfo=UTC)
                t0 = t1 - timedelta(days=1)
                hs0, hs1 = depth_at(depth, t0, 0.5), depth_at(depth, t1, 0.5)
                kind = "obs_06utc"
            else:
                t0, t1 = local_day_window(d, 1)
                hs0, hs1 = depth_at(depth, t0), depth_at(depth, t1)
                kind = "local_day"
            hours = [t0 + timedelta(hours=k + 1) for k in range(24)]
            p = [rr.get(h) for h in hours]
            prec = round(sum(p), 1) if all(v is not None for v in p) else None
            tv = [tt.get(h) for h in hours if tt.get(h) is not None]
            hnv = hn.get(t1) if manual else None
            rows.append(row(s, kind, t0, t1, hs0_cm=hs0, hs1_cm=hs1, hn_cm=hnv, prec_mm=prec,
                            tmean_c=round(sum(tv) / len(tv), 1) if len(tv) >= 20 else None))
            if not manual and hn.get(datetime(d.year, d.month, d.day, 6, tzinfo=UTC)) is not None:
                t1h = datetime(d.year, d.month, d.day, 6, tzinfo=UTC)
                rows.append(row(s, "hn_06utc", t1h - timedelta(days=1), t1h, hn_cm=hn[t1h]))


# ---------------------------------------------------------------- GeoSphere
def geosphere(stations, rows, log):
    sts = [s for s in stations.values() if s["provider"] == "GeoSphere"]
    base = "https://dataset.api.hub.geosphere.at/v1/station/historical/tawes-v1-10min"
    for s in sts:
        readings, rr, tl = [], {}, {}
        for name, (a, b) in WINDOWS.items():
            m0 = date(a.year, a.month, 1)
            while m0 <= b:
                m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
                start = datetime(m0.year, m0.month, m0.day) - timedelta(hours=2)
                end = datetime(m1.year, m1.month, m1.day) + timedelta(hours=2)
                url = (f"{base}?parameters=SCHNEE,RR,TL&station_ids={s['id']}"
                       f"&start={start:%Y-%m-%dT%H:%M}&end={end:%Y-%m-%dT%H:%M}")
                try:
                    data = get(url, timeout=180, pause=0.5)
                except Refused as e:
                    log.append(f"GeoSphere {s['id']}: refused {e}")
                    m0 = m1
                    continue
                stamps = [datetime.fromisoformat(t) for t in data.get("timestamps", [])]
                for feat in data.get("features", []):
                    P = feat["properties"]["parameters"]
                    for t, v in zip(stamps, P.get("SCHNEE", {}).get("data") or []):
                        if v is not None and 0 <= v < MAX_DEPTH_CM:
                            readings.append((t, float(v)))
                    for t, v in zip(stamps, P.get("RR", {}).get("data") or []):
                        if v is not None:
                            rr[t] = float(v)
                    for t, v in zip(stamps, P.get("TL", {}).get("data") or []):
                        if v is not None:
                            tl[t] = float(v)
                m0 = m1
        readings = sorted(set(readings))
        kept, nd = despike(readings)
        log.append(f"GeoSphere {s['id']} ({s['name']}): {len(readings)} 10-min depth reads, {nd} spikes dropped")
        for d in all_days():
            t0, t1 = local_day_window(d, 1)
            slots = [t0 + timedelta(minutes=10 * (k + 1)) for k in range(144)]
            p = [rr.get(x) for x in slots]
            prec = round(sum(p), 1) if all(v is not None for v in p) else None   # no gap-filling
            tv = [tl[x] for x in slots if x in tl]
            rows.append(row(s, "local_day", t0, t1, hs0_cm=depth_at(kept, t0), hs1_cm=depth_at(kept, t1),
                            prec_mm=prec, tmean_c=round(sum(tv) / len(tv), 1) if len(tv) >= 120 else None))


# ---------------------------------------------------------------- Meteocat
def meteocat(stations, rows, log):
    sts = [s for s in stations.values() if s["provider"] == "Meteocat"]
    base = "https://analisi.transparenciacatalunya.cat/resource/nzvn-apee.json"
    for s in sts:
        vals = defaultdict(dict)
        for name, (a, b) in WINDOWS.items():
            for var in ("38", "35", "32"):
                since = (datetime(a.year, a.month, a.day) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
                until = (datetime(b.year, b.month, b.day) + timedelta(days=1, hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
                qs = urllib.parse.urlencode({"codi_estacio": s["id"], "codi_variable": var,
                                             "$where": f"data_lectura >= '{since}' AND data_lectura <= '{until}'",
                                             "$limit": "50000"}, quote_via=urllib.parse.quote)
                try:
                    data = get(f"{base}?{qs}", timeout=180, pause=1.0)
                except Refused as e:
                    log.append(f"Meteocat {s['id']}: refused {e}")
                    continue
                for dct in data:
                    try:
                        t = datetime.fromisoformat(dct["data_lectura"][:19]).replace(tzinfo=UTC)
                        vals[var][t] = float(dct["valor_lectura"])
                    except (KeyError, ValueError):
                        pass
        readings = sorted((t, v) for t, v in vals["38"].items() if 0 <= v < MAX_DEPTH_CM)
        kept, nd = despike(readings)
        log.append(f"Meteocat {s['id']} ({s['name']}): {len(readings)} depth reads, {nd} spikes dropped, "
                   f"{len(vals['35'])} precip reads")
        for d in all_days():
            t0, t1 = local_day_window(d, 1)
            slots = [t0 + timedelta(minutes=30 * (k + 1)) for k in range(48)]
            p = [vals["35"].get(x) for x in slots]
            prec = round(sum(p), 1) if all(v is not None for v in p) else None
            tv = [vals["32"][x] for x in slots if x in vals["32"]]
            rows.append(row(s, "local_day", t0, t1, hs0_cm=depth_at(kept, t0), hs1_cm=depth_at(kept, t1),
                            prec_mm=prec, tmean_c=round(sum(tv) / len(tv), 1) if len(tv) >= 40 else None))


def main():
    stations = stations_from_product()
    rows, log = [], []
    for fn in (snotel, meteofrance, geosphere, meteocat):
        try:
            fn(stations, rows, log)
        except Refused as e:
            log.append(f"{fn.__name__}: REFUSED, provider skipped: {e}")
        except Exception as e:  # keep the other providers; the failure is logged in the output
            log.append(f"{fn.__name__}: FAILED {type(e).__name__}: {e}")
    skipped = sorted({f"{s['provider']} {s['id']}" for s in stations.values()
                      if s["provider"] in ("SLF", "JMA", "Tirol")})
    log.append("No free history through the product's endpoints (snapshot analysis only): " + ", ".join(skipped))
    out = OUT / "obs_windows.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "obs_fetch_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
