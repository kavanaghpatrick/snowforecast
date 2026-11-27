"""Permanent skill tracker: keep every forecast the site issues, and score it once the snow is measured.

The backtest (docs/model_validation_report.md) was one look at last winter. This
script makes the check continuous. Each refresh it appends what the current data
file says to a growing history, and `summary` scores everything collected so far.

    python3 scripts/validation/track_skill.py                # record (default)
    python3 scripts/validation/track_skill.py summary        # print current skill
    python3 scripts/validation/track_skill.py summary --json data/validation/out/skill_summary.json
    options: --data PATH (default data/forecasts_v2.json), --history DIR (default data/validation/history)

Stdlib only, like the fetcher, so it runs on the refresh workflow without installs.

History: one JSON object per line, in monthly files <history>/YYYY-MM.jsonl
(month of the run's `updated`). Append-only. Record types:

  fc      one day's forecast at every band, as issued: resort, date, lead (the
          day's index in that run: 0 = the resort's today, -1 = yesterday's
          stitched model day), raw snowfall/precip/rain per band, plus the
          inputs the page's snow-line recount reads, so the recount can be
          rebuilt later. Leads 0..6 keep the FIRST run of each local day that
          issued them; past days (lead < 0) keep the first sighting of the date.
  hero    the 3-day hero as the page computes its inputs: raw next_72h at the
          hero band, the window, and the likely range for that raw figure
          (dashboard_v2/skill.py). The page prints the range only when
          page_shows_range is true (no recounted rain in the window; None where
          the page's data module could not be imported). One per resort per
          local issue day.
  obs     a measurement. SNOTEL: each 15-day series row {date, depth_cm}
          (end-of-day depth; the backtest's lag test confirmed the dating).
          Other providers: each reading {observed_utc, depth_cm, new_24h_cm,
          change_24h_cm}. Written when new or changed, so later corrections win.
  recent  recent.measured against recent.modelled at the hero band, as shown.
          One per resort per UTC day.

Nothing here alters a number. Scoring uses the same definitions as
scripts/validation/analyze.py: event days are model or station >= 1 cm, ratio is
total measured over total forecast, and a group with fewer than 30 days or 3
stations gets no conclusion.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "forecasts_v2.json"
HISTORY = ROOT / "data" / "validation" / "history"
BANDS = ("base", "mid", "summit")
BAND_ELEV = {"base": "elevation_m", "mid": "mid_elevation_m", "summit": "summit_elevation_m"}
MIN_N, MIN_STATIONS = 30, 3
EVENT_CM = 1.0
SPIKE_CM = 100.0          # same sensor-spike rule as the backtest

sys.path.insert(0, str(ROOT / "dashboard_v2"))
try:
    import skill as page_skill   # stdlib-only module; the page's own range table
except Exception:  # pragma: no cover - the page module is part of the repo
    page_skill = None
try:
    # The page hides the range when the window holds recounted rain (render.hero_range_words).
    # data.py needs streamlit; where it is not installed the flag is recorded as None (unknown).
    import data as page_data
except Exception:  # pragma: no cover
    page_data = None


def recount_in_window(r, band):
    if page_data is None:
        return None
    try:
        return any(page_data.seg_recounted(r, s, band) for s in g(r, "next_72h", "by_day", band) or [])
    except Exception:
        return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def g(o, *path):
    for k in path:
        if isinstance(o, dict):
            o = o.get(k)
        elif isinstance(o, list) and isinstance(k, int) and -len(o) <= k < len(o):
            o = o[k]
        else:
            return None
    return o


def parse_utc(s):
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def resort_tz(r):
    if ZoneInfo is not None and r.get("timezone"):
        try:
            return ZoneInfo(r["timezone"])
        except Exception:
            pass
    off = r.get("utc_offset_s")
    return timezone(timedelta(seconds=off)) if is_num(off) else timezone.utc


def hero_band(r):
    """Same rule as dashboard_v2/data.hero_elevation: mid, else summit, else base."""
    for b in ("mid", "summit", "base"):
        if r.get(BAND_ELEV[b]) is not None and is_num(g(r, "next_72h", f"{b}_cm")):
            return b
    return "base"


def nearest_band(r, elev):
    have = [(b, r.get(BAND_ELEV[b])) for b in BANDS if is_num(r.get(BAND_ELEV[b]))]
    if not have:
        return None
    if not is_num(elev):
        return hero_band(r)
    return min(have, key=lambda x: abs(x[1] - elev))[0]


def month_file(hist: Path, updated: str):
    return hist / f"{updated[:7]}.jsonl"


def read_history(hist: Path):
    out = []
    for p in sorted(hist.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        print(f"warning: unreadable line in {p.name}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------

def fc_key(x):
    return ("fc", x["resort"], x["date"], "past" if x["lead"] < 0 else x["lead"])


def obs_key(x):
    return ("obs", x["station"], x.get("date") or x.get("observed_utc"))


def obs_val(x):
    return (x.get("depth_cm"), x.get("new_24h_cm"), x.get("change_24h_cm"))


def build_records(data):
    if not isinstance(data.get("schema_version"), int) or data["schema_version"] < 3 \
            or not isinstance(data.get("resorts"), list):
        raise SystemExit("data file is not schema version 3 or later; nothing recorded")
    upd = data.get("updated")
    issued = parse_utc(upd)
    if issued is None:
        raise SystemExit("data file has no usable 'updated' time; nothing recorded")
    recs = []
    for r in data.get("resorts") or []:
        rid, country = r.get("id"), r.get("country")
        hb = hero_band(r)
        days = r.get("days") or []
        day0 = next((d for d in days if d.get("index") == 0), None)
        issue_local = g(day0, "date")
        for d in days:
            i = d.get("index")
            if not isinstance(i, int) or i > 6 or d.get("horizon") == "extended":
                continue
            dv = d.get("derived") or {}
            recs.append({
                "t": "fc", "issued": upd, "issue_local": issue_local, "resort": rid, "country": country,
                "date": d.get("date"), "lead": i, "horizon": d.get("horizon"), "hero_band": hb,
                "snow": {b: g(d, b, "snowfall_cm") for b in BANDS},
                "precip": {b: g(d, b, "precip_mm") for b in BANDS},
                "rain": {b: g(d, b, "rain_mm") for b in BANDS},
                "tmax": {b: g(d, b, "temp_max_c") for b in BANDS},
                # inputs of the page's recount (dashboard_v2/data._recount_mm)
                "fl_precip_max_m": dv.get("freezing_level_precip_max_m"),
                "phase": dv.get("phase") or {}, "rain_phase_mm": dv.get("rain_phase_mm") or {},
                "powder_temp_c": dv.get("powder_temp_c"),
            })
        raw = g(r, "next_72h", f"{hb}_cm")
        rc = recount_in_window(r, hb)
        # likely_* is the range for the raw figure. The page prints it only when
        # recount_in_window is False; True means the page shows none, None means unknown.
        rng = page_skill.likely_range(country, raw) if (page_skill and is_num(raw)) else None
        recs.append({
            "t": "hero", "issued": upd, "issue_local": issue_local, "resort": rid, "country": country,
            "band": hb, "raw_cm": raw, "by_band": {b: g(r, "next_72h", f"{b}_cm") for b in BANDS},
            "window_start_local": g(r, "next_72h", "window_start_local"),
            "window_end_local": g(r, "next_72h", "window_end_local"),
            "likely_lo_cm": None if rng is None else round(rng[0], 1),
            "likely_hi_cm": None if rng is None else round(rng[1], 1),
            "recount_in_window": rc,
            "page_shows_range": None if rc is None else (rng is not None and not rc),
            "models_agree_through": g(r, "models", "agree_through"),
        })
        for s in r.get("stations") or []:
            sid = f"{s.get('provider')}:{s.get('id')}"
            base = {"t": "obs", "station": sid, "provider": s.get("provider"), "resort": rid,
                    "country": country, "elev": s.get("elevation_m"), "seen": upd}
            if s.get("provider") == "SNOTEL":
                for row in s.get("series") or []:
                    if is_num(row.get("depth_cm")):
                        recs.append({**base, "kind": "series_day", "date": row.get("date"),
                                     "depth_cm": row["depth_cm"]})
            elif is_num(s.get("depth_cm")) or is_num(s.get("new_24h_cm")):
                ou = s.get("observed_utc")
                if not ou:
                    continue  # a reading without a time cannot be placed on a day
                recs.append({**base, "kind": "reading", "observed_utc": ou,
                             "depth_cm": s.get("depth_cm"), "new_24h_cm": s.get("new_24h_cm"),
                             "change_24h_cm": s.get("change_24h_cm")})
        m = g(r, "recent", "measured") or {}
        mod = g(r, "recent", "modelled", hb) or {}
        recs.append({
            "t": "recent", "issued": upd, "resort": rid, "country": country, "band": hb,
            "kind": m.get("kind"), "provider": m.get("provider"),
            "measured": {k: m.get(k) for k in ("h24_cm", "h48_cm", "h72_cm")},
            "modelled": {k: mod.get(k) for k in ("h24_cm", "h48_cm", "h72_cm")},
        })
    return recs


def record(data_path: Path, hist: Path):
    data = json.loads(data_path.read_text())
    hist.mkdir(parents=True, exist_ok=True)
    existing = read_history(hist)
    seen = set()
    obs_latest = {}
    for x in existing:
        t = x.get("t")
        if t == "fc":
            seen.add(fc_key(x))
        elif t == "obs":
            obs_latest[obs_key(x)] = obs_val(x)
        elif t == "hero":
            seen.add(("hero", x["resort"], x.get("issue_local")))
        elif t == "recent":
            seen.add(("recent", x["resort"], (x.get("issued") or "")[:10]))
        elif t == "run":
            seen.add(("run", x.get("updated")))
    upd = data.get("updated")
    if ("run", upd) in seen:
        print(f"run {upd} already recorded; nothing appended")
        return 0
    new, counts = [], defaultdict(int)
    for x in build_records(data):
        t = x["t"]
        if t == "fc":
            k = fc_key(x)
        elif t == "hero":
            k = ("hero", x["resort"], x.get("issue_local"))
        elif t == "recent":
            k = ("recent", x["resort"], x["issued"][:10])
        else:
            k = obs_key(x)
            if obs_latest.get(k) == obs_val(x):
                continue
            obs_latest[k] = obs_val(x)
            new.append(x)
            counts[t] += 1
            continue
        if k in seen:
            continue
        seen.add(k)
        new.append(x)
        counts[t] += 1
    run = {"t": "run", "updated": upd, "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "resorts": len(data.get("resorts") or []), "appended": dict(counts),
           "range_table": None if page_skill is None else str(page_skill.RANGES_PATH.name)}
    path = month_file(hist, upd)
    with path.open("a") as f:
        for x in new + [run]:
            f.write(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"recorded run {upd} -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def score(pairs):
    """pairs: [(station, forecast_cm, measured_cm)] already on event days."""
    n = len(pairs)
    st = len({p[0] for p in pairs})
    out = {"n": n, "n_stations": st}
    if n == 0:
        return out
    f = [p[1] for p in pairs]
    m = [p[2] for p in pairs]
    out["sum_forecast_cm"] = round(sum(f), 1)
    out["sum_measured_cm"] = round(sum(m), 1)
    out["ratio_meas_over_fc"] = round(sum(m) / sum(f), 2) if sum(f) > 0 else None
    r = pearson(f, m)
    out["pearson_r"] = None if r is None else round(r, 2)
    out["mae_cm"] = round(sum(abs(a - b) for a, b in zip(f, m)) / n, 1)
    out["verdict"] = "ok" if (n >= MIN_N and st >= MIN_STATIONS) else "too few for a conclusion"
    return out


def first_by(recs, keyf, order="issued"):
    out = {}
    for x in sorted(recs, key=lambda x: x.get(order) or ""):
        out.setdefault(keyf(x), x)
    return out


def snotel_gains(obs):
    """{station: {date: new snow cm}} from end-of-day depths (latest sighting wins)."""
    depth = defaultdict(dict)
    meta = {}
    for x in sorted(obs, key=lambda x: x.get("seen") or ""):
        if x.get("kind") == "series_day":
            depth[x["station"]][x["date"]] = x["depth_cm"]
            meta[x["station"]] = x
    gains = defaultdict(dict)
    for st, dm in depth.items():
        for ds, v in dm.items():
            prev = dm.get((date.fromisoformat(ds) - timedelta(days=1)).isoformat())
            if prev is None:
                continue
            dhs = v - prev
            if abs(dhs) > SPIKE_CM:
                continue
            gains[st][ds] = max(dhs, 0.0)
    return gains, meta


def reading_gains(obs, resorts_tz):
    """Other providers: the 24 h ending at a reading, paired with local day D only when
    the reading falls within 6 h of the midnight that ends D (so at least 18 of its 24 h
    are D's). Of several such readings the one nearest midnight wins. new_24h_cm (a
    new-snow measurement) wins over a positive depth change. Kept as its own group,
    never mixed with SNOTEL's exact local days."""
    best = {}
    meta = {}
    for x in sorted(obs, key=lambda x: x.get("seen") or ""):
        if x.get("kind") != "reading":
            continue
        t = parse_utc(x.get("observed_utc"))
        tz = resorts_tz.get(x["resort"])
        if t is None or tz is None:
            continue
        v = x.get("new_24h_cm")
        if not is_num(v):
            c = x.get("change_24h_cm")
            if not is_num(c) or abs(c) > SPIKE_CM:
                continue
            v = max(c, 0.0)
        lt = t.astimezone(tz)
        midnight = datetime.combine((lt + timedelta(hours=12)).date(), datetime.min.time(), tzinfo=lt.tzinfo)
        off_h = abs((lt - midnight).total_seconds()) / 3600
        if off_h > 6:
            continue
        day = (midnight - timedelta(days=1)).date().isoformat()
        k = (x["station"], day)
        if k not in best or off_h <= best[k][0]:
            best[k] = (off_h, v)
        meta[x["station"]] = x
    gains = defaultdict(dict)
    for (st, day), (_off, v) in best.items():
        gains[st][day] = v
    return gains, meta


def summary(hist: Path, data_path: Path, json_out: Path | None):
    H = read_history(hist)
    runs = [x for x in H if x.get("t") == "run"]
    fcs = [x for x in H if x.get("t") == "fc"]
    obs = [x for x in H if x.get("t") == "obs"]
    heroes = [x for x in H if x.get("t") == "hero"]
    recents = [x for x in H if x.get("t") == "recent"]
    resorts = {}
    if data_path.exists():
        for r in json.loads(data_path.read_text()).get("resorts") or []:
            resorts[r["id"]] = r
    tzs = {k: resort_tz(r) for k, r in resorts.items()}

    S = {"history_dir": str(hist), "runs": len(runs),
         "first_run": min((x["updated"] for x in runs), default=None),
         "last_run": max((x["updated"] for x in runs), default=None),
         "records": {"fc": len(fcs), "obs": len(obs), "hero": len(heroes), "recent": len(recents)}}

    # forecast lookup: (resort, date, lead-group) -> first issued record
    fc = first_by(fcs, lambda x: (x["resort"], x["date"], "past" if x["lead"] < 0 else x["lead"]))

    def pair_group(gains, meta):
        by_lead = defaultdict(list)
        for st, dm in gains.items():
            m = meta[st]
            r = resorts.get(m["resort"])
            band = nearest_band(r, m.get("elev")) if r else None
            if band is None:
                continue
            for ds, meas in dm.items():
                for lead in ["past"] + list(range(7)):
                    x = fc.get((m["resort"], ds, lead))
                    if x is None:
                        continue
                    f = g(x, "snow", band)
                    if not is_num(f):
                        continue
                    if f >= EVENT_CM or meas >= EVENT_CM:
                        by_lead[lead].append((st, f, meas, m["country"]))
        return by_lead

    sg, sm = snotel_gains(obs)
    rg, rm = reading_gains(obs, tzs)
    groups = {"SNOTEL (local day, end-of-day depth)": pair_group(sg, sm),
              "Other stations (24 h to a reading within 6 h of midnight)": pair_group(rg, rm)}
    S["daily_by_lead"] = {}
    for gname, by_lead in groups.items():
        S["daily_by_lead"][gname] = {("past (stitched)" if k == "past" else f"day {k}"): score([p[:3] for p in v])
                                     for k, v in sorted(by_lead.items(), key=lambda kv: -1 if kv[0] == "past" else kv[0])}
        countries = sorted({p[3] for v in by_lead.values() for p in v})
        for c in countries:
            for k, v in by_lead.items():
                sub = [p[:3] for p in v if p[3] == c]
                if sub:
                    S["daily_by_lead"].setdefault(f"{gname} / {c}", {})[
                        "past (stitched)" if k == "past" else f"day {k}"] = score(sub)

    # likely-range coverage: 3 whole days from one issue (days 0..2 at their own lead),
    # nearest band, raw model, exactly as scripts/validation/hero_ranges.py built the table
    cov = defaultdict(lambda: [0, 0])
    hero_rc = first_by(heroes, lambda x: (x["resort"], x.get("issue_local")))
    if page_skill is not None:
        for gains, meta in ((sg, sm), (rg, rm)):
            for st, dm in gains.items():
                m = meta[st]
                r = resorts.get(m["resort"])
                band = nearest_band(r, m.get("elev")) if r else None
                if band is None:
                    continue
                issues = {x["issue_local"] for x in fcs if x["resort"] == m["resort"] and x["lead"] == 0}
                for iss in issues:
                    # the page shows no range when the window holds recounted rain
                    if (hero_rc.get((m["resort"], iss)) or {}).get("recount_in_window") is True:
                        continue
                    d0 = date.fromisoformat(iss)
                    ds = [(d0 + timedelta(days=k)).isoformat() for k in range(3)]
                    xs = [fc.get((m["resort"], ds[k], k)) for k in range(3)]
                    if not all(xs) or not all(d in dm for d in ds):
                        continue
                    f = [g(x, "snow", band) for x in xs]
                    if not all(is_num(v) for v in f):
                        continue
                    rng = page_skill.likely_range(m["country"], sum(f))
                    if rng is None:
                        continue
                    meas = sum(dm[d] for d in ds)
                    inside = rng[0] <= meas <= rng[1]
                    for key in (m["country"], "all"):
                        cov[key][0] += inside
                        cov[key][1] += 1
    S["likely_range_coverage"] = {
        "target": "about 0.66 (the range is the middle two thirds of last winter's misses)",
        "groups": {k: {"inside": v[0], "n": v[1], "share": round(v[0] / v[1], 2),
                       "verdict": "ok" if v[1] >= MIN_N else "too few for a conclusion"}
                   for k, v in sorted(cov.items())}}

    # recent: measured new snow vs the page's modelled figure, latest run per resort per day
    rec = {}
    for x in sorted(recents, key=lambda x: x.get("issued") or ""):
        rec[(x["resort"], x["issued"][:10])] = x
    S["recent_measured_vs_modelled"] = {}
    for hk in ("h24_cm", "h72_cm"):
        per = defaultdict(list)
        for (rid, _day), x in rec.items():
            if x.get("kind") not in ("resort_reported_new_snow", "measured_new_snow"):
                continue
            a, b = g(x, "measured", hk), g(x, "modelled", hk)
            if is_num(a) and is_num(b) and (a >= EVENT_CM or b >= EVENT_CM):
                per[x["country"]].append((rid, b, a))
                per["all"].append((rid, b, a))
        S["recent_measured_vs_modelled"][hk] = {
            c: {**score(v), "note": "resorts, not stations; windows overlap across runs"} for c, v in sorted(per.items())}

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(S, indent=1, sort_keys=True) + "\n")
    print_summary(S)
    return S


def fmt(s):
    if not s or not s.get("n"):
        return "no pairs yet"
    bits = [f"n={s['n']}", f"sites={s['n_stations']}"]
    if s.get("ratio_meas_over_fc") is not None:
        bits.append(f"measured/forecast={s['ratio_meas_over_fc']}")
    if s.get("pearson_r") is not None:
        bits.append(f"r={s['pearson_r']}")
    bits.append(f"avg miss={s['mae_cm']} cm")
    bits.append(s["verdict"])
    return ", ".join(bits)


def print_summary(S):
    print(f"Skill history: {S['runs']} runs recorded ({S['first_run']} to {S['last_run']}); "
          f"{S['records']['fc']} forecast days, {S['records']['obs']} measurements, "
          f"{S['records']['hero']} hero figures, {S['records']['recent']} recent-snow pairs.")
    print("\nDaily snow, forecast against measured (event days, model or station >= 1 cm):")
    for gname, leads in S["daily_by_lead"].items():
        print(f"  {gname}")
        if not leads:
            print("    no pairs yet")
        for k, s in leads.items():
            print(f"    {k:16s} {fmt(s)}")
    print("\nLikely-range coverage on 3-day totals (" + S["likely_range_coverage"]["target"] + "):")
    gr = S["likely_range_coverage"]["groups"]
    if not gr:
        print("  no 3-day issue with all three days measured yet")
    for k, v in gr.items():
        print(f"  {k:8s} {v['inside']}/{v['n']} inside = {v['share']} ({v['verdict']})")
    print("\nRecent snow on the page: measured new snow against the model's figure at the hero band:")
    for hk, groups in S["recent_measured_vs_modelled"].items():
        for c, s in groups.items():
            print(f"  {hk[:-3]:4s} {c:12s} {fmt(s)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="record", choices=("record", "summary"))
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--json", type=Path, default=None, help="summary: also write the numbers to this file")
    a = ap.parse_args(argv)
    if a.mode == "record":
        return record(a.data, a.history)
    summary(a.history, a.data, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
