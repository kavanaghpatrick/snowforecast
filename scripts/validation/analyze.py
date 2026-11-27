"""Backtest analysis: pairs, regressions, temperature test, recount skill, lead time.

Inputs (built by fetch_obs.py and fetch_model.py, all cached):
  data/validation/out/obs_windows.csv, model_hf.json, model_prev.json, data/forecasts_v2.json
Outputs:
  data/validation/out/pairs.csv        every (station-window, model) pair with its flags
  data/validation/out/exclusions.csv   every station-window not used, with the reason
  data/validation/out/results.json     every number quoted in docs/model_validation_report.md
  data/validation/out/tables.md        the same numbers as tables
Usage: python3 scripts/validation/analyze.py
Deterministic: the bootstrap uses a fixed seed.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, RECOUNT_MIN_MM, SNOW_CM_PER_MM, load_product  # noqa: E402

RNG = np.random.default_rng(20261127)
N_BOOT = 2000
MIN_N = 30          # fewer paired events than this: no conclusion
MIN_STATIONS = 3    # fewer independent stations than this: no conclusion
EVENT_CM = 1.0      # a pair is an "event" when model or measurement reaches this
SPIKE_DAY_CM = 100.0
TH = load_product()["thresholds"]

# ------------------------------------------------------------------ model helpers

def stamps(t0: str, t1: str):
    a = datetime.fromisoformat(t0)
    b = datetime.fromisoformat(t1)
    out = []
    t = a + timedelta(hours=1)
    while t <= b:
        out.append(t.strftime("%Y-%m-%dT%H:%M"))
        t += timedelta(hours=1)
    return out


def hour_split(elev, precip, rain, fl, sh):
    """Copy of scripts/fetch_data_v2.py hour_split (product phase rule)."""
    if precip is None or rain is None or (fl is None and sh is None):
        return None
    line = sh if sh is not None else fl - TH["snowline_below_fl_m"]
    liquid = min(rain, precip)
    frozen = max(precip - liquid, 0.0)
    if elev < line:
        return liquid, frozen, 0.0
    return 0.0, liquid, frozen


def window_model(H, elev, t0, t1):
    """Model aggregates over (t0, t1] at one point; None if any hour is missing."""
    hs = stamps(t0, t1)
    rows = [H.get(h) for h in hs]
    if any(r is None for r in rows):
        return None
    if any(r.get(v) is None for r in rows for v in ("snowfall", "rain", "precipitation", "temperature_2m")):
        return None
    snow = sum(r["snowfall"] for r in rows)
    precip = sum(r["precipitation"] for r in rows)
    rain = sum(r["rain"] for r in rows)
    tw = (sum(r["temperature_2m"] * r["snowfall"] for r in rows) / snow) if snow > 0 else None
    pw = sum(r["precipitation"] for r in rows)
    tp = (sum(r["temperature_2m"] * r["precipitation"] for r in rows) / pw) if pw > 0 else None
    tot = {"rain": 0.0, "mixed": 0.0, "snow": 0.0}
    phase_ok = True
    fl_wet = []
    for r in rows:
        p = r["precipitation"]
        if p <= 0:
            continue
        fl_wet.append(r.get("freezing_level_height"))
        sp = hour_split(elev, p, r["rain"], r.get("freezing_level_height"), r.get("snowfall_height"))
        if sp is None:
            phase_ok = False
            break
        tot["rain"] += sp[0]
        tot["mixed"] += sp[1]
        tot["snow"] += sp[2]
    phase = None
    if phase_ok:
        sig = {k for k, v in tot.items() if v >= TH["phase_min_mm"] and v >= TH["phase_share_min"] * precip}
        if precip < TH["phase_day_min_mm"]:
            phase = "none"
        else:
            sp_ = sig or {max(tot, key=tot.get)}
            phase = next(iter(sp_)) if len(sp_) == 1 else "mixed"
    flmax = max(fl_wet) if fl_wet and all(v is not None for v in fl_wet) else None
    # product recount (dashboard_v2/data.py _recount_mm), on this point's own freezing level
    extra = 0.0
    if phase in ("snow", "mixed") and flmax is not None and elev >= flmax and tot["rain"] < RECOUNT_MIN_MM:
        liquid = min(rain, precip)
        if liquid >= RECOUNT_MIN_MM:
            extra = liquid * SNOW_CM_PER_MM
    return {"snow": snow, "precip": precip, "rain": rain, "frozen_mm": precip - min(rain, precip),
            "tw": tw, "tp": tp, "phase": phase, "rain_phase_mm": tot["rain"] if phase_ok else None,
            "fl_wet_max": flmax, "recount_cm": extra, "snow_rc": snow + extra}


def window_prev(H, t0, t1, n):
    hs = stamps(t0, t1)
    key = "snowfall" if n == 0 else f"snowfall_previous_day{n}"
    pkey = "precipitation" if n == 0 else f"precipitation_previous_day{n}"
    v = [(H.get(h) or {}).get(key) for h in hs]
    p = [(H.get(h) or {}).get(pkey) for h in hs]
    s = sum(v) if all(x is not None for x in v) else None
    pp = sum(p) if all(x is not None for x in p) else None
    return s, pp


# ------------------------------------------------------------------ stats helpers

def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": n, "slope": float(b), "intercept": float(a),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
            "resid_sd": math.sqrt(ss_res / (n - 2))}


def cluster_boot(groups, fn):
    """95% CI of fn(pairs) resampling whole stations (days within a station are correlated)."""
    keys = list(groups)
    if len(keys) < MIN_STATIONS:
        return None
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        pairs = [p for i in pick for p in groups[keys[i]]]
        v = fn(pairs)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < N_BOOT * 0.5:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(pairs, xk="pred", yk="meas"):
    """pairs: list of dicts with xk, yk, 'station'. Returns the full regression block."""
    n = len(pairs)
    st = {p["station"] for p in pairs}
    out = {"n": n, "n_stations": len(st)}
    if n == 0:
        out["verdict"] = "no data"
        return out
    x = np.array([p[xk] for p in pairs], float)
    y = np.array([p[yk] for p in pairs], float)
    out["sum_pred"] = float(x.sum())
    out["sum_meas"] = float(y.sum())
    out["ratio_meas_over_pred"] = float(y.sum() / x.sum()) if x.sum() > 0 else None
    out["mae"] = float(np.mean(np.abs(y - x)))
    out["bias_mean"] = float(np.mean(x - y))
    out["rmse"] = float(np.sqrt(np.mean((y - x) ** 2)))
    out["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if n > 2 and np.std(x) > 0 and np.std(y) > 0 else None
    reg = ols(x, y)
    if reg:
        out.update({k: reg[k] for k in ("slope", "intercept", "r2", "resid_sd")})
    out["resid_p10_p90"] = [float(np.percentile(y - x, 10)), float(np.percentile(y - x, 90))]
    groups = defaultdict(list)
    for p in pairs:
        groups[p["station"]].append(p)

    def ratio(ps):
        sx = sum(p[xk] for p in ps)
        return sum(p[yk] for p in ps) / sx if sx > 0 else None

    def slope(ps):
        r = ols([p[xk] for p in ps], [p[yk] for p in ps])
        return r["slope"] if r else None

    if n >= MIN_N and len(st) >= MIN_STATIONS:
        out["ratio_ci95"] = cluster_boot(groups, ratio)
        out["slope_ci95"] = cluster_boot(groups, slope)
        out["verdict"] = "ok"
    else:
        out["verdict"] = f"insufficient (need n>={MIN_N} and >={MIN_STATIONS} stations)"
    return out


def contingency(pairs, thr, fk="pred", ok="meas"):
    a = sum(1 for p in pairs if p[fk] >= thr and p[ok] >= thr)
    b = sum(1 for p in pairs if p[fk] >= thr and p[ok] < thr)
    c = sum(1 for p in pairs if p[fk] < thr and p[ok] >= thr)
    d = sum(1 for p in pairs if p[fk] < thr and p[ok] < thr)
    return {"hits": a, "false_alarms": b, "misses": c, "correct_neg": d,
            "pod": a / (a + c) if a + c else None, "far": b / (a + b) if a + b else None,
            "csi": a / (a + b + c) if a + b + c else None}


def r(x, k=2):
    if isinstance(x, float):
        return round(x, k)
    if isinstance(x, list):
        return [r(v, k) for v in x]
    if isinstance(x, dict):
        return {a: r(b, k) for a, b in x.items()}
    return x


# ------------------------------------------------------------------ build pairs

def nearest_band(resort, elev):
    bands = [(b, resort.get(k)) for b, k in (("base", "elevation_m"), ("mid", "mid_elevation_m"),
                                              ("summit", "summit_elevation_m")) if resort.get(k) is not None]
    b, e = min(bands, key=lambda be: abs(be[1] - elev))
    return b, e


def build():
    prod = load_product()
    resorts = {x["id"]: x for x in prod["resorts"]}
    hf = json.loads((OUT / "model_hf.json").read_text())
    prev_path = OUT / "model_prev.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    obs = list(csv.DictReader((OUT / "obs_windows.csv").open()))

    def f(v):
        return float(v) if v not in ("", None, "None") else None

    pairs, excl = [], []
    for o in obs:
        sk = f"S:{o['provider']}:{o['station_id']}"
        rid = o["resort_ids"].split("|")[0]
        res = resorts[rid]
        base = {"provider": o["provider"], "station": f"{o['provider']}:{o['station_id']}",
                "station_name": o["station_name"], "resort": rid, "country": res["country"],
                "elev": f(o["elev_m"]), "kind": o["window_kind"], "t0": o["t0_utc"], "t1": o["t1_utc"],
                "window": "W2" if o["t0_utc"] >= "2026-06" else "W1"}
        hn, dhs = f(o["hn_cm"]), f(o["dhs_cm"])
        if hn is not None and hn >= 0:
            meas, mkind = hn, "HN"
        elif dhs is not None:
            if abs(dhs) > SPIKE_DAY_CM:
                excl.append({**base, "reason": f"depth change {dhs} cm in one window (spike)"})
                continue
            meas, mkind = max(dhs, 0.0), "dHS+"
        else:
            excl.append({**base, "reason": "no depth reading at window start or end (stale/missing)"
                         if o["window_kind"] != "hn_06utc" else "no HN value"})
            continue
        S = hf.get(sk)
        if S is None:
            excl.append({**base, "reason": "no model series for station point"})
            continue
        m = window_model(S["hourly"], base["elev"], o["t0_utc"], o["t1_utc"])
        if m is None:
            excl.append({**base, "reason": "model hours incomplete"})
            continue
        band, belev = nearest_band(res, base["elev"])
        B = hf.get(f"B:{rid}:{band}")
        mb = window_model(B["hourly"], belev, o["t0_utc"], o["t1_utc"]) if B else None
        p = {**base, "meas": meas, "meas_kind": mkind, "dhs": dhs, "hn": hn,
             "hs0": f(o["hs0_cm"]), "hs1": f(o["hs1_cm"]),
             "swe": f(o["swe_gain_mm"]), "gauge": f(o["prec_mm"]), "tmean_obs": f(o["tmean_c"]),
             "pred": m["snow"], "pred_rc": m["snow_rc"], "recount_cm": m["recount_cm"],
             "precip": m["precip"], "rain": m["rain"], "frozen_mm": m["frozen_mm"],
             "tw": m["tw"], "tp": m["tp"], "phase": m["phase"], "fl_wet_max": m["fl_wet_max"],
             "band": band, "band_elev": belev, "band_gap_m": belev - base["elev"],
             "pred_band": mb["snow"] if mb else None, "pred_band_rc": mb["snow_rc"] if mb else None,
             "cell_lat": S["cell_lat"], "cell_lon": S["cell_lon"]}
        P = prev.get(sk)
        if P and o["window_kind"] == "local_day":
            for n in range(0, 8):
                s, pp = window_prev(P["hourly"], o["t0_utc"], o["t1_utc"], n)
                p[f"lead{n}"] = s
                p[f"lead{n}_p"] = pp
        pairs.append(p)
    return prod, resorts, hf, pairs, excl


# ------------------------------------------------------------------ analyses

def events(ps, pk="pred"):
    return [p for p in ps if p[pk] >= EVENT_CM or p["meas"] >= EVENT_CM]


def by(ps, keyf):
    g = defaultdict(list)
    for p in ps:
        g[keyf(p)].append(p)
    return dict(sorted(g.items(), key=lambda kv: str(kv[0])))


def treg(tw):
    if tw is None:
        return "no snow"
    if tw < TH["powder_cold_below_c"]:
        return "a: below -8 C"
    if tw <= TH["powder_warm_above_c"]:
        return "b: -8 to -3 C"
    return "c: above -3 C"


def three_day(ps, pk="pred"):
    """Sum consecutive local days into non-overlapping 3-day blocks per station (all 3 present)."""
    out = []
    for st, g in by([p for p in ps if p["kind"] == "local_day"], lambda p: p["station"]).items():
        g = sorted(g, key=lambda p: p["t0"])
        dmap = {p["t0"][:10]: p for p in g}
        days = sorted(dmap)
        if not days:
            continue
        d = date.fromisoformat(days[0])
        end = date.fromisoformat(days[-1])
        while d <= end:
            blk = [dmap.get((d + timedelta(days=k)).isoformat()) for k in range(3)]
            if all(blk):
                out.append({"station": st, "country": blk[0]["country"], "provider": blk[0]["provider"],
                            "window": blk[0]["window"], "pred": sum(b[pk] for b in blk),
                            "meas": sum(b["meas"] for b in blk), "t0": blk[0]["t0"]})
            d += timedelta(days=3)
    return out


def main():
    prod, resorts, hf, pairs, excl = build()
    R = {"settings": {"MIN_N": MIN_N, "MIN_STATIONS": MIN_STATIONS, "EVENT_CM": EVENT_CM, "N_BOOT": N_BOOT}}

    # --- write pairs and exclusions
    if pairs:
        keys = sorted({k for p in pairs for k in p})
        with (OUT / "pairs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(pairs)
    with (OUT / "exclusions.csv").open("w", newline="") as fh:
        keys = sorted({k for p in excl for k in p}) or ["reason"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(excl)
    R["inventory"] = {
        "station_windows_total": len(pairs) + len(excl),
        "pairs_used": len(pairs),
        "excluded": len(excl),
        "excluded_by_reason": Counter(e["reason"].split(" (")[0] if "spike" not in e["reason"] else "spike"
                                      for e in excl),
        "excluded_by_station": Counter(e["station"] for e in excl).most_common(),
        "pairs_by_provider_window": Counter(f"{p['provider']} {p['window']} {p['meas_kind']}" for p in pairs),
        "stations_used": sorted({p["station"] for p in pairs}),
    }
    # Stations whose depth record is almost all exclusions are reported, never silently dropped.

    # Primary set: station-point model vs station measurement, local-day windows and HN windows.
    day_ps = [p for p in pairs if p["kind"] in ("local_day", "obs_06utc")]
    hn_ps = [p for p in pairs if p["meas_kind"] == "HN"]

    # --- 1. overall + strata (daily, events only)
    E = events(day_ps)
    R["daily_overall"] = summarize(E)
    R["daily_overall_recount"] = summarize(events(day_ps, "pred_rc"), xk="pred_rc")
    R["daily_by_country"] = {k: summarize(v) for k, v in by(E, lambda p: p["country"]).items()}
    R["daily_by_provider"] = {k: summarize(v) for k, v in by(E, lambda p: f"{p['provider']} ({p['meas_kind']})").items()}
    R["daily_by_window"] = {k: summarize(v) for k, v in by(E, lambda p: p["window"]).items()}
    R["daily_by_meas_kind"] = {k: summarize(v) for k, v in by(E, lambda p: p["meas_kind"]).items()}
    R["daily_by_temp_regime"] = {k: summarize(v) for k, v in by(E, lambda p: treg(p["tw"])).items()}
    # band: model at the product's nearest band elevation (what the page shows) vs station
    EB = [dict(p, pred=p["pred_band"]) for p in day_ps if p["pred_band"] is not None]
    EB = events(EB)
    R["daily_band_pairs_by_band"] = {k: summarize(v) for k, v in by(EB, lambda p: p["band"]).items()}
    R["daily_band_pairs_by_gap"] = {k: summarize(v) for k, v in by(
        EB, lambda p: "band within 150 m of station" if abs(p["band_gap_m"]) <= 150 else
        ("band 150-400 m above station" if p["band_gap_m"] > 0 and p["band_gap_m"] <= 400 else
         ("band 150-400 m below station" if p["band_gap_m"] < 0 and p["band_gap_m"] >= -400 else
          "band >400 m from station"))).items()}
    R["daily_band_pairs_overall"] = summarize(EB)

    # --- HN-only (cleanest measurement) and depth-change vs new-snow mismatch
    R["hn_overall"] = summarize(events(hn_ps))
    both = defaultdict(dict)
    for p in pairs:
        if p["provider"] == "Meteo-France":
            key = (p["station"], p["t1"])
            if p["meas_kind"] == "HN":
                both[key]["hn"] = p["hn"]
            if p["dhs"] is not None:
                both[key]["dhs"] = p["dhs"]
    mm = [v for v in both.values() if "hn" in v and "dhs" in v and v["hn"] >= 5]
    R["hn_vs_depth_change"] = {
        "n_days_hn_ge_5": len(mm),
        "sum_hn": sum(v["hn"] for v in mm), "sum_dhs_pos": sum(max(v["dhs"], 0) for v in mm),
        "ratio_dhs_over_hn": (sum(max(v["dhs"], 0) for v in mm) / sum(v["hn"] for v in mm)) if mm else None,
        "median_ratio": float(np.median([max(v["dhs"], 0) / v["hn"] for v in mm])) if mm else None}

    # --- 3-day totals
    T3 = [b for b in three_day(day_ps) if b["pred"] >= 3 or b["meas"] >= 3]
    R["three_day_overall"] = summarize(T3)
    R["three_day_by_country"] = {k: summarize(v) for k, v in by(T3, lambda p: p["country"]).items()}
    T3rc = [b for b in three_day(day_ps, "pred_rc") if b["pred"] >= 3 or b["meas"] >= 3]
    R["three_day_overall_recount"] = summarize(T3rc)
    T3b = [b for b in three_day([dict(p, pred=p["pred_band"]) for p in day_ps if p["pred_band"] is not None])
           if b["pred"] >= 3 or b["meas"] >= 3]
    R["three_day_band_overall"] = summarize(T3b)

    # --- 2. the 7:1 hypothesis
    # (a) measured/predicted ratio vs snowfall-weighted model temperature (pred >= 3 cm)
    tr = [p for p in day_ps if p["pred"] >= 3 and p["tw"] is not None]
    x = [p["tw"] for p in tr]
    y = [p["meas"] / p["pred"] for p in tr]
    R["ratio_vs_tw"] = {"all": ols(x, y), "n_stations": len({p["station"] for p in tr})}
    R["ratio_vs_tw"]["log_ratio"] = ols(x, [math.log((p["meas"] + 0.5) / p["pred"]) for p in tr])
    g = defaultdict(list)
    for p in tr:
        g[p["station"]].append(p)
    R["ratio_vs_tw"]["slope_ci95"] = cluster_boot(g, lambda ps: (ols([q["tw"] for q in ps], [q["meas"] / q["pred"] for q in ps]) or {}).get("slope"))
    bins = [(-99, -12), (-12, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]
    R["ratio_by_tw_bin"] = {}
    for lo, hi in bins:
        ps = [p for p in tr if lo <= p["tw"] < hi]
        s = summarize(ps)
        R["ratio_by_tw_bin"][f"{lo}..{hi}"] = {k: s.get(k) for k in ("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "verdict")}
    R["ratio_by_tw_bin_hn_only"] = {}
    for lo, hi in [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]:
        s = summarize([p for p in tr if p["meas_kind"] == "HN" and lo <= p["tw"] < hi])
        R["ratio_by_tw_bin_hn_only"][f"{lo}..{hi}"] = {k: s.get(k) for k in ("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "verdict")}
    for prov in ("SNOTEL", "Meteo-France", "GeoSphere", "Meteocat"):
        ps = [p for p in tr if p["provider"] == prov]
        R.setdefault("ratio_vs_tw_by_provider", {})[prov] = {"ols": ols([p["tw"] for p in ps], [p["meas"] / p["pred"] for p in ps]),
                                                            "n": len(ps)}
    # (b) decomposition at SNOTEL (snow pillow SWE): water ratio and density ratio, by temperature
    sn = [p for p in day_ps if p["provider"] == "SNOTEL" and p["pred"] >= 3 and p["tw"] is not None
          and p["swe"] is not None and p["frozen_mm"] > 0]
    R["snotel_decomposition"] = {}
    for lo, hi in bins:
        ps = [p for p in sn if lo <= p["tw"] < hi]
        if not ps:
            continue
        sw = sum(max(p["swe"], 0) for p in ps)
        fw = sum(p["frozen_mm"] for p in ps)
        sm = sum(p["meas"] for p in ps)
        R["snotel_decomposition"][f"{lo}..{hi}"] = {
            "n": len(ps), "n_stations": len({p["station"] for p in ps}),
            "water_ratio_swe_over_model_frozen": sw / fw if fw else None,
            "obs_cm_per_mm_swe": sm / sw if sw else None,
            "model_cm_per_mm": SNOW_CM_PER_MM}
    # (c) observed snow-to-liquid ratio vs temperature where water is clearly measured
    slr = [p for p in day_ps if p["provider"] == "SNOTEL" and p["swe"] is not None and p["swe"] >= 5
           and p["dhs"] is not None and p["dhs"] > 0 and p["tw"] is not None]
    R["snotel_slr_vs_tw"] = {"ols": ols([p["tw"] for p in slr], [p["dhs"] / p["swe"] for p in slr]),
                             "median_cm_per_mm": float(np.median([p["dhs"] / p["swe"] for p in slr])) if slr else None,
                             "n": len(slr), "n_stations": len({p["station"] for p in slr})}
    slr_o = [p for p in slr if p["tmean_obs"] is not None]
    R["snotel_slr_vs_station_tavg"] = {"ols": ols([p["tmean_obs"] for p in slr_o], [p["dhs"] / p["swe"] for p in slr_o]),
                                       "n": len(slr_o)}
    # (d) precipitation amount alone: model precip vs gauge, wet days
    pg = [dict(p, pred=p["precip"], meas=p["gauge"]) for p in day_ps if p["gauge"] is not None
          and (p["precip"] >= 1 or p["gauge"] >= 1)]
    R["precip_vs_gauge_by_provider"] = {k: summarize(v) for k, v in by(pg, lambda p: p["provider"]).items()}
    sw = [dict(p, pred=p["frozen_mm"], meas=max(p["swe"], 0)) for p in day_ps if p["provider"] == "SNOTEL"
          and p["swe"] is not None and (p["frozen_mm"] >= 1 or p["swe"] >= 1)]
    R["snotel_frozen_water_vs_swe"] = summarize(sw)

    # --- recount skill
    rc = [p for p in day_ps if p["recount_cm"] > 0]
    R["recount"] = {"n_station_days_recounted": len(rc), "n_stations": len({p["station"] for p in rc}),
                    "raw": summarize([dict(p) for p in rc]),
                    "recounted": summarize([dict(p, pred=p["pred_rc"]) for p in rc])}
    if rc:
        R["recount"]["mae_raw"] = float(np.mean([abs(p["meas"] - p["pred"]) for p in rc]))
        R["recount"]["mae_recounted"] = float(np.mean([abs(p["meas"] - p["pred_rc"]) for p in rc]))
        R["recount"]["closer_after_recount"] = sum(abs(p["meas"] - p["pred_rc"]) < abs(p["meas"] - p["pred"]) for p in rc)
        R["recount"]["worse_after_recount"] = sum(abs(p["meas"] - p["pred_rc"]) > abs(p["meas"] - p["pred"]) for p in rc)
        R["recount"]["by_provider"] = Counter(p["provider"] for p in rc)
        R["recount"]["by_provider_detail"] = {
            k: {"n": len(v), "n_stations": len({q["station"] for q in v}),
                "sum_meas": sum(q["meas"] for q in v), "sum_raw": sum(q["pred"] for q in v),
                "sum_recounted": sum(q["pred_rc"] for q in v),
                "mae_raw": float(np.mean([abs(q["meas"] - q["pred"]) for q in v])),
                "mae_recounted": float(np.mean([abs(q["meas"] - q["pred_rc"]) for q in v]))}
            for k, v in by(rc, lambda p: p["provider"]).items()}
        gr = defaultdict(list)
        for p in rc:
            gr[p["station"]].append(p)
        R["recount"]["mae_gain_ci95"] = cluster_boot(gr, lambda ps: float(np.mean([abs(q["meas"] - q["pred"]) - abs(q["meas"] - q["pred_rc"]) for q in ps])))
        R["recount"]["obs_tmean_on_recount_days"] = [p["tmean_obs"] for p in rc if p["tmean_obs"] is not None]
        R["recount"]["examples"] = [{k: p[k] for k in ("station_name", "t0", "pred", "pred_rc", "meas", "meas_kind", "tmean_obs", "fl_wet_max", "elev")}
                                    for p in sorted(rc, key=lambda p: -p["recount_cm"])[:15]]
    rcb = [p for p in day_ps if p["pred_band_rc"] is not None and p["pred_band_rc"] > p["pred_band"]]
    R["recount_band"] = {"n": len(rcb), "raw": summarize([dict(p, pred=p["pred_band"]) for p in rcb]),
                         "recounted": summarize([dict(p, pred=p["pred_band_rc"]) for p in rcb])}

    # --- phase words vs SNOTEL (gauge >= 5 mm)
    ph = [p for p in day_ps if p["provider"] in ("SNOTEL",) and p["gauge"] is not None and p["gauge"] >= 5
          and p["phase"] is not None and p["dhs"] is not None]

    def obs_phase(p):
        if p["dhs"] >= 2.5:
            return "snow accumulated (depth +2.5 cm or more)"
        if p["dhs"] <= 0 and p["tmean_obs"] is not None and p["tmean_obs"] > 0.5:
            return "no accumulation, station above +0.5 C (rain)"
        return "ambiguous"
    tab = defaultdict(Counter)
    for p in ph:
        tab[p["phase"]][obs_phase(p)] += 1
    R["phase_table_snotel"] = {k: dict(v) for k, v in tab.items()}
    ph_all = [p for p in day_ps if p["gauge"] is not None and p["gauge"] >= 5 and p["phase"] is not None
              and p["dhs"] is not None and p["tmean_obs"] is not None]
    tab2 = defaultdict(Counter)
    for p in ph_all:
        tab2[f"{p['provider']}|{p['phase']}"][obs_phase(p)] += 1
    R["phase_table_all"] = {k: dict(v) for k, v in tab2.items()}

    # --- powder temperature label vs observed density (SNOTEL)
    pw = [p for p in slr if p["pred"] >= 5]
    R["powder_label_vs_obs_slr"] = {}
    for lab, lo, hi in (("cold_dry (< -8 C)", -99, TH["powder_cold_below_c"]),
                        ("average (-8..-3 C)", TH["powder_cold_below_c"], TH["powder_warm_above_c"] + 1e-9),
                        ("warm_heavy (> -3 C)", TH["powder_warm_above_c"] + 1e-9, 99)):
        v = [p["dhs"] / p["swe"] for p in pw if lo <= p["tw"] < hi]
        R["powder_label_vs_obs_slr"][lab] = {"n": len(v), "median_cm_per_mm": float(np.median(v)) if v else None,
                                             "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None}

    # --- per-band split: station pairs in one resort, >= 300 m apart
    by_res = defaultdict(set)
    for p in day_ps:
        by_res[p["resort"]].add((p["station"], p["elev"]))
    split = []
    idx = defaultdict(dict)
    for p in day_ps:
        if p["kind"] == "local_day":
            idx[p["station"]][p["t0"]] = p
    for res, sts in by_res.items():
        sts = sorted(sts, key=lambda s: s[1])
        for i in range(len(sts)):
            for j in range(i + 1, len(sts)):
                lo, hi = sts[i], sts[j]
                if hi[1] - lo[1] < 300:
                    continue
                common = sorted(set(idx[lo[0]]) & set(idx[hi[0]]))
                common = [t for t in common if max(idx[lo[0]][t]["pred"], idx[hi[0]][t]["pred"],
                                                   idx[lo[0]][t]["meas"], idx[hi[0]][t]["meas"]) >= EVENT_CM]
                if len(common) < 10:
                    continue
                ml = sum(idx[lo[0]][t]["meas"] for t in common)
                mh = sum(idx[hi[0]][t]["meas"] for t in common)
                pl = sum(idx[lo[0]][t]["pred"] for t in common)
                ph_ = sum(idx[hi[0]][t]["pred"] for t in common)
                split.append({"resort": res, "low": lo[0], "low_m": lo[1], "high": hi[0], "high_m": hi[1],
                              "days": len(common), "meas_low": ml, "meas_high": mh, "pred_low": pl, "pred_high": ph_,
                              "meas_high_minus_low": mh - ml, "pred_high_minus_low": ph_ - pl,
                              "same_model_cell": (hf[f"S:{lo[0]}"]["cell_lat"], hf[f"S:{lo[0]}"]["cell_lon"]) ==
                                                 (hf[f"S:{hi[0]}"]["cell_lat"], hf[f"S:{hi[0]}"]["cell_lon"])})
    R["band_split_station_pairs"] = split
    if split:
        R["band_split_sign_agreement"] = sum((s["meas_high_minus_low"] > 0) == (s["pred_high_minus_low"] > 0) for s in split)

    # --- mid/summit share a cell
    q = [x["quality"]["bands"].get("mid_summit_same_cell") for x in prod["resorts"] if x.get("quality")]
    R["mid_summit_same_cell"] = {"true": sum(1 for v in q if v is True), "of": sum(1 for v in q if v is not None)}

    # --- 3. lead time (previous-runs), station point, local-day windows
    L = [p for p in day_ps if p["kind"] == "local_day" and all(p.get(f"lead{n}") is not None for n in range(8))]
    R["lead_time"] = {}
    for n in range(8):
        ps = [dict(p, pred=p[f"lead{n}"]) for p in L]
        ev = events(ps)
        s = summarize(ev)
        s["ge10cm"] = contingency(ps, 10.0)
        s["ge5cm"] = contingency(ps, 5.0)
        pp = [dict(p, pred=p[f"lead{n}_p"], meas=p["gauge"]) for p in L if p["gauge"] is not None and p[f"lead{n}_p"] is not None]
        pp = [p for p in pp if p["pred"] >= 1 or p["meas"] >= 1]
        sp = summarize(pp)
        s["precip_vs_gauge"] = {k: sp.get(k) for k in ("n", "mae", "pearson_r", "ratio_meas_over_pred")}
        R["lead_time"][f"day{n}"] = {k: s.get(k) for k in ("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "mae",
                                                          "rmse", "pearson_r", "slope", "r2", "ge10cm", "ge5cm",
                                                          "precip_vs_gauge", "verdict")}
    R["lead_time_by_country"] = {}
    for c in sorted({p["country"] for p in L}):
        for n in (0, 1, 3, 5, 7):
            ps = events([dict(p, pred=p[f"lead{n}"]) for p in L if p["country"] == c])
            s = summarize(ps)
            pp = [dict(p, pred=p[f"lead{n}_p"], meas=p["gauge"]) for p in L if p["country"] == c
                  and p["gauge"] is not None]
            R["lead_time_by_country"][f"{c} day{n}"] = {
                **{k: s.get(k) for k in ("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "mae", "pearson_r", "verdict")},
                "model_precip_sum": sum(p["pred"] for p in pp), "gauge_sum": sum(p["meas"] for p in pp)}
    # the 7:1 test with the day-1 forecast as predictor (removes the stitched-archive dry start)
    t1 = [dict(p, pred=p["lead1"]) for p in L if p["lead1"] >= 3 and p["tw"] is not None]
    R["ratio_by_tw_bin_lead1"] = {}
    for lo, hi in [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]:
        s = summarize([p for p in t1 if lo <= p["tw"] < hi])
        R["ratio_by_tw_bin_lead1"][f"{lo}..{hi}"] = {k: s.get(k) for k in ("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "verdict")}
    # next-3 totals and best day by issue date
    Lmap = defaultdict(dict)
    for p in L:
        Lmap[p["station"]][p["t0"][:10]] = p
    n3, best, n3_pairs = [], [], []
    for st, dm in Lmap.items():
        for iss in sorted(dm):
            d0 = date.fromisoformat(iss)
            wk = [dm.get((d0 + timedelta(days=k)).isoformat()) for k in range(7)]
            if all(wk[:3]):
                f3 = sum(wk[k][f"lead{k}"] for k in range(3))
                o3 = sum(wk[k]["meas"] for k in range(3))
                n3_pairs.append({"station": st, "pred": f3, "meas": o3})
            if all(wk):
                # ski score analog: yesterday's snow + today's snow (whole days, not lift hours)
                prevday = dm.get((d0 - timedelta(days=1)).isoformat())
                if prevday is None:
                    continue
                fs = [wk[k][f"lead{k}"] + (wk[k - 1][f"lead{k - 1}"] if k else prevday["lead0"]) for k in range(7)]
                os_ = [wk[k]["meas"] + (wk[k - 1]["meas"] if k else prevday["meas"]) for k in range(7)]
                if max(fs) >= 5.0:   # the page names a best day only at >= 5 cm (POWDER_MIN_CM)
                    fi, oi = int(np.argmax(fs)), int(np.argmax(os_))
                    best.append({"station": st, "issue": iss, "f_idx": fi, "o_idx": oi, "o_max": max(os_),
                                 "o_on_f": os_[fi], "f_max": max(fs)})
    ev3 = [p for p in n3_pairs if p["pred"] >= 3 or p["meas"] >= 3]
    R["next3_by_issue"] = summarize(ev3)
    R["next3_by_issue"]["ge15cm"] = contingency(n3_pairs, 15.0)
    if best:
        named = best
        R["best_day"] = {
            "n_issues_named": len(named), "n_stations": len({b["station"] for b in named}),
            "exact": sum(b["f_idx"] == b["o_idx"] for b in named) / len(named),
            "within_1_day": sum(abs(b["f_idx"] - b["o_idx"]) <= 1 for b in named) / len(named),
            "obs_on_named_day_ge5": sum(b["o_on_f"] >= 5 for b in named) / len(named),
            "random_day_baseline_exact": 1 / 7,
            "named_with_obs_storm": sum(b["o_max"] >= 5 for b in named),
        }
        for k in (1, 3, 5):
            sub = [b for b in named if b["f_idx"] >= k]
            if sub:
                R["best_day"][f"exact_when_named_day_ge_{k}"] = {"n": len(sub), "exact": sum(b["f_idx"] == b["o_idx"] for b in sub) / len(sub),
                                                               "obs_on_named_day_ge5": sum(b["o_on_f"] >= 5 for b in sub) / len(sub)}

    # --- check: does the historical-forecast archive reproduce the product's own past days?
    chk = []
    for rid, res in resorts.items():
        for band in ("base", "mid", "summit"):
            B = hf.get(f"B:{rid}:{band}")
            if not B:
                continue
            for dd in res.get("days") or []:
                if dd["index"] >= 0 or dd.get(band) is None or dd[band].get("snowfall_cm") is None:
                    continue
                d0 = date.fromisoformat(dd["date"])
                t0 = datetime(d0.year, d0.month, d0.day) - timedelta(seconds=res["utc_offset_s"])
                m = window_model(B["hourly"], 0, t0.strftime("%Y-%m-%dT%H:%M"),
                                 (t0 + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M"))
                if m:
                    chk.append({"resort": rid, "band": band, "date": dd["date"], "product": dd[band]["snowfall_cm"],
                                "archive": round(m["snow"], 1)})
    if chk:
        a = np.array([c["product"] for c in chk]); b = np.array([c["archive"] for c in chk])
        R["archive_reproduces_product"] = {"n": len(chk), "pearson_r": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else None,
                                           "mae": float(np.mean(np.abs(a - b))), "sum_product": float(a.sum()), "sum_archive": float(b.sum()),
                                           "exact_within_0_2": int(np.sum(np.abs(a - b) <= 0.2))}

    # --- snapshot: the product's own recent.measured vs recent.modelled (one run, all providers)
    snap = []
    for x in prod["resorts"]:
        m = x["recent"].get("measured")
        if not m or m.get("h72_cm") is None:
            continue
        mod = x["recent"]["modelled"]
        snap.append({"resort": x["id"], "country": x["country"], "kind": m["kind"], "provider": m["provider"],
                     "meas": max(m["h72_cm"], 0.0) if m["kind"] == "depth_change" else m["h72_cm"],
                     "base": (mod.get("base") or {}).get("h72_cm"), "mid": (mod.get("mid") or {}).get("h72_cm"),
                     "summit": (mod.get("summit") or {}).get("h72_cm"), "station": x["id"]})
    R["snapshot_72h"] = {}
    for band in ("base", "mid", "summit"):
        for kind, v in by([s for s in snap if s[band] is not None], lambda s: s["kind"]).items():
            R["snapshot_72h"][f"{band}|{kind}"] = summarize([dict(s, pred=s[band]) for s in v])
    R["snapshot_rows"] = len(snap)

    (OUT / "results.json").write_text(json.dumps(r(R, 3), indent=1, default=str))
    write_tables(R)
    print(json.dumps(r(R["inventory"], 2), default=str)[:3000])


def fmt(v, k=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{k}f}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x, k) for x in v) + "]"
    return str(v)


def table(title, blocks, cols=("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "slope", "slope_ci95",
                               "intercept", "r2", "resid_sd", "mae", "pearson_r", "verdict")):
    lines = [f"### {title}", "", "| stratum | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k, b in blocks.items():
        lines.append(f"| {k} | " + " | ".join(fmt(b.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_tables(R):
    parts = [table("Daily, station-point model vs station (events)", {"all": R["daily_overall"], "all, with recount": R["daily_overall_recount"]}),
             table("Daily by country", R["daily_by_country"]),
             table("Daily by provider (measurement kind)", R["daily_by_provider"]),
             table("Daily by window", R["daily_by_window"]),
             table("Daily by snowfall-weighted model temperature", R["daily_by_temp_regime"]),
             table("Daily, product band nearest the station", {"all": R["daily_band_pairs_overall"], **R["daily_band_pairs_by_band"]}),
             table("Daily, product band by elevation gap", R["daily_band_pairs_by_gap"]),
             table("Manual new-snow boards only (HN)", {"HN": R["hn_overall"]}),
             table("3-day totals", {"all": R["three_day_overall"], "all, with recount": R["three_day_overall_recount"],
                                    "product band": R["three_day_band_overall"], **R["three_day_by_country"]}),
             table("Ratio by temperature bin (pred >= 3 cm)", R["ratio_by_tw_bin"], cols=("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "verdict")),
             table("Model precipitation vs gauge", R["precip_vs_gauge_by_provider"]),
             table("Recount days", {"raw model": R["recount"]["raw"], "recounted": R["recount"]["recounted"]}),
             table("Lead time (snow)", R["lead_time"], cols=("n", "n_stations", "ratio_meas_over_pred", "ratio_ci95", "mae", "rmse", "pearson_r", "r2")),
             table("Next-3 by issue date", {"next3": R["next3_by_issue"]}),
             table("Snapshot 72 h (one run)", R["snapshot_72h"], cols=("n", "ratio_meas_over_pred", "mae", "pearson_r", "slope", "r2", "verdict"))]
    (OUT / "tables.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
