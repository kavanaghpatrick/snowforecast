"""Adversarial re-check of the backtest's conclusions (docs/model_validation_report.md,
docs/model_improvements.md). Offline: reads only data/validation/out/pairs.csv
(analyze.py) and dashboard_v2/skill_ranges.json (hero_ranges.py). Writes
data/validation/out/audit.json and prints the same numbers. Deterministic (fixed seed).

Checks, each answering "does the stated conclusion survive?":
  1. alignment   lag test: station day D against model days D-1, D, D+1
  2. robust      daily and 3-day fits: OLS vs Theil-Sen vs trimmed; ratio with the
                 biggest 5% of days removed; median of per-station ratios
  3. temp        the 7:1 slope on ratios vs on log ratios, Theil-Sen, per-station
                 slopes (sign test), and leave-one-station-out
  4. range       hero range: cluster-bootstrap CI on LOSO coverage; coverage per
                 forecast bucket; out-of-time (Dec-Jan table -> Feb-Mar and back);
                 one-day-staler leads (lead 1..3) as a stand-in for the page's single
                 72 h run; non-overlapping blocks only
  5. cold        day-1 cold-storm medians with station-cluster CIs
  6. recount     how many stations carry the improvement; LOSO on the MAE gain
  7. lead        correlation by lead with station-cluster CIs, and without the US
  8. exclusions  model snow on excluded vs kept windows (is missingness storm-linked?)
  9. strata      count of stratified ratios reported, and how many CIs exclude 1
Usage: python3 scripts/validation/audit.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT  # noqa: E402

RNG = np.random.default_rng(20261128)
NB = 2000
LO_Q, HI_Q = 17.0, 83.0
BUCKETS = [(5.0, 10.0), (10.0, 20.0), (20.0, None)]
TESTED = ("USA", "France", "Spain")


def f(v):
    return float(v) if v not in ("", None, "None") else None


def load():
    rows = list(csv.DictReader((OUT / "pairs.csv").open()))
    for r in rows:
        for k in ("meas", "pred", "pred_rc", "tw", "gauge", "precip", "dhs", "recount_cm", "tmean_obs",
                  *[f"lead{n}" for n in range(8)]):
            r[k] = f(r.get(k))
    return rows


def boot_ci(groups, fn, nb=NB):
    keys = list(groups)
    vals = []
    for _ in range(nb):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        v = fn([p for i in pick for p in groups[keys[i]]])
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def by_station(ps):
    g = defaultdict(list)
    for p in ps:
        g[p["station"]].append(p)
    return g


def theil_sen(x, y, max_pairs=400000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        k = RNG.choice(len(i), max_pairs, replace=False)
        i, j = i[k], j[k]
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def ratio(ps, xk="pred", yk="meas"):
    sx = sum(p[xk] for p in ps)
    return sum(p[yk] for p in ps) / sx if sx > 0 else None


def robust_block(ps, xk="pred"):
    x = [p[xk] for p in ps]
    y = [p["meas"] for p in ps]
    res = np.array(y) - np.array(x)
    keep = np.abs(res) <= np.percentile(np.abs(res), 99)
    big = sorted(ps, key=lambda p: -max(p[xk], p["meas"]))
    cut = big[int(0.05 * len(big)):]
    st = by_station(ps)
    per_st = [ratio(v, xk) for v in st.values() if sum(q[xk] for q in v) >= 20]
    return {"n": len(ps), "n_stations": len(st),
            "ratio_of_totals": round(ratio(ps, xk), 3),
            "ratio_without_top5pct_days": round(ratio(cut, xk), 3),
            "median_of_station_ratios": round(float(np.median([v for v in per_st if v is not None])), 3),
            "share_stations_ratio_above_1": round(float(np.mean([v > 1 for v in per_st if v is not None])), 3),
            "ols_slope": round(ols_slope(x, y), 3),
            "theil_sen_slope": round(theil_sen(x, y), 3),
            "ols_slope_trim1pct_resid": round(ols_slope(np.array(x)[keep], np.array(y)[keep]), 3),
            "median_day_ratio_pred_ge3": round(float(np.median([p["meas"] / p[xk] for p in ps if p[xk] >= 3])), 3)}


def events(ps, xk="pred"):
    return [p for p in ps if p[xk] is not None and (p[xk] >= 1 or p["meas"] >= 1)]


# ------------------------------------------------------------------ 1. alignment
def alignment(rows):
    out = {}
    for prov in ("SNOTEL", "Meteo-France", "Meteocat"):
        m = defaultdict(dict)
        for r in rows:
            if r["provider"] == prov and r["kind"] == "local_day":
                m[r["station"]][r["t0"][:10]] = r
        res = {}
        for k in (-1, 0, 1):
            x, y = [], []
            for dm in m.values():
                for d, r in dm.items():
                    d2 = (date.fromisoformat(d) + timedelta(days=k)).isoformat()
                    if d2 in dm and r["gauge"] is not None and dm[d2]["precip"] is not None:
                        x.append(r["gauge"])
                        y.append(dm[d2]["precip"])
            res[f"model_day_shift_{k:+d}"] = {"n": len(x), "r_gauge_vs_model_precip": round(float(np.corrcoef(x, y)[0, 1]), 3)}
        out[prov] = res
    return out


# ------------------------------------------------------------------ 3-day blocks
def blocks(rows, leads=(0, 1, 2)):
    L = [r for r in rows if r["kind"] == "local_day" and all(r[f"lead{n}"] is not None for n in range(8))]
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
                            "pred": sum(wk[k][f"lead{leads[k]}"] for k in range(3)),
                            "meas": sum(wk[k]["meas"] for k in range(3))})
    return out


def bucket_of(v):
    for i, (lo, hi) in enumerate(BUCKETS):
        if v >= lo and (hi is None or v < hi):
            return i
    return None


def build_table(bs):
    ev = [b for b in bs if b["pred"] >= 5 and b["country"] in TESTED]
    t = {}
    for i in range(len(BUCKETS)):
        inb = [b for b in ev if bucket_of(b["pred"]) == i]
        pooled = np.array([b["meas"] / b["pred"] for b in inb])
        for c in TESTED:
            own = [b for b in inb if b["country"] == c]
            ok = len(own) >= 30 and len({b["station"] for b in own}) >= 5
            r = np.array([b["meas"] / b["pred"] for b in own]) if ok else pooled
            t[(c, i)] = (float(np.percentile(r, LO_Q)), float(np.percentile(r, HI_Q)))
    return t


def inside(t, b):
    lo, hi = t[(b["country"], bucket_of(b["pred"]))]
    return lo * b["pred"] <= b["meas"] <= hi * b["pred"]


def loso(bs):
    ev = [b for b in bs if b["pred"] >= 5 and b["country"] in TESTED]
    res = []
    for st in sorted({b["station"] for b in ev}):
        t = build_table([b for b in bs if b["station"] != st])
        for b in ev:
            if b["station"] == st:
                res.append(dict(b, inside=inside(t, b)))
    return res


def coverage_stats(res):
    g = by_station(res)
    share = lambda ps: float(np.mean([p["inside"] for p in ps]))  # noqa: E731
    out = {"all": {"n": len(res), "share": round(share(res), 3), "station_cluster_ci95": boot_ci(g, share, 1000)}}
    for c in TESTED:
        sub = [p for p in res if p["country"] == c]
        out[c] = {"n": len(sub), "n_stations": len({p["station"] for p in sub}), "share": round(share(sub), 3),
                  "station_cluster_ci95": boot_ci(by_station(sub), share, 1000)}
    for i, (lo, hi) in enumerate(BUCKETS):
        sub = [p for p in res if bucket_of(p["pred"]) == i]
        above = float(np.mean([p["meas"] > p["pred"] * build_table_cache[(p["country"], i)][1] for p in sub]))
        below = float(np.mean([p["meas"] < p["pred"] * build_table_cache[(p["country"], i)][0] for p in sub]))
        out[f"bucket {lo:.0f}-{'' if hi is None else int(hi)} cm"] = {
            "n": len(sub), "share": round(share(sub), 3),
            "share_above_range_full_table": round(above, 3), "share_below_range_full_table": round(below, 3)}
    return out


build_table_cache = {}


def range_checks(rows):
    global build_table_cache
    bs = blocks(rows)
    build_table_cache = build_table(bs)
    out = {}
    # page table matches a rebuild?
    page = json.loads((ROOT / "dashboard_v2" / "skill_ranges.json").read_text())
    mism = []
    for c, lst in page["countries"].items():
        for i, row in enumerate(lst):
            lo, hi = build_table_cache[(c, i)]
            if abs(lo - row["lo_ratio"]) > 0.001 or abs(hi - row["hi_ratio"]) > 0.001:
                mism.append((c, i, row["lo_ratio"], lo, row["hi_ratio"], hi))
    out["page_table_rebuild_mismatches"] = mism
    out["loso"] = coverage_stats(loso(bs))
    # out of time: Dec-Jan table -> Feb-Mar blocks and back
    ev = [b for b in bs if b["pred"] >= 5 and b["country"] in TESTED]
    early = [b for b in bs if b["issue"] < "2026-02-01"]
    late = [b for b in bs if b["issue"] >= "2026-02-01"]
    for name, train, test in (("train Dec-Jan, test Feb-Mar", early, late), ("train Feb-Mar, test Dec-Jan", late, early)):
        t = build_table(train)
        tv = [b for b in test if b["pred"] >= 5 and b["country"] in TESTED]
        res = [dict(b, inside=inside(t, b)) for b in tv]
        g = by_station(res)
        out[name] = {"n": len(res), "share": round(float(np.mean([p["inside"] for p in res])), 3),
                     "station_cluster_ci95": boot_ci(g, lambda ps: float(np.mean([p["inside"] for p in ps])), 1000),
                     "by_country": {c: round(float(np.mean([p["inside"] for p in res if p["country"] == c])), 3) for c in TESTED},
                     "median_ratio_train": round(float(np.median([b["meas"] / b["pred"] for b in train if b["pred"] >= 5])), 3),
                     "median_ratio_test": round(float(np.median([b["meas"] / b["pred"] for b in tv])), 3)}
    # staler leads: the page's hero is one run, 0-72 h ahead; the backtest's day 0 is the
    # stitched first hours (lead ~0) and days 1, 2 are leads 1, 2. Leads 1..3 are a day staler.
    stale = blocks(rows, leads=(1, 2, 3))
    st_ev = [b for b in stale if b["pred"] >= 5 and b["country"] in TESTED]
    res = [dict(b, inside=inside(build_table_cache, b)) for b in st_ev if bucket_of(b["pred"]) is not None]
    def side(p, k):
        lo, hi = build_table_cache[(p["country"], bucket_of(p["pred"]))]
        return p["meas"] > hi * p["pred"] if k == "above" else p["meas"] < lo * p["pred"]
    out["fresh table applied to leads 1-3 blocks"] = {
        "n": len(res), "share": round(float(np.mean([p["inside"] for p in res])), 3),
        "by_country": {c: {"inside": round(float(np.mean([p["inside"] for p in res if p["country"] == c])), 3),
                           "above": round(float(np.mean([side(p, "above") for p in res if p["country"] == c])), 3),
                           "below": round(float(np.mean([side(p, "below") for p in res if p["country"] == c])), 3)}
                       for c in TESTED}}
    # station height vs the page's band: the previous-runs archive was fetched only at
    # station points, so use the stitched archive (available at both) for the DIFFERENCE.
    def stitched_blocks(key):
        by = defaultdict(dict)
        for r in rows:
            if r["kind"] == "local_day" and r["window"] == "W1" and f(r.get(key)) is not None:
                by[r["station"]][r["t0"][:10]] = r
        outb = []
        for st, dm in by.items():
            for iss in sorted(dm):
                d0 = date.fromisoformat(iss)
                wk = [dm.get((d0 + timedelta(days=k)).isoformat()) for k in range(3)]
                if all(wk):
                    outb.append({"station": st, "country": wk[0]["country"], "issue": iss,
                                 "pred": sum(f(w[key]) for w in wk), "meas": sum(w["meas"] for w in wk)})
        return outb
    sb = {k: {(b["station"], b["issue"]): b for b in stitched_blocks(k)} for k in ("pred", "pred_band")}
    both = [k for k in sb["pred"] if k in sb["pred_band"]
            and sb["pred"][k]["pred"] >= 5 and sb["pred_band"][k]["pred"] >= 5 and sb["pred"][k]["country"] in TESTED]
    out["station height vs page band (stitched archive, same blocks)"] = {
        "n": len(both),
        "inside_station_height": round(float(np.mean([inside(build_table_cache, sb["pred"][k]) for k in both])), 3),
        "inside_page_band": round(float(np.mean([inside(build_table_cache, sb["pred_band"][k]) for k in both])), 3),
        "note": "absolute level reflects the stitched day 0; only the difference between the two is the point"}
    # non-overlapping (every third issue day per station)
    no = []
    for st, g in by_station([b for b in ev]).items():
        g = sorted(g, key=lambda b: b["issue"])
        last = None
        for b in g:
            d = date.fromisoformat(b["issue"])
            if last is None or (d - last).days >= 3:
                no.append(b)
                last = d
    lo_res = [p for p in loso(bs) if (p["station"], p["issue"]) in {(b["station"], b["issue"]) for b in no}]
    out["loso_non_overlapping_blocks"] = {"n": len(lo_res), "share": round(float(np.mean([p["inside"] for p in lo_res])), 3)}
    # day-0 share of the 3-day forecast: how much of the "forecast" is the stitched first hours
    out["day0_share_of_3day_forecast"] = None
    return out


# ------------------------------------------------------------------ main
def main():
    rows = load()
    A = {}
    A["1_alignment"] = alignment(rows)

    day_ps = [p for p in rows if p["kind"] in ("local_day", "obs_06utc")]
    E = events(day_ps)
    A["2_robust_daily"] = {"all": robust_block(E)}
    for c in ("USA", "France", "Spain"):
        A["2_robust_daily"][c] = robust_block([p for p in E if p["country"] == c])
    for w in ("W1", "W2"):
        A["2_robust_daily"][w] = robust_block([p for p in E if p["window"] == w])

    # 3. temperature slope
    tr = [p for p in day_ps if p["pred"] >= 3 and p["tw"] is not None]
    x = np.array([p["tw"] for p in tr])
    yr = np.array([p["meas"] / p["pred"] for p in tr])
    yl = np.log((np.array([p["meas"] for p in tr]) + 0.5) / np.array([p["pred"] for p in tr]))
    st = by_station(tr)
    per = [ols_slope([q["tw"] for q in v], [q["meas"] / q["pred"] for q in v]) for v in st.values() if len(v) >= 10]
    per = [s for s in per if s is not None]
    lo_slopes = []
    for k in st:
        rest = [p for p in tr if p["station"] != k]
        lo_slopes.append(ols_slope([p["tw"] for p in rest], [p["meas"] / p["pred"] for p in rest]))
    A["3_temperature"] = {
        "n": len(tr), "n_stations": len(st),
        "ols_slope_ratio_per_degC": round(ols_slope(x, yr), 4),
        "theil_sen_slope_ratio": round(theil_sen(x, yr), 4),
        "ols_slope_ratio_trim_ratio_gt5": round(ols_slope(x[yr <= 5], yr[yr <= 5]), 4),
        "ols_slope_log_ratio": round(ols_slope(x, yl), 4),
        "log_slope_ci95_station_cluster": boot_ci(st, lambda ps: ols_slope([q["tw"] for q in ps],
                                                  [math.log((q["meas"] + 0.5) / q["pred"]) for q in ps]), 1000),
        "share_ratio_gt5": round(float(np.mean(yr > 5)), 3),
        "per_station_slopes_n": len(per),
        "per_station_slopes_negative_share": round(float(np.mean([s < 0 for s in per])), 3),
        "loso_slope_range": [round(min(lo_slopes), 4), round(max(lo_slopes), 4)],
        "note": "negative slope = ratio rises as storms get colder (tw falls)"}
    # by bin: totals vs median day, cluster CI on the median
    bins = {}
    for lo, hi in [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]:
        ps = [p for p in tr if lo <= p["tw"] < hi]
        g = by_station(ps)
        bins[f"{lo}..{hi}"] = {"n": len(ps), "n_stations": len(g), "ratio_of_totals": round(ratio(ps), 3),
                               "median_day": round(float(np.median([p["meas"] / p["pred"] for p in ps])), 3),
                               "median_day_ci95": boot_ci(g, lambda q: float(np.median([p["meas"] / p["pred"] for p in q])), 1000),
                               "ratio_without_top5pct": round(ratio(sorted(ps, key=lambda p: -p["meas"])[max(1, int(.05 * len(ps))):]), 3)}
    A["3_temperature"]["stitched_bins"] = bins

    # 5. cold storm, day-1 forecast
    L = [p for p in rows if p["kind"] == "local_day" and all(p[f"lead{n}"] is not None for n in range(8))]
    t1 = [dict(p, pred=p["lead1"]) for p in L if p["lead1"] >= 3 and p["tw"] is not None]
    cold = {}
    for lo, hi in [(-99, -8), (-8, -5), (-5, -3), (-3, -1), (-1, 99)]:
        ps = [p for p in t1 if lo <= p["tw"] < hi]
        g = by_station(ps)
        cold[f"{lo}..{hi}"] = {"n": len(ps), "n_stations": len(g), "ratio_of_totals": round(ratio(ps), 3),
                               "ratio_ci95": boot_ci(g, ratio, 1000),
                               "median_day": round(float(np.median([p["meas"] / p["pred"] for p in ps])), 3),
                               "median_day_ci95": boot_ci(g, lambda q: float(np.median([p["meas"] / p["pred"] for p in q])), 1000),
                               "share_meas_above_fc": round(float(np.mean([p["meas"] > p["pred"] for p in ps])), 3),
                               "share_above_ci95": boot_ci(g, lambda q: float(np.mean([p["meas"] > p["pred"] for p in q])), 1000),
                               "share_meas_zero": round(float(np.mean([p["meas"] == 0 for p in ps])), 3),
                               "providers": dict(sorted({k: sum(1 for p in ps if p["provider"] == k) for k in {p["provider"] for p in ps}}.items()))}
    A["5_cold_storm_day1"] = cold
    A["5_note"] = ("lead-time pairs are local_day windows only: no new-snow boards, every measurement is a depth "
                   "change, which captured 73% of board totals where both exist")

    # 4. hero range
    A["4_hero_range"] = range_checks(rows)
    bs = blocks(rows)
    L_by = defaultdict(dict)
    for p in L:
        L_by[p["station"]][p["t0"][:10]] = p
    sh = []
    for b in bs:
        if b["pred"] >= 5:
            d0 = L_by[b["station"]][b["issue"]]
            sh.append(d0["lead0"] / b["pred"])
    A["4_hero_range"]["day0_share_of_3day_forecast"] = {"median": round(float(np.median(sh)), 3), "mean": round(float(np.mean(sh)), 3)}

    # 6. recount
    rc = [p for p in day_ps if p["recount_cm"] and p["recount_cm"] > 0]
    gr = by_station(rc)
    gain = {k: float(np.mean([abs(q["meas"] - q["pred"]) - abs(q["meas"] - q["pred_rc"]) for q in v])) for k, v in gr.items()}
    tot_gain = {k: float(np.sum([abs(q["meas"] - q["pred"]) - abs(q["meas"] - q["pred_rc"]) for q in v])) for k, v in gr.items()}
    order = sorted(tot_gain.items(), key=lambda kv: -kv[1])
    allg = sum(tot_gain.values())
    loso_g = [float(np.mean([abs(q["meas"] - q["pred"]) - abs(q["meas"] - q["pred_rc"]) for q in rc if q["station"] != k])) for k in gr]
    A["6_recount"] = {"n": len(rc), "n_stations": len(gr), "mae_gain_cm": round(allg / len(rc), 3),
                      "stations_improved": sum(v > 0 for v in gain.values()),
                      "stations_worse": sum(v < 0 for v in gain.values()),
                      "top3_stations_share_of_total_gain": round(sum(v for _, v in order[:3]) / allg, 3) if allg else None,
                      "top3": [(k, round(v, 1), len(gr[k])) for k, v in order[:3]],
                      "loso_gain_range": [round(min(loso_g), 3), round(max(loso_g), 3)],
                      "by_country_gain_cm": {c: round(float(np.mean([abs(q["meas"] - q["pred"]) - abs(q["meas"] - q["pred_rc"])
                                                                      for q in rc if q["country"] == c])), 3)
                                             for c in sorted({q["country"] for q in rc})},
                      "share_days_tmean_above_0": round(float(np.mean([q["tmean_obs"] > 0 for q in rc if q["tmean_obs"] is not None])), 3)}

    # 7. lead correlation with cluster CIs
    lead = {}
    for n in range(8):
        ps = events([dict(p, pred=p[f"lead{n}"]) for p in L])
        g = by_station(ps)
        rfun = lambda q: float(np.corrcoef([p["pred"] for p in q], [p["meas"] for p in q])[0, 1])  # noqa: E731
        nonus = [p for p in ps if p["country"] != "USA"]
        lead[f"day{n}"] = {"n": len(ps), "r": round(rfun(ps), 3), "r_ci95": boot_ci(g, rfun, 1000),
                           "r_non_us": round(rfun(nonus), 3), "r_non_us_ci95": boot_ci(by_station(nonus), rfun, 1000),
                           "spearman": round(float(np.corrcoef(np.argsort(np.argsort([p["pred"] for p in ps])),
                                                               np.argsort(np.argsort([p["meas"] for p in ps])))[0, 1]), 3)}
    A["7_lead"] = lead

    # 8. exclusions: is missingness linked to storms?
    excl = list(csv.DictReader((OUT / "exclusions.csv").open()))
    A["8_exclusions"] = {"n": len(excl), "by_provider_window": {}}
    for k in sorted({(e["provider"], e["window"]) for e in excl}):
        A["8_exclusions"]["by_provider_window"][" ".join(k)] = sum(1 for e in excl if (e["provider"], e["window"]) == k)
    # model snow at the station on excluded windows is not in exclusions.csv; compare kept-window
    # gauge totals around excluded MF manual days instead: share of kept MF manual days with snow
    mf_kept = [p for p in day_ps if p["provider"] == "Meteo-France" and p["kind"] == "obs_06utc"]
    A["8_exclusions"]["mf_manual_kept_days"] = len(mf_kept)
    A["8_exclusions"]["mf_manual_kept_share_model_snow_ge1"] = round(float(np.mean([p["pred"] >= 1 for p in mf_kept])), 3)

    # 8b. model snow at the station on excluded windows vs kept ones (uses analyze.py's own window rule)
    import analyze  # noqa: E402  (loads thresholds from the product file, as analyze.py does)
    hf = json.loads((OUT / "model_hf.json").read_text())
    ex_snow = defaultdict(list)
    for e in excl:
        S = hf.get(f"S:{e['station']}")
        if not S or not e.get("elev"):
            continue
        m = analyze.window_model(S["hourly"], float(e["elev"]), e["t0"], e["t1"])
        if m:
            ex_snow[(e["provider"], e["window"])].append(m["snow"])
    kept_snow = defaultdict(list)
    for p in day_ps:
        kept_snow[(p["provider"], p["window"])].append(p["pred"])
    A["8_exclusions"]["model_snow_excluded_vs_kept"] = {
        " ".join(k): {"excluded_n": len(v), "excluded_mean_model_cm": round(float(np.mean(v)), 2),
                      "excluded_share_ge1": round(float(np.mean([x >= 1 for x in v])), 3),
                      "kept_mean_model_cm": round(float(np.mean(kept_snow[k])), 2) if kept_snow[k] else None,
                      "kept_share_ge1": round(float(np.mean([x >= 1 for x in kept_snow[k]])), 3) if kept_snow[k] else None}
        for k, v in sorted(ex_snow.items()) if v}

    # 8c. best day: the report compares with a random pick (1/7). A fairer floor is a
    # rule that ignores the forecast: always name the day index measured best most often.
    best = []
    for st, dm in L_by.items():
        for iss in sorted(dm):
            d0 = date.fromisoformat(iss)
            wk = [dm.get((d0 + timedelta(days=k)).isoformat()) for k in range(7)]
            prevday = dm.get((d0 - timedelta(days=1)).isoformat())
            if not all(wk) or prevday is None:
                continue
            fs = [wk[k][f"lead{k}"] + (wk[k - 1][f"lead{k - 1}"] if k else prevday["lead0"]) for k in range(7)]
            os_ = [wk[k]["meas"] + (wk[k - 1]["meas"] if k else prevday["meas"]) for k in range(7)]
            if max(fs) >= 5.0:
                best.append((int(np.argmax(fs)), int(np.argmax(os_)), os_))
    from collections import Counter
    oc = Counter(b[1] for b in best)
    mode_idx, mode_n = oc.most_common(1)[0]
    A["8c_best_day"] = {"n": len(best), "exact": round(float(np.mean([b[0] == b[1] for b in best])), 3),
                        "observed_best_index_counts": dict(sorted(oc.items())),
                        "always_pick_most_common_index": {"index": mode_idx, "exact": round(mode_n / len(best), 3)},
                        "always_pick_day0_ge5_share": round(float(np.mean([b[2][0] >= 5 for b in best])), 3),
                        "note": "ties in measured (often all zero) resolve to index 0 in argmax"}

    # 9. strata count
    res = json.loads((OUT / "results.json").read_text())
    n_strata = n_excl1 = 0
    for key in ("daily_by_country", "daily_by_provider", "daily_by_window", "daily_by_meas_kind", "daily_by_temp_regime",
                "daily_band_pairs_by_band", "daily_band_pairs_by_gap", "three_day_by_country", "ratio_by_tw_bin",
                "ratio_by_tw_bin_hn_only", "ratio_by_tw_bin_lead1", "precip_vs_gauge_by_provider"):
        for v in (res.get(key) or {}).values():
            ci = v.get("ratio_ci95")
            if ci:
                n_strata += 1
                n_excl1 += (ci[0] > 1 or ci[1] < 1)
    A["9_strata"] = {"stratified_ratio_cis": n_strata, "cis_excluding_1": n_excl1,
                     "note": "95% intervals, not adjusted for the number of strata"}

    (OUT / "audit.json").write_text(json.dumps(A, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(A, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
