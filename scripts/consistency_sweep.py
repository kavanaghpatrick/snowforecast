#!/usr/bin/env python3
"""Logical-consistency sweep over data/forecasts_v2.json (schema 3).

Scans every resort and day for contradictions between fields, per the formulas
in docs/dashboard_v2_redesign_spec.md section 1. Read-only; prints a report.

    python3 scripts/consistency_sweep.py [path] [--json out.json] [--examples N]

Exit code 0 always (this is a diagnostic, not a gate). Each rule reports a
violation count and the worst examples, ranked by a per-rule severity number.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BANDS = ("base", "mid", "summit")
HOURLY_DERIVED = ("freezing_level_max_m", "freezing_level_precip_min_m", "freezing_level_precip_max_m",
                  "snowfall_height_precip_min_m", "snowfall_height_precip_max_m", "powder_temp_c",
                  "wind_700hpa_lift_max_kmh", "flat_light_hours", "wind_chill_min_c", "lift_hours_snowfall_cm",
                  "rain_limit_m", "rain_limit_kind", "wind_gust_lift_max_kmh", "cloud_cover_lift_mean_pct")
HOURLY_BAND_DICTS = ("phase", "rain_phase_mm")
DAYPOINT_KEYS = ("snowfall_cm", "rain_mm", "precip_mm", "precip_hours", "precip_prob_pct", "temp_max_c",
                 "temp_min_c", "wind_max_kmh", "gust_max_kmh", "wind_dir_deg", "weather_code", "sunshine_s", "uv_max")


def g(obj, *keys):
    for k in keys:
        if obj is None:
            return None
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def elev(r, band):
    return {"base": r.get("elevation_m"), "mid": r.get("mid_elevation_m"),
            "summit": r.get("summit_elevation_m")}[band]


def ssum(vals):
    if any(v is None for v in vals):
        return None
    return sum(vals)


class Rule:
    def __init__(self, rid, title, judgement):
        self.rid, self.title, self.judgement = rid, title, judgement
        self.hits = []          # (severity, resort_id, text)
        self.checked = 0
        self.resorts = set()

    def hit(self, sev, r, text):
        self.hits.append((sev, r["id"], text))
        self.resorts.add(r["id"])


def sweep(data):
    th = data.get("thresholds") or {}
    gust_t = th.get("wind_gust_hold_kmh", 60)
    w700_t = th.get("wind_700hpa_hold_kmh", 70)
    cold_t = th.get("powder_cold_below_c", -8)
    warm_t = th.get("powder_warm_above_c", -3)
    snowy_min = th.get("snowy_day_min_cm", 1.0)
    storm_min = th.get("storm_day_min_cm", 10)
    storm_codes = set(th.get("storm_codes", [73, 75, 85, 86]))
    storm_prob = th.get("storm_prob_pct", 50)
    storm_3d = th.get("storm_3day_cm", 15)
    bb_prev = th.get("bluebird_prev24_cm", 10)
    bb_codes = set(th.get("bluebird_codes", [0, 1, 2]))
    bb_cloud = th.get("bluebird_cloud_mean_pct", 40)
    bb_flat = th.get("bluebird_flat_light_max_h", 2)
    w700_min = th.get("wind_700hpa_min_summit_m", 2500)
    ph_min = th.get("phase_min_mm", 1.0)

    R = {}

    def rule(rid, title, judgement):
        R[rid] = Rule(rid, title, judgement)
        return R[rid]

    r01 = rule("R01", "Heavy snow (>=10 cm) and rain_risk true at the same band and day",
               "Since spec 1.9.2 the remaining hits are days with real rain hours below the snow line plus snow in "
               "other hours (genuinely mixed days, e.g. Hakuba at +8 C). Not a bug; the page shows the per-day "
               "phase line instead of a bare rain word.")
    r01b = rule("R01b", "Any snow (>=1 cm) and rain_risk true at the same band/day, band temp_max_c < 0",
                "Rain flagged at a band whose own daily max temperature never got above freezing. Comes from "
                "Open-Meteo rain_sum not being re-split by the elevation-downscaled temperature. Since spec 1.9.2 "
                "the remaining hits are bands that sit below the column snow line (freezing level - 250 m) in the "
                "rain hours; the point's own -1 C is a fixed lapse-rate value, so the column wins by design.")
    r01c = rule("R01c", "Band reports rain_mm > 0 although band temp_max_c <= -2 (physically suspect phase split)",
                "Upstream (Open-Meteo) partitions rain/snow at grid-cell temperature, not at the requested "
                "elevation. Not a fetcher bug; the page must not present this rain as fact at that band.")
    r02 = rule("R02", "Bluebird true on a day with summit precip_prob >= 40% or summit snowfall > 0",
               "Spec formula for bluebird looks only at yesterday's snow and today's code/sunshine; it never "
               "requires today to be dry. Spec-formula gap.")
    r02b = rule("R02b", "Bluebird flag does not match recomputation from spec 1.9.1 (dry today, code 0-2, "
                "lift-hour cloud mean <= 40, flat light <= 2 h)", "Fetcher bug if any.")
    r03a = rule("R03a", "wind_hold true but lift-hour gust <= threshold and (700 hPa <= threshold or summit "
                "below wind_700hpa_min_summit_m)", "Fetcher bug if any.")
    r03b = rule("R03b", "Lift-hour summit gust > threshold (short days) or daily gust > threshold (extended) "
                "but wind_hold not true", "Fetcher bug if any.")
    r03d = rule("R03d", "wind_hold true from 700 hPa wind alone at a resort whose summit is below "
                "wind_700hpa_min_summit_m", "Fetcher bug if any (spec 1.9.1 excludes these).")
    r03c = rule("R03c", "Extended day with wind_hold == false (spec: never false)",
                "Fetcher bug if any.")
    r04 = rule("R04", "next_72h.E_cm differs from sum of days 0..2 E.snowfall_cm by > 1 cm",
               "By design: next_72h is a rolling (H, H+72h] window from the hourly call; days 0..2 are calendar "
               "days from the daily call, so the window includes the rest of day 3 morning and excludes earlier "
               "hours of day 0. Data consistent, but two '3-day' numbers on the page can disagree.")
    r05 = rule("R05", "week.E_cm != sum(days 0..6 E.snowfall_cm), or legacy totals disagree",
               "Fetcher bug if any.")
    r06 = rule("R06", "Summit snowfall less than base snowfall by > 3 cm and < 50% of base (same day)",
               "Physically suspect; upstream model artefact (inversion or orographic grid differences).")
    r06b = rule("R06b", "Mid and summit DayPoint snowfall/rain/precip identical on every day 0..14 "
                "(and next_72h + recent identical)",
                "Upstream: Open-Meteo returns the same precipitation and phase split for the mid and summit "
                "elevations (only temperature is downscaled). Data honest but the elevation screen shows two "
                "identical numbers, which reads as a bug.")
    r06c = rule("R06c", "Summit 7-day snowfall within 5% (or 0.5 cm) of mid 7-day snowfall",
                "Upstream: Open-Meteo's elevation adjustment can only turn snow into rain where the downscaled "
                "point is warm; it never adds snow above the model grid cell, so every point above the grid "
                "cell gets the grid cell's snowfall. Data honest, display misleading.")
    r09d = rule("R09d", "Freezing level during precipitation above band elevation, yet band temp_max_c <= -1 "
                "(freezing level and downscaled temperature disagree)",
                "Upstream: freezing_level_height comes from the model column, temperature_2m is lapse-rate "
                "downscaled to the requested elevation. They can disagree by several degrees; rain_risk "
                "trusts the freezing level. Spec-formula gap.")
    r07 = rule("R07", "Storm reason 'heavy snow code' on a day with < storm_day_min_cm, or storm flag mismatches "
               "recomputation", "Fetcher bug if any.")
    r08 = rule("R08", "powder_band inconsistent with powder_temp_c, or powder_temp_c present on a day below "
               "snowy_day_min_cm, or outside summit temp_min..temp_max",
               "Fetcher bug if band mismatch; range check is upstream rounding.")
    r09a = rule("R09a", "rain_risk true but rain_phase_mm < phase_min_mm, or rain_risk/phase/mixed_risk "
                "mutually inconsistent (spec 1.9.2)", "Fetcher bug if any.")
    r09b = rule("R09b", "Band daily rain_mm >= 1 but rain_risk false (informational: the phase rule judged "
                "the model's liquid as wet snow/mixed because the band is above the snow line)",
                "By design since spec 1.9.2; these days carry mixed_risk / phase instead of a rain flag.")
    r09c = rule("R09c", "rain_risk true at band but the base-call freezing level during precipitation, minus "
                "250 m, stays at or below that band in every wet hour",
                "Should be 0 when the band and base calls share one freezing level; any hit means the band's own "
                "cell had a higher freezing level (spec 1.9.2 uses each band's own call).")
    r12 = rule("R12", "next_72h.by_day segments do not sum to next_72h.E_cm (within 0.2 cm), or window "
               "fields missing", "Fetcher bug if any.")
    r13 = rule("R13", "Mid and summit identical over days 0..6 but quality.bands.mid_summit_same_week is not "
               "true, or the flag is true while values differ; same for next_72h.mid_equals_summit",
               "Fetcher bug if any.")
    r14 = rule("R14", "ski_score_cm non-null although the previous day had rain_risk at mid (summit if no mid)",
               "Fetcher bug if any.")
    r15 = rule("R15", "Band-day says rain (rain_risk true) while the same band's day phase is snow or none",
               "Fetcher bug if any.")
    r10 = rule("R10", "Recent measured new snow > 0 while nearest station depth fell by more than that amount, "
               "or h24 > h48 > h72 ordering broken", "Different sources (resort report vs station); mostly "
               "settling/rain/location. Ordering breaks would be a fetcher bug.")
    r11 = rule("R11", "Derived/hourly field non-null where spec says null (past days, extended days), or past "
               "DayPoint field non-null beyond snowfall/rain/precip", "Fetcher bug if any.")
    r11b = rule("R11b", "Extended day rain_risk true (spec: can only be false or null)", "Fetcher bug if any.")

    for r in data.get("resorts", []):
        days = r.get("days") or []
        by = {d["index"]: d for d in days}
        has_mid = r.get("mid_elevation_m") is not None
        flags = r.get("flags") or {}

        # ---- R06b mid == summit identical
        if r.get("summit_elevation_m") is not None:
            r06b.checked += 1
            same = True
            nd = 0
            for i in range(0, 15):
                m, s = g(by.get(i), "mid"), g(by.get(i), "summit")
                if m is None or s is None:
                    continue
                nd += 1
                if any(m.get(k) != s.get(k) for k in ("snowfall_cm", "rain_mm", "precip_mm")):
                    same = False
                    break
            n72 = g(r, "next_72h", "mid_cm") == g(r, "next_72h", "summit_cm")
            rec = g(r, "recent", "modelled", "mid") == g(r, "recent", "modelled", "summit")
            wk_same = all(
                g(by.get(i), "mid") is not None and g(by.get(i), "summit") is not None and all(
                    g(by.get(i), "mid", k) == g(by.get(i), "summit", k) for k in ("snowfall_cm", "rain_mm", "precip_mm"))
                for i in range(7))
            qb = g(r, "quality", "bands") or {}
            r13.checked += 1
            if wk_same != (qb.get("mid_summit_same_week") is True):
                r13.hit(1, r, f"days0..6 mid==summit {wk_same} but quality.bands.mid_summit_same_week "
                              f"{qb.get('mid_summit_same_week')}")
            me = g(r, "next_72h", "mid_equals_summit")
            if me is True and not n72:
                r13.hit(1, r, f"next_72h.mid_equals_summit True but mid {g(r,'next_72h','mid_cm')} != summit "
                              f"{g(r,'next_72h','summit_cm')}")
            if me is True and qb.get("mid_summit_same_72h") is not True:
                r13.hit(1, r, "next_72h.mid_equals_summit True but quality.bands.mid_summit_same_72h not True")
            if same and nd and n72 and rec:
                wk = g(r, "week", "summit_cm") or 0
                r06b.hit(wk, r, f"mid {r['mid_elevation_m']} m / summit {r['summit_elevation_m']} m: identical "
                         f"snow/rain/precip on all {nd} days; next_72h {g(r,'next_72h','summit_cm')} cm both; "
                         f"recent h72 {g(r,'recent','modelled','summit','h72_cm')} cm both; week {wk} cm")
            # near-identical: summit week within 5% (or 0.5 cm) of mid week although summit is higher
            wm, ws, wb = g(r, "week", "mid_cm"), g(r, "week", "summit_cm"), g(r, "week", "base_cm")
            if wm is not None and ws is not None:
                r06c.checked += 1
                if abs(ws - wm) <= max(0.5, 0.05 * max(ws, wm)):
                    r06c.hit(ws, r, f"week base {wb} / mid {wm} / summit {ws} cm "
                             f"({r['elevation_m']} / {r['mid_elevation_m']} / {r['summit_elevation_m']} m)")

        # ---- R05 totals
        r05.checked += 1
        for e in BANDS:
            vals = [g(by.get(i), e, "snowfall_cm") if i in by else None for i in range(7)]
            s = ssum(vals)
            w = g(r, "week", f"{e}_cm")
            if (s is None) != (w is None) or (s is not None and abs(round(s, 1) - w) > 0.15):
                r05.hit(abs((s or 0) - (w or 0)), r, f"week.{e}_cm={w} vs sum days0..6={None if s is None else round(s,1)}")
        fs = ssum([f.get("new_snow_cm") for f in r.get("forecast") or []])
        if (fs is None) != (r.get("seven_day_total_cm") is None) or (
                fs is not None and abs(round(fs, 1) - r["seven_day_total_cm"]) > 0.15):
            r05.hit(abs((fs or 0) - (r.get("seven_day_total_cm") or 0)), r,
                    f"seven_day_total_cm={r.get('seven_day_total_cm')} vs sum forecast={fs}")
        if r.get("summit_seven_day_total_cm") != g(r, "week", "summit_cm"):
            r05.hit(1, r, f"summit_seven_day_total_cm={r.get('summit_seven_day_total_cm')} vs "
                          f"week.summit_cm={g(r,'week','summit_cm')}")
        for f in r.get("forecast") or []:
            bv = g(by.get(f["day"]), "base", "snowfall_cm")
            if bv != f.get("new_snow_cm"):
                r05.hit(abs((bv or 0) - (f.get("new_snow_cm") or 0)), r,
                        f"legacy forecast day {f['day']} new_snow_cm={f.get('new_snow_cm')} vs days base={bv}")

        # ---- R04 next_72h vs days 0..2
        for e in BANDS:
            n = g(r, "next_72h", f"{e}_cm")
            s = ssum([g(by.get(i), e, "snowfall_cm") for i in range(3)])
            if n is None or s is None:
                continue
            r04.checked += 1
            if abs(n - s) > 1.0:
                r04.hit(abs(n - s), r, f"{e}: next_72h={n} cm vs days0..2 sum={round(s,1)} cm "
                        f"(H={g(r,'now','local_time')})")

        # ---- R12 by_day segments sum to the headline
        nx = r.get("next_72h") or {}
        if nx.get("window_end_local") is not None or any(nx.get(f"{e}_cm") is not None for e in BANDS):
            r12.checked += 1
            if not nx.get("window_end_local") or "by_day" not in nx:
                r12.hit(5, r, "next_72h window_end_local/by_day missing")
            else:
                for e in BANDS:
                    tot, segs = nx.get(f"{e}_cm"), (nx.get("by_day") or {}).get(e) or []
                    if tot is None:
                        continue
                    ss = ssum([sg.get("snowfall_cm") for sg in segs])
                    hrs = sum(sg.get("hours", 0) for sg in segs)
                    if ss is None or abs(ss - tot) > 0.2 or hrs != 72:
                        r12.hit(abs((ss or 0) - tot), r, f"{e}: next_72h {tot} cm vs by_day sum {ss} ({hrs} h)")

        # ---- R07 storm
        use_summit = r.get("summit_elevation_m") is not None
        se = "summit" if use_summit else "base"
        for rs in flags.get("storm_reasons") or []:
            if rs.get("reason") == "heavy snow code":
                sf = g(by.get(rs["index"]), se, "snowfall_cm")
                if sf is None or sf < storm_min:
                    r07.hit(storm_min - (sf or 0), r, f"day {rs['index']} heavy-snow-code reason with {se} {sf} cm")
        # recompute
        conds = []
        s3 = ssum([g(by.get(i), se, "snowfall_cm") for i in (1, 2, 3)])
        conds.append(None if s3 is None else s3 >= storm_3d)
        for i in range(7):
            dp = g(by.get(i), se)
            if not dp:
                conds.append(None)
                continue
            parts = [None if dp.get("weather_code") is None else dp["weather_code"] in storm_codes,
                     None if dp.get("precip_prob_pct") is None else dp["precip_prob_pct"] >= storm_prob,
                     None if dp.get("snowfall_cm") is None else dp["snowfall_cm"] >= storm_min]
            conds.append(False if False in parts else (None if None in parts else True))
        exp = True if True in conds else (None if None in conds else False)
        r07.checked += 1
        if exp != flags.get("storm_incoming"):
            r07.hit(5, r, f"storm_incoming={flags.get('storm_incoming')} but recomputed {exp}")

        # ---- R10 recent
        meas = g(r, "recent", "measured")
        if meas:
            r10.checked += 1
            h = [meas.get("h24_cm"), meas.get("h48_cm"), meas.get("h72_cm")]
            if meas.get("kind") == "resort_reported_new_snow" and None not in h and not (h[0] <= h[1] <= h[2]):
                r10.hit(max(h) - min(h), r, f"reported new snow not monotonic h24/48/72={h}")
            st = next((s for s in r.get("stations") or [] if s.get("change_72h_cm") is not None), None)
            if meas.get("kind") == "resort_reported_new_snow" and st and h[2] and h[2] > 0:
                if st["change_72h_cm"] < -h[2]:
                    r10.hit(-st["change_72h_cm"] - h[2], r,
                            f"resort reports {h[2]} cm new in 72h but station {st['name']} "
                            f"({st['distance_km']} km) depth changed {st['change_72h_cm']} cm")
        for e in BANDS:
            m = g(r, "recent", "modelled", e)
            if m and None not in m.values() and not (m["h24_cm"] <= m["h48_cm"] <= m["h72_cm"]):
                r10.hit(5, r, f"modelled {e} not monotonic {m}")

        # ---- per-day rules
        for d in days:
            i, hz, dv = d["index"], d["horizon"], d.get("derived") or {}
            rr = dv.get("rain_risk") or {}
            summit = d.get("summit") or {}
            prev = by.get(i - 1)
            label = f"{d['date']} (day {i})"

            # R11 null-ness
            if hz == "past_model":
                r11.checked += 1
                nn = [k for k, v in dv.items() if k != "rain_risk" and v is not None
                      and not (isinstance(v, dict) and all(x is None for x in v.values()))]
                nn += [f"rain_risk.{k}" for k, v in rr.items() if v is not None]
                for e in BANDS:
                    p = d.get(e) or {}
                    nn += [f"{e}.{k}" for k in DAYPOINT_KEYS if k not in ("snowfall_cm", "rain_mm", "precip_mm")
                           and p.get(k) is not None]
                if nn:
                    r11.hit(len(nn), r, f"{label} past day non-null: {nn[:6]}")
            elif hz == "extended":
                r11.checked += 1
                nn = [k for k in HOURLY_DERIVED + ("ski_score_cm", "bluebird") if dv.get(k) is not None]
                nn += [f"{k}.{e}" for k in HOURLY_BAND_DICTS for e in BANDS if (dv.get(k) or {}).get(e) is not None]
                if dv.get("powder_band") is not None:
                    nn.append("powder_band")
                if nn:
                    r11.hit(len(nn), r, f"{label} extended non-null: {nn}")
                eg = summit.get("gust_max_kmh") if r.get("summit_elevation_m") is not None else None
                if eg is not None and eg > gust_t and dv.get("wind_hold") is not True:
                    r03b.hit(eg, r, f"{label} extended: daily gust {eg} km/h, wind_hold {dv.get('wind_hold')}")
                r03c.checked += 1
                if dv.get("wind_hold") is False:
                    r03c.hit(1, r, f"{label} extended wind_hold False")
                r11b.checked += 1
                for e in BANDS:
                    if rr.get(e) is True or (dv.get("mixed_risk") or {}).get(e) is True:
                        r11b.hit(1, r, f"{label} extended rain_risk/mixed_risk.{e} True")
            if hz != "short":
                continue

            fl = dv.get("freezing_level_max_m")
            flp = dv.get("freezing_level_precip_max_m")
            for e in BANDS:
                p = d.get(e)
                el = elev(r, e)
                if not p or el is None:
                    continue
                sf, rain, tmax = p.get("snowfall_cm"), p.get("rain_mm"), p.get("temp_max_c")
                for ru in (r01, r01b, r01c, r09a, r09b, r09c, r09d):
                    ru.checked += 1
                if flp is not None and flp > el and tmax is not None and tmax <= -1:
                    r09d.hit(tmax * -1 + (flp - el) / 100, r, f"{label} {e} {el} m: FL during precip up to {flp} m, "
                             f"band tmax {tmax} C, rain {rain} mm, snow {sf} cm, rain_risk {rr.get(e)}")
                if rr.get(e) is True and sf is not None and sf >= 10:
                    r01.hit(sf + (rain or 0), r, f"{label} {e} {el} m: snow {sf} cm, rain {rain} mm, "
                            f"tmax {tmax} C, FLmax {fl} m, FL-in-precip max {flp} m -> rain_risk True")
                if rr.get(e) is True and sf is not None and sf >= 1 and tmax is not None and tmax < 0:
                    r01b.hit(sf - tmax, r, f"{label} {e} {el} m: snow {sf} cm, rain {rain} mm, tmax {tmax} C, "
                             f"FLmax {fl} m -> rain_risk True")
                if rain is not None and rain > 0 and tmax is not None and tmax <= -2:
                    r01c.hit(rain - tmax, r, f"{label} {e} {el} m: rain {rain} mm at tmax {tmax} C "
                             f"(snow {sf} cm, FLmax {fl} m)")
                # R09
                ph = (dv.get("phase") or {}).get(e)
                rpm = (dv.get("rain_phase_mm") or {}).get(e)
                mx = (dv.get("mixed_risk") or {}).get(e)
                bad = []
                if rr.get(e) is True and (rpm is None or rpm < ph_min):
                    bad.append(f"rain_risk True with rain_phase_mm {rpm}")
                if ph == "rain" and rr.get(e) is not True and rpm is not None and rpm > ph_min + 0.05:  # stored 1 dp
                    bad.append("phase rain but rain_risk not True")
                if ph == "mixed" and mx is not True:
                    bad.append("phase mixed but mixed_risk not True")
                if (ph is None) != (rr.get(e) is None):
                    bad.append(f"phase {ph} vs rain_risk {rr.get(e)} null mismatch")
                if bad:
                    r09a.hit(len(bad), r, f"{label} {e} {el} m: " + "; ".join(bad))
                if rain is not None and rain >= 1 and rr.get(e) is False:
                    r09b.hit(rain, r, f"{label} {e} {el} m: daily rain {rain} mm, rain_risk False, phase {ph}, "
                             f"mixed_risk {mx}")
                if rr.get(e) is True and flp is not None and flp - 250 <= el and \
                        dv.get("snowfall_height_precip_max_m") is None:
                    r09c.hit(el - flp + 250, r, f"{label} {e} {el} m: base-call FL during precip max {flp} m, "
                             f"rain {rain} mm -> rain_risk True")
                r15.checked += 1
                if rr.get(e) is True and ph in ("snow", "none"):
                    r15.hit(1, r, f"{label} {e}: rain_risk True but phase {ph}")

            # R06 summit < base
            b = g(d, "base", "snowfall_cm")
            s = summit.get("snowfall_cm")
            if b is not None and s is not None and r.get("summit_elevation_m") is not None:
                r06.checked += 1
                if b - s > 3 and s < 0.5 * b:
                    r06.hit(b - s, r, f"{label}: base {b} cm vs summit {s} cm")

            # R03 wind
            sm = r.get("summit_elevation_m")
            gust = dv.get("wind_gust_lift_max_kmh") if sm is not None else None
            w7 = dv.get("wind_700hpa_lift_max_kmh")
            w7_counts = sm is not None and sm >= w700_min
            wh = dv.get("wind_hold")
            r03a.checked += 1
            r03b.checked += 1
            r03d.checked += 1
            if wh is True and (gust is None or gust <= gust_t) and (w7 is None or w7 <= w700_t or not w7_counts):
                r03a.hit(1, r, f"{label}: wind_hold True, lift gust {gust}, 700hPa {w7}, summit {sm} m")
            if gust is not None and gust > gust_t and wh is not True:
                r03b.hit(gust, r, f"{label}: lift gust {gust} km/h, 700hPa {w7}, wind_hold {wh}")
            if wh is True and not w7_counts and (gust is None or gust <= gust_t):
                r03d.hit(w7 or 0, r, f"{label}: summit {sm} m, 700hPa {w7} km/h, lift gust {gust} -> wind_hold True")
            # R14 ski score after a rained-on day
            if 0 <= i <= 6:
                key = "mid" if has_mid else "summit"
                r14.checked += 1
                if dv.get("ski_score_cm") is not None and g(prev, "derived", "rain_risk", key) is True:
                    r14.hit(dv["ski_score_cm"], r, f"{label}: ski_score {dv['ski_score_cm']} after rain at {key}")

            # R02 bluebird
            if 0 <= i <= 6:
                bb = dv.get("bluebird")
                pp = summit.get("precip_prob_pct")
                r02.checked += 1
                if bb is True and ((pp is not None and pp >= 40) or (s is not None and s > 0)):
                    r02.hit((pp or 0) + 10 * (s or 0), r,
                            f"{label}: bluebird True, summit precip prob {pp}%, snow {s} cm, code "
                            f"{summit.get('weather_code')}, sunshine {summit.get('sunshine_s')} s of "
                            f"{d.get('daylight_s')} s, prev-day summit {g(prev,'summit','snowfall_cm')} cm")
                ps = g(prev, "summit", "snowfall_cm")
                code = summit.get("weather_code")
                cm, fh = dv.get("cloud_cover_lift_mean_pct"), dv.get("flat_light_hours")
                parts = [None if ps is None else ps >= bb_prev,
                         None if s is None else s < snowy_min,
                         None if code is None else code in bb_codes,
                         None if cm is None else cm <= bb_cloud,
                         None if fh is None else fh <= bb_flat]
                exp = False if False in parts else (None if None in parts else True)
                r02b.checked += 1
                if exp != bb:
                    r02b.hit(1, r, f"{label}: bluebird {bb}, recomputed {exp}")

            # R08 powder
            pt, pb = dv.get("powder_temp_c"), dv.get("powder_band")
            r08.checked += 1
            exp = None if pt is None else ("cold_dry" if pt < cold_t else ("average" if pt <= warm_t else "warm_heavy"))
            if exp != pb:
                r08.hit(5, r, f"{label}: powder_temp {pt} C band {pb}, expected {exp}")
            if pt is not None and (s is None or s < snowy_min):
                r08.hit(2, r, f"{label}: powder_temp {pt} C on day with summit snow {s} cm")
            tmin, tmx = summit.get("temp_min_c"), summit.get("temp_max_c")
            if pt is not None and tmin is not None and tmx is not None and not (tmin - 0.1 <= pt <= tmx + 0.1):
                r08.hit(abs(pt - tmx), r, f"{label}: powder_temp {pt} C outside summit {tmin}..{tmx} C")

    return R


def report(R, n_examples):
    out = []
    for rid, ru in R.items():
        worst = sorted(ru.hits, key=lambda h: -h[0])[:n_examples]
        out.append({"rule": rid, "title": ru.title, "checked": ru.checked, "violations": len(ru.hits),
                    "resorts_affected": len(ru.resorts), "judgement": ru.judgement,
                    "worst": [f"{rid_}: {t}" for _, rid_, t in worst]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(Path(__file__).resolve().parent.parent / "data" / "forecasts_v2.json"))
    ap.add_argument("--json", help="also write the report as JSON here")
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args(argv)
    data = json.load(open(a.path))
    rep = report(sweep(data), a.examples)
    print(f"File: {a.path}  updated {data.get('updated')}  resorts {len(data.get('resorts', []))}\n")
    for x in rep:
        print(f"[{x['rule']}] {x['title']}")
        print(f"  violations {x['violations']} (of {x['checked']} checked), resorts affected {x['resorts_affected']}")
        for w in x["worst"]:
            print(f"    - {w}")
        if x["violations"]:
            print(f"  judgement: {x['judgement']}")
        print()
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
