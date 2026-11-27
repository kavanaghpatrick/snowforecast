#!/usr/bin/env python3
"""Validate data/forecasts_v2.json against schema_version 3 (spec section 2.9).

Importable:
    validate(obj, expected_count=86) -> list[str]   (empty list means valid)
    health(obj, max_age_h=None) -> list[str]         (empty list means healthy)

CLI:
    python3 scripts/validate_forecasts_v2.py data/forecasts_v2.json [--health] [--max-age-h H]
    (--expected-count defaults to the file's own run.resort_count)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone

EXPECTED_RESORTS = 86

TOP_KEYS = ["schema_version", "updated", "run", "thresholds", "sources", "resorts"]
RUN_KEYS = ["started_utc", "finished_utc", "duration_s", "resort_count", "open_meteo", "coverage", "warnings"]
RESORT_KEYS = [
    "id", "name", "country", "region", "lat", "lon", "timezone", "utc_offset_s",
    "elevation_m", "summit_elevation_m", "mid_elevation_m", "elevation_note",
    "base_depth_cm", "base_depth_source", "temp_c", "summit_temp_c", "forecast",
    "seven_day_total_cm", "summit_seven_day_total_cm",
    "depth", "stations", "recent", "now", "next_72h", "days", "week", "extended",
    "flags", "models", "quality",
]
DEPTH_KEYS = ["value_cm", "kind", "provider", "source_label", "observed_utc", "report"]
STATION_KEYS = ["provider", "id", "name", "lat", "lon", "elevation_m", "distance_km", "depth_cm",
                "observed_utc", "observed_date", "change_24h_cm", "change_48h_cm", "change_72h_cm",
                "new_24h_cm", "new_48h_cm", "new_72h_cm",
                "swe_change_24h_mm", "swe_mm", "series"]
DAY_KEYS = ["date", "index", "horizon", "sunrise_local", "sunset_local", "daylight_s",
            "base", "mid", "summit", "derived"]
DAYPOINT_KEYS = ["snowfall_cm", "rain_mm", "precip_mm", "precip_hours", "precip_prob_pct",
                 "temp_max_c", "temp_min_c", "wind_max_kmh", "gust_max_kmh", "wind_dir_deg",
                 "weather_code", "sunshine_s", "uv_max"]
DERIVED_KEYS = ["freezing_level_max_m", "freezing_level_precip_min_m", "freezing_level_precip_max_m",
                "snowfall_height_precip_min_m", "snowfall_height_precip_max_m", "rain_risk",
                "powder_temp_c", "powder_band", "wind_700hpa_lift_max_kmh", "wind_hold",
                "flat_light_hours", "wind_chill_min_c", "lift_hours_snowfall_cm", "ski_score_cm", "bluebird"]
HOURLY_DERIVED = ["freezing_level_max_m", "freezing_level_precip_min_m", "freezing_level_precip_max_m",
                  "snowfall_height_precip_min_m", "snowfall_height_precip_max_m", "powder_temp_c",
                  "wind_700hpa_lift_max_kmh", "flat_light_hours", "wind_chill_min_c", "lift_hours_snowfall_cm"]
WEEK_KEYS = ["base_cm", "mid_cm", "summit_cm", "best_snow_day", "best_ski_day", "best_ski_day_score_cm"]
FLAG_KEYS = ["storm_incoming", "storm_reasons", "bluebird_days", "wind_hold_days", "rain_days"]
QUALITY_KEYS = ["forecast_ok", "summit_forecast_ok", "depth_ok", "notes"]
STATION_PROVIDERS = ("SNOTEL", "GeoSphere", "SLF", "JMA", "Meteo-France", "Meteocat", "Tirol", "AEMET")
PROVIDERS = {None, "OnTheSnow", "tenki.jp", *STATION_PROVIDERS}
MAX_STATION_DISTANCE_KM = 15
MEASURED_KINDS = {"depth_change", "resort_reported_new_snow", "measured_new_snow",
                  "station_modelled_new_snow"}
MEASURED_KEYS = ["kind", "provider", "label", "h24_cm", "h48_cm", "h72_cm", "observed_utc"]
COUNTRIES = {"USA", "Austria", "Switzerland", "Japan", "France", "Spain"}
NONNEG_SUFFIXES = ("snowfall_cm", "rain_mm", "precip_mm")
# Keys that are never negative wherever they appear (depths, SWE, new snow, report values,
# forecast totals). Depth changes (change_*_cm) and recent.measured of kind depth_change can be.
NONNEG_KEYS = {"depth_cm", "value_cm", "swe_mm", "new_24h_cm", "new_48h_cm", "new_72h_cm",
               "last24_cm", "last48_cm", "last72_cm", "base_cm", "mid_cm", "summit_cm"}
# Snow depths the fetcher keeps are below 600 cm (MAX_REPORT_DEPTH_CM there).
MAX_DEPTH_CM = 600
DEPTH_KEYS_CAPPED = {"depth_cm", "value_cm"}
FUTURE_SLACK = timedelta(minutes=15)


def _missing(d, keys, where, errs):
    if not isinstance(d, dict):
        errs.append(f"{where}: not an object")
        return True
    miss = [k for k in keys if k not in d]
    if miss:
        errs.append(f"{where}: missing keys {miss}")
        return True
    return False


def _walk(x, path, errs):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            errs.append(f"{path}: NaN or infinite")
    elif isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if v < 0 and (k.endswith(NONNEG_SUFFIXES) or k in NONNEG_KEYS or k == "new_snow_cm"):
                    errs.append(f"{p}: negative value {v}")
                if k in DEPTH_KEYS_CAPPED and v >= MAX_DEPTH_CM:
                    errs.append(f"{p}: snow depth {v} cm is not below {MAX_DEPTH_CM} cm")
            _walk(v, p, errs)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _walk(v, f"{path}[{i}]", errs)


def _is_num_or_none(v):
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def _is_bool_or_none(v):
    return v is None or isinstance(v, bool)


def _parse_utc(s):
    """datetime (aware, UTC) for an ISO time, date for a bare date, None if unreadable."""
    if not isinstance(s, str):
        return None
    try:
        if len(s) == 10:
            return date.fromisoformat(s)
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else None


def _future(s, updated):
    """True when an observation time lies after the file's own update time."""
    t = _parse_utc(s)
    if updated is None or t is None:
        return False
    if isinstance(t, datetime):
        return t > updated + FUTURE_SLACK
    return t > (updated + timedelta(days=1)).date()   # a local date can be one day ahead of UTC


def _check_times(where, value, updated, errs):
    if value is None:
        return
    if _parse_utc(value) is None:
        errs.append(f"{where}: unreadable time {value!r}")
    elif _future(value, updated):
        errs.append(f"{where}: {value} is after the file's update time")


def validate(obj, expected_count=EXPECTED_RESORTS):
    errs: list[str] = []
    if _missing(obj, TOP_KEYS, "top", errs):
        return errs
    if obj["schema_version"] != 3:
        errs.append(f"schema_version is {obj['schema_version']}, expected 3")
    updated = _parse_utc(obj["updated"])
    if not isinstance(updated, datetime):
        errs.append(f"updated is not an ISO UTC time: {obj['updated']!r}")
        updated = None
    if not _missing(obj["run"], RUN_KEYS, "run", errs):
        run = obj["run"]
        for k, typ in (("coverage", dict), ("open_meteo", dict), ("warnings", list)):
            if not isinstance(run[k], typ):
                errs.append(f"run.{k} is not a {typ.__name__}")
        if "provider_status" in run and not isinstance(run["provider_status"], dict):
            errs.append("run.provider_status is not an object")
    if not isinstance(obj["sources"], dict):
        errs.append("sources is not an object")
    if not isinstance(obj["thresholds"], dict) or not obj["thresholds"]:
        errs.append("thresholds is not a non-empty object")
    resorts = obj["resorts"]
    if not isinstance(resorts, list):
        return errs + ["resorts is not a list"]
    if expected_count is not None and len(resorts) != expected_count:
        errs.append(f"resort count {len(resorts)} != {expected_count}")
    rcount = obj["run"].get("resort_count") if isinstance(obj["run"], dict) else None
    if rcount is not None and rcount != len(resorts):
        errs.append(f"run.resort_count {rcount} != {len(resorts)} resorts in the file")
    ids = [r.get("id") if isinstance(r, dict) else None for r in resorts]
    if len(ids) != len(set(ids)):
        errs.append("resort ids are not unique")

    for r in resorts:
        w = f"resort[{r.get('id') if isinstance(r, dict) else '?'}]"
        if _missing(r, RESORT_KEYS, w, errs):
            continue
        try:
            _check_resort(r, w, updated, errs)
        except Exception as e:   # a malformed resort is an error in the list, never a crash
            errs.append(f"{w}: could not be checked ({type(e).__name__}: {e})")

    ok = sum(1 for r in resorts if isinstance(r, dict) and isinstance(r.get("quality"), dict)
             and r["quality"].get("forecast_ok") is True)
    if resorts and ok < 0.9 * len(resorts):
        errs.append(f"only {ok} of {len(resorts)} resorts have forecast_ok")
    _walk(obj, "$", errs)
    return errs


_RESORT_TYPES = (("stations", list), ("days", list), ("forecast", list), ("depth", dict), ("recent", dict),
                 ("now", dict), ("next_72h", dict), ("week", dict), ("extended", (dict, type(None))),
                 ("flags", dict), ("quality", dict))


def _check_resort(r, w, updated, errs):
    bad_types = [k for k, typ in _RESORT_TYPES if not isinstance(r[k], typ)]
    if r["models"] is not None and not isinstance(r["models"], dict):
        bad_types.append("models")
    if bad_types:
        errs.append(f"{w}: wrong type for {bad_types}")
        return
    q = r["quality"]
    if q.get("forecast_ok") and (not isinstance(r["timezone"], str) or not r["timezone"]
                                 or not isinstance(r["utc_offset_s"], int)
                                 or isinstance(r["utc_offset_s"], bool)):
        errs.append(f"{w}: forecast_ok but timezone or utc_offset_s missing")
    for k in ("base_cm", "mid_cm", "summit_cm"):
        if not _is_num_or_none(r["next_72h"].get(k)):
            errs.append(f"{w}: next_72h.{k} wrong type")
    for k in ("base", "summit"):
        if r["now"].get(k) is not None and not isinstance(r["now"][k], dict):
            errs.append(f"{w}: now.{k} not an object")
    if r["country"] not in COUNTRIES:
        errs.append(f"{w}: bad country {r['country']}")
    if not isinstance(r["elevation_m"], int):
        errs.append(f"{w}: elevation_m not int")
    if r["summit_elevation_m"] is not None and not r["elevation_note"]:
        errs.append(f"{w}: summit without elevation_note")
    # depth
    d = r["depth"]
    if not _missing(d, DEPTH_KEYS, f"{w}.depth", errs):
        if d["provider"] not in PROVIDERS:
            errs.append(f"{w}: depth.provider {d['provider']} not allowed")
        text = json.dumps(d).lower()
        if "modeled" in text or "modelled" in text:
            errs.append(f"{w}: depth mentions modelled data")
        if not _is_num_or_none(d["value_cm"]):
            errs.append(f"{w}: depth.value_cm wrong type")
    # legacy keys
    if r["base_depth_cm"] != d.get("value_cm"):
        errs.append(f"{w}: base_depth_cm != depth.value_cm")
    if r["base_depth_source"] != d.get("source_label"):
        errs.append(f"{w}: base_depth_source != depth.source_label")
    if r["temp_c"] != ((r["now"] or {}).get("base") or {}).get("temp_c"):
        errs.append(f"{w}: temp_c != now.base.temp_c")
    if r["summit_temp_c"] != ((r["now"] or {}).get("summit") or {}).get("temp_c"):
        errs.append(f"{w}: summit_temp_c != now.summit.temp_c")
    if r["summit_seven_day_total_cm"] != (r["week"] or {}).get("summit_cm"):
        errs.append(f"{w}: summit_seven_day_total_cm != week.summit_cm")
    fvals = [f.get("new_snow_cm") for f in r["forecast"]]
    exp7 = round(sum(fvals), 1) if len(fvals) == 7 and all(v is not None for v in fvals) else None
    if exp7 is not None and r["seven_day_total_cm"] is not None and abs(exp7 - r["seven_day_total_cm"]) > 0.11:
        errs.append(f"{w}: seven_day_total_cm {r['seven_day_total_cm']} != sum(forecast) {exp7}")
    if exp7 is None and r["seven_day_total_cm"] is not None:
        errs.append(f"{w}: seven_day_total_cm set although a forecast day is null")
    _check_times(f"{w}.depth.observed_utc", d.get("observed_utc") if isinstance(d, dict) else None,
                 updated, errs)
    # stations
    for s in r["stations"]:
        if _missing(s, STATION_KEYS, f"{w}.station", errs):
            continue
        _check_times(f"{w}.station {s['id']}.observed_utc", s["observed_utc"], updated, errs)
        _check_times(f"{w}.station {s['id']}.observed_date", s["observed_date"], updated, errs)
        if not _is_num_or_none(s["depth_cm"]) or not _is_num_or_none(s["distance_km"]) \
                or s["distance_km"] is None:
            errs.append(f"{w}: station {s['id']} depth_cm or distance_km wrong type")
            continue
        if s["distance_km"] > MAX_STATION_DISTANCE_KM:
            errs.append(f"{w}: station {s['id']} is {s['distance_km']} km away")
        if s["provider"] not in STATION_PROVIDERS:
            errs.append(f"{w}: bad station provider {s['provider']}")
    dists = [s.get("distance_km") for s in r["stations"] if isinstance(s, dict)
             and _is_num_or_none(s.get("distance_km")) and s.get("distance_km") is not None]
    if dists != sorted(dists):
        errs.append(f"{w}: stations not sorted by distance")
    meas = (r.get("recent") or {}).get("measured")
    if meas is not None and (not isinstance(meas, dict) or meas.get("kind") not in MEASURED_KINDS):
        errs.append(f"{w}: bad recent.measured kind")
    elif meas is not None and not _missing(meas, MEASURED_KEYS, f"{w}.recent.measured", errs):
        _check_times(f"{w}.recent.measured.observed_utc", meas["observed_utc"], updated, errs)
        for k in ("h24_cm", "h48_cm", "h72_cm"):
            v = meas[k]
            if not _is_num_or_none(v):
                errs.append(f"{w}: recent.measured.{k} wrong type")
            elif v is not None and v < 0 and meas["kind"] != "depth_change":
                errs.append(f"{w}: recent.measured.{k} negative ({v}) for kind {meas['kind']}")
    # days
    for day in r["days"]:
        if _missing(day, DAY_KEYS, f"{w}.day", errs):
            continue
        if not isinstance(day["index"], int) or isinstance(day["index"], bool):
            errs.append(f"{w}: day index {day['index']!r} is not an integer")
            return
        exp_h = "past_model" if day["index"] < 0 else ("short" if day["index"] <= 6 else "extended")
        if day["horizon"] != exp_h:
            errs.append(f"{w}: day {day['index']} horizon {day['horizon']}")
        for e in ("base", "mid", "summit"):
            if day[e] is not None:
                _missing(day[e], DAYPOINT_KEYS, f"{w}.day{day['index']}.{e}", errs)
        der = day["derived"]
        if _missing(der, DERIVED_KEYS, f"{w}.day{day['index']}.derived", errs):
            continue
        if day["horizon"] == "past_model":
            bad = [k for k in DERIVED_KEYS if k != "rain_risk" and der[k] is not None]
            bad += [f"rain_risk.{k}" for k, v in der["rain_risk"].items() if v is not None]
            if bad:
                errs.append(f"{w}: past_model day {day['index']} has derived values {bad}")
        if day["horizon"] == "extended":
            bad = [k for k in HOURLY_DERIVED if der[k] is not None]
            if der["wind_hold"] is False:
                bad.append("wind_hold=false")
            if bad:
                errs.append(f"{w}: extended day {day['index']} has hourly-derived values {bad}")
        for k in ("wind_hold", "bluebird"):
            if not _is_bool_or_none(der[k]):
                errs.append(f"{w}: day {day['index']} {k} not bool/null")
        if der["powder_band"] not in (None, "cold_dry", "average", "warm_heavy"):
            errs.append(f"{w}: bad powder_band")
    idxs = [d["index"] for d in r["days"]]
    if idxs != sorted(idxs):
        errs.append(f"{w}: days not ascending")
    if q.get("forecast_ok") and idxs != list(range(-3, 15)):
        errs.append(f"{w}: forecast_ok but days cover {idxs[:1]}..{idxs[-1:]}, not -3..14")
    _missing(r["week"], WEEK_KEYS, f"{w}.week", errs)
    _missing(r["flags"], FLAG_KEYS, f"{w}.flags", errs)
    if not _missing(r["quality"], QUALITY_KEYS, f"{w}.quality", errs):
        if r["quality"]["depth_ok"] != (d.get("value_cm") is not None):
            errs.append(f"{w}: quality.depth_ok inconsistent")
    if r["summit_elevation_m"] is None and r["next_72h"].get("summit_cm") is not None:
        errs.append(f"{w}: summit forecast without summit elevation")


def health(obj, max_age_h=None, now=None):
    """Source health (spec 2.9 --health). max_age_h also flags a file older than that."""
    problems = []
    run = obj.get("run") if isinstance(obj.get("run"), dict) else {}
    for w in run.get("warnings") or []:
        problems.append(f"run warning: {w}")
    resorts = [r for r in obj.get("resorts") or [] if isinstance(r, dict)]

    def sts(r):
        return [s for s in (r.get("stations") or []) if isinstance(s, dict)]

    for prov in STATION_PROVIDERS:
        assigned = [r for r in resorts if any(s.get("provider") == prov for s in sts(r))]
        ok = [r for r in assigned
              if any(s.get("provider") == prov and s.get("depth_cm") is not None for s in sts(r))]
        if len(assigned) >= 5 and not ok:
            problems.append(f"{prov}: 0 of {len(assigned)} resorts have a reading")
    if max_age_h is not None:
        upd = _parse_utc(obj.get("updated"))
        now = now or datetime.now(timezone.utc)
        if not isinstance(upd, datetime):
            problems.append(f"updated time unreadable: {obj.get('updated')!r}")
        elif now - upd > timedelta(hours=max_age_h):
            problems.append(f"file is stale: updated {obj['updated']}, more than {max_age_h:g} hours ago")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--health", action="store_true")
    # Default: the count the fetcher itself recorded (run.resort_count), so adding resorts
    # does not need a matching edit here or in the workflow. The fetcher checks the count
    # against its own resort list before it writes.
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--max-age-h", type=float, default=None,
                    help="with --health: also flag a file whose 'updated' is older than this")
    args = ap.parse_args(argv)
    with open(args.path, encoding="utf-8") as f:
        obj = json.load(f)
    expected = args.expected_count
    if expected is None:
        rc_ = (obj.get("run") or {}).get("resort_count") if isinstance(obj, dict) else None
        expected = rc_ if isinstance(rc_, int) and not isinstance(rc_, bool) else EXPECTED_RESORTS
    errs = validate(obj, expected_count=expected)
    for e in errs:
        print(f"INVALID: {e}")
    rc = 1 if errs else 0
    if args.health:
        probs = health(obj, max_age_h=args.max_age_h)
        for p in probs:
            print(f"HEALTH: {p}")
        if probs:
            rc = 1
    if rc == 0:
        print(f"OK: {len(obj['resorts'])} resorts, schema_version {obj['schema_version']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
