# Snow forecast dashboard v2 redesign: implementation spec

Status: ready to build. Date: 27 November 2026.
Owner of this spec: lead architect. Builders: one data engineer, one frontend engineer.

Read this file top to bottom once. Then read only your own parts:

| Builder | Must read | Owns these files (nobody else edits them) |
|---|---|---|
| Data engineer | 0, 1, 2, 5 (data items) | `scripts/fetch_data_v2.py`, `scripts/validate_forecasts_v2.py` (new), `.github/workflows/refresh_v2.yml`, `data/forecasts_v2.json`, `tests/test_fetch_v2.py` (new) |
| Frontend engineer | 0, 1, 3, 4, 5 (page items), and `docs/design_brief_v2.md` sections 1, 2, 11 | `dashboard_v2/app.py`, `dashboard_v2/theme.py` (new), `dashboard_v2/render.py` (new), `dashboard_v2/data.py` (new), `dashboard_v2/requirements.txt` (new), `.streamlit/config.toml`, `tests/fixtures/forecasts_v2_fixture.json` (new), `tests/test_dashboard_v2.py` (new) |

The JSON file in section 1 is the only contract between the two of you. If something is not in section 1, the frontend must not expect it and the data engineer must not invent it.

---

## 0. Ground rules that apply to both builders

1. No made-up numbers. If an input is missing, the output is `null`. Never fill with zero, an average, a neighbour's value or a model value dressed up as a measurement. This is the project's hard rule (CLAUDE.md, "Data Integrity").
2. Zero is a real value and must survive. A forecast of 0 cm is `0.0`, a station reading of 0 cm is `0.0`. Code must test `is None`, never truthiness (the current app does `if base_depth else None`, which turns 0 into missing; that bug must not come back).
3. Three-valued flags. Every boolean flag in the JSON is `true`, `false` or `null`. When a flag is an OR of several conditions: `true` if any known condition is true; `false` only if every condition is known and false; otherwise `null`. When a flag is an AND: `false` if any known condition is false; `true` only if every condition is known and true; otherwise `null`.
4. Model output is labelled as model output, measurements as measurements, resort reports as resort reports. The `kind` field (section 1.6) carries this, and the page shows it in words.
5. Real tests only, no mocks (user rule). Tests hit the real APIs with a small resort subset.
6. Units in the JSON are metric and fixed: cm for snow, mm for rain and precipitation, °C, km/h, metres, seconds, percent. The page converts to inches/feet/°F only for display.
7. Precision: the JSON keeps snow and rain to 1 decimal, temperatures to 1 decimal, elevations and freezing levels to whole metres, wind to whole km/h. The page rounds for display (section 4.8).

---

## 1. JSON schema for `data/forecasts_v2.json` (schema_version 3)

Same path as today. The file is rewritten whole on every run, atomically (write `forecasts_v2.json.tmp`, then `os.replace`). All old top-level and per-resort keys stay, with the corrected meanings given in 1.3, so an old reader does not crash. Everything new is added alongside.

Write it compact: `json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`. Any NaN must already have been turned into `null`. Expected size is roughly 0.5 to 1 MB for 86 resorts.

Type notation below: `str`, `int`, `float`, `bool`, `T|null` (nullable), `[T]` (array), `"a"|"b"` (enum string). Times are ISO 8601. `*_utc` fields end in `Z`. `*_local` fields have no offset and are in the resort's `timezone`.

### 1.1 Top level

```
{
  "schema_version": 3,                                   int, always 3
  "updated": str,                                        UTC finish time, "2026-11-27T01:19:26Z" (kept from v2)
  "run": Run,                                            1.2
  "thresholds": Thresholds,                              1.2, the display-rule constants actually used
  "sources": Sources,                                    1.2
  "resorts": [Resort]                                    1.3, one per resort in the fetcher's list, always all of them
}
```

### 1.2 Run, thresholds, sources

```
Run {
  "started_utc": str,
  "finished_utc": str,
  "duration_s": int,
  "resort_count": int,                                   len(resorts)
  "open_meteo": {
    "requests": int,                                     HTTP requests sent, retries included
    "retries": int,
    "failed_chunks": [str]                               chunk labels that failed after all retries, e.g. "daily:3"
  },
  "coverage": {
    "forecast": int,                                     resorts with quality.forecast_ok == true
    "depth": int,                                        resorts with depth.value_cm != null
    "depth_by_provider": {"SNOTEL": int, "GeoSphere": int, "SLF": int, "OnTheSnow": int, "tenki.jp": int},
    "stations_measured": int                             station rows (all resorts) with depth_cm != null
  },
  "warnings": [str]                                      plain sentences, e.g. "OnTheSnow parsed 0 of 52 pages"
}

Thresholds {                                             constants copied from the fetcher so the page can print them
  "wind_gust_hold_kmh": 60,
  "wind_700hpa_hold_kmh": 70,
  "storm_3day_cm": 15,
  "storm_codes": [73, 75, 85, 86],
  "storm_prob_pct": 50,
  "storm_day_min_cm": 10,
  "powder_cold_below_c": -8,
  "powder_warm_above_c": -3,
  "snowy_day_min_cm": 1.0,
  "bluebird_prev24_cm": 10,
  "bluebird_codes": [0, 1, 2],
  "flat_light_visibility_m": 1000,
  "flat_light_cloud_pct": 90
}

Sources {
  "base_depth": {"USA": str, "Austria": str, "Switzerland": str, "Japan": str, "France": str, "Spain": str},
                                                         kept for old readers; now written from what actually succeeded
                                                         this run, e.g. "OnTheSnow resort reports (14 of 22), SNOTEL stations"
  "forecast": "Open-Meteo (open-meteo.com), best_match model",
  "list": [
    {"id": "open-meteo", "name": "Open-Meteo", "url": "https://open-meteo.com", "kind": "model",
     "used_for": "forecasts, recent modelled snowfall, freezing level, wind"},
    {"id": "snotel", "name": "USDA NRCS SNOTEL (AWDB)", "url": "https://wcc.sc.egov.usda.gov", "kind": "measured", "used_for": "..."},
    {"id": "geosphere", "name": "GeoSphere Austria TAWES", "url": "https://dataset.api.hub.geosphere.at", "kind": "measured", "used_for": "..."},
    {"id": "slf", "name": "WSL Institute for Snow and Avalanche Research SLF, IMIS", "url": "https://measurement-api.slf.ch", "kind": "measured", "used_for": "..."},
    {"id": "onthesnow", "name": "OnTheSnow", "url": "https://www.onthesnow.com", "kind": "resort_report", "used_for": "..."},
    {"id": "tenki", "name": "tenki.jp", "url": "https://tenki.jp", "kind": "resort_report", "used_for": "..."}
  ]                                                      include a source only if it returned data for at least one resort this run
}
```

### 1.3 Resort

```
Resort {
  // identity
  "id": str,                       url-safe slug of name: lowercase, ASCII-folded, non-alphanumerics -> "-",
                                   e.g. "alta", "val-d-isere", "solden", "st-anton". Unique. Stable between runs.
  "name": str,                     display name, unchanged from fetcher list ("Sölden", "Val d'Isère")
  "country": "USA"|"Austria"|"Switzerland"|"Japan"|"France"|"Spain",
  "region": str,                   state / province / range from the fetcher list
  "lat": float, "lon": float,      from the fetcher list
  "timezone": str|null,            IANA name returned by Open-Meteo for the base point, e.g. "America/Denver"
  "utc_offset_s": int|null,        Open-Meteo utc_offset_seconds for the base point at run time

  // elevations (metres). Configuration, not data: see 2.3 for where they come from
  "elevation_m": int,              base elevation (legacy key, kept)
  "summit_elevation_m": int|null,  published top-lift elevation, null if not sourced (legacy key, meaning corrected)
  "mid_elevation_m": int|null,     round((elevation_m + summit_elevation_m) / 2), null if summit is null
  "elevation_note": str|null,      citation for base/summit, e.g. "alta.com/mountain-stats"

  // legacy v2 keys, kept so an old reader still works, values now honest
  "base_depth_cm": float|null,     == depth.value_cm
  "base_depth_source": str|null,   == depth.source_label
  "temp_c": float|null,            == now.base.temp_c
  "summit_temp_c": float|null,     == now.summit.temp_c (real summit-elevation call, no lapse rate)
  "forecast": [                    days 0..6 at BASE elevation, exactly as v2
    {"day": int, "date": "YYYY-MM-DD", "new_snow_cm": float|null, "source": "open-meteo"}
  ],
  "seven_day_total_cm": float|null,         sum of forecast[].new_snow_cm; null if any day is null
  "summit_seven_day_total_cm": float|null,  == week.summit_cm

  // new
  "depth": Depth,                  1.4
  "stations": [Station],           1.5, may be empty
  "recent": Recent,                1.6
  "now": Now,                      1.7
  "next_72h": Next72,              1.8
  "days": [Day],                   1.9, local dates from today-3 to today+14, ascending, 18 entries when complete
  "week": Week,                    1.10
  "extended": Extended,            1.10
  "flags": Flags,                  1.11
  "models": Models|null,           1.12
  "model_cells": {                 1.13; the grid cell each elevation's hourly call landed on (response latitude,
    "base": Cell|null, "mid": Cell|null, "summit": Cell|null      longitude, elevation), null when that call is absent
  },
  "quality": Quality               1.13
}
Cell {"lat": float, "lon": float, "model_elevation_m": int}
```

### 1.4 Depth (the one "base depth" number shown on the page)

```
Depth {
  "value_cm": float|null,
  "kind": "resort_report"|"measured"|null,
  "provider": "OnTheSnow"|"tenki.jp"|"SNOTEL"|"GeoSphere"|"SLF"|null,
  "source_label": str|null,        short words for the page, e.g. "Resort report via OnTheSnow",
                                   "SNOTEL Snowbird station, 2,795 m, 2 km away"
  "observed_utc": str|null,        when the reading was taken or the report updated; null if unknown
  "report": {                      only when kind == "resort_report", else null
    "base_cm": float|null, "mid_cm": float|null, "summit_cm": float|null,
    "last24_cm": float|null, "last48_cm": float|null, "last72_cm": float|null
  }|null
}
```

Selection order (first non-null wins; no blending, no averaging):

| Country | 1st | 2nd | 3rd |
|---|---|---|---|
| USA | OnTheSnow report: `mid_cm`, else `base_cm` (never summit) | nearest SNOTEL station in `stations` with non-null `depth_cm` | null |
| Austria | GeoSphere station `depth_cm` | null | |
| Switzerland | SLF station `depth_cm` | null | |
| Japan | OnTheSnow report (`mid_cm`, else `base_cm`) | tenki.jp strict match (2.6) | null |
| France, Spain | OnTheSnow report (`mid_cm`, else `base_cm`) | null | |

The Open-Meteo modelled snow depth is never used here (section 2.7 removes it).

### 1.5 Station (measured, one row per station; never averaged)

```
Station {
  "provider": "SNOTEL"|"GeoSphere"|"SLF",
  "id": str,                       "766:UT:SNTL", "11110", "ADE2"
  "name": str,                     from the provider's metadata endpoint, not from our hardcoded label
  "lat": float, "lon": float,      provider metadata
  "elevation_m": int|null,         provider metadata (SNOTEL feet * 0.3048, rounded)
  "distance_km": float,            great-circle distance from resort lat/lon, 1 decimal (haversine, R = 6371.0 km)
  "depth_cm": float|null,          latest reading
  "observed_utc": str|null,        time of that reading for SLF and GeoSphere; always null for SNOTEL, whose daily
                                   values have a date but no time
  "observed_date": "YYYY-MM-DD"|null,   date of that reading (all providers)
  "change_24h_cm": float|null,     see formulas
  "change_48h_cm": float|null,     SNOTEL only, else null
  "change_72h_cm": float|null,     SNOTEL only, else null
  "swe_change_24h_mm": float|null, SNOTEL only (WTEQ), else null
  "swe_mm": float|null,            SNOTEL only, latest WTEQ in mm
  "series": [{"date": "YYYY-MM-DD", "depth_cm": float|null}]
                                   SNOTEL: the last 15 days, one entry per calendar day, null for days with no value.
                                   GeoSphere and SLF: [] (no daily history verified)
}
```

Sort `stations` by `distance_km` ascending.

Formulas (all null if any input is null):

- SNOTEL, with `D[t]` = daily SNWD in cm for date t (inches * 2.54) and `latest` = the most recent date with a non-null SNWD inside the 15-day window:
  - `change_24h_cm = D[latest] - D[latest - 1 day]`
  - `change_48h_cm = D[latest] - D[latest - 2 days]`
  - `change_72h_cm = D[latest] - D[latest - 3 days]`
  - `swe_change_24h_mm = (W[latest] - W[latest - 1 day]) * 25.4` with W = daily WTEQ in inches.
  - These are depth changes, not new snowfall; settling makes them negative. The page labels them "depth change".
- SLF, with readings sorted by `measure_date` and `HS` in cm: let `t1` = latest reading with non-null HS, `t0` = the reading whose time is closest to `t1 - 24h`. If `|t0 - (t1 - 24h)| > 2h`, `change_24h_cm = null`, else `HS[t1] - HS[t0]`.
- GeoSphere, with 10-minute SCHNEE readings for the last 72 h: `depth_cm` = latest non-null value; `change_24h_cm = S[t1] - S[t0]` where `t0` is the non-null reading closest to `t1 - 24h`, null if more than 1 h away.

### 1.6 Recent (last 24, 48, 72 hours)

```
Recent {
  "measured": {                    null when no measurement exists for this resort
    "kind": "depth_change"|"resort_reported_new_snow",
    "provider": "SNOTEL"|"SLF"|"GeoSphere"|"OnTheSnow",
    "label": str,                  e.g. "Depth change at Snowbird station" or "New snow reported by the resort"
    "h24_cm": float|null, "h48_cm": float|null, "h72_cm": float|null
  }|null,
  "modelled": {                    always "kind": "model"
    "kind": "model",
    "base":   {"h24_cm": float|null, "h48_cm": float|null, "h72_cm": float|null},
    "mid":    {...same}|null,      null if mid_elevation_m is null
    "summit": {...same}|null       null if summit_elevation_m is null
  }
}
```

`measured` selection: OnTheSnow `last24/48/72` if the report exists (kind `resort_reported_new_snow`), else the nearest station's `change_*` (kind `depth_change`), else null.

`modelled` formula: let `H` = the run time converted to the point's local time with `utc_offset_seconds`, floored to the hour. Open-Meteo hourly snowfall at timestamp `t` is the sum over the hour ending at `t`. Then `hN_cm = sum(hourly.snowfall[t] for t in (H - N h, H])`. Null if any of those hourly values is null or missing.

### 1.7 Now

```
Now {
  "local_time": str|null,          H from 1.6, e.g. "2026-11-27T08:00"
  "base":   Point|null,
  "summit": Point|null
}
Point {                            from the Open-Meteo `current` block of that elevation's call
  "temp_c": float|null, "apparent_temp_c": float|null,
  "weather_code": int|null,
  "wind_kmh": int|null, "gust_kmh": int|null,
  "freezing_level_m": int|null,
  "snowfall_cm": float|null,       current-interval snowfall
  "is_day": bool|null
}
```

### 1.8 Next 72 hours (the hero number)

```
Next72 {
  "base_cm": float|null, "mid_cm": float|null, "summit_cm": float|null,
  "summit_by_period": [            summit snowfall in 6-hour blocks, 12 entries, for the sentence "most of it falls ..."
    {"start_local": "YYYY-MM-DDTHH:00", "label": "night"|"morning"|"afternoon"|"evening", "snowfall_cm": float|null}
  ],
  "precip_prob_max_pct": int|null, max of hourly precipitation_probability (base call) over the same 72 h
  "window_start_local": str|null,  H, e.g. "2026-11-27T03:00" (the window is (H, H+72h])
  "window_end_local": str|null,    H + 72 h, e.g. "2026-11-30T03:00"; the page names this when it labels the hero number
  "by_day": {                      per calendar day touched by (H, H+72h], same hourly arrays as X_cm, so the
    "base": [Seg], "mid": [Seg], "summit": [Seg]   snowfall_cm of a band's segments sums to X_cm (within rounding)
  },
  "mid_equals_summit": bool|null,  hourly snowfall, rain and precipitation identical for mid and summit over the window
  "base_equals_mid": bool|null     same for base and mid
}
Seg {
  "date": "YYYY-MM-DD", "hours": int,          hours of this date inside the window
  "snowfall_cm": float|null, "rain_mm": float|null, "precip_mm": float|null,   sums of that band's hourly values
  "phase": "snow"|"mixed"|"rain"|"none"|null,  1.9.2, over these hours only
  "rain_phase_mm": float|null, "mixed_phase_mm": float|null                    1.9.2 water totals
}
```

Formula: `X_cm = sum(hourly.snowfall[t] for t in (H, H + 72 h])` for that elevation's call; null if any value is null. Blocks are aligned to local 00, 06, 12, 18: `night` covers hours ending 01..06, `morning` 07..12, `afternoon` 13..18, `evening` 19..24. The first and last blocks may be partial; mark them by including only the hours inside (H, H+72h].

### 1.9 Day

One entry per local calendar date. `index` is days from today (local, at run time): -3, -2, -1, 0, 1 ... 14. Day 15 from the API is always null and is dropped.

```
Day {
  "date": "YYYY-MM-DD",
  "index": int,
  "horizon": "past_model"|"short"|"extended",       index < 0, 0..6, 7..14
  "sunrise_local": str|null, "sunset_local": str|null,   from base call
  "daylight_s": int|null,                           sunset - sunrise
  "base":   DayPoint|null,
  "mid":    DayPoint|null,
  "summit": DayPoint|null,
  "derived": Derived                                1.9.1
}
DayPoint {                                          straight from Open-Meteo `daily` for that elevation's call
  "snowfall_cm": float|null,                        snowfall_sum
  "rain_mm": float|null,                            rain_sum
  "precip_mm": float|null,                          precipitation_sum
  "precip_hours": float|null,                       precipitation_hours
  "precip_prob_pct": int|null,                      precipitation_probability_max
  "temp_max_c": float|null, "temp_min_c": float|null,
  "wind_max_kmh": int|null, "gust_max_kmh": int|null,
  "wind_dir_deg": int|null,                         wind_direction_10m_dominant
  "weather_code": int|null,
  "sunshine_s": int|null,                           sunshine_duration
  "uv_max": float|null
}
```

For `horizon == "past_model"` only `snowfall_cm`, `rain_mm` and `precip_mm` are kept in each DayPoint (others null); these are model values, not observations.

#### 1.9.1 Derived (per day)

Every field is null when `horizon` is `past_model`. Fields marked (hourly) need the hourly call, which covers today-3 to today+6, so they are null for `extended` days. "Hours of day d" means hourly timestamps whose local date is d (00:00 to 23:00). "Lift hours" means hourly timestamps t with `sunrise_local < t <= sunset_local` of that day.

```
Derived {
  "freezing_level_max_m": int|null,            (hourly) max hourly freezing_level_height over hours of d (base call)
  "freezing_level_precip_min_m": int|null,     (hourly) min freezing_level_height over hours of d with base precipitation > 0;
  "freezing_level_precip_max_m": int|null,     (hourly) max of the same; both null if no hour has precipitation > 0
  "snowfall_height_precip_min_m": int|null,    (hourly) same as above using snowfall_height; null wherever the API returns null
  "snowfall_height_precip_max_m": int|null,    (hourly)   (always null in the US; never filled from freezing level)
  "rain_risk": {"base": bool|null, "mid": bool|null, "summit": bool|null},     1.9.2
  "mixed_risk": {"base": bool|null, "mid": bool|null, "summit": bool|null},    1.9.2
  "phase": {"base": str|null, "mid": str|null, "summit": str|null},            (hourly) 1.9.2: snow|mixed|rain|none
  "rain_phase_mm": {"base": float|null, "mid": float|null, "summit": float|null},   (hourly) 1.9.2
  "rain_limit_m": int|null,                    (hourly) highest rain line in hours it rains at the base, see below
  "rain_limit_kind": "snow_line"|"freezing_level_minus_250"|null,
  "powder_temp_c": float|null,                 (hourly)
  "powder_band": "cold_dry"|"average"|"warm_heavy"|null,
  "wind_700hpa_lift_max_kmh": int|null,        (hourly)
  "wind_hold": bool|null,
  "wind_gust_lift_max_kmh": int|null,          (hourly) max hourly wind_gusts_10m over lift hours (summit call)
  "cloud_cover_lift_mean_pct": int|null,       (hourly) mean hourly cloud_cover over lift hours (summit call)
  "flat_light_hours": int|null,                (hourly)
  "wind_chill_min_c": float|null,              (hourly)
  "lift_hours_snowfall_cm": float|null,        (hourly) summit
  "ski_score_cm": float|null,
  "bluebird": bool|null
}
```

Formulas. Inputs come only from the Open-Meteo responses described in 2.4. If any input is null, the output is null, subject to the three-valued rule in section 0.

| Field | Formula |
|---|---|
| `freezing_level_*` | As described in the block above. Round to whole metres. |
| `rain_risk.E`, `mixed_risk.E`, `phase.E`, `rain_phase_mm.E` | See 1.9.2 (hour by hour, on band E's own hourly call). For extended days (no hourly): `rain_risk.E` = `mixed_risk.E` = `false` if `days[d].E.rain_mm == 0`, otherwise `null`; `phase.E` and `rain_phase_mm.E` are null. |
| `rain_limit_m` | Over hours t of d where the base call's hourly `rain >= 0.1` and the base elevation is below `line(t)` (1.9.2): `max(line(t))`. `rain_limit_kind` is `snow_line` only if every such hour used `snowfall_height`. Null if there is no such hour. It is a classification threshold, not a measured snow line; never write it into `snowfall_height_*`. |
| `powder_temp_c` | Only if `days[d].summit.snowfall_cm >= thresholds.snowy_day_min_cm`. Then `sum(T(t) * S(t)) / sum(S(t))` over hours of d, with T = hourly temperature_2m and S = hourly snowfall from the summit call. Null otherwise, or if `sum(S) == 0`. 1 decimal. |
| `powder_band` | `cold_dry` if `powder_temp_c < -8`; `average` if `-8 <= powder_temp_c <= -3`; `warm_heavy` if `> -3`; null if `powder_temp_c` is null. It is a temperature band, not a density; never compute a snow-to-liquid ratio (Open-Meteo snowfall is a fixed 0.7 cm per mm of precipitation). |
| `wind_700hpa_lift_max_kmh` | Max of hourly wind_speed_700hPa over lift hours (summit call). |
| `wind_hold` | Short days: OR of `wind_gust_lift_max_kmh > 60` and (only when `summit_elevation_m >= 2500`, `thresholds.wind_700hpa_min_summit_m`) `wind_700hpa_lift_max_kmh > 70`; for lower summits the 700 hPa term is `false`, because 700 hPa (about 2,800 to 3,100 m) is free air well above those lifts. Daily `gust_max_kmh` is not used on short days because it includes overnight gusts when no lift runs. Extended days: `days[d].summit.gust_max_kmh > 60` only, so they can be `true` or `null`, never `false`. These are display rules, not predictions of lift closures. |
| `flat_light_hours` | Count of lift hours where summit hourly `visibility < 1000` OR `max(cloud_cover_low, cloud_cover_mid) > 90`. High thin cloud (in total `cloud_cover` only) does not flatten light. Null if any lift-hour value of these variables is null. |
| `wind_chill_min_c` | Min of hourly apparent_temperature over lift hours (summit call). |
| `lift_hours_snowfall_cm` | Sum of hourly snowfall over lift hours (summit call). |
| `ski_score_cm` | Only for index 0..6: `days[d-1].summit.snowfall_cm + lift_hours_snowfall_cm`. For index 0, `days[-1]` is the past_model day (model value, acceptable here because the score is a ranking aid, labelled as such). Set to null if `wind_hold == true`, or `rain_risk.mid == true` on day d, or `rain_risk.mid == true` on day d-1 (the counted snow was rained on); use `rain_risk.summit` when mid is null. Day -1 has null rain_risk, so it never nulls index 0. For index 0, also null if the run's local time H is after `sunset_local`. |
| `bluebird` | AND of: `days[d-1].summit.snowfall_cm >= 10`; `days[d].summit.snowfall_cm < 1.0` (`snowy_day_min_cm`, today is dry); `days[d].summit.weather_code in {0,1,2}`; `cloud_cover_lift_mean_pct <= 40` (`bluebird_cloud_mean_pct`); `flat_light_hours <= 2` (`bluebird_flat_light_max_h`). Index 0..6 only. `sunshine_duration` is not used: Open-Meteo derives it from modelled radiation and it reported 6 of 9 daylight hours of sun on a day whose hourly cloud cover was 100% throughout. |

#### 1.9.2 Precipitation phase (rain, mixed, snow)

Why this exists. Open-Meteo's `&elevation=` picks a grid cell and adjusts only temperature for height (a fixed 0.65 C per 100 m). Precipitation, and the rain/snow split inside it, are the grid cell's values. The split is re-done only where the adjusted point is warmer than about +1 C, which can turn snow into rain but never rain into snow. So `rain` at a mid or summit point is often the lower cell's rain: Crystal Mountain's 2,131 m summit got 11.8 mm of "rain" on a day its own temperature stayed at -3 to -4 C. Mid and summit often land on the same cell and get identical precipitation (1.8 `mid_equals_summit`, quality.bands). The point's lapse-rate temperature is not a reliable phase signal either: Timberline's 1,829 m base read -0.1 to -2 C with the freezing level 300 to 800 m above it. The freezing level is a property of the model column and is the best phase signal available. Snowfall and rain amounts are always shown exactly as the API returns them; only the phase label is derived.

For each hour t with hourly `precipitation(E,t) > 0`, on band E's own hourly call (each band can land on a different cell; the freezing level differed by up to 1,190 m between Alta's base and summit cells):

- `line(t)` = `snowfall_height(t)` when non-null (Europe), otherwise `freezing_level_height(t) - 250` (`snowline_below_fl_m`; snow usually reaches 200 to 300 m below the freezing level; always the fallback in the US).
- liquid = `min(rain(E,t), precipitation(E,t))`, frozen = `precipitation - liquid`.
- If `elev(E) < line(t)`: liquid counts as **rain**, frozen counts as **mixed**.
- Otherwise: liquid counts as **mixed** (wet snow or rain), frozen counts as **snow**.
- If `precipitation`, `rain`, or both `snowfall_height` and `freezing_level_height` are null for a wet hour, every output for the period is null.

Over a period (a day, or a `by_day` segment), with `total` = sum of hourly precipitation and a phase *significant* when its water is `>= 1.0 mm` (`phase_min_mm`) and `>= 20%` of `total` (`phase_share_min`):

- `rain_risk.E` = rain is significant. `mixed_risk.E` = mixed is significant, or `phase.E == "mixed"`.
- `phase.E` = `"none"` if `total < 0.5 mm` (`phase_day_min_mm`); otherwise the single significant phase; `"mixed"` if two or more are significant or mixed is; if none is significant, the phase with the most water.
- `rain_phase_mm.E` = rain water, 1 decimal.

Checked on the 2026-11-27 run: Crystal summit Sunday (freezing level 1,920 to 2,260 m, so line 1,670 to 2,010 m, below 2,131 m) gives `phase = mixed`, `rain_risk = false`; mid (1,736 m) gives `rain_risk = true` because the line sat at 1,910 to 2,010 m from 09:00 to 13:00; base gives `phase = rain`. |

### 1.10 Week and extended

```
Week {                                   index 0..6
  "base_cm": float|null, "mid_cm": float|null, "summit_cm": float|null,
                                         sum of days[0..6].E.snowfall_cm, null if any of the 7 is null
  "best_snow_day": int|null,             index (0..6) of max days[i].summit.snowfall_cm; earliest index wins ties;
                                         null if every value is 0 or any is null
  "best_ski_day": int|null,              index (0..6) of max ski_score_cm among non-null scores;
                                         ties broken by higher days[i].summit.sunshine_s, then earliest index;
                                         null if no score > 0
  "best_ski_day_score_cm": float|null
}
Extended {                               index 7..14, labelled lower confidence on the page
  "summit_cm": float|null, "base_cm": float|null,
                                         sum over index 7..14, null if any is null
  "days_available": int                  how many of indices 7..14 exist in `days` (normally 8)
}
```

### 1.11 Flags

```
Flags {
  "storm_incoming": bool|null,
  "storm_reasons": [{"index": int, "date": "YYYY-MM-DD", "reason": "3-day total"|"heavy snow code"}],
  "bluebird_days": [int],                indices with derived.bluebird == true
  "wind_hold_days": [int],               indices 0..14 with derived.wind_hold == true
  "rain_days": [int],                    indices 0..6 with derived.rain_risk.mid == true (summit when mid is null)
  "mixed_days": [int]                    indices 0..6 with derived.mixed_risk.mid == true (summit when mid is null)
}
```

`storm_incoming` = OR of:
- `days[1].summit.snowfall_cm + days[2].summit.snowfall_cm + days[3].summit.snowfall_cm >= 15` (null if any of the three is null). If true, add `{index: 1, reason: "3-day total"}`.
- for any index 0..6: `days[i].summit.weather_code in {73, 75, 85, 86}` AND `days[i].summit.precip_prob_pct >= 50` AND `days[i].summit.snowfall_cm >= 10` (`storm_day_min_cm`). Add one reason per triggering day. Each of the three inputs follows three-valued logic: a missing value makes that day's condition null (unknown) unless another input is already known false. The snowfall requirement was added after QA found the code-and-probability rule alone flagged 33 of 86 resorts, including resorts forecasting 0 cm.

If summit is null for this resort, compute both conditions from `base` instead. The page knows this from `summit_elevation_m == null` and labels the elevation as base.

### 1.12 Models (optional model agreement)

```
Models {
  "compared": ["best_match", "ecmwf_ifs025"] (+ "icon_d2" for Austria and Switzerland),
  "days": [{"index": int, "snowfall_cm": {"best_match": float|null, "ecmwf_ifs025": float|null, "icon_d2": float|null}}],
                                         index 0..6, summit elevation
  "agree_through": int|null              highest index k such that for every i in 0..k all non-null values are within
                                         max(3.0, 0.3 * larger value) cm of each other and at least two are non-null;
                                         null if day 0 already fails or has fewer than two values
}
```

Keep the nulls that short-range models return at the end of their range; never fill them. If the verification in 2.4 step 5 fails, write `"models": null` for every resort and the page falls back to its horizon sentence.

### 1.13 Quality

```
Quality {
  "forecast_ok": bool,                   true when base daily snowfall is non-null for all of index 0..6
  "summit_forecast_ok": bool,
  "depth_ok": bool,                      depth.value_cm != null
  "bands": {                             whether the model can separate the elevations (1.9.2). Facts only, never filled:
    "mid_summit_same_cell": bool|null,   model_cells.mid lat/lon == model_cells.summit lat/lon
    "mid_summit_same_72h": bool|null,    == next_72h.mid_equals_summit
    "mid_summit_same_week": bool|null,   days[0..6] mid and summit snowfall_cm, rain_mm and precip_mm all equal
    "base_mid_same_72h": bool|null       == next_72h.base_equals_mid
  },                                     When mid_summit_same_72h is true a note is added: "Mid-mountain and summit get the
                                         same forecast precipitation for the next three days; the model cannot separate
                                         these heights here". The page merges the two bands (screen 4) instead of showing
                                         two identical numbers. Never invent a separate summit value.
  "notes": [str]                         plain sentences, e.g. "Summit elevation not sourced, summit forecast skipped",
                                         "OnTheSnow page returned no snow object", "SNOTEL 332:UT:SNTL is Ben Lomond Peak, removed"
}
```

### 1.14 Shape example (types only, no values)

The frontend engineer builds `tests/fixtures/forecasts_v2_fixture.json` from this shape for development before real data exists. The fixture must carry `"fixture": true` at the top level, and the app must refuse to load a file with that key unless the environment variable `SNOW_FIXTURE=1` is set.

```json
{
  "schema_version": 3, "updated": "<utc>", "run": {"...": "..."}, "thresholds": {"...": "..."}, "sources": {"...": "..."},
  "resorts": [{
    "id": "alta", "name": "Alta", "country": "USA", "region": "Utah", "lat": 0.0, "lon": 0.0,
    "timezone": "America/Denver", "utc_offset_s": 0,
    "elevation_m": 0, "summit_elevation_m": 0, "mid_elevation_m": 0, "elevation_note": "<cite>",
    "base_depth_cm": null, "base_depth_source": null, "temp_c": null, "summit_temp_c": null,
    "forecast": [{"day": 0, "date": "<date>", "new_snow_cm": 0.0, "source": "open-meteo"}],
    "seven_day_total_cm": 0.0, "summit_seven_day_total_cm": 0.0,
    "depth": {"value_cm": null, "kind": null, "provider": null, "source_label": null, "observed_utc": null, "report": null},
    "stations": [{"provider": "SNOTEL", "id": "766:UT:SNTL", "name": "Snowbird", "lat": 0.0, "lon": 0.0, "elevation_m": 0,
                  "distance_km": 0.0, "depth_cm": null, "observed_utc": null, "observed_date": null,
                  "change_24h_cm": null, "change_48h_cm": null, "change_72h_cm": null,
                  "swe_change_24h_mm": null, "swe_mm": null, "series": []}],
    "recent": {"measured": null, "modelled": {"kind": "model", "base": {"h24_cm": null, "h48_cm": null, "h72_cm": null}, "mid": null, "summit": null}},
    "now": {"local_time": null, "base": null, "summit": null},
    "next_72h": {"base_cm": null, "mid_cm": null, "summit_cm": null, "summit_by_period": [], "precip_prob_max_pct": null},
    "days": [{"date": "<date>", "index": 0, "horizon": "short", "sunrise_local": null, "sunset_local": null, "daylight_s": null,
              "base": null, "mid": null, "summit": null,
              "derived": {"freezing_level_max_m": null, "freezing_level_precip_min_m": null, "freezing_level_precip_max_m": null,
                          "snowfall_height_precip_min_m": null, "snowfall_height_precip_max_m": null,
                          "rain_risk": {"base": null, "mid": null, "summit": null},
                          "powder_temp_c": null, "powder_band": null, "wind_700hpa_lift_max_kmh": null, "wind_hold": null,
                          "flat_light_hours": null, "wind_chill_min_c": null, "lift_hours_snowfall_cm": null,
                          "ski_score_cm": null, "bluebird": null}}],
    "week": {"base_cm": null, "mid_cm": null, "summit_cm": null, "best_snow_day": null, "best_ski_day": null, "best_ski_day_score_cm": null},
    "extended": {"summit_cm": null, "base_cm": null, "days_available": 0},
    "flags": {"storm_incoming": null, "storm_reasons": [], "bluebird_days": [], "wind_hold_days": [], "rain_days": [], "mixed_days": []},
    "models": null,
    "quality": {"forecast_ok": false, "summit_forecast_ok": false, "depth_ok": false, "notes": []}
  }]
}
```

The fixture must include at least: one resort with every field filled, one with `summit_elevation_m: null`, one with `depth.value_cm: 0.0`, one with `depth.value_cm: null` and no stations, one with all-zero snowfall, one with `models` non-null, and enough resorts (at least 12, across at least 3 countries) to exercise the nearby list and grouping. Fixture numbers are test inputs only and never ship.

---

## 2. Fetcher changes (`scripts/fetch_data_v2.py`)

Keep the single file. Keep stdlib only (`urllib`, `json`, `re`, `math`, `datetime`, `zoneinfo`, `time`, `os`, `argparse`, `logging`). Python 3.11.

### 2.1 Keep

- All six resort lists and all 86 resorts, with their names, coordinates, regions and base elevations unchanged (US_SKI_AREAS 22, AUSTRIA 10, SWISS 11, JAPAN 10, FRANCE 20, SPAIN 13). Do not add or remove resorts. Hakuba Valley and Happo-One share coordinates and will show the same forecast; leave both and add a `quality.notes` line to each saying so.
- SNOTEL, GeoSphere and SLF as the measured sources, with the station ids already in the lists (subject to the validation in 2.5).
- OnTheSnow slug map and tenki.jp map, with the fixes in 2.6.

### 2.2 Remove

- The `importlib` load of `src/snowforecast/cache/elevation_bands.py` and every use of `get_summit_elevation`, `apply_lapse_rate`, `get_precip_type`, `adjust_new_snow`. Do not edit `elevation_bands.py` itself (v1 still imports it).
- `fetch_open_meteo` against `/v1/gfs` and the four duplicated `process_*` functions. Replace with one pipeline (2.8).
- `get_openmeteo_snow_depth` and the Austria modelled-depth fallback.
- Weighted multi-station averaging (`get_snotel_multi_station`) and the `weight` element of station tuples (ignore it; you may leave the third tuple element in place to keep the list diff small).
- `scrape_snow_forecast`, `SNOW_FORECAST_SLUGS` (dead code).
- The loose tenki.jp regex `積雪[^0-9]*(\d+)\s*cm` and the second fallback; it matches the site's "積雪が100cm以上" navigation link and produces a fake 100 cm for every Japanese resort.
- `datetime.utcnow()` and naive `datetime.now()`. Use `datetime.now(timezone.utc)` everywhere.

### 2.3 Elevations

Add one table `RESORT_ELEVATIONS = {name: (base_m, summit_m, note)}` in the fetcher.

- For every resort in `RESORT_VERTICAL_M` (in `elevation_bands.py`, which covers USA, Japan, France and Spain), copy `summit_m = base_m + RESORT_VERTICAL_M[name]` once into the table, with `note = "carried over from elevation_bands.RESORT_VERTICAL_M"`.
- Austria and Switzerland are not in `RESORT_VERTICAL_M`, and their list `elevation_m` values are station elevations, not base elevations (St. Anton's 2079 is the Galzig station; Sölden's 3340 is its top). For these 21 resorts, set `base_m` and `summit_m` from the resort's own published mountain stats, with the URL in `note`. Starting values to check, not to copy blindly:

| Resort | Base m | Summit m | Resort | Base m | Summit m |
|---|---|---|---|---|---|
| St. Anton | 1304 | 2811 | Zermatt | 1620 | 3883 |
| Sölden | 1350 | 3340 | Saas-Fee | 1800 | 3600 |
| Obergurgl | 1793 | 3080 | Verbier | 1500 | 3330 |
| Ischgl | 1377 | 2872 | St. Moritz | 1822 | 3303 |
| Kitzbühel | 800 | 2000 | Davos | 1560 | 2844 |
| Lech am Arlberg | 1450 | verify | Laax | 1100 | 3018 |
| Pitztal Glacier | 1740 | 3440 | Lenzerheide | 1229 | 2865 |
| Obertauern | 1630 | 2313 | Mürren | 1650 | 2970 |
| Zell am See | 757 | 2000 | Andermatt | 1444 | 2961 |
| Schladming | 745 | 2015 | Arosa | 1800 | 2653 |
| | | | Engelberg | 1050 | 3020 |

- If you cannot confirm a value from a published source, set that summit to `None`. A null summit means the summit and mid calls are skipped for that resort and every summit/mid field is null. Never fall back to `base + 800` (the old `DEFAULT_VERTICAL_M` rule is removed).
- For Austria and Switzerland, `elevation_m` in the output becomes the resort base from this table. The station elevation moves to `stations[].elevation_m`, read from the station metadata endpoint.

### 2.4 Open-Meteo calls

All three calls go to `https://api.open-meteo.com/v1/forecast` (not `/v1/gfs`; the env variable `OPEN_METEO_URL` may point a run at `https://previous-runs-api.open-meteo.com/v1/forecast`, which serves the same current forecast, when the main host's daily quota is exhausted for that IP), with the default `best_match` model. Points are batched: each resort contributes up to three points (base, mid, summit; mid and summit only when summit is not null). Send 30 points per request (10 resorts), as comma-separated `latitude`, `longitude` and `elevation` lists of equal length. The response is a JSON array in the same order; if a single point is sent, it is an object, so wrap it in a list.

Call A, daily (16 days plus 3 past days):

```
https://api.open-meteo.com/v1/forecast
  ?latitude=LAT1,LAT2,...&longitude=LON1,LON2,...&elevation=E1,E2,...
  &daily=snowfall_sum,rain_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,
         temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,
         wind_direction_10m_dominant,weather_code,sunrise,sunset,sunshine_duration,uv_index_max
  &forecast_days=16&past_days=3&timezone=auto
  &wind_speed_unit=kmh&precipitation_unit=mm&temperature_unit=celsius
```

Drop the last daily entry (day 16, always null). Map dates to `index` by comparing with the point's local "today" (run time plus `utc_offset_seconds`).

Call B, hourly plus current (7 days plus 3 past days):

```
https://api.open-meteo.com/v1/forecast
  ?latitude=...&longitude=...&elevation=...
  &hourly=snowfall,rain,precipitation,temperature_2m,apparent_temperature,cloud_cover,cloud_cover_low,cloud_cover_mid,visibility,
          wind_speed_10m,wind_gusts_10m,freezing_level_height,snowfall_height,precipitation_probability,
          wind_speed_700hPa,weather_code,is_day
  &current=temperature_2m,apparent_temperature,snowfall,weather_code,wind_speed_10m,wind_gusts_10m,
           freezing_level_height,is_day
  &forecast_days=7&past_days=3&timezone=auto
  &wind_speed_unit=kmh&precipitation_unit=mm&temperature_unit=celsius
```

Hourly data is used only to compute the fields in 1.6 to 1.9.1. Do not write hourly arrays into the JSON.

Call C, model agreement (summit points only, one point per resort with a summit):

```
https://api.open-meteo.com/v1/forecast
  ?latitude=...&longitude=...&elevation=SUMMIT1,...
  &daily=snowfall_sum&models=best_match,ecmwf_ifs025[,icon_d2]
  &forecast_days=7&timezone=auto
```

Add `icon_d2` only in batches made of Austria and Switzerland resorts. Read the suffixed keys (`snowfall_sum_best_match`, `snowfall_sum_ecmwf_ifs025`, `snowfall_sum_icon_d2`).

Before wiring Call C, do this verification once and record the result in your handoff note: send Call C for Alta (40.59, -111.64, elevation 3215) and Zermatt (46.02, 7.75, elevation 3883). The per-model daily keys were verified only in hourly form, at Zermatt. If either response lacks the suffixed daily keys, or ecmwf_ifs025 is all null, ship `"models": null` for all resorts and skip Call C.

Request handling (all three calls, plus every other HTTP call in the fetcher):

- One helper `http_json(url, *, attempts, backoff_s, timeout_s, headers=None)`. Set a `User-Agent` of `snowforecast/3 (+https://github.com/kavanaghpatrick/snowforecast)` for API calls; keep the browser UA for OnTheSnow and tenki.jp.
- A response counts as a failure when: a network or HTTP error happens, the body is not JSON, or the body is an object with `"error": true` (Open-Meteo returns HTTP 200 with `{"error":true,"reason":"The service is overloaded"}` at times; the current code does not catch this).
- Open-Meteo: 5 attempts, waits of 8, 16, 32, 60 seconds between them, 60-second timeout. Sleep 1 second between batches. On HTTP 429, wait 60 seconds before the next attempt.
- When a batch fails after all attempts, retry it once more as single-resort requests (3 attempts each). Any resort still failing gets null forecast fields, `quality.forecast_ok = false`, and its label goes into `run.open_meteo.failed_chunks`.
- Count requests and retries into `run.open_meteo`.

Expected volume per run: about 9 batches each for A and B, 9 for C, so about 27 requests and about 750 location-calls by Open-Meteo's weighting, well under the free tier's 10,000 a day at four runs a day.

### 2.5 Measured sources

SNOTEL, switched from metloom to the AWDB REST API (verified keyless JSON). Two requests for all US stations together:

```
https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?stationTriplets=T1,T2,...
https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=T1,T2,...&elements=SNWD,WTEQ&duration=DAILY&beginDate=YYYY-MM-DD&endDate=YYYY-MM-DD
```

`beginDate` = today (UTC) minus 15 days, `endDate` = today (UTC). Station metadata gives `name`, `elevation` (feet), `latitude`, `longitude`. 3 attempts, 5 and 15 seconds apart.

Station validation. Compare each hardcoded station label with the metadata `name`, case-insensitively. A mismatch means the triplet is wrong. Example found while writing this spec: Alta's list uses `332:UT:SNTL` labelled "Alta", but AWDB says 332:UT:SNTL is **Ben Lomond Peak**, at 41.376, -111.944, about 90 km north. On mismatch, drop that station from the resort, add a `quality.notes` line, and log a warning. Do not guess a replacement triplet in code; fixing the list is a one-time manual edit, and only with a triplet whose metadata name and coordinates you have checked against the same endpoint. Also drop any station more than 60 km from the resort, with a note.

GeoSphere (Austria). Current-endpoint SCHNEE was null for St. Anton's and Sölden's stations on 27 November, so read the last 72 hours instead:

```
https://dataset.api.hub.geosphere.at/v1/station/historical/tawes-v1-10min?parameters=SCHNEE&station_ids=ID1,ID2,...&start=YYYY-MM-DDTHH:MM&end=YYYY-MM-DDTHH:MM
https://dataset.api.hub.geosphere.at/v1/station/current/tawes-v1-10min/metadata
```

One request for all 10 stations (St. Anton and Lech share 11110, so 9 unique). The metadata gives `name`, `altitude`, `lat`, `lon`. When every SCHNEE value in 72 h is null, `depth_cm` is null and the resort has no Austrian depth. That is the honest result.

SLF (Switzerland). Download `https://measurement-api.slf.ch/public/api/imis/measurements` once per run (not once per resort as now) and `https://measurement-api.slf.ch/public/api/imis/stations` once for metadata (`label`, `elevation`, `lat`, `lon`). Filter by `station_code`.

### 2.6 Resort reports

OnTheSnow. The site moved to the Next.js App Router, so `<script id="__NEXT_DATA__">` no longer exists; that is why it returned nothing for all 86 resorts. The data is now in the page's RSC payload as escaped JSON. Verified on 27 November:

```
\"snow\":{\"base\":null,\"middle\":45.72,\"summit\":null,\"last24\":0,\"last48\":0,\"last72\":5.08}   (Vail)
\"snow\":{\"base\":80,\"middle\":null,\"summit\":150,\"last24\":4,\"last48\":6,\"last72\":6}         (Val Thorens)
```

- Find every match of `\\"snow\\":\{(.*?)\}` in the HTML, unescape (`.replace('\\"', '"')`), and parse as JSON. Use the first object that has a `base`, `middle` or `summit` key. Other `snow` objects on the page (`forecast72`, `{date, snow}`) are not the report.
- Values are cm. Store all six fields in `depth.report`. Reject a depth >= 600 cm (keep the existing guard). Note that all three depths can be null while `last24/48/72` are present (Niseko on 27 November); then `depth.value_cm` falls through to the next source, but `recent.measured` still uses the report's `last*` values.
- `observed_utc`: collect all `\"updatedAt\":\"...\"` values; if exactly one distinct value exists, use it; otherwise null.
- European slugs now redirect (`/france/val-thorens` returns 308 to `onthesnow.co.uk/northern-alps/val-thorens`). Python 3.11's `urllib` follows 308. Run a one-time slug check over all 81 entries in `ONTHESNOW_SLUGS`, record each final URL, and update the map so requests go straight to the final URL. Slugs that do not lead to a page with a report object are removed from the map and noted.
- Sleep 1 second between OnTheSnow requests. 2 attempts each.

tenki.jp. Keep only the strict pattern `class="depth">積雪<br><span>(\d+)</span>cm` (or the current equivalent if you verify one on a live page). If nothing matches, the result is null. If across a run every Japanese resort that reached tenki.jp returns the same value, discard all of them and add a run warning; this catches a navigation-link match.

### 2.7 Removing the modelled depth

There is no modelled snow depth anywhere in the output. Open-Meteo `snow_depth` disagreed with the Snowbird station by a factor of three on the same week and moved the wrong way on a snow day.

### 2.8 Pipeline shape

```
main(argv):
  args: --only "Alta,Zermatt,Val Thorens" (optional subset), --out PATH (default data/forecasts_v2.json), --no-write
  1. build RESORTS: one list of dicts from the six country lists + RESORT_ELEVATIONS + station tuples + slugs
  2. measured: snotel_batch(), geosphere_batch(), slf_once()        -> stations per resort
  3. reports:  onthesnow for resorts with a slug; tenki for Japan when no OnTheSnow depth
  4. open-meteo: calls A, B, C in batches                           -> raw per-point responses
  5. derive:   pure functions, one per section 1.6 to 1.12, each taking parsed arrays and returning the fields
  6. assemble: resort dicts in list order, legacy keys filled from new ones
  7. validate: validate_forecasts_v2.validate(obj) (2.9); on failure exit 2 and do NOT replace the file
  8. write:    tmp + os.replace; exit 0
```

Keep each derivation in its own small pure function (`modelled_recent`, `next_72h`, `derive_day`, `storm_flags`, `week_summary`, `model_agreement`) so tests can call them with real API responses saved during the test run.

### 2.9 Validation script (`scripts/validate_forecasts_v2.py`)

Importable `validate(obj) -> list[str]` (empty list means valid) and a CLI: `python scripts/validate_forecasts_v2.py data/forecasts_v2.json [--health]`.

`validate` (blocks the write) checks:
1. `schema_version == 3`, all top-level keys present.
2. Resort count equals the fetcher's list length (86), ids unique.
3. Every key in 1.3 present on every resort (null allowed where the schema allows null).
4. At least 90% of resorts have `quality.forecast_ok == true`.
5. No value anywhere is NaN or infinite; no negative snowfall, rain or precipitation.
6. No `depth.provider` value outside the enum; no source string contains "modeled" or "modelled" in `depth`.
7. Legacy keys equal their new counterparts (`base_depth_cm == depth.value_cm`, etc.).

`--health` (runs after the commit, fails the job to alert, does not block data) fails when any provider with at least 5 resorts assigned has 0 successes this run, or when any `run.warnings` entry exists.

### 2.10 Workflow (`.github/workflows/refresh_v2.yml`)

```yaml
name: Refresh Dashboard V2 Data
on:
  schedule:
    - cron: '0 0,6,12,18 * * *'
  workflow_dispatch:
concurrency:
  group: refresh-v2
  cancel-in-progress: false
jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Fetch forecast data
        run: python scripts/fetch_data_v2.py          # no continue-on-error; exit 2 leaves the old file in place
      - name: Commit and push if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -f data/forecasts_v2.json
          git diff --staged --quiet && exit 0
          git commit -m "Update forecast data [skip ci]"
          git pull --rebase -X ours origin main || { git rebase --abort; git pull --rebase origin main; }
          git push
      - name: Source health
        run: python scripts/validate_forecasts_v2.py data/forecasts_v2.json --health
```

No pip install step: the fetcher is stdlib only. Remove `metloom` and `pandas` from the workflow. A failed fetch or failed health check turns the run red, so GitHub emails the repo owner, which replaces today's silent `continue-on-error`.

After the code is in, run the fetcher locally once and commit the fresh `data/forecasts_v2.json` (the local copy is from 24 January with 53 resorts).

### 2.11 Tests (`tests/test_fetch_v2.py`, real network)

1. Run `main(["--only", "Alta,Snowbird,Zermatt,Davos,St. Anton,Val Thorens,Niseko United", "--out", tmp, ...])` against the live APIs; the output passes `validate` with the resort-count check relaxed to the subset.
2. Alta's stations do not include `332:UT:SNTL` (or the list has been corrected).
3. No Japanese resort has `depth.provider == "tenki.jp"` with value exactly 100.0 unless at least one other tenki.jp resort differs.
4. For Zermatt, `days[0..1].derived.snowfall_height_precip_*` may be non-null; for Alta they are null.
5. For every resort, `next_72h.summit_cm` is null exactly when `summit_elevation_m` is null or the summit call failed.
6. Pure-function tests on saved real responses: a day with 0 snowfall gives `powder_band == null`; `wind_hold` on an extended day is never `false`; `ski_score_cm` is null on a wind-hold day.

---

## 3. Dashboard information architecture

The page answers one question: where should I ski this week, and why. It is one long page about one resort, framed by how that resort compares with the rest. No sidebar, no tabs, no expanders. The order follows the design brief's eight screens, with the decisions below.

### 3.1 State and URLs

All state lives in `st.query_params`, so every view is a shareable link:

| Param | Values | Default |
|---|---|---|
| `resort` | a resort `id` | the resort with the highest `next_72h.summit_cm` across all resorts (falling back to `next_72h.base_cm` for resorts with no summit). Ties: highest `week.summit_cm`, then name. |
| `scope` | `nearby`, `country`, `all` | `nearby` |
| `sort` | `next3`, `week`, `best`, `depth`, `name` | `next3` |
| `units` | `cm`, `in` | `cm` |

Links in our HTML are `<a href="?resort=alta&scope=nearby&sort=next3&units=cm" target="_self">`, carrying the other current params. The resort selectbox writes `st.query_params["resort"]` on change. Unknown values fall back to the default silently.

`nearby` means the 10 resorts with the smallest great-circle distance to the selected resort, including itself. `country` means every resort with the same `country`. `all` means all 86.

### 3.2 Data loading (`dashboard_v2/data.py`)

- `load()` with `@st.cache_data(ttl=300, show_spinner=False)`. Fetch the GitHub raw URL (10 s timeout) and read the local file. Use whichever has `schema_version == 3` and the later `updated`. If neither is schema 3, show the one-sentence error in 3.4 screen 0.
- Return the parsed dict plus `origin: "github"|"local"`. The close prints which one was used.
- Never load a file with `"fixture": true` unless `SNOW_FIXTURE=1`.
- No pandas. Plain dicts and lists.

### 3.3 Freshness

`age_h = now_utc - updated`. The update line reads "Updated 06:14 MST, Thursday 27 November" in the selected resort's `timezone`, plus "(13:14 UTC)". When `age_h > 13`, add the sentence "This forecast is N hours old; updates normally run every six hours." When `age_h > 36`, the hero sentence is replaced by "This forecast is N days old, so treat it as out of date." and the hero numeral is still shown. The line appears under the hero (screen 1) and in the close (screen 8).

### 3.4 Screens

Screen 0 (only on error). One sentence in ink at body size at the top-left: "The forecast file could not be loaded." or "The forecast file is from an older version of this app and has no data this page can show." Nothing else renders.

Screen 1, "is it going to snow here, and when". 5:7 split (text left 7, facts right 5 on desktop; stacked on phone).
- Label "Resort" (12px small caps), then the picker styled as the 61px name. Options are every resort, sorted by country then name, shown as "Alta, Utah" / "Zermatt, Valais".
- Caption line: "Utah, USA. Base 2,600 m, summit 3,215 m." (summit part omitted when null: "Base 2,600 m, summit not known.").
- Hero numeral: `next_72h.summit_cm` in accent, with unit at 40px, and a 12px label after it: "new snow in the next three days, at the summit". If summit is null, use `base_cm` and say "at the base". If both are null, no numeral; write "No forecast for this resort in this update." at body size.
- One sentence (body size), built by the rules in 3.5.
- The freshness line (3.3).
- Right column, three facts as a hanging list, each a label and a number:
  1. "Base depth": `depth.value_cm` and, in a caption under it, `depth.source_label` and the observation date ("reported Wednesday 26 November"). Null: "No reading".
  2. Second fact: "Snow line today" from `days[0].derived.snowfall_height_precip_min_m`–`max_m` if not null; otherwise "Freezing level today" from `freezing_level_max_m`. Null: "No reading".
  3. "Chance of precipitation, next three days": `next_72h.precip_prob_max_pct`%. Note the wording: Open-Meteo gives precipitation probability, not snow probability.

Screen 2, "the shape of the week". 4:8 split.
- Left 4: "The week" (40px), `week.summit_cm` at 91px with unit, one sentence: "Best day to ski: Saturday, after 24 cm overnight." from `week.best_ski_day` and `best_ski_day_score_cm`; if null, "No day this week stands out." Label "at the summit" (or "at the base").
- Right 8: seven rows, index 0..6. Each row on one 27px line: day label ("Today", then "Sat 29"), text-weight bar, numeral (summit snowfall), then in caption size: precipitation probability, "freezing level 2,400 m" or "snow line 1,700–2,100 m" (Europe), powder word, and any of the words "wind hold", "rain at mid-mountain" (`rain_risk.mid`), "mixed at mid-mountain" (`mixed_risk.mid` true and `rain_risk.mid` not true), "bluebird". Words, never icons. The row for `week.best_snow_day` has its bar and numeral in accent.
- After a `--s3` gap, the label "Days 8 to 15, lower confidence" and eight more rows in the same format but with only the numeral, bar, probability and "wind hold" if true. Then one line: "Days 8 to 15 total: 18 cm." from `extended.summit_cm`.
- Bar scale: one fixed maximum for both lists, `max(50, ceil(max_value / 25) * 25)` cm (in inches: `max(20, ceil(max / 10) * 10)`). State it once in the label: "bars to 50 cm".

Screen 3, "how far to trust it". One sentence at 27px, nothing else, built by 3.6.

Screen 4, "where on the mountain". Three groups, summit, mid, base, `--s3` apart, top to bottom. Each group: label with the elevation ("Summit, 3,215 m"), `next_72h.E_cm` at 27px with a bar (same scale as screen 2), and one line built from `next_72h.by_day.E` (the same hours as the headline, so its cm add up to it): one segment per day with `phase != "none"`: "Fri 7 cm" (snow), "Sun 8 cm + 9 mm wet" (mixed, meaning wet snow or rain), "Mon 18 mm rain" (rain), joined with " · ". "dry" if every segment is none; a dash if any is null. The word "rain" may appear only for a segment whose phase is rain or mixed. When `quality.bands.mid_summit_same_72h` is true, show one merged group "Mid-mountain to summit, 1,736–2,131 m" with the caption "Model can't separate these heights here" instead of two identical groups. (See docs/logic_fixes_render.md.) Then "Last three days, model: 12 cm" from `recent.modelled.E.h72_cm`, with the word "model" always present. Groups for null elevations are left out, with one caption: "Summit elevation not known for this resort, so only the base is shown."

Screen 5, "compare with its neighbours". 5:7 split.
- Left 5: "Across the range" (40px). One line of scope words: "Nearby · Utah · Everywhere" (current one underlined; the country word is the resort's country, "USA"). One sentence: "Snowiest in the next three days: Alta 40 cm, Snowbird 38 cm, Brighton 31 cm." (top three in scope). If any resort in scope has `flags.storm_incoming == true`, a second sentence: "A storm is on the way for Alta, Snowbird and Park City." (names, max five, then "and N more").
- Then the ranked list, top 10 in scope by `next_72h.summit_cm` (base where summit is null; nulls last). Each row: name (a link to `?resort=`), distance in caption size when scope is nearby ("12 km"), bar, numeral, best ski day in caption size ("Sat"). The selected resort's row is accent.
- Right 7: the map (4.7) showing every resort in scope.

Screen 6, "check against real measurements". "On the ground" (40px).
- If `depth.kind == "resort_report"`: one row group "Resort report via OnTheSnow, updated Wednesday 26 November": base, mid, summit depths, and "new snow reported: 4 cm in 24 hours, 6 in 48, 6 in 72". Each null part says "not reported".
- Then one row per `stations` entry: name, distance ("2 km"), elevation, latest depth, "depth change +5 cm in 24 hours" (SNOTEL also 48/72 and "water added 8 mm"), and for SNOTEL the 15-day depth sparkline with the latest value beside it. Observation date in caption size.
- Then one line: "Model, last 24, 48 and 72 hours at the summit: 4, 6, 6 cm." labelled model.
- When there is no report and no station: "No measuring station or resort report in this feed for {country}." Never anything else in their place.

Screen 7, "find any other resort". "All resorts" (40px). Sort words: "Sort by next three days · week · best day · base depth · name". The table shows resorts in the current scope, with a final link "Show all 86 resorts" when the scope is not `all`.

| Column | Value | Align |
|---|---|---|
| Resort | name, link to `?resort=` | left |
| Where | region, country | left |
| Next 3 days | `next_72h.summit_cm` (base where no summit) | right |
| Week | 7-bar inline sparkline of `days[0..6].summit.snowfall_cm` then `week.summit_cm` | right |
| Days 8–15 | `extended.summit_cm` | right |
| Best day | weekday of `week.best_ski_day`, "none" if null | left |
| Base depth | `depth.value_cm`, followed by a 12px small-caps word "report" or "station" | right |
| Last 24 h | `recent.measured.h24_cm` with sign, else `recent.modelled.summit.h24_cm` followed by "model" | right |
| Storm | the word "storm" when `flags.storm_incoming` is true, otherwise empty | left |

Sorting: `next3`, `week`, `depth` descending with nulls last; `best` by `best_ski_day_score_cm` descending; `name` A–Z and then grouped by country and region with a 27px group name above each group. Missing values in the table show "—", and a caption under the table says "— means no data in this update." On phones (under 700px) the table sits in an `overflow-x:auto` wrapper and Resort stays the first column.

Screen 8, the close. Caption size, on the left edge:
- Sources, built from `sources.list`, each with its kind in words ("Open-Meteo, model forecasts", "SNOTEL, measured snow depth", "OnTheSnow, resort reports").
- "Base depths this update: 61 of 86 resorts." from `run.coverage.depth`.
- Update line (3.3) and "Data loaded from GitHub" or "from the copy bundled with the app".
- Units: "cm / in", the current one underlined.
- "Refresh" as the one `st.button` (clears `load` cache and reruns).
- Map attribution: "Map tiles © CARTO, © OpenStreetMap contributors."
- `?debug=1` only: `run` object as plain text lines.

### 3.5 Hero sentence rules (screen 1)

Pick the first that applies. Numbers rounded as in 4.8; days in words.

1. Summit and base both null: "No forecast for this resort in this update."
2. `next_72h.summit_cm < 1`: "No meaningful snow in the next three days." Then, if `week.summit_cm >= 5`: " The week brings {week} cm, mostly on {weekday of best_snow_day}."
3. Otherwise: "Most of it falls {period} {weekday}" where the period is the `summit_by_period` block with the largest snowfall ("Friday night", "Saturday morning"). Then, if `best_ski_day` is not null: ". {Weekday} is the day to ski." Then, if any of `flags.rain_days` is in 0..2: " Rain up to about {rain_limit_m rounded to 10} m {weekday}." using that day's `derived.rain_limit_m` (skip the clause if null); never `freezing_level_precip_max_m`, which is the warmest hour's 0 C level and overstates the rain line by 250 m or more.

### 3.6 Confidence sentence rules (screen 3)

1. If `models` is not null and `agree_through` is not null: "The models agree through {weekday of agree_through}; after that they differ." If `agree_through >= 6`: "The models agree on the whole week."
2. If `models` is not null and `agree_through` is null: "The models already disagree about today, so treat these numbers loosely."
3. Otherwise: "The next three days are the most reliable part of this forecast; days 8 to 15 are a rough guide."

---

## 4. Visual spec

The design brief (`docs/design_brief_v2.md`) is the authority. This section applies it to the IA above and settles the few places where the brief and this spec differ. Where they conflict, this section wins and says why.

### 4.1 Files and injection

- `dashboard_v2/theme.py`: `apply_theme(page_title)` calls `st.set_page_config(page_title=page_title, page_icon=None, layout="wide", initial_sidebar_state="collapsed")` first, then injects one `<link>` to Google Fonts and one `<style>` block (the full CSS below).
- `dashboard_v2/render.py`: one function per screen returning an HTML string, plus helpers `bar(v, vmax, best=False)`, `spark_bars(values, best_index)`, `spark_line(values)`, `num(v, unit, units)`, `fmt_date(...)`. Every data string passes through `html.escape`.
- `app.py`: loads data, reads params, calls `st.markdown(html, unsafe_allow_html=True)` once per screen. Only three widgets: the selectbox, `st.pydeck_chart`, and the Refresh `st.button`.
- If `st.markdown` strips the inline `<svg>` on the installed Streamlit version, use `st.html(...)` for that screen instead. Check this first.

### 4.2 Theme file

Replace `.streamlit/config.toml` at the repo root with the server settings kept and the brief's theme (Streamlit Cloud reads the root file). Delete `dashboard_v2/.streamlit/` so there is one config.

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#2C5B8C"
backgroundColor = "#F2F3F0"
secondaryBackgroundColor = "#F2F3F0"
textColor = "#1A2230"
font = "serif"

[client]
toolbarMode = "minimal"
showSidebarNavigation = false
showErrorDetails = false
```

### 4.3 Requirements

`dashboard_v2/requirements.txt` (Streamlit Cloud uses the entrypoint folder's file before the root one), which drops the heavy root dependencies:

```
streamlit==1.52.2
pydeck==0.9.1
```

Both versions are installed in this environment today. Do not touch the root `requirements.txt`.

### 4.4 Fonts and the tabular-figure check

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500&display=swap" rel="stylesheet">
```

First build step: render `<span class="data" id="tnum-a">1111</span>` and `<span class="data" id="tnum-b">8888</span>` and compare their widths in the browser (Playwright `boundingBox`). If they differ by more than 0.5px, add Source Serif 4 (`family=Source+Serif+4:opsz,wght@8..60,400;8..60,500`) and set `.data, td, .num, .hero, .big, .n27 { font-family: "Source Serif 4", "EB Garamond", serif; }` for numerals only. Record the result in a comment at the top of `theme.py`.

### 4.5 CSS

The whole stylesheet, in one `<style>` block. Values come from the brief. Selectors for Streamlit chrome are from the brief and must be rechecked against 1.52.2 in the browser.

```css
:root{
  --paper:#F2F3F0; --ink:#1A2230; --accent:#2C5B8C;
  --l:27px;
  --s0:calc(var(--l)*.5); --s1:var(--l); --s1h:calc(var(--l)*1.5);
  --s2:calc(var(--l)*2); --s3:calc(var(--l)*3); --s4:calc(var(--l)*4); --s6:calc(var(--l)*6);
  --m:clamp(24px,6vw,120px);
  --hang:clamp(24px,11vw,200px);
  --measure:24em;
}
@media (max-width:700px){ :root{ --l:26px; --s6:calc(var(--l)*5); --m:24px; --hang:24px } }
@media (min-width:1921px){ :root{ --l:30px } }

/* chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stElementToolbar"]
{ display:none !important; }

/* ground */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]
{ background:var(--paper) !important; color:var(--ink); }
html{ overflow-x:clip; }
.block-container, [data-testid="stMainBlockContainer"]
{ padding:0 var(--m) 0 var(--hang) !important; max-width:none !important; }
[data-testid="stVerticalBlock"]{ gap:0 !important; }
[data-testid="stHorizontalBlock"]{ gap:var(--l) !important; }
[data-testid="stMarkdownContainer"] p{ margin:0; }

/* type */
.stApp, .stApp *{ font-family:"EB Garamond", Garamond, "Times New Roman", serif !important; }
body, .stApp{ font-size:18px; line-height:var(--l); -webkit-font-smoothing:antialiased; font-kerning:normal; }
em, i{ font-style:normal; }
.data, td, .num{ font-variant-numeric:lining-nums tabular-nums; }
.prose .num{ font-variant-numeric:oldstyle-nums proportional-nums; }
::selection{ background:var(--accent); color:var(--paper); }
.page{ max-width:min(1180px, calc(100vw - var(--hang) - var(--m))); }
.screen{ padding-top:var(--s6); }
.screen:first-child{ padding-top:var(--s4); }
.label{ font-size:12px; line-height:1.5; font-variant-caps:all-small-caps; letter-spacing:.12em; }
.caption{ font-size:14px; line-height:20px; }
.body{ font-size:18px; line-height:var(--l); }
.n27{ font-size:27px; line-height:1.35; }
.h40{ font-size:40px; line-height:1.1; letter-spacing:-.01em; font-weight:400; margin:0; text-wrap:balance; }
.big{ font-size:91px; line-height:.95; letter-spacing:-.02em; font-variant-numeric:lining-nums proportional-nums; }
.hero{ font-size:clamp(120px,14vw,200px); line-height:.9; letter-spacing:-.03em; color:var(--accent);
       font-variant-numeric:lining-nums proportional-nums; }
.hero .unit{ font-size:40px; letter-spacing:0; }
.big .unit{ font-size:40px; letter-spacing:0; }
.n27 .unit{ font-size:18px; }
.accent{ color:var(--accent); }
.sentence{ max-width:var(--measure); text-wrap:pretty; }
.breath{ font-size:27px; line-height:1.35; max-width:var(--measure); padding-block:var(--s6); }

/* splits */
.split-5-7{ display:grid; grid-template-columns:7fr 5fr; column-gap:var(--l); }   /* screen 1: text 7, facts 5 */
.split-4-8{ display:grid; grid-template-columns:4fr 8fr; column-gap:var(--l); }
@media (max-width:700px){ .split-5-7, .split-4-8{ grid-template-columns:1fr; row-gap:var(--s2); } }

/* rows on the line grid */
.rows{ display:grid; grid-template-columns:auto 10em 4ch 1fr; column-gap:1em; align-items:baseline; }
.rows > *{ line-height:var(--l); min-height:var(--l); }
.rows .n{ text-align:right; font-variant-numeric:lining-nums tabular-nums; font-weight:500; }
.rows .c{ font-size:14px; }
.group + .group{ margin-top:var(--s3); }
@media (max-width:700px){ .rows{ grid-template-columns:auto 6em 4ch; } .rows .c{ grid-column:1 / -1; } }

/* bars and sparklines */
.bar{ display:inline-block; height:.18em; width:calc(var(--v)*10em); background:var(--ink); vertical-align:.32em; border-radius:0; }
.bar.is-best{ background:var(--accent); }
.spark{ height:1.2em; width:7em; vertical-align:-.2em; overflow:visible; }
.spark polyline{ fill:none; stroke:var(--ink); stroke-width:1.25; vector-effect:non-scaling-stroke; shape-rendering:geometricPrecision; }
.spark rect{ fill:var(--ink); } .spark rect.is-best{ fill:var(--accent); } .spark circle{ fill:var(--accent); }

/* tables */
.tablewrap{ overflow-x:auto; }
table.t{ border-collapse:collapse; border:0; font-size:14px; }
table.t th, table.t td{ border:0; padding:0 1em 0 0; height:var(--l); line-height:var(--l); background:transparent; text-align:left; font-weight:400; }
table.t th{ font-size:12px; font-variant-caps:all-small-caps; letter-spacing:.12em; padding-bottom:var(--s0); }
table.t td.r, table.t th.r{ text-align:right; }
table.t tr.is-selected td{ color:var(--accent); }
table.t .grp td{ font-size:27px; line-height:1.35; padding-top:var(--s3); }

/* links */
a, a:visited{ color:inherit; text-decoration:underline; text-decoration-thickness:1px; text-underline-offset:.2em; }
a.plain{ text-decoration:none; }
.words a{ display:inline-block; padding-block:10px; }
.words a.is-current{ text-decoration:underline; } .words a:not(.is-current){ text-decoration:none; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:3px; }

/* the picker */
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:transparent !important; border:0 !important; border-bottom:1px solid var(--ink) !important;
  border-radius:0 !important; box-shadow:none !important; padding:0 !important; min-height:0 !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] *{
  font-size:61px !important; line-height:1.05 !important; letter-spacing:-.01em; color:var(--ink) !important; }
[data-testid="stSelectbox"] svg{ width:24px; height:24px; color:var(--ink); }
@media (max-width:700px){ [data-testid="stSelectbox"] [data-baseweb="select"] *{ font-size:44px !important } }
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"]{
  background:var(--paper) !important; border:1px solid var(--ink) !important; border-radius:0 !important; box-shadow:none !important; }
[role="option"]{ font-size:18px !important; line-height:var(--l) !important; color:var(--ink) !important; background:transparent !important; }
[role="option"][aria-selected="true"], [role="option"]:hover{ color:var(--accent) !important; background:transparent !important; }

/* the map */
[data-testid="stDeckGlJsonChart"]{
  margin-right:calc(var(--m) * -1);
  -webkit-mask-image:linear-gradient(to right, transparent 0, rgba(0,0,0,.15) 6%, rgba(0,0,0,.6) 14%, #000 24%),
                     linear-gradient(to bottom, transparent 0, #000 10%, #000 90%, transparent 100%);
  -webkit-mask-composite:source-in;
  mask-image:linear-gradient(to right, transparent 0, rgba(0,0,0,.15) 6%, rgba(0,0,0,.6) 14%, #000 24%),
             linear-gradient(to bottom, transparent 0, #000 10%, #000 90%, transparent 100%);
  mask-composite:intersect;
  border:0 !important; border-radius:0 !important; box-shadow:none !important; }
.deck-tooltip{ background:var(--paper) !important; color:var(--ink) !important; border:1px solid var(--ink) !important;
  border-radius:0 !important; box-shadow:none !important; font-size:14px !important; }
.mapboxgl-ctrl, .maplibregl-ctrl, .mapboxgl-ctrl-attrib, .maplibregl-ctrl-attrib{ display:none !important; }

/* the one button */
.stButton > button{ background:var(--paper) !important; border:1px solid var(--accent) !important; border-radius:0 !important;
  color:var(--ink) !important; height:48px; font-size:18px !important; box-shadow:none !important; padding:0 var(--s1) !important; }
.stButton > button:hover{ color:var(--accent) !important; background:var(--paper) !important; }

/* large screens */
@media (min-width:1921px){
  body, .stApp, .body{ font-size:20px; } .n27{ font-size:30px; } .h40{ font-size:46px; } .big{ font-size:112px; }
  .hero{ font-size:clamp(160px,12vw,240px); } .label{ font-size:14px; }
  [data-testid="stSelectbox"] [data-baseweb="select"] *{ font-size:70px !important; }
}
@media (max-width:700px){
  body, .stApp, .body{ font-size:17px; } .n27{ font-size:24px; } .h40{ font-size:34px; } .big{ font-size:56px; }
  .hero{ font-size:clamp(96px,28vw,136px); }
}
```

### 4.6 Element by element

| Element | Visual rule |
|---|---|
| Hero numeral (screen 1) | `.hero`, accent, whole number, `&thinsp;` then `<span class="unit">cm</span>`, baseline-aligned. Label after it in `.label`. This is the only element at the jump size. |
| Hero sentence | `.body .sentence .prose`, ink. `--s1` below the hero. |
| Three facts | `.split-5-7` right column. Each: `.label` line, then the number at 27px (`.n27`) in ink with unit, then the source in `.caption`. Groups `--s2` apart. |
| Week total (screen 2) | `.big`, ink, with unit. |
| Day rows | `.rows` grid: label, bar column (10em), numeral column (4ch, right-aligned, weight 500), caption column. Row height exactly one line, no gap between rows, no rule. Best-snow day bar and numeral accent. Today's label reads "Today". |
| Snowfall amounts in general | Shown only as a numeral plus a `.bar`. Bar width = `value / scale_max` as `--v`; any non-zero value gets at least 2px (`max(var, 2px)` via inline `min-width:2px`); zero draws no bar at all; null draws no bar and the numeral cell shows "—". No progress bars, no boxes, no tracks, no colour ramps. |
| Week sparkline (screen 7) | Inline SVG, `viewBox="0 0 35 12"`, seven `<rect>` 3 wide with 2 gaps, height proportional to the table's own maximum, baseline at the bottom. Null day: no rect. Followed by the week numeral. |
| Depth sparkline (screen 6) | Inline SVG polyline of `series[].depth_cm`, nulls break the line into segments (separate polylines), one accent `<circle r="2">` on the latest non-null point, then the latest value as a numeral. |
| Breath line (screen 3) | `.breath`, ink. Nothing else on the screen. |
| Elevation groups (screen 4) | `.label` with elevation, `.n27` numeral with bar (bar height `.18em` of 27px), sentence in `.body`. `.group + .group` gives `--s3`. |
| Ranked list (screen 5) | `.rows`, selected resort's row in `.accent` (name, bar `.is-best`, numeral). |
| Scope and sort words | `.words` paragraphs; separators are a middle dot with spaces; the current word underlined, others not. |
| Tables (screen 7) | `table.t` inside `.tablewrap`. No borders, no fills, no zebra, no hover. Header labels in small caps with `--s0` space below. |
| Close (screen 8) | `.caption` lines on the left edge, ends `--s4` above the page bottom (`padding-bottom:var(--s4)` on the last screen). |
| Missing values | Prose: "No reading", "not reported", "No forecast". Tables and row numeral cells: "—" with the legend caption. See 4.9. |

### 4.7 Map (pydeck)

- `pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json", initial_view_state=fit to the scope's points, layers=[points, labels], tooltip={...})`, `st.pydeck_chart(deck, height=324)` (270 on phones is not detectable server-side; keep 324). Use the container-width argument the installed version expects.
- Points: `ScatterplotLayer`, `get_fill_color` ink `[26,34,48]`, selected resort accent `[44,91,140]`, `radius_units="pixels"`, `get_radius = 3 + 2.2 * sqrt(max(cm, 0))` capped at 28, where cm is the same value the ranked list uses; null gets radius 3 and is drawn as an outline only (`stroked=True`, `filled=False` for those, via a second layer).
- Labels: `TextLayer` with the names of the five snowiest resorts in scope, `get_size=14`, `font_family='"EB Garamond", serif'`, ink, `get_text_anchor="start"`, `get_pixel_offset=[10,0]`.
- Tooltip: `{"html": "<b>{name}</b><br>{label}", "style": {"background": "#F2F3F0", "color": "#1A2230", "border": "1px solid #1A2230", "font-family": "EB Garamond, serif", "font-size": "14px", "box-shadow": "none"}}`. Note that `<b>` renders weight 500 via CSS, not bold 700.

### 4.8 Number and date formatting

- Snow: whole cm (`round`), or whole inches (`cm / 2.54`, rounded) when `units=in`. Values between 0 and 0.5 cm display as "0"; never "0.3".
- Changes carry a sign and a real minus sign: "+5 cm", "−3 cm".
- Temperatures: whole degrees with a real minus sign, "−8°C" or "18°F".
- Elevations and freezing levels: nearest 10 m with a thousands comma, "2,460 m"; in `units=in` mode, nearest 10 ft, "8,070 ft".
- Wind: whole km/h, or mph in `units=in` mode.
- Probability: whole percent, "67%".
- Dates: "Saturday 29 November" in sentences, "Sat 29" in row labels.
- Ranges with an en dash and no spaces: "1,700–2,100 m".
- Numbers inside sentences use `.prose .num` (old-style figures).

### 4.9 Missing data (where the brief and this spec differ)

The brief says never to write a dash for missing data. The QA checklist for this round asks for "—". The resolution: in table cells and row numeral cells, missing is "—" and every table or list that can show one carries a caption "— means no data in this update"; in sentences and the three facts, missing is written in words ("No reading", "not reported"). Zero is always "0". A null is never shown as 0 and a 0 is never shown as "—".

### 4.10 Never list (from the brief, enforced in code review)

No italics, no emoji (including the page title and favicon), no sans-serif, no third font, no grey or faded text, no accent on headings, no boxes or cards or `st.metric`/`st.expander`/`st.info`/`st.warning`/`st.error`/`st.success`/`st.dataframe`/`st.line_chart`/`st.bar_chart`/`st.divider`/`st.progress`/`st.toggle`/`st.checkbox`/`st.radio`/`st.slider`/`st.tabs`/`st.columns([1,1])`, no section backgrounds, no more than one visible rule per screen (the picker underline and link underlines only), no colour ramps, no made-up numbers.

---

## 5. Acceptance criteria

QA verifies each item with screenshots at 1440x900 and 390x844 (and 1920x1080 and 2560x1440 where noted), with Playwright against a local `streamlit run dashboard_v2/app.py` reading a real schema-3 file produced by the new fetcher, and by reading the JSON. Every item is pass or fail.

### Data

1. `data/forecasts_v2.json` has `schema_version` 3, passes `python scripts/validate_forecasts_v2.py`, and contains all 86 resorts with unique `id`s.
2. At least 90% of resorts have `quality.forecast_ok == true`, and `days` covers index -3 to 14 for those resorts.
3. No resort's `depth.provider` is an Open-Meteo value; the string "modeled" or "modelled" never appears inside any `depth` object.
4. No more than one Japanese resort has `depth.value_cm == 100.0` from tenki.jp.
5. Alta's `stations` do not contain `332:UT:SNTL` (Ben Lomond Peak), unless its name has been corrected in AWDB, and every station `name` comes from the provider's metadata.
6. No `stations` entry is more than 60 km from its resort.
7. For each resort with `summit_elevation_m` not null and a working summit call, `now.summit.temp_c` differs from `now.base.temp_c` or the elevations are within 100 m (proves the summit value is a real call, not a copy).
8. For every resort, `summit_elevation_m` is either null or has an `elevation_note`; no summit equals base + 800 unless the note cites a source.
9. Spot check three resorts by hand against a fresh Open-Meteo request using the URL in 2.4: `days[1].summit.snowfall_cm`, `days[1].derived.freezing_level_max_m` and `next_72h.summit_cm` match within 0.1 cm / 1 m (allowing for a model update between the two calls; if they differ, re-run both back to back).
10. Every derived field in 1.9.1 is null on `past_model` days, and the hourly-based fields are null on `extended` days.
11. Every 0.0 forecast or depth in the source is 0.0 in the JSON, not null.
12. The workflow file has no `continue-on-error`, has a `concurrency` group, rebases before push, and has no pip install step. A manual `workflow_dispatch` run succeeds and commits.

### Page structure and behaviour

13. With no query params, the selected resort is the one with the highest `next_72h.summit_cm` (base where summit is null) in the file.
14. Screens appear in this order, each present: picker and hero, the week, confidence sentence, by elevation, across the range with map, on the ground, all resorts, close.
15. The picker lists all 86 resorts as "Name, Region", is searchable by typing, and changing it updates `?resort=` and every screen.
16. "Nearby", the country word and "Everywhere" change the ranked list, the map and the table; "Nearby" shows exactly 10 resorts.
17. Every sort word reorders the table as specified, nulls last; "name" groups by country and region.
18. With `?scope=all`, the table shows 86 rows.
19. "cm / in" switches every snow number, elevation, temperature and wind speed on the page; the current unit word is underlined.
20. The hero shows `next_72h.summit_cm` rounded to a whole number, and the label says "at the summit" (or "at the base" when summit is null).
21. The week list shows 7 day rows plus 8 extended rows under the "lower confidence" label, and each row's numeral equals the JSON value rounded.
22. Derived metrics are visible in words: at least one of "wind hold", "rain at mid-mountain", "bluebird", "cold, dry" / "average" / "warm, heavy" appears for a resort whose JSON has that flag (QA picks such a resort from the file), and freezing level or snow line appears in the day rows.
23. A resort with `flags.storm_incoming == true` in scope makes the storm sentence appear in screen 5 and "storm" appear in its table row.
24. Screen 6 for a US resort shows each station's name, distance, elevation, depth, 24-hour depth change and a 15-day sparkline with a numeral beside it; for a French resort with a resort report it shows the report; for a resort with neither it shows the one "No measuring station or resort report" sentence.
25. Every value labelled as recent modelled snowfall carries the word "model" next to it.
26. Null values: a resort with `depth.value_cm == null` shows "No reading" in screen 1 and "—" in its table cell, never "0"; a resort with `depth.value_cm == 0.0` shows "0". The legend "— means no data in this update" sits under the table.
27. The update line shows the time in the resort's timezone and in UTC; with an `updated` value set 40 hours in the past (fixture), the out-of-date sentence appears.
28. "Refresh" reloads data without a full page error; the page never shows a Python traceback (`showErrorDetails = false`).
29. Loading with `?resort=not-a-resort&sort=zzz` falls back to defaults without an error.

### Design (from the brief)

30. One ground: every visible background pixel of the app, including the opened picker list and the map tooltip, is `#F2F3F0`, except the map tiles.
31. Two text colours only: computed `color` of every text node is `rgb(26, 34, 48)` or `rgb(44, 91, 140)`. Accent text appears only on the hero, the best-snow-day row, and the selected resort's rows.
32. Serif type loads: `document.fonts.check('18px "EB Garamond"')` is true, and the computed `font-family` of every text node starts with "EB Garamond" (or "Source Serif 4" for numerals if the tabular check failed and the fallback is documented in `theme.py`).
33. Tabular figures: in the screen 7 "Next 3 days" column, the right edges of all numerals line up within 1px.
34. No boxes: no element (outside the map canvas) has a non-zero `border` other than the picker's bottom border, the opened list's 1px outline, the Refresh button's outline and link underlines; no element has a `box-shadow`, a non-zero `border-radius`, or a background other than paper or transparent. Specifically none of `stMetric`, `stExpander`, `stDataFrame`, `stAlert`, `stProgress`, `stTabs`, `stArrowVegaLiteChart` exists in the DOM.
35. No Streamlit chrome visible: no header bar, toolbar, hamburger menu, "Made with Streamlit" footer, sidebar or sidebar toggle.
36. One left edge: at 1440 wide, the left x of every heading, label, sentence and table's first column is the same value (within 2px), with numeral columns as the only second edge per list.
37. Splits are 5:7 or 4:8 at 1440; no two side-by-side columns of equal width anywhere.
38. Sizes: every computed `font-size` on the page is one of 12, 14, 17, 18, 20, 24, 27, 30, 34, 40, 44, 46, 56, 61, 70, 91, 112, or the hero's clamp value.
39. Vertical gaps between blocks are 0, 14, 27, 41, 54, 81, 108 or 162 px at 1440 (phone: 0, 13, 26, 39, 52, 78, 104, 130), within 2px.
40. Bars: every `.bar` has `height` equal to 0.18 of its font size, no background track behind it, `border-radius: 0`; a zero value has no bar; one numeral sits beside every bar and sparkline.
41. The map has no border, its right edge reaches the viewport edge, its left edge fades into the paper, and no zoom or attribution controls are visible.
42. No italics (`font-style: italic` computed nowhere), no emoji anywhere in the page text, the browser tab title is "Snow forecast, {resort name}", and there is no emoji favicon.
43. At 390x844 there is no horizontal page scroll; the table scrolls inside its own wrapper; tap targets for links and the button are at least 44px tall; hero and picker use the phone sizes.
44. At 2560x1440 the line unit is 30px and the type uses the large-screen sizes.
45. `dashboard_v2/requirements.txt` exists with the two pins, and the app starts with only those installed (`pip install -r dashboard_v2/requirements.txt` in a fresh venv, then `streamlit run dashboard_v2/app.py`).
