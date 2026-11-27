# Powder view: what the fetcher needs to add

The Powder view (`?view=powder`, `dashboard_v2/powder.py`) ships today with forecast claims only for powder quality. It can't check a powder forecast against what actually fell, because the file is missing four things. Everything below is a change to `scripts/fetch_data_v2.py`, which belongs to the fetcher team. The dashboard side will pick each field up as soon as it appears; nothing on the page is faked in the meantime.

The score itself (`data.powder_score`) needs no new fields. It already runs on `days[].{mid,summit}.{snowfall_cm,precip_mm,temp_max_c,gust_max_kmh}` and `days[].derived.{powder_temp_c,rain_risk,mixed_risk,wind_hold}`.

## 1. Snow water in every SNOTEL series row (needed for measured density)

Today `stations[].series[]` holds `{date, depth_cm}` only. The fetcher already downloads 15 days of WTEQ for the same stations, so this is a small change.

- Add `swe_mm` (WTEQ converted from 0.1 in to mm, rounded to 0.1) to every SNOTEL series row, next to `depth_cm`. Use `null` for a missing reading, never 0.
- Optional: add `prec_mm` (PREC, the accumulated precipitation gauge) the same way. It catches snow that fell and then settled or blew off before the depth sensor read it.

With this the page can compute measured density for a forecast day D as `(swe(D+1) − swe(D)) / ((depth(D+1) − depth(D)) × 10)`, which is how much water is in the new snow. It covers 24 SNOTEL stations at 18 US resorts. The other 68 resorts have no water measurement and stay forecast-only.

## 2. Keep the forecast as it was issued (needed to check any forecast at all)

Once a day passes it becomes a `past_model` day, and its `derived` block is all null while its amounts are rewritten from the latest run. So the file never holds what we forecast for a day after that day has happened. Without that, there's nothing to check.

Add a per-resort array, for example `powder_issued[]`, that keeps each forecast day from earlier runs for 10 days:

```
{ "date": "2026-11-30", "issued_utc": "2026-11-27T12:27:13Z", "lead_days": 3,
  "band": "mid",
  "snowfall_cm": 15.8, "precip_mm": 23.1, "temp_max_c": -1.1, "gust_max_kmh": 15,
  "powder_temp_c": -3.8, "rain_risk": false, "mixed_risk": false,
  "prev_rain_risk": false, "wind_hold": false }
```

Keep the latest issue for each lead time from 1 to 3 days. These are the exact inputs `data.powder_score` reads, so the page can rebuild the score as it was issued and set it beside the SNOTEL change in snow water from item 1.

## 3. Day −1 rain risk (needed to score day 0)

Gate 3 of the score is "the day before was not rained on". For day 0 the day before is index −1, a `past_model` day whose `derived.rain_risk` and `derived.mixed_risk` are null. So day 0 can never be scored today, and no resort can show "powder today".

Compute `derived.rain_risk`, `derived.mixed_risk` and `derived.phase` for index −1 with the same hourly rules used for days 0 to 6. Asking the hourly call for `past_days=1` is enough.

## 4. When a resort report was read (needed for the 36-hour freshness gate)

A fallen-snow row needs a reading less than 36 hours old. For resort reports the only date in the file is `depth.observed_utc`, and that date belongs to the resort report only when `depth.kind == "resort_report"`. Where the depth comes from a station instead, the report has no date. Today that drops Niseko United and Rusutsu, which both report 23 cm in 72 hours.

- Add `recent.measured.observed_utc`, the time of the reading or report behind `h24_cm`, `h48_cm` and `h72_cm`, for every kind.
- Add `recent.measured.station_id` for station kinds. Today the page finds the station by matching the name in `label` against `stations[].name`, which works but breaks if a label changes.

## 5. Nice to have

- AWDB daily values look like start-of-day readings. One check against the hourly endpoint, written into `sources`, would settle which calendar day a daily change belongs to before item 1 is used for checking.
- `recent.modelled.{band}.h24_mm`, the model's own water for the same windows as `recent.measured`. Measured snow often runs 1.5 to 2 times the model's centimetres, because the model counts all frozen water at a fixed 7 cm per 10 mm. So amounts should be compared as water, never by rescaling the model's centimetres.

## 6. Past-day temperatures and gusts (BLOCKING the fallen-snow groups)

The fallen-snow quality gates (fell cold, no thaw since, no wind-pack since) need
`days[].{mid,summit}.{temp_max_c, temp_min_c, gust_max_kmh, rain_mm, precip_mm,
snowfall_cm}` on PAST days (index -3..-1). The fetcher currently blanks
temperature and gust fields on past days, so no fallen snow can ever pass the
gates and the Powder page's fallen-snow groups stay empty. Open-Meteo's
past_days parameter returns these raw observations-era model fields; keep them
populated on past days exactly as on forecast days. Only the DERIVED fields
(rain_risk, ski_score, flags) stay null on past days per the spec — raw weather
is a record, not a forecast. (Identified 27 Nov by the powder-purity team.)
