# Render fixes from the logic review (dashboard_v2/render.py)

Written by the data-layer fix pass on 2026-11-27. The data team did not touch dashboard_v2/*. Everything below uses fields the fetcher now writes (spec sections 1.8, 1.9.1, 1.9.2, 1.13). Keep wording dense: numbers first, few words, no "likely" on a yes/no flag.

New fields the page can use:

| Field | Meaning |
|---|---|
| `next_72h.window_start_local`, `next_72h.window_end_local` | The headline window is (start, end], e.g. Fri 03:00 to Mon 03:00. |
| `next_72h.by_day.{base,mid,summit}[]` | One segment per calendar day in that same window: `date`, `hours`, `snowfall_cm`, `rain_mm`, `precip_mm`, `phase` (`snow`/`mixed`/`rain`/`none`/null), `rain_phase_mm`, `mixed_phase_mm`. A band's segment snowfall sums to `next_72h.{band}_cm` (checked by sweep rule R12). |
| `days[d].derived.phase.{E}` | Day phase per band, same rule. |
| `days[d].derived.mixed_risk.{E}` | Wet snow or rain is a real part of the day at that band. |
| `days[d].derived.rain_risk.{E}` | Now hour-by-hour against the snow line (freezing level minus 250 m). False at Crystal's summit on Sunday. |
| `days[d].derived.rain_limit_m`, `rain_limit_kind` | Highest rain line in hours it rains at the base. `freezing_level_minus_250` means approximate (US always). |
| `flags.mixed_days` | Indices 0..6 with `mixed_risk` at mid (summit when no mid). |
| `quality.bands.mid_summit_same_72h`, `mid_summit_same_week`, `mid_summit_same_cell`, `base_mid_same_72h` | The model gave two elevations identical precipitation. |
| `model_cells.{base,mid,summit}` | The grid cell each call landed on. |

---

## 1. Screen 4 "By elevation": the rain column contradicts the snow column (the user's complaint)

File and function: `dashboard_v2/render.py`, `_rain_word()` and `screen4()`.

Current behaviour: the Rain column prints the first day in days 0..2 with `rain_risk.E` true ("Sun"), next to a 72-hour snow number. It covers calendar days 0..2, not the headline's (H, H+72h] window, gives no amount, and never says that the snow falls Friday and the wet part is Sunday. At the Crystal summit it printed "Rain Sunday" under 12 cm of snow while the summit stayed at -3 C.

Required behaviour:
- Replace `_rain_word()` with a per-band line built from `next_72h.by_day[E]`, skipping segments whose `phase` is `"none"`.
- Segment text by phase:
  - `snow`: `{Ddd} {snow} cm`
  - `mixed`: `{Ddd} {snow} cm + {rain_mm} mm wet`
  - `rain`: `{Ddd} {rain_mm} mm rain`
- Join segments with ` · `. If every segment is `none`, show `dry`. If any segment's phase is null, show the em dash.
- The word "rain" may appear only in a `rain` segment, and "wet" only in a `mixed` segment. The cm figures in the line must add up to the headline number (they come from the same hours).
- Rename the column header from "Rain" to "Snow / wet".

Example (Crystal Mountain, run of 2026-11-27):
- Summit 2,131 m, next 3 days 16 cm: `Fri 7 cm · Sun 8 cm + 9 mm wet · Mon 0 cm + 2 mm wet`
- Base 1,341 m, next 3 days 1 cm: `Fri 1 cm + 9 mm wet · Sun 20 mm rain · Mon 2 mm rain`

## 2. Screen 4: mid-mountain and summit show the same number

File and function: `dashboard_v2/render.py`, `screen4()`.

Current behaviour: three rows are always drawn. At Crystal, mid and summit both show 12 cm next and 24 cm past, which looks like a bug or a copy-paste. The model's grid cell is the same for both, and Open-Meteo adjusts only temperature for height. In this run that happens at many resorts; see quality.bands.

Required behaviour:
- If `quality.bands.mid_summit_same_72h` is true, draw one row labelled `Mid–summit`, with the elevation `1,736–2,131 m`, one bar and one number. Below the table add the caption `Model can't separate mid and summit here.`
- If `quality.bands.base_mid_same_72h` is also true, merge all three rows into `Base–summit` with the same caption.
- Never compute or invent a different summit value.

Example: `Mid–summit  1,736–2,131 m  ▇▇▇▇  16  24  Fri 7 cm · Sun 8 cm + 9 mm wet` and the caption `Model can't separate mid and summit here.`

## 3. Hero sentence: the rain clause uses the wrong line and the wrong flag

File and function: `dashboard_v2/render.py`, `_rain_phrase()` (called from `hero_sentence()`).

Current behaviour: the clause uses `freezing_level_precip_max_m`, the warmest wet hour's 0 C level. At Crystal on Sunday that was 2,210 m, above the 2,131 m summit, so the page said "rain to summit" although the summit is above the snow line. The rain line is usually 250 to 300 m lower than that.

Required behaviour:
- Use `days[i].derived.rain_limit_m` for the first i in `flags.rain_days` within 0..2.
- If `rain_limit_kind == "freezing_level_minus_250"`, write `~` before the number.
- If `rain_limit_m >= summit_elevation_m`, write `rain to summit {Ddd}`. If `rain_limit_m` is null, write `rain at base {Ddd}`.
- If no day in 0..2 is in `rain_days` but one is in `flags.mixed_days`, write `wet snow {Ddd}` instead.
- Never read `freezing_level_precip_max_m` here.

Examples:
- `Falls Friday morning; rain below ~2,010 m Sun.`
- `Falls tonight; wet snow Sun.`

## 4. Hero number: say which window it covers

File and function: `dashboard_v2/render.py`, `screen1_main()` and its hero label.

Current behaviour: "next 3 days" is ambiguous. The 72-hour total runs from the run hour (03:00 Friday) to 03:00 Monday, while the week table uses calendar days. Sweep rule R04 found differences of up to 7.5 cm between the two "three day" figures (Mt Hood Meadows mid: 18.5 cm against 26.0 cm).

Required behaviour: label the hero number with `to {Ddd} {HH}h`, taken from `next_72h.window_end_local`. Don't show a calendar-day 0..2 sum next to it.

Example: `16 cm · to Mon 03h`

## 5. Week rows: the rain flag word

File and function: `dashboard_v2/render.py`, `_day_flags()`.

Current behaviour: prints `rain mid` when `rain_risk.mid` is true, and nothing for the much more common mixed days.

Required behaviour:
- If `rain_risk.mid` is true (summit when there is no mid), print `rain mid` as now.
- Otherwise, if `mixed_risk.mid` is true, print `wet mid`.
- Keep the limit of three words.
- On a row whose summit `phase` is `snow`, never print a word containing "rain" about the summit.

Example row: `Sun 8  92  2,260  wet mid · wind hold`

## 6. Week rows: the "Freeze" column

File and function: `dashboard_v2/render.py`, `_day_row()`.

Current behaviour: shows `freezing_level_max_m`, the day's maximum including dry hours. That's fine as a fact, but readers take it as the rain line.

Required behaviour: keep the number, and rename the header to `0 °C lvl`. Optionally, when `rain_limit_m` is not null, show it after a slash, as in the example. There is no change to the data.

Example: `2,260 / ~2,010`

## 7. Don't show `rain_mm` at a band as fact

File and function: anywhere the page shows `days[d].{mid,summit}.rain_mm` (currently nowhere in render.py; keep it that way).

Required behaviour: Open-Meteo's `rain_mm` at mid and summit is the grid cell's rain. Sweep rule R01c found 62 band-days with rain while that band's own maximum temperature was -2 C or colder. If an amount is ever needed, use `next_72h.by_day[E].rain_phase_mm` and `mixed_phase_mm`, and label the mixed amount `wet`, not `rain`.

## 8. Best ski day and spoiled-snow sentence

File and function: `dashboard_v2/render.py`, `_snow_spoiled()`.

Current behaviour: treats `rain_risk.summit` as spoiling a day. With the new rule, the summit rarely has `rain_risk`. The fetcher now also nulls `ski_score_cm` when the previous day had `rain_risk` at mid.

Required behaviour: also count `ski_score_cm is None` when the previous day's `rain_risk.mid` is true, so that "Wind or rain spoils every snow day." stays accurate. There is no change to the wording.

## 9. Station "depth change" against a resort's "new snow" (Big Sky)

No change. The page already labels station numbers as "depth change". This is listed only so nobody "fixes" it.

## Validator invariants for the render team's QA

- Screen 4: for each displayed band, the sum of the cm figures in its line equals the band's headline number within 1 cm.
- The words "rain" or "wet" appear in a band line only on a segment whose `phase` is `rain` or `mixed` respectively.
- When `quality.bands.mid_summit_same_72h` is true, screen 4 never shows two rows with the same number.
