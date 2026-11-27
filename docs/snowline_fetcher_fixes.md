# Snow line: fixes for the fetcher (scripts/fetch_data_v2.py)

These follow docs/snowline_decision.md. The page already does what it can from the current file; everything here needs the hourly data only the fetcher has. Each item adds new, separately named fields. No existing field changes meaning, so the page keeps working while these land, and every derived field says it is derived.

In order of value.

## 1. Snow amounts split at each band's own height (the main fix)

Problem: Open-Meteo splits rain from snow at the grid cell's terrain height and does not redo the split for the `elevation` we ask for. At Timberline on Monday 30 Nov the freezing level never rises above 1,970 m, yet the 2,603 m summit gets 0.8 cm of snow and 23.2 mm of "rain". The page now recounts only the clear-cut days (band above the freezing level all day); days where the line moves through the band, like Timberline's Sunday, need hour-by-hour work.

Add, per band, hour by hour with the existing `hour_split()` line (snowfall height where given, else freezing level minus 250 m):

- `days[].{band}.snow_fl_cm`: water in hours where the band is at or above the line, times 0.7 cm per mm (Open-Meteo's own factor).
- `days[].{band}.rain_fl_mm`: water in hours where the band is below the line.
- the same two in `next_72h.by_day[band][]` and as `next_72h.{band}_fl_cm`, and `week.{band}_fl_cm`.
- `derived.snow_split_kind`: `"freezing_level"` so the page can say the amount is derived.

Check before shipping: at Timberline this should give about 17 cm at the summit on Monday (Weather Service 16.5 cm at 2,794 m) and about 7 cm on Sunday (Weather Service 18.0 cm at 2,794 m, 6.6 cm at 2,126 m). A four-model check this week gave 11 to 22 cm on Monday from each model's own data.

When it lands, the page swaps `data._recount_mm()` for these fields and drops its daily-only test.

## 2. A wet-snow zone, not a hard cutoff

The phase rule flips from rain to snow at one height. The Weather Service and OpenSnow both describe a few hundred metres where rain and snow mix, and heavy precipitation drags the snow level 450 m or more below the freezing level. At Timberline on Friday the Weather Service's snow level ran 1,740 to 2,230 m while our freezing level minus 250 m ran 1,500 to 2,470 m.

Add `derived.snow_line_precip_min_m` / `_max_m` (the line over wet hours, the range the page now draws itself from the freezing level) and, per band, `wet_snow_fl_mm` for hours where the band is within 250 m above the line. Keep these as phase information only: do not scale amounts by a guessed snow ratio.

## 3. Store the snow line's range, not its peak

`derived.rain_limit_m` is the highest line in hours with rain at the base. That single peak is what showed Timberline as "8,100 ft". The page now builds a range from `freezing_level_precip_min_m` / `_max_m`, but those come from the base cell's wet hours only. Store the range over each band's own wet hours, and the precipitation-weighted median line, so the page can quote the line where most of the water fell.

## 4. Cross-model consensus on the re-split amounts

Today `models` compares best_match and ECMWF on raw summit snowfall. The raw numbers are not comparable: ECMWF's coarse cell sits low, so it shows 1.4 cm at Timberline on Friday against 65 cm for best_match, mostly because its cell calls the water rain. With the item 1 split applied to each model:

| Model | Summit precip, 7 days | Summit snow, raw | Summit snow, re-split |
|---|---|---|---|
| GFS (our default in the US) | 120 mm | 56 cm | 84 cm |
| ECMWF | 66 mm | 9 cm | 46 cm |
| ICON | 100 mm | 5 cm | 69 cm |
| GEM | 91 mm | 37 cm | 64 cm |

Add `models.days[].snow_fl_cm.{model}` and `models.days[].precip_mm.{model}` at the hero band (mid), and compute `agree_through` on the re-split values. ECMWF and GEM don't publish a freezing level through the API; derive it from each model's own pressure-level temperatures (this landed within about 15 m of the published value for GFS and ICON in the check) and mark it `freezing_level_kind: "pressure_levels"`.

Also add `models.precip_spread` per day (max over min of the models' precipitation). On Friday GFS gives 77 mm at Timberline's summit against 29 to 50 mm for the others, and the Weather Service gives about 40 mm. The page can then say when the hero rests on the wettest model, without inventing a corrected number.

## 5. Best ski day per band

`week.best_ski_day` and `best_ski_day_score_cm` score the summit (65.3 at Timberline), while the page now quotes mid. Add `week.best_ski_day_by_band.{base,mid,summit}` with `{index, score_cm}` each, scored on the item 1 amounts, so the best-day cell, the verdict and the sort all come from one band. The page then drops its own 5 cm check at mid.

## 6. Snow the model reports below the snow line

The reverse of item 1: a band below the line in every wet hour while the cell reports snow. Seven band-days this week (Mt. Bachelor's base today: 1.4 cm with the line at 1,910 m or higher, 37 mm of rain; Snoqualmie and Mt. Baker on Sunday). Item 1 handles this automatically (the water goes to `rain_fl_mm`); until then the page leaves the cell's snow in place.

## 7. Tests

Add fixtures to `tests/` from today's Timberline hourly responses (scratchpad copies of the per-band calls exist from this session's model check) and assert:
- Monday summit `snow_fl_cm` is between 15 and 19 cm and `rain_fl_mm` is 0;
- no band above the freezing level in every wet hour has `rain_fl_mm` above 0;
- `snow_fl_cm` / 0.7 + `rain_fl_mm` equals the band's `precip_mm` to within 0.2 mm;
- `rain_limit_m` is unchanged (it stays for compatibility).
