# Model improvements after the backtest (27 Nov 2026)

This turns the findings in `docs/model_validation_report.md` into product changes. One rule governs all of it: a number presented as the model's stays the model's. What the backtest learned may be disclosed next to it (a range, a "~", a sentence), but it is never folded into the number.

## 1. Decisions

### 1.1 The 3-day hero runs low and its misses are wide: show a likely range beside it

The finding: measured 3-day snow was 1.25 times the forecast on average, and 80% of misses fell between 8 cm too high and 15 cm too low.

Three options were on the table.

- (a) A range next to the model figure, built from the measured misses. **Chosen.**
- (b) A calibrated hero ("~22 cm", with the model's 15 cm somewhere below it). **Rejected.** The page would lead with a number no model produced, and the factor isn't stable enough to apply: it was 1.11 this season against 1.59 last winter, 1.04 in France against 1.26 in the US, and below 1 for forecasts over 40 cm. A second figure for the same claim also breaks the page's one-figure-per-claim rule (the density law).
- (c) Wording only ("usually more than forecast"). **Rejected** as the main fix. It says less than a range and would be false for big storms, where the median ratio is 0.91 to 0.99.

The design brief already gives ranges a form ("12 cm, 6–20", section 6.7), so the range is a caption phrase and not a new element.

How the range is built (`scripts/validation/hero_ranges.py`):

- The data is real 3-day forecasts from last winter (days 0, 1 and 2 at their own lead, from the previous-runs archive), compared with measured snow at 38 stations: 1,341 blocks of 5 cm or more.
- For each forecast size (5–10, 10–20 and 20+ cm) and each country (USA, France, Spain), the range is the 17th to 83rd percentile of measured ÷ forecast. That middle two thirds is what "likely" means.
- A cell needs 30 blocks and 5 stations. Otherwise it falls back to the pooled value for its size.
- Checked by leaving one station out at a time, the range held 64% of blocks from stations it had never seen (USA 65%, France 65%, Spain 58%), against a target of 66%.

| Country | 5–10 cm forecast | 10–20 cm | 20 cm or more |
|---|---|---|---|
| USA | ×0.35 to ×2.52 | ×0.51 to ×2.01 | ×0.63 to ×1.81 |
| France | ×0.14 to ×2.06 | ×0.49 to ×1.61 | ×0.67 to ×1.34 |
| Spain | ×0.00 to ×2.26 | ×0.35 to ×1.96 | ×0.54 to ×1.42 |

When the page shows no range:

- Switzerland, Austria and Japan were never backtested, so they get none. Borrowing the US or France spread would be a guess presented as evidence.
- Forecasts under 5 cm get none. A quarter of those real blocks measured nothing, so the ratio means little.
- Windows that include the snow-line recount get none. The table was built from the raw model, and the recount already moves the figure up.

Known limits:

- The table is built at station height. The page's band fits a little worse (R² 0.40 against 0.51), so the live range may hold slightly less than two thirds.
- Blocks overlap by issue day, so the effective sample is roughly a third of the count.
- The size buckets make small jumps at 10 and 20 cm.
- It is built from depth change, which caught about 73% of new-snow board totals, so it describes snow on the ground at the end of the window. Snow that fell can sit above its top.
- It is one winter, last winter, when the bias ran higher (1.59) than this season so far (1.11). Its upward lean (US median ×1.27 to ×1.41) carries last winter's bias. If this season stays near 1, the range will read high, and the tracker is what will show it.
- Day 0 in the table is the stitched first hours, which last winter were drier in the US than a real forecast. With every day a day staler (leads 1 to 3, nearer the page's single run), the US range still held 66%, with 21% of blocks below it and 14% above.

Re-checked independently (`scripts/validation/audit.py`): the table rebuilds exactly from `pairs.csv`. Leave-one-station-out coverage is 64% (95% range 59 to 68%, resampling stations). Built on Dec to Jan and tested on Feb to Mar it holds 66%, and built the other way round 62%. On the same blocks, the page's band holds about 1 point less than station height. Spain is 58% (49 to 67%) on 5 stations, so Spain's range is the weakest of the three.

The harness (section 4) measures the live coverage. If it settles away from about two thirds, the table should be rebuilt from the live history by a person running the script. It never updates itself.

### 1.2 The cold-storm note: not shipped

The 7:1 hypothesis holds on the stitched archive: measured was 1.81 times the model below −8 °C. But on what readers actually see, the forecast a day ahead, it does not hold day by day.

| Storm temperature | Days | Measured ÷ day-1 forecast (totals) | Median day | Share of days where measured beat the forecast |
|---|---|---|---|---|
| below −8 °C | 73 | 1.47 (1.02–2.00) | 0.96 | 49% |
| −8 to −5 °C | 149 | 1.15 | 0.80 | 46% |
| −5 to −3 °C | 183 | 1.04 | 0.99 | 49% |
| −3 to −1 °C | 221 | 1.18 | 1.17 | 57% |
| above −1 °C | 205 | 0.70 | 0.55 | 32% |

The cold-storm excess comes from a few big days. On a typical cold day the forecast was about as likely to be high as low. So a line like "cold storm: usually deeper than forecast" is not supported, and it is not shipped. That is not the same as showing it false: below −8 °C the typical day's ratio has a 95% range of 0.20 to 1.68 (73 days, 23 stations, depth change only, a third of them measuring zero), so the data can't say which way a typical cold day goes (`audit.py`, section 5). The cold effect still shows up inside the hero range, which is skewed upward for 5–20 cm forecasts.

Near freezing (above −1 °C), measured was usually below the forecast. Part of that is how it's measured (wet snow settles fast, so a change in depth under-reads it), and the page already labels those days "wet snow" or "dense snow". No new wording.

Reproduce: `python3 scripts/validation/hero_ranges.py` prints this table and saves it as `cold_storm_day1` in `data/validation/out/hero_ranges.json`. A live version needs a per-band snow temperature (F4).

### 1.3 The page's modelled recent snow runs low: mark it "~"

The ground line's "Last 3 days" falls back to the model's hindcast when there's no measured new snow. That happens at 43 resorts today: 16 in France, 13 in Spain, 10 in Austria, 3 in the US and 1 in Japan. The backtest found measured snow was 1.2 times this figure in France and 1.8 times in the US, but that is mostly last winter: this season so far it is 0.9 in France and 1.17 in the US. The "~" (this is a model figure) holds either way; any stronger wording would not.

- Shipped: the model figure now reads "~23 cm", using the "~" the page already uses for model figures (verdict, snow-cover column). Zero stays "0".
- Not shipped: a stronger phrase for the US such as "at least", because a single day can still come in lower. The fix for the US is better data (F1), not wording.

### 1.4 Days 5 to 7 carry little skill: one sentence under the table

Correlation was 0.48 at day 4, 0.31 at day 5 and about 0.12 at days 6 and 7. Shipped: "Amounts from Wednesday on are a rough guide." It appears under the day table, and only when a day five or more days ahead (counted from the resort's own today) shows 5 cm or more. When nothing far out would tempt a decision, the page says nothing. The numbers stay as the model gave them.

### 1.5 Claims that need no change

- Phase words: high trust.
- "Mixed": read as wet snow, which the page already says.
- Best day: moderate, exactly right 48% of the time against 14% for a random pick.
- The recount: supported, though the Pyrenees overshoot at one station is worth watching.
- "Warm/heavy" powder label: real.

"Cold/dry" against "average" is unproven, but the label is wording about the snow type, and the model's temperature is real. No change.

## 2. What shipped on the page (dashboard_v2/, no new data)

| Where | Before | After | Code |
|---|---|---|---|
| Hero caption | `Mid-mountain · Fri to Sun night · Forecasts agree through tomorrow.` | `Mid-mountain · Fri to Sun night · Likely 8–32 cm · Forecasts agree through tomorrow.` (Crystal Mountain, 16 cm) | `render.hero_range_words`, `skill.likely_range`, `skill_ranges.json` |
| Ground line, model fallback | `Last 3 days ▁█▃ 23 cm` | `Last 3 days ▁█▃ ~23 cm` | `render._ground_num(approx=)` |
| Under the day table | (nothing) | `Amounts from Wednesday on are a rough guide.` | `render.rough_note`, `skill.ROUGH_FROM_INDEX` |

Today the range shows at 14 resorts, all US (no French or Spanish resort has 5 cm or more in its window). Examples: Jackson Hole 33 cm, likely 21–59. Park City 12 cm, likely 6–23. Inches convert the same way (Jackson Hole 8–23 in). Timberline and Stevens Pass show none, because their windows include the recount.

`dashboard_v2/skill_ranges.json` is generated. Rebuild it only with `python3 scripts/validation/hero_ranges.py`, which is byte-identical on rerun.

## 3. Fetcher queue (scripts/fetch_data_v2.py, not edited today)

Each item adds new, separately named fields. No existing field changes meaning.

- **F1. Past days from the day-1 forecast (aimed at last winter's US "half of what fell", which this season's data does not repeat: 1.17).** The stitched first hours of each run held 9,693 mm where the next-day forecast held 15,940 mm (gauges 14,367 mm). For past days −3..−1, also fetch the previous-runs archive's `snowfall_previous_day1`, `precipitation_previous_day1` and `rain_previous_day1` at each band. Publish them as `days[].{band}.snowfall_prev1_cm`, `precip_prev1_mm` and `rain_prev1_mm`, plus `recent.modelled.{band}.h24_prev1_cm`, `h48_prev1_cm` and `h72_prev1_cm`. The endpoint is `previous-runs-api.open-meteo.com`, which has its own quota, so test it with one call first. The page decides on a switch only after `track_skill.py` shows the prev1 figures beat the stitched ones live. Both stay in the file, so nothing is hidden.
- **F2. Model water beside recent snow.** This is the same as `docs/powder_fetcher_fixes.md` item 5: `recent.modelled.{band}.h24_mm` and `h72_mm`. It lets the harness compare water at SNOTEL, which tells too little water apart from too-dense snow.
- **F3. Keep issued forecasts.** `docs/powder_fetcher_fixes.md` item 2 is now covered outside the data file: `track_skill.py` keeps every issued day 0..6 in `data/validation/history/`. The fetcher item can be dropped, or reduced to reading from that history.
- **F4. Per-band snow temperature.** `days[].derived.snow_temp_c.{base,mid,summit}`, the snowfall-weighted hourly temperature at each band (today's `powder_temp_c` is summit only). This is for measurement only. It lets the harness test the cold-storm question at the page's own band before anyone writes a cold-storm sentence.
- **F5. `observed_utc` on every station reading, SNOTEL included.** Other-provider readings without a time can't be placed on a day and are skipped by the harness.

## 4. Permanent skill harness (scripts/validation/track_skill.py)

- `record` (the default) appends the current file to `data/validation/history/YYYY-MM.jsonl`. It stores issued forecasts (days 0..6 from the first run of each local day, past days from the first sighting), the hero and its likely range, every station reading (SNOTEL series rows, other providers' readings with their time), and the page's measured-versus-modelled recent snow. A run already recorded appends nothing. It uses the standard library only and makes no network calls. The first run added 556 KB. Later runs should add far less, an estimated 40 KB each, because only new days and changed readings are written.
- Hero records carry `recount_in_window` and `page_shows_range`, because the page hides the range when the window holds recounted rain. Before this was added, the first run's hero records gave a range at 17 resorts where the page showed 14 (Stevens Pass, Mt. Bachelor and Timberline had a range on record but none on the page). The flags are None where the page's data module can't be imported (it needs streamlit), and coverage skips issues flagged as recounted. The first run's records, written before the fix, have no flag.
- `summary [--json PATH]` scores everything so far, using the backtest's own definitions:
  - daily forecast against measured by lead (stitched past day, and days 0..6), for SNOTEL's exact local days and separately for other stations' 24-hour readings taken within 6 hours of midnight
  - live coverage of the likely range
  - recent measured against modelled, by country
  - groups under 30 days or 3 sites print "too few for a conclusion"

  This also starts the Switzerland and Japan history that the backtest couldn't get, because their feeds keep only the last day.

### Workflow queue: wire it into `.github/workflows/refresh_v2.yml`

Add after "Fetch forecast data" and before "Commit and push":

```yaml
      - name: Record forecast skill history
        continue-on-error: true      # never blocks a data refresh
        run: python scripts/validation/track_skill.py
```

In "Commit and push if changed", stage the history next to the data file (`data/` is gitignored, hence `-f`):

```bash
          git add -f data/forecasts_v2.json data/validation/history/*.jsonl
```

Optionally add a weekly job (or a `workflow_dispatch` input) that runs `python scripts/validation/track_skill.py summary --json data/validation/history/summary.json` and commits that file, so trust numbers are visible without a local checkout.

## 5. Reproduce

```
python3 scripts/validation/analyze.py        # pairs.csv (offline, from the backtest cache)
python3 scripts/validation/hero_ranges.py    # dashboard_v2/skill_ranges.json + data/validation/out/hero_ranges.json
python3 scripts/validation/track_skill.py    # append the current data file to the history
python3 scripts/validation/track_skill.py summary --json data/validation/out/skill_summary.json
```
