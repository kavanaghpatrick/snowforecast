# Model validation report (27 Nov 2026)

This is the first backtest of the snow numbers the site shows. It pairs the model's own numbers with what 57 weather stations measured, day by day, over last winter (1 Dec 2025 to 31 Mar 2026) and this season so far (15 Oct to 26 Nov 2026). Every number below comes from the scripts in `scripts/validation/` and can be rebuilt from their cached downloads (see "Reproduce" at the end). Nothing here changes a number on the site.

## The answer first

- **The model gets the timing and the storms right, and last winter it got the amounts low.** Over 2,096 station-days with snow, measured snow was 1.48 times the model's (95% range 1.35 to 1.65). Almost all of that is last winter (1.59); this season so far it is 1.11, with the median station at 1.00. The day-to-day match is decent (correlation 0.70) but loose: on a single day the model is off by 5 cm on average, and one day in ten the station got 11 cm or more beyond what the model said.
- **The 7:1 hypothesis holds in direction, on past-day data only.** The colder the storm, the more the model under-counts: measured was 1.8 times the model in storms colder than -8 °C (a figure carried by a few big days), 1.35 times between -3 and -1 °C, and about equal above -1 °C. At SNOTEL sites, where the snow's water is weighed, the observed snow per mm of water was 0.75 to 0.87 cm in cold storms and 0.42 cm near freezing, against the model's fixed 0.7. Temperature explains only a small part of the day-to-day scatter, though; the bigger error in the US is the amount of water.
- **The forecasts people read (days 1 to 3) are less biased than the past-day numbers.** A day ahead, measured snow was 1.10 times the forecast; three days ahead, 1.23 times. Skill fades steadily and is nearly gone by day 6.
- **Last winter, the site's US "past days" and "recent modelled" snow were about half of what fell.** Those numbers come from the first hours of each model run stitched together, and in the US those first hours held about a third less water than the same model's next-day forecast (9,693 mm against 15,940 mm over the same days, with rain gauges at 14,367 mm). This season the same comparison gives 1.17, not 2, so "about half" is not established for now.
- **The snow-line recount is supported in the US.** In the Pyrenees it fixes the totals but not the daily miss (5 stations). On the 356 station-days where it fired, the stations were below freezing (median -3.7 °C; only 21 days above 0 °C), and the recount cut the average error from 8.1 to 7.4 cm. It still leaves the model low, not high.
- **Not tested at all:** Switzerland and Japan (their station feeds keep only the last day or so), and the size of the base/mid/summit split (only 4 usable station pairs).

## 1. What was paired with what

### Measurements

Stations are exactly the ones the site already uses (`stations[]` in `data/forecasts_v2.json`), for every provider that publishes free history:

| Provider | Stations | What it measures | Window used |
|---|---|---|---|
| SNOTEL (US) | 22 | daily snow depth, snow water (snow pillow), gauge precipitation, mean temperature | both windows |
| Météo-France | 23 | hourly snow depth at automatic sites; at manual sites a 07:00 depth and a new-snow board reading (24 h new snow) | both windows |
| GeoSphere (Austria) | 8 | 10-minute snow depth, precipitation, temperature | this season only (the 10-minute archive no longer holds last winter) |
| Meteocat (Catalonia) | 4 | half-hourly snow depth, precipitation, temperature | both windows |

"Measured new snow" is the new-snow board where one exists (10 Météo-France sites, 318 snow days) and otherwise the gain in snow depth over the day (depth change, negative values counted as zero). Depth change under-reads new snow, because new snow settles and old snow melts: on the 164 days where a Météo-France site reported both and the board showed at least 5 cm, depth change captured 73% of the board's total (median 77%). So every ratio below that rests on depth change is, if anything, too kind to the model.

### Model

The model numbers come from Open-Meteo's historical-forecast archive, with the same `best_match` model and the same `elevation=` adjustment the site uses, hour by hour in UTC, summed over exactly the hours each measurement covers. Two versions were pulled:

- at each station's own position and height (the main test of the model), and
- at the resort's position and the site's own base, mid and summit heights (what the page shows). Each station is compared with the band nearest its height.

To check that this archive really is what the site shows, it was compared with the site's own past three days (24 to 26 Nov) for every resort and band: 441 band-days, correlation 0.995, average difference 0.26 cm, totals 1,160 cm against 1,170 cm. It is the same data.

Forecasts at longer lead times come from Open-Meteo's previous-runs archive (the same model's forecast made 1 to 7 days before), for last winter at 38 station sites.

The main Open-Meteo endpoint's daily quota was used up on this machine. The two archive endpoints have their own quotas; one test call each confirmed them before use, and none refused a request.

### Exclusions (none silent)

9,348 station-windows were built; 7,824 were used. Every excluded one is listed with its reason in `data/validation/out/exclusions.csv`:

| Reason | Count | Where |
|---|---|---|
| No depth reading at the start or end of the window | 1,523 | GeoSphere last winter (968: 8 stations x 121 days, archive empty), Météo-France (501, of which 400 this season), Meteocat and GeoSphere short gaps (55). No SNOTEL window was excluded. |
| Depth jump over 100 cm in one day (sensor spike) | 1 | |
| Readings more than 20 cm from the median of the hour around them (dropped before the daily sums, same rule as the fetcher) | 15 | Meteocat |

The exclusions are not random this season. On the 400 Météo-France windows dropped this season, the model had 3.0 cm of snow on average and 1 cm or more on 31% of them; on the kept ones, 2.0 cm and 20%. Missing readings cluster on snowy days, so this season's French and overall ratios lean on the drier days (`audit.py`, section 8).

Not backtested, with the reason:

- SLF (Switzerland, 12 stations), JMA (Japan, 7 stations) and the Tirol feed (5 stations): the endpoints the site uses return only the latest day or hours, and no free archive was found in the time available. Switzerland and Japan therefore have no backtest.
- Resort-reported new snow (OnTheSnow) has no history.
- The site's own single run (`recent.measured` against `recent.modelled`, 64 resorts) was checked too, but split by measurement kind every group has fewer than 30 resorts and it is one storm, so no conclusion is drawn from it.

One pitfall found on the way: SNOTEL's daily values dated D are the readings at the end of day D, not the start. Read the other way, model and station correlate at 0.42 for precipitation; read correctly, 0.88. The backtest uses the correct reading. The site's own SNOTEL code only uses these dates as labels, so the site is not affected.

## 2. Regressions

All regressions are measured = intercept + slope x model, by least squares, on "event" days (model or station at least 1 cm). "Ratio" is total measured over total model. The 95% ranges come from resampling whole stations 2,000 times, because days at one station are not independent. A group with fewer than 30 days or fewer than 3 stations gets no conclusion. Residual spread is the standard deviation of the misses around the line; "p10 to p90" is the range that holds 80% of daily misses (measured minus model).

### Daily, model at the station's own height

| Group | Days | Stations | Measured / model | 95% range | Slope | Intercept (cm) | R² | Residual spread (cm) | Avg miss (cm) |
|---|---|---|---|---|---|---|---|---|---|
| All | 2,096 | 57 | 1.48 | 1.35–1.65 | 1.02 | 2.6 | 0.49 | 7.3 | 5.2 |
| All, with recount | 2,101 | 57 | 1.36 | 1.24–1.50 | 0.95 | 2.5 | 0.51 | 7.1 | 5.0 |
| USA | 852 | 22 | 1.77 | 1.48–2.18 | 0.94 | 4.0 | 0.37 | 7.9 | 6.0 |
| France | 945 | 22 | 1.27 | 1.16–1.41 | 1.06 | 1.4 | 0.63 | 6.5 | 4.5 |
| Spain (Catalonia, plus Puigmal) | 235 | 5 | 1.62 | 1.22–2.08 | 1.26 | 1.6 | 0.42 | 8.0 | 5.2 |
| Austria (this season only) | 64 | 8 | 1.68 | 1.15–2.64 | 1.42 | 0.7 | 0.45 | 4.5 | 3.5 |
| New-snow boards only (France) | 318 | 10 | 1.51 | 1.39–1.64 | 1.15 | 2.5 | 0.69 | 6.7 | 5.1 |
| Last winter | 1,670 | 49 | 1.59 | 1.45–1.76 | 1.15 | 2.4 | 0.51 | 7.3 | 5.4 |
| This season | 426 | 53 | 1.11 | 0.98–1.30 | 0.75 | 2.1 | 0.55 | 5.9 | 4.4 |

The daily misses run from 4 cm too high to 11 cm too low (p10 to p90). The new-snow boards, the cleanest measurement, give the best fit (R² 0.69) and still say 1.5 times.

This season's ratio is near 1 (1.11; the median station's ratio is 1.00, and only half the stations read above the model). In the US it is 1.17 this season against 1.99 last winter. Warm storms on thin ground, where new snow melts on contact, may explain part of that, but nothing here tests that explanation, and this season's snowy French days are under-sampled (see Exclusions). The honest reading is that the low bias is a last-winter result that has not shown up so far this season.

The slopes depend on a few big days. A robust fit (Theil-Sen, the median of all pairwise slopes) gives 0.84 overall against least squares' 1.02, and 0.70 in the US against 0.94. The ratios do not depend on them: dropping the biggest 5% of days moves the overall ratio from 1.48 to 1.45, and 87% of stations read above the model (`audit.py`, section 2).

### Daily, the band the page shows

| Group | Days | Stations | Measured / model | 95% range | Slope | R² | Avg miss (cm) |
|---|---|---|---|---|---|---|---|
| All | 2,120 | 57 | 1.44 | 1.25–1.67 | 0.87 | 0.39 | 5.6 |
| Base band | 657 | 22 | 1.55 | 1.31–1.95 | 1.00 | 0.43 | 5.1 |
| Mid band | 1,346 | 32 | 1.38 | 1.15–1.67 | 0.82 | 0.39 | 5.8 |
| Summit band | 117 | 3 | 1.77 | 0.99–4.18 | 0.93 | 0.27 | 5.5 |
| Band within 150 m of the station | 1,285 | 33 | 1.46 | 1.24–1.75 | 0.94 | 0.40 | 5.7 |

The band numbers fit a little worse than the station-height numbers (R² 0.39 against 0.49), because the resort's grid cell is not the station's. The summit group has only 3 stations, so its wide range means "unknown", not "worse".

### 3-day totals (the "next 3 days" window)

Non-overlapping 3-day blocks with at least 3 cm from model or station:

| Group | Blocks | Stations | Measured / model | 95% range | Slope | R² | Residual spread (cm) | p10 to p90 miss (cm) |
|---|---|---|---|---|---|---|---|---|
| All | 788 | 43 | 1.48 | 1.32–1.69 | 0.92 | 0.51 | 10.9 | -6 to +17 |
| All, with recount | 796 | 43 | 1.30 | 1.17–1.47 | 0.85 | 0.56 | 10.3 | |
| The band the page shows | 796 | 44 | 1.42 | 1.21–1.73 | 0.73 | 0.40 | 12.1 | |
| France | 244 | 11 | 1.11 | 1.00–1.29 | 0.86 | 0.66 | 8.4 | |
| USA | 415 | 22 | 1.76 | 1.48–2.12 | 0.96 | 0.48 | 11.6 | |
| Spain | 106 | 5 | 1.62 | 1.22–2.12 | 1.20 | 0.52 | 12.1 | |
| Austria | 23 | 5 | | | | | | too few blocks, no conclusion |

### The 7:1 test

Open-Meteo turns all frozen water into snow at 0.7 cm per mm, whatever the temperature. Real snow is fluffier in cold storms and denser near freezing, so the hypothesis says the model's under-count should grow as storms get colder. Tested on days where the model gave at least 3 cm, against the model's own temperature weighted by when the snow fell.

Measured / model by storm temperature:

| Storm temperature | Days | Stations | All stations | New-snow boards only | Using the day-1 forecast instead |
|---|---|---|---|---|---|
| below -12 °C | 3 | 3 | too few | | |
| -12 to -8 °C | 59 | 27 | 1.81 (1.29–2.43) | 2.09, too few (9 days, 2 stations) | 1.47 (1.02–2.00), below -8 |
| -8 to -5 °C | 160 | 41 | 1.56 (1.38–1.77) | 1.66 (1.47–2.26) | 1.15 (0.96–1.38) |
| -5 to -3 °C | 226 | 48 | 1.44 (1.28–1.61) | 1.66 (1.38–1.98) | 1.04 (0.88–1.23) |
| -3 to -1 °C | 319 | 52 | 1.35 (1.24–1.50) | 1.34 (1.20–1.52) | 1.18 (1.06–1.31) |
| above -1 °C | 305 | 48 | 0.91 (0.79–1.04) | 1.02 (0.88–1.17) | 0.70 (0.57–0.88) |

As one regression over 1,072 days at 55 stations, the measured/model ratio rises by 0.094 for every degree colder (95% range 0.043 to 0.149). The direction is clear, but temperature explains only 4% of the day-to-day scatter (R² 0.04); most of the miss on any one day is about how much water fell, not how fluffy it was.

The direction survives every robustness check in `audit.py`: a robust slope (−0.099), the slope on log ratios (95% range −0.117 to −0.015), leaving out any one station (−0.083 to −0.106), and 76% of the 42 stations with enough days having their own slope in the same direction. The size of the coldest bin does not: below −8 °C the typical day's ratio is 1.42 with a 95% range of 0.87 to 1.96, and dropping that bin's biggest 5% of days takes the total from 1.81 to 1.52. "About 1.8 times in cold storms" rests on a few big days at 28 stations.

SNOTEL weighs the snow's water, which splits the miss into its two parts:

| Storm temperature | Days | Water measured / model's frozen water | Snow per mm of water, measured (cm) | Model (cm) |
|---|---|---|---|---|
| -12 to -8 °C | 27 | 1.96 | 0.75 | 0.70 |
| -8 to -5 °C | 54 | 1.60 | 0.87 | 0.70 |
| -5 to -3 °C | 72 | 1.55 | 0.70 | 0.70 |
| -3 to -1 °C | 110 | 1.73 | 0.62 | 0.70 |
| above -1 °C | 141 | 1.44 | 0.42 | 0.70 |

In the US the larger problem is water: the model's frozen water was about half the snow pillow's at every temperature (all wet days: 2.1 times, 95% range 1.8 to 2.6). That is mostly the stitched-archive dryness described under lead time below, not the forecast. Density adds a second, temperature-dependent part on top. The measured density figures come from one day's depth gain, which settles, so the real fresh-snow figures are higher by roughly a quarter (from the 73% depth-change capture above). Against the weighed water, measured snow per mm falls by 0.048 cm for every degree warmer (519 days, 22 stations, R² 0.12).

Against rain gauges, the model's precipitation was low at SNOTEL (gauges 1.48 times the model), close in Austria (1.14) and Catalonia (1.17), and high in France (gauges 0.77 times the model). Gauges catch 20 to 50% too little snow in wind, so the French result does not show that the model is too wet there.

### Snow-line recount

The recount (docs/snowline_decision.md) counts the model's "rain" as snow when the band sat above the model's own freezing level all day. The backtest applies the same rule at each station's height, using that point's own freezing level.

| | Station-days | Stations | Measured / model | 95% range | Avg miss (cm) |
|---|---|---|---|---|---|
| As the model gave it | 356 | 27 | 1.91 | 1.61–2.30 | 8.1 |
| With the recount | 356 | 27 | 1.30 | 1.11–1.55 | 7.4 |

The recount brought the number closer on 232 of the 356 days and further on 124. The average error fell by 0.7 cm (95% range 0.15 to 1.2 cm). The thing it assumes, that it was below freezing at that height, held: the stations' own mean temperature on those days had a median of -3.7 °C, and only 21 of 356 days were above 0 °C. At the page's band heights (386 band-days) it moves measured / model from 1.82 to 1.21.

By provider: SNOTEL (278 days, 22 stations) improves from an 8.5 to a 7.4 cm average miss; Meteocat (61 days, 4 stations) gets its total right (801 cm measured, 549 raw, 788 recounted) but its daily miss grows slightly (7.4 to 7.6 cm); Météo-France has a single station (17 days), where it overshoots (134 cm measured, 142 raw, 201 recounted). Across the Pyrenees as a whole (5 stations) the recount makes the average daily miss 0.5 cm worse. The improvement is spread across stations rather than carried by one (23 of 27 improve, the top three give 41% of the gain, and leaving out any one station keeps the gain between 0.62 and 0.88 cm), but all of that is in the US.

### Phase words

On days where a gauge caught at least 5 mm, the model's phase at the station height (the site's own rule) was checked against what the station saw: depth up by at least 2.5 cm means snow; no gain while the station averaged above +0.5 °C means rain; anything else is not counted.

| Model said | SNOTEL: snow accumulated | SNOTEL: rain | SNOTEL: unclear | Météo-France: snow | Météo-France: rain | Météo-France: unclear |
|---|---|---|---|---|---|---|
| snow | 155 | 7 | 38 | 115 | 2 | 19 |
| mixed | 352 | 56 | 35 | 69 | 31 | 13 |
| rain | 19 | 149 | 12 | 13 | 82 | 2 |
| none | 39 | 29 | 39 | | | |

"Snow" was followed by rain on 3.5% of SNOTEL days and 1.5% of French days. "Rain" was followed by settling snow on 11% and 13%. "Mixed" usually ended as snow on the ground at station height (79% at SNOTEL, 61% in France). And 107 SNOTEL days with at least 5 mm in the gauge had no phase from the model at all, because the stitched archive was dry (see lead time).

### Band split

The site's base/mid/summit numbers differ only by Open-Meteo's height adjustment, and in 30 of 86 resorts mid and summit share one grid cell. Testing the split needs two stations at one resort at least 300 m apart with history; there are only 4 such pairs. The model got the direction right in all 4 but the size badly wrong in 2 (Vail: model +36 cm higher up, measured +8; Telluride: model +22, measured +82). Four pairs cannot support a conclusion.

## 3. Forecast skill by lead time

Previous-runs archive, last winter, 38 station sites (SNOTEL, Météo-France automatic sites, Meteocat). "Day 0" is the stitched archive, the same data as the site's past days.

| Days ahead | Days | Measured / forecast | 95% range | Correlation | Avg miss (cm) | 10 cm+ days caught | False alarms | Score (CSI) |
|---|---|---|---|---|---|---|---|---|
| 0 (stitched) | 1,257 | 1.63 | 1.44–1.88 | 0.67 | 5.6 | 35% | 18% | 0.32 |
| 1 | 1,475 | 1.10 | 1.00–1.21 | 0.63 | 5.1 | 55% | 28% | 0.45 |
| 2 | 1,486 | 1.18 | 1.06–1.32 | 0.61 | 5.2 | 45% | 29% | 0.38 |
| 3 | 1,534 | 1.23 | 1.09–1.38 | 0.56 | 5.4 | 42% | 29% | 0.36 |
| 4 | 1,570 | 1.24 | 1.11–1.39 | 0.48 | 6.0 | 40% | 38% | 0.32 |
| 5 | 1,447 | see note | | 0.31 | 6.7 | 17% | 35% | 0.16 |
| 6 | 1,538 | see note | | 0.13 | 7.0 | 12% | 55% | 0.10 |
| 7 | 1,639 | see note | | 0.12 | 7.0 | 17% | 59% | 0.14 |

The CSI score is the share of real-or-forecast 10 cm days that were both forecast and real; 1 is perfect.

By country, a day ahead: USA 1.12 (correlation 0.57), France 1.16 (0.76), Spain 0.92 (0.63). Three days ahead: USA 1.36 (0.57), France 1.05 (0.67), Spain 1.16 (0.36).

Two things in this table are about the archive, not the forecast:

- **Day 0 is drier than day 1.** In the US the stitched first hours of each run held 9,693 mm against 15,940 mm for the next-day forecast over the same days (gauges 14,367 mm). Spain shows the same pattern (1,830 against 3,017 mm); France does not (4,822 against 4,969 mm). The site's "past days" and "recent modelled" numbers are this stitched data, so in the US they under-read what fell by about half.
- **Correlation at days 6 and 7 comes from a few big days.** Its 95% ranges (station resampling) are 0.05 to 0.21 at day 6 and 0.06 to 0.19 at day 7, and the rank correlation, which ignores how big a day was, is zero or below from day 5 on (0.03, −0.15, −0.20). Outside the US the Pearson figures are similar (0.31, 0.19, 0.22). So "little skill from day 5" stands, and if anything is generous.
- **Bias at day 5 and later is not trustworthy in the US.** In the previous-runs archive, the US forecasts lose about two thirds of their water between day 4 and day 5. The same single model (GFS) holds 304 mm at Alta three days ahead and 105 mm five days ahead (Stevens Pass: 796 against 325 mm), which no real forecast does (`model_mix_check.py`). So the day 5 to 7 bias is left out. Correlation does not depend on a constant scale, so the fall in correlation is real.

Next-3 totals by issue day (days 0, 1 and 2 at their own lead): 1,917 issues at 38 sites, measured / forecast 1.25 (95% range 1.15 to 1.39), slope 0.90, R² 0.53, correlation 0.73, average miss 8.2 cm, 80% of misses between 8 cm too high and 15 cm too low. For a storm of 15 cm or more over three days, the forecast caught 55% of them, and 18% of forecast storms did not happen.

The git history also holds the site's own forecasts from 4 to 24 January 2026 (old pipeline). That stretch was nearly dry at the US stations (38 cm in 125 resort-days), so it cannot test anything; the script is kept (`archived_forecasts.py`).

### Best day

The site's best ski day is the day with the most snow from yesterday plus today. The backtest uses the same idea with whole days (the site counts only lift hours today) and asks: on days when the forecast would name a best day (at least 5 cm), did the station's own best day match?

2,025 such issue days at 38 sites. The named day was exactly right 48% of the time and within one day 74% of the time; picking a day at random would be right 14% of the time, but that is too easy a baseline. A rule that ignores the forecast and always names today (the most common measured best day) is right 26% of the time, so the forecast's real edge is 48% against 26%. On 80% of named days the station really did get at least 5 cm. When the named day was 3 or more days out, it was exactly right 42% of the time and had 5 cm or more on 75% of days.

### Powder score

The powder score has two parts that can be tested. Its 10 cm gate is the "10 cm+ days" row above (a day ahead it catches 55%, with 28% false alarms). Its temperature labels were checked against measured snow per mm of water at SNOTEL on model days of 5 cm or more: "warm / heavy" days (above -3 °C) gave 0.57 cm per mm (122 days), clearly denser than "average" days (-8 to -3 °C) at 0.79 cm per mm (86 days). "Cold / dry" (below -8 °C) had only 13 days at 0.74 cm per mm, too few to tell apart from "average". Wind and crust parts were not tested.

## 4. How much to trust each claim on the page

These verdicts were revised after an independent re-check (`scripts/validation/audit.py`, output `data/validation/out/audit.json`). Every verdict below is from one winter (1 Dec 2025 to 31 Mar 2026) unless it says otherwise. The forecast-lead rows rest on depth change only (no new-snow boards are in the lead-time data), which caught about 73% of board totals, so they describe snow on the ground, not snow that fell.

| Claim | Trust | Why |
|---|---|---|
| Snow next 3 days (total) | Good for "is a storm coming and roughly how big". Last winter the amount ran low; this season is untested at forecast lead | Correlation 0.73 by issue day. Measured 1.25 times the forecast on average last winter. The cold-storm excess seen in past-day data does not show up at a day's lead (−8 to −5 °C: 1.15, 95% range 0.97 to 1.38). 80% of misses between 8 cm too high and 15 cm too low. |
| The "likely" range beside it | Fair, for the US and France at station height | Holds 64% of blocks from stations it never saw (95% range 59 to 68%), 66% and 62% when built on one half of the winter and tested on the other, 65% with forecasts a day staler. At the page's band it holds about 1 point less. Spain 58% (49 to 67%, 5 stations). Not tested on this season, where the bias has differed. |
| Days 1 to 3 individually | Moderate | A day ahead: measured 1.10 times, correlation 0.63, 55% of 10 cm days caught. By day 3: 1.23 times, 0.56, 42%. |
| Days 4 to 7 and the 7-day total | Low after day 4; near none from day 5 | Correlation 0.48 at day 4, 0.31 at day 5, about 0.12 by days 6 and 7, and the rank correlation is zero or negative from day 5. In the US the day 5+ amount could not be checked at all. |
| Past days and "recent modelled" snow | Low in the US last winter; this season unclear | US stitched data last winter: measured about 2 times the model. This season 1.17 (22 stations). France about 1.4 last winter, 0.9 this season. |
| Base / mid / summit split | Unknown | 4 station pairs. Direction right in all 4, size wrong by a factor of 3 or more in 2. In 30 of 86 resorts mid and summit share a grid cell, so the split is the height adjustment only. |
| Phase words ("snow", "rain") | High | "Snow" was followed by rain on 1.5 to 3.5% of wet days; "rain" was followed by settling snow on 11 to 13%. Days where the station could not tell were left out (38 of 200 SNOTEL "snow" days). |
| "Mixed / wet snow" | Read as "probably snow, wet" | Ended as snow on the ground at station height on 61 to 79% of days. |
| Recounted days | Supported in the US; not supported in the Pyrenees | US (22 stations): error down 1.0 cm. Pyrenees (5 stations): totals closer but daily error 0.5 cm worse. Stations below freezing on 94% of those days. |
| Cold storms run deeper | Not supported at forecast lead; unknown on a typical day | Past days below −8 °C: 1.81 overall, but 1.52 without the biggest 5% of days. Day-1 forecasts below −8 °C: typical day 0.96 with a 95% range of 0.20 to 1.68 (73 days, 23 stations), which can't tell "deeper" from "shallower". |
| Powder score | Partly | The 10 cm gate inherits the 10 cm row above. "Warm / heavy" really is denser snow; "cold / dry" versus "average" is unproven (13 days). |
| Best ski day | Moderate | Right day 48%, within a day 74%, at least 5 cm on the named day 80%. Always naming today, ignoring the forecast, is right 26%. |

Switzerland, Austria (for anything past a single day) and Japan: no evidence either way. Nothing above applies to them except by analogy.

About the number of comparisons: this report shows 43 group ratios with 95% ranges, and 34 of them exclude 1. Chance alone would put about two of 43 outside, so the general low bias is not a fluke, but any single small group (Austria, Spain, the coldest bin) should be read as a lead, not a finding.

## What would change our minds

Each weak or downgraded conclusion, with the data that would settle it:

- **The low bias this season.** Last winter 1.59, this season 1.11. A full season of the skill tracker (`track_skill.py`, which keeps every issued forecast) at the 22 SNOTEL sites, plus the Météo-France days this season's archive is missing, would show whether the bias returned with colder storms or was specific to last winter. If this season's ratio stays near 1 through January, the "amounts run low" wording and the upward lean of the likely range should both be revisited.
- **US past days at half of what fell.** Two winters disagree (about 2 against 1.17). The fetcher item that adds the day-1 forecast for past days (F1), run side by side with the stitched figures for a few weeks, would show directly which one matches SNOTEL.
- **The likely range.** It was built and tested on last winter only, at station height, from depth change. It needs about 30 live 3-day issues per country with all three days measured (the tracker counts these); coverage well outside 55 to 75% would mean rebuilding it. A version of the previous-runs archive fetched at the page's band heights would test the band directly instead of by proxy.
- **Spain's range and the Pyrenees recount.** Five stations. More Catalan and Andorran stations with snow-depth history (Meteocat has more than the site uses) would make either result a finding.
- **Cold storms.** The day-1 typical-day ratio below −8 °C spans 0.20 to 1.68. Snow temperature at each band (F4), new-snow boards rather than depth change at forecast lead, and at least 150 cold days across 30 or more stations would narrow it enough to say yes or no.
- **Days 5 to 7.** The rank correlation is already near zero. A second winter that showed rank correlation above 0.3 at day 5 would justify removing the "rough guide" sentence; nothing short of that would.
- **Band split.** At least 10 station pairs 300 m or more apart at one resort (SNOTEL has candidates near Alta, Snowbird and Tahoe).
- **Switzerland and Japan.** Any free daily history from SLF or JMA, or a winter of the tracker's own record, which started on 27 Nov 2026.

## What could be disclosed (not applied)

What the product did with these options is in `docs/model_improvements.md`. The 3-day range shipped. The cold-storm line did not, because on day-1 forecasts the median cold day was not under-forecast. The "~" on modelled recent snow and the days 5 to 7 sentence also shipped. The live skill tracker is `scripts/validation/track_skill.py`.

Under the no-recalibration rule, none of this changes a number. Things the page could say honestly, from these results:

- A range on the 3-day total from the residuals: the 3-day misses' 10th to 90th percentiles are about 6 cm too high to 17 cm too low, so "18 cm (likely 12 to 35)" style wording is defensible. It should be computed per country, since France is much tighter than the US.
- (Withdrawn.) A cold-storm line such as "Cold storm: measured snow often runs 1.5 times the model's figure." The 1.44 to 1.81 times across 445 days is past-day data. On day-1 forecasts the same bins give 1.04 to 1.47 on totals and about 1 on a typical day, so the line is not supported for what readers see.
- On US "recent modelled" snow: "Model estimate; it has run about half of measured at SNOTEL sites." Or show the measured value first, which the page often has.
- Days 5 to 7: words rather than numbers, or a "low confidence" mark.

## Open work

- A free history source for SLF and JMA would add Switzerland and Japan (both are big on the site).
- GeoSphere's daily climate dataset (`klima-v2-1d`) has last winter, but under different station numbers. Only St. Anton (11311) matched; mapping the rest by name would add Austria's winter.
- The US day 5+ drop in the previous-runs archive should be reported to Open-Meteo or checked against a saved live run before anyone draws conclusions about it.
- The band split needs more station pairs (SNOTEL has several near Alta/Snowbird and Tahoe that the site does not use).

## Reproduce

```
python3 scripts/validation/fetch_obs.py        # station history -> data/validation/out/obs_windows.csv
python3 scripts/validation/fetch_model.py hf   # historical-forecast archive (about 70 requests)
python3 scripts/validation/fetch_model.py prev # previous-runs archive (about 10 requests)
python3 scripts/validation/model_mix_check.py  # one request
python3 scripts/validation/analyze.py          # pairs.csv, exclusions.csv, results.json, tables.md
python3 scripts/validation/archived_forecasts.py
python3 scripts/validation/audit.py            # independent re-check -> audit.json (offline, fixed seed)
```

Every download is cached under `data/validation/raw/` (keyed by URL; Météo-France files under `raw/mf/`), so a rerun is offline and gives the same output. Two runs of `analyze.py` produced byte-identical `results.json`, because the resampling uses a fixed seed. Every number in this report is in `data/validation/out/results.json`, and the tables are also in `data/validation/out/tables.md`. `data/` is not in git, so the cache stays on this machine; deleting it refetches from the same public sources.
