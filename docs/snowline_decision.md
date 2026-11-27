# Snow line decision (27 Nov 2026)

The question came from Timberline on Mt. Hood: the resort runs from 1,829 m (6,000 ft) at the lodge to 2,603 m at the top of Palmer, and our page said the snow line was 8,100 ft (2,470 m). Does that mean most of the precipitation falls as rain?

No. The Weather Service and the Northwest Avalanche Center (NWAC) put Friday's snow level at 1,500 to 2,230 m (5,000 to 7,300 ft). The lodge sits right at the rain/snow line and gets wet, mixed snow; everything above about 2,200 m gets all snow. Our 8,100 ft was the day's single highest freezing level (2,720 m) minus 250 m, so it overstated the line. Our numbers do have two real problems, but they are the amounts, not the phase:

1. On Friday our summit precipitation is more than double the Weather Service's, so our summit snow is too (65 cm against 34 cm).
2. On Sunday and Monday our data shows rain at every height and almost no snow, while the Weather Service shows 35 to 60 cm of snow up high. On Monday the freezing level never rises above 1,970 m, 630 m below our summit, yet the model's grid cell still calls 23 of its 25 mm "rain". That is the model contradicting itself, not the weather.

All sources below were fetched live on 27 Nov 2026.

## What the sources say, against our numbers

| | Weather Service / NWAC | Ours before this change |
|---|---|---|
| Friday snow level | 1,760–1,880 m early, 2,030–2,230 m midday, 1,740 m at night (NWS grid); 6,000–6,500 ft (NWAC) | One number: 2,470 m (8,100 ft) |
| Friday snow, lodge cell (1,932 m) | 11.7 cm from 40.6 mm, about 3:1, so mostly wet | Base 22.7 cm |
| Friday snow, 2,126 m cell | 22.1 cm | Mid (2,216 m) 38.0 cm |
| Friday snow, 2,794 m cell | 34.3 cm from 40.1 mm, about 8.5:1, all snow | Summit 65.3 cm from 94.5 mm |
| Sunday | Lodge 1.0 cm (essentially rain); 2,794 m cell 18.0 cm | 10.2 mm, 0 cm at every band |
| Monday | Snow level falling from 1,980 m to 1,300 m; lodge "Snow before 4pm", 5.8 cm; 2,126 m cell 12.2 cm; 2,794 m cell 16.5 cm | 25.3 mm, 0.8 cm at every band, freezing level at most 1,970 m |
| Timberline rank | | #1 on best ski day (Sat 65) |

A check of four models (GFS, ECMWF, ICON, GEM) over the same week shows the Monday problem in every one of them: 16 to 31 mm of precipitation with freezing levels of 1,230 to 2,370 m, yet 1 to 3 cm of summit snow. Counting the water as snow wherever the band is above that model's own snow line gives 11 to 22 cm, which brackets the Weather Service's 12 to 16 cm. It also shows our default model (GFS) is the wettest of the four on Friday (77 mm at the summit against 29 to 50 mm for the others), which is why Timberline ranked first.

## The decisions

### 1. Rankings and the headline use mid-mountain snow

Every headline and ranking number (the hero, "Snow next 3 days", "Snow 7 days", "Best ski day", Days 8–15, the daily table, the Most snow line, the nearby line) now speaks for mid-mountain. Summit is used only where a resort has no mid height (none today), then base.

Why:
- Summit flatters wet resorts. In a mild storm the top is the only band above the snow line, so a summit number ranks a resort on snow most of its terrain never sees. That is exactly how Timberline reached #1 on a 65 cm figure the Weather Service puts at 34 cm.
- Base punishes high resorts unfairly.
- At Timberline the mid band (2,216 m) is the closest of our three to the Weather Service's own 2,126 m cell: 38 cm ours against 22 cm theirs, against 65 against 34 at the top. It is still high, because our precipitation is, but it no longer ranks on the flattering band.
- A "skiable snow weighted by terrain" figure is out: we have no terrain-share data, so the weights would be our own estimate, which the no-estimation rule forbids. Switching band only on rainy days is out too: it would make resorts' numbers incomparable in one table.
- Mid is already the band the rain flags, the "Watch for" words and the fetcher's ski score speak for. Moving the headline there fixes the spec's "one elevation per screenful" rule (docs/ux_hierarchy_spec.md, rule 1) for free. The mid height itself is (base + summit) / 2, plain arithmetic on real heights.

Summit and base numbers stay in the mountain strip, where each line names its band. The table captions read "Mid-mountain snow." and the daily table's label reads "Snow by day, mid-mountain".

On today's data the top of the 3-day list becomes Myoko Kogen 36, Timberline 33, Jackson Hole 33 cm (it was Myoko 52, Hakuba 49, Happo-One 49, Timberline 46). Timberline's best ski day becomes Sat 38 (from Sat 65). It is still near the top of the best-day sort, because our model's Friday really is that wet; the fetcher fixes below address that.

### 2. When the snow line cuts through the resort, say where, in words

The verdict gets a second sentence when, on the snowiest day of the next three (at least 5 cm at the top band), the top band gets snow and a lower band gets rain or wet snow with at least 2 mm of water:

> Today: snow above 7,270 ft; wet snow below.

The height is the lowest of our bands that is all snow by the model's own hour-by-hour phase, not the day's peak freezing level. So the sentence is true at the heights we have and never quotes a line the precipitation didn't fall at. For Timberline it says snow above the mid band (2,216 m, 7,270 ft) and wet snow at the lodge, which is what the Weather Service says too ("Snow before 10am, then rain and snow" at the lodge; snow level peaking at 7,300 ft). 15 of 86 resorts get this sentence today.

Why words and not a number in a table: the Weather Service puts the line in the sentence ("Snow level rising to 7300 feet"), snow-forecast.com keeps each elevation in its own table, and OpenSnow colours the day "snow/rain". None of them invents an amount at a height the model didn't give one for.

### 3. The snow line column shows a range, not the peak

The daily table's Snow line cell now shows the line's range over the day's wet hours, for example "1,500–2,470 m" (4,920–8,100 ft) for Timberline on Friday. Before, it showed only the top of that range, which is what misled. "below base" and "above summit" still replace the numbers when the whole range sits outside the resort. The range uses the model's snowfall height where Open-Meteo gives one, else its freezing level minus 250 m, the same line the fetcher's phase rule uses. The Weather Service's Friday range (1,740 to 2,230 m) sits inside ours.

### 4. A model that contradicts itself: count its rain as snow, only when its own freezing level says so

When a band sits at or above the freezing level in every wet hour of a day, and that band's own hour-by-hour phase has no rain, the grid cell's "rain" at that band is snow by the same model's temperatures. The page counts that water as snow at Open-Meteo's own 0.7 cm per mm (the median ratio in our own file is 0.70). This is derivation from the model's own water and the model's own freezing level, not an estimate. Every other day is left exactly as the model gave it.

The test is deliberately strict:
- the band must be above the freezing level itself, not just the snow line 250 m under it, in every hour with precipitation;
- its own phase must show less than 1 mm of rain;
- the day must be in the hourly range (days 0 to 6); days 8–15 have no hourly freezing level and are never touched;
- at least 1 mm of the cell's liquid is needed before anything is recounted.

It changes 37 band-days across 13 resorts this week (the Cascades, the Sierra, Park City, Niseko and Rusutsu), all in the "too little snow" direction. Timberline's Monday becomes 17 cm at mid and summit (0.8 cm + 23.2 mm × 0.7), against the Weather Service's 12.2 cm at 2,126 m and 16.5 cm at 2,794 m. Its Thursday becomes 4 cm.

The opposite case (a band below the snow line all day while the cell reports snow) shows up 7 times this week. The page does not remove that snow: the fetcher's phase rule already counts frozen water below the line as wet snow rather than rain, and deleting a model's snow is a bigger claim than recounting its rain. It is on the fetcher list instead.

Labelling: one sentence under the daily table, only on resorts where a number on the page includes recounted rain. It names the day when it can:

> Monday and Thursday: model rain counted as snow. It's below freezing up there.

No table cell carries a mark, no legend explains one (design brief, rule 12). The All resorts table uses the same numbers with no note; the resort page is where the reader can see the day and the sentence.

A recounted band-day is treated as snow everywhere on the page: it no longer triggers "wet snow" in Watch for, in the mountain strip, or in the "wet snow arrives" verdict clause.

### 5. What the page still cannot fix

- Our precipitation is too high on wet Cascade days (about double the Weather Service at Timberline on Friday). Nothing in the file says by how much, so the page shows the model's water as it is. The hero caption already says "Forecasts disagree from today." when the second model differs.
- Sunday at Timberline stays as the model gave it (10 mm of rain everywhere). The freezing level that day reaches 2,670 m, above the summit, so the strict test above does not apply. Splitting a day like that needs the hourly data the fetcher has and the page doesn't.
- The best ski day is still chosen by the fetcher on summit snow. The page now also requires at least 5 cm at mid before it names one.

## Code

- `dashboard_v2/data.py`: `hero_elevation()` returns mid first; `next3()`, `week_total()`, `extended_total()` and `day_snow()` follow it. New `_recount_mm()`, `recounted()`, `seg_snow()`, `seg_recounted()`, `next3_band()` do the recount. `best_ski_index()` also needs 5 cm at the hero band. `rain_threat_72h()` treats recounted segments as snow.
- `dashboard_v2/render.py`: `split_line()` (the verdict's second sentence), `recount_note()`, `snow_line_range()` and the new `_snow_line()`; captions name mid-mountain; the mountain strip totals use `next3_band()`; the Snow line column is wider for ranges.
- `dashboard_v2/everywhere.py`: caption "Mid-mountain snow."
- Where docs/ux_hierarchy_spec.md names the summit as the hero band (section 0, item 1, and the "Summit snow." caption), this note supersedes it. The rule itself (one elevation per screenful) stands, now at mid-mountain.

Fetcher work that would do this properly is in docs/snowline_fetcher_fixes.md.
