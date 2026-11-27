# Skim pass: text to marks

Status: implementation spec, 27 Nov 2026. Sits under `design_brief_v2.md` and `ux_hierarchy_spec.md`; where this file and those disagree, those win except on the points this file decides explicitly (marked **Decision**).

Inputs: two expert reviews (a lawful-swap review and a 3-second-scan review), `qa_screenshots/skim_audit/*.png` (16 screenshots, 1440 and 390), `qa_screenshots/skim_audit/text_inventory.md`, `dashboard_v2/render.py`, `dashboard_v2/everywhere.py`.

User directive: "make sure we don't have so much text — replace with useful visual indicators that make this easier to skim without having to read."

## 0. The test every proposal went through

A swap survives only if all three hold:

1. **It is a data mark.** Its length, height or position means a value. A shape that stands for a category (rain, wind, wet, sunny) is a pictogram, which is chrome, and dies.
2. **It needs no legend.** If a reader can't tell what the mark means from the row label, its numeral or a numeral printed at the axis end, it dies.
3. **It loses nothing the skimmer needs.** If the scan review says a decision fact (how much, which elevation, which day, spoiled or not, where else) disappears, it dies.

What the shared vocabulary is: the existing `.bar` (0.18em ink, no track, square ends, `vertical-align:.32em`), the existing `spark_bars` and `spark_line`, numerals from the type scale, position, whitespace, the one accent (`#2C5B8C`, best day / selected row / hero only), and the one new mark defined in 1.2 (the snow-line range). Everything is ink `var(--ink)` on `#F2F3F0`. No grey, no opacity, no new colour.

## 1. The swap list (survivors)

### 1.1 Band rows replace the three band sentences (resort page, under the hero)

Before (Timberline):
`Summit 2,600 m: 51 cm snow, wet snow Sun` / `Mid 2,220 m: 33 cm snow, wet snow Sun` / `Base 1,830 m: 18 cm snow, then rain Sun (10 mm)`

After: three rows on the line grid, summit on top, base at the bottom. Row order is the elevation encoding.

```
Summit 2,600 m   ██████████████████████  51
Mid 2,220 m      ██████████████          33   wet Sun
Base 1,830 m     ████████                18   rain Sun
```

Rendering:

- One `<div class="bands">` using CSS grid, rows 0 apart, each row exactly one line (`--l`: 27px, 26px phone, 30px at 1921+). Sits `--s1` under the hero caption.
- Columns, desktop: `grid-template-columns: max-content 10em 3em auto; column-gap: .5em`. Label column is `max-content` capped at `11em`.
- Columns, phone (≤700px): `max-content 4em 2.5em auto`, label capped at `9em`. On phone the label drops its unit ("Summit 2,600"), because the base and summit with unit are already in the caption under the resort name. This keeps the row to one line at 390 (342px of text width: about 9 + 4 + 2.5 + 4.5 em at 17px).
- Label: body size (18 / 17 / 20), roman, ink, weight 400. "Summit 2,600 m", "Mid 2,220 m", "Base 1,830 m". Three words or fewer.
- Bar: existing `bar(v, vmax)`, ink, never accent. `vmax = scale_max([summit, mid, base], units)`, the existing helper, so the top band is long but rarely full and the 2px minimum for non-zero values holds. Zero or null: no bar.
- Numeral: same size as the label, `tabular-nums lining-nums`, weight 500, right-aligned in its column, no unit (the hero states "cm" or "in" directly above). Null snowfall: "—".
- Word column: 14px, ink, at most two words, only one of:
  - `rain <day>` when that band gets rain in the 3-day window (the existing rain-threat test, first day it happens);
  - `wet <day>` when it gets wet snow and no rain;
  - nothing otherwise.
  The mm figure, "then", "snow" and "wet snow" spelled out are dropped. Day is the short day word already used in the day table ("Sun", "Today").
- Merged band (two bands share one model height, e.g. Crystal's "Mid–summit 1,740–2,130 m"): one row with that label. On phone it becomes "Mid–summit 1,740", the lower edge only, since the caption carries the summit. Then the base row below it. Two rows, same grid.
- Missing band elevation: omit that row. All three missing: omit the block, no sentence in its place.
- No snow-line mark is drawn across these rows. See section 2.

Saves about 12 to 16 words per resort page.

### 1.2 Snow-line range mark replaces the range strings (resort page, day table "Snow line" column)

Before: `1,500–2,470 m`, `840–2,010 m`, `below base`, `above summit`, one string per wet day.

After: one inline SVG per row on a fixed base-to-summit axis that is labelled once, in the header.

Header cell (replaces the single word "Snow line" header):

```
SNOW LINE              (12px small caps, as today)
1,830        2,600 m   (14px ink numerals, tabular)
```

The numeral line is a 120px-wide `display:flex; justify-content:space-between` box, left-aligned in the column, with `padding:0 6px` so the left numeral's left edge sits on the axis start (x=6) and the right numeral's right edge on the axis end (x=114). These are axis end values, not a legend: they are the resort's real base and summit, the same numbers printed under the resort name. Inches mode: feet, "6,000 … 8,530 ft".

Row mark:

```html
<svg class="sl" width="120" height="12" viewBox="0 0 120 12"
     shape-rendering="crispEdges" role="img" aria-label="Snow line 1,500–2,470 m">
  <title>Snow line 1,500–2,470 m</title>
  <line x1="X_LO" y1="6.5" x2="X_HI" y2="6.5" />           <!-- hairline -->
  <line x1="X_LO" y1="3" x2="X_LO" y2="10" />               <!-- end tick, only if unclipped -->
  <line x1="X_HI" y1="3" x2="X_HI" y2="10" />               <!-- end tick, only if unclipped -->
</svg>
```

- CSS: `.sl{ display:inline-block; vertical-align:calc(.41em - 6px); } .sl line{ stroke:var(--ink); stroke-width:1; vector-effect:non-scaling-stroke; fill:none }`. The hairline's centre sits at the same height as the `.bar` centre in the same row (within 1px; check on the screenshot).
- Axis: x = 6 at the resort's base elevation, x = 114 at its summit. `x(v) = 6 + (v - base) / (summit - base) * 108`, rounded to an integer, then +0.5 for the vertical ticks so they land on whole pixels. Same scale on every row of the table, days 1 to 15.
- The hairline is 1px, the snowfall bar is about 3px. The two marks in one row are never the same shape, so they cannot be confused.
- Both ends inside the axis: hairline from lo to hi with a 1px × 7px tick at each end. If lo and hi round to the same pixel, a single tick.
- One end beyond the axis: clip the hairline at x = 6 or x = 114 and draw no tick on that end. An open end means "keeps going".
- Whole range below base (`hi <= base`): a 4px stub from x = 0 to x = 4 at y = 6.5, no ticks. It sits outside the left end of the axis, left of the base numeral in the header, so it reads as "lower than the base". Exact text goes in `<title>` ("Snow line below base").
- Whole range above summit (`lo >= summit`): a 4px stub from x = 116 to x = 120, mirror image. `<title>` "Snow line above summit". The "rain" word in Watch for still carries the threat.
- Dry day (the existing `wet_day` test in `_snow_line` fails): empty cell, as today. A dry day has no snow line, so this is not missing data.
- Wet day with no snow-line data (`snow_line_range` returns None): "—" in the cell, 14px, left-aligned at x = 6. This is a change from today, which leaves it blank.
- Base or summit elevation missing for the resort: fall back to today's range strings for the whole column. Never invent an axis.
- Column width stays 8.5em desktop / 7.6em phone. 120px fits both (153px and 129px).

**Decision on the verdict's snow-line clause.** "Today: snow above 2,220 m; wet snow below." is cut from the verdict. The verdict is one sentence of 8 words or fewer about one day, per the hierarchy spec. The Today row's mark now shows where the line sits. Exception: when the verdict's own day is spoiled low on the mountain (rain or wet snow below the mid band on the recommended day), one threat sentence of 6 words or fewer may follow, naming snow: "Wet snow below 2,220 m Sunday." This is a threat sentence, so it is on the keep list.

Saves about 10 to 24 words per resort page.

### 1.3 Days 8 to 15 summary gets a sparkline (resort page, the collapsed line)

Before: `Days 8–15, less certain: 42 cm snow.`

After: `Days 8–15` ▁▃▁█▂▆▁ `42 cm`

- Summary text "Days 8–15", 18px, underlined as now (it is the disclosure control).
- Then, after a normal space, an 8-bar `spark_bars` variant: `viewBox="0 0 38 18"`, rendered 38 × 18px (17px tall on phone, by CSS height `1em`), bars 3px wide with 2px gaps (x = 5i), ink, height `v / vmax × 18` where vmax is the same `scale_max` the day table uses, 1px minimum for any non-zero value, zero = no rect, null = no rect. Baseline on the text baseline (`vertical-align:baseline`). `shape-rendering:crispEdges`. No accent (best day is a days 1–7 idea).
- Then the total numeral, tabular, weight 500, with the unit one step smaller after a thin space.
- All eight days null: no sparkline, total "—".
- "less certain" and the trailing "snow." are dropped: the collapse already says this is the long range.
- Generalise `spark_bars(values, best_index, vmax, height=12)` to compute `viewBox` width as `5n - 2` so the 7-day and 8-day forms share one function.

Saves 3 words.

### 1.4 Recent snow sparkline on the ground line (resort page)

Before: `Snow depth 48 cm mid-mountain. Last 3 days: 23 cm snow (model).`

After: `Snow depth, mid-mountain 48 cm    Last 3 days ▁█▃ 23 cm`

- Depth stays as a label and numeral. The label is 4 words or fewer and says where the depth is measured (Never 11 allows that in the label).
- "Last 3 days" (label), then a 3-bar sparkline: `viewBox="0 0 13 18"`, 13 × 18px, same bar spec and same vmax as the day table, then the total numeral.
- Only when the per-day series exists. If only a total exists, the total numeral alone. Null: "—".
- "(model)", "(reported)", "(measured)" are deleted everywhere. They are provenance captions (Never 11). Measured-beats-modelled is enforced where the value is chosen, not in a tag.
- Null depth keeps the prose form from the hierarchy spec: "No snow depth report."

Saves 3 words.

### 1.5 Station change: keep 24 h only (resort page, Snow stations)

- Drop the "48 h" and "72 h" columns. The 15-day `spark_line` next to them already shows the multi-day slope, and the dropped columns are what clip at 390.
- Header "Change 24 h". Signed numeral, tabular. Null: "—".

Saves 4 words per station row plus headers.

### 1.6 Cuts that need no mark (all lawful, all keep information)

| Where | Cut | Why it is safe |
|---|---|---|
| Comparison table, resort page | Country small-caps word after the resort name, when every visible row has the same country | The "Show:" control already names the country. Mixed-country views (Nearby across a border) keep it. All resorts keeps it. |
| All resorts and comparison, Best ski day | `none` becomes "—"; `windy` becomes "— wind"; `rain` becomes "— rain" | "—" is the null, not a blank, so the never-blank rule holds. The reason stays attached for the user's 27 Nov bug report (a bare weather word read as broken). |
| Every view | Sources paragraph goes inside a closed `<details>` whose summary is the one underlined word "Sources" | Licence text stays one click away (CC BY 4.0, Licence Ouverte 2.0). Same disclosure pattern as Days 8–15. |
| Every view | "Snow depth reported at 71 of 86 resorts." | The "—" cells already show which resorts lack a report. |
| Resort page, verdict line 2 | "Most snow nearby." / "Little snow nearby either." | Repeats the selected row's rank in the comparison table. "More snow: Timberline, 33 cm." stays (it says where to go). |
| Resort page | "Snow by day, mid-mountain" label becomes "Snow by day" | "Mid-mountain" is already in the hero caption one screen up. The comparison table caption "Mid-mountain snow." stays, because that table is far from the hero. |
| Resort page, hero label | "New snow, next 3 days" becomes "Snow, next 3 days" | 4-word label law. |
| Watch for, every table | "sunny" | Not a threat; the column is for spoilers. |
| Powder view | "Snow type" column and the "Mid-mountain snow." caption | Every row on that page is powder, so the column says nothing. |
| Day table caveat | "Monday and Thursday: model rain counted as snow. It's below freezing up there." becomes "Mon and Thu rain counted as snow." | Keeps the honest caveat, 7 words instead of 13. |
| All resorts verdict | "Most snow: Myoko Kogen 36, Timberline 33, Jackson Hole 33 cm." becomes "Most snow: Myoko Kogen, 36 cm." | Verdict-first stays; the second and third names repeated rows 2 and 3. |

### 1.7 Layout fixes that the skim test needs (390 and 1440)

- Day table at 390: Watch for words must stay on the row's line. Move the Watch for column directly after the numeral (before Snow line) on both widths, which also closes the roughly 300px gap at 1440 between a day's bar and its "rain". At 390, if the row still overflows, Watch for drops to 14px; it never wraps under the bar.
- All resorts at 390: the bar column must render. Hide the km column (empty on All) and the Snow 7 days column on phone instead; the bar sits right after the name at 4em.
- Comparison and All tables at 390: no clipped words ("strong wir"). The table sits in its `overflow-x:auto` wrapper and Watch for cells get `white-space:nowrap`.
- Stations table at 390: fits after 1.5 removes two columns; verify no clip.

## 2. The elevation strip: decided, and it dies as a combined diagram

**Decision: no combined mountain diagram that puts the snow line over the band bars.**

The band totals cover three days. The snow line is a per-day value that moves by hundreds of metres from day to day (Crystal: 840–2,010 m on Sunday, below base on Tuesday). One line drawn across three-day bars would say "the line sits here for all of this snow", which is false. A three-day min-to-max snow-line band across the strip would be honest but would need a legend to explain that it is a span over several days, and a legend kills the swap.

What survives is the two halves, each honest on its own:

- Elevation × amount: the band rows in 1.1 (the brief's own section 5, level 4 "By elevation, hanging list"). Rows ordered summit to base, bars with numerals. Long-on-top, short-at-bottom reads as "spoiled low" at a glance, and the "rain Sun" word names the spoiler.
- Elevation × day: the snow-line marks in 1.2, one per day, on the resort's own base-to-summit axis. Read down the column, the marks move left and right as the snow line falls and rises across the week.

The 390 form and the merged-band case are in 1.1 and 1.2.

## 3. Rejected swaps

| Proposal | Why it dies |
|---|---|
| Model-spread whisker on each day bar, replacing "Forecasts agree through tomorrow" | A reader can't tell what the whisker means without being told. Needs a legend. The confidence sentence stays. |
| Split snow bar (thick dry part, thin wet part) replacing "wet snow" | The thin segment's meaning isn't readable without a legend. "wet snow" stays as words. |
| Gust numeral column with bold weight at the lift-hold threshold, replacing "strong wind" | The bold cue only works if the reader knows the threshold, which is a legend in disguise. Weight 400 against 500 in Garamond at 14px is too faint to skim. It also adds a numeral to every row (more reading, not less) and one more column to tables that already clip at 390. "strong wind" stays. |
| Depth line as a 15-day sparkline, and "—" when only modelled depth exists | Throws away a real value the page has, which loses information. The station table already carries the sparkline. Section 1.4 is used instead. |
| Position mark for Powder's "Snow down to" | The hierarchy spec (5b) requires the elevation numeral there. A mark alone loses the number. |
| Snow-line mark across the band rows | See section 2. |
| Any category mark (rain drop, wind line, sun, snowflake) | Pictograms, banned as chrome. |
| Probability bar beside the snowfall bar | No probability column exists today. If one is added, it is a 14px numeral, because two bars with different meanings in one row need a legend. |

## 4. Keep list

These stay as text because they carry something no mark can.

- **The verdict**, one sentence, 8 words or fewer, one day, naming snow ("Ski Mon, on Sunday's 36 cm."). Any number in it must equal a number visible on the same screen, normally the day-table value (hierarchy spec rule 3). This fixes the clash the scan review found (Crystal says 23 in the verdict against a hero of 16).
- **One threat sentence** after the verdict, only when the recommended day is spoiled low or by wind, 6 words or fewer ("Wet snow below 2,220 m Sunday.", "Wind or rain spoils every snow day.").
- **"More snow: Timberline, 33 cm."** when another resort beats this one.
- **The hero caption** "Mid-mountain · Fri to Sun night · Forecasts disagree from today." It gives the window, the elevation and how far to trust the number. No legend-free mark can carry trust.
- **Watch for words**: rain, wet snow, strong wind, dense snow, dry powder. Categories, not amounts. At most two per cell, rain first.
- **"rain <day>" / "wet <day>"** in the band rows. Rain is the deal-breaker and must not be left to a subtle mark.
- **The shortened caveat** "Mon and Thu rain counted as snow." (honest about how the data was handled).
- **Day names and the 7-day total** in the day table. Bars can't give the exact sum or the day.
- **Every numeral column** in the tables (3-day, 7-day, depth, days 8–15). Bars give rank; numerals give the amount people compare. No second bar per row.
- **Best ski day cell** "Sat 38", or "—", "— wind", "— rain".
- **Country word** on All resorts and on mixed-country comparison views.
- **Nav, "Updated 1 h ago", Sort and Show controls, "Refresh forecast"**. Controls and freshness.
- **"Sources"** as a one-word closed disclosure.
- **"No snow depth report."** and "Snow depth: last report in April." in prose.

## 5. Word budget

Counted the way `text_inventory.md` counts (visible words, closed `<details>` excluded, "—" counts as zero words). Targets are ceilings.

| View | Now 1440 | Target 1440 | Now 390 | Target 390 |
|---|---|---|---|---|
| Default resort (Myoko Kogen) | 371 | 250 | 354 | 240 |
| Rain-threat resort (Crystal) | 412 | 285 | 395 | 275 |
| Snow-line resort (Timberline, Days 8–15 open) | 429 | 300 | 412 | 290 |
| No-data resort (Masella) | 282 | 190 | 265 | 180 |
| All resorts, by 3-day snow | 867 | 720 | 776 | 650 |
| All resorts, A to Z | 867 | 720 | 776 | 650 |
| Powder | 92 | 40 | 91 | 40 |
| Default resort, inches | 377 | 255 | 360 | 245 |

The All view stays word-heavy because "strong wind" and "wet snow" stay (section 3). That is the price of no legends.

## 6. Acceptance

### 6.1 Three-second scan, per view

Show the screenshot for three seconds, then ask. Each item must be answerable without reading a sentence (verdict excepted).

| View | Must be extractable at a glance |
|---|---|
| Any resort, 1440 | The hero amount. Whether the top of the mountain gets more than the bottom (band bar lengths). Whether any band is spoiled (a "rain" word in the band rows). Which day has the most snow (accent bar in the day table). At least one bar visible above the fold at 1440×900 (the band rows put three there). |
| Any resort, 390 | The same, with the band rows each on one line and at least the hero and band rows in the first two screens. |
| Timberline / Crystal | Which days the snow line drops below or near the base (marks hugging the left end or a left stub) and which days it sits high (marks at the right). Days 8–15 shape visible while collapsed. |
| Masella (no data) | That there is nothing: no bars, "—" where a value is missing, "No snow depth report." No empty sparkline frames, no zero-length marks. |
| All resorts, 1440 and 390 | The top resort (accent row 1, longest bar) and the rough ranking (bar lengths), at both widths. |
| Powder | Which resort, how much, which day, from the one row. |
| Inches mode | Same as resort page; axis header numerals in feet. |

### 6.2 No-legend rule

- No text anywhere explains a mark, a symbol or a weight ("bars to 50 cm", "— no report", "thin = wet" are all forbidden).
- Every new mark is self-explaining by one of: a numeral on the same row, a labelled row, or the axis end numerals in its own column header.
- Search the rendered HTML for any new caption added near a mark; there must be none.

### 6.3 Mark rules

- All marks ink. Accent only on the day table's best day (existing), the selected resort row, and the hero numeral.
- Pixel-crisp: `shape-rendering:crispEdges` on every new SVG, 1px strokes on half-pixel coordinates, `vector-effect:non-scaling-stroke`.
- Null is "—", never an empty mark. Zero is no mark. Dry day has no snow-line cell content. An SVG is never rendered with no shapes in it.
- Every SVG has `<title>` and `aria-label` with the exact value.
- No new mark is wider than its column at 390; nothing causes body sideways scroll.

### 6.4 Standing invariants (unchanged, re-checked)

- One ground `#F2F3F0`, one ink, one accent `#2C5B8C`, EB Garamond only, sizes only from the scale (12 small caps, 14, 18/17/20, 27/24/30, display, hero), weight 500 only on numerals at 18px and below.
- One left edge; rows on the 27px line (26 phone, 30 at 1921+), 0 apart inside a group.
- Never 1–12 hold: no boxes, cards, pills, chips, badges, gauges, progress bars, icons as chrome, emoji, italics, grey text, provenance captions, fragment captions, legends.
- Verdict first. Sentences 8 words or fewer and name snow. Labels 4 words or fewer. Table cells bare: one value or at most two Watch-for words, nothing appended.
- Measured beats modelled, chosen upstream, never tagged.
- No made-up values. A missing value is "—" in a cell or a plain sentence in prose.

### 6.5 How to verify

1. Re-run the skim capture (same 16 screenshots and `text_inventory.md`) and compare word counts to section 5.
2. Run `scripts/consistency_sweep.py` and the label/sentence length check from the hierarchy spec.
3. Look at Crystal, Timberline and Masella at both widths. Confirm the band rows, the snow-line column header numerals and marks, and the Days 8–15 sparkline match this spec to the pixel, and that the snow-line hairline lines up with the bar centre within 1px.
