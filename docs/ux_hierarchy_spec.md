# Dashboard v2: information hierarchy and language spec

Status: implementation spec, supersedes the layout and wording parts of `dashboard_v2_redesign_spec.md` and `design_brief_v2.md` where they conflict.
Inputs: four expert reviews (decision design, cold read, reading flow, competitor benchmark), `qa_screenshots/ux_audit/text_inventory.md`, screenshots 01, 05 and the render code (`dashboard_v2/render.py`).

The rule behind everything below: **dense and self-explanatory, not dense or self-explanatory.** Every number has a subject and a unit next to it. Every sentence names snow when it quantifies snow. The page runs in the order a skier makes decisions.

## 0. The four decisions, and how each element is judged

| # | Decision | Question the skier asks |
|---|---|---|
| D1 | Go or not | Is there meaningful new snow at this resort? |
| D2 | Where | Is another resort better? |
| D3 | When | Which day do I ski? |
| D4 | Will it be good | Will rain, wind or warmth wreck it? What's on the ground already? |

An element that serves none of these is deleted. An element that serves one late in the order goes lower on the page. Where the reviewers disagreed, the one whose proposal serves a decision earlier won.

## 0.1 Data contradictions to fix before any layout work

Wording cannot fix a page that contradicts itself. These go to the render logic first (coordinate with the render-logic work in `docs/logic_fixes_render.md`).

1. **One elevation per screenful.** The hero is the summit number (52), but the hero sentence quotes the rain-flag band, which is mid (36), because `rain_threat_72h` takes `snow_cm` from `flag_band()`. Every snow number in the verdict, the hero and the daily table must come from the hero's elevation. Mid and base numbers appear only in the mountain strip, where each line names its band.
2. **Best-day citations must name the real day.** "Best today: Thursday's 3 cm" (Formigal) means *yesterday*, 26 Nov, because `rel_word` returns a weekday name for any day that isn't today or tomorrow. All four reviewers read it as Thu 3 Dec, which proves the wording fails. Past days must render as "yesterday" or "Wed 25". Never show a weekday name that also appears in the daily table for a different date.
3. **Best-day amounts must match something visible.** "Best today: 26 cm fresh" (Crystal) is `best_ski_day_score_cm`, an accumulation across several days, shown next to a table where today is 7. The sentence must state the window ("26 cm snow since Wed"). If the window can't be derived from the data, quote the day's own table value instead. Never print a number in a sentence that the reader can't find or add up on the same screen.
4. **A 7-day total can't be smaller than a 3-day total.** Timberline shows 46 for the next 3 days and 43 for the week, because the 72-hour window (from now) and the 7-day window (calendar days) don't nest. Make the 7-day window contain the 3-day window, or label both windows with their dates. Until that's fixed, the Everywhere table will look broken to anyone who reads two columns.
5a. **Best-ski-day sort is chronological (user directive, 27 Nov).** Sorting any table by Best ski day orders rows by the DAY, soonest first — Today, then tomorrow, then each later day — with amount descending as the tie-break inside a day, and reason-word rows ("none — wind", "none") after all dated rows. It is a planning sort ("who is good today, who is good Saturday"), not a size sort; the size sorts (Snow next 3 days, Snow 7 days) already exist. The descending/ascending toggle flips the day order (soonest↔latest), never switches to amount-first.

5b. **"Snow down to" precision (user request, 27 Nov).** The Powder view's "Snow down to" cell must not print a band word ("mid") — that reads as a claim that snow stops at mid-mountain, when it only means the lowest of our three bands clearing the 10 cm all-snow test. Print the elevation instead: the day's snow line in the page's units ("to ~1,450 m"), from the same per-day snow-line data the daily table's Snow line column uses, with "~" marking a freezing-level-derived value per docs/snowline_decision.md. Fall back to the band's own elevation in meters only when no snow-line value exists that day; "—" when neither exists. Never a bare band word.

5c. **"Ski {day}" needs a base (user concern, 27 Nov: "not sure best ski day makes sense for areas where we have no snow depth").** Astún topped the chronological best-day sort with "Today 26" while its depth was "—": fresh snow on an unknown or bare base is not a ski recommendation. The rule governs the CLAIM, not the ranking. The verdict's imperative forms ("Ski Mon, on Sunday's 48 cm snow.", "Ski today: ...", "Ski Mon; rain arrives Tue.") require an established base: a measured or resort-reported `depth.value_cm` of at least 20 cm. Otherwise the verdict states the same day and amount as a fact, worded "Fresh snow" (snow lying on that day) rather than "Most snow", because the snow often fell the day before and "Most snow" elsewhere on the page means snowfall: "Fresh snow Mon: 26 cm." / "Fresh snow Mon: 48 cm from Sun." / "Fresh snow Mon: 26 cm; rain Tue." / "26 cm snow down, last 3 days." / "Model says 26 cm snow down." A null depth adds nothing more, because the ground line already says "No snow depth report." The 20 cm line is about whether "go ski" is supportable, not about ski quality. Depth never re-ranks, suppresses or filters anything: a thin station reading does not prove thin slopes (Sölden's glacier is open at a 22 cm station reading; stations sit off-piste at their own elevations), so which day is best, the sort order, the Best ski day cell (day and amount, a fact), the ground line and Powder qualification are all unchanged. Code: `render.GO_SKI_BASE_MIN_CM`, `render.has_base`. When the fetcher publishes open/closed status (docs/depth_coverage_plan.md §7), "closed" replaces depth as the stronger signal.

5. **Blank best days on big storms.** Hakuba Valley and Happo-One (49 cm) show a blank Best cell. If no day qualifies, the cell says why — but it must read as an ANSWER, not a weather observation (user bug report, 27 Nov: a bare "windy" next to "Snow 7 days: 49" read as broken). The cell form is "none — wind" / "none — rain" / "none" (≤4 words with the reason attached to the answer), and it is never blank. The resort page's verdict already carries the full sentence ("Wind or rain spoils every snow day."); the cell only echoes its conclusion.
6. **The "best" sort is opaque.** Sorted by best, Mammoth (0 cm next 3 days) ranks above Stevens Pass (14 cm). Sort by the best day's fresh-snow amount, nones last, and show that amount in the cell.
7. **Best-day threshold.** A "best day" with 1 to 4 cm oversells. Name a ski day only when its fresh snow is at least 5 cm. Below that, say "No powder day this week."

None of these fixes invents a value. Where data is missing, the page says so in words.

---

## 1. The hierarchy

### 1.1 Resort page, top to bottom

| Level | Element | Serves | Change |
|---|---|---|---|
| 0 | Top bar: tabs "One resort · All resorts", units "cm / in", "Updated 2 h ago" | none (controls) | Units toggle and update time **move up** from the footer and from under the hero. One line, small type. |
| 1 | Resort picker, then "Japan · 730–1,860 m" | context | Keep as is. |
| 2 | **Verdict sentence**, second-largest type on the page | D3, D1 | **Promoted** from below the fold (it was the "Best …" line under the duplicate Week heading). Always names snow, a day and an amount at the hero's elevation. Templates in section 2.2. |
| 3 | **Hero number** "52 cm" with the label "New snow, next 3 days" and caption "Summit · Fri to Mon" | D1 | Kept, but the label is **rewritten** and set at body size next to the number, not in 7px caps. Confidence tag sits on the caption line: "Forecasts agree only today." |
| 4 | **Mountain strip**: one line per band, Summit / Mid / Base, each with height, snow amount and any rain | D4 | **Promoted and merged.** Replaces the right rail's "Freeze today" and "Rain next 3" tiles, the "Rain tomorrow, buried…" phrase and the whole "By elevation" table. This is the feature no competitor has, so it sits right under the hero. |
| 5 | **Nearby line**: "Most snow nearby." or "More snow: Jackson Hole, 35 cm." | D2 | **New**, built from the existing Nearby compare rows. Puts the "where" answer in the first screenful. |
| 6 | **Ground line**: "Snow depth 48 cm · Morse Lake station, Thu. Last 3 days: 24 cm new." or "No snow depth report. Last 3 days: 21 cm new (forecast estimate)." | D4 | **Merged** from the Depth tile, "On the ground", the "Model summit" row and the "Past 3" column. One line, small type. |
| 7 | **Daily table, days 1–7**: Day · new snow (bar + number) · Snow line · Watch for. Best day highlighted. Last row "7-day total 52 cm". | D3, D4 | Kept, columns **rewritten**. The duplicate "Week / 52 cm" block is **deleted**; its number becomes the total row. |
| 8 | **Days 8–15**: one line, "Days 8–15, less certain: 2 cm snow." Expands to rows on click. Opens by default only if a day has 5 cm or more. | D3 (late) | **Collapsed** from 9 rows to 1. The ambiguous "Total 2" is gone. |
| 9 | **Compare resorts**: "Show: Nearby · Japan · All 86 resorts", then "Sort by: …", then the table | D2 | **Moved up** and made contiguous. Scope links, sort links and table sit together with nothing in between. Same columns as the All resorts page. |
| 10 | **Snow stations** table | D4 (detail) | Shown **only when a station exists** (Crystal). Otherwise the section doesn't render at all. |
| 11 | Footer: sources in plain words, "Snow depth reported at 41 of 86 resorts.", Refresh | none | Kept, reworded. |

**Deleted outright from the resort page:**

- The right rail as a block (Depth / Freeze today / Rain next 3).
- The "Week" heading and the second 52 cm.
- The standalone "Models agree on today only." section. It becomes the confidence tag at level 3.
- The "By elevation" table, its "Past 3" column, its orphan headers ("m", "0–75 cm") and the "Model can't separate mid and summit here." line.
- The "On the ground" heading, the "New snow · cm" caption, the "Model summit 0 0 21" row and "No stations or reports."
- The compare map. It shows unlabelled dots, serves no decision and splits the Compare heading from its table. The table already has a distance column.
- The compare legend "cm · summit · * report · ~ model" and the `*` and `~` marks.
- The compare "24 h" column, the "8–15" column and the generic "storm" flag.
- "Bundled data", the UTC time and the time-zone code.
- The "0–50" bar-scale header.

### 1.2 All resorts page (was "Everywhere"), top to bottom

| Level | Element | Serves | Change |
|---|---|---|---|
| 0 | Top bar, same as the resort page | controls | Same. |
| 1 | Title "All resorts", with caption "86 resorts. Snow in cm at each summit." | context | Replaces "86 resorts · cm · m · summit". |
| 2 | **Leaders line**, heading size: "Most snow: Myoko 52, Hakuba 49, Timberline 46 cm." | D2, D1 | **New.** The page now opens with a conclusion. |
| 3 | Sort line: "Sort by: 3-day snow · 7-day snow · best day · depth · A–Z" | D2 | Moved above the table and reworded. Sort also stays clickable on the headers. |
| 4 | Table: Resort (country in grey after the name) · Snow next 3 days (bar + number) · Snow 7 days · Best ski day · Watch for · Snow depth · Days 8–15 | D2, D3, D4 | Columns **renamed and reordered**. "0 °C" column **deleted**. The country column **merged** into the resort cell. "Days 8–15" **moved last**, in grey. |
| 5 | Footer | none | Same as the resort page. |

The compare scope "Everywhere" on the resort page is **deleted**. It duplicated this page with different columns. "All 86 resorts" now links here.

---

## 2. The language system

### 2.1 Rules

1. Every line names its subject. Wherever snow is counted, the word "snow" is in the same line: "Snow falls Sunday evening", never "Falls this evening".
2. Labels are 4 words or fewer. Sentences are 8 words or fewer, counting each number as a word.
3. No insider abbreviations: no "lvl", "Δ", "SWE", "~", "*", "20h", "Next 3", "8–15" without a noun, "Flags", "Bundled".
4. A skier's vocabulary. Say "snow line" (the height where rain turns to snow), not "freezing level" or "0 °C". Say "snow depth", never "base" for depth (on a ski page "base" means the bottom of the mountain).
5. One name per idea, everywhere: "Snow next 3 days", "Snow 7 days", "Days 8–15", "Best ski day", "Snow depth", "Snow line", "Watch for".
6. Every snow number in a sentence uses the hero's elevation unless the sentence names a different band.
7. Times are words, not clock codes: "Mon evening", "Fri to Mon". Hours map as 00–05 night, 06–11 morning, 12–17 afternoon, 18–23 evening.
8. Days in the past are "yesterday" or a dated short form ("Wed 25"). A bare weekday name always means a future day in the table.
9. Never print a zero amount of rain ("+ 0 mm wet"). Never print rain in mm in inches mode.
10. Missing data is a phrase in prose ("No snow depth report") and a dash only inside table cells, where the table caption says "— no report".
11. Rain only appears as a threat to snow: what day, which band, and whether later snow covers it.

### 2.2 Verdict sentence templates (level 2)

Pick the first that applies. All quote the hero elevation. `{d}` is a day word (today, tomorrow, Sun). `{n}` is a number with its unit.

| Situation | Template | Example |
|---|---|---|
| Ski day after a storm day | "Ski {d}, on {d}'s {n} snow." | "Ski Mon, on Sunday's 48 cm snow." |
| Ski day is the storm day | "Ski {d}: {n} new snow that day." | "Ski Sat: 30 cm new snow that day." |
| Ski day on snow already fallen | "Ski {d}: {n} snow since {past d}." | "Ski today: 26 cm snow since Wed." |
| Ski day, then rain or wet snow arrives | "Ski {d}; wet snow arrives {d}." | "Ski today; wet snow arrives Sun." |
| Snow, but every snow day is spoiled | "Wind or rain spoils every snow day." | |
| Under 5 cm in 7 days, snow in days 8–15 | "No snow next 7 days. Maybe {d} {date}." | "No snow next 7 days. Maybe Sun 6." |
| Under 5 cm in 7 days, light snow only | "Only {n} snow this week." | "Only 1 cm snow this week." |
| Nothing in 15 days | "No snow in the next 15 days." | |
| No forecast | "No forecast for this resort." | |
| Forecast older than 36 hours (overrides all) | "Warning: forecast is {k} days old." | |

When the verdict has no ski day but snow does fall in the next 3 days, the timing template is "Most snow falls {period}." ("Most snow falls Sunday evening."). This replaces "Falls this evening."

The four "Ski {d}" templates apply only on an established base (rule 5c: measured or resort-reported depth of 20 cm or more). Without one, the same day and amount read as a fact: "Fresh snow {d}: {n}." / "Fresh snow {d}: {n} from {past d}." / "Fresh snow {d}: {n} since {past d}." / "Fresh snow {d}: {n}; wet snow {d}." / "{n} snow down, last 3 days." / "Model says {n} snow down."

### 2.3 Mountain strip templates (level 4)

One line per band that has data. Format: `{Band} {height}: {snow part}{, rain part}`.

| Case | Line |
|---|---|
| Snow only | "Summit 1,860 m: 52 cm snow" |
| Rain first, then snow | "Mid 1,290 m: rain Sat, then 36 cm snow" |
| Rain, no snow | "Base 730 m: rain Sat, no snow" |
| Rain then wet snow | "Base 730 m: rain Sat, wet snow Sun" |
| Snow then rain (threat) | "Base 1,340 m: 1 cm snow, then rain Sun" |
| Dry | "Mid 3,250 m: dry" |
| Mid and summit identical | one line, "Mid–summit 1,740–2,130 m: 16 cm snow" |
| Rain amount, only when 5 mm or more | "rain Sat (18 mm)" ("0.7 in" in inches mode) |

### 2.4 Replacement table, old to new

Every flagged string, in page order.

**Top of page and hero**

| Old | New | Note |
|---|---|---|
| RESORT / EVERYWHERE (tabs) | One resort / All resorts | |
| Updated Fri 27 Nov 20:39 JST · 11:39 UTC | Updated 2 h ago | Top bar. Full local time on hover. |
| 52 cm (no label) | 52 cm, labelled "New snow, next 3 days" | |
| To Mon 20h · summit | Summit · Fri to Mon | Window as days, never clock codes. |
| To Mon 04h · summit (Telluride) | Summit · Fri to Sun night | |
| Rain tomorrow, buried by Sun's 36 cm. | Verdict: "Ski Mon, on Sunday's 48 cm snow." Strip: "Mid 1,290 m: rain Sat, then 36 cm snow" | Split. The snow number in the verdict is the summit number. |
| Wet snow Sun on today's 7 cm. | Ski today; wet snow arrives Sun. | |
| Falls this evening. (and every "Falls {period}.") | Most snow falls this evening. | |
| Dry week. | No snow next 7 days. Maybe Sun 6. | Second sentence only if days 8–15 has a day of 5 cm or more (Telluride, 16 cm). |
| Nothing worth chasing. | Only 1 cm snow this week. | |
| Dry until Wed. | No snow until Wed. | |
| No forecast. | No forecast for this resort. | |
| Forecast is N days old. | Warning: forecast is N days old. | |
| Depth / — | No snow depth report. | Ground line, small type. |
| Depth / — (Formigal, stale report) | Snow depth: last report in April. | Use the real report month. |
| Depth / 48 cm / SNOTEL Morse Lake, 1,646 m, 3 km | Snow depth 48 cm · Morse Lake station, Thu. | Height and distance move to the stations table. |
| Freeze today / 2,980 m | Deleted. Strip says "rain to summit" or gives the snow line. | |
| Snow line today / 1,200–1,500 m (tile) | Deleted. Same reason. | |
| Rain next 3 / Tomorrow, buried | Deleted. Strip: "Mid 1,290 m: rain Sat, then 36 cm snow" | |
| Rain next 3 / Sun 18 mm | Deleted. Strip: "Base 1,340 m: rain Sun (21 mm)" | |
| Rain next 3 / None | Deleted. No rain means no rain clause. | |
| Models agree on today only. | Forecasts agree only today. | Caption line under the hero. |
| Models agree through tomorrow. | Forecasts agree through tomorrow. | |
| Models agree all week. | Forecasts agree all week. | |
| Models split from today. | Forecasts disagree from today. | |
| Days 1–3 firm; 8–15 rough. | Next 3 days firmer than later. | |

**Best day and week**

| Old | New |
|---|---|
| Week / 52 cm (heading and big number) | Deleted. Total row "7-day total 52 cm". |
| Best Monday: Sunday's 48 cm. | Ski Mon, on Sunday's 48 cm snow. (verdict) |
| Best today: 26 cm fresh. | Ski today: 26 cm snow since Wed. (verdict, real window) |
| Best today: Thursday's 3 cm. | No powder day this week. (below the 5 cm threshold) |
| No standout day. | No powder day this week. |
| Wind or rain spoils every snow day. | Unchanged. |

**Daily table**

| Old | New |
|---|---|
| Daily · summit | Snow by day, summit |
| Day | Day |
| 0–50 (bar header) | Deleted. |
| cm (header) | New snow, cm |
| 0 °C lvl | Snow line |
| 2,980 / ~2,730 | 2,730 m, or "above summit" when it is above the summit, or "below base" when it is below the base. Only on days with precipitation; blank on dry days. |
| Flags | Watch for |
| rain | rain |
| wet | wet snow |
| rain heavy | rain, dense snow |
| heavy | dense snow |
| powder | dry powder |
| wind | strong wind |
| bluebird | sunny |
| line 1,200–1,500 (flag) | Deleted. It's in the Snow line column. |
| (no total for days 1–7) | 7-day total 52 cm |
| Days 8–15 · rough | Days 8–15, less certain: 2 cm snow. (collapsed line) |
| Total / 2 | Deleted. Folded into the line above. |

**By elevation (deleted section, strings for the record)**

| Old | New |
|---|---|
| By elevation | Deleted. Content moves to the mountain strip. |
| m / 0–75 cm / Next 3 / Past 3 / Snow / wet | Deleted. |
| Fri 1 mm rain · Sat 4 cm + 0 mm wet · Sun 48 cm | Summit 1,860 m: 52 cm snow |
| Sat 18 mm rain · Sun 36 cm | Mid 1,290 m: rain Sat (18 mm), then 36 cm snow |
| Sat 12 mm rain · Sun 0 cm + 35 mm wet | Base 730 m: rain Sat, wet snow Sun |
| Model can't separate mid and summit here. | Deleted. The row label "Mid–summit" says it. |
| Past 3 (21) | Ground line: "Last 3 days: 21 cm new (forecast estimate)." |

**On the ground and stations**

| Old | New |
|---|---|
| On the ground | Deleted as a section. Ground line at level 6. |
| New snow · cm / New 24 h 48 h 72 h / Model summit 0 0 21 | Deleted. |
| No stations or reports. | Deleted. |
| Stations · cm | Snow stations, cm |
| Station · km · m · Depth · Δ 24 h · 48 h · 72 h · SWE mm · 15 days · Read | Station · km · Height · Snow depth · Change 24 h · 48 h · 72 h · 15-day trend · Last reading |
| SWE mm column | Deleted. It doesn't serve a skier decision. |

**Compare**

| Old | New |
|---|---|
| Compare | Compare resorts |
| Nearby · Japan · Everywhere | Show: Nearby · Japan · All 86 resorts |
| Sort next 3 · week · best · depth · name | Sort by: 3-day snow · 7-day snow · best day · depth · A–Z |
| cm · summit · * report · ~ model | Snow in cm at each summit. — no report. |
| km | km |
| Next 3 | Snow next 3 days |
| Week | Snow 7 days |
| 8–15 | Deleted here (lives on the All resorts page). |
| Best | Best ski day |
| Best (blank cell) | none, or the reason in one word (windy, rain) |
| Depth | Snow depth |
| 124* / 30* | 124 / 30 (source and date on hover) |
| 24 h / ~0 / 0* / +2* | Deleted column. |
| Flags: storm | Deleted. The snow number already says it. |
| Flags: storm wet / wet | Watch for: wet snow |
| Flags: rain / storm rain | Watch for: rain |
| All 86 | See all 86 resorts |
| (map) | Deleted. |

**All resorts page**

| Old | New |
|---|---|
| 86 resorts · cm · m · summit | All resorts. Caption: 86 resorts. Snow in cm at each summit. |
| (no conclusion) | Most snow: Myoko 52, Hakuba 49, Timberline 46 cm. |
| Country (own column) | Grey country after the resort name |
| Next 3↓ | Snow next 3 days ↓ |
| Week | Snow 7 days |
| 8–15 | Days 8–15 (last column, grey) |
| Base | Snow depth |
| 0 °C | Deleted. |
| Best | Best ski day |
| Flags: storm wind | strong wind (storm deleted) |
| Flags: wind rain | rain, strong wind (rain first) |
| Flags: bluebird | sunny |

**Footer**

| Old | New |
|---|---|
| Sources / model Open-Meteo · depth USDA NRCS SNOTEL (AWDB), GeoSphere Austria TAWES, WSL Institute for Snow and Avalanche Research SLF, IMIS · reports OnTheSnow | Sources. Forecast: Open-Meteo. Snow depth: SNOTEL, GeoSphere, SLF. Resort reports: OnTheSnow. |
| Depth 41/86 · Bundled data · Map © CARTO, © OpenStreetMap | Snow depth reported at 41 of 86 resorts. (Map credit goes with the map.) |
| cm / in | Moved to the top bar. |
| Refresh | Refresh forecast |

---

## 3. Screen by screen

Screen numbers follow `qa_screenshots/ux_audit/`.

### 01 Myoko Kogen (default, storm with rain first)

Before: tabs · picker · "Japan · 730–1,860 m" · **52 cm** · "To Mon 20h · summit" · "Rain tomorrow, buried by Sun's 36 cm." · "Updated … JST · UTC" · right rail (Depth —, Freeze today 2,980 m, Rain next 3 Tomorrow, buried) · *fold* · Week 52 cm · "Best Monday: Sunday's 48 cm." · Daily table (7 rows + 8 rough rows + Total 2) · "Models agree on today only." · By elevation table · Compare · Nearby · Japan · Everywhere · map · On the ground (Model summit row, No stations) · Sort links · legend · compare table · All 86 · Sources · Depth 41/86 · cm / in · Refresh.

After:
1. Top bar: One resort · All resorts · cm / in · Updated 2 h ago
2. Myoko Kogen, Niigata ▾ · Japan · 730–1,860 m
3. **Ski Mon, on Sunday's 48 cm snow.**
4. **52 cm**, "New snow, next 3 days" · Summit · Fri to Mon · Forecasts agree only today.
5. Summit 1,860 m: 52 cm snow / Mid 1,290 m: rain Sat (18 mm), then 36 cm snow / Base 730 m: rain Sat, wet snow Sun
6. Most snow nearby.
7. No snow depth report. Last 3 days: 21 cm new (forecast estimate).
   *(fold at 1440)*
8. Snow by day, summit: Today 0 · rain to summit / Sat 4 · rain, dense snow / **Sun 48** · snow line 1,220 m · strong wind / Mon 0 · sunny / … / 7-day total 52 cm
9. Days 8–15, less certain: 2 cm snow.
10. Compare resorts: Show · Sort by · table (Hakuba gets a best day or a reason, not a blank)
11. Footer

Dies: right rail, Week block, standalone models line, By elevation table, On the ground, map, legend, 24 h column.

### 02 Telluride (dry)

Before: 0 cm · "To Mon 04h · summit" · "Dry week." · Depth — · Freeze today 3,430 m · Rain next 3 None · Week 0 cm · "No standout day." · table of zeros with Sun 6 16 buried in the rough rows · "Models agree on today only." · By elevation all zeros · compare (Jackson Hole 35 cm at the top) · …

After:
1. Top bar
2. Telluride, Colorado ▾ · USA · 2,660–3,830 m
3. **No snow next 7 days. Maybe Sun 6.**
4. **0 cm**, "New snow, next 3 days" · Summit · Fri to Sun night (no confidence tag, since nothing falls)
5. Strip: Summit 3,830 m: dry / Mid 3,250 m: dry / Base 2,660 m: dry. Collapse to one line, "Whole mountain: dry", when all bands are dry.
6. More snow: Jackson Hole, 35 cm.
7. No snow depth report. Last 3 days: 0 cm new.
8. Daily table (7 rows of 0, total 0)
9. Days 8–15, less certain: 20 cm snow. **Open by default**, because Sun 6 has 16 cm.
10. Compare, footer.

Dies: "Dry week.", "No standout day.", freeze tile, "Rain next 3 None".

### 03 Crystal Mountain (snow now, wet snow coming)

Before: 16 cm · "To Mon 03h · summit" · "Wet snow Sun on today's 7 cm." · Depth 48 cm SNOTEL Morse Lake… · Freeze today 1,510 m · Rain next 3 Sun 18 mm · Week 46 cm · "Best today: 26 cm fresh." · table (today 7) · "Models agree through tomorrow." · By elevation (Mid–summit, Base) · "Model can't separate…" · Compare · map · On the ground with stations table · compare table …

After:
1. Top bar
2. Crystal Mountain, Washington ▾ · USA · 1,340–2,130 m
3. **Ski today: 26 cm snow since Wed.** (window from data; if it can't be derived, "Ski today: 7 cm new snow today.")
4. **16 cm**, "New snow, next 3 days" · Summit · Fri to Sun night · Forecasts agree through tomorrow.
5. Mid–summit 1,740–2,130 m: 16 cm snow, wet snow Sun / Base 1,340 m: 1 cm snow, then rain Sun (21 mm)
6. More snow: Timberline, 46 cm.
7. Snow depth 48 cm · Morse Lake station, Thu. Last 3 days: 24 cm new.
8. Daily table, 7-day total 46 cm
9. Days 8–15, less certain: 38 cm snow. Open by default (Wed 9 has 17).
10. Compare resorts
11. Snow stations table (exists here, so it shows; no SWE column)
12. Footer

### 04 Formigal (no depth data, almost dry)

Before: 1 cm · "To Mon 12h · summit" · "Nothing worth chasing." · Depth — · Freeze today 3,410 m · Rain next 3 Sun 1 mm · Week 1 cm · "Best today: Thursday's 3 cm." · table · "Models agree all week." · By elevation ("Sun 1 cm + 1 mm wet" twice) · …

After:
1. Top bar
2. Formigal, Pyrenees ▾ · Spain · 1,500–2,250 m
3. **Only 1 cm snow this week.** (If days 8–15 has a 5 cm day: "Only 1 cm snow this week. Maybe Wed 9.")
4. **1 cm**, "New snow, next 3 days" · Summit · Fri to Mon · Forecasts agree all week.
5. Summit 2,250 m: 1 cm snow / Mid 1,880 m: 1 cm snow / Base 1,500 m: dry (the 1 mm of rain is under the 5 mm threshold and isn't shown)
6. More snow: Baqueira-Beret, … or "Little snow nearby either." when no nearby resort has 5 cm more.
7. Snow depth: last report in April. Last 3 days: 10 cm new (forecast estimate).
8. Daily table, Days 8–15 line (open, since Wed 9 has 7 cm and Fri 11 has 6), Compare, footer.

Dies: "Best today: Thursday's 3 cm." (below threshold, and misreadable).

### 05–07 All resorts (sorted by 3-day snow, depth, name)

Before: tabs · "86 resorts · cm · m · summit" · Resort · Country · bar · Next 3↓ · Week · 8–15 · Base · 0 °C · Best · Flags (nearly every top row "storm wind", Hakuba Best blank).

After:
1. Top bar
2. All resorts · "86 resorts. Snow in cm at each summit."
3. **Most snow: Myoko 52, Hakuba 49, Timberline 46 cm.** (always the top 3 by 3-day snow, whatever the sort)
4. Sort by: 3-day snow · 7-day snow · best day · depth · A–Z
5. Table: Resort *Japan* · Snow next 3 days (bar + 52) · Snow 7 days · Best ski day · Watch for · Snow depth · Days 8–15
6. Footer

Dies: 0 °C column, Country column, "storm", the units caption.

On 06 (sorted by depth), rows with no depth report sort last and show "—", with the caption saying "— no report".

### 08 Compare, scope Japan, sorted by 7-day snow

Same resort page as 01. The compare block reads "Show: Nearby · **Japan** · All 86 resorts" and "Sort by: … **7-day snow** …", directly above the table. The km column hides when the scope isn't Nearby.

### 09 Compare, scope all, sorted by best

This scope is **deleted**. Its link becomes "All 86 resorts", which opens the All resorts page. The best-day sort there follows section 0.1 item 6.

### 10 Inches

Same as 01 in inches. Rain converts to inches ("rain Sat (0.7 in)"). Heights in ft. No "°C" anywhere on the page.

---

## 4. Acceptance criteria

### 4.1 Cold-read test

Give the page to someone who has never seen it. They pass each item without help.

1. They can say what the big number counts (new snow), where (summit) and over which days.
2. They can say which day to ski, or that there isn't one, from the first sentence.
3. They can say whether rain is a problem and where on the mountain, from the strip.
4. Every label is 4 words or fewer and every sentence is 8 words or fewer. A script checks this against the rendered text.
5. The rendered text contains none of these: `lvl`, `Δ`, `SWE`, `Flags`, `Next 3` (without "days"), `To Mon`, `\d\dh`, `buried`, `fresh.`, `rough`, `Bundled`, `*` or `~` next to a number, `UTC`, `0 mm`, `+ 0`, `°C` in inches mode, `Everywhere`.
6. Every sentence that contains a snow amount contains the word "snow".
7. No weekday name in a sentence refers to a past date.
8. Every number cited in a sentence appears in, or can be added up from, the table on the same page.
9. "Base" never labels snow depth.
10. A blank cell never appears in the Best ski day column.

### 4.2 Five-second test

In the first screenful (1440 × 900 and 390 × 844, no scrolling):

1. Resort page: verdict sentence (best day), labelled hero number, mountain strip, nearby line and ground line are all visible. At 390 this means the hero numeral is capped at 96px and the strip lines don't wrap.
2. All resorts page: the leaders line and at least the top 8 rows are visible, and each of those rows has a Best ski day value or a one-word reason.
3. A tester asked "where and when should I ski?" answers correctly within 5 seconds on screens 01, 02 and 05.

### 4.3 No-orphan rule

1. Every number has its subject and unit within one eye fixation: the same line, or the column header directly above in a table.
2. No section heading opens onto a "no data" line. If a section has no data, it doesn't render. The one exception is the ground line, which states the missing depth in words.
3. No sentence stands alone in its own section. Confidence sits on the hero caption line.
4. No number appears twice at heading size. The hero 3-day number appears once large; the 7-day total appears once, as a table row.

### 4.4 Density rules that survive

1. One ink colour and one accent. The accent marks the best day and the current resort only.
2. Two gap sizes: tight (16–24px) inside a unit (verdict, hero and strip; table and its total), wide (about 96px) between units (forecast, compare, footer).
3. Prose comes only from the templates in section 2. No free text.
4. Watch for holds at most 2 items, rain first. "storm" is never shown.
5. Never fabricate. Missing values are words in prose and "—" in table cells, explained once in the table caption.
6. Snow is the product. Rain appears only as a threat to snow, with its day and band.
