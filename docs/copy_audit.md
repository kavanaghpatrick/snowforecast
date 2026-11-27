# Copy audit replacement table

(Recovered from the audit workflow journal, 27 Nov 2026. **Verified row by row against the live
`dashboard_v2/` code on 27 Nov 2026**: every "Now says" template was grepped against today's
source, every "Change to" was checked mechanically against the copy law and semantically against
the function that generates it. Rows whose old string no longer exists are in section 2 (stale);
rows whose replacement was wrong and unfixable are in section 5 (rejected). The earlier checker's
row 1–9 verdicts were rederived from the code, not copied.)

**Conventions used in the replacements**

- **{amt}** is a unit-bearing amount placeholder: one shared helper renders it as "21 cm" in cm
  mode and "8″" in inches mode. No replacement hard-codes "cm" or "″"; the draft table did, which
  broke inches mode on every amount row.
- **~** is prefixed only when the figure is model-derived. Measured or resort-reported figures
  stay bare, even in a template that usually carries ~ (rows 16, 45, 48 say so explicitly).
- **{day}** is a short weekday / today / tomorrow; **{date}** is month-form ("Dec 8").
- Elevations render in the display unit (m / ft), as the code already does.

## 1. Replacement table (68 surviving rows)

| # | View | Now says (live code) | Change to | Severity |
|---|---|---|---|---|
| 1 | Verdict | Ski {day}: {n} snow since {day}. | Ski {day}: ~{amt} snow {day} and {day}. The window is always the two adjacent snow days, e.g. "Ski Tue: ~21 cm snow Mon and Tue." ("since Mon" claimed an open-ended window.) | confusing |
| 2 | Verdict | Ski {day}, on {day}'s {n} snow. | Ski {day} after {day}'s ~{amt} snow. | confusing |
| 3 | Verdict | Ski tomorrow: {n} snow since today. | Ski tomorrow: ~{amt} snow today and tomorrow. (The amount covers both days; the draft's "Ski tomorrow: ~{n} cm snow." dropped the window.) | confusing |
| 4 | Verdict | Ski {day}: ~{n} fresh snow. (model figure, no measurement) | Ski today: ~{amt} new snow. ("of new snow" in the draft added a needless word.) | confusing |
| 5 | Verdict | Fresh snow {day}: {n} from {day}. / {n} since {day}. | New snow through {day}: ~{amt}. ("Through" is cumulative up to and including that day, which matches what the figure covers.) | confusing |
| 6 | Verdict | Fresh snow tomorrow: {n} from today. | New snow through tomorrow: ~{amt}. | confusing |
| 7 | Verdict | Fresh snow {day}: {n}; {wet word} {day}. | New snow through {day}: ~{amt}. Wet snow on {day}. (Or "Rain on {day}." when the threat phase is rain.) | minor |
| 8 | Verdict | Only {n} cm snow this week. Maybe {n} cm snow {date}. | Only ~{amt} snow next 7 days. Next snow on {date}: ~{amt}. | confusing |
| 9 | Verdict | No snow next 7 days. Maybe {n} cm snow {date}. | Models show no snow next 7 days. Next snow on {date}: ~{amt}. | confusing |
| 10 | Verdict | No snow in the next 15 days. / No snow next 7 days. | Models show no snow next {n} days. | minor |
| 11 | Verdict | No snow until {day}. | Models show no snow until {day}. | minor |
| 12 | Verdict | Wind or rain spoils every snow day. | Models show wind or rain on some snow days. (Also more honest: the code only proves some snowy day lost its ski score, not every one.) | confusing |
| 13 | Verdict | Most snow falls {phrase}. | Models put most snow {phrase}. | minor |
| 15 | Verdict | About {n} fresh snow, {window}. | Last {n} days: ~{amt} new snow. (24-hour window: "Since yesterday: ~{amt} new snow."; the figure here is the model's, so ~ is right.) | minor |
| 16 | Verdict | Ski {day}: {n} fresh snow, {window}. (measured/reported figure) | Ski today: {amt} new snow, last {n} days. **No ~**: this branch only fires on a measured or reported figure. The draft added ~ to a measured number and quoted the same amount twice. | minor |
| 17 | Verdict | Warning: forecast is {n} days old. (any age over 36 h shows at least "2 days") | Under 48 hours: Warning: forecast is {n} hours old. From 48 hours: Warning: forecast is {n} days old. | confusing |
| 18 | All resorts, top line | Most snowfall next 3 days: {resort}, {n} cm. (draft's "Most snow:" was stale; this is today's string) | Most snowfall next 3 days: {resort}, ~{amt}. | confusing |
| 19 | Hero label | Snow, next 3 days | Snow next 3 days | confusing |
| 20 | Hero label | Snow depth, mid-mountain / Snow depth at base / near summit / below base | Snow depth at base / at mid-mountain / near summit / below base. (Keep the live "near summit" and "below base" variants the draft dropped: the reading is at the station's height, and "at summit" would overstate it.) | minor |
| 21 | Hero number | {n} cm (bare) | ~{amt} for the 3-day hero number (model figure). The depth numeral beside it is measured and **stays bare**. | confusing |
| 22 | Hero caption | Forecasts agree all week. | Models agree on snow all week. | confusing |
| 23 | Hero caption | Forecasts agree only today. / agree through {day}. / disagree from today. | Models agree on snow only today. / Models agree on snow through {day}. / Models disagree on snow from today. (The disagree form only when agree_through is a real value; see Deletions.) | confusing |
| 24 | Hero line | Mid-mountain · {day} to {day} night · Forecasts agree only yesterday. | Mid-mountain: {day} to {day} night. (Drop the agreement sentence whenever day 0 is yesterday.) | confusing |
| 25 | Hero line | Mid-mountain · {day} to {day} night · Forecasts disagree from today (agree_through missing). | Mid-mountain: {day} to {day} night. (No agreement sentence: a missing value must not become a claim.) | confusing |
| 26 | Hero line | Mid-mountain · {day} to {day} · Forecasts agree all week. | Mid-mountain: {day} to {day} night. Models agree on snow all week. | minor |
| 27 | Days table row | {day} {n}, e.g. Sat 28 (short_day, also the stations' Last reading column and past-day words) | {day} {month} {n}, e.g. Sat Nov 28 | confusing |
| 28 | Days table header | Snow covers (day table and the Powder forecast table; **not** on All resorts — the draft's scope note was wrong) | Snow reaches | confusing |
| 29 | Days table cell | all | ~100% | confusing |
| 30 | Days table cell | none | ~0% | minor |
| 31 | Days table cell | top ~{n}% | upper ~{n}% | minor |
| 32 | Days table cell | dense snow; strong wind, dense snow; rain, dense snow; wet snow, dense snow | heavy snow; strong wind, heavy snow; rain, heavy snow; wet snow, heavy snow | confusing |
| 34 | Days table header | New snow, cm (narrow screens: bare "cm") | New snow, at both widths. The unit moves into the cells (row 35). | confusing |
| 35 | Days table cells | {n} (bare numbers — the draft's "{n} in" cells were never in the code; the unit sat in the header) | ~{amt} in every day cell. All seven day rows are model figures, so the draft's "{n}″ for measured days" clause had nothing to attach to. Applies with row 34. | confusing |
| 36 | Days table total row | 7-day total | Snow next 7 days | minor |
| 38 | Days table, Snow reaches column (share unknown) | {n}–{n} m | Snow line ~{n}–{n} m (ft mode: ~{n}–{n} ft) | confusing |
| 39 | Ground line spark label | Last {n} days, oldest first: {n} in, {n} in, {n} in | This string is an SVG title/aria-label, so "″" (not reliably voiced by screen readers) is wrong there: spell it — "Last {n} days, oldest first: {n} inches, {n} inches, {n} inches." cm mode stays "{n} cm". | confusing |
| 40 | Mountain strip numeral | {n} (bare) | ~{amt} (model figures). The numeral column is 2.5–3 em wide; widen it in the same edit. | confusing |
| 41 | Mountain strip band labels | Narrow labels only: Base {n}, Mid {n}, Summit {n} (wide labels already carry m/ft — the draft overstated the fault) | Narrow labels gain the unit: Base {n} m / Base {n} ft. | confusing |
| 42 | Mountain strip condition | wet Today / rain Today | wet snow today / rain today | confusing |
| 43 | Mountain strip condition | wet {Day} / rain {Day} | wet snow on {day} / rain on {day} (lowercase day word) | confusing |
| 44 | Compare caption | Mid-mountain snow. / Base snow. (also the All resorts caption) | All figures are for mid-mountain. (or base, or summit). A missing figure shows "—". | confusing |
| 45 | Compare / All resorts cells | {n} (bare, every numeric column; headers carry no unit either) | ~{amt} for model cells (3-day, 7-day, Days 8–15). Bare {amt} for measured cells: Snow depth always; Last 3 days only when recent_fallen says measured/reported — that column is per-resort model-or-measured, so the mark is decided per cell. | confusing |
| 46 | Compare cell | — wind / — rain | No ski day, wind / No ski day, rain. A missing value shows "—" alone. | confusing |
| 47 | Compare header | Best ski day (narrow: Best day) | Models' best day. No short form needed: phone headers already wrap to two lines (white-space:normal in both tables' CSS). | confusing |
| 48 | Compare cell | Sat 38 | ~{amt} on {day}, e.g. "~38 cm on Sat". Bare {amt} when the amount is a measured fallen figure (best_ski_display tag "measured"/"reported"). | confusing |
| 49 | Compare cell | Today 23 | ~{amt} today (same measured-figure exception as row 48) | confusing |
| 50 | Compare header | Snow 7 days (narrow: 7-day snow) | Snow next 7 days | confusing |
| 51 | Compare header | Depth (narrow form) / depth | Snow depth (narrow wraps) | minor |
| 52 | Depth tooltip | SNOTEL {station} station ({n} m), {n.n} km away — built in scripts/fetch_data_v2.py, shown as the title attribute on depth cells (_depth_title) | SNOTEL {station} station, {n} m, {n.n} km away, {date}. (Use "—" for the date if it is missing.) The label is baked into the data file, so it cannot switch to ft/mi with the page; acceptable for a tooltip, noted here so nobody "fixes" it in render.py. | minor |
| 53 | Depth tooltip | Meteo-France {station} station … (ALL CAPS names from the provider) | Météo-France {station} station, {n} m, {n.n} km away, {date}. Names in title case — but leave all-caps tokens that are acronyms/codes (e.g. "SAPC") alone rather than producing "Sapc". | minor |
| 54 | Sort control | Sort by: 3-day snow · 7-day snow · best day · depth · A–Z (All resorts also has "last 3 days") | Sort by: Snow next 3 days · Snow next 7 days · Last 3 days · Models' best day · Snow depth · A–Z. (The draft omitted "Last 3 days", which the All resorts page sorts by.) | minor |
| 55 | Show line | Show: Nearby · {country} · All {n} resorts | Show: Nearby · {country} · All 1 resort (for 1); All {n} resorts otherwise | minor |
| 56 | Powder line, resort page | {amt} {kind} forecast {day}. (+ optional "Warmth may crust the snow.") | {Kind} on {day}: ~{amt}. — Kind keeps the quality word, e.g. "Dry powder snow on Sun: ~16 cm." The crust-warning sentence stays. (The draft dropped "dry", losing information.) | confusing |
| 57 | Powder page row | {resort}: {amt} {kind} forecast {day}. | {resort}: {kind} on {day}, ~{amt}. Keep the existing degrade ladder (drop quality word, then the name) for long names past eight words. | minor |
| 58 | Powder page, no data | No powder snow forecast next 4 days. (zero resorts or no forecasts at all) | No forecast for any resort. (Needs a code branch: the live lead() cannot tell this case from row 59's.) | confusing |
| 59 | Powder page, forecasts exist, none qualify | No powder snow forecast next 4 days. | Models show no powder snow next 4 days. | minor |
| 60 | Powder headers | Snow, cm / Snow, in (and the companion "Snow depth, {u}") | New snow; the companion header becomes Snow depth. Units move into the cells, same treatment as row 45 (forecast cells ~, measured/reported fallen-snow and depth cells bare). | confusing |
| 61 | Top bar tooltip | Updated {day} {n} {Mon} {time} CET · {time} UTC (update_line; shown as the hover title on the age) | Updated {day} {Mon} {n}, {time} CET — month before day, drop the UTC half. | confusing |
| 62 | Top bar | Updated {n} h ago / Updated {n} min ago | Updated {n} hours ago / Updated {n} minutes ago (singular at 1: Updated 1 hour ago) | minor |
| 65 | Snow stations header | Change 24 h | 24-hour depth change | minor |
| 66 | Snow stations heading | Snow stations, cm / Snow stations, in | Measuring stations. The depth and change cells then carry their unit (headers name the measure only). | minor |
| 67 | Snow stations header | 15-day trend | Snow depth, last 15 days. (Confirmed: the series is the stations' depth_cm.) | minor |
| 68 | Ground line | Since yesterday {n} cm | New snow since yesterday {amt}. The figure is fallen new snow (recent_fallen, which excludes depth changes), so the draft's "24-hour depth change" mislabelled it — see Rejected. | minor |
| 69 | Picker label | Sierra Nevada, Sierra Nevada | Sierra Nevada, Spain (when region equals the resort name, show the country) | confusing |
| 70 | Where line | Spain · 1,240–1,780 m | Spain · 1,240–1,780 m elevation (ft mode: … ft elevation) | confusing |
| 71 | Footer sources | Forecast: Open-Meteo. Snow depth: USDA NRCS SNOTEL (AWDB), GeoSphere Austria TAWES, WSL … SLF …, Japan Meteorological Agency, AMeDAS, Météo-France, hourly climatological data (Licence Ouverte 2.0), Servei Meteorològic de Catalunya, XEMA, Land Tirol weather stations, LWD and HD Tirol (CC BY 4.0), AEMET OpenData, … Resort reports: OnTheSnow, tenki.jp. | Forecast: Open-Meteo. Snow depth: SNOTEL, GeoSphere, SLF, Japan Meteorological Agency (AMeDAS), Météo-France, Servei Meteorològic de Catalunya (XEMA), AEMET OpenData, Tirol's weather, avalanche and river services. Resort reports: OnTheSnow, tenki.jp. Licences: Licence Ouverte 2.0; CC BY 4.0. (The draft dropped AEMET OpenData and tenki.jp, both live sources; provider names stay for attribution.) | confusing |
| 72 | Error screen | Forecast file is from an older version. | Forecast data is out of date. | confusing |
| 73 | Error screen | Forecast file failed to load. | Forecast data failed to load. | confusing |

## 2. Stale rows (old string no longer in the code)

- **Row 33** (days table footer "{day} and {day} rain counted as snow."). Replaced today by
  recount_note(): "Freezing air {day}: rain falls as snow." The draft's replacement ("Snow totals
  include rain on {day}") would also have regressed the user's 27 Nov ruling that "rain counted
  as snow" reads as malpractice without the why. No action.
- **Row 63** (singular "Updated 1 day ago" / "forecast is 1 day old"). Unreachable: updated_ago
  only says "days" from 48 hours (so ≥ 2), and the verdict warning floors at max(2, …). No action.
- **Row 74** (All resorts "No new snow anywhere next {n} days."). Not in the code. The live
  leaders() renders nothing on all-null data and says "No snowfall next 3 days." / "Little
  snowfall next 3 days." otherwise — which already separates no-data from no-snow. Row 18 covers
  the one live string that still needs a fix.

## 3. Rejected rows

- **Row 14** ("Wet snow below {n} ft today." → "below ~{n} ft"). The height in low_threat() is
  the resort's own band elevation — a fixed, known figure, not a model output. "~" on it is the
  decorative use the law forbids. The sentence stays as it is.
- **Row 37** ("Yesterday" → "yesterday" in the day column). The column's convention is
  capitalized labels ("Today", "Sat Nov 28"); lowercasing one row breaks it. The glossary's
  lowercase rule is for condition cells, not day labels.
- **Row 64** ("Resort searchAll resortsPowder" → add separators). Checked against the rendering:
  the topbar emits separate anchors in a flex row with column-gap, so the items never render
  joined — the concatenation existed only in the audit's text extraction. (This closes the old
  open item.)
- **Row 68's draft replacement** ("24-hour depth change") is rejected as written — the figure is
  new snow, not a depth delta — but the row survives with a corrected replacement in the table.

## 4. Deletions

- "Next 3 days firmer than later." Delete it, both the fallback in `confidence_tag()` and the
  no-models case. The code has no data behind it. (Verified still present at render.py:818.)
- "Forecasts agree only yesterday." Delete the clause whenever day 0 is yesterday. Keep the
  window label alone. (Row 24.)
- "Forecasts disagree from today." and "Forecasts disagree from {day}." Delete the clause
  whenever `agree_through` is None. A missing value must not become a claim. (Row 25.)
- "Maybe", "About", and any second hedge word. The "~" mark does this job.
- "Ski {day}: {n} new snow that day." — **replace, do not merely delete**: two live branches emit
  it (today with snow still falling, and a future day with no snow the day before), and deleting
  it would leave them with no sentence. New form: "Ski {day}: ~{amt} new snow." (the "that day"
  tail was the redundancy).
- "Ski Sun, on tomorrow's …" — covered by row 2's new form ("Ski {day} after {day}'s ~{amt}
  snow."), which cannot name one day twice.
- "Base snow." as a caption. Replaced by row 44.
- "Snow figures are for the …" and "Mid-mountain snow." as captions. Replaced by row 44.
- "Warning: forecast is 2 days old." for ages under 48 hours. Replaced by row 17.

## 5. Glossary (the codebase must follow these)

- **Model figures get "~", measured figures stay bare.** "~" is the only mark. Do not add
  "About", "Maybe", or "Models show" in front of an amount.
- **Inches are ″ in prose and values.** Never "in" as a unit. Exception: spoken/assistive text
  (SVG titles, aria-labels) spells "inches" (row 39).
- **Units sit in each cell, not in headers.** Headers name the measure only.
- **N-day snow is "Snow next N days"** in the hero, headers, and sort line. Retire "N-day snow",
  "N-day total", "Snow N days", and "this week".
- **Forward windows end with "through".** "Through Sun" and "through Dec 9" include that day.
  Retire "since" and "from tomorrow" for forward windows. Backward windows use "last N days".
  A two-day window names both days: "{day} and {day}".
- **New snow**, not "fresh snow" and not bare "snow", for the new-snow column and verdicts.
- **Snow depth**, not "Depth", "depth", or "Snow depth, mid-mountain".
- **Models**, not "forecasts", for model output. "Forecast:" stays only in the source credit line.
- **Models' best day**, not "Best ski day" or "Best day".
- **Snow reaches**, not "Snow covers". Cells read "~100%", "upper ~60%", or "~0%".
- **Condition words:** wet snow, rain, heavy snow. Never "dense", and never a bare "wet".
  Condition cells are lowercase, with "on {day}" after a day name.
- **Dates:** month then day ("Dec 6", "Sat Nov 28"). Weekdays in verdicts and cells use the
  short form ("Mon").
- **Nulls:** "—" alone.
- **Elevations** always carry m or ft. Bands are named ("Base 1,240 m").
- **Dry or timing claims** with no amount may use "Models show …". Amount sentences use "~" and
  no prefix.
- **Labels** have four words or fewer (a number with its unit counts as one word).
- **Time stamps:** "Updated Fri Nov 27, 10:00 CET". Spell out "hours". Drop UTC.

## 6. Rejected findings (from the first audit pass, still standing)

- **"Sat ~38"** for the best-day cell. It keeps the weekday-plus-number pattern the user
  rejected. Use "~38 cm on Sat".
- **"— too windy to ski" / "— too wet to ski".** A dash in front of words breaks the null rule.
  Use "No ski day, wind".
- **"Models show" before every amount sentence.** It repeats the "~" mark and pushes sentences
  past eight words.
- **"Maybe ~16 cm snow Dec 6."** "Maybe" is a hedge. Use "Next snow on Dec 6: ~16 cm."
- **"No base snow report."** A null must be "—".
- **"Models roughly agree".** "Roughly" is a hedge.
- **"Fresh snow Sun: ~8 cm from tomorrow."** "From tomorrow" has the start-day problem. Use
  "through".
- **Dropping AMeDAS, XEMA, LWD, and HD Tirol** from the footer. The CC BY and open-licence terms
  need attribution, so the provider names stay. LWD and HD Tirol are expanded in plain words.
- **"Base–summit {n}–{n} m"** for the bare range in the days table. The cell sits in the Snow
  reaches column, so "Snow line ~680–1,240 m" is used.
- **"Stevens Pass: powder snow forecast Mon, ~16 cm."** The comma stacks two nouns. Use the
  colon form in row 57.
- **Any shortened weekday in a sentence next to "Monday's".** Mixed forms break the rule, so
  short forms are used throughout.

## IMPLEMENTATION ORDER

Build one `{amt}` helper first (render.py, next to `_snow_amount`): display-unit amount with its
unit glued on, "21 cm" / "8″", plus an optional model flag that prefixes "~". Every batch below
uses it. Batches are grouped so each touches one file (or one file pair that shares a constant),
worst severity first. Within a batch, rows list the function the string lives in.

**Batch 1 — render.py, verdict sentences (confusing).**
Rows 1–6 (`_ski_sentence`), 8–9 and 12 (`verdict`, with `_first_ext_snow_day` feeding the
"Next snow on {date}" clause), 17 (`verdict`), 21 (`hero_block`). Also the Deletions items for
"About/Maybe" and the replaced "new snow that day" form (`_ski_sentence`).

**Batch 2 — render.py, hero caption and labels (confusing).**
Rows 19–20 (`hero_block`), 22–26 (`confidence_tag` + the caption join in `hero_block` — the
" · " joined caption becomes the "Mid-mountain: {window}." sentence form), plus the
`confidence_tag` fallback deletion.

**Batch 3 — render.py, days table (confusing, then minor).**
Rows 27 (`short_day`, which also feeds `_day_label`, `rel_word` and the stations' Last reading),
28 (`_sl_head`), 29 and 32 (`snow_cover_str`, `POWDER_WORD`/`watch_day`), 34+35 (`days_section`
header + `_day_row` cells), 38 (`_snow_line_text`); minor rows 30–31 (`snow_cover_str`),
36 (`days_section` total row).

**Batch 4 — render.py, mountain strip and ground line (confusing, then minor).**
Rows 40–41 (`bands_block` / `band_rows`, with the numeral-column CSS widened in `CSS`),
42–43 (`_band_spoiler`, and the matching phrases in `strip_text`), 39 (`ground_line` spark
label); minor rows 15–16 and 7 (`_ski_sentence` measured/threat branches, if not already taken
in Batch 1), 68 (`ground_line` / `_fallen_words`).

**Batch 5 — render.py + everywhere.py, resort tables (confusing, then minor).**
Rows 44 (`compare_section` cap; `everywhere.page` cap), 45 (`resort_row`), 46 and 48–49
(`best_cell`, using `best_display_num`'s tag for the measured exception), 47 and 50–51
(`NARROW_HEADS`, `compare_section` head, `everywhere.COLS`), 18 (`everywhere.leaders`);
minor rows 54 (`compare_section` sorts + `everywhere.SORT_WORDS`), 55 (`compare_section` show).
render.py and everywhere.py share `NARROW_HEADS`/`th_text`, so this is one batch.

**Batch 6 — powder.py (confusing, then minor).**
Rows 56 (`resort_line`), 58 (`lead`, new no-data branch), 60 (`page` table headers + cells);
minor rows 57 (`forecast_sentence`), 59 (`lead`). Row 28's powder copy ("Snow covers" header in
`page`) lands here too.

**Batch 7 — render.py chrome + app-level (confusing, then minor).**
Rows 61 (`update_line`), 69 (`option_label`), 70 (`where_line`), 71 (`footer` `SOURCE_SHORT` /
`KIND_WORDS`, with the source names themselves in `scripts/fetch_data_v2.py`'s sources list —
footer copy can be fixed in render.py alone via `SOURCE_SHORT`), 72–73 (`screen_error`);
minor rows 62 (`updated_ago`), 65–67 (`stations_section` headers), 52–53 (labels built in
`scripts/fetch_data_v2.py` around line 2700, displayed by `_depth_title`; a data-refresh item,
not a dashboard edit).

After each batch: re-run the eight-word check on every sentence the batch touched (powder.py's
`_words` counter is the reference), in both cm and inches mode.
