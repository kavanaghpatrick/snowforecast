# Snow forecast dashboard: design brief v2

This brief turns the wedding design library (`/Users/patrickkavanagh/weddingwebsite/design/`: TASTE, STYLE_GUIDE, ARCHETYPES, CHECKLIST) into rules for the Streamlit snow dashboard. The library was made for a painting-led wedding site. Here the data plays the part the paintings played there: the numbers are the thing people came to see, and everything built around them is a quiet, exact frame.

Every builder reads sections 1, 2 and 11 first. The rest is reference.

## 1. The idea in one sentence

A forecast set like the front page of a good newspaper's weather section: one flat snow-coloured page, one dark ink, one cold blue accent, serif type with very big numbers and very small labels, and no boxes anywhere.

What survives from the library, adapted:

| Library rule | What it means here |
|---|---|
| The art carries the page; code is the frame | The forecast numbers carry the page. Chrome is near invisible. |
| One ground | One paper colour for the whole app, sidebar and popovers included. No cards, metric tiles, expanders, bands or section backgrounds. |
| One ink, one accent | All text is ink. The accent marks only fresh snow: the biggest number, the best day's bar, the selected resort on the map. |
| Serif-led, big scale contrast | EB Garamond. A hero numeral 7 to 10 times body size. Labels in small caps. |
| One left edge | Every heading, label and sentence starts on the same vertical line. Numeral columns right-align on one second edge. |
| Uneven splits | 5:7 or 4:8. Never `st.columns(2)` with equal widths, never four equal metric columns, never seven equal day columns. |
| Space from one line unit | 27px line, every vertical gap a half or whole multiple of it. |
| At most one visible rule per screen | The resort picker's underline is the one rule. Data marks are not rules (see 6.4). |
| No flat app chrome | No st.metric, no progress bars, no pills, no badges, no filled toggles, no emoji. |
| Pictures float, never rectangles | The map bleeds off the right edge and fades into the paper on its left. It never sits in a bordered box. |

What does not carry over: paintings, cut-outs, surreal objects, live paintings, Avery & Blake, Lago Maggiore, the Italian copy rules.

## 2. Colour

| Token | Hex | Use | Contrast on paper |
|---|---|---|---|
| `--paper` | `#F2F3F0` | The one ground. Cool off-white, the colour of snow under an overcast sky. Not pure white (too clinical), not cream (too wedding). | |
| `--ink` | `#1A2230` | All text, all data marks, all lines. A blue-black like a winter night. | 14.3:1 |
| `--accent` | `#2C5B8C` | Glacier blue. Fresh-snow emphasis only. | 6.3:1 (passes AA at every size) |

Rules:

- Two text colours on the page, ink and accent, and the accent only on the numbers listed in 6.1. Headings are always ink.
- No second ink made with opacity or grey. Hierarchy comes from size and position, never from fading text. If something seems to need a grey, it is either a label (make it 14px small caps in ink) or it should be cut.
- Accent appears on at most two kinds of thing per screen.
- Data colours are ink and accent only. Map points, bars and sparklines use the same two colours. Amount is shown by length, height or size, never by a colour ramp.
- Force the light theme. Do not build a dark mode for this round; one ground is the rule.

```css
:root{ --paper:#F2F3F0; --ink:#1A2230; --accent:#2C5B8C; }
```

`.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#2C5B8C"
backgroundColor = "#F2F3F0"
secondaryBackgroundColor = "#F2F3F0"   # same as paper, so widgets get no tinted fill
textColor = "#1A2230"
font = "serif"

[client]
toolbarMode = "minimal"
showSidebarNavigation = false
showErrorDetails = false
```

## 3. Type

### 3.1 Family and figures

One family: **EB Garamond**, roman only, weights 400 and 500. Load from Google Fonts with the 400 and 500 roman weights only:

```html
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500&display=swap" rel="stylesheet">
```

Why Garamond still works for numbers: its lining figures are elegant at display size, and it ships tabular figures (`tnum`), so columns align. Its weakness is a small x-height, so nothing data-bearing goes below 14px.

Figure strategy:

| Where | Setting |
|---|---|
| All data: numerals in lists, tables, the hero, bars, axis-free labels | `font-variant-numeric: lining-nums tabular-nums;` |
| Numbers inside running sentences ("about 18 cm by Saturday") | `font-variant-numeric: oldstyle-nums proportional-nums;` |
| Hero numeral (136px and up) | `lining-nums proportional-nums` (tabular spacing looks gappy at that size) |

Put `font-variant-numeric: lining-nums tabular-nums` on `.data` and on every `td`. Weight 500 is allowed only for numerals at 18px and below, so small data reads as firmly as the big numbers do. Everything 27px and up is weight 400; big type gets its weight from size.

Verification step before building anything else: render `1111 / 8888` in a `.data` span and confirm both are the same width. If Google's EB Garamond build does not apply `tnum` in the browser, switch the numerals (only the numerals) to **Source Serif 4** with the same `font-variant-numeric`, using a `.num` class. That is the one permitted second family. Do not mix the two by preference; it is a fallback for a missing feature.

No italics anywhere: `em,i{font-style:normal}`. No sans-serif in the visible page. Labels are small caps, not a sans.

### 3.2 Scale

Base 18px, ratio 1.5, plus one jump size for the hero number.

| Role | Desktop | Phone (700px and under) | 1921px and up | Line height | Tracking |
|---|---|---|---|---|---|
| Label (small caps), units after small numbers | 12 | 12 | 14 | 1.5 | 0.12em, small caps only |
| Caption, data table row text, source notes | 14 | 14 | 14 | 20px | 0 |
| Body, data rows, day list | 18 | 17 | 20 | 27px (26 phone) | 0 |
| Secondary numerals (day totals, elevation bands), section standfirst | 27 | 24 | 30 | 1.35 | 0 |
| Section heading ("The week", "Across the range") | 40 | 34 | 46 | 1.1 | -0.01em |
| Resort name (the picker) | 61 | 44 | 70 | 1.05 | -0.01em |
| Big moment: week total, best day | 91 | 56 | 112 | 0.95 | -0.02em |
| Hero numeral: next 72 hours of new snow | `clamp(120px, 14vw, 200px)` | `clamp(96px, 28vw, 136px)` | up to 240 | 0.9 | -0.03em |

Rules:

- No sizes outside this table. If something is not body and not a heading, it is a label.
- Each screen uses at most two sizes plus the label size, at least twice apart. The first screen uses the hero, the resort name, body and label.
- Units sit next to their number one or two steps smaller: `147` at 200px, `cm` at 40px, aligned to the numeral's baseline. Use a thin space (`&thinsp;`) between number and unit.
- Labels are `font-variant-caps: all-small-caps; letter-spacing: .12em`, at most four words, at most one label per group. No label above every heading ("kickers" are banned).
- Ragged right, never justified. `text-wrap: balance` on headings, `text-wrap: pretty` on sentences.
- Measure: sentences at most 24em. Data rows may run wider, because they are tables, not prose.

### 3.3 Writing numbers and dates

- Dates in words: "Saturday 29 November". Day lists may shorten to "Sat 29" in the label column only.
- Times: "Updated 06:00 MST, Thursday 27 November". One update line, at caption size, at the close and under the hero.
- Ranges with an en dash and no spaces: "10–15 cm". Real minus sign for temperatures: "−8°C".
- Round to what the model can actually claim: whole centimetres for snowfall and depth, whole percent for probability, metres to the nearest 10. No "12.3 cm".
- Never write a number for missing data. Write "No reading" or leave the cell empty with a label; never a dash, zero or estimate that could be read as a value.
- Plain words for headings: "Next three days", "The week", "By elevation", "Across the range", "On the ground" (SNOTEL), "All resorts". No emoji, no clever labels.

## 4. Space

Every vertical gap is a whole or half multiple of the 27px line (26px on phones, 30px at 1921 and up). Never write a raw pixel value for vertical space.

| Token | Lines | Desktop | Phone | Use |
|---|---|---|---|---|
| `--s0` | 0.5 | 13.5 | 13 | heading to its first line; number to its label |
| `--s1` | 1 | 27 | 26 | sentence to sentence; label row to data rows |
| `--s1h` | 1.5 | 40.5 | 39 | map to the text below it (phone) |
| `--s2` | 2 | 54 | 52 | standfirst to the list under it; last row to a closing link |
| `--s3` | 3 | 81 | 78 | between groups inside a section (between elevation bands, between the hero and the week) |
| `--s4` | 4 | 108 | 104 | top of the page to the resort name; close to the end of the page |
| `--s6` | 6 (5 on phone) | 162 | 130 | every section break |

```css
:root{
  --l:27px;
  --s0:calc(var(--l)*.5); --s1:var(--l); --s1h:calc(var(--l)*1.5);
  --s2:calc(var(--l)*2); --s3:calc(var(--l)*3); --s4:calc(var(--l)*4); --s6:calc(var(--l)*6);
  --m:clamp(24px,6vw,120px);          /* side margin */
  --hang:clamp(24px,11vw,200px);      /* the one left edge, about 1/9 of the width */
  --measure:24em;
}
@media (max-width:700px){ :root{ --l:26px; --s6:calc(var(--l)*5); --m:24px; --hang:24px } }
@media (min-width:1921px){ :root{ --l:30px } }
```

Data density rule (the adaptation that matters most): rows of data sit directly on the line grid, one row per 27px line, with no gap and no rule between rows. Space separates groups, not rows. A gap between groups is at least three times the largest gap inside a group: rows are 0 apart, groups `--s3` apart, sections `--s6` apart.

## 5. Layout

### 5.1 The page

- One scrolling page, no sidebar, no tabs. Everything a person needs for one resort is on one long page, read top to bottom as a sequence of screens.
- Content starts at `--hang` from the left. The right side runs to `--m` from the right edge, except the map, which bleeds off the screen.
- Maximum text block width is `min(1180px, 100vw - var(--hang) - var(--m))`. At 2560 wide, scale type and line unit up (table above) rather than leaving a small page in a big field.
- Every text element on the page shares one left x. The only second edge is the right edge of the numeral column in data rows.
- Splits are 4:8 or 5:7 on a 12-column grid with a 27px gutter. In Streamlit use `st.columns([5,7], gap="large")` or `[4,8]`, and better still build the split in one HTML block with CSS grid so the gutter is exact. Never `[1,1]`, never four or seven equal columns.
- The space below the main block on a screen is 1.5 to 2 times the space above it.
- Below 700px wide everything stacks into one column on the side margin, in reading order.

### 5.2 Screens, top to bottom

Write the page as screens, the way the library's ARCHETYPES does. Neighbouring screens never use the same move.

| # | Move (adapted archetype) | What is on it | The screen's one job |
|---|---|---|---|
| 1 | Cover with one huge number (archetype 6, "one huge word") | Top left: the resort name at 61px, which is the picker itself (see 8.3). Under it one caption line: region, base and summit elevation in words and figures. Then the hero numeral, new snow in the next 72 hours, at the jump size in accent with `cm` beside it. Under the hero one plain sentence in body size that says what it means ("Most of it falls Friday night above 2,000 m. Saturday morning is the day."). The 5 column on the right holds three small facts as a hanging list: current base depth, snow level, chance of any snow. | Answer "is it going to snow, and when" in two seconds. |
| 2 | Hanging list (archetype 9) in a 4:8 split | Left 4: "The week" at 40px, the week total at 91px, one sentence. Right 8: seven day rows, each one line: day label, text-weight bar, numeral, then probability and snow level at caption size. The best day's bar and numeral in accent. | Show the shape of the week. |
| 3 | Breath line (archetype 7) | One sentence at 27px, nothing else: the honest confidence statement ("Models agree through Saturday; after that the spread is wide."). 75 to 85% empty. | Say how much to trust it. |
| 4 | By elevation, hanging list | Three groups (summit, mid, base), each with its elevation as a label, the 72-hour numeral at 27px, a bar, and rain/snow line in a sentence. Groups `--s3` apart. | Show where on the mountain the snow lands. |
| 5 | Anchor (archetype 4) with the map | Left 5: "Across the range" at 40px and a ranked list of the ten snowiest resorts nearby, numerals right-aligned with bars. Right 7: the map, running off the right screen edge, fading into the paper on its left side. | Compare this resort with its neighbours. |
| 6 | On the ground (SNOTEL) | Nearest stations as rows: name, distance, elevation, observed depth, 24-hour change, a 14-day depth sparkline. Observed values only; missing readings say so. | Check the forecast against real measurements. |
| 7 | All resorts | Full ranked table as HTML, sortable by plain-word links ("Sort by new snow / base / name"). Rows on the line grid, no rules. | Find any other resort. |
| 8 | Close | Caption-size block on the left edge: sources in plain words (NOAA HRRR, Copernicus DEM, SNOTEL), model update time, "Refresh" as an underlined word, units switch "cm / in" as two plain words. Ends `--s4` above the page foot. | End the page the same way every time. |

On phones the same order holds; the map drops to full width, bleeding off both sides only if it fades top and bottom, otherwise 100% of the content width with its fade on the top edge.

## 6. Data under these rules

### 6.1 Numerals

- The hero numeral is the only thing at the jump size. It is accent.
- Accent numerals: the hero, the week's best day, the selected resort's row in ranked lists. Everything else is ink.
- Secondary numbers step down the scale: 91 (week total), 27 (day and elevation numbers), 18 (rows), 14 (captions).
- A number is never alone without its unit or label. Units in small caps or one size smaller, baseline-aligned.
- Change values carry a sign and say direction in words when it matters: "+12 cm in 24 hours", "−4 cm since Monday (settling)".

### 6.2 Text-weight bars

The bar is a typographic mark, not a control. It sits in a row of text like a long dash.

- Height: `0.18em` of the row's font size (about 3px at 18px, 5px at 27px). Vertically centred on the x-height.
- Colour: ink, or accent for the one highlighted row. No track, no background behind the bar, no rounded ends (`border-radius:0`), no gradient, no shadow.
- Length: proportional to the value against a fixed maximum for that list (state it once in the label, "bars to 50 cm"), so bars compare across rows. Minimum visible length 2px for any non-zero value; zero is no bar at all.
- The numeral sits right after the bar's end column, right-aligned in a fixed numeral column, so numbers line up even when bars differ.
- Build it as an inline `<span class="bar" style="--v:.42">` with `width:calc(var(--v) * 10em)`. Never with `st.progress`.

```css
.bar{display:inline-block;height:.18em;width:calc(var(--v)*10em);background:var(--ink);vertical-align:.32em;border-radius:0}
.bar.is-best{background:var(--accent)}
```

### 6.3 Sparklines and small charts

- Inline SVG, drawn in the same line as the text, height 1em to 1.5em of the row, width 6em to 8em.
- Snow depth over time: one polyline, `stroke-width:1.25`, ink, `fill:none`, `shape-rendering:geometricPrecision`, `vector-effect:non-scaling-stroke`. One accent dot (r=2) on the latest value only. No axis, no gridlines, no area fill, no markers on every point.
- Daily snowfall over time: vertical bars, 3px wide with 2px gaps, ink, best day in accent, baseline at the bottom of the text line.
- Every sparkline sits beside a numeral that states its latest value, so the line never has to carry the number.
- If a larger chart is truly needed (hourly accumulation in the detail view), use Plotly with `paper_bgcolor` and `plot_bgcolor` set to `rgba(0,0,0,0)`, `showgrid:false`, `zeroline:false`, axis lines off, tick labels in EB Garamond 14px ink with tabular figures, `displayModeBar:false`, and the chart bleeding to the right screen edge like the map. Do not use `st.line_chart` or `st.bar_chart` (their chrome cannot be fully removed).

### 6.4 The line budget

The library allows at most one visible line per screen. Here:

- The one structural line is the resort picker's 1px underline on screen 1, and the 1px underline under text links.
- Data marks (bars, sparklines) are data, not lines, and do not count. They must never be used to separate things: no bar or line as a divider, no rule under a table header, no rule between rows, no border around the map.
- A table header row is small-caps labels in ink, separated from the rows by `--s0` of space, nothing else.

### 6.5 Tables

- Built as HTML `<table>` via `st.markdown`, never `st.dataframe` (it renders on a canvas with its own grid lines, header fill and toolbar, and cannot be restyled).
- `border-collapse:collapse; border:0`, every cell `border:0; padding:0 1em 0 0; height:var(--l)`. Text columns left-aligned on the left edge; numeral columns right-aligned with tabular figures.
- No zebra striping, no hover fill, no header background.
- The selected resort's row is in accent. That is the only highlight.
- Long lists (all resorts) group by region or state with a 27px region name above each group and `--s3` between groups, so the eye has places to rest without rules.

### 6.6 The map

- The map is the only "picture" on the page and follows the library's floating-art rules: no border, no rounded corners, no shadow, no legend box.
- It runs off the right screen edge (a negative right margin equal to `--m`), and its left edge fades into the paper with a long multi-stop mask: `mask-image: linear-gradient(to right, transparent 0, rgba(0,0,0,.15) 6%, rgba(0,0,0,.6) 14%, #000 24%)`. Its top and bottom fade the same way over the outer 10%.
- Basemap: Carto Positron without labels (`light_nolabels`), so its land is close to the paper. Use pydeck with `map_style` set to that URL; folium only if pydeck is not available, with the same tiles.
- Points: ink circles, radius scaled by 72-hour new snow (square-root scale so area tracks the amount), the selected resort in accent. No colour ramp, no heatmap.
- Labels on the map: at most the five snowiest resort names, in EB Garamond 14px ink, set beside their points. Tooltips styled as paper with 14px ink text and a 1px ink outline, no shadow.
- Hide the map's own controls (zoom buttons, attribution box styled as a pill); put the tile attribution as plain caption text in the close.

### 6.7 Probability and confidence

- Probability is a number with a percent sign ("80%"), never a gauge, dial, coloured badge or pill.
- Confidence is a sentence (screen 3) plus, in day rows, a range in caption size after the main numeral ("12 cm, 6–20").
- Snow quality (powder, wet, wind-affected) is a word in the row, never an icon or a coloured chip.

## 7. Controls

The library's rule: plain words, one button style, nothing that looks like an app.

- Links and secondary actions: plain words with a 1px underline, `text-underline-offset:.2em`, at least 44px tap height via padding.
- If a button is unavoidable (Refresh), it is the library's one button: 1px accent outline, square corners, paper fill, 18px roman label, 48px tall. One per page at most. Prefer an underlined word.
- Favourites: the word "Save" beside the resort name, underlined; when saved it reads "Saved". No star icon, no filled toggle.
- Units: "cm / in" as two words; the current one is ink with an underline, the other plain ink.
- Focus: always visible, `outline:2px solid var(--accent); outline-offset:3px`.
- No `st.toggle`, `st.checkbox`, `st.radio` pills, `st.slider`, `st.metric`, `st.expander`, `st.progress`, `st.status`, `st.toast`, `st.info/warning/success/error` (all of them draw boxes, fills or icons). Status and errors are one sentence in ink.

## 8. Restyling Streamlit

The strategy: let Streamlit run the page and hold state, and draw almost everything visible as HTML through `st.markdown(..., unsafe_allow_html=True)`. Keep only three Streamlit widgets: `st.selectbox` (the resort picker), `st.pydeck_chart` (the map) and, if needed, `st.button` (refresh). Hide every other piece of chrome.

### 8.1 Injection

Put the whole stylesheet in one module, for example `src/snowforecast/dashboard/theme.py`, with one function `apply_theme()` that calls `st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title=..., page_icon=None)` first and then one `st.markdown("<style>...</style>", unsafe_allow_html=True)` with the font link. Every render helper returns an HTML string; the app calls `st.markdown(html, unsafe_allow_html=True)` once per screen, so each screen is one block whose internal spacing we fully control. Escape every data string with `html.escape` before putting it in markup.

### 8.2 Hide and flatten the chrome

Streamlit's test ids are the stable selectors (checked against 1.33 to 1.40; recheck after upgrades).

```css
/* chrome */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]
{ display:none !important; }

/* the ground */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]
{ background:var(--paper) !important; color:var(--ink); }
html{ overflow-x:clip; }

/* remove default padding; our blocks own the spacing */
.block-container, [data-testid="stMainBlockContainer"]
{ padding:0 var(--m) 0 var(--hang) !important; max-width:none !important; }
[data-testid="stVerticalBlock"]{ gap:0 !important; }
[data-testid="stHorizontalBlock"]{ gap:var(--l) !important; }
[data-testid="stMarkdownContainer"] p{ margin:0; }
[data-testid="stElementToolbar"]{ display:none !important; }   /* hover toolbars on charts */

/* type everywhere */
.stApp, .stApp *{ font-family:"EB Garamond", Garamond, "Times New Roman", serif !important; }
body{ font-size:18px; line-height:var(--l); -webkit-font-smoothing:antialiased; font-kerning:normal; }
em,i{ font-style:normal; }
.data, td{ font-variant-numeric:lining-nums tabular-nums; }
::selection{ background:var(--accent); color:var(--paper); }
```

Also hide Streamlit's running indicator flash and the "Made with Streamlit" footer (covered above), and set `[data-testid="stSpinner"]` text to caption size ink with no icon if a spinner is used at all. Prefer `st.cache_data` with no visible spinner (`show_spinner=False`).

### 8.3 The resort picker (the one kept widget, styled as the page title)

The selectbox becomes the resort name at 61px with a single 1px ink underline. That underline is the screen's one rule. Its label is hidden (`label_visibility="collapsed"`) and a 12px small-caps label "Resort" sits above it in our own HTML.

```css
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:transparent !important; border:0 !important; border-bottom:1px solid var(--ink) !important;
  border-radius:0 !important; box-shadow:none !important; padding:0 !important; min-height:0 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *{
  font-size:61px !important; line-height:1.05 !important; letter-spacing:-.01em; color:var(--ink) !important;
}
[data-testid="stSelectbox"] svg{ width:24px; height:24px; color:var(--ink); }   /* keep the caret, small */
@media (max-width:700px){ [data-testid="stSelectbox"] [data-baseweb="select"] *{ font-size:44px !important } }

/* the open list */
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"]{
  background:var(--paper) !important; border:1px solid var(--ink) !important; border-radius:0 !important; box-shadow:none !important;
}
[role="option"]{ font-size:18px !important; line-height:var(--l) !important; color:var(--ink) !important; background:transparent !important; }
[role="option"][aria-selected="true"], [role="option"]:hover{ color:var(--accent) !important; background:transparent !important; }
```

The open popover is temporary, so its outline does not count against the screen's line budget. It has no fill other than paper and no shadow. Make the list searchable (Streamlit's selectbox already filters on typing) and group by state or country in the option labels ("Alta, Utah") rather than with a second state selectbox.

### 8.4 Other kept pieces

- `st.pydeck_chart`: wrap in `<div class="map-bleed">` is not possible around a widget, so target its container: `[data-testid="stDeckGlJsonChart"]{ margin-right:calc(var(--m) * -1); mask-image:...; }` with the mask from 6.6. Set `use_container_width=True` and a height of 12 lines (324px) on desktop, 10 lines on phones.
- `st.button`, if kept: `.stButton > button{ background:var(--paper); border:1px solid var(--accent); border-radius:0; color:var(--ink); height:48px; font-size:18px; box-shadow:none; padding:0 var(--s1); }` and the same on hover with the label in accent. No filled hover.
- `st.columns`: only for placing the map beside an HTML block. Always `[5,7]` or `[4,8]`.
- `st.tabs`: not used. If a later round brings them back: hide `[data-baseweb="tab-highlight"]` and `[data-baseweb="tab-border"]`, tab labels 18px ink, the current tab with a 1px underline only.
- Sort links and the units switch use `st.query_params` and plain `<a href="?sort=new">` anchors in our HTML, so they stay words, not widgets.

### 8.5 Theme file and favicon

Set `.streamlit/config.toml` as in section 2. `page_icon=None` (no snowflake emoji). Page title: "Snow forecast, <resort name>".

## 9. Copy

- Lead with a sentence that answers the question. "18 cm by Saturday morning, most of it above 2,000 m." beats any tile.
- Short, plain headings, listed in 3.3.
- No emoji anywhere: not in headings, tabs, buttons, captions or the page title.
- Debug information (database path, cache counts) never appears on the page. Put it behind `?debug=1` at caption size in the close, or in logs.
- When data is missing, say so in a plain sentence: "No SNOTEL station within 30 km." Never fill the gap with an estimate.

## 10. Responsive

- Check at 390x844, 768x1024, 1280x800, 1440x900, 1920x1080 and 2560x1440.
- Phones: one column on a 24px margin, body 17/26, hero at `clamp(96px,28vw,136px)`, resort name 44px. Day rows keep bar and numeral on one line; probability and snow level wrap under them at caption size.
- No sideways scrolling at any width except inside a wide table, which sits in an `overflow-x:auto` wrapper with no visible scrollbar styling beyond the browser default.
- Tap targets at least 44px.

## 11. Never

Adapted from the library's Never list. Any one of these fails the round.

1. Italics, for any reason, and emoji anywhere on the page or in its title.
2. A third font family, a sans-serif in the visible page, or any font but EB Garamond (Source Serif 4 numerals only as the documented `tnum` fallback).
3. Text in more than one ink plus one accent, a grey or faded "secondary" text colour, or the accent on headings.
4. Boxes of any kind: cards, metric tiles (`st.metric`), expanders, info/warning/success boxes, bordered containers, rounded panels, translucent overlays.
5. Section backgrounds, a sidebar of a different tint, header or footer bars, bands, zebra rows or table header fills. One ground, flat, everywhere.
6. More than one visible rule per screen: no dividers (`st.markdown("---")`, `st.divider`), no rules between table rows or under headers, no borders around the map or charts.
7. Flat app controls: progress bars, gauges, glossy pills, badges, coloured status chips, filled toggles, checkboxes, sliders, drop shadows.
8. Equal splits: `st.columns(2)` at 1:1, rows of three, four or seven equal tiles or day cards. Use 5:7 or 4:8 and vertical hanging lists.
9. Streamlit's default chrome left showing: the header, toolbar, hamburger menu, "Made with Streamlit" footer, `st.dataframe` grids, `st.line_chart` axes and gridlines, chart toolbars.
10. Made-up numbers: estimated or filled-in values where data is missing, false precision (decimals the model cannot support), or a colour ramp standing in for a number the reader cannot read.
11. Provenance captions under statistics (user directive, 27 Nov 2026): never print a data source, station name, provider or observation date beneath a number ("OnTheSnow (mid-mountain) · 26 Nov" was rejected as clutter). A stat is its label and its number, nothing under it. Attribution lives only in the single sources line at the foot of the page; where a depth is measured (base vs mid-mountain) belongs in the stat's own ≤4-word label if it matters. The only exception is a staleness warning on genuinely old data, and that is a warning, not a citation.
12. Caption fragment chains and symbol legends (user directive, 27 Nov 2026): never string sentence fragments into a caption ("86 resorts. Snow in cm at each summit. — no report." was rejected as AI slop), and never print a legend explaining a symbol ("— no report", "* estimate"). The row count is visible in the table, units live in the cm/in toggle, and a dash explains itself. A table caption is at most one ≤4-word phrase that adds a fact the headers don't carry ("Summit snow."), or nothing.

## 12. Before showing anyone

- Screenshot every screen at 1440x900 and 390x844 and look at them, not just the code.
- List every computed font size; each is in the scale table.
- List every computed text colour; there are two.
- Count visible lines per screen; at most one, and it is the picker or a link underline.
- List the left x of every text element at 1440; at most two values (text edge and numeral edge).
- Every vertical gap between blocks is one of 0, 14, 27, 41, 54, 81, 108, 162 (phone 0, 13, 26, 39, 52, 78, 104, 130), allowing 2px for rounding.
- Search the rendered page for `st-emotion` boxes with borders or fills, `font-style: italic`, emoji, and any grey text.
- Confirm tabular figures: a column of numbers is aligned on its right edge and digits do not jitter when values change.
