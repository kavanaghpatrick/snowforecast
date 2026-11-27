"""Theme for the v2 dashboard: one ground, one ink, one accent, EB Garamond.

Tabular-figure check (spec 4.4), run in Chromium with Playwright against
Streamlit 1.52.2 on 27 November 2026: "1111" and "8888" in a .data span
measured the same width, so EB Garamond's tnum works and no Source Serif 4
fallback is loaded.
"""

from __future__ import annotations

import streamlit as st

FONT_IMPORT = "https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500&display=swap"

CSS = r"""
@import url('%(font)s');
:root{
  --paper:#F2F3F0; --ink:#1A2230; --accent:#2C5B8C;
  --l:27px;
  --s0:calc(var(--l)*.5); --s1:var(--l); --s1h:calc(var(--l)*1.5);
  --s2:calc(var(--l)*2); --s3:calc(var(--l)*3); --s4:calc(var(--l)*4); --s6:calc(var(--l)*6);
  --m:clamp(24px,6vw,120px);
  --hang:clamp(24px,11vw,200px);
  --measure:24em;
}
@media (max-width:700px){ :root{ --l:26px; --s6:calc(var(--l)*5); --m:24px; --hang:24px } }
@media (min-width:1921px){ :root{ --l:30px } }

/* chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stElementToolbar"], [data-testid="stHeaderActionElements"], [data-testid="stAppDeployButton"]
{ display:none !important; }

/* ground */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] > div
{ background:var(--paper) !important; color:var(--ink); }
html{ overflow-x:clip; }
.block-container, [data-testid="stMainBlockContainer"]
{ padding:0 var(--m) 0 var(--hang) !important; max-width:none !important; }
[data-testid="stVerticalBlock"]{ gap:0 !important; }
[data-testid="stHorizontalBlock"]{ gap:var(--l) !important; }
[data-testid="stMarkdownContainer"] p{ margin:0; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdown"]{ color:var(--ink); }
/* Streamlit gives every markdown block margin-bottom:-1rem, which shaved 16px off every
   section break (146 instead of 162). Our blocks own their spacing. */
[data-testid="stMarkdownContainer"]{ margin-bottom:0 !important; }
.stElementContainer, [data-testid="stElementContainer"]{ margin:0 !important; }

/* type */
.stApp, .stApp *{ font-family:"EB Garamond", Garamond, "Times New Roman", serif !important; }
body, .stApp{ font-size:18px; line-height:var(--l); -webkit-font-smoothing:antialiased; font-kerning:normal; }
[data-testid="stMarkdownContainer"]{ font-size:18px; line-height:var(--l); }
em, i{ font-style:normal; }
b, strong{ font-weight:500; }
.data, td, .num{ font-variant-numeric:lining-nums tabular-nums; }
.prose .num{ font-variant-numeric:oldstyle-nums proportional-nums; }
::selection{ background:var(--accent); color:var(--paper); }
.page{ max-width:min(1180px, calc(100vw - var(--hang) - var(--m))); }
.screen{ padding-top:var(--s6); }
.screen.top{ padding-top:var(--s4); }
.screen.top .label{ margin-bottom:var(--s0); }
.screen.after-breath{ padding-top:0; }
.label{ font-size:12px; line-height:1.5; font-variant-caps:all-small-caps; letter-spacing:.12em; font-weight:400; }
.caption{ font-size:14px; line-height:20px; }
.body{ font-size:18px; line-height:var(--l); }
.n27{ font-size:27px; line-height:1.35; }
.h40, [data-testid="stMarkdownContainer"] h2.h40{ font-size:40px; line-height:1.1; letter-spacing:-.01em; font-weight:400;
  margin:0; padding:0; text-wrap:balance; color:var(--ink); }
.big{ font-size:91px; line-height:.95; letter-spacing:-.02em; font-variant-numeric:lining-nums proportional-nums; }
.herorow{ display:flex; align-items:baseline; gap:54px; flex-wrap:wrap; }
.herostat{ display:flex; flex-direction:column; }
.herodepth .big{ color:var(--ink); }
.hero{ font-size:clamp(120px,14vw,200px); line-height:.9; letter-spacing:-.03em; color:var(--accent);
       font-variant-numeric:lining-nums proportional-nums; }
.hero .unit{ font-size:40px; letter-spacing:0; }
.big .unit{ font-size:40px; letter-spacing:0; }
.n27 .unit{ font-size:18px; }
.accent{ color:var(--accent); }
.sentence{ max-width:var(--measure); text-wrap:pretty; }
.breath{ font-size:27px; line-height:1.35; max-width:var(--measure); padding-block:var(--s6); text-wrap:pretty; }
/* .stApp prefix: beats Streamlit's "[data-testid=stMarkdownContainer] p{margin:0}" so gaps on <p> hold */
.stApp .gap0{ margin-top:var(--s0); } .stApp .gap1{ margin-top:var(--s1); } .stApp .gap2{ margin-top:var(--s2); } .stApp .gap3{ margin-top:var(--s3); }
.nowrap{ white-space:nowrap; }

/* type classes win over Streamlit's own markdown rules */
.stApp .label{ font-size:12px; line-height:1.5; }
.stApp .caption{ font-size:14px; line-height:20px; }
.stApp .body{ font-size:18px; line-height:var(--l); }
.stApp .n27{ font-size:27px; line-height:1.35; }
.stApp .breath{ font-size:27px; line-height:1.35; }
.stApp .rows .c{ font-size:14px; }
@media (min-width:1921px){ .stApp .body{ font-size:20px; } .stApp .n27, .stApp .breath{ font-size:30px; } .stApp .label{ font-size:14px; } }
@media (max-width:700px){ .stApp .body{ font-size:17px; line-height:var(--l); } .stApp .n27, .stApp .breath{ font-size:24px; } }

/* splits */
.split-5-7{ display:grid; grid-template-columns:7fr 5fr; column-gap:var(--l); }
.split-4-8{ display:grid; grid-template-columns:4fr 8fr; column-gap:var(--l); }
@media (max-width:700px){ .split-5-7, .split-4-8{ grid-template-columns:1fr; row-gap:var(--s2); } }
/* tablets: the 8 column is too narrow for the day rows' captions (they wrapped
   one word per line at 768), so the week stacks above its day list */
@media (min-width:701px) and (max-width:1000px){ .split-4-8{ grid-template-columns:1fr; row-gap:var(--s2); } }

/* rows on the line grid */
.rows{ display:grid; grid-template-columns:max-content 10em 4ch 1fr; column-gap:1em; align-items:baseline; }
.rows > *{ line-height:var(--l); min-height:var(--l); }
.rows .n{ text-align:right; font-variant-numeric:lining-nums tabular-nums; font-weight:500; white-space:nowrap; }
.rows .c{ font-size:14px; }
.rows .bc{ white-space:nowrap; }
.rows > *:empty::before{ content:"\200b"; }
.rows .c.nowrap{ white-space:nowrap; }
.group + .group{ margin-top:var(--s3); }
/* day lists: fixed label column so the week and days 8-15 share one numeral edge */
.rows.days{ grid-template-columns:4em 10em 4ch 1fr; }
/* elevation groups: bar in a fixed 10em slot so the three numerals share one left edge */
.bc4{ display:inline-block; width:10em; margin-right:.25em; }
.n4{ display:inline-block; min-width:2ch; text-align:right; }
@media (max-width:700px){
  .rows, .rows.days{ grid-template-columns:max-content minmax(0,1fr) 4ch; }
  .rows .c{ grid-column:1 / -1; line-height:20px; min-height:0; margin-bottom:6px; }
  .rows .c:empty{ display:none; }
  .rows .bc{ overflow:hidden; }
  .rows .bar{ max-width:100%%; }
}

/* bars and sparklines */
.bar{ display:inline-block; height:.18em; width:calc(var(--v)*10em); background:var(--ink); vertical-align:.32em; border-radius:0; }
.bar.is-best{ background:var(--accent); }
.spark{ height:1.2em; width:7em; vertical-align:-.2em; overflow:visible; }
.spark.week{ width:3.5em; height:1em; vertical-align:-.1em; }
.spark polyline{ fill:none; stroke:var(--ink); stroke-width:1.25; vector-effect:non-scaling-stroke; shape-rendering:geometricPrecision; }
.spark rect{ fill:var(--ink); } .spark rect.is-best{ fill:var(--accent); } .spark circle{ fill:var(--accent); }

/* tables */
.tablewrap{ overflow:auto clip; }
[data-testid="stMarkdownContainer"] table.t{ border-collapse:collapse; border:0 !important; font-size:14px; margin:0; width:auto; }
[data-testid="stMarkdownContainer"] table.t tr{ border:0 !important; background:transparent !important; }
[data-testid="stMarkdownContainer"] table.t th, [data-testid="stMarkdownContainer"] table.t td{
  border:0 !important; padding:0 1em 0 0; height:var(--l); line-height:var(--l); background:transparent !important;
  text-align:left; font-weight:400; color:var(--ink); white-space:nowrap; vertical-align:baseline; font-size:14px; }
[data-testid="stMarkdownContainer"] table.t th{ font-size:12px; font-variant-caps:all-small-caps; letter-spacing:.12em; padding-bottom:var(--s0); }
[data-testid="stMarkdownContainer"] table.t td.r, [data-testid="stMarkdownContainer"] table.t th.r{ text-align:right; }
[data-testid="stMarkdownContainer"] table.t td.first{ font-size:18px; padding-right:2em; }
[data-testid="stMarkdownContainer"] table.t tr.is-selected td{ color:var(--accent); }
[data-testid="stMarkdownContainer"] table.t tr.grp td{ font-size:27px; line-height:1.35; padding-top:var(--s3); padding-bottom:var(--s0); height:auto; }
[data-testid="stMarkdownContainer"] table.t tr.grp.first td{ padding-top:0; }
[data-testid="stMarkdownContainer"] table.t .label{ font-size:12px; }

/* links */
.stApp a, .stApp a:visited{ color:inherit !important; text-decoration:underline; text-decoration-thickness:1px; text-underline-offset:.2em; }
.stApp a.plain{ text-decoration:none; }
.words a{ display:inline-block; padding-block:10px; }
/* 44px tap targets for inline links without moving the line grid: vertical padding on an
   inline box enlarges its hit area but not the line box */
.stApp a:not(.words a){ padding-block:11px; }
.stApp .caption a:not(.words a){ padding-block:13px; }
.stApp .words a.is-current{ text-decoration:underline; } .stApp .words a:not(.is-current){ text-decoration:none; }
:focus-visible{ outline:2px solid var(--accent) !important; outline-offset:3px; }

/* the picker */
[data-testid="stSelectbox"]{ max-width:min(100%%, 820px); margin-top:var(--s1); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:transparent !important; border:0 !important; border-bottom:1px solid var(--ink) !important;
  border-radius:0 !important; box-shadow:none !important; padding:0 !important; min-height:0 !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] *{
  font-size:61px !important; line-height:1.05 !important; letter-spacing:-.01em; color:var(--ink) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] input{ caret-color:var(--accent); }
/* focus on the picker shows as its underline turning accent, not a box round the text input */
[data-testid="stSelectbox"] input:focus, [data-testid="stSelectbox"] input:focus-visible{ outline:none !important; box-shadow:none !important; }
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div{ border-bottom-color:var(--accent) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] div{ height:auto !important; max-height:none !important; overflow:visible !important;
  white-space:normal !important; text-overflow:clip !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child{ padding:0 0 8px 0 !important; }
[data-testid="stSelectbox"] svg{ width:24px !important; height:24px !important; color:var(--ink); }
@media (max-width:700px){ [data-testid="stSelectbox"] [data-baseweb="select"] *{ font-size:44px !important } }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"]{
  background:var(--paper) !important; border-radius:0 !important; box-shadow:none !important; }
[data-baseweb="popover"] > div{ border:1px solid var(--ink) !important; }
[data-baseweb="popover"] ul[role="listbox"]{ border:0 !important; }
[role="option"]{ font-size:18px !important; line-height:var(--l) !important; color:var(--ink) !important; background:transparent !important; }
[role="option"] *{ font-size:18px !important; color:inherit !important; }
[role="option"][aria-selected="true"], [role="option"]:hover{ color:var(--accent) !important; background:transparent !important; }

/* the map */
[data-testid="stDeckGlJsonChart"]{
  margin-right:calc(var(--m) * -1);
  width:calc(100%% + var(--m)) !important;
  -webkit-mask-image:linear-gradient(to right, transparent 0, rgba(0,0,0,.15) 6%%, rgba(0,0,0,.6) 14%%, #000 24%%),
                     linear-gradient(to bottom, transparent 0, #000 10%%, #000 90%%, transparent 100%%);
  -webkit-mask-composite:source-in;
  mask-image:linear-gradient(to right, transparent 0, rgba(0,0,0,.15) 6%%, rgba(0,0,0,.6) 14%%, #000 24%%),
             linear-gradient(to bottom, transparent 0, #000 10%%, #000 90%%, transparent 100%%);
  mask-composite:intersect;
  border:0 !important; border-radius:0 !important; box-shadow:none !important; }
[data-testid="stDeckGlJsonChart"] *{ border-radius:0 !important; }
.deck-tooltip{ background:var(--paper) !important; color:var(--ink) !important; border:1px solid var(--ink) !important;
  border-radius:0 !important; box-shadow:none !important; font-size:14px !important; }
.mapboxgl-ctrl, .maplibregl-ctrl, .mapboxgl-ctrl-attrib, .maplibregl-ctrl-attrib,
.mapboxgl-ctrl-bottom-right, .maplibregl-ctrl-bottom-right, .mapboxgl-ctrl-bottom-left, .maplibregl-ctrl-bottom-left
{ display:none !important; }
.map-pad{ padding-top:var(--s6); }
.facts-pad{ height:calc(var(--s4) + var(--s6) + var(--s2)); }  /* facts start level with the hero numeral */
/* phone: stacked columns already have a one-line gap between them, so pad one more line for --s2 */
@media (max-width:700px){ .facts-pad{ height:var(--s1); } .map-pad{ padding-top:var(--s1); } }

/* the one button */
.stButton > button, [data-testid="stBaseButton-secondary"]{ background:var(--paper) !important; border:1px solid var(--accent) !important; border-radius:0 !important;
  color:var(--ink) !important; height:48px; min-height:48px; font-size:18px !important; box-shadow:none !important; padding:0 var(--s1) !important; }
.stButton > button p{ font-size:18px !important; }
.stButton > button:hover, .stButton > button:focus, .stButton > button:active{ color:var(--accent) !important; background:var(--paper) !important; border-color:var(--accent) !important; }
.stButton{ margin-top:var(--s1); }

.close-end{ height:var(--s4); }

/* large screens */
@media (min-width:1921px){
  body, .stApp, .body, [data-testid="stMarkdownContainer"]{ font-size:20px; } .n27{ font-size:30px; } .h40, [data-testid="stMarkdownContainer"] h2.h40{ font-size:46px; } .big{ font-size:112px; }
  .hero{ font-size:clamp(160px,12vw,240px); } .label{ font-size:14px; }
  [data-testid="stMarkdownContainer"] table.t th, [data-testid="stMarkdownContainer"] table.t .label{ font-size:14px; }
  [data-testid="stSelectbox"] [data-baseweb="select"] *{ font-size:70px !important; }
}
@media (max-width:700px){
  body, .stApp, .body, [data-testid="stMarkdownContainer"]{ font-size:17px; } .n27{ font-size:24px; } .h40, [data-testid="stMarkdownContainer"] h2.h40{ font-size:34px; } .big{ font-size:56px; }
  .hero{ font-size:96px; }  /* capped so verdict to ground line fit the first phone screen */
  .herorow{ gap:26px; }
  .hero .unit, .big .unit{ font-size:34px; }
}
""" % {"font": FONT_IMPORT}


def apply_theme(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
