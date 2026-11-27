"""State-aware paintings at the page's breathing points (design law, design_assets/web/motifs.html).

The images (lossless WebP, paper-exact #F2F3F0) sit on the page ground, so one placed in
whitespace or bleeding off a viewport edge reads as paint on the page, never
as a pasted rectangle. Every crop box edge lands on paper (the ASSETS table
below holds each painting's measured subject box; _fit computes the crop), or
is dissolved with a mask where a painting's own cable or track runs to its
frame edge. Every image is aria-hidden with an empty alt, and every positioned
piece is pointer-events none with no in-flow height, so the data layout never
shifts. The in-flow exceptions sit below all data: the dump dissolve at the
very foot, and the small foot object.

The resort page is a run of screenfuls: verdict/hero, snow by day, compare,
stations/sources, foot. One painted subject per screenful, never two in a row
on the same side, heavy then light (ARCHETYPES rhythm):

  state        screen 1 (heavy)   days gap (light)  compare  stations   foot
  storm        pool, right EDGE   pool, small left  clean    right      left
  dry          pool, right EDGE   pool, small left  clean    right      left
  bluebird     pool, right EDGE   pool, small left  clean    none       left
  stale        pool, LEFT         pool, right       clean    none       left
  dump         none               none              clean    none       dissolve
  (none)       none               pool, left        clean    right      left

Each slot draws from its own themed pool, picked by a stable hash of the
resort id, so the same resort always shows the same painting for a given
state, and neighbours differ. Japanese resorts swap in the Japan set for
storm and dry. The powder page keeps the storm glass right of the title and
may add one small companion beside the measured-snow group; the all-resorts
page stays art-free. Phones show only the dump dissolve and a small hare.
"""

from __future__ import annotations

import zlib

from data import g, is_num, next3, week_total
from render import age_hours

ART = "app/static/art"

DUMP_CM = 40
STORM_CM = 20
DRY_WEEK_CM = 5
STALE_H = 36


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def _bluebird_next3(resort):
    for day in resort.get("days") or []:
        if isinstance(day, dict) and isinstance(day.get("index"), int) and 0 <= day["index"] <= 2:
            if g(day, "derived", "bluebird") is True:
                return True
    return False


def state(resort, updated):
    """Which family of paintings this resort's page gets today, or None."""
    age = age_hours(updated)
    if age is not None and age > STALE_H:
        return "stale"
    n3 = next3(resort)
    if is_num(n3) and n3 >= DUMP_CM:
        return "dump"
    if is_num(n3) and n3 >= STORM_CM:
        return "storm"
    if _bluebird_next3(resort):
        return "bluebird"
    wk = week_total(resort)
    if is_num(wk) and wk < DRY_WEEK_CM:
        return "dry"
    return None


# --------------------------------------------------------------------------
# Catalog: exported size and measured subject box (x0, y0, x1, y1 fractions)
# --------------------------------------------------------------------------

ASSETS = {
    "g-alpenglow": (800, 1200, 0.021, 0.300, 0.975, 0.675),
    "g-bathtub": (1200, 800, 0.058, 0.195, 0.936, 0.809),
    "g-boots": (1200, 800, 0.148, 0.191, 0.875, 0.829),
    "g-cornice": (1200, 800, 0.029, 0.256, 0.968, 0.780),
    "g-daychair": (1200, 800, 0.000, 0.061, 0.999, 0.936),
    "g-drawer": (1024, 1024, 0.110, 0.259, 0.885, 0.781),
    "g-fox": (1200, 800, 0.181, 0.135, 0.799, 0.858),
    "g-funicular": (800, 1200, 0.031, 0.000, 0.962, 0.999),
    "g-gondola": (1200, 800, 0.000, 0.141, 0.999, 0.808),
    "g-icicles": (1200, 800, 0.117, 0.105, 0.866, 0.838),
    "g-mirror": (1200, 800, 0.205, 0.190, 0.803, 0.840),
    "g-monkey": (1200, 800, 0.134, 0.170, 0.845, 0.849),
    "g-origami": (1200, 800, 0.213, 0.280, 0.790, 0.749),
    "g-owl": (1200, 800, 0.000, 0.180, 0.999, 0.999),
    "g-pistebasher": (1200, 800, 0.180, 0.269, 0.821, 0.755),
    "g-probe": (800, 1200, 0.212, 0.066, 0.799, 0.911),
    "g-pylon": (1200, 800, 0.162, 0.281, 0.833, 0.739),
    "g-rope": (800, 1200, 0.246, 0.030, 0.739, 0.978),
    "g-skintrack": (1200, 800, 0.023, 0.175, 0.977, 0.939),
    "g-skis": (800, 1200, 0.098, 0.032, 0.885, 0.965),
    "g-snowshoes": (800, 1200, 0.118, 0.072, 0.880, 0.932),
    "g-stbernard": (800, 1200, 0.151, 0.235, 0.760, 0.805),
    "g-thermos": (800, 1200, 0.091, 0.218, 0.891, 0.818),
    "g-torii": (800, 1200, 0.014, 0.260, 0.979, 0.849),
    "g-woolhat": (800, 1200, 0.296, 0.232, 0.679, 0.812),
    "v2-bough": (1200, 800, 0.433, 0.000, 0.999, 0.611),
    "v2-buried-village": (1200, 800, 0.128, 0.299, 0.872, 0.661),
    "v2-gauge": (1200, 800, 0.188, 0.058, 0.786, 0.955),
    "v2-goggles": (1200, 800, 0.208, 0.279, 0.782, 0.755),
    "v2-hare": (800, 1200, 0.222, 0.360, 0.782, 0.674),
    "v2-night-chair": (800, 1200, 0.034, 0.213, 0.948, 0.818),
    "v2-storm-glass": (800, 1200, 0.308, 0.314, 0.688, 0.683),
    "v3-cedar": (800, 1200, 0.254, 0.028, 0.769, 0.730),
    "v3-lantern": (800, 1200, 0.129, 0.145, 0.861, 0.864),
    "v3-stormlift": (800, 1200, 0.085, 0.256, 0.880, 0.782),
}

# Paintings whose cable, wire or track runs to the frame edge: the cut side
# dissolves into the paper with a mask instead of ending in a hard line.
FADES = {"g-daychair": "x", "g-gondola": "x", "g-pylon": "x", "g-owl": "x",
         "g-funicular": "y"}

# Subject height on screen, px. Screen-1 pieces are clearly large (they commit
# to the right page edge); days-gap pieces are clearly small (punctuation
# beside the section label). Nothing sits at headline height mid-field.
SIZE = {
    "v2-bough": 260, "g-cornice": 250, "g-pistebasher": 235, "v3-stormlift": 330,
    "v2-hare": 250, "g-woolhat": 235, "g-skintrack": 290, "g-thermos": 235,
    "g-torii": 270, "g-monkey": 225, "g-origami": 170, "v3-lantern": 270,
    "v2-goggles": 270, "g-alpenglow": 230, "g-mirror": 225,
    "v2-night-chair": 240, "g-pylon": 185, "g-gondola": 170,
    "g-skis": 104, "g-snowshoes": 100, "g-boots": 50, "g-rope": 104,
    "v2-gauge": 200, "g-probe": 235, "v3-cedar": 330,
    "g-icicles": 135, "g-stbernard": 155, "g-fox": 135, "g-owl": 150,
    "g-daychair": 215, "g-funicular": 260,
    "g-drawer": 235, "g-bathtub": 215,
}

# How much of a screen-1 subject runs off the right viewport edge: the cut
# commits the bleed, so the screen edge itself crops the paint (never a crop
# box stopping short of the edge). Wide scenes give up more than objects.
CUT = {"g-skintrack": 0.45, "v2-goggles": 0.38, "g-cornice": 0.38}
CUT_DEFAULT = 0.30

# Themed pools; every pick is crc32(resort id) deep, so it is stable across
# reloads and varied across resorts.
S1_POOLS = {
    "storm": ("v2-bough", "g-cornice", "g-pistebasher", "v3-stormlift"),
    "dry": ("v2-hare", "g-woolhat", "g-thermos", "g-skintrack"),
    "bluebird": ("v2-goggles", "g-alpenglow", "g-mirror"),
    "stale": ("v2-night-chair", "g-pylon", "g-gondola"),
}
# Regional flavour: Japanese resorts draw storm and dry from the Japan set.
# (Pool order matters: picks are positional under the hash.)
S1_POOLS_JP = {
    "storm": ("g-torii", "v2-bough", "g-monkey", "v3-lantern"),
    "dry": ("g-torii", "g-monkey", "g-origami", "v3-lantern"),
}
DUMP_POOL = ("v2-buried-village", "g-drawer", "g-bathtub")
DAYS_POOL = ("g-skis", "g-snowshoes", "g-boots", "g-rope")
STATIONS_POOL = ("v2-gauge", "g-probe", "v3-cedar")
FOOT_POOL = ("g-icicles", "g-stbernard", "g-fox", "g-owl")
POWDER_POOL = ("g-daychair", "g-funicular")


def _pick(pool, key, salt=""):
    return pool[zlib.crc32((str(key) + salt).encode("utf-8")) % len(pool)]


def _s1_pool(st, resort):
    if resort.get("country") == "Japan" and st in S1_POOLS_JP:
        return S1_POOLS_JP[st]
    return S1_POOLS.get(st)


# --------------------------------------------------------------------------
# Crop engine
# --------------------------------------------------------------------------

def _fit(name, pad=0.06):
    """Wrapper and image geometry showing the whole subject plus a paper margin.

    Returns (wrap_w, wrap_h, img_w, ml, mt) in px for a rendered image whose
    subject stands SIZE[name] px tall. Crop edges land on paper because the
    pad stays inside the frame (clamped where the subject meets the edge; those
    assets carry a FADES mask instead).
    """
    w, h, x0, y0, x1, y1 = ASSETS[name]
    sh = SIZE[name]
    img_h = sh / max(y1 - y0, 0.05)
    img_w = img_h * (w / h)
    cx0, cy0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    cx1, cy1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
    return (round((cx1 - cx0) * img_w), round((cy1 - cy0) * img_h),
            round(img_w), round(-cx0 * img_w), round(-cy0 * img_h))


def _mask_css(name):
    fade = FADES.get(name)
    if fade == "x":
        m = "linear-gradient(to right, transparent 0, #000 52px, #000 calc(100% - 52px), transparent 100%)"
    elif fade == "y":
        m = "linear-gradient(to bottom, transparent 0, #000 52px, #000 calc(100% - 52px), transparent 100%)"
    else:
        return ""
    return f"-webkit-mask-image:{m};mask-image:{m};"


def _img(name, iw, ml, mt):
    # loading="lazy": a display:none piece (phone media queries) never fetches,
    # so phones stop downloading the hidden desktop art set (audit round 1).
    return (f'<img src="{ART}/{name}.webp" alt="" aria-hidden="true" '
            f'loading="lazy" decoding="async" style="width:{iw}px;margin:{mt}px 0 0 {ml}px">')


def _raw(cls, name):
    """A hand-placed piece whose geometry lives wholly in the CSS class."""
    return (f'<div class="art-anchor" aria-hidden="true"><div class="art {cls}">'
            f'<img src="{ART}/{name}.webp" alt="" aria-hidden="true" '
            f'loading="lazy" decoding="async"></div></div>')


def _piece(name, pos, cls="art"):
    """One positioned painting inside a zero-height anchor; pos is inline CSS."""
    ww, wh, iw, ml, mt = _fit(name)
    pos = pos.format(w=ww, h=wh)
    style = f"width:{ww}px;height:{wh}px;{pos}{_mask_css(name)}"
    return (f'<div class="art-anchor" aria-hidden="true">'
            f'<div class="{cls}" style="{style}">{_img(name, iw, ml, mt)}</div></div>')


def _left_bleed(name, top, cls="art"):
    """A piece hanging off the left viewport edge: the subject's right edge sits
    20 px inside a 148 px window (the hang margin is at least 158 px wherever
    .art shows, so the window never reaches the text column), the rest runs off
    screen. Pieces whose paint reaches their own frame edge dissolve under
    their FADES mask."""
    w, h, x0, y0, x1, y1 = ASSETS[name]
    sh = SIZE[name]
    img_h = sh / max(y1 - y0, 0.05)
    img_w = img_h * (w / h)
    ww = 148
    ml = round(ww - 20 - x1 * img_w)
    cy0 = max(0.0, y0 - 0.06)
    cy1 = min(1.0, y1 + 0.06)
    wh = round((cy1 - cy0) * img_h)
    mt = round(-cy0 * img_h)
    style = (f"width:{ww}px;height:{wh}px;top:{top}px;"
             f"left:calc(var(--hang)*-1);{_mask_css(name)}")
    return (f'<div class="art-anchor" aria-hidden="true">'
            f'<div class="{cls} art-lb" style="{style}">{_img(name, round(img_w), ml, mt)}</div></div>')


def _right_bleed(name, top, cut=None, cls="art"):
    """A piece committed to the right viewport edge: the window's right edge IS
    the page edge, and the subject's outer share (CUT) runs off screen, so the
    screen edge itself crops the paint (crops at the screen edge are the one
    hard cut the law allows). The window's left edge lands on paper: a 6%
    margin sits before the subject. `top` is an int (px) or a css fragment
    that may use {h} for the window height (for bottom-anchored pieces)."""
    w, h, x0, y0, x1, y1 = ASSETS[name]
    sh = SIZE[name]
    img_h = sh / max(y1 - y0, 0.05)
    img_w = img_h * (w / h)
    pad = round(0.06 * img_w)
    sw = (x1 - x0) * img_w
    if cut is None:
        cut = CUT.get(name, CUT_DEFAULT)
    ww = round(sw * (1 - cut)) + pad
    ml = pad - round(x0 * img_w)
    cy0 = max(0.0, y0 - 0.06)
    cy1 = min(1.0, y1 + 0.06)
    wh = round((cy1 - cy0) * img_h)
    mt = round(-cy0 * img_h)
    top_css = f"top:{top}px;" if isinstance(top, int) else top.format(h=wh)
    style = (f"width:{ww}px;height:{wh}px;{top_css}"
             f"right:calc(var(--m)*-1);{_mask_css(name)}")
    return (f'<div class="art-anchor" aria-hidden="true">'
            f'<div class="{cls}" style="{style}">{_img(name, round(img_w), ml, mt)}</div></div>')


# --------------------------------------------------------------------------
# Slots, in page order
# --------------------------------------------------------------------------

def top_art(st, rid, resort):
    """First block on the page: the bough owns the top-right viewport corner;
    stale pieces anchor the left edge."""
    if st in ("storm", "dry", "bluebird") and _pick(_s1_pool(st, resort), rid) == "v2-bough":
        return _raw("art-bough", "v2-bough")
    if st == "stale":
        name = _pick(S1_POOLS["stale"], rid)
        if name == "v2-night-chair":
            return _raw("art-chair art-lb", name)
        return _left_bleed(name, 140)
    return ""


def verdict_art(st, rid, resort):
    """After the nav, so top:0 is the picker line: the heavy screen-1 piece,
    large and cut by the right page edge (never floating mid-field). Storm
    keeps its ink high; bluebird seats its one saturated mass bottom-right."""
    if st not in ("storm", "dry", "bluebird"):
        return ""
    name = _pick(_s1_pool(st, resort), rid)
    if name == "v2-bough":
        return ""  # the bough already owns the corner
    # storm sits at 140 so the crop window clears the picker glyphs by the full
    # 54 px by box measurement, not just by eye (audit round 1, minor 7).
    top = {"storm": 140, "dry": 140, "bluebird": 400}[st]
    return _right_bleed(name, top)


def low_art(st):
    return ""  # screen 1's lower half stays clear; the rhythm resumes at the days gap


def table_art(st, rid):
    """Before the snow-by-day section: a light companion in the section gap, on
    the side opposite screen 1's art, plus the phone hare on a dry page."""
    out = ""
    if st is None:
        return out  # ruled: "resorts with nothing notable show no art" (verify r1)
    if st == "dump":
        return out
    if st == "bluebird":
        # One painted subject per screenful: bluebird's S1 mass sits low
        # (top:400), so a days sliver entered the same landing screenful.
        # Bluebird is the more-blank-paper state; bare beats a second object
        # (audit round 1, major 1).
        return out
    if st == "stale":
        name = _pick(DAYS_POOL, rid, "days")
        out += _piece(name, "top:8px;left:min(950px, calc(100% - {w}px - 8px));")
    else:
        # Small punctuation in the hang margin beside the SNOW BY DAY block:
        # window right edge held 54px clear of the text column. top:230 (was
        # 104) keeps the piece out of the first 900px screenful at 1440 on
        # every live page (two-line verdicts shift the anchor up ~40px), so
        # it enters only with the S2 screenful (audit round 1).
        name = _pick(DAYS_POOL, rid, "days")
        ww, wh, iw, ml, mt = _fit(name)
        style = (f"width:{ww}px;height:{wh}px;top:230px;"
                 f"right:calc(100% + 54px);{_mask_css(name)}")
        out += (f'<div class="art-anchor" aria-hidden="true">'
                f'<div class="art art-lb" style="{style}">{_img(name, iw, ml, mt)}</div></div>')
    if st == "dry":
        # Sized and seated so the painted peak clears both the ground line
        # above and the table header below by the full 54px at 390 (audit
        # round 1, minor: it sat 29px from the header).
        out += ('<div class="art-anchor" aria-hidden="true"><div class="art-m art-hare-m">'
                f'<img src="{ART}/v2-hare.webp" alt="" aria-hidden="true" '
                'decoding="async" style="width:130px;margin:-66px 0 0 -26px"></div></div>')
    return out


def footer_art(st, rid, resort=None):
    """Before the footer: the measuring still life (or the tall cedar) beside
    stations and sources, bleeding off the right page edge, 1440 px and up.
    Stale and dump pages skip it (stale already carries left-right-left; the
    dump foot belongs to the village). Bluebird is the more-blank-paper state:
    bare beats a second floater, so it skips too. A short stations block (two
    rows or fewer) also skips it: the stations piece and the foot piece would
    share one screenful (audit round 1, major; bluebird's own drop rule,
    applied generally)."""
    if st is None:
        return ""  # ruled: "resorts with nothing notable show no art" (verify r1)
    if st in ("dump", "stale", "bluebird"):
        return ""
    if resort is not None:
        stations = [s for s in (resort.get("stations") or []) if isinstance(s, dict)]
        if len(stations) <= 2:
            return ""
    name = _pick(STATIONS_POOL, rid, "stations")
    return _right_bleed(name, "top:calc(30px - {h}px);", cut=0.25, cls="art art-st")


def foot_art(st, rid, far=False):
    """After the last block. Dump: the buried village dissolving at the very
    foot, on every viewport (the brief's dump row: the page ends in the
    in-flow dissolve; the centred, headline-height pool mate was the round's
    own "reference wrong" — audit round 1, major). Otherwise a light small
    object, left side, in flow below everything. far=True when the page also
    carries the stations piece: the foot then opens one empty screenful of
    bare paper above itself so no 900px window frames two painted subjects."""
    if st is None:
        return ""  # ruled: "resorts with nothing notable show no art" (verify r1)
    if st == "dump":
        return ('<div class="art-foot" aria-hidden="true">'
                f'<img src="{ART}/v2-buried-village.webp" alt="" aria-hidden="true" '
                'loading="lazy" decoding="async"></div>')
    name = _pick(FOOT_POOL, rid, "foot")
    ww, wh, iw, ml, mt = _fit(name)
    cls = "art-flow far" if far else "art-flow"
    style = f"width:{ww}px;height:{wh}px;left:-28px;top:0;{_mask_css(name)}"
    return (f'<div class="{cls}" aria-hidden="true" style="height:{wh + 10}px">'
            f'<div class="art" style="{style}">{_img(name, iw, ml, mt)}</div></div>')


def powder_art(empty=False):
    """Storm in a glass, right of the Powder title, cropped by the viewport
    edge. On the empty state (no powder table at all) the lower half of the
    sheet was dead paper, so the top slot stays bare and powder_art_low()
    closes the sheet on the glass instead (audit round 1, minor 5)."""
    if empty:
        return ""
    return _raw("art-glass", "v2-storm-glass")


def powder_art_low():
    """The empty state's close: the glass, smaller, low on the sheet, still
    cut by the right viewport edge, anchored after the Refresh button so the
    page ends on the object, not on dead paper."""
    return _raw("art-glass-low", "v2-storm-glass")


def powder_companion(page_html, updated):
    """One small companion beside the measured-snow group, left side (the glass
    holds the right). Injected before the group's label; the forecast group
    stays clean. No measured group today means no companion."""
    marker = '<div class="label gap2">Snow measured'
    if marker not in page_html:
        return page_html
    key = updated.date().isoformat() if updated is not None else "powder"
    piece = _left_bleed(_pick(POWDER_POOL, key), 30)
    return page_html.replace(marker, piece + marker, 1)


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

CSS = """
/* paintings: zero-height anchors, crops whose edges always land on paper */
.art-anchor{ position:relative; height:0; overflow:visible; }
.art{ position:absolute; overflow:hidden; pointer-events:none; z-index:0; }
.art img, .art-m img, .art-foot img{ display:block; max-width:none; pointer-events:none; }

/* storm: the laden bough enters from the top-right viewport corner */
.art-bough{ top:0; right:calc(var(--m)*-1 - 30px); width:440px; height:330px; }
.art-bough img{ width:690px; margin:0 0 0 -250px; }

/* stale: the night chair anchors the left edge */
.art-chair{ top:120px; left:calc(var(--hang)*-1); width:150px; height:390px; }
.art-chair img{ width:375px; margin:-90px 0 0 -215px; }

/* powder page: storm in a glass, cropped by the right viewport edge */
.art-glass{ top:-10px; right:calc(var(--m)*-1 - 150px); width:420px; height:520px; }
.art-glass img{ width:520px; margin:-195px 0 0 -78px; }
/* empty state: a smaller glass after the button closes the sheet on the
   object (one screenful; the window still bleeds the right viewport edge).
   On tall windows it rides down so the sheet still ends on the object. */
.art-glass-low{ top:-60px; right:calc(var(--m)*-1 - 121px); width:340px; height:420px; }
.art-glass-low img{ width:420px; margin:-157px 0 0 -63px; }
@media (min-height:960px){ .art-glass-low{ top:calc(100vh - 981px); } }

/* dump: the buried village dissolves at the very foot of the page (in flow) */
.art-foot{ margin-left:calc(var(--hang)*-1); width:100vw; height:30vw;
  overflow:hidden; pointer-events:none; }
.art-foot img{ width:100vw; margin-top:-17vw; }

/* the light foot object: in flow below every block, so nothing can shift */
.art-flow{ position:relative; overflow:visible; pointer-events:none; margin-top:var(--s2); }
/* a page that also shows the stations piece opens one empty screenful of
   bare paper before its foot object, so no scroll position frames two
   painted subjects (the wedding close; audit round 1, major 2) */
@media (min-width:1440px){ .art-flow.far{ margin-top:900px; } }

/* phone hare: sits in the section gap above the daily table, right side,
   54px clear of the ground line above and the table header below */
.art-m{ position:absolute; overflow:hidden; pointer-events:none; display:none; }
.art-hare-m{ top:9px; right:4px; width:80px; height:77px; }

/* the stations still life joins only at 1440 and up */
.art-st{ display:none; }
@media (min-width:1440px){ .art-st{ display:block; } }

/* narrow screens: margins and whitespace shrink, positioned art steps back;
   left-margin pieces need the full 158px hang, so they go first.
   .art-flow hides with .art (not at 700): its fixed height was a dead blank
   strip above the footer on tablets while its child was display:none. */
@media (max-width:1359px){ .art-lb{ display:none !important; } }
@media (max-width:1100px){ .art{ display:none !important; } .art-flow{ display:none; } }
@media (max-width:700px){
  .art-flow{ display:none; }
  .art-m{ display:block; }
  /* the dissolve strip reads too timid at 100vw on a phone: scale the painting
     up and keep the crop edges on paper (village spans y 28-65% of the frame) */
  .art-foot{ height:46vw; }
  .art-foot img{ width:150vw; margin:-24vw 0 0 -25vw; }
}
"""
