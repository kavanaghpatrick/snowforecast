#!/usr/bin/env python3
"""Snow forecast fetcher for Dashboard V2 (schema_version 3).

Writes data/forecasts_v2.json as described in docs/dashboard_v2_redesign_spec.md,
sections 1 and 2.

Sources:
- Open-Meteo /v1/forecast (best_match model): daily, hourly and current values at
  base, mid and summit elevation, plus a per-model daily snowfall comparison.
- Measured snow depth and new snow: SNOTEL (USDA NRCS AWDB REST API), GeoSphere
  Austria TAWES, Land Tirol (LWD/HD), WSL SLF (IMIS and study plots), JMA AMeDAS,
  Météo-France hourly files (data.gouv.fr), Meteocat XEMA, and AEMET OpenData
  when AEMET_API_KEY is set. See docs/depth_coverage_plan.md.
- Resort reports: OnTheSnow (RSC payload), tenki.jp (strict pattern only).

Hard rule: no value is ever estimated or filled. A missing input yields null.

Stdlib only. Python 3.11.

Usage:
    python3 scripts/fetch_data_v2.py
    python3 scripts/fetch_data_v2.py --only "Alta,Zermatt,Val Thorens" --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import http.client
import io
import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_forecasts_v2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_v2")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "forecasts_v2.json"

SCHEMA_VERSION = 3

# =============================================================================
# Display-rule constants (copied into the JSON under "thresholds")
# =============================================================================
THRESHOLDS = {
    "wind_gust_hold_kmh": 60,
    "wind_700hpa_hold_kmh": 70,
    "storm_3day_cm": 15,
    "storm_codes": [73, 75, 85, 86],
    "storm_prob_pct": 50,
    "storm_day_min_cm": 10,
    "powder_cold_below_c": -8,
    "powder_warm_above_c": -3,
    "snowy_day_min_cm": 1.0,
    "bluebird_prev24_cm": 10,
    "bluebird_codes": [0, 1, 2],
    "flat_light_visibility_m": 1000,
    "flat_light_cloud_pct": 90,          # applied to max(cloud_cover_low, cloud_cover_mid)
    # Precipitation phase (spec 1.9.2). Open-Meteo's rain/snow split at a
    # mountain point is the model grid cell's split (only temperature is
    # elevation-adjusted), so phase is classified per hour from the column
    # freezing level / snow line plus the point's own temperature.
    "snowline_below_fl_m": 250,          # snow line proxy = freezing level - 250 m where snowfall_height is null
    "phase_min_mm": 1.0,                 # rain_risk / mixed_risk need at least this much water in that phase
    "phase_share_min": 0.2,              # ... and at least this share of the period's water
    "phase_day_min_mm": 0.5,             # below this total the period's phase is "none"
    "bluebird_cloud_mean_pct": 40,
    "bluebird_flat_light_max_h": 2,
    "wind_700hpa_min_summit_m": 2500,    # 700 hPa wind counts toward wind_hold only for summits this high
}

# Station acceptance (docs/depth_coverage_plan.md 0.1). A station outside either
# limit is dropped with a resort note.
MAX_STATION_DISTANCE_KM = 15.0
STATION_ELEV_BELOW_BASE_M = 300   # station may sit at most 300 m below the resort base
STATION_ELEV_ABOVE_TOP_M = 300    # and at most 300 m above the summit (base + 1500 if summit unknown)
MAX_REPORT_DEPTH_CM = 600.0
STATION_PROVIDERS = ("SNOTEL", "GeoSphere", "SLF", "JMA", "Meteo-France", "Meteocat", "Tirol", "AEMET")

# Max age of a station reading before it stops counting as current depth or
# recent snow (plan 0.4): each source's update cycle with one missed update.
# Keyed by provider, or provider:station_type where a provider has two kinds.
STATION_MAX_AGE_H = {
    "JMA": 6,
    "Meteo-France:auto": 36,
    "Meteo-France:manual": 72,
    "Meteocat": 12,
    "Tirol": 6,
    "GeoSphere": 6,
    "SLF:imis": 6,
    "SLF:study_plot": 36,
    "SNOTEL": 48,          # date only: the reading's date may be at most 2 days before the run date
    "AEMET": 6,
}

API_UA = "snowforecast/3 (+https://github.com/kavanaghpatrick/snowforecast)"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# =============================================================================
# USA - SNOTEL stations
# Format: (name, lat, lon, region, elevation_m, stations_list)
# stations_list: [(station_triplet, station_name, weight), ...]
# The weight element is ignored (no averaging). Triplets corrected on
# 27 Nov 2026 against the AWDB stations endpoint (name and coordinates checked).
# =============================================================================
US_SKI_AREAS = [
    # Washington
    ("Stevens Pass", 47.7448, -121.089, "Washington", 1238, [
        ("791:WA:SNTL", "Stevens Pass", 1.0),
        ("672:WA:SNTL", "Olallie Meadows", 0.5),
    ]),
    ("Crystal Mountain", 46.9282, -121.5045, "Washington", 1341, [
        ("642:WA:SNTL", "Morse Lake", 1.0),  # was 679 (AWDB: Paradise)
    ]),
    ("Mt. Baker", 48.857, -121.6695, "Washington", 1067, [
        ("909:WA:SNTL", "Wells Creek", 1.0),
    ]),
    ("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 945, [
        ("672:WA:SNTL", "Olallie Meadows", 1.0),  # 5.6 km; replaces 788 Stampede Pass (17.1 km)
    ]),
    # Oregon
    ("Mt. Hood Meadows", 45.3311, -121.6647, "Oregon", 1379, [
        ("651:OR:SNTL", "Mt Hood Test Site", 1.0),
    ]),
    ("Mt. Bachelor", 43.9792, -121.6886, "Oregon", 1920, [
        ("733:OR:SNTL", "Santiam Jct.", 1.0),  # was 729 (AWDB: Salt Creek Falls)
    ]),
    ("Timberline", 45.3309, -121.7109, "Oregon", 1829, [
        ("651:OR:SNTL", "Mt Hood Test Site", 1.0),
    ]),
    # California
    ("Mammoth Mountain", 37.6308, -119.0326, "California", 2424, [
        ("846:CA:SNTL", "Virginia Lakes Ridge", 1.0),
    ]),
    ("Palisades Tahoe", 39.1969, -120.2358, "California", 1890, [
        ("784:CA:SNTL", "Palisades Tahoe", 1.0),
        ("809:CA:SNTL", "Tahoe City Cross", 0.7),
        ("541:CA:SNTL", "Independence Lake", 0.5),
    ]),
    ("Heavenly", 38.9353, -119.9396, "California", 1906, [
        ("518:CA:SNTL", "Heavenly Valley", 1.0),  # was 473 (AWDB: Fallen Leaf)
    ]),
    ("Kirkwood", 38.6848, -120.0655, "California", 2377, [
        ("1067:CA:SNTL", "Carson Pass", 1.0),  # 5.6 km, 2,548 m
    ]),
    # Colorado
    ("Vail", 39.6403, -106.3742, "Colorado", 2454, [
        ("842:CO:SNTL", "Vail Mountain", 1.0),
        ("1041:CO:SNTL", "Beaver Ck Village", 0.7),
        ("415:CO:SNTL", "Copper Mountain", 0.5),
    ]),
    ("Breckenridge", 39.4817, -106.0384, "Colorado", 2926, [
        ("415:CO:SNTL", "Copper Mountain", 1.0),
    ]),
    ("Aspen Snowmass", 39.2084, -106.949, "Colorado", 2623, [
        # 32.6 km away: kept here so the 15 km rule removes it with a note on every run.
        ("542:CO:SNTL", "Independence Pass", 1.0),
    ]),
    ("Telluride", 37.9375, -107.8123, "Colorado", 2659, [
        ("1344:CO:SNTL", "Alta Lakes", 1.0),  # 6.1 km, ski-area side
        ("538:CO:SNTL", "Idarado", 0.5),      # 11.9 km, over the ridge
    ]),
    # Utah
    ("Park City", 40.6514, -111.508, "Utah", 2103, [
        ("814:UT:SNTL", "Thaynes Canyon", 1.0),  # 3.8 km
        ("628:UT:SNTL", "Mill-D North", 0.7),
    ]),
    ("Snowbird", 40.583, -111.6508, "Utah", 2365, [
        ("766:UT:SNTL", "Snowbird", 1.0),
        ("366:UT:SNTL", "Brighton", 0.7),
    ]),
    ("Alta", 40.5884, -111.6386, "Utah", 2600, [
        # was 332:UT:SNTL "Alta", which AWDB says is Ben Lomond Peak (~90 km north).
        ("1308:UT:SNTL", "Atwater", 1.0),
        ("766:UT:SNTL", "Snowbird", 0.8),
    ]),
    # Montana
    ("Big Sky", 45.2618, -111.4018, "Montana", 2290, [
        ("590:MT:SNTL", "Lone Mountain", 1.0),  # 2.4 km; replaces 754 Shower Falls (38 km)
    ]),
    ("Whitefish", 48.4833, -114.355, "Montana", 1361, [
        ("664:MT:SNTL", "Noisy Basin", 1.0),  # was 656 (AWDB: Mule Creek)
    ]),
    # Wyoming
    ("Jackson Hole", 43.5875, -110.8281, "Wyoming", 1924, [
        ("689:WY:SNTL", "Phillips Bench", 1.0),  # 10.4 km; replaces 497 Granite Creek (41.8 km)
    ]),
    # Idaho
    ("Sun Valley", 43.6804, -114.4075, "Idaho", 1753, [
        ("895:ID:SNTL", "Chocolate Gulch", 1.0),  # 9.8 km; replaces 537 Hyndman (20.3 km)
    ]),
]

# =============================================================================
# AUSTRIA - GeoSphere TAWES and Land Tirol stations
# Format: (name, lat, lon, region, station_elevation_m, stations)
# stations: [(provider, station_id, label), ...] in priority order; the first
# station with a fresh reading gives the depth (plan 0.2). The 5th element is
# the old single-station elevation and is unused; the resort base/summit come
# from RESORT_ELEVATIONS. Station swaps per docs/depth_coverage_plan.md 1.4:
# the old GeoSphere stations at Galzig, Brunnenkogel, Galtür and Schmittenhöhe
# have no snow-depth sensor.
# =============================================================================
TIROL_SOLDEN_HD = "80b5d623-5fb3-4abe-ed2e-ad5cfe789785"   # HD Tirol "Sölden" (no LWD number)
AUSTRIA_SKI_AREAS = [
    ("St. Anton", 47.1297, 10.2303, "Tyrol", 2079, [
        ("GeoSphere", "11311", "St. Anton/Arlberg"),
        ("Tirol", "SKAP2", "Kapall"),
    ]),
    ("Sölden", 46.9128, 10.8617, "Tyrol", 3340, [
        ("Tirol", TIROL_SOLDEN_HD, "Sölden (HD Tirol)"),
        ("Tirol", "SGIG2", "Giggijoch"),
    ]),
    ("Obergurgl", 46.8669, 11.0244, "Tyrol", 1941, [("GeoSphere", "11127", "Obergurgl")]),
    ("Ischgl", 46.9681, 10.1856, "Tyrol", 1587, [
        ("Tirol", "IPIS2", "Pischgraben"),
        ("Tirol", "INAC2", "Nachtweide"),
    ]),
    ("Kitzbühel", 47.4183, 12.3592, "Tyrol", 1794, [("GeoSphere", "8989044", "Hahnenkamm")]),
    ("Lech am Arlberg", 47.1575, 10.2128, "Vorarlberg", 2079, [
        ("GeoSphere", "11308", "Warth"),
        ("GeoSphere", "11307", "Langen/Arlberg"),
    ]),
    ("Pitztal Glacier", 46.9269, 10.8792, "Tyrol", 2864, [("GeoSphere", "11316", "Pitztaler Gletscher")]),
    # Flattnitz is 49.7 km away: kept so the 15 km rule removes it with a note on every run.
    ("Obertauern", 47.2489, 13.5597, "Salzburg", 1630, [("GeoSphere", "11222", "Flattnitz")]),
    ("Zell am See", 47.3286, 12.7381, "Salzburg", 1956, [("GeoSphere", "11366", "Saalbach")]),
    ("Schladming", 47.4678, 13.6264, "Styria", 2015, [("GeoSphere", "11351", "Ramsau/Dachstein")]),
]

# =============================================================================
# SWITZERLAND - SLF IMIS stations and study plots
# Format: (name, lat, lon, region, station_elevation_m, stations)
# stations: [(station_code, label, station_type)] in priority order;
# station_type "imis" (30-min automatic) or "study_plot" (daily manual).
# =============================================================================
SWISS_SKI_AREAS = [
    ("Zermatt", 45.9872, 7.7836, "Valais", 2953, [("GOR2", "Gornergratsee", "imis")]),
    ("Saas-Fee", 46.1275, 7.9814, "Valais", 2480, [("SAA2", "Seetal", "imis")]),
    ("Verbier", 46.0989, 7.2856, "Valais", 2550, [("ATT2", "Lac des Vaux", "imis")]),
    ("St. Moritz", 46.4761, 9.8438, "Graubünden", 2512, [("BEV2", "Valetta", "imis")]),
    ("Davos", 46.8131, 9.8439, "Graubünden", 2540, [("WFJ2", "Weissfluhjoch", "imis")]),
    ("Laax", 46.8356, 9.2317, "Graubünden", 2325, [("CMA2", "La Fuorcla", "imis")]),
    # PMA2 Colms da Parsonz is 16.4 km away; ROT3 Plang Bi (3.9 km) is in the linked area.
    ("Lenzerheide", 46.7267, 9.5542, "Graubünden", 2429, [("ROT3", "Plang Bi", "imis")]),
    ("Mürren", 46.5573, 7.8352, "Bern", 2332, [
        ("SCH2", "Türliboden", "imis"),          # silent in Nov 2026, kept in case it returns
        ("1MR0", "Mürren", "study_plot"),        # 4.2 km, at the base
        ("LHO2", "Russisprung", "imis"),         # 8.8 km
    ]),
    ("Andermatt", 46.6544, 8.6106, "Uri", 2209, [("GOS3", "Gütsch", "imis")]),
    ("Arosa", 46.7594, 9.6669, "Graubünden", 2495, [("ROT3", "Plang Bi", "imis")]),
    ("Engelberg", 46.7711, 8.4306, "Uri", 2149, [("TIT2", "Titlisboden", "imis")]),
]

# =============================================================================
# JAPAN - JMA AMeDAS snow stations (plan 1.1). {resort: [(station_id, label)]}
# Kiroro's Akaigawa and Zao's Yamagata fail the elevation rule; they stay
# configured so the rule removes them with a note on every run.
# =============================================================================
JMA_STATIONS = {
    "Niseko United": [("16217", "Kutchan")],
    "Furano": [("12626", "Furano")],
    "Rusutsu": [("16286", "Kimobetsu")],
    "Kiroro": [("16126", "Akaigawa")],
    "Hakuba Valley": [("48141", "Hakuba")],
    "Happo-One": [("48141", "Hakuba")],
    "Nozawa Onsen": [("48031", "Nozawa Onsen")],
    "Shiga Kogen": [("42121", "Kusatsu")],
    "Myoko Kogen": [("48061", "Shinano-Machi")],
    "Zao Onsen": [("35426", "Yamagata")],
}

# =============================================================================
# FRANCE (and La Molina / Masella) - Météo-France hourly climatological files
# (plan 1.2 / 1.3). {resort: [(NUM_POSTE, label, station_type)]} in priority order.
# station_type is "manual" for once-a-day ski-patrol (nivo) posts and "auto" for
# stations that report hourly, checked against each station's reporting pattern
# (an ID pattern is not reliable: Puigmal-Nivose 66067402 is automatic).
# =============================================================================
METEO_FRANCE_STATIONS = {
    "Val Thorens": [("73257005", "Val Thorens SAPC", "auto")],
    "Les Arcs": [("73054401", "Les Arcs (manual)", "manual"), ("73150004", "La Plagne SAPC", "auto")],
    "Val d'Isère": [("73304005", "Val d'Isere Jos", "auto")],
    "Les Menuires": [("73257400", "Les Menuires (manual)", "manual"), ("73257005", "Val Thorens SAPC", "auto")],
    "Avoriaz": [("74191406", "Avoriaz (manual)", "manual")],
    "La Plagne": [("73150004", "La Plagne SAPC", "auto"), ("73150400", "La Plagne (manual)", "manual")],
    "Courchevel": [("73227400", "Courchevel (manual)", "manual"), ("73015400", "Meribel Mottaret (manual)", "manual")],
    "Chamonix": [("74056001", "Chamonix", "auto")],
    "Morzine": [("74134400", "Les Gets (manual)", "manual"), ("74191406", "Avoriaz (manual)", "manual")],
    "Tignes": [("73296005", "Tignes SAPC", "auto")],
    "Méribel": [("73015400", "Meribel Mottaret (manual)", "manual")],
    "Megève": [("74173402", "Megeve Neige", "manual")],
    "La Clusaz": [("74136005", "Le Grand-Bornand", "auto")],
    "Flaine": [("74014004", "Flaine SAPC", "auto")],
    "Les Gets": [("74134400", "Les Gets (manual)", "manual")],
    "Alpe d'Huez": [("38191002", "Alpe-d'Huez", "auto")],
    "Les 2 Alpes": [("38253400", "Les 2 Alpes Toura NE (manual)", "manual"),
                    ("38375009", "St Christophe en Oisans SAPC", "auto")],
    "Serre Chevalier": [("05133400", "Serre Che-Chantemerle (manual)", "manual"), ("05183001", "Villar St Pancrace", "auto")],
    "Vars": [("05026400", "Ceillac Nivo (manual)", "manual")],
    "Isola 2000": [("06073005", "Isola 2000", "auto")],
    # Spain, French side of the massif
    "La Molina": [("66067402", "Puigmal-Nivose", "auto")],
    "Masella": [("66067402", "Puigmal-Nivose", "auto")],   # 16.3 km: the distance rule removes it with a note
}
METEO_FRANCE_DATASET = "6569b4473bedf2e7abad3b72"
METEO_FRANCE_FALLBACK_URL = ("https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR/"
                             "H_{dep}_latest-2025-2026.csv.gz")

# =============================================================================
# SPAIN - Meteocat XEMA (plan 1.3). {resort: [(codi_estacio, label)]}
# Espot (Z7) sits about 400 m above Espot's listed (unverified) summit, so the
# elevation rule removes it; it becomes valid with no other change if the
# summit is confirmed near 2,500 m.
# Bonabé (YT) is left out: its snow-depth sensor jumps between about 4 and 157 cm
# within hours (26-27 Nov 2026), so its readings cannot be trusted.
# =============================================================================
METEOCAT_STATIONS = {
    "Baqueira-Beret": [("Z1", "Bonaigua")],
    "Boí Taüll": [("Z2", "Boí")],
    "Vallter 2000": [("ZC", "Ulldeter")],
    "Espot Esquí": [("Z7", "Espot")],
}

# AEMET OpenData (needs a free key in the AEMET_API_KEY environment variable /
# repo secret). Unverified: used only when a key is configured (plan 3).
AEMET_RESORTS = ("Formigal", "Panticosa", "Candanchú", "Astún", "Cerler", "Sierra Nevada")

# Why a resort has no usable station (plan 1.x "unresolvable" lists). Added to
# the resort's notes on every run so an empty depth explains itself.
DEPTH_COVERAGE_NOTES = {
    "Aspen Snowmass": "No snow station within 15 km (the nearest SNOTEL site is 22.2 km away); "
                      "depth is shown only when the resort report gives one",
    "Mammoth Mountain": "No SNOTEL site within 15 km; depth comes from the resort report only",
    "Mt. Bachelor": "No SNOTEL site within 15 km; depth comes from the resort report only",
    "Whitefish": "No SNOTEL site within 15 km; depth comes from the resort report only",
    "Obertauern": "No station with a snow-depth sensor within 15 km (GeoSphere Obertauern has none, and the "
                  "Salzburg avalanche service publishes only chart images), so there is no measured depth",
    "Kiroro": "The only JMA snow station within 15 km (Akaigawa) is 372 m below the base, so it is not used; "
              "depth comes from the resort report only",
    "Zao Onsen": "The only JMA snow station within 15 km is in Yamagata city, 627 m below the base, "
                 "so there is no measured depth",
    "Formigal": "No keyless snow station within 15 km (the nearest are French stations 19.5 to 24 km away); "
                "AEMET OpenData needs an API key and is used only when one is configured",
    "Panticosa": "No keyless snow station within 15 km (the nearest are French stations 19.5 to 24 km away); "
                 "AEMET OpenData needs an API key and is used only when one is configured",
    "Candanchú": "No keyless snow station within 15 km (the nearest are French stations 19.5 to 24 km away); "
                 "AEMET OpenData needs an API key and is used only when one is configured",
    "Astún": "No keyless snow station within 15 km (the nearest are French stations 19.5 to 24 km away); "
             "AEMET OpenData needs an API key and is used only when one is configured",
    "Cerler": "No keyless snow station within 15 km (the nearest are French stations 19.5 to 24 km away); "
              "AEMET OpenData needs an API key and is used only when one is configured",
    "Sierra Nevada": "No keyless live snow-depth source found; AEMET OpenData needs an API key and is used "
                     "only when one is configured",
    "Masella": "The nearest suitable snow stations (Puigmal and Meteocat Malniu) are 16.3 km away, "
               "over the 15 km limit, so there is no measured depth",
    "Port Ainé": "The nearest suitable snow station (Meteocat Salòria) is 16.7 km away, and Sort is about "
                 "1 km below the base, so there is no measured depth",
    "Espot Esquí": "Meteocat Espot (4.6 km) sits about 400 m above the listed 2,110 m summit, which is "
                   "unverified, so it is not used",
}

# =============================================================================
# JAPAN / FRANCE / SPAIN - resort reports + Open-Meteo
# Format: (name, lat, lon, region, elevation_m)
# =============================================================================
JAPAN_SKI_AREAS = [
    # Hokkaido
    ("Niseko United", 42.8635, 140.6981, "Hokkaido", 260),
    ("Furano", 43.2818, 142.4735, "Hokkaido", 245),
    ("Rusutsu", 42.75, 140.88, "Hokkaido", 400),
    ("Kiroro", 43.0701, 140.9891, "Hokkaido", 520),
    # Nagano
    ("Hakuba Valley", 36.70, 137.83, "Nagano", 760),
    ("Happo-One", 36.70, 137.827, "Nagano", 760),
    ("Nozawa Onsen", 36.9228, 138.4406, "Nagano", 565),
    ("Shiga Kogen", 36.70, 138.50, "Nagano", 1300),
    # Niigata
    ("Myoko Kogen", 36.88, 138.12, "Niigata", 731),
    # Yamagata
    ("Zao Onsen", 38.16, 140.40, "Yamagata", 780),
]

FRANCE_SKI_AREAS = [
    # Northern Alps (Savoie / Haute-Savoie)
    ("Val Thorens", 45.2922, 6.5747, "Northern Alps", 2300),
    ("Les Arcs", 45.5427, 6.7587, "Northern Alps", 1200),
    ("Val d'Isère", 45.4485, 6.9802, "Northern Alps", 1850),
    ("Les Menuires", 45.3200, 6.5300, "Northern Alps", 1850),
    ("Avoriaz", 46.1878, 6.7717, "Northern Alps", 1800),
    ("La Plagne", 45.5064, 6.6772, "Northern Alps", 1250),
    ("Courchevel", 45.4154, 6.6349, "Northern Alps", 1300),
    ("Chamonix", 45.9238, 6.8693, "Northern Alps", 1035),
    ("Morzine", 46.1792, 6.7089, "Northern Alps", 1000),
    ("Tignes", 45.4686, 6.9063, "Northern Alps", 1550),
    ("Méribel", 45.4010, 6.5655, "Northern Alps", 1100),
    ("Megève", 45.8569, 6.6178, "Northern Alps", 1113),
    ("La Clusaz", 45.9050, 6.4239, "Northern Alps", 1050),
    ("Flaine", 46.0067, 6.6895, "Northern Alps", 1600),
    ("Les Gets", 46.1577, 6.6694, "Northern Alps", 1172),
    # Southern Alps (Isère / Hautes-Alpes)
    ("Alpe d'Huez", 45.0500, 6.0333, "Southern Alps", 1125),
    ("Les 2 Alpes", 45.0043, 6.1197, "Southern Alps", 1280),
    ("Serre Chevalier", 44.9400, 6.5667, "Southern Alps", 1200),
    ("Vars", 44.5938, 6.6899, "Southern Alps", 1650),
    ("Isola 2000", 44.1866, 7.1579, "Southern Alps", 1800),
]

SPAIN_SKI_AREAS = [
    # Pyrenees
    ("Baqueira-Beret", 42.701, 0.971, "Pyrenees", 1500),
    ("Formigal", 42.774, -0.361, "Pyrenees", 1500),
    ("Boí Taüll", 42.473, 0.821, "Pyrenees", 2020),
    ("Candanchú", 42.788, -0.528, "Pyrenees", 1560),
    ("La Molina", 42.344, 1.957, "Pyrenees", 1700),
    ("Masella", 42.351, 1.901, "Pyrenees", 1600),
    ("Cerler", 42.593, 0.538, "Pyrenees", 1500),
    ("Astún", 42.810, -0.506, "Pyrenees", 1700),
    ("Panticosa", 42.747, -0.358, "Pyrenees", 1500),
    ("Vallter 2000", 42.427, 2.265, "Pyrenees", 1959),
    ("Port Ainé", 42.421, 1.212, "Pyrenees", 1650),
    ("Espot Esquí", 42.564, 1.094, "Pyrenees", 1510),
    # Sierra Nevada (Andalusia)
    ("Sierra Nevada", 37.095, -3.400, "Sierra Nevada", 2100),
]

# =============================================================================
# Resort base and summit elevations (metres). Configuration, not data.
# name: (base_m, summit_m or None, note)
# Summits are lift-served tops with the source URL in the comment above each
# entry (researched 27 Nov 2026). Entries marked "unverified" have no
# trustworthy source yet and keep the figure carried over from
# elevation_bands.RESORT_VERTICAL_M.
# =============================================================================
RESORT_ELEVATIONS = {
    # USA
    # https://en.wikipedia.org/wiki/Stevens_Pass_Ski_Area
    "Stevens Pass": (1238, 1782, "en.wikipedia.org (base 1,238 m, lift-served top 1,782 m)"),
    # https://en.wikipedia.org/wiki/Crystal_Mountain_(Washington)
    "Crystal Mountain": (1341, 2131, "en.wikipedia.org (base 1,341 m, lift-served top 2,131 m)"),
    # https://en.wikipedia.org/wiki/Mt._Baker_Ski_Area
    "Mt. Baker": (1067, 1551, "en.wikipedia.org (base 1,067 m, lift-served top 1,551 m)"),
    # https://en.wikipedia.org/wiki/The_Summit_at_Snoqualmie
    "Snoqualmie Pass": (945, 1652, "en.wikipedia.org (base 945 m, lift-served top 1,652 m)"),
    # https://archive.skihood.com/en/the-mountain/hours-of-operations/mountain-statistics
    "Mt. Hood Meadows": (1379, 2225, "archive.skihood.com (base 1,379 m, lift-served top 2,225 m)"),
    # https://www.mtbachelor.com/the-mountain/resort-policies-safety/mountain-stats/
    "Mt. Bachelor": (1737, 2763, "mtbachelor.com (base 1,737 m, lift-served top 2,763 m; base changed from 1,920 m)"),
    # https://www.timberlinelodge.com/mountain/ski-area/
    # Base kept at the lodge (1,829 m), which is the resort's forecast coordinate; the 1,219 m
    # figure on the official site is the separate Summit Pass area in Government Camp.
    "Timberline": (1829, 2603, "timberlinelodge.com (lodge base 1,829 m, lift-served top 2,603 m)"),
    # https://www.skiresort.com/en/ski-resort/mammoth-mountain/
    "Mammoth Mountain": (2424, 3369, "skiresort.com (base 2,424 m, lift-served top 3,369 m)"),
    # https://www.palisadestahoe.com/footer/mountain-statistics
    "Palisades Tahoe": (1890, 2745, "palisadestahoe.com (base 1,890 m, lift-served top 2,745 m)"),
    # https://en.wikipedia.org/wiki/Heavenly_Mountain_Resort
    "Heavenly": (1906, 3068, "en.wikipedia.org (base 1,906 m, lift-served top 3,068 m)"),
    # https://news.vailresorts.com/mountain/kirkwood/fact-sheet/
    "Kirkwood": (2377, 2987, "news.vailresorts.com (base 2,377 m, lift-served top 2,987 m)"),
    # https://en.wikipedia.org/wiki/Vail_Ski_Resort
    "Vail": (2454, 3527, "en.wikipedia.org (base 2,454 m, lift-served top 3,527 m)"),
    # https://en.wikipedia.org/wiki/Breckenridge_Ski_Resort
    "Breckenridge": (2926, 3914, "en.wikipedia.org (base 2,926 m, lift-served top 3,914 m)"),
    # https://en.wikipedia.org/wiki/Snowmass_(ski_area)
    "Aspen Snowmass": (2623, 3813, "en.wikipedia.org (base 2,623 m, lift-served top 3,813 m)"),
    # https://en.wikipedia.org/wiki/Telluride_Ski_Resort
    "Telluride": (2659, 3831, "en.wikipedia.org (base 2,659 m, lift-served top 3,831 m)"),
    # https://en.wikipedia.org/wiki/Park_City_Mountain_Resort
    "Park City": (2103, 3056, "en.wikipedia.org (base 2,103 m, lift-served top 3,056 m)"),
    # https://en.wikipedia.org/wiki/Snowbird_ski_resort
    "Snowbird": (2365, 3353, "en.wikipedia.org (base 2,365 m, lift-served top 3,353 m)"),
    # https://slopelab.com/alta-review/
    "Alta": (2600, 3216, "slopelab.com (base 2,600 m, lift-served top 3,216 m)"),
    # https://www.skiresort.com/en/ski-resort/big-sky-resort/
    "Big Sky": (2073, 3403, "skiresort.com (base 2,073 m, lift-served top 3,403 m; base changed from 2,290 m)"),
    # https://www.skiresort.com/en/ski-resort/whitefish-mountain-resort/
    "Whitefish": (1361, 2078, "skiresort.com (base 1,361 m, lift-served top 2,078 m)"),
    # https://www.skiresort.com/en/ski-resort/jackson-hole/
    "Jackson Hole": (1924, 3185, "skiresort.com (base 1,924 m, lift-served top 3,185 m)"),
    # https://en.wikipedia.org/wiki/Bald_Mountain_(Idaho)
    "Sun Valley": (1753, 2789, "en.wikipedia.org (base 1,753 m, lift-served top 2,789 m)"),
    # Austria (resort base and top from published mountain stats)
    "St. Anton": (1304, 2811, "skiarlberg.at/en/st-anton (village 1,304 m, Valluga 2,811 m)"),
    "Sölden": (1350, 3340, "soelden.com/en/activities/winter/skiing-snowboarding/ski-area-information"),
    "Obergurgl": (1800, 3030, "gurgl.com/en/activities/winter/skiing-snowboarding/ski-area-information"),
    "Ischgl": (1377, 2872, "ischgl.com/en/winter/silvretta-arena"),
    "Kitzbühel": (800, 2000, "kitzbueheler-alpen.com/en/bri/wi/ski/kitzski.html"),
    "Lech am Arlberg": (1444, 2811, "en.wikipedia.org/wiki/Lech_(Vorarlberg) village 1,444 m; "
                                    "Ski Arlberg top (Valluga) 2,811 m per en.wikipedia.org/wiki/Ski_Arlberg"),
    "Pitztal Glacier": (1740, 3440, "pitztal.com/en/pitztaler-glacier-rifflsee-ski-resort.html"),
    "Obertauern": (1630, 2313, "skiresort.com/en/ski-resorts/schladming-tauern (Obertauern 1,630-2,313 m)"),
    "Zell am See": (757, 2000, "snow-online.com/ski-resort/schmittenhoehe.html"),
    "Schladming": (750, 2015, "en.wikipedia.org/wiki/Schladminger_4-Berge-Schaukel"),
    # Switzerland
    "Zermatt": (1620, 3883, "valais.ch/en/explore/activities/ski-snowboard/ski-areas/zermatt-matterhorn"),
    "Saas-Fee": (1800, 3600, "valais.ch/en/explore/activities/ski-snowboard/ski-areas/saas-fee"),
    "Verbier": (1500, 3330, "en.wikipedia.org/wiki/Verbier (village 1,500 m, Mont Fort 3,330 m)"),
    "St. Moritz": (1772, 3057, "skiresort.com/en/ski-resorts/swiss-alps (St. Moritz Corviglia 1,772-3,057 m)"),
    "Davos": (1560, 2844, "en.wikipedia.org/wiki/Davos (Davos 1,560 m); Parsenn top 2,844 m per snowtrex/skiresort.info"),
    "Laax": (1100, 3018, "skiresort.info/ski-resorts/graubuenden (Laax 1,100-3,018 m, Vorab)"),
    "Lenzerheide": (1229, 2865, "skiresort.com/en/ski-resort/arosa-lenzerheide (1,229-2,865 m)"),
    "Mürren": (1650, 2970, "deinskigebiet.ch/en/ski-resorts/muerren-schilthorn (1,650-2,970 m)"),
    "Andermatt": (1444, 2961, "skiresort.info/ski-resorts/switzerland (Andermatt-Gemsstock 1,444-2,961 m)"),
    "Arosa": (1800, 2653, "myswitzerland.com/en-us/experiences/weisshorn (Arosa 1,800 m, Weisshorn 2,653 m)"),
    "Engelberg": (1003, 3020, "skiresort.info/ski-resorts/switzerland (Titlis-Engelberg 1,003-3,020 m)"),
    # Japan
    # https://en.wikipedia.org/wiki/Niseko_Mt._Resort_Grand_Hirafu
    "Niseko United": (260, 1200, "en.wikipedia.org (base 260 m, lift-served top 1,200 m)"),
    # https://www.skiresort.com/en/ski-resort/furano/
    "Furano": (245, 1074, "skiresort.com (base 245 m, lift-served top 1,074 m)"),
    # https://www.skiresort.com/en/ski-resort/rusutsu/
    "Rusutsu": (400, 994, "skiresort.com (base 400 m, lift-served top 994 m)"),
    # https://www.kiroro.co.jp/media/brochure/pdf-1035.pdf
    "Kiroro": (520, 1180, "kiroro.co.jp (base 520 m, lift-served top 1,180 m)"),
    # https://www.skiresort.com/en/ski-resort/happo-one-hakuba/
    "Hakuba Valley": (760, 1831, "skiresort.com (base 760 m, lift-served top 1,831 m)"),
    "Happo-One": (760, 1831, "en.wikipedia.org/wiki/Hakuba_Happo-one_Snow_Resort (base 760 m, top 1,831 m)"),
    # https://www.skiresort.com/en/ski-resort/nozawa-onsen/
    "Nozawa Onsen": (565, 1650, "skiresort.com (base 565 m, lift-served top 1,650 m)"),
    # https://www.shigakogen-ski.or.jp/lift/yokoteyama-shibutoge/index-en.html
    "Shiga Kogen": (1300, 2307, "shigakogen-ski.or.jp (base 1,300 m, lift-served top 2,307 m)"),
    # https://www.princehotels.com/en/ski/myoko_kogen/
    "Myoko Kogen": (731, 1855, "princehotels.com (base 731 m, lift-served top 1,855 m)"),
    # https://www.skiresort.com/en/ski-resort/zao-onsen/
    "Zao Onsen": (780, 1661, "skiresort.com (base 780 m, lift-served top 1,661 m)"),
    # France
    # https://en.wikipedia.org/wiki/Val_Thorens
    "Val Thorens": (2300, 3230, "en.wikipedia.org (base 2,300 m, lift-served top 3,230 m)"),
    # https://www.skiresort.com/en/ski-resort/les-arcs-peisey-vallandry-paradiski/
    "Les Arcs": (1200, 3226, "skiresort.com (base 1,200 m, lift-served top 3,226 m)"),
    # https://www.valdisere.ski/en/top-picks-pisaillas
    "Val d'Isère": (1850, 3197, "valdisere.ski (base 1,850 m, lift-served top 3,197 m)"),
    # https://snowcarbon.co.uk/ski-resorts/les-menuires/les-menuires-slopes
    "Les Menuires": (1850, 2850, "snowcarbon.co.uk (base 1,850 m, lift-served top 2,850 m)"),
    # https://en.wikipedia.org/wiki/Avoriaz
    "Avoriaz": (1800, 2466, "en.wikipedia.org (base 1,800 m, lift-served top 2,466 m)"),
    # https://www.mon-sejour-en-montagne.com/actualites/a-la-plagne-une-nouvelle-telecabine-pour-monter-a-3080-metres/
    "La Plagne": (1250, 3080, "mon-sejour-en-montagne.com (base 1,250 m, lift-served top 3,080 m)"),
    # https://en.wikipedia.org/wiki/Courchevel
    "Courchevel": (1300, 2738, "en.wikipedia.org (base 1,300 m, lift-served top 2,738 m)"),
    # https://www.skiresort.com/en/ski-resort/grands-montets-argentiere-chamonix/
    # Base kept at Chamonix town (1,035 m), which is the resort's forecast coordinate and the
    # Brévent lift base; 1,252 m is Argentière, a different village in the valley.
    "Chamonix": (1035, 2765, "skiresort.com (town base 1,035 m, lift-served top 2,765 m)"),
    # Unverified. Best source found (low confidence): https://www.seemorzine.com/morzine/ski-area/
    "Morzine": (1000, 2000, "summit unverified: carried over from elevation_bands.RESORT_VERTICAL_M; best source found (seemorzine.com, low confidence) gives 2,002 m"),
    # https://www.skiresort.com/en/ski-resort/tignes-val-disere/
    "Tignes": (1550, 3456, "skiresort.com (base 1,550 m, lift-served top 3,456 m)"),
    # https://en.wikipedia.org/wiki/M%C3%A9ribel
    "Méribel": (1100, 2952, "en.wikipedia.org (base 1,100 m, lift-served top 2,952 m)"),
    # https://www.skiresort.com/en/ski-resort/megeve-saint-gervais/
    "Megève": (1113, 2353, "skiresort.com (base 1,113 m, lift-served top 2,353 m)"),
    # https://www.skiresort.com/en/ski-resort/la-clusaz-manigod/
    "La Clusaz": (1050, 2485, "skiresort.com (base 1,050 m, lift-served top 2,485 m)"),
    # https://fr.wikipedia.org/wiki/Grandes_Plati%C3%A8res
    "Flaine": (1600, 2480, "fr.wikipedia.org (base 1,600 m, lift-served top 2,480 m)"),
    # https://en.wikipedia.org/wiki/Mont_Ch%C3%A9ry
    "Les Gets": (1172, 1826, "en.wikipedia.org (base 1,172 m, lift-served top 1,826 m)"),
    # https://www.skiresort.com/en/ski-resort/alpe-dhuez/
    "Alpe d'Huez": (1125, 3330, "skiresort.com (base 1,125 m, lift-served top 3,330 m)"),
    # https://www.les2alpes.com/winter/discover/skiing-in-les-2-alpes/ski-area/
    "Les 2 Alpes": (1280, 3600, "les2alpes.com (base 1,280 m, lift-served top 3,600 m)"),
    # https://www.skiresort.com/en/ski-resort/serre-chevalier-briancon-chantemerle-villeneuve-la-salle-le-monetier-les-bains/
    "Serre Chevalier": (1200, 2800, "skiresort.com (base 1,200 m, lift-served top 2,800 m)"),
    "Vars": (1650, 2750, "skiresort.com/en/ski-resort/vars-risoul-la-foret-blanche (Forêt Blanche 1,650-2,750 m)"),
    # https://www.skideals.fr/station/isola-2000/infos
    "Isola 2000": (1800, 2610, "skideals.fr (base 1,800 m, lift-served top 2,610 m)"),
    # Spain
    # https://www.skiresort.com/en/ski-resort/baqueira-beret/
    "Baqueira-Beret": (1500, 2610, "skiresort.com (base 1,500 m, lift-served top 2,610 m)"),
    # https://en.wikipedia.org/wiki/Formigal_Ski_Resort
    "Formigal": (1500, 2250, "en.wikipedia.org (base 1,500 m, lift-served top 2,250 m)"),
    # https://en.wikipedia.org/wiki/Bo%C3%AD_Ta%C3%BCll_Resort
    "Boí Taüll": (2020, 2751, "en.wikipedia.org (base 2,020 m, lift-served top 2,751 m)"),
    # https://www.candanchu.com/
    "Candanchú": (1560, 2400, "candanchu.com (base 1,560 m, lift-served top 2,400 m)"),
    # https://en.wikipedia.org/wiki/La_Molina_(ski_resort)
    "La Molina": (1700, 2537, "en.wikipedia.org (base 1,700 m, lift-served top 2,537 m)"),
    # https://www.masella.com/
    "Masella": (1600, 2535, "masella.com (base 1,600 m, lift-served top 2,535 m)"),
    # https://www.skiresort.com/en/ski-resort/cerler/
    "Cerler": (1500, 2630, "skiresort.com (base 1,500 m, lift-served top 2,630 m)"),
    # https://www.lugaresdenieve.com/?q=es%2Festacion%2Fastun
    "Astún": (1700, 2300, "lugaresdenieve.com (base 1,700 m, lift-served top 2,300 m)"),
    # https://www.lugaresdenieve.com/?q=es%2Festacion%2Fpanticosa-aramon
    "Panticosa": (1500, 2220, "lugaresdenieve.com (base 1,500 m, lift-served top 2,220 m)"),
    # https://pirineu365.cat/wp-content/uploads/2025/10/Fitxa-Tecnica_Vallter-25-26_V2.docx.pdf
    "Vallter 2000": (1959, 2535, "pirineu365.cat (base 1,959 m, lift-served top 2,535 m)"),
    # https://en.wikipedia.org/wiki/Port_Ain%C3%A9
    "Port Ainé": (1650, 2440, "en.wikipedia.org (base 1,650 m, lift-served top 2,440 m)"),
    # Unverified. Best source found (low confidence): https://pirineu365.cat/en/espotesqui/resort/trail-map/
    "Espot Esquí": (1510, 2110, "summit unverified: carried over from elevation_bands.RESORT_VERTICAL_M; best source found (pirineu365.cat, low confidence) gives 2,500 m"),
    # https://www.skiresort.com/en/ski-resort/sierra-nevada-pradollano/
    "Sierra Nevada": (2100, 3300, "skiresort.com (base 2,100 m, lift-served top 3,300 m)"),
}

# =============================================================================
# OnTheSnow report pages: final URLs after the one-time redirect check
# (27 Nov 2026). European slugs redirect to onthesnow.co.uk. Japanese entries
# that returned 404 (happo-one, nozawa-onsen, shiga-kogen, myoko-kogen,
# zao-onsen) were removed: OnTheSnow lists only Hakuba Valley in Nagano.
# =============================================================================
_OTS = "https://www.onthesnow.com/"
_OTS_UK = "https://www.onthesnow.co.uk/"
ONTHESNOW_URLS = {
    # USA
    "Stevens Pass": _OTS + "washington/stevens-pass-resort/skireport",
    "Crystal Mountain": _OTS + "washington/crystal-mountain-wa/skireport",
    "Mt. Baker": _OTS + "washington/mt-baker/skireport",
    "Snoqualmie Pass": _OTS + "washington/the-summit-at-snoqualmie/skireport",
    "Mt. Hood Meadows": _OTS + "oregon/mt-hood-meadows/skireport",
    "Mt. Bachelor": _OTS + "oregon/mt-bachelor/skireport",
    "Timberline": _OTS + "oregon/timberline-lodge/skireport",
    "Mammoth Mountain": _OTS + "california/mammoth-mountain-ski-area/skireport",
    "Palisades Tahoe": _OTS + "california/palisades-tahoe/skireport",
    "Heavenly": _OTS + "california/heavenly-mountain-resort/skireport",
    "Kirkwood": _OTS + "california/kirkwood/skireport",
    "Vail": _OTS + "colorado/vail/skireport",
    "Breckenridge": _OTS + "colorado/breckenridge/skireport",
    "Aspen Snowmass": _OTS + "colorado/aspen-snowmass/skireport",
    "Telluride": _OTS + "colorado/telluride/skireport",
    "Park City": _OTS + "utah/park-city-mountain-resort/skireport",
    "Snowbird": _OTS + "utah/snowbird/skireport",
    "Alta": _OTS + "utah/alta-ski-area/skireport",
    "Big Sky": _OTS + "montana/big-sky-resort/skireport",
    "Whitefish": _OTS + "montana/whitefish-mountain-resort/skireport",
    "Jackson Hole": _OTS + "wyoming/jackson-hole/skireport",
    "Sun Valley": _OTS + "idaho/sun-valley/skireport",
    # Japan
    "Niseko United": _OTS + "hokkaido/niseko-united/skireport",
    "Furano": _OTS + "hokkaido/furano-ski-resort/skireport",
    "Rusutsu": _OTS + "hokkaido/rusutsu/skireport",
    "Kiroro": _OTS + "hokkaido/kiroro/skireport",
    "Hakuba Valley": _OTS + "nagano/hakuba-valley/skireport",
    # France
    "Val Thorens": _OTS_UK + "northern-alps/val-thorens/skireport",
    "Les Arcs": _OTS_UK + "northern-alps/les-arcs-bourg-st-maurice/skireport",
    "Val d'Isère": _OTS_UK + "northern-alps/val-disere/skireport",
    "Les Menuires": _OTS_UK + "northern-alps/les-menuires/skireport",
    "Avoriaz": _OTS_UK + "northern-alps/avoriaz/skireport",
    "La Plagne": _OTS_UK + "northern-alps/la-plagne/skireport",
    "Courchevel": _OTS_UK + "northern-alps/courchevel/skireport",
    "Chamonix": _OTS_UK + "northern-alps/chamonix-mont-blanc/skireport",
    "Morzine": _OTS_UK + "northern-alps/morzine/skireport",
    "Tignes": _OTS_UK + "northern-alps/tignes/skireport",
    "Méribel": _OTS_UK + "northern-alps/meribel/skireport",
    "Megève": _OTS_UK + "northern-alps/megeve/skireport",
    "La Clusaz": _OTS_UK + "northern-alps/la-clusaz/skireport",
    "Flaine": _OTS_UK + "northern-alps/flaine/skireport",
    "Les Gets": _OTS_UK + "northern-alps/les-gets/skireport",
    "Alpe d'Huez": _OTS_UK + "northern-alps/alpe-dhuez/skireport",
    "Les 2 Alpes": _OTS_UK + "northern-alps/les-2-alpes/skireport",
    "Serre Chevalier": _OTS_UK + "southern-alps/serre-chevalier/skireport",
    "Vars": _OTS_UK + "southern-alps/vars/skireport",
    "Isola 2000": _OTS_UK + "southern-alps/isola-2000/skireport",
    # Spain
    "Baqueira-Beret": _OTS_UK + "pirineo-catalan/baqueira-beret/skireport",
    "Formigal": _OTS_UK + "aragon/formigal/skireport",
    "Boí Taüll": _OTS_UK + "pirineo-catalan/boi-tauell/skireport",
    "Candanchú": _OTS_UK + "aragon/candanchu/skireport",
    "La Molina": _OTS_UK + "pirineo-catalan/la-molina/skireport",
    "Masella": _OTS + "spain/masella/skireport",  # timed out during the slug check; follows redirect at run time
    "Cerler": _OTS_UK + "aragon/cerler/skireport",
    "Astún": _OTS_UK + "aragon/astun/skireport",
    "Panticosa": _OTS_UK + "aragon/panticosa/skireport",
    "Vallter 2000": _OTS_UK + "pirineo-catalan/vallter-2000/skireport",
    "Port Ainé": _OTS_UK + "pirineo-catalan/port-aine/skireport",
    "Espot Esquí": _OTS_UK + "pirineo-catalan/espot-esqui/skireport",
    "Sierra Nevada": _OTS_UK + "sistema-penibetico/sierra-nevada/skireport",
}

ONTHESNOW_REMOVED_NOTE = {
    "Happo-One": "OnTheSnow has no page for Happo-One (404); Hakuba Valley's report is not reused",
    "Nozawa Onsen": "OnTheSnow has no page for Nozawa Onsen (404)",
    "Shiga Kogen": "OnTheSnow has no page for Shiga Kogen (404)",
    "Myoko Kogen": "OnTheSnow has no page for Myoko Kogen (404)",
    "Zao Onsen": "OnTheSnow has no page for Zao Onsen (404)",
}

# tenki.jp resort pages. Checked 27 Nov 2026: Rusutsu (1/2/11718), Kiroro
# (1/2/11711) and Happo-One (3/23/15100) return 404, and Hakuba Valley's id
# 3/23/15101 is Kurohime Kogen Snow Park, a different resort. Those are removed.
TENKI_JP_RESORTS = {
    "Niseko United": "1/2/11708",      # Niseko Tokyu Grand Hirafu
    "Furano": "1/2/11717",             # Furano Ski Resort
    "Nozawa Onsen": "3/23/15115",      # Nozawa Onsen
    "Shiga Kogen": "3/23/15123",       # Shiga Kogen Yakebitaiyama
    "Myoko Kogen": "4/18/13142",       # Myoko Suginohara
    "Zao Onsen": "2/9/15191",          # Zao Onsen
}
TENKI_REMOVED_NOTE = {
    "Rusutsu": "tenki.jp page for Rusutsu returns 404, not used",
    "Kiroro": "tenki.jp page for Kiroro returns 404, not used",
    "Happo-One": "tenki.jp page for Happo-One returns 404, not used",
    "Hakuba Valley": "tenki.jp id listed for Hakuba Valley is Kurohime Kogen, not used",
}
TENKI_DEPTH_RE = re.compile(r'class="depth">積雪<br><span>(\d+)</span>cm')
TENKI_OFF_SEASON_RE = re.compile(r"スキー積雪情報の更新は終了しました")

SHARED_COORD_NOTE = ("Hakuba Valley and Happo-One share coordinates in this list, "
                     "so their forecasts are identical")

# =============================================================================
# Small helpers
# =============================================================================


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def r1(x):
    return None if x is None else round(float(x), 1)


def rint(x):
    return None if x is None else int(math.floor(float(x) + 0.5))


def slugify(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def norm_name(s: str) -> str:
    folded = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def tri_and(*vals):
    """Three-valued AND: False if any known False, True if all True, else None."""
    if any(v is False for v in vals):
        return False
    if all(v is True for v in vals):
        return True
    return None


def tri_or(*vals):
    """Three-valued OR: True if any True, False if all known False, else None."""
    if any(v is True for v in vals):
        return True
    if all(v is False for v in vals):
        return False
    return None


def gt(a, b):
    return None if a is None or b is None else a > b


def strict_sum(values):
    vals = list(values)
    if not vals or any(v is None for v in vals):
        return None
    return sum(vals)


# =============================================================================
# HTTP
# =============================================================================


class HttpCounter:
    """Request and retry counts. Locked, because the fetch chains run in threads."""

    def __init__(self):
        self.requests = 0
        self.retries = 0
        self._lock = threading.Lock()

    def count(self, retry):
        with self._lock:
            self.requests += 1
            if retry:
                self.retries += 1


OM_COUNTER = HttpCounter()
OTHER_COUNTER = HttpCounter()

# Time budget. The workflow kills the job at 25 minutes and then nothing is written, so the
# fetcher stops itself first. RUN_BUDGET_S covers the whole run; the measured-source and
# resort-report phases must leave OM_RESERVE_S of it for Open-Meteo. Raise
# FETCH_RUN_BUDGET_S together with the workflow's timeout-minutes (resort expansion plans).
RUN_BUDGET_S = float(os.environ.get("FETCH_RUN_BUDGET_S") or 22 * 60)
OM_RESERVE_S = float(os.environ.get("FETCH_OM_RESERVE_S") or 10 * 60)
_RUN_DEADLINE = None     # time.monotonic() value; None means no budget (library use, tests)
_PHASE_DEADLINE = None   # tighter deadline for the current phase, or None
# The fetch chains run in threads, and only the measured-source and report chains have a
# phase deadline, so a thread can carry its own (set by _run_chain), overriding the global.
_THREAD = threading.local()


def _deadline():
    phase = getattr(_THREAD, "phase_deadline", _PHASE_DEADLINE)
    ds = [d for d in (_RUN_DEADLINE, phase) if d is not None]
    return min(ds) if ds else None


def time_left():
    """Seconds left in the current phase's budget (inf when no budget is set)."""
    d = _deadline()
    return math.inf if d is None else d - time.monotonic()


# Headers that carry no secret and may follow a redirect to another host.
_PORTABLE_HEADERS = {"user-agent", "accept", "accept-language"}
_MAX_RETRY_AFTER_S = 300


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to http(s), and drop the caller's own headers (AEMET's
    api_key) when a redirect changes host or goes from https to http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_u, new_u = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(new.full_url)
        if new_u.scheme not in ("http", "https"):
            raise urllib.error.HTTPError(newurl, code, "redirect to a non-HTTP URL refused", headers, fp)
        if ((new_u.hostname or "").lower() != (old_u.hostname or "").lower()
                or (old_u.scheme == "https" and new_u.scheme != "https")):
            for k in list(new.headers):
                if k.lower() not in _PORTABLE_HEADERS:
                    del new.headers[k]
        return new


def _retry_after_s(e):
    v = (e.headers or {}).get("Retry-After") if getattr(e, "headers", None) is not None else None
    if not v:
        return None
    try:
        return max(0.0, min(float(v.strip()), _MAX_RETRY_AFTER_S))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        when = parsedate_to_datetime(v)
        return max(0.0, min((when - datetime.now(timezone.utc)).total_seconds(), _MAX_RETRY_AFTER_S))
    except (TypeError, ValueError, IndexError):
        return None


def _error_reason(e):
    """Short reason from an HTTP error body, for the log only (never headers)."""
    try:
        txt = e.read(400).decode("utf-8", "replace")
    except Exception:
        return None
    try:
        j = json.loads(txt)
        if isinstance(j, dict) and j.get("reason"):
            return str(j["reason"])[:160]
    except ValueError:
        pass
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt)).strip()
    return txt[:160] or None


def _read_body(resp, deadline):
    """Read the whole body in small pieces, stopping at the wall-clock deadline.

    A per-read socket timeout alone never fires against a server that keeps sending a
    few bytes, so the deadline is checked between reads."""
    sock = getattr(getattr(resp.fp, "raw", None), "_sock", None)
    parts = []
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("read deadline passed before the body was complete")
        if sock is not None:
            try:
                sock.settimeout(max(0.05, left))
            except OSError:
                pass
        chunk = resp.read1(65536)
        if not chunk:
            break
        parts.append(chunk)
    raw = b"".join(parts)
    if resp.length:   # Content-Length promised more than arrived
        raise http.client.IncompleteRead(raw, resp.length)
    return raw


def _fetch(url, *, attempts, backoff_s, timeout_s, headers, counter, want_json,
           wait_on_429=None, want_bytes=False):
    """Fetch a URL with retries. Returns (body, final_url) or (None, error_text).

    timeout_s bounds each attempt from connect to the last byte (wall clock), and no
    attempt or retry wait runs past the run's time budget. 4xx answers other than 408
    and 429 are not retried. Header values are never logged (AEMET passes its key as a
    header), and custom headers are dropped on a redirect to another host."""
    if urllib.parse.urlsplit(url or "").scheme not in ("http", "https"):
        return None, "refused non-HTTP URL"
    hdrs = {"User-Agent": API_UA}
    if headers:
        hdrs.update(headers)
    opener = urllib.request.build_opener(_SafeRedirect)
    last_err = None
    for attempt in range(attempts):
        hard = _deadline()
        now = time.monotonic()
        if hard is not None and now >= hard:
            last_err = (last_err + "; " if last_err else "") + "time budget used up"
            break
        counter.count(retry=attempt > 0)
        got_429 = False
        retry_after = None
        no_retry = False
        att_deadline = now + timeout_s if hard is None else min(now + timeout_s, hard)
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with opener.open(req, timeout=max(0.05, min(timeout_s, att_deadline - now))) as resp:
                raw = _read_body(resp, att_deadline)
                final_url = resp.geturl()
            if want_bytes:
                return raw, final_url
            text = raw.decode("utf-8", "replace")
            if not want_json:
                return text, final_url
            data = json.loads(text)
            if isinstance(data, dict) and data.get("error") is True:
                raise ValueError(f"API error: {data.get('reason')}")
            return data, final_url
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            got_429 = e.code == 429
            retry_after = _retry_after_s(e)
            reason = _error_reason(e)
            if reason:
                logger.info(f"  HTTP {e.code} reason: {reason} ({url[:90]})")
            # A 4xx other than timeout/rate limit is the same on every retry.
            no_retry = 400 <= e.code < 500 and e.code not in (408, 429)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                http.client.HTTPException) as e:
            last_err = f"{type(e).__name__}: {e}"
        if no_retry:
            break
        if attempt < attempts - 1:
            wait = backoff_s[min(attempt, len(backoff_s) - 1)] if backoff_s else 0
            if got_429 and wait_on_429 is not None:
                wait = max(wait, wait_on_429)
            if retry_after is not None:
                wait = max(wait, retry_after)
            hard = _deadline()
            if hard is not None and time.monotonic() + wait >= hard:
                last_err += "; no time left in the run budget to retry"
                break
            logger.info(f"  retry in {wait}s after {last_err} ({url[:90]})")
            time.sleep(wait)
    return None, last_err


def http_json(url, *, attempts, backoff_s, timeout_s, headers=None, counter=None,
              wait_on_429=None):
    """GET JSON. Returns parsed body, or None on failure (error logged)."""
    body, info = _fetch(url, attempts=attempts, backoff_s=backoff_s, timeout_s=timeout_s,
                        headers=headers, counter=counter or OTHER_COUNTER, want_json=True,
                        wait_on_429=wait_on_429)
    if body is None:
        logger.warning(f"Request failed after {attempts} attempt(s): {info} ({url[:120]})")
    return body


def http_text(url, *, attempts, backoff_s, timeout_s, headers=None):
    """GET text. Returns (text, final_url) or (None, error)."""
    return _fetch(url, attempts=attempts, backoff_s=backoff_s, timeout_s=timeout_s,
                  headers=headers, counter=OTHER_COUNTER, want_json=False)


def http_bytes(url, *, attempts, backoff_s, timeout_s, headers=None):
    """GET raw bytes. Returns (bytes, final_url) or (None, error)."""
    return _fetch(url, attempts=attempts, backoff_s=backoff_s, timeout_s=timeout_s,
                  headers=headers, counter=OTHER_COUNTER, want_json=False, want_bytes=True)


# =============================================================================
# Resort list
# =============================================================================


def build_resorts():
    """One list of resort dicts in output order."""
    out = []

    def base(name, lat, lon, region, country):
        if name not in RESORT_ELEVATIONS:
            raise KeyError(f"RESORT_ELEVATIONS has no entry for {name}")
        base_m, summit_m, note = RESORT_ELEVATIONS[name]
        mid_m = rint((base_m + summit_m) / 2) if summit_m is not None else None
        return {
            "name": name, "id": slugify(name), "country": country, "region": region,
            "lat": lat, "lon": lon,
            "base_m": int(base_m), "summit_m": summit_m, "mid_m": mid_m, "elev_note": note,
            # (provider, id, label, station_type) in priority order; the index is the row's _rank
            "station_cfg": [],
            "ots_url": ONTHESNOW_URLS.get(name),
            "tenki": TENKI_JP_RESORTS.get(name) if country == "Japan" else None,
            "notes": [],
        }

    for name, lat, lon, region, _elev, stations in US_SKI_AREAS:
        r = base(name, lat, lon, region, "USA")
        r["station_cfg"] = [("SNOTEL", t[0], t[1], None) for t in stations]
        if not stations:
            r["notes"].append("No SNOTEL station with a verified triplet is configured for this resort")
        out.append(r)
    for name, lat, lon, region, _elev, stations in AUSTRIA_SKI_AREAS:
        r = base(name, lat, lon, region, "Austria")
        r["station_cfg"] = [(p, sid, label, None) for p, sid, label in stations]
        out.append(r)
    for name, lat, lon, region, _elev, stations in SWISS_SKI_AREAS:
        r = base(name, lat, lon, region, "Switzerland")
        r["station_cfg"] = [("SLF", sid, label, stype) for sid, label, stype in stations]
        out.append(r)
    for country, areas in (("Japan", JAPAN_SKI_AREAS), ("France", FRANCE_SKI_AREAS),
                           ("Spain", SPAIN_SKI_AREAS)):
        for name, lat, lon, region, _elev in areas:
            r = base(name, lat, lon, region, country)
            cfg = [("JMA", sid, label, None) for sid, label in JMA_STATIONS.get(name, [])]
            cfg += [("Meteocat", sid, label, None) for sid, label in METEOCAT_STATIONS.get(name, [])]
            cfg += [("Meteo-France", sid, label, stype)
                    for sid, label, stype in METEO_FRANCE_STATIONS.get(name, [])]
            if name in AEMET_RESORTS:
                cfg.append(("AEMET", "nearest", "nearest AEMET station reporting snow depth", None))
            r["station_cfg"] = cfg
            out.append(r)

    for r in out:
        if r["name"] in DEPTH_COVERAGE_NOTES:
            r["notes"].append(DEPTH_COVERAGE_NOTES[r["name"]])
        if r["name"] in ("Hakuba Valley", "Happo-One"):
            r["notes"].append(SHARED_COORD_NOTE)
        if r["name"] in ONTHESNOW_REMOVED_NOTE:
            r["notes"].append(ONTHESNOW_REMOVED_NOTE[r["name"]])
        if r["name"] in TENKI_REMOVED_NOTE:
            r["notes"].append(TENKI_REMOVED_NOTE[r["name"]])
        if r["summit_m"] is None:
            r["notes"].append("Summit elevation not sourced, summit forecast skipped")

    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("resort ids are not unique")
    return out


# =============================================================================
# Measured sources
# =============================================================================


def _cfg(r, provider):
    """[(rank, station_id, label, station_type)] for one provider, rank = index in the resort's config."""
    return [(i, sid, label, stype) for i, (p, sid, label, stype) in enumerate(r["station_cfg"]) if p == provider]


def _provider_down(provider, resorts, warnings, provider_status, warning, why):
    """A provider outage: status 'failed', one run warning, and a note on each resort that uses it.

    Used when a request fails and when a 200 answer is empty or the wrong shape, so an
    outage is not written up as every station having been removed from the metadata."""
    warnings.append(warning)
    provider_status[provider] = "failed"
    for r in resorts:
        for _, sid, label, _ in _cfg(r, provider):
            r["notes"].append(f"{provider} {sid} not available this run ({why})")


def _list_of_dicts(x, nonempty=True):
    return isinstance(x, list) and (bool(x) or not nonempty) and all(isinstance(i, dict) for i in x)


def _plausible_depth(v):
    """Station snow depth in cm within the same 0 to 600 cm window the report parsers use."""
    return v is not None and 0 <= v < MAX_REPORT_DEPTH_CM


def _station_row(provider, sid, name, lat, lon, elev_m, resort, rank=0, age_key=None):
    return {
        "provider": provider, "id": sid, "name": name,
        "lat": round(float(lat), 5), "lon": round(float(lon), 5),
        "elevation_m": rint(elev_m),
        "distance_km": round(haversine_km(resort["lat"], resort["lon"], lat, lon), 1),
        "depth_cm": None, "observed_utc": None, "observed_date": None,
        "change_24h_cm": None, "change_48h_cm": None, "change_72h_cm": None,
        "new_24h_cm": None, "new_48h_cm": None, "new_72h_cm": None,
        "swe_change_24h_mm": None, "swe_mm": None, "series": [],
        # private, stripped before writing: config priority and staleness rule key
        "_rank": rank, "_age_key": age_key or provider,
    }


def _station_rule_failure(row, resort):
    """Plan 0.1: distance and elevation-window checks. Returns the reason, or None if accepted."""
    label = f"{row['provider']} {row['id']} ({row['name']})"
    if row["distance_km"] > MAX_STATION_DISTANCE_KM:
        return f"{label} is {row['distance_km']} km away, over {int(MAX_STATION_DISTANCE_KM)} km, removed"
    elev = row["elevation_m"]
    if elev is not None:
        base = resort["base_m"]
        top = resort["summit_m"] if resort["summit_m"] is not None else base + 1500
        if elev < base - STATION_ELEV_BELOW_BASE_M:
            return (f"{label} at {elev:,} m is {base - elev:,} m below the {base:,} m base "
                    f"(limit {STATION_ELEV_BELOW_BASE_M} m), removed")
        if elev > top + STATION_ELEV_ABOVE_TOP_M:
            return (f"{label} at {elev:,} m is {elev - top:,} m above the {top:,} m summit "
                    f"(limit {STATION_ELEV_ABOVE_TOP_M} m), removed")
    return None


def _accept_station(row, resort, warnings):
    msg = _station_rule_failure(row, resort)
    if msg:
        resort["notes"].append(msg)
        logger.warning(f"{resort['name']}: {msg}")
        return False
    return True


def _set_latest(row, t1, v1):
    row["depth_cm"] = r1(v1)
    row["observed_utc"] = utc_iso(t1)
    row["observed_date"] = t1.astimezone(timezone.utc).date().isoformat()


def _depth_changes(row, readings, max_gap):
    """change_24/48/72 from a sorted list of (datetime, depth_cm); nearest reading within max_gap."""
    t1, v1 = readings[-1]
    for n in (24, 48, 72):
        v0 = _closest_reading(readings, t1 - timedelta(hours=n), max_gap)
        row[f"change_{n}h_cm"] = r1(v1 - v0) if v0 is not None else None


def _sum_chain(first, *rest):
    """h24, h48, h72 from consecutive period totals; each total is null once any period is missing."""
    out = [first]
    for v in rest:
        out.append(None if out[-1] is None or v is None else out[-1] + v)
    return [r1(x) for x in out]


def snotel_batch(resorts, run_utc, warnings, provider_status):
    """SNOTEL via AWDB REST: one metadata request and one data request."""
    triplets = sorted({sid for r in resorts for _, sid, _, _ in _cfg(r, "SNOTEL")})
    if not triplets:
        return
    base_url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/"
    q = urllib.parse.quote(",".join(triplets), safe=",:")
    meta = http_json(f"{base_url}stations?stationTriplets={q}", attempts=3, backoff_s=[5, 15],
                     timeout_s=60)
    end_d = run_utc.date()
    begin_d = end_d - timedelta(days=15)
    data = http_json(
        f"{base_url}data?stationTriplets={q}&elements=SNWD,WTEQ&duration=DAILY"
        f"&beginDate={begin_d.isoformat()}&endDate={end_d.isoformat()}",
        attempts=3, backoff_s=[5, 15], timeout_s=90)
    if meta is None or data is None:
        _provider_down("SNOTEL", resorts, warnings, provider_status,
                       "SNOTEL (AWDB) request failed; no SNOTEL station data this run", "AWDB request failed")
        return
    # A 200 answer that is empty or the wrong shape is an outage, not every station removed.
    meta_by = {m.get("stationTriplet"): m for m in meta if isinstance(m, dict)} if isinstance(meta, list) else {}
    if not meta_by or not _list_of_dicts(data, nonempty=False):
        _provider_down("SNOTEL", resorts, warnings, provider_status,
                       "SNOTEL (AWDB) answered with empty or malformed data; no SNOTEL station data this run",
                       "AWDB answered with empty or malformed data")
        return
    series_by = {}
    for item in data:
        trip = item.get("stationTriplet")
        el = {}
        for block in item.get("data", []) or []:
            if not isinstance(block, dict):
                continue
            code = (block.get("stationElement") or {}).get("elementCode")
            vals = {}
            for v in block.get("values", []) or []:
                if isinstance(v, dict) and v.get("date") and _num(v.get("value")) is not None:
                    vals[v["date"][:10]] = float(v["value"])
            el[code] = vals
        series_by[trip] = el

    window = [end_d - timedelta(days=i) for i in range(14, -1, -1)]  # 15 days, ascending

    for r in resorts:
        for rank, sid, label, _ in _cfg(r, "SNOTEL"):
            m = meta_by.get(sid)
            if m is None:
                msg = f"SNOTEL {sid} not found in AWDB metadata, removed"
                r["notes"].append(msg)
                warnings.append(f"{r['name']}: {msg}")
                continue
            if norm_name(m.get("name", "")) != norm_name(label):
                msg = f"SNOTEL {sid} is {m.get('name')}, not {label}, removed"
                r["notes"].append(msg)
                warnings.append(f"{r['name']}: {msg}")
                logger.warning(f"{r['name']}: {msg}")
                continue
            elev_ft = m.get("elevation")
            row = _station_row("SNOTEL", sid, m.get("name"), m.get("latitude"), m.get("longitude"),
                               elev_ft * 0.3048 if elev_ft is not None else None, r, rank)
            if not _accept_station(row, r, warnings):
                continue
            el = series_by.get(sid, {})
            snwd_in = el.get("SNWD", {})
            wteq_in = el.get("WTEQ", {})
            D = {d: snwd_in[d] * 2.54 for d in snwd_in}  # cm
            bad = [d for d in D if not _plausible_depth(D[d])]
            if bad:
                r["notes"].append(f"SNOTEL {sid}: {len(bad)} snow-depth readings outside 0 to 600 cm dropped")
                D = {d: v for d, v in D.items() if d not in bad}
            row["series"] = [{"date": d.isoformat(), "depth_cm": r1(D.get(d.isoformat()))}
                             for d in window]
            dates_with = sorted(d for d in D if begin_d.isoformat() <= d <= end_d.isoformat())
            if dates_with:
                latest = dates_with[-1]
                ld = date.fromisoformat(latest)

                def diff(n, _ld=ld, _D=D, _latest=latest):
                    prev = (_ld - timedelta(days=n)).isoformat()
                    if prev not in _D:
                        return None
                    return r1(_D[_latest] - _D[prev])

                row["depth_cm"] = r1(D[latest])
                row["observed_date"] = latest
                row["change_24h_cm"] = diff(1)
                row["change_48h_cm"] = diff(2)
                row["change_72h_cm"] = diff(3)
                if latest in wteq_in:
                    row["swe_mm"] = r1(wteq_in[latest] * 25.4)
                    prev = (ld - timedelta(days=1)).isoformat()
                    if prev in wteq_in:
                        row["swe_change_24h_mm"] = r1((wteq_in[latest] - wteq_in[prev]) * 25.4)
            r.setdefault("stations", []).append(row)
    provider_status["SNOTEL"] = "ok"


def _closest_reading(readings, target, max_gap):
    """readings: list of (datetime, value) with non-null values. Closest to target."""
    best = None
    for t, v in readings:
        gap = abs((t - target).total_seconds())
        if best is None or gap < best[0]:
            best = (gap, t, v)
    if best is None or best[0] > max_gap.total_seconds():
        return None
    return best[2]


def geosphere_batch(resorts, run_utc, warnings, provider_status):
    ids = sorted({sid for r in resorts for _, sid, _, _ in _cfg(r, "GeoSphere")})
    if not ids:
        return
    end = run_utc.replace(second=0, microsecond=0)
    start = end - timedelta(hours=74)   # 74 h so that t1 - 72 h is inside the window
    fmt = "%Y-%m-%dT%H:%M"
    data = http_json(
        "https://dataset.api.hub.geosphere.at/v1/station/historical/tawes-v1-10min"
        f"?parameters=SCHNEE&station_ids={','.join(ids)}"
        f"&start={start.strftime(fmt)}&end={end.strftime(fmt)}",
        attempts=3, backoff_s=[5, 15], timeout_s=90)
    meta = http_json("https://dataset.api.hub.geosphere.at/v1/station/current/tawes-v1-10min/metadata",
                     attempts=3, backoff_s=[5, 15], timeout_s=60)
    if data is None or meta is None:
        _provider_down("GeoSphere", resorts, warnings, provider_status,
                       "GeoSphere request failed; no Austrian station data this run", "request failed")
        return
    if (not isinstance(meta, dict) or not _list_of_dicts(meta.get("stations"))
            or not isinstance(data, dict) or not isinstance(data.get("timestamps"), list)
            or not _list_of_dicts(data.get("features"), nonempty=False)):
        _provider_down("GeoSphere", resorts, warnings, provider_status,
                       "GeoSphere answered with empty or malformed data; no Austrian station data this run",
                       "GeoSphere answered with empty or malformed data")
        return
    meta_by = {str(s.get("id")): s for s in meta["stations"]}
    stamps = [datetime.fromisoformat(t) for t in data["timestamps"]]
    readings_by = {}
    dropped = {}
    for f in data["features"]:
        props = f.get("properties") or {}
        sid = str(props.get("station"))
        vals = ((props.get("parameters") or {}).get("SCHNEE") or {}).get("data") or []
        rd = [(t, _num(v)) for t, v in zip(stamps, vals) if _num(v) is not None]
        readings_by[sid] = [(t, v) for t, v in rd if _plausible_depth(v)]
        dropped[sid] = len(rd) - len(readings_by[sid])

    for r in resorts:
        for rank, sid, label, _ in _cfg(r, "GeoSphere"):
            m = meta_by.get(sid)
            if m is None:
                msg = f"GeoSphere station {sid} not in metadata, removed"
                r["notes"].append(msg)
                warnings.append(f"{r['name']}: {msg}")
                continue
            row = _station_row("GeoSphere", sid, m.get("name"), m.get("lat"), m.get("lon"),
                               m.get("altitude"), r, rank)
            if not _accept_station(row, r, warnings):
                continue
            readings = readings_by.get(sid, [])
            if dropped.get(sid):
                r["notes"].append(f"GeoSphere {m.get('name')}: {dropped[sid]} snow-depth readings "
                                  "outside 0 to 600 cm dropped")
            if readings:
                t1, v1 = readings[-1]
                _set_latest(row, t1, v1)
                _depth_changes(row, readings, timedelta(hours=1))
            else:
                r["notes"].append(f"GeoSphere {m.get('name')} reported no snow depth in the last 72 hours")
            r.setdefault("stations", []).append(row)
    provider_status["GeoSphere"] = "ok"


SLF_API = "https://measurement-api.slf.ch/public/api/"


def slf_once(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "SLF")]
    if not cfgs:
        return
    plot_codes = {sid for _, (_, sid, _, st) in cfgs if st == "study_plot"}
    meas = http_json(SLF_API + "imis/measurements", attempts=3, backoff_s=[5, 15], timeout_s=90)
    stations = http_json(SLF_API + "imis/stations", attempts=3, backoff_s=[5, 15], timeout_s=60)
    if meas is None or stations is None:
        _provider_down("SLF", resorts, warnings, provider_status,
                       "SLF request failed; no Swiss station data this run", "request failed")
        return
    if not _list_of_dicts(meas) or not _list_of_dicts(stations):
        _provider_down("SLF", resorts, warnings, provider_status,
                       "SLF answered with empty or malformed data; no Swiss station data this run",
                       "SLF answered with empty or malformed data")
        return
    codes = {sid for _, (_, sid, _, st) in cfgs if st != "study_plot"}
    meta_by = {s.get("code"): s for s in stations}
    readings_by = {}
    dropped = {}
    for m in meas:
        c = m.get("station_code")
        hs = _num(m.get("HS"))
        if c in codes and m.get("measure_date") and hs is not None:
            if not _plausible_depth(hs):
                dropped[c] = dropped.get(c, 0) + 1
                continue
            t = datetime.fromisoformat(m["measure_date"].replace("Z", "+00:00"))
            readings_by.setdefault(c, []).append((t, hs))
    for c in readings_by:
        readings_by[c].sort(key=lambda x: x[0])

    # New snow (HN_1D, cm per day) for IMIS stations, plan 2.2. SLF's API describes this
    # value as "1 day new snow (modeled by the snowpack model)", so it is published as
    # modelled, not measured (see recent_measured). Only study plots measure new snow.
    status_extra = []
    daily = http_json(SLF_API + "imis/daily-snow?period_in_days=7", attempts=3, backoff_s=[5, 15],
                      timeout_s=90)
    hn_by = {}
    if isinstance(daily, list):
        for d in daily:
            c = d.get("station_code")
            if c in codes and d.get("measure_date") and _num(d.get("HN_1D")) is not None:
                hn_by.setdefault(c, {})[d["measure_date"][:10]] = _num(d["HN_1D"])
    else:
        warnings.append("SLF daily new-snow request failed; Swiss stations show depth change only this run")
        status_extra.append("daily-snow failed")

    plot_meas, plot_meta = {}, {}
    if plot_codes:
        pm = http_json(SLF_API + "study-plot/measurements", attempts=3, backoff_s=[5, 15], timeout_s=60)
        ps = http_json(SLF_API + "study-plot/stations", attempts=3, backoff_s=[5, 15], timeout_s=60)
        if isinstance(pm, list) and isinstance(ps, list):
            plot_meta = {s.get("code"): s for s in ps}
            for m in pm:
                c = m.get("station_code")
                if c in plot_codes and m.get("measure_date"):
                    prev = plot_meas.get(c)
                    if prev is None or m["measure_date"] > prev["measure_date"]:
                        plot_meas[c] = m
        else:
            warnings.append("SLF study-plot request failed; study-plot stations missing this run")
            status_extra.append("study-plot failed")

    run_day = run_utc.date()
    for r in resorts:
        for rank, sid, label, stype in _cfg(r, "SLF"):
            stype = stype or "imis"
            if stype == "study_plot":
                if not plot_meta:
                    r["notes"].append(f"SLF study plot {sid} not available this run")
                    continue
                m = plot_meta.get(sid)
                if m is None:
                    msg = f"SLF study plot {sid} not in metadata, removed"
                    r["notes"].append(msg)
                    warnings.append(f"{r['name']}: {msg}")
                    continue
                row = _station_row("SLF", sid, m.get("label"), m.get("lat"), m.get("lon"),
                                   m.get("elevation"), r, rank, "SLF:study_plot")
                if not _accept_station(row, r, warnings):
                    continue
                pmv = plot_meas.get(sid)
                hs = _num((pmv or {}).get("HS"))
                if hs is not None and not _plausible_depth(hs):
                    r["notes"].append(f"SLF study plot {sid}: snow height {hs} cm outside 0 to 600 cm dropped")
                    hs = None
                if pmv and hs is not None:
                    t1 = datetime.fromisoformat(pmv["measure_date"].replace("Z", "+00:00"))
                    if t1.tzinfo is None:
                        t1 = t1.replace(tzinfo=timezone.utc)
                    _set_latest(row, t1, hs)
                    # The endpoint returns the latest day only, so 48 h and 72 h stay null.
                    hn = _num(pmv.get("HN_1D"))
                    row["new_24h_cm"] = r1(hn) if hn is not None and hn >= 0 else None
                else:
                    r["notes"].append(f"SLF study plot {m.get('label')} ({sid}) has no snow height reading today")
                r.setdefault("stations", []).append(row)
                continue

            m = meta_by.get(sid)
            if m is None:
                msg = f"SLF station {sid} not in metadata, removed"
                r["notes"].append(msg)
                warnings.append(f"{r['name']}: {msg}")
                continue
            row = _station_row("SLF", sid, m.get("label"), m.get("lat"), m.get("lon"),
                               m.get("elevation"), r, rank, "SLF:imis")
            if not _accept_station(row, r, warnings):
                continue
            readings = readings_by.get(sid, [])
            if dropped.get(sid):
                r["notes"].append(f"SLF {m.get('label')} ({sid}): {dropped[sid]} snow-height readings "
                                  "outside 0 to 600 cm dropped")
            if readings:
                t1, v1 = readings[-1]
                _set_latest(row, t1, v1)
                # The bulk feed covers 24 hours, so only the 24 h change is possible.
                v0 = _closest_reading(readings, t1 - timedelta(hours=24), timedelta(hours=2))
                row["change_24h_cm"] = r1(v1 - v0) if v0 is not None else None
            else:
                r["notes"].append(f"SLF {m.get('label')} ({sid}) has no snow height reading in the current feed")
            hn = hn_by.get(sid, {})
            if hn:
                latest = max(hn)
                ld = date.fromisoformat(latest)
                if (run_day - ld).days <= 1:
                    days = [hn.get((ld - timedelta(days=i)).isoformat()) for i in range(3)]
                    if all(v is None or v >= 0 for v in days):
                        row["new_24h_cm"], row["new_48h_cm"], row["new_72h_cm"] = _sum_chain(*days)
            r.setdefault("stations", []).append(row)
    provider_status["SLF"] = "ok" + (f" ({', '.join(status_extra)})" if status_extra else "")


# -----------------------------------------------------------------------------
# JMA AMeDAS (Japan), plan 1.1
# -----------------------------------------------------------------------------
JMA_BASE = "https://www.jma.go.jp/bosai/amedas/"


def _jma_val(m, sid, key):
    """Value from a map file entry [value, quality_flag]; only flag 0 (normal) is used."""
    v = ((m or {}).get(sid) or {}).get(key)
    if isinstance(v, list) and len(v) >= 2 and v[1] == 0:
        return _num(v[0])
    return None


def jma_batch(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "JMA")]
    if not cfgs:
        return
    latest, info = http_text(JMA_BASE + "data/latest_time.txt", attempts=3, backoff_s=[5, 15], timeout_s=30)
    table = http_json(JMA_BASE + "const/amedastable.json", attempts=3, backoff_s=[5, 15], timeout_s=60)
    if latest is None or not isinstance(table, dict):
        _provider_down("JMA", resorts, warnings, provider_status,
                       "JMA AMeDAS request failed; no Japanese station data this run", "request failed")
        return
    try:
        t_latest = datetime.fromisoformat(latest.strip()).replace(minute=0, second=0, microsecond=0)
    except ValueError:
        t_latest = None
    if not table or t_latest is None or t_latest.tzinfo is None:
        _provider_down("JMA", resorts, warnings, provider_status,
                       "JMA AMeDAS answered with empty or malformed data; no Japanese station data this run",
                       "JMA answered with empty or malformed data")
        return
    # Snow keys appear only in the on-the-hour files; step back one hour, up to 3 times.
    T, m0 = None, None
    for back in range(4):
        cand = t_latest - timedelta(hours=back)
        m = http_json(JMA_BASE + f"data/map/{cand:%Y%m%d%H}0000.json", attempts=2, backoff_s=[5],
                      timeout_s=60)
        if isinstance(m, dict) and any(isinstance(v, dict) and "snow" in v for v in m.values()):
            T, m0 = cand, m
            break
    if T is None:
        warnings.append("JMA AMeDAS hourly map with snow values not found; no Japanese station data this run")
        provider_status["JMA"] = "failed"
        return
    hist = {}
    for n in (24, 48, 72):
        hist[n] = http_json(JMA_BASE + f"data/map/{T - timedelta(hours=n):%Y%m%d%H}0000.json",
                            attempts=2, backoff_s=[5], timeout_s=60)
    t_utc = T.astimezone(timezone.utc)

    for r, (rank, sid, label, _) in cfgs:
        meta = table.get(sid)
        if not isinstance(meta, dict) or not meta.get("lat") or not meta.get("lon"):
            msg = f"JMA station {sid} not in AMeDAS metadata, removed"
            r["notes"].append(msg)
            warnings.append(f"{r['name']}: {msg}")
            continue
        lat = meta["lat"][0] + meta["lat"][1] / 60
        lon = meta["lon"][0] + meta["lon"][1] / 60
        row = _station_row("JMA", sid, meta.get("enName") or label, lat, lon, meta.get("alt"), r, rank)
        if not _accept_station(row, r, warnings):
            continue
        depth = _jma_val(m0, sid, "snow")
        if depth is not None and not _plausible_depth(depth):
            r["notes"].append(f"JMA {row['name']}: snow depth {depth} cm outside 0 to 600 cm dropped")
            depth = None
        if depth is not None:
            _set_latest(row, t_utc, depth)
            for n in (24, 48, 72):
                prev = _jma_val(hist[n], sid, "snow")
                prev = prev if _plausible_depth(prev) else None
                row[f"change_{n}h_cm"] = r1(depth - prev) if prev is not None else None
        else:
            r["notes"].append(f"JMA {row['name']} gave no valid snow depth at {utc_iso(t_utc)}")
        # snow24h is JMA's own 24-hour snowfall figure; sums need every period present.
        n24 = [_jma_val(m0, sid, "snow24h"), _jma_val(hist[24], sid, "snow24h"),
               _jma_val(hist[48], sid, "snow24h")]
        if n24[0] is not None and depth is not None:
            row["new_24h_cm"], row["new_48h_cm"], row["new_72h_cm"] = _sum_chain(*n24)
        r.setdefault("stations", []).append(row)
    missing_hist = [n for n in (24, 48, 72) if hist[n] is None]
    provider_status["JMA"] = "ok" + (f" (history missing for {missing_hist} h)" if missing_hist else "")


# -----------------------------------------------------------------------------
# Météo-France hourly climatological files (France, plus department 66), plan 1.2
# -----------------------------------------------------------------------------
MF_WINDOW_H = 14 * 24   # rows kept per station; older readings are reported as "not since"


def _mf_resource_urls(depts):
    urls = {}
    ds = http_json(f"https://www.data.gouv.fr/api/1/datasets/{METEO_FRANCE_DATASET}/",
                   attempts=3, backoff_s=[5, 15], timeout_s=60)
    pat = re.compile(r"/BASE/HOR/H_(\d{2})_latest-(\d{4})-(\d{4})\.csv\.gz$")
    best = {}
    if isinstance(ds, dict):
        for res in ds.get("resources") or []:
            m = pat.search(res.get("url") or "")
            if m and m.group(1) in depts:
                key = (m.group(2), m.group(3))
                if m.group(1) not in best or key > best[m.group(1)][0]:
                    best[m.group(1)] = (key, res["url"])
    fallback = []
    for dep in depts:
        if dep in best:
            urls[dep] = best[dep][1]
        else:
            urls[dep] = METEO_FRANCE_FALLBACK_URL.format(dep=dep)
            fallback.append(dep)
    return urls, fallback


def meteofrance_batch(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "Meteo-France")]
    if not cfgs:
        return
    wanted = {c[1] for _, c in cfgs}
    depts = sorted({sid[:2] for sid in wanted})
    urls, fallback = _mf_resource_urls(depts)
    if fallback:
        logger.info(f"Météo-France: data.gouv.fr lookup gave no URL for {fallback}; using the fixed 2025-2026 name")

    cutoff_key = (run_utc - timedelta(hours=MF_WINDOW_H)).strftime("%Y%m%d%H")
    meta, last_depth, recs = {}, {}, {}
    failed = []
    cols = ("NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH", "NEIGETOT", "HNEIGEF")
    for dep in depts:
        raw, info = http_bytes(urls[dep], attempts=3, backoff_s=[5, 15], timeout_s=180)
        if raw is None:
            failed.append(dep)
            warnings.append(f"Météo-France department {dep} file could not be downloaded ({info})")
            continue
        try:
            with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8", errors="replace", newline="") as f:
                rd = csv.reader(f, delimiter=";")
                header = [h.strip() for h in next(rd)]
                ix = [header.index(c) for c in cols]
                n_max = max(ix)
                i_id, i_name, i_lat, i_lon, i_alt, i_t, i_nt, i_hn = ix
                for rec in rd:
                    if len(rec) <= n_max:
                        continue
                    sid = rec[i_id].strip()
                    if sid not in wanted:
                        continue
                    hh = rec[i_t].strip()
                    prev = meta.get(sid)
                    if prev is None or hh > prev[0]:
                        meta[sid] = (hh, rec[i_name].strip(), rec[i_lat], rec[i_lon], rec[i_alt])
                    nt = rec[i_nt].strip()
                    if nt and hh > last_depth.get(sid, ""):
                        last_depth[sid] = hh
                    if hh >= cutoff_key:
                        recs.setdefault(sid, []).append((hh, nt, rec[i_hn].strip()))
        except (OSError, EOFError, ValueError, StopIteration, csv.Error, zlib.error, IndexError) as e:
            # zlib.error: some gzip corruption raises it instead of BadGzipFile (an OSError).
            failed.append(dep)
            warnings.append(f"Météo-France department {dep} file could not be read ({type(e).__name__}: {e})")
        finally:
            del raw

    def to_dt(hh):
        return datetime.strptime(hh, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    for r, (rank, sid, label, stype) in cfgs:
        dep = sid[:2]
        if dep in failed:
            r["notes"].append(f"Météo-France station {sid} ({label}) not available this run")
            continue
        m = meta.get(sid)
        if m is None:
            msg = f"Météo-France station {sid} ({label}) is not in the department {dep} file, removed"
            r["notes"].append(msg)
            warnings.append(f"{r['name']}: {msg}")
            continue
        _, name, lat, lon, alt = m
        try:
            row = _station_row("Meteo-France", sid, name, float(lat), float(lon), float(alt) if alt else None,
                               r, rank, f"Meteo-France:{stype}")
        except ValueError:
            r["notes"].append(f"Météo-France station {sid} has unreadable coordinates, removed")
            continue
        if not _accept_station(row, r, warnings):
            continue
        rows = sorted(recs.get(sid, []))
        readings = []
        for hh, nt, _hn in rows:
            v = _num(nt) if nt else None
            if v is not None and 0 <= v < MAX_REPORT_DEPTH_CM:
                readings.append((to_dt(hh), v))
        if readings:
            t1, v1 = readings[-1]
            _set_latest(row, t1, v1)
            _depth_changes(row, readings, timedelta(hours=1))
            if stype == "manual":
                # HNEIGEF on the daily ~06 UTC row is the measured fresh snow since the last
                # observation, which can be several days back. A day's value counts as 24-hour
                # snow only when the post also observed 24 h earlier; otherwise the chain stops.
                hn = [(to_dt(hh), _num(h)) for hh, _nt, h in rows if h and _num(h) is not None and _num(h) >= 0]
                obs_hours = sorted({to_dt(hh) for hh, nt, h in rows if nt or h})

                def observed_near(t):
                    return any(abs(o - t) <= timedelta(hours=1) for o in obs_hours)

                days = []
                for i in range(3):
                    v = _closest_reading(hn, t1 - timedelta(hours=24 * i), timedelta(hours=1))
                    if v is not None and not observed_near(t1 - timedelta(hours=24 * (i + 1))):
                        v = None
                    days.append(v)
                if days[0] is not None:
                    row["new_24h_cm"], row["new_48h_cm"], row["new_72h_cm"] = _sum_chain(*days)
        else:
            last = last_depth.get(sid)
            when = to_dt(last).strftime("%-d %b %Y") if last else None
            r["notes"].append(f"Météo-France {name} has not reported a snow depth since {when}" if when else
                              f"Météo-France {name} has not reported a snow depth in this season's file")
        r.setdefault("stations", []).append(row)
    ok = [d for d in depts if d not in failed]
    provider_status["Meteo-France"] = "ok" if not failed else f"{len(ok)}/{len(depts)} departments"
    if failed and not ok:
        provider_status["Meteo-France"] = "failed"


# -----------------------------------------------------------------------------
# Meteocat XEMA (Catalonia), plan 1.3
# -----------------------------------------------------------------------------
METEOCAT_DATA = "https://analisi.transparenciacatalunya.cat/resource/nzvn-apee.json"
METEOCAT_STATIONS_URL = "https://analisi.transparenciacatalunya.cat/resource/yqwd-vj5e.json?$limit=1000"


def meteocat_batch(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "Meteocat")]
    if not cfgs:
        return
    codes = sorted({c[1] for _, c in cfgs})
    since = (run_utc - timedelta(hours=80)).strftime("%Y-%m-%dT%H:%M:%S")
    where = ("codi_estacio in(" + ",".join(f"'{c}'" for c in codes) + f") AND data_lectura >= '{since}'")
    qs = urllib.parse.urlencode({"codi_variable": "38", "$where": where, "$order": "data_lectura DESC",
                                 "$limit": "5000"}, quote_via=urllib.parse.quote)
    data = http_json(f"{METEOCAT_DATA}?{qs}", attempts=3, backoff_s=[5, 15], timeout_s=60)
    time.sleep(1)  # robots.txt asks for a 1-second crawl delay
    meta = http_json(METEOCAT_STATIONS_URL, attempts=3, backoff_s=[5, 15], timeout_s=60)
    if data is None or meta is None:
        _provider_down("Meteocat", resorts, warnings, provider_status,
                       "Meteocat request failed; no Catalan station data this run", "request failed")
        return
    if not _list_of_dicts(data, nonempty=False) or not _list_of_dicts(meta):
        _provider_down("Meteocat", resorts, warnings, provider_status,
                       "Meteocat answered with empty or malformed data; no Catalan station data this run",
                       "Meteocat answered with empty or malformed data")
        return
    raw = []
    for d in data:
        v = _num(d.get("valor_lectura"))
        t = d.get("data_lectura")
        if v is None or not t or d.get("codi_estacio") not in codes:
            continue
        raw.append((d["codi_estacio"], datetime.fromisoformat(t[:19]), v))
    # The portal's timestamps are UTC; if the newest is in the future they are local time.
    local = False
    now = datetime.now(timezone.utc)
    if raw and max(t for _, t, _ in raw).replace(tzinfo=timezone.utc) > now + timedelta(minutes=15):
        local = True
        warnings.append("Meteocat timestamps are ahead of UTC; treated as Europe/Madrid local time")
    readings_by = {}
    for c, t, v in raw:
        t = (t.replace(tzinfo=ZoneInfo("Europe/Madrid")).astimezone(timezone.utc) if local
             else t.replace(tzinfo=timezone.utc))
        if 0 <= v < MAX_REPORT_DEPTH_CM:
            readings_by.setdefault(c, []).append((t, v))
    for c in readings_by:
        readings_by[c].sort(key=lambda x: x[0])
    # Meteocat publishes half-hourly sensor values with no quality flag. Drop a reading
    # that is more than 20 cm from the median of the station's readings within ±60 min.
    spikes = {}
    for c, rd in readings_by.items():
        kept = []
        for t, v in rd:
            near = [x for u, x in rd if abs(u - t) <= timedelta(minutes=60)]
            if abs(v - statistics.median(near)) <= 20:
                kept.append((t, v))
        spikes[c] = len(rd) - len(kept)
        readings_by[c] = kept
    meta_by = {m.get("codi_estacio"): m for m in meta if isinstance(m, dict)}

    for r, (rank, sid, label, _) in cfgs:
        m = meta_by.get(sid)
        if m is None or _num(m.get("latitud")) is None or _num(m.get("longitud")) is None:
            msg = f"Meteocat station {sid} not in metadata, removed"
            r["notes"].append(msg)
            warnings.append(f"{r['name']}: {msg}")
            continue
        name = re.sub(r"\s*\([^)]*\)\s*$", "", m.get("nom_estacio") or label)
        row = _station_row("Meteocat", sid, name, _num(m["latitud"]), _num(m["longitud"]),
                           _num(m.get("altitud")), r, rank)
        if not _accept_station(row, r, warnings):
            continue
        readings = readings_by.get(sid, [])
        if spikes.get(sid):
            r["notes"].append(f"Meteocat {name}: {spikes[sid]} snow-depth readings dropped as sensor spikes")
        if readings:
            # Latest value is the median of the last hour's readings, timed at the newest one.
            t1 = readings[-1][0]
            v1 = statistics.median([v for t, v in readings if t >= t1 - timedelta(minutes=60)])
            _set_latest(row, t1, v1)
            _depth_changes(row, readings[:-1] + [(t1, v1)], timedelta(hours=1))
        else:
            r["notes"].append(f"Meteocat {name} reported no snow depth in the last 80 hours")
        r.setdefault("stations", []).append(row)
    provider_status["Meteocat"] = "ok"


# -----------------------------------------------------------------------------
# Land Tirol weather stations (LWD / HD Tirol), plan 1.4
# -----------------------------------------------------------------------------
TIROL_URL = "https://wiski.tirol.gv.at/lawine/produkte/ogd.geojson"


def tirol_batch(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "Tirol")]
    if not cfgs:
        return
    data = http_json(TIROL_URL, attempts=3, backoff_s=[5, 15], timeout_s=60)
    if data is None:
        _provider_down("Tirol", resorts, warnings, provider_status,
                       "Land Tirol station request failed; no Tirol station data this run", "request failed")
        return
    if not isinstance(data, dict) or not _list_of_dicts(data.get("features")):
        _provider_down("Tirol", resorts, warnings, provider_status,
                       "Land Tirol answered with empty or malformed data; no Tirol station data this run",
                       "Land Tirol answered with empty or malformed data")
        return
    by = {}
    for f in data["features"]:
        p = f.get("properties") or {}
        key = p.get("LWD-Nummer") or f.get("id")
        if key:
            by[str(key)] = f
    for r, (rank, sid, label, _) in cfgs:
        f = by.get(sid)
        coords = ((f or {}).get("geometry") or {}).get("coordinates") or []
        if f is None or len(coords) < 2:
            msg = f"Tirol station {sid} ({label}) not in the Land Tirol file, removed"
            r["notes"].append(msg)
            warnings.append(f"{r['name']}: {msg}")
            continue
        p = f.get("properties") or {}
        row = _station_row("Tirol", sid, p.get("name") or label, coords[1], coords[0],
                           coords[2] if len(coords) > 2 else None, r, rank)
        if not _accept_station(row, r, warnings):
            continue
        hs = _num(p.get("HS"))
        t = None
        if p.get("date"):
            try:
                t = datetime.fromisoformat(p["date"])
                if t.tzinfo is None:
                    t = None   # no offset: the time cannot be placed, so the value is not used
            except ValueError:
                t = None
        if hs is not None and t is not None and 0 <= hs < MAX_REPORT_DEPTH_CM:
            _set_latest(row, t, hs)
            for n in (24, 48, 72):
                row[f"change_{n}h_cm"] = r1(_num(p.get(f"HSD{n}")))
        else:
            r["notes"].append(f"Tirol {row['name']} has no current snow depth")
        r.setdefault("stations", []).append(row)
    provider_status["Tirol"] = "ok"


# -----------------------------------------------------------------------------
# AEMET OpenData (Spain), plan 3. Runs only with AEMET_API_KEY set; unverified.
# -----------------------------------------------------------------------------
AEMET_OBS_URL = "https://opendata.aemet.es/opendata/api/observacion/convencional/todas"
AEMET_DATOS_PREFIX = "https://opendata.aemet.es/"


def _aemet_time(s):
    if not s:
        return None
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s.strip())
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def aemet_batch(resorts, run_utc, warnings, provider_status):
    cfgs = [(r, c) for r in resorts for c in _cfg(r, "AEMET")]
    if not cfgs:
        return
    key = (os.environ.get("AEMET_API_KEY") or "").strip()
    if not key:
        provider_status["AEMET"] = "skipped (no key)"
        return
    # The key travels as a header, never in a URL, so request logging cannot leak it.
    hdr = {"api_key": key, "Accept": "application/json"}
    idx = http_json(AEMET_OBS_URL, attempts=3, backoff_s=[5, 15], timeout_s=60, headers=hdr)
    datos = idx.get("datos") if isinstance(idx, dict) and idx.get("estado") == 200 else None
    # The data URL comes from AEMET's own answer; follow it only on AEMET's own https host.
    if datos is not None and not (isinstance(datos, str) and datos.startswith(AEMET_DATOS_PREFIX)):
        logger.warning("AEMET 'datos' URL is not on https://opendata.aemet.es/; not followed")
        datos = None
    obs = http_json(datos, attempts=3, backoff_s=[5, 15], timeout_s=90) if datos else None
    if not isinstance(obs, list):
        _provider_down("AEMET", resorts, warnings, provider_status,
                       "AEMET request failed; no AEMET station data this run", "request failed")
        return
    readings_by, meta_by = {}, {}
    for o in obs:
        if not isinstance(o, dict) or not o.get("idema"):
            continue
        v, t = _num(o.get("nieve")), _aemet_time(o.get("fint"))
        lat, lon = _num(o.get("lat")), _num(o.get("lon"))
        if v is None or t is None or lat is None or lon is None or not 0 <= v < MAX_REPORT_DEPTH_CM:
            continue
        readings_by.setdefault(o["idema"], []).append((t, v))
        meta_by[o["idema"]] = (o.get("ubi") or o["idema"], lat, lon, _num(o.get("alt")))
    for c in readings_by:
        readings_by[c].sort(key=lambda x: x[0])
    for r, (rank, _sid, _label, _) in cfgs:
        best = None
        for c, (name, lat, lon, alt) in meta_by.items():
            row = _station_row("AEMET", c, name, lat, lon, alt, r, rank)
            if _station_rule_failure(row, r) is None and (best is None or row["distance_km"] < best["distance_km"]):
                best = row
        if best is None:
            r["notes"].append("AEMET: no station reporting snow depth within 15 km and the elevation window this run")
            continue
        readings = readings_by[best["id"]]
        t1, v1 = readings[-1]
        _set_latest(best, t1, v1)
        _depth_changes(best, readings, timedelta(hours=1))
        r.setdefault("stations", []).append(best)
    provider_status["AEMET"] = "ok"


# -----------------------------------------------------------------------------
# Staleness (plan 0.4)
# -----------------------------------------------------------------------------


def _age_words(h):
    return f"{int(round(h))} hours" if h < 48 else f"{int(round(h / 24))} days"


def mark_freshness(r, run_utc):
    """Set _fresh on each station row; a stale reading stays in the row but is not used."""
    for s in r.get("stations", []):
        s["_fresh"] = False
        if s["depth_cm"] is None and s["new_24h_cm"] is None:
            continue
        key = s.get("_age_key") or s["provider"]
        limit = STATION_MAX_AGE_H.get(key, STATION_MAX_AGE_H.get(s["provider"]))
        if s["observed_utc"]:
            t = datetime.fromisoformat(s["observed_utc"].replace("Z", "+00:00"))
            age_h = (run_utc - t).total_seconds() / 3600
        elif s["observed_date"]:
            age_h = (run_utc.date() - date.fromisoformat(s["observed_date"])).days * 24
        else:
            continue
        if limit is None or age_h <= limit:
            s["_fresh"] = True
        else:
            r["notes"].append(f"{s['provider']} {s['name']} reading is {_age_words(age_h)} old, not used")


def _best_station(r):
    """Lowest config rank among fresh stations with a depth (plan 0.2)."""
    cands = [s for s in r.get("stations", []) if s.get("_fresh") and s["depth_cm"] is not None]
    return min(cands, key=lambda s: (s["_rank"], s["distance_km"])) if cands else None


# =============================================================================
# Resort reports
# =============================================================================

OTS_SNOW_RE = re.compile(r'\\"snow\\":\{(.*?)\}')
# The resort's own report date, from the page's structured data. It appears both
# plain and backslash-escaped; the page-wide updatedAt values belong to the
# nearby-resort carousel and are not this resort's report time.
OTS_REPORT_DATE_RE = re.compile(
    r'Last snow report update\\?",\\?"value\\?":\\?"(\d{4}-\d{2}-\d{2})\\?"')
OTS_MAX_REPORT_AGE_DAYS = 3
# The resort object carries its own status right after the snow object:
# openFlag 1 = open, 2 = closed (seen 27 Nov 2026: Tignes 1, Val d'Isere and Meribel 2).
OTS_STATUS_RE = re.compile(r'\\"status\\":\{\\"openFlag\\":(\d+)(?:,\\"openingDate\\":\\"([0-9-]+)\\")?')


def parse_onthesnow(html):
    """Return (report dict or None, report date "YYYY-MM-DD" or None).

    The report dict gains "_open_flag" and "_opening_date" when the page states them.
    """
    report = None
    for m in OTS_SNOW_RE.finditer(html):
        try:
            obj = json.loads("{" + m.group(1).replace('\\"', '"') + "}")
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in ("base", "middle", "summit")):
            report = obj
            st = OTS_STATUS_RE.search(html, m.end(), m.end() + 600)
            if st:
                report["_open_flag"] = int(st.group(1))
                report["_opening_date"] = st.group(2)
            break
    if report is None:
        return None, None
    dates = set(OTS_REPORT_DATE_RE.findall(html))
    observed = None
    if len(dates) == 1:
        try:
            observed = date.fromisoformat(next(iter(dates))).isoformat()
        except ValueError:
            observed = None
    return report, observed


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _budget_mark(source, remaining, warnings, n_total):
    """A note per resort whose page was skipped because the report phase's time budget is
    spent, so the run goes on to Open-Meteo instead of being killed by the workflow timeout."""
    for r in remaining:
        r["notes"].append(f"{source} page not fetched this run (time budget for report pages used up)")
    warnings.append(f"{source}: {len(remaining)} of {n_total} pages skipped, time budget for "
                    "report pages used up")


# Report pages (OnTheSnow, tenki.jp): request starts stay at least REPORT_SPACING_S apart, the
# same 1-second spacing the serial loop used to put between pages, but up to REPORT_WORKERS
# pages are in flight at once instead of one. Results are still handled one by one in page
# order, so notes, warnings and the site-down rule come out exactly as before.
REPORT_SPACING_S = max(1.0, float(os.environ.get("FETCH_REPORT_SPACING_S") or 1.1))
REPORT_WORKERS = max(1, int(os.environ.get("FETCH_REPORT_WORKERS") or 4))
_BUDGET = "time budget for report pages used up"   # sentinel info: page not requested
_STOPPED = "stopped"                                 # sentinel info: run of pages ended early


def _fetch_pages(items, fetch_one, min_left_s=1):
    """Yield fetch_one(item) -> (body, info) for each item, in order.

    Request starts are spaced REPORT_SPACING_S apart and happen in item order. A page whose
    start would come with min_left_s or less of the phase budget left is not requested and
    yields (None, _BUDGET); every later page does the same. Closing the generator stops
    the pages not yet started.

    Site-down rule, kept as polite as the serial loop: a page is not requested when the
    SITE_DOWN_AFTER pages before it all ended in a timeout or connection error, because
    the caller then stops at the earlier page anyway. When the page just before that
    window failed this way, the page waits for the window to finish before deciding."""
    if not items:
        return
    phase = getattr(_THREAD, "phase_deadline", _PHASE_DEADLINE)
    cond = threading.Condition()
    stop = threading.Event()
    state = {"turn": 0, "next": None}
    conn_fail = {}   # page index -> True if it ended in a timeout or connection error

    def window_known(i):
        w = range(max(0, i - SITE_DOWN_AFTER), i)
        return any(conn_fail.get(j) is False for j in w) or all(j in conn_fail for j in w)

    def work(i, item):
        _THREAD.phase_deadline = phase
        failed = False
        try:
            with cond:
                cond.wait_for(lambda: stop.is_set() or (state["turn"] == i and window_known(i)))
                if stop.is_set():
                    return None, _STOPPED
                state["turn"] += 1
                cond.notify_all()
                if i >= SITE_DOWN_AFTER and all(conn_fail[j] for j in range(i - SITE_DOWN_AFTER, i)):
                    failed = True   # so the pages after it are skipped too
                    return None, _STOPPED
                now = time.monotonic()
                start = now if state["next"] is None else max(now, state["next"])
                state["next"] = start + REPORT_SPACING_S
            if start > now:
                stop.wait(start - now)
            if stop.is_set():
                return None, _STOPPED
            if time_left() <= min_left_s:
                return None, _BUDGET
            body, info = fetch_one(item)
            failed = body is None and not str(info).startswith("HTTP ")
            return body, info
        finally:
            with cond:
                conn_fail[i] = failed
                cond.notify_all()

    ex = ThreadPoolExecutor(min(REPORT_WORKERS, len(items)))
    try:
        futs = [ex.submit(work, i, it) for i, it in enumerate(items)]
        for fut in futs:
            yield fut.result()
    finally:
        stop.set()
        with cond:
            cond.notify_all()
        ex.shutdown(wait=True, cancel_futures=True)


SITE_DOWN_AFTER = 4   # consecutive timeouts or connection errors before a report site counts as down


def _site_down(source, down, remaining, warnings):
    if down < SITE_DOWN_AFTER or not remaining:
        return False
    for r in remaining:
        r["notes"].append(f"{source} page not fetched this run (site not responding)")
    warnings.append(f"{source} stopped after {down} pages in a row timed out or could not connect; "
                    f"{len(remaining)} pages skipped")
    return True


def onthesnow_all(resorts, warnings, provider_status):
    targets = [r for r in resorts if r["ots_url"]]
    pages = _fetch_pages(targets, lambda r: http_text(r["ots_url"], attempts=2, backoff_s=[5], timeout_s=30,
                                                      headers={"User-Agent": BROWSER_UA}))
    try:
        return _onthesnow_pages(targets, pages, warnings, provider_status)
    finally:
        pages.close()


def _onthesnow_pages(targets, pages, warnings, provider_status):
    parsed = 0
    down = 0   # consecutive pages lost to timeouts or connection errors (not HTTP statuses)
    for i, (r, (html, info)) in enumerate(zip(targets, pages)):
        if info is _BUDGET:
            _budget_mark("OnTheSnow", targets[i:], warnings, len(targets))
            break
        if html is None:
            r["notes"].append(f"OnTheSnow page could not be fetched ({info})")
            logger.warning(f"OnTheSnow {r['name']}: {info}")
            down = 0 if str(info).startswith("HTTP ") else down + 1
            if _site_down("OnTheSnow", down, targets[i + 1:], warnings):
                break
            continue
        down = 0
        report, observed = parse_onthesnow(html)
        if report is None:
            r["notes"].append("OnTheSnow page returned no snow object")
            logger.warning(f"OnTheSnow {r['name']}: no snow object")
            continue
        parsed += 1
        if observed is None:
            r["notes"].append("OnTheSnow page gives no date for its snow report, so the report is not used")
            logger.info(f"OnTheSnow {r['name']}: no report date, report not used")
            continue
        age_days = (datetime.now(timezone.utc).date() - date.fromisoformat(observed)).days
        if age_days > OTS_MAX_REPORT_AGE_DAYS:
            r["notes"].append(f"OnTheSnow report was last updated on {observed}, {age_days} days ago, "
                              "so it is not used")
            logger.info(f"OnTheSnow {r['name']}: report from {observed} is stale, not used")
            continue
        flag = report.get("_open_flag")
        closed = flag is not None and flag != 1
        opening = report.get("_opening_date")
        try:
            open_words = datetime.strptime(opening, "%Y-%m-%d").strftime("%A %-d %B") if opening else None
        except ValueError:
            open_words = None
        closed_txt = "The resort is closed" + (f" until {open_words}" if open_words else "")
        if closed and all(_num(report.get(k)) in (None, 0.0)
                          for k in ("base", "middle", "summit", "last24", "last48", "last72")):
            # A closed resort's page shows 0 for every depth and new-snow field.
            # That is a placeholder, not a measurement, so the report is not used.
            r["notes"].append(closed_txt + ", and OnTheSnow shows zeros for closed resorts, so its report is not used")
            logger.info(f"OnTheSnow {r['name']}: closed, report not used")
            continue
        if closed:
            # Closed but reporting new snow: keep the snowfall, drop 0 cm depths (placeholders).
            for k in ("base", "middle", "summit"):
                if _num(report.get(k)) == 0.0:
                    report[k] = None
                    r["notes"].append(f"{closed_txt}; OnTheSnow's 0 cm {k} depth for it is a placeholder, not used")
        rep = {
            "base_cm": r1(_num(report.get("base"))),
            "mid_cm": r1(_num(report.get("middle"))),
            "summit_cm": r1(_num(report.get("summit"))),
            "last24_cm": r1(_num(report.get("last24"))),
            "last48_cm": r1(_num(report.get("last48"))),
            "last72_cm": r1(_num(report.get("last72"))),
        }
        for k in ("base_cm", "mid_cm", "summit_cm"):
            v = rep[k]
            if v is not None and (v >= MAX_REPORT_DEPTH_CM or v < 0):
                r["notes"].append(f"OnTheSnow {k.split('_')[0]} depth {v} cm rejected as implausible")
                rep[k] = None
        for k in ("last24_cm", "last48_cm", "last72_cm"):
            if rep[k] is not None and rep[k] < 0:
                rep[k] = None
        if rep["base_cm"] is None and rep["mid_cm"] is None:
            if rep["summit_cm"] is not None:
                r["notes"].append("OnTheSnow report gives only a summit depth, which is not used as base depth")
            else:
                r["notes"].append("OnTheSnow report has no base or mid-mountain depth this update")
        r["ots_report"] = rep
        r["ots_observed"] = observed
        logger.info(f"OnTheSnow {r['name']}: {rep}")
    if targets:
        provider_status["OnTheSnow"] = f"{parsed}/{len(targets)}"
        if parsed == 0 and len(targets) >= 5:
            warnings.append(f"OnTheSnow parsed 0 of {len(targets)} pages")
    return parsed, len(targets)


def tenki_all(resorts, warnings, provider_status):
    """tenki.jp for Japanese resorts with no OnTheSnow depth. Strict pattern only."""
    targets = [r for r in resorts if r["tenki"] and _report_depth(r) is None]
    pages = _fetch_pages(targets, lambda r: http_text(
        f"https://tenki.jp/season/ski/{r['tenki']}/", attempts=2, backoff_s=[5], timeout_s=30,
        headers={"User-Agent": BROWSER_UA, "Accept-Language": "ja,en;q=0.9"}))
    try:
        _tenki_pages(targets, pages, warnings, provider_status)
    finally:
        pages.close()


def _tenki_pages(targets, pages, warnings, provider_status):
    results = {}
    off_season = 0
    reached = 0
    down = 0
    for i, (r, (html, info)) in enumerate(zip(targets, pages)):
        if info is _BUDGET:
            _budget_mark("tenki.jp", targets[i:], warnings, len(targets))
            break
        if html is None:
            r["notes"].append(f"tenki.jp page could not be fetched ({info})")
            down = 0 if str(info).startswith("HTTP ") else down + 1
            if _site_down("tenki.jp", down, targets[i + 1:], warnings):
                break
            continue
        down = 0
        reached += 1
        if TENKI_OFF_SEASON_RE.search(html):
            off_season += 1
            r["notes"].append("tenki.jp snow reports are paused for the off season")
            continue
        m = TENKI_DEPTH_RE.search(html)
        if not m:
            r["notes"].append("tenki.jp page has no snow depth value")
            continue
        v = float(m.group(1))
        if v >= MAX_REPORT_DEPTH_CM:
            r["notes"].append(f"tenki.jp depth {v} cm rejected as implausible")
            continue
        results[r["name"]] = v
    vals = list(results.values())
    if len(vals) >= 2 and len(set(vals)) == 1 and len(vals) == reached - off_season:
        warnings.append(f"tenki.jp returned the same depth ({vals[0]} cm) for every Japanese resort; "
                        "all discarded")
        for r in targets:
            if r["name"] in results:
                r["notes"].append("tenki.jp value discarded (identical across all resorts)")
        results = {}
    for r in targets:
        if r["name"] in results:
            r["tenki_depth"] = results[r["name"]]
    provider_status["tenki.jp"] = (f"{len(results)}/{len(targets)}"
                                   + (f", {off_season} off season" if off_season else ""))
    if targets and len(targets) - off_season >= 5 and not results:
        warnings.append(f"tenki.jp returned no depth for any of {len(targets) - off_season} resorts")


def _report_depth(r):
    rep = r.get("ots_report")
    if not rep:
        return None
    if rep["mid_cm"] is not None:
        return rep["mid_cm"]
    return rep["base_cm"]


# =============================================================================
# Open-Meteo
# =============================================================================

# OPEN_METEO_URL lets a run use an equivalent Open-Meteo host (for example
# previous-runs-api.open-meteo.com, which serves the same current forecast)
# when this IP's daily quota on the main host is exhausted.
OM_URL = os.environ.get("OPEN_METEO_URL") or "https://api.open-meteo.com/v1/forecast"
OM_UNITS = "&wind_speed_unit=kmh&precipitation_unit=mm&temperature_unit=celsius"
DAILY_VARS = ("snowfall_sum,rain_sum,precipitation_sum,precipitation_hours,precipitation_probability_max,"
              "temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,"
              "wind_direction_10m_dominant,weather_code,sunrise,sunset,sunshine_duration,uv_index_max")
HOURLY_VARS = ("snowfall,rain,precipitation,temperature_2m,apparent_temperature,cloud_cover,"
               "cloud_cover_low,cloud_cover_mid,visibility,"
               "wind_speed_10m,wind_gusts_10m,freezing_level_height,snowfall_height,precipitation_probability,"
               "wind_speed_700hPa,weather_code,is_day")
CURRENT_VARS = ("temperature_2m,apparent_temperature,snowfall,weather_code,wind_speed_10m,wind_gusts_10m,"
                "freezing_level_height,is_day")
OM_BATCH_RESORTS = 10
OM_BACKOFF = [8, 16, 32, 60]
# Pause between Open-Meteo calls. Open-Meteo is limited by its per-minute quota, not by
# latency, so its calls stay one at a time; this pause may be raised but not lowered below 1 s.
OM_PAUSE_S = max(1.0, float(os.environ.get("FETCH_OM_PAUSE_S") or 1.0))


def _om_points(resort):
    pts = [("base", resort["base_m"])]
    if resort["summit_m"] is not None:
        pts.append(("mid", resort["mid_m"]))
        pts.append(("summit", resort["summit_m"]))
    return pts


def _om_url(kind, lats, lons, elevs, models=None):
    q = (f"?latitude={','.join(f'{x:.4f}' for x in lats)}"
         f"&longitude={','.join(f'{x:.4f}' for x in lons)}"
         f"&elevation={','.join(str(int(e)) for e in elevs)}")
    if kind == "daily":
        return f"{OM_URL}{q}&daily={DAILY_VARS}&forecast_days=16&past_days=3&timezone=auto{OM_UNITS}"
    if kind == "hourly":
        return (f"{OM_URL}{q}&hourly={HOURLY_VARS}&current={CURRENT_VARS}"
                f"&forecast_days=7&past_days=3&timezone=auto{OM_UNITS}")
    if kind == "models":
        return f"{OM_URL}{q}&daily=snowfall_sum&models={','.join(models)}&forecast_days=7&timezone=auto"
    raise ValueError(kind)


def _om_request(url, n_points, attempts):
    body = http_json(url, attempts=attempts, backoff_s=OM_BACKOFF, timeout_s=60,
                     counter=OM_COUNTER, wait_on_429=60)
    if body is None:
        return None
    if isinstance(body, dict):
        body = [body]
    if not isinstance(body, list) or len(body) != n_points:
        logger.warning(f"Open-Meteo returned {len(body) if isinstance(body, list) else 'non-list'} "
                       f"results for {n_points} points")
        return None
    if not all(isinstance(p, dict) and p for p in body):
        # null, a string, a list or {} in place of a point: the whole answer is unusable
        logger.warning(f"Open-Meteo answer has a point that is not a non-empty object ({n_points} points)")
        return None
    return body


def openmeteo_fetch(kind, resorts, failed_chunks, point_fn, models_fn=None, batch_size=OM_BATCH_RESORTS,
                    group_fn=None):
    """Batched Open-Meteo calls. Returns {resort_id: {label: response}}.

    point_fn(resort) -> [(label, elevation_m)]. Resorts with no points are skipped.
    group_fn(resort) -> key; batches never mix keys (used for icon_d2 regions).
    """
    out = {}
    groups = []
    for r in resorts:
        if not point_fn(r):
            continue
        key = group_fn(r) if group_fn else None
        if groups and groups[-1][0] == key and len(groups[-1][1]) < batch_size:
            groups[-1][1].append(r)
        else:
            groups.append((key, [r]))
    for bi, (key, batch) in enumerate(groups):
        label = f"{kind}:{bi}"
        if time_left() <= 0:
            failed_chunks.append(label)
            for r in batch:
                failed_chunks.append(f"{kind}:{r['id']}")
                r["notes"].append(f"Open-Meteo {kind} call skipped for this resort (run time budget used up)")
            continue
        if bi:
            time.sleep(OM_PAUSE_S)
        models = models_fn(key) if models_fn else None
        pts = [(r, lab, e) for r in batch for lab, e in point_fn(r)]
        url = _om_url(kind, [r["lat"] for r, _, _ in pts], [r["lon"] for r, _, _ in pts],
                      [e for _, _, e in pts], models)
        logger.info(f"Open-Meteo {label}: {len(batch)} resorts, {len(pts)} points")
        body = _om_request(url, len(pts), attempts=5)
        if body is not None:
            for (r, lab, _), resp in zip(pts, body):
                out.setdefault(r["id"], {})[lab] = resp
            continue
        failed_chunks.append(label)
        logger.warning(f"Open-Meteo {label} failed; retrying resort by resort")
        for r in batch:
            if time_left() <= 0:
                failed_chunks.append(f"{kind}:{r['id']}")
                r["notes"].append(f"Open-Meteo {kind} call skipped for this resort (run time budget used up)")
                continue
            time.sleep(OM_PAUSE_S)
            rp = point_fn(r)
            url = _om_url(kind, [r["lat"]] * len(rp), [r["lon"]] * len(rp), [e for _, e in rp], models)
            body = _om_request(url, len(rp), attempts=3)
            if body is None:
                failed_chunks.append(f"{kind}:{r['id']}")
                r["notes"].append(f"Open-Meteo {kind} call failed for this resort")
                continue
            for (lab, _), resp in zip(rp, body):
                out.setdefault(r["id"], {})[lab] = resp
    return out


def verify_model_call():
    """Spec 2.4 step 5: Alta and Zermatt summit, check suffixed daily keys."""
    url = _om_url("models", [40.59, 46.02], [-111.64, 7.75], [3215, 3883],
                  ["best_match", "ecmwf_ifs025", "icon_d2"])
    body = _om_request(url, 2, attempts=5)
    if body is None:
        return False, "model-agreement verification request failed"
    for p in body:
        d = p.get("daily") or {}
        for k in ("snowfall_sum_best_match", "snowfall_sum_ecmwf_ifs025"):
            if k not in d:
                return False, f"verification response lacks {k}"
        if all(v is None for v in d["snowfall_sum_ecmwf_ifs025"]):
            return False, "ecmwf_ifs025 all null in verification response"
    if "snowfall_sum_icon_d2" not in (body[1].get("daily") or {}):
        return False, "verification response lacks snowfall_sum_icon_d2 for Zermatt"
    return True, "ok"


# =============================================================================
# Pure derivations (spec 1.6 - 1.12)
# =============================================================================


def local_hour(run_utc, offset_s):
    """H: run time in the point's local time, floored to the hour (naive)."""
    loc = (run_utc + timedelta(seconds=offset_s)).replace(tzinfo=None)
    return loc.replace(minute=0, second=0, microsecond=0)


@lru_cache(maxsize=8192)
def _hfmt_naive(dt):
    return dt.strftime("%Y-%m-%dT%H:00")


def _hfmt(dt):
    # Cached for naive local times: the derive step asks for the same few hundred hour keys
    # about 200,000 times a run. Aware times are not cached, since two aware datetimes at
    # the same instant compare equal but format differently.
    return _hfmt_naive(dt) if dt.tzinfo is None else dt.strftime("%Y-%m-%dT%H:00")


class Hourly:
    """Index of one Open-Meteo hourly block by local timestamp string."""

    def __init__(self, resp):
        h = (resp or {}).get("hourly") or {}
        self.h = h
        self.idx = {t: i for i, t in enumerate(h.get("time") or [])}

    def get(self, var, t):
        i = self.idx.get(_hfmt(t) if isinstance(t, datetime) else t)
        if i is None:
            return None
        arr = self.h.get(var)
        if arr is None or i >= len(arr):
            return None
        return arr[i]

    def has(self, t):
        return (_hfmt(t) if isinstance(t, datetime) else t) in self.idx

    def series(self, var, times):
        """Values for times; None for missing hours."""
        return [self.get(var, t) for t in times]

    def day_times(self, d: date):
        ds = d.isoformat()
        return [t for t in self.idx if t[:10] == ds]


def modelled_recent(resp, H):
    """hN = sum of hourly snowfall over (H - N h, H]."""
    if resp is None or H is None:
        return {"h24_cm": None, "h48_cm": None, "h72_cm": None}
    hr = Hourly(resp)
    out = {}
    for n in (24, 48, 72):
        times = [H - timedelta(hours=k) for k in range(n)]
        out[f"h{n}_cm"] = r1(strict_sum(hr.series("snowfall", times)))
    return out


def hour_split(elev, precip, rain, fl, sh):
    """Split one hour's water at elevation elev into rain / mixed / snow (spec 1.9.2).

    The phase comes from the model column, not from Open-Meteo's rain/snow split
    (that split is the grid cell's) and not from the point's lapse-rate
    temperature (a fixed 0.65 C per 100 m adjustment that can sit below 0 C
    with the freezing level hundreds of metres higher).
      line = snowfall_height if the API gives it, else freezing level - 250 m
             (a classification threshold only, never stored as a snow line)
      elev <  line: the model's liquid counts as rain, its frozen part as mixed
      elev >= line: the model's liquid counts as mixed (wet snow or rain), its frozen part as snow
    Returns (rain_mm, mixed_mm, snow_mm) of water, or None if an input is missing.
    """
    if precip is None or rain is None or (fl is None and sh is None):
        return None
    line = sh if sh is not None else fl - THRESHOLDS["snowline_below_fl_m"]
    liquid = min(rain, precip)
    frozen = max(precip - liquid, 0.0)
    if elev < line:
        return liquid, frozen, 0.0
    return 0.0, liquid, frozen


def phase_summary(h, times, elev):
    """Phase totals over the given hours of one band's hourly call.

    Returns {"rain_mm", "mixed_mm", "snow_mm", "rain_risk", "mixed_risk", "phase"}; every value
    is None if any needed input is missing (no estimation). A phase counts when it has at
    least phase_min_mm of water and at least phase_share_min of the period's water.
    """
    empty = {"rain_mm": None, "mixed_mm": None, "snow_mm": None,
             "rain_risk": None, "mixed_risk": None, "phase": None}
    if h is None or elev is None or not times:
        return empty
    P = h.series("precipitation", times)
    if any(p is None for p in P):
        return empty
    tot = {"rain": 0.0, "mixed": 0.0, "snow": 0.0}
    for t, p in zip(times, P):
        if p <= 0:
            continue
        sp = hour_split(elev, p, h.get("rain", t), h.get("freezing_level_height", t), h.get("snowfall_height", t))
        if sp is None:
            return empty
        tot["rain"] += sp[0]
        tot["mixed"] += sp[1]
        tot["snow"] += sp[2]
    total = sum(P)
    sig = {k for k, v in tot.items()
           if v >= THRESHOLDS["phase_min_mm"] and v >= THRESHOLDS["phase_share_min"] * total}
    if total < THRESHOLDS["phase_day_min_mm"]:
        phase = "none"
    else:
        if not sig:
            sig_p = {max(tot, key=tot.get)}
        else:
            sig_p = sig
        phase = next(iter(sig_p)) if len(sig_p) == 1 else "mixed"
    return {"rain_mm": r1(tot["rain"]), "mixed_mm": r1(tot["mixed"]), "snow_mm": r1(tot["snow"]),
            "rain_risk": "rain" in sig, "mixed_risk": "mixed" in sig or phase == "mixed",
            "phase": phase}


def _same_series(a, b, times, vars_=("snowfall", "rain", "precipitation")):
    """True if two hourly calls give identical values over times; None if either is missing."""
    if a is None or b is None or not times:
        return None
    for v in vars_:
        sa, sb = a.series(v, times), b.series(v, times)
        if any(x is None for x in sa + sb):
            return None
        if sa != sb:
            return False
    return True


def _period_label(start_hour):
    return {0: "night", 6: "morning", 12: "afternoon", 18: "evening"}[start_hour]


def next_72h(base_resp, mid_resp, summit_resp, H, elevations=None):
    out = {"base_cm": None, "mid_cm": None, "summit_cm": None, "summit_by_period": [],
           "precip_prob_max_pct": None, "window_start_local": None, "window_end_local": None,
           "by_day": {"base": [], "mid": [], "summit": []},
           "mid_equals_summit": None, "base_equals_mid": None}
    if H is None:
        return out
    elevations = elevations or {}
    times = [H + timedelta(hours=k) for k in range(1, 73)]
    out["window_start_local"] = _hfmt(H)
    out["window_end_local"] = _hfmt(times[-1])
    hrs = {"base": Hourly(base_resp) if base_resp is not None else None,
           "mid": Hourly(mid_resp) if mid_resp is not None else None,
           "summit": Hourly(summit_resp) if summit_resp is not None else None}
    for key, resp in (("base_cm", base_resp), ("mid_cm", mid_resp), ("summit_cm", summit_resp)):
        if resp is not None:
            out[key] = r1(strict_sum(Hourly(resp).series("snowfall", times)))
    # Per calendar day segments over the same (H, H+72] hours as the headline,
    # so a sentence built from them always adds up to the headline number.
    for e, h in hrs.items():
        if h is None:
            continue
        dates = list(dict.fromkeys(t.date() for t in times))
        for dd in dates:
            ts = [t for t in times if t.date() == dd]
            ps = phase_summary(h, [_hfmt(t) for t in ts], elevations.get(e))
            out["by_day"][e].append({
                "date": dd.isoformat(),
                "hours": len(ts),
                "snowfall_cm": r1(strict_sum(h.series("snowfall", ts))),
                "rain_mm": r1(strict_sum(h.series("rain", ts))),
                "precip_mm": r1(strict_sum(h.series("precipitation", ts))),
                "phase": ps["phase"],
                "rain_phase_mm": ps["rain_mm"],
                "mixed_phase_mm": ps["mixed_mm"],
            })
    out["mid_equals_summit"] = _same_series(hrs["mid"], hrs["summit"], times)
    out["base_equals_mid"] = _same_series(hrs["base"], hrs["mid"], times)
    if summit_resp is not None:
        hr = Hourly(summit_resp)
        blocks = {}
        order = []
        for t in times:
            prev = t - timedelta(hours=1)  # hour ending at t belongs to the block containing t-1h
            start = prev.replace(hour=(prev.hour // 6) * 6)
            if start not in blocks:
                blocks[start] = []
                order.append(start)
            blocks[start].append(hr.get("snowfall", t) if hr.has(t) else None)
        for start in order:
            vals = blocks[start]
            out["summit_by_period"].append({
                "start_local": _hfmt(start),
                "label": _period_label(start.hour),
                "snowfall_cm": r1(strict_sum(vals)),
            })
    if base_resp is not None:
        probs = Hourly(base_resp).series("precipitation_probability", times)
        out["precip_prob_max_pct"] = rint(max(probs)) if probs and all(p is not None for p in probs) else None
    return out


def now_point(resp):
    if resp is None:
        return None
    c = resp.get("current")
    if not c:
        return None
    is_day = c.get("is_day")
    return {
        "temp_c": r1(c.get("temperature_2m")),
        "apparent_temp_c": r1(c.get("apparent_temperature")),
        "weather_code": rint(c.get("weather_code")),
        "wind_kmh": rint(c.get("wind_speed_10m")),
        "gust_kmh": rint(c.get("wind_gusts_10m")),
        "freezing_level_m": rint(c.get("freezing_level_height")),
        "snowfall_cm": r1(c.get("snowfall")),
        "is_day": None if is_day is None else bool(is_day),
    }


def _daily_by_date(resp):
    d = (resp or {}).get("daily") or {}
    times = d.get("time") or []
    if times:
        times = times[:-1]  # drop day 16 (always null)
    out = {}
    for i, t in enumerate(times):
        out[t] = {k: (v[i] if isinstance(v, list) and i < len(v) else None) for k, v in d.items()}
    return out


def day_point(row, past):
    if row is None:
        return None
    p = {
        "snowfall_cm": r1(row.get("snowfall_sum")),
        "rain_mm": r1(row.get("rain_sum")),
        "precip_mm": r1(row.get("precipitation_sum")),
        "precip_hours": r1(row.get("precipitation_hours")),
        "precip_prob_pct": rint(row.get("precipitation_probability_max")),
        "temp_max_c": r1(row.get("temperature_2m_max")),
        "temp_min_c": r1(row.get("temperature_2m_min")),
        "wind_max_kmh": rint(row.get("wind_speed_10m_max")),
        "gust_max_kmh": rint(row.get("wind_gusts_10m_max")),
        "wind_dir_deg": rint(row.get("wind_direction_10m_dominant")),
        "weather_code": rint(row.get("weather_code")),
        "sunshine_s": rint(row.get("sunshine_duration")),
        "uv_max": r1(row.get("uv_index_max")),
    }
    if past:
        for k in p:
            if k not in ("snowfall_cm", "rain_mm", "precip_mm"):
                p[k] = None
    return p


def _parse_local(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


EMPTY_DERIVED = {
    "freezing_level_max_m": None, "freezing_level_precip_min_m": None, "freezing_level_precip_max_m": None,
    "snowfall_height_precip_min_m": None, "snowfall_height_precip_max_m": None,
    "rain_risk": {"base": None, "mid": None, "summit": None},
    "powder_temp_c": None, "powder_band": None, "wind_700hpa_lift_max_kmh": None, "wind_hold": None,
    "flat_light_hours": None, "wind_chill_min_c": None, "lift_hours_snowfall_cm": None,
    "ski_score_cm": None, "bluebird": None,
    # spec 1.9.2 additions
    "mixed_risk": {"base": None, "mid": None, "summit": None},
    "phase": {"base": None, "mid": None, "summit": None},
    "rain_phase_mm": {"base": None, "mid": None, "summit": None},
    "rain_limit_m": None, "rain_limit_kind": None,
    "wind_gust_lift_max_kmh": None, "cloud_cover_lift_mean_pct": None,
}

_BAND_DICTS = ("rain_risk", "mixed_risk", "phase", "rain_phase_mm")


def _empty_derived():
    d = dict(EMPTY_DERIVED)
    for k in _BAND_DICTS:
        d[k] = dict(EMPTY_DERIVED[k])
    return d


def powder_band(t):
    if t is None:
        return None
    if t < THRESHOLDS["powder_cold_below_c"]:
        return "cold_dry"
    if t <= THRESHOLDS["powder_warm_above_c"]:
        return "average"
    return "warm_heavy"


def derive_day(day, prev_day, hourly, elevations, H):
    """Derived fields for one Day (spec 1.9.1).

    day / prev_day: assembled Day dicts (prev_day may be None).
    hourly: {"base": Hourly|None, "mid": ..., "summit": ...}
    elevations: {"base": m, "mid": m|None, "summit": m|None}
    H: run local hour (naive) for this resort.
    """
    out = _empty_derived()
    if day["horizon"] == "past_model":
        return out
    idx = day["index"]
    d = date.fromisoformat(day["date"])
    short = day["horizon"] == "short"
    hb, hs = hourly.get("base"), hourly.get("summit")
    summit = day.get("summit")

    sunrise = _parse_local(day.get("sunrise_local"))
    sunset = _parse_local(day.get("sunset_local"))

    # ---- hourly-based fields (short horizon only)
    fl_max = None
    if short and hb is not None:
        times = hb.day_times(d)
        if times:
            fl = hb.series("freezing_level_height", times)
            if all(v is not None for v in fl):
                fl_max = max(fl)
                out["freezing_level_max_m"] = rint(fl_max)
            pr = hb.series("precipitation", times)
            if all(v is not None for v in pr):
                wet = [t for t, p in zip(times, pr) if p > 0]
                if wet:
                    flw = hb.series("freezing_level_height", wet)
                    if all(v is not None for v in flw):
                        out["freezing_level_precip_min_m"] = rint(min(flw))
                        out["freezing_level_precip_max_m"] = rint(max(flw))
                    shw = hb.series("snowfall_height", wet)
                    if all(v is not None for v in shw):
                        out["snowfall_height_precip_min_m"] = rint(min(shw))
                        out["snowfall_height_precip_max_m"] = rint(max(shw))
            # rain limit: highest snow line (or freezing level - 250 m) in hours
            # with rain at the base point. Never written into snowfall_height_*.
            rn = hb.series("rain", times)
            if all(v is not None for v in rn):
                rainy = [t for t, v in zip(times, rn) if v >= 0.1]
                lines, kinds = [], set()
                base_m = elevations.get("base")
                for t in rainy:
                    sh, flh = hb.get("snowfall_height", t), hb.get("freezing_level_height", t)
                    if sh is not None:
                        line, kind = sh, "snow_line"
                    elif flh is not None:
                        line, kind = flh - THRESHOLDS["snowline_below_fl_m"], "freezing_level_minus_250"
                    else:
                        lines = None
                        break
                    # only hours that are rain at the base by the phase rule (base below the line)
                    if base_m is not None and base_m < line:
                        lines.append(line)
                        kinds.add(kind)
                if lines:
                    out["rain_limit_m"] = rint(max(lines))
                    out["rain_limit_kind"] = "snow_line" if kinds == {"snow_line"} else "freezing_level_minus_250"

    # ---- precipitation phase per elevation (spec 1.9.2), hour by hour on each
    # band's own hourly call
    for e in ("base", "mid", "summit"):
        elev = elevations.get(e)
        if elev is None:
            continue
        if short:
            h = hourly.get(e)
            ps = phase_summary(h, h.day_times(d) if h is not None else [], elev)
            out["rain_risk"][e] = ps["rain_risk"]
            out["mixed_risk"][e] = ps["mixed_risk"]
            out["phase"][e] = ps["phase"]
            out["rain_phase_mm"][e] = ps["rain_mm"]
        else:
            # Extended days have no hourly data: false only when there is no rain at all
            dp = day.get(e)
            rain_mm = dp.get("rain_mm") if dp else None
            v = tri_and(None, None if rain_mm is None else rain_mm > 0)
            out["rain_risk"][e] = v
            out["mixed_risk"][e] = v

    # ---- summit hourly fields
    lift_times = []
    if short and hs is not None and sunrise is not None and sunset is not None:
        lift_times = [t for t in hs.day_times(d) if sunrise < datetime.fromisoformat(t) <= sunset]

    if short and hs is not None and summit is not None:
        sday = summit.get("snowfall_cm")
        if sday is not None and sday >= THRESHOLDS["snowy_day_min_cm"]:
            times = hs.day_times(d)
            T = hs.series("temperature_2m", times)
            S = hs.series("snowfall", times)
            if times and all(v is not None for v in T + S):
                ssum = sum(S)
                if ssum > 0:
                    out["powder_temp_c"] = r1(sum(t * s for t, s in zip(T, S)) / ssum)
        out["powder_band"] = powder_band(out["powder_temp_c"])

    if lift_times:
        w7 = hs.series("wind_speed_700hPa", lift_times)
        if all(v is not None for v in w7):
            out["wind_700hpa_lift_max_kmh"] = rint(max(w7))
        vis = hs.series("visibility", lift_times)
        cl = hs.series("cloud_cover_low", lift_times)
        cm = hs.series("cloud_cover_mid", lift_times)
        if all(v is not None for v in vis + cl + cm):
            # low or mid cloud flattens light; high thin cloud does not
            out["flat_light_hours"] = sum(
                1 for v, a, b in zip(vis, cl, cm)
                if v < THRESHOLDS["flat_light_visibility_m"] or max(a, b) > THRESHOLDS["flat_light_cloud_pct"])
        cc = hs.series("cloud_cover", lift_times)
        if all(v is not None for v in cc):
            out["cloud_cover_lift_mean_pct"] = rint(sum(cc) / len(cc))
        gl = hs.series("wind_gusts_10m", lift_times)
        if all(v is not None for v in gl):
            out["wind_gust_lift_max_kmh"] = rint(max(gl))
        at = hs.series("apparent_temperature", lift_times)
        if all(v is not None for v in at):
            out["wind_chill_min_c"] = r1(min(at))
        out["lift_hours_snowfall_cm"] = r1(strict_sum(hs.series("snowfall", lift_times)))

    # ---- wind hold (OR). Short days: summit gusts during lift hours, plus 700 hPa
    # wind only where the summit reaches toward that level. Extended days: daily
    # gust only, so true or null.
    if short:
        summit_m = elevations.get("summit")
        w7_cond = (gt(out["wind_700hpa_lift_max_kmh"], THRESHOLDS["wind_700hpa_hold_kmh"])
                   if summit_m is not None and summit_m >= THRESHOLDS["wind_700hpa_min_summit_m"]
                   else (False if summit_m is not None else None))
        out["wind_hold"] = tri_or(gt(out["wind_gust_lift_max_kmh"], THRESHOLDS["wind_gust_hold_kmh"]), w7_cond)
    else:
        gust = summit.get("gust_max_kmh") if summit else None
        out["wind_hold"] = tri_or(gt(gust, THRESHOLDS["wind_gust_hold_kmh"]), None)

    # ---- ski score and bluebird (index 0..6)
    if 0 <= idx <= 6:
        prev_snow = (prev_day.get("summit") or {}).get("snowfall_cm") if prev_day else None
        key = "mid" if elevations.get("mid") is not None else "summit"
        rr = out["rain_risk"][key]
        prev_rr = ((prev_day or {}).get("derived") or {}).get("rain_risk", {}).get(key)
        score = None
        if prev_snow is not None and out["lift_hours_snowfall_cm"] is not None:
            score = r1(prev_snow + out["lift_hours_snowfall_cm"])
        # also null when the snow being counted was rained on the day before
        if out["wind_hold"] is True or rr is True or prev_rr is True:
            score = None
        if idx == 0 and (sunset is None or H is None or H > sunset):
            score = None
        out["ski_score_cm"] = score

        c1 = None if prev_snow is None else prev_snow >= THRESHOLDS["bluebird_prev24_cm"]
        code = summit.get("weather_code") if summit else None
        snow_today = summit.get("snowfall_cm") if summit else None
        cmean, flh = out["cloud_cover_lift_mean_pct"], out["flat_light_hours"]
        out["bluebird"] = tri_and(
            c1,
            None if snow_today is None else snow_today < THRESHOLDS["snowy_day_min_cm"],
            None if code is None else code in THRESHOLDS["bluebird_codes"],
            None if cmean is None else cmean <= THRESHOLDS["bluebird_cloud_mean_pct"],
            None if flh is None else flh <= THRESHOLDS["bluebird_flat_light_max_h"],
        )
    return out


def build_days(resp_by_label, elevations, run_utc, H):
    """Days -3..14 from the daily call, with derived fields."""
    base_resp = resp_by_label.get("daily_base")
    if not base_resp:   # the same truthiness test as the daily dict below ({} included)
        return []
    offset = base_resp.get("utc_offset_seconds") or 0
    today = (run_utc + timedelta(seconds=offset)).date()
    daily = {e: _daily_by_date(resp_by_label.get(f"daily_{e}")) if resp_by_label.get(f"daily_{e}") else None
             for e in ("base", "mid", "summit")}
    hourly = {e: Hourly(resp_by_label[f"hourly_{e}"]) if resp_by_label.get(f"hourly_{e}") else None
              for e in ("base", "mid", "summit")}
    days = []
    for ds in sorted(daily["base"].keys()):
        idx = (date.fromisoformat(ds) - today).days
        if idx < -3 or idx > 14:
            continue
        horizon = "past_model" if idx < 0 else ("short" if idx <= 6 else "extended")
        brow = daily["base"][ds]
        sr, ss = brow.get("sunrise"), brow.get("sunset")
        srd, ssd = _parse_local(sr), _parse_local(ss)
        day = {
            "date": ds, "index": idx, "horizon": horizon,
            "sunrise_local": sr, "sunset_local": ss,
            "daylight_s": int((ssd - srd).total_seconds()) if srd and ssd else None,
        }
        for e in ("base", "mid", "summit"):
            day[e] = day_point(daily[e].get(ds), horizon == "past_model") if daily[e] is not None else None
        days.append(day)
    by_idx = {d["index"]: d for d in days}
    for d in days:
        d["derived"] = derive_day(d, by_idx.get(d["index"] - 1), hourly, elevations, H)
    return days


def week_summary(days):
    by = {d["index"]: d for d in days}

    def total(e, idxs):
        vals = []
        for i in idxs:
            dp = by.get(i, {}).get(e) if i in by else None
            vals.append(dp.get("snowfall_cm") if dp else None)
        s = strict_sum(vals)
        return r1(s)

    week = {"base_cm": total("base", range(7)), "mid_cm": total("mid", range(7)),
            "summit_cm": total("summit", range(7)), "best_snow_day": None,
            "best_ski_day": None, "best_ski_day_score_cm": None}
    svals = [((by.get(i) or {}).get("summit") or {}).get("snowfall_cm") if i in by else None for i in range(7)]
    if all(v is not None for v in svals) and any(v > 0 for v in svals):
        m = max(svals)
        week["best_snow_day"] = svals.index(m)
    cands = []
    for i in range(7):
        d = by.get(i)
        if not d:
            continue
        sc = d["derived"]["ski_score_cm"]
        if sc is None:
            continue
        sun = (d.get("summit") or {}).get("sunshine_s")
        cands.append((sc, sun if sun is not None else -1, -i, i))
    if cands:
        best = max(cands)
        if best[0] > 0:
            week["best_ski_day"] = best[3]
            week["best_ski_day_score_cm"] = best[0]
    ext_idx = [i for i in range(7, 15) if i in by]
    extended = {"summit_cm": total("summit", range(7, 15)) if len(ext_idx) == 8 else None,
                "base_cm": total("base", range(7, 15)) if len(ext_idx) == 8 else None,
                "days_available": len(ext_idx)}
    return week, extended


def storm_flags(days, use_summit):
    e = "summit" if use_summit else "base"
    by = {d["index"]: d for d in days}

    def snow(i):
        dp = (by.get(i) or {}).get(e)
        return dp.get("snowfall_cm") if dp else None

    reasons = []
    conds = []
    s3 = strict_sum([snow(1), snow(2), snow(3)])
    c1 = None if s3 is None else s3 >= THRESHOLDS["storm_3day_cm"]
    conds.append(c1)
    if c1 is True:
        reasons.append({"index": 1, "date": by[1]["date"], "reason": "3-day total"})
    for i in range(7):
        dp = (by.get(i) or {}).get(e)
        if not dp:
            conds.append(None)
            continue
        code, prob = dp.get("weather_code"), dp.get("precip_prob_pct")
        sf = dp.get("snowfall_cm")
        # Spec 1.11: a heavy-snow code alone is not enough; the day must also
        # forecast at least storm_day_min_cm of snow. Missing input -> unknown.
        c = tri_and(None if code is None else code in THRESHOLDS["storm_codes"],
                    None if prob is None else prob >= THRESHOLDS["storm_prob_pct"],
                    None if sf is None else sf >= THRESHOLDS["storm_day_min_cm"])
        conds.append(c)
        if c is True:
            reasons.append({"index": i, "date": by[i]["date"], "reason": "heavy snow code"})
    flags = {
        "storm_incoming": tri_or(*conds) if days else None,
        "storm_reasons": reasons,
        "bluebird_days": [d["index"] for d in days if d["derived"]["bluebird"] is True],
        "wind_hold_days": [d["index"] for d in days if 0 <= d["index"] <= 14 and d["derived"]["wind_hold"] is True],
        "rain_days": [],
        "mixed_days": [],
    }
    for d in days:
        if 0 <= d["index"] <= 6:
            key = "mid" if d.get("mid") is not None else "summit"
            if d["derived"]["rain_risk"][key] is True:
                flags["rain_days"].append(d["index"])
            if d["derived"]["mixed_risk"][key] is True:
                flags["mixed_days"].append(d["index"])
    return flags


def model_agreement(resp, models, today):
    if resp is None:
        return None
    d = resp.get("daily") or {}
    times = d.get("time") or []
    days = []
    for i, t in enumerate(times):
        idx = (date.fromisoformat(t) - today).days
        if not 0 <= idx <= 6:
            continue
        vals = {}
        for m in ("best_match", "ecmwf_ifs025", "icon_d2"):
            arr = d.get(f"snowfall_sum_{m}") if m in models else None
            vals[m] = r1(arr[i]) if arr is not None and i < len(arr) else None
        days.append({"index": idx, "snowfall_cm": vals})
    days.sort(key=lambda x: x["index"])
    agree = None
    for day in days:
        vs = [v for v in day["snowfall_cm"].values() if v is not None]
        if len(vs) < 2:
            break
        hi, lo = max(vs), min(vs)
        if hi - lo > max(3.0, 0.3 * hi):
            break
        if agree is None and day["index"] != 0:
            break
        agree = day["index"]
    return {"compared": list(models), "days": days, "agree_through": agree}


# =============================================================================
# Assemble
# =============================================================================


def choose_depth(r):
    """Spec 1.4 selection. Returns Depth dict."""
    empty = {"value_cm": None, "kind": None, "provider": None, "source_label": None,
             "observed_utc": None, "report": None}
    rep = r.get("ots_report")
    country = r["country"]

    def from_report():
        v = _report_depth(r)
        if v is None:
            return None
        which = "mid-mountain" if rep["mid_cm"] is not None else "base"
        return {"value_cm": v, "kind": "resort_report", "provider": "OnTheSnow",
                "source_label": f"Resort report via OnTheSnow ({which})",
                "observed_utc": r.get("ots_observed"), "report": rep}

    def from_station():
        s = _best_station(r)
        if s is None:
            return None
        elev = f", {s['elevation_m']:,} m" if s["elevation_m"] is not None else ""
        return {"value_cm": s["depth_cm"], "kind": "measured", "provider": s["provider"],
                "source_label": f"{s['provider']} {s['name']} station{elev}, {s['distance_km']:g} km away",
                "observed_utc": s["observed_utc"], "report": None}

    # Plan 0.3. Resort reports first where they exist (they describe the slopes);
    # stations in config order, fresh readings only.
    if country == "USA":
        chain = [from_report, from_station]
    elif country in ("Austria", "Switzerland"):
        chain = [from_station]
    elif country == "Japan":
        def tenki():
            v = r.get("tenki_depth")
            if v is None:
                return None
            return {"value_cm": r1(v), "kind": "resort_report", "provider": "tenki.jp",
                    "source_label": "Resort report via tenki.jp", "observed_utc": None, "report": None}
        chain = [from_report, tenki, from_station]
    else:  # France, Spain
        chain = [from_report, from_station]
    for fn in chain:
        d = fn()
        if d is not None:
            return d
    return empty


def recent_measured(r):
    """Plan 2.1: resort report -> measured new snow at the best fresh station -> its depth change.

    A depth change is never labelled as new snow."""
    rep = r.get("ots_report")
    if rep is not None:
        return {"kind": "resort_reported_new_snow", "provider": "OnTheSnow",
                "label": "New snow reported by the resort",
                "h24_cm": rep["last24_cm"], "h48_cm": rep["last48_cm"], "h72_cm": rep["last72_cm"],
                "observed_utc": r.get("ots_observed")}
    s = _best_station(r)
    if s is None:
        return None
    observed = s["observed_utc"] or s["observed_date"]
    if any(s[k] is not None for k in ("new_24h_cm", "new_48h_cm", "new_72h_cm")):
        if s.get("_age_key") == "SLF:imis":
            return {"kind": "station_modelled_new_snow", "provider": s["provider"],
                    "label": f"New snow at {s['name']} station (SLF snowpack model)",
                    "h24_cm": s["new_24h_cm"], "h48_cm": s["new_48h_cm"], "h72_cm": s["new_72h_cm"],
                    "observed_utc": observed}
        return {"kind": "measured_new_snow", "provider": s["provider"],
                "label": f"New snow measured at {s['name']} station",
                "h24_cm": s["new_24h_cm"], "h48_cm": s["new_48h_cm"], "h72_cm": s["new_72h_cm"],
                "observed_utc": observed}
    return {"kind": "depth_change", "provider": s["provider"],
            "label": f"Depth change at {s['name']} station",
            "h24_cm": s["change_24h_cm"], "h48_cm": s["change_48h_cm"], "h72_cm": s["change_72h_cm"],
            "observed_utc": observed}


def assemble_resort(r, om, run_utc, models_ok, models_for):
    labels = om.get(r["id"], {})
    has_summit = r["summit_m"] is not None
    hb, hm, hs = labels.get("hourly_base"), labels.get("hourly_mid"), labels.get("hourly_summit")
    tz = None
    offset = None
    for k in ("daily_base", "hourly_base"):
        if labels.get(k):
            tz = labels[k].get("timezone")
            offset = labels[k].get("utc_offset_seconds")
            break
    H = local_hour(run_utc, offset) if offset is not None else None

    elevations = {"base": r["base_m"], "mid": r["mid_m"], "summit": r["summit_m"]}
    days = build_days(labels, elevations, run_utc, H)
    by_i = {d["index"]: d for d in days}
    week, extended = week_summary(days)
    flags = storm_flags(days, use_summit=has_summit)

    stations = sorted(r.get("stations", []), key=lambda s: s["distance_km"])
    r["stations"] = stations
    mark_freshness(r, run_utc)
    depth = choose_depth(r)
    measured = recent_measured(r)
    stations = [{k: v for k, v in s.items() if not k.startswith("_")} for s in stations]

    def hr_offset(resp):
        return local_hour(run_utc, resp.get("utc_offset_seconds") or 0) if resp else None

    recent = {
        "measured": measured,
        "modelled": {
            "kind": "model",
            "base": modelled_recent(hb, hr_offset(hb)),
            "mid": modelled_recent(hm, hr_offset(hm)) if has_summit else None,
            "summit": modelled_recent(hs, hr_offset(hs)) if has_summit else None,
        },
    }
    nxt = next_72h(hb, hm, hs, H, elevations)

    # Which model grid cell each elevation's call landed on, and whether two
    # bands got identical precipitation (Open-Meteo adjusts only temperature
    # for elevation; precipitation is the cell's). Facts only, nothing filled.
    model_cells = {}
    for e, resp in (("base", hb), ("mid", hm), ("summit", hs)):
        model_cells[e] = ({"lat": resp.get("latitude"), "lon": resp.get("longitude"),
                           "model_elevation_m": rint(resp.get("elevation"))} if resp else None)
    same_cell = None
    if model_cells["mid"] and model_cells["summit"]:
        same_cell = (model_cells["mid"]["lat"], model_cells["mid"]["lon"]) == \
                    (model_cells["summit"]["lat"], model_cells["summit"]["lon"])
    week_same = None
    if has_summit:
        week_same = True
        for i in range(7):
            m, s_ = (by_i.get(i) or {}).get("mid"), (by_i.get(i) or {}).get("summit")
            if not m or not s_ or any(m.get(k) is None or s_.get(k) is None
                                      for k in ("snowfall_cm", "rain_mm", "precip_mm")):
                week_same = None
                break
            if any(m[k] != s_[k] for k in ("snowfall_cm", "rain_mm", "precip_mm")):
                week_same = False
                break
    bands = {"mid_summit_same_cell": same_cell,
             "mid_summit_same_72h": nxt["mid_equals_summit"],
             "mid_summit_same_week": week_same,
             "base_mid_same_72h": nxt["base_equals_mid"]}
    if nxt["mid_equals_summit"] is True:
        r["notes"].append("Mid-mountain and summit get the same forecast precipitation for the next three days; "
                          "the model cannot separate these heights here")
    now = {"local_time": _hfmt(H) if H else None, "base": now_point(hb),
           "summit": now_point(hs) if has_summit else None}

    models = None
    if models_ok and has_summit and labels.get("models_summit") and offset is not None:
        today = (run_utc + timedelta(seconds=offset)).date()
        models = model_agreement(labels["models_summit"], models_for(r), today)

    by = {d["index"]: d for d in days}
    forecast = []
    for i in range(7):
        d = by.get(i)
        if d:
            forecast.append({"day": i, "date": d["date"],
                             "new_snow_cm": (d["base"] or {}).get("snowfall_cm"), "source": "open-meteo"})
    forecast_ok = len(forecast) == 7 and all(f["new_snow_cm"] is not None for f in forecast)
    summit_ok = has_summit and all(
        by.get(i) and by[i].get("summit") and by[i]["summit"].get("snowfall_cm") is not None for i in range(7))
    if not forecast_ok:
        r["notes"].append("Base forecast incomplete for days 0 to 6")

    return {
        "id": r["id"], "name": r["name"], "country": r["country"], "region": r["region"],
        "lat": r["lat"], "lon": r["lon"], "timezone": tz, "utc_offset_s": offset,
        "elevation_m": r["base_m"], "summit_elevation_m": r["summit_m"], "mid_elevation_m": r["mid_m"],
        "elevation_note": r["elev_note"],
        "base_depth_cm": depth["value_cm"], "base_depth_source": depth["source_label"],
        "temp_c": (now["base"] or {}).get("temp_c"),
        "summit_temp_c": (now["summit"] or {}).get("temp_c"),
        "forecast": forecast,
        "seven_day_total_cm": week["base_cm"],
        "summit_seven_day_total_cm": week["summit_cm"],
        "depth": depth,
        "stations": stations,
        "recent": recent,
        "now": now,
        "next_72h": nxt,
        "days": days,
        "week": week,
        "extended": extended,
        "flags": flags,
        "models": models,
        "model_cells": model_cells,
        "quality": {
            "bands": bands,
            "forecast_ok": forecast_ok,
            "summit_forecast_ok": bool(summit_ok),
            "depth_ok": depth["value_cm"] is not None,
            "notes": list(dict.fromkeys(r["notes"])),
        },
    }


def sources_block(resorts_out):
    counts = {}
    for r in resorts_out:
        p = r["depth"]["provider"]
        if p:
            counts.setdefault(r["country"], {}).setdefault(p, 0)
            counts[r["country"]][p] += 1
    totals = {}
    for r in resorts_out:
        totals[r["country"]] = totals.get(r["country"], 0) + 1
    names = {"OnTheSnow": "OnTheSnow resort reports", "tenki.jp": "tenki.jp resort reports",
             "SNOTEL": "SNOTEL stations", "GeoSphere": "GeoSphere TAWES stations", "SLF": "SLF stations",
             "JMA": "JMA AMeDAS stations", "Meteo-France": "Météo-France stations",
             "Meteocat": "Meteocat XEMA stations", "Tirol": "Land Tirol stations",
             "AEMET": "AEMET stations"}
    base_depth = {}
    for c in ("USA", "Austria", "Switzerland", "Japan", "France", "Spain"):
        n = totals.get(c, 0)
        parts = [f"{names[p]} ({k} of {n})" for p, k in sorted(counts.get(c, {}).items(), key=lambda x: -x[1])]
        base_depth[c] = ", ".join(parts) if parts else "none this run"

    used = {"open-meteo"}
    for r in resorts_out:
        for s in r["stations"]:
            if s["depth_cm"] is not None:
                used.add(s["provider"].lower())
        if r["depth"]["report"] is not None or (r["recent"]["measured"] or {}).get("provider") == "OnTheSnow":
            used.add("onthesnow")
        if r["depth"]["provider"] == "tenki.jp":
            used.add("tenki")
    catalogue = [
        {"id": "open-meteo", "name": "Open-Meteo", "url": "https://open-meteo.com", "kind": "model",
         "used_for": "forecasts, recent modelled snowfall, freezing level, wind"},
        {"id": "snotel", "name": "USDA NRCS SNOTEL (AWDB)", "url": "https://wcc.sc.egov.usda.gov",
         "kind": "measured", "used_for": "measured snow depth and 15-day depth history at US stations"},
        {"id": "geosphere", "name": "GeoSphere Austria TAWES", "url": "https://dataset.api.hub.geosphere.at",
         "kind": "measured", "used_for": "measured snow depth at Austrian stations"},
        {"id": "slf", "name": "WSL Institute for Snow and Avalanche Research SLF, IMIS and study plots",
         "url": "https://measurement-api.slf.ch", "kind": "measured",
         "used_for": "measured snow depth; new snow modelled by SLF's snowpack model (study plots measured)"},
        {"id": "jma", "name": "Japan Meteorological Agency, AMeDAS", "url": "https://www.jma.go.jp/bosai/amedas/",
         "kind": "measured",
         "used_for": "measured snow depth and new snow at Japanese stations "
                     "(出典：気象庁ホームページ (https://www.jma.go.jp/); selected and summed by Snowforecast)"},
        {"id": "meteo-france", "name": "Météo-France, hourly climatological data (Licence Ouverte 2.0)",
         "url": f"https://www.data.gouv.fr/datasets/{METEO_FRANCE_DATASET}/", "kind": "measured",
         "used_for": "measured snow depth at French and Pyrenean stations"},
        {"id": "meteocat", "name": "Servei Meteorològic de Catalunya, XEMA",
         "url": "https://analisi.transparenciacatalunya.cat", "kind": "measured",
         "used_for": "measured snow depth at Catalan stations"},
        {"id": "tirol", "name": "Land Tirol weather stations, LWD and HD Tirol (CC BY 4.0)",
         "url": "https://www.data.gv.at", "kind": "measured",
         "used_for": "measured snow depth at Tyrolean stations (Wetterstationsdaten Tirol, data.tirol.gv.at)"},
        {"id": "aemet", "name": "AEMET OpenData", "url": "https://opendata.aemet.es", "kind": "measured",
         "used_for": "measured snow depth at Spanish stations"},
        {"id": "onthesnow", "name": "OnTheSnow", "url": "https://www.onthesnow.com", "kind": "resort_report",
         "used_for": "resort-reported base, mid and summit depth and new snow"},
        {"id": "tenki", "name": "tenki.jp", "url": "https://tenki.jp", "kind": "resort_report",
         "used_for": "resort-reported snow depth in Japan"},
    ]
    return {
        "base_depth": base_depth,
        "forecast": "Open-Meteo (open-meteo.com), best_match model",
        "list": [s for s in catalogue if s["id"] in used],
    }


def write_atomic(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, out_path)


def _guard(resorts, warnings, provider_status):
    """guarded(provider, fn, *a): one provider's parsing surprise must not take down the run."""
    def guarded(provider, fn, *a):
        if provider in STATION_PROVIDERS and time_left() <= 0:
            _provider_down(provider, resorts, warnings, provider_status,
                           f"{provider} skipped: time budget for measured sources used up",
                           "time budget used up")
            return
        try:
            fn(*a)
        except Exception as e:
            logger.exception(f"{provider} failed")
            provider_status[provider] = "failed"
            warnings.append(f"{provider} failed with {type(e).__name__}: {e}")
            for r in resorts:
                if any(p == provider for p, *_ in r["station_cfg"]):
                    r["notes"].append(f"{provider} data not available this run (processing error)")
    return guarded


PARALLEL_CHAINS = (os.environ.get("FETCH_PARALLEL_CHAINS") or "1") != "0"


def _run_chains(resorts, warnings, provider_status, chains):
    """Run chains [(fn(resorts, warnings, provider_status), phase_deadline)] and merge.

    Each chain gets shallow copies of the resort dicts with an empty notes list, and its
    own warnings list and provider_status dict. After all chains finish, notes, warnings,
    statuses and any keys a chain set on a resort are merged back in chain order, the
    order a serial run would have produced. Chains must not set the same resort key."""
    jobs = []
    for fn, phase in chains:
        views = [dict(r, notes=[]) for r in resorts]
        jobs.append((fn, phase, views, [], {}))

    def run(job):
        fn, phase, views, w, ps = job
        _THREAD.phase_deadline = phase
        try:
            fn(views, w, ps)
        finally:
            del _THREAD.phase_deadline

    if PARALLEL_CHAINS:
        with ThreadPoolExecutor(len(jobs)) as ex:
            for fut in [ex.submit(run, j) for j in jobs]:
                fut.result()
    else:
        for j in jobs:
            run(j)
    for fn, _, views, w, ps in jobs:
        for r, v in zip(resorts, views):
            r["notes"].extend(v["notes"])
            for k, val in v.items():
                if k != "notes" and (k not in r or r[k] is not val):
                    r[k] = val
        warnings.extend(w)
        provider_status.update(ps)


def main(argv=None):
    global _RUN_DEADLINE, _PHASE_DEADLINE
    try:
        return _main(argv)
    finally:
        _RUN_DEADLINE = _PHASE_DEADLINE = None


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated resort names (subset run)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    global _RUN_DEADLINE, _PHASE_DEADLINE
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    _RUN_DEADLINE = t0 + RUN_BUDGET_S
    # Measured sources and resort reports must leave OM_RESERVE_S of the budget for Open-Meteo.
    _PHASE_DEADLINE = t0 + max(RUN_BUDGET_S - OM_RESERVE_S, 0)
    OM_COUNTER.requests = OM_COUNTER.retries = 0
    warnings: list[str] = []
    provider_status: dict[str, str] = {}

    resorts = build_resorts()
    full_count = len(resorts)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in {r["name"] for r in resorts}]
        if unknown:
            raise SystemExit(f"unknown resort(s): {unknown}")
        resorts = [r for r in resorts if r["name"] in wanted]
    logger.info(f"Resorts: {len(resorts)}")

    # 2-4. Three independent fetch chains on different hosts, run at the same time:
    #   measured  - the station providers, one after another
    #   reports   - OnTheSnow, then tenki.jp (tenki only asks for resorts OnTheSnow left without a depth)
    #   openmeteo - every Open-Meteo call, still one at a time
    # Each chain works on its own view of the resorts (own notes list) and its own warnings
    # and provider_status, merged afterwards in that fixed order, so the output is the same
    # as running the chains one after another. FETCH_PARALLEL_CHAINS=0 runs them serially.
    phase_deadline = _PHASE_DEADLINE   # measured sources and reports must leave OM_RESERVE_S

    def measured_chain(rs, warnings, provider_status):
        logger.info("--- measured sources ---")
        g = _guard(rs, warnings, provider_status)
        g("SNOTEL", snotel_batch, rs, started, warnings, provider_status)
        g("GeoSphere", geosphere_batch, rs, started, warnings, provider_status)
        g("SLF", slf_once, rs, started, warnings, provider_status)
        g("JMA", jma_batch, rs, started, warnings, provider_status)
        g("Meteo-France", meteofrance_batch, rs, started, warnings, provider_status)
        g("Meteocat", meteocat_batch, rs, started, warnings, provider_status)
        g("Tirol", tirol_batch, rs, started, warnings, provider_status)
        g("AEMET", aemet_batch, rs, started, warnings, provider_status)

    def reports_chain(rs, warnings, provider_status):
        logger.info("--- resort reports ---")
        g = _guard(rs, warnings, provider_status)
        g("OnTheSnow", onthesnow_all, rs, warnings, provider_status)
        g("tenki.jp", tenki_all, rs, warnings, provider_status)

    alpine = {"Austria", "Switzerland"}

    def models_for_key(key):
        return ["best_match", "ecmwf_ifs025"] + (["icon_d2"] if key == "alpine" else [])

    def models_for(r):
        return models_for_key("alpine" if r["country"] in alpine else "other")

    om: dict[str, dict] = {}
    failed_chunks: list[str] = []
    models_state = {}

    def openmeteo_chain(rs, warnings, provider_status):
        # From here on only the run budget applies.
        logger.info(f"--- Open-Meteo --- ({int(time_left())} s of run budget left)")
        for kind in ("daily", "hourly"):
            res = openmeteo_fetch(kind, rs, failed_chunks, _om_points)
            for rid, labs in res.items():
                for lab, resp in labs.items():
                    om.setdefault(rid, {})[f"{kind}_{lab}"] = resp
            time.sleep(OM_PAUSE_S)
        try:
            models_ok, why = verify_model_call()
        except Exception as e:   # a malformed answer switches the optional feature off, no more
            logger.exception("model-agreement verification failed")
            models_ok, why = False, f"verification response malformed ({type(e).__name__}: {e})"
        logger.info(f"Model-agreement verification: {models_ok} ({why})")
        if models_ok:
            time.sleep(OM_PAUSE_S)
            res = openmeteo_fetch(
                "models", rs, failed_chunks,
                point_fn=lambda r: [("summit", r["summit_m"])] if r["summit_m"] is not None else [],
                models_fn=models_for_key,
                group_fn=lambda r: "alpine" if r["country"] in alpine else "other")
            for rid, labs in res.items():
                om.setdefault(rid, {})["models_summit"] = labs.get("summit")
        else:
            warnings.append(f"Model agreement skipped: {why}")
        models_state["ok"] = models_ok

    _run_chains(resorts, warnings, provider_status, [
        (measured_chain, phase_deadline),
        (reports_chain, phase_deadline),
        (openmeteo_chain, None),
    ])
    _PHASE_DEADLINE = None
    models_ok = models_state["ok"]

    # 5-6. derive + assemble. One resort's malformed forecast must not stop the other resorts:
    # first drop only its model-agreement answer, then all of its Open-Meteo data, so it
    # is written with forecast_ok false and its measured data intact.
    out_resorts = []
    for r in resorts:
        notes0 = list(r["notes"])
        try:
            out_resorts.append(assemble_resort(r, om, started, models_ok, models_for))
            continue
        except Exception as e:
            logger.exception(f"{r['name']}: assembling the forecast failed")
            err = f"{type(e).__name__}: {e}"
        labels = om.get(r["id"], {})
        for keep in ([k for k in labels if k != "models_summit"], []):
            r["notes"] = notes0 + [f"Open-Meteo answer for this resort could not be processed ({err}); "
                                   + ("model agreement dropped" if keep else "forecast not used")]
            try:
                out_resorts.append(assemble_resort(r, {r["id"]: {k: labels[k] for k in keep}}, started,
                                                   models_ok and bool(keep), models_for))
                warnings.append(f"{r['name']}: Open-Meteo answer could not be processed ({err})")
                break
            except Exception as e2:
                logger.exception(f"{r['name']}: assembling without that data failed too")
                err = f"{type(e2).__name__}: {e2}"
        else:
            raise RuntimeError(f"{r['name']}: resort could not be assembled even without forecast data")

    finished = datetime.now(timezone.utc)
    by_provider = {"SNOTEL": 0, "GeoSphere": 0, "SLF": 0, "OnTheSnow": 0, "tenki.jp": 0,
                   "JMA": 0, "Meteo-France": 0, "Meteocat": 0, "Tirol": 0, "AEMET": 0}
    for r in out_resorts:
        p = r["depth"]["provider"]
        if p in by_provider:
            by_provider[p] += 1
    n_fail = sum(1 for r in out_resorts if not r["quality"]["forecast_ok"])
    if n_fail:
        warnings.append(f"Open-Meteo forecast incomplete for {n_fail} of {len(out_resorts)} resorts")
    for prov in STATION_PROVIDERS:
        assigned = [r for r in out_resorts if any(s["provider"] == prov for s in r["stations"])]
        ok = [r for r in assigned if any(s["provider"] == prov and s["depth_cm"] is not None for s in r["stations"])]
        if len(assigned) >= 5 and not ok:
            warnings.append(f"{prov} returned no depth for any of {len(assigned)} resorts")

    obj = {
        "schema_version": SCHEMA_VERSION,
        "updated": utc_iso(finished),
        "run": {
            "started_utc": utc_iso(started),
            "finished_utc": utc_iso(finished),
            "duration_s": int((finished - started).total_seconds()),
            "resort_count": len(out_resorts),
            "open_meteo": {"requests": OM_COUNTER.requests, "retries": OM_COUNTER.retries,
                           "failed_chunks": failed_chunks},
            "coverage": {
                "forecast": sum(1 for r in out_resorts if r["quality"]["forecast_ok"]),
                "depth": sum(1 for r in out_resorts if r["depth"]["value_cm"] is not None),
                "depth_by_provider": by_provider,
                "stations_measured": sum(1 for r in out_resorts for s in r["stations"] if s["depth_cm"] is not None),
                "recent_measured": sum(1 for r in out_resorts if r["recent"]["measured"] is not None),
            },
            "warnings": warnings,
            "provider_status": dict(provider_status),
        },
        "thresholds": THRESHOLDS,
        "sources": sources_block(out_resorts),
        "resorts": out_resorts,
    }

    # 7. validate
    expected = full_count if not args.only else len(out_resorts)
    errors = validate_forecasts_v2.validate(obj, expected_count=expected)
    logger.info(f"Provider status: {provider_status}")
    logger.info(f"Coverage: {obj['run']['coverage']}")
    logger.info(f"Open-Meteo: {obj['run']['open_meteo']}")
    for w in warnings:
        logger.warning(f"RUN WARNING: {w}")
    if errors:
        for e in errors[:50]:
            logger.error(f"VALIDATION: {e}")
        logger.error(f"Validation failed with {len(errors)} error(s); file NOT written")
        return 2

    # 8. write
    if not args.no_write:
        write_atomic(obj, Path(args.out))
        logger.info(f"Wrote {args.out} ({Path(args.out).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
