#!/usr/bin/env python3
"""Hybrid snow depth + Open-Meteo forecast fetcher for Dashboard V2.

Data sources by region:
- USA: SNOTEL (NRCS) for base depth + Open-Meteo for forecasts
- Austria: GeoSphere Austria TAWES for base depth + Open-Meteo for forecasts
- Switzerland: SLF IMIS for base depth + Open-Meteo for forecasts

Usage:
    python scripts/fetch_data_v2.py
"""

import json
import logging
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add parent directory to path for snowforecast imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snowforecast.cache.elevation_bands import (
    get_summit_elevation,
    apply_lapse_rate,
    get_precip_type,
    adjust_new_snow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# USA - SNOTEL stations with multi-station averaging
# Format: (name, lat, lon, region, elevation_m, stations_list)
# stations_list: [(station_id, station_name, weight), ...]
# =============================================================================
US_SKI_AREAS = [
    # Washington
    ("Stevens Pass", 47.7448, -121.089, "Washington", 1238, [  # Base: 4,061ft
        ("791:WA:SNTL", "Stevens Pass", 1.0),
        ("672:WA:SNTL", "Olallie Meadows", 0.5),
    ]),
    ("Crystal Mountain", 46.9282, -121.5045, "Washington", 1341, [  # Base: 4,400ft (was 2134 summit)
        ("679:WA:SNTL", "Morse Lake", 1.0),
    ]),
    ("Mt. Baker", 48.857, -121.6695, "Washington", 1067, [  # Base: 3,500ft (was 1524 mid-mtn)
        ("909:WA:SNTL", "Wells Creek", 1.0),
    ]),
    ("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 945, [  # Base: 3,100ft (was 1067)
        ("817:WA:SNTL", "Stampede Pass", 1.0),
    ]),
    # Oregon
    ("Mt. Hood Meadows", 45.3311, -121.6647, "Oregon", 1379, [  # Base: 4,523ft (was 1829)
        ("651:OR:SNTL", "Mt Hood Test Site", 1.0),
    ]),
    ("Mt. Bachelor", 43.9792, -121.6886, "Oregon", 1920, [  # Base: 6,300ft (was 2743 summit)
        ("729:OR:SNTL", "Santiam Jct", 1.0),
    ]),
    ("Timberline", 45.3309, -121.7109, "Oregon", 1829, [  # Lodge elevation: 6,000ft (correct)
        ("651:OR:SNTL", "Mt Hood Test Site", 1.0),
    ]),
    # California
    ("Mammoth Mountain", 37.6308, -119.0326, "California", 2424, [  # Base: 7,953ft (was 3369 summit)
        ("846:CA:SNTL", "Virginia Lakes Ridge", 1.0),  # Closest available (~31mi)
    ]),
    ("Palisades Tahoe", 39.1969, -120.2358, "California", 1890, [  # Base: 6,200ft (was 2500)
        ("784:CA:SNTL", "Palisades Tahoe", 1.0),
        ("809:CA:SNTL", "Tahoe City Cross", 0.7),
        ("541:CA:SNTL", "Independence Lake", 0.5),
    ]),
    ("Heavenly", 38.9353, -119.9396, "California", 1906, [  # Base: 6,255ft (was 3060 summit)
        ("473:CA:SNTL", "Heavenly Valley", 1.0),
    ]),
    ("Kirkwood", 38.6848, -120.0655, "California", 2377, [  # Base: 7,800ft (correct)
        ("518:CA:SNTL", "Kirkwood", 1.0),
    ]),
    # Colorado
    ("Vail", 39.6403, -106.3742, "Colorado", 2454, [  # Base: 8,120ft (was 3527 summit)
        ("842:CO:SNTL", "Vail Mountain", 1.0),
        ("1041:CO:SNTL", "Beaver Ck Village", 0.7),
        ("415:CO:SNTL", "Copper Mountain", 0.5),
    ]),
    ("Breckenridge", 39.4817, -106.0384, "Colorado", 2926, [  # Base: 9,600ft (was 3914 summit)
        ("415:CO:SNTL", "Copper Mountain", 1.0),
    ]),
    ("Aspen Snowmass", 39.2084, -106.949, "Colorado", 2623, [  # Base: 8,604ft (was 3813 summit)
        ("505:CO:SNTL", "Independence Pass", 1.0),
    ]),
    ("Telluride", 37.9375, -107.8123, "Colorado", 2659, [  # Base: 8,725ft (was 3831 summit)
        ("797:CO:SNTL", "Telluride", 1.0),
    ]),
    # Utah
    ("Park City", 40.6514, -111.508, "Utah", 2103, [  # Base: 6,900ft (was 3049 summit)
        ("628:UT:SNTL", "Mill D North", 1.0),
    ]),
    ("Snowbird", 40.583, -111.6508, "Utah", 2365, [  # Base: 7,760ft (was 3353 summit)
        ("766:UT:SNTL", "Snowbird", 1.0),
        ("366:UT:SNTL", "Brighton", 0.7),
    ]),
    ("Alta", 40.5884, -111.6386, "Utah", 2600, [  # Base: 8,530ft (was 3215)
        ("332:UT:SNTL", "Alta", 1.0),
        ("766:UT:SNTL", "Snowbird", 0.8),
    ]),
    # Montana
    ("Big Sky", 45.2618, -111.4018, "Montana", 2290, [  # Base: 7,500ft (was 3403 summit)
        ("609:MT:SNTL", "Shower Falls", 1.0),
    ]),
    ("Whitefish", 48.4833, -114.355, "Montana", 1361, [  # Base: 4,464ft (was 2133 summit)
        ("656:MT:SNTL", "Noisy Basin", 1.0),
    ]),
    # Wyoming
    ("Jackson Hole", 43.5875, -110.8281, "Wyoming", 1924, [  # Base: 6,311ft (was 3185 summit)
        ("481:WY:SNTL", "Granite Creek", 1.0),
    ]),
    # Idaho
    ("Sun Valley", 43.6804, -114.4075, "Idaho", 1753, [  # Base: 5,750ft (was 2789 summit)
        ("489:ID:SNTL", "Hyndman", 1.0),
    ]),
]

# =============================================================================
# AUSTRIA - GeoSphere TAWES mountain stations (using zamg library)
# Format: (name, lat, lon, region, elevation_m, station_id, station_name)
# IMPORTANT: Use MOUNTAIN station IDs, not town stations!
# =============================================================================
AUSTRIA_SKI_AREAS = [
    ("St. Anton", 47.1297, 10.2303, "Tyrol", 2079, "11110", "Galzig"),
    ("Sölden", 46.9128, 10.8617, "Tyrol", 3340, "11318", "Brunnenkogel"),  # Summit: 3,340m (was 3437)
    ("Obergurgl", 46.8669, 11.0244, "Tyrol", 1941, "11127", "Obergurgl"),
    ("Ischgl", 46.9681, 10.1856, "Tyrol", 1587, "11312", "Galtür"),
    ("Kitzbühel", 47.4183, 12.3592, "Tyrol", 1794, "8989044", "Hahnenkamm"),
    ("Lech am Arlberg", 47.1575, 10.2128, "Vorarlberg", 2079, "11110", "Galzig (Ski Arlberg)"),
    ("Pitztal Glacier", 46.9269, 10.8792, "Tyrol", 2864, "11316", "Pitztaler Gletscher"),
    ("Obertauern", 47.2489, 13.5597, "Salzburg", 1630, "11222", "Flattnitz"),  # Base: 1,630m (was 1437)
    ("Zell am See", 47.3286, 12.7381, "Salzburg", 1956, "11340", "Schmittenhöhe"),
    ("Schladming", 47.4678, 13.6264, "Styria", 2015, "11268", "Dachstein-Schladminger Gletscher"),  # Summit: 2,015m (was 2520)
]

# =============================================================================
# SWITZERLAND - SLF IMIS stations (using snow sensor stations - type 2/3)
# Format: (name, lat, lon, region, elevation_m, station_id, station_name)
# =============================================================================
SWISS_SKI_AREAS = [
    ("Zermatt", 45.9872, 7.7836, "Valais", 2953, "GOR2", "Gornergratsee"),
    ("Saas-Fee", 46.1275, 7.9814, "Valais", 2480, "SAA2", "Seetal"),
    ("Verbier", 46.0989, 7.2856, "Valais", 2550, "ATT2", "Lac des Vaux"),
    ("St. Moritz", 46.4761, 9.8438, "Graubünden", 2512, "BEV2", "Valetta"),
    ("Davos", 46.8131, 9.8439, "Graubünden", 2540, "WFJ2", "Weissfluhjoch"),
    ("Laax", 46.8356, 9.2317, "Graubünden", 2325, "CMA2", "La Fuorcla"),
    ("Lenzerheide", 46.7267, 9.5542, "Graubünden", 2429, "PMA2", "Colms da Parsonz"),
    ("Mürren", 46.5573, 7.8352, "Bern", 2332, "SCH2", "Türliboden"),
    ("Andermatt", 46.6544, 8.6106, "Uri", 2209, "GOS3", "Gütsch"),
    ("Arosa", 46.7594, 9.6669, "Graubünden", 2495, "ROT3", "Plang Bi"),
    ("Engelberg", 46.7711, 8.4306, "Uri", 2149, "TIT2", "Titlisboden"),
]

# =============================================================================
# JAPAN - OnTheSnow for snow depth + Open-Meteo for forecasts
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

# =============================================================================
# OnTheSnow.com URL slugs for scraping resort-reported base depths
# Pattern: https://www.onthesnow.com/{slug}/skireport
# =============================================================================
ONTHESNOW_SLUGS = {
    # USA - Washington
    "Stevens Pass": "washington/stevens-pass-resort",
    "Crystal Mountain": "washington/crystal-mountain-wa",
    "Mt. Baker": "washington/mt-baker",
    "Snoqualmie Pass": "washington/the-summit-at-snoqualmie",
    # USA - Oregon
    "Mt. Hood Meadows": "oregon/mt-hood-meadows",
    "Mt. Bachelor": "oregon/mt-bachelor",
    "Timberline": "oregon/timberline-lodge",
    # USA - California
    "Mammoth Mountain": "california/mammoth-mountain-ski-area",
    "Palisades Tahoe": "california/palisades-tahoe",
    "Heavenly": "california/heavenly-mountain-resort",
    "Kirkwood": "california/kirkwood",
    # USA - Colorado
    "Vail": "colorado/vail",
    "Breckenridge": "colorado/breckenridge",
    "Aspen Snowmass": "colorado/aspen-snowmass",
    "Telluride": "colorado/telluride",
    # USA - Utah
    "Park City": "utah/park-city-mountain-resort",
    "Snowbird": "utah/snowbird",
    "Alta": "utah/alta-ski-area",
    # USA - Montana
    "Big Sky": "montana/big-sky-resort",
    "Whitefish": "montana/whitefish-mountain-resort",
    # USA - Wyoming
    "Jackson Hole": "wyoming/jackson-hole",
    # USA - Idaho
    "Sun Valley": "idaho/sun-valley",
    # Japan - Hokkaido
    "Niseko United": "hokkaido/niseko-united",
    "Furano": "hokkaido/furano-ski-resort",
    "Rusutsu": "hokkaido/rusutsu",
    "Kiroro": "hokkaido/kiroro",
    # Japan - Nagano
    "Hakuba Valley": "nagano/hakuba-valley",
    "Happo-One": "nagano/happo-one",
    "Nozawa Onsen": "nagano/nozawa-onsen",
    "Shiga Kogen": "nagano/shiga-kogen",
    # Japan - Niigata
    "Myoko Kogen": "niigata/myoko-kogen",
    # Japan - Yamagata
    "Zao Onsen": "yamagata/zao-onsen",
}

# snow-forecast.com slugs for Japan resorts (fallback for OnTheSnow gaps)
SNOW_FORECAST_SLUGS = {
    "Happo-One": "Happo-One",
    "Nozawa Onsen": "Nozawa-Onsen",
    "Shiga Kogen": "ShigaKogenGiant",  # Main Shiga resort
    "Myoko Kogen": "MyokoSuginohara",  # Largest Myoko resort
    "Zao Onsen": "Yamagata-Zao-Onsen",
}

# Resorts that share the same ski area (use data from parent resort)
# Happo-One is the main resort within Hakuba Valley
JAPAN_RESORT_ALIASES = {
    "Happo-One": "Hakuba Valley",  # Happo-One is the flagship resort of Hakuba Valley
}

# tenki.jp resort IDs - reliable Japanese source with server-rendered HTML
# URL pattern: https://tenki.jp/season/ski/{region}/{prefecture}/{resort_id}/
# Updates twice daily at 9:00 and 15:00 JST
TENKI_JP_RESORTS = {
    # Hokkaido (region=1, prefecture=2)
    "Niseko United": "1/2/11708",      # Niseko Grand Hirafu
    "Furano": "1/2/11717",             # Furano Ski Resort
    "Rusutsu": "1/2/11718",            # Rusutsu Resort
    "Kiroro": "1/2/11711",             # Kiroro Snow World
    # Nagano (region=3, prefecture=23)
    "Hakuba Valley": "3/23/15101",     # Hakuba Goryu
    "Happo-One": "3/23/15100",         # Hakuba Happo-One
    "Nozawa Onsen": "3/23/15115",      # Nozawa Onsen
    "Shiga Kogen": "3/23/15123",       # Shiga Kogen Yokoteyama
    # Niigata (region=4, prefecture=18)
    "Myoko Kogen": "4/18/13142",       # Myoko Suginohara
    # Yamagata (region=2, prefecture=9)
    "Zao Onsen": "2/9/15191",          # Zao Onsen
}


def scrape_onthesnow(resort_name: str) -> tuple[float | None, str | None]:
    """Scrape OnTheSnow for resort-reported base depth.

    OnTheSnow uses Next.js SSR - snow data is in <script id="__NEXT_DATA__"> JSON.

    Args:
        resort_name: Name of the resort (must be in ONTHESNOW_SLUGS)

    Returns:
        (depth_cm, source) tuple, or (None, None) if scraping fails
    """
    slug = ONTHESNOW_SLUGS.get(resort_name)
    if not slug:
        return None, None

    url = f"https://www.onthesnow.com/{slug}/skireport"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')

        # Extract __NEXT_DATA__ JSON
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', html, re.DOTALL)
        if not match:
            logger.debug(f"OnTheSnow: No __NEXT_DATA__ for {resort_name}")
            return None, None

        data = json.loads(match.group(1))
        resort = data.get('props', {}).get('pageProps', {}).get('fullResort', {})
        snow = resort.get('snow', {})

        # Try middle (mid-mountain), then base, then summit
        depth = snow.get('middle') or snow.get('base') or snow.get('summit')

        # Validate: reject obviously erroneous values (>600cm = 6 meters)
        # Example: Niseko returns 1108cm from OnTheSnow but actual is 80-235cm
        if depth is not None and 0 < depth < 600:
            return round(depth, 1), "OnTheSnow"
        elif depth is not None and depth >= 600:
            logger.warning(f"OnTheSnow: Rejecting implausible depth {depth}cm for {resort_name}")

        return None, None

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.debug(f"OnTheSnow HTTP error for {resort_name}: {e}")
        return None, None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"OnTheSnow parse error for {resort_name}: {e}")
        return None, None
    except (TimeoutError, OSError) as e:
        logger.debug(f"OnTheSnow timeout for {resort_name}: {e}")
        return None, None


def scrape_snow_forecast(resort_name: str) -> tuple[float | None, str | None]:
    """Scrape snow-forecast.com for resort snow depth (fallback for Japan).

    snow-forecast.com has consistent page structure with snow depth in text.
    We extract the "top" snow depth which is most relevant for skiing.

    Args:
        resort_name: Name of the resort (must be in SNOW_FORECAST_SLUGS)

    Returns:
        (depth_cm, source) tuple, or (None, None) if scraping fails
    """
    slug = SNOW_FORECAST_SLUGS.get(resort_name)
    if not slug:
        return None, None

    url = f"https://www.snow-forecast.com/resorts/{slug}/6day/top"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')

        # Look for snow depth pattern: "X.Xm" or "Xcm" in the snow depth section
        # The page shows depth like "Top: 2.4m" or similar
        # Pattern for meters (e.g., "2.4m", "1.7m")
        match_m = re.search(r'(?:top|summit)[^0-9]*?(\d+\.?\d*)\s*m(?:eter|etre)?s?\b', html, re.IGNORECASE)
        if match_m:
            depth_m = float(match_m.group(1))
            depth_cm = depth_m * 100
            return round(depth_cm, 1), "snow-forecast.com"

        # Pattern for centimeters (e.g., "240cm")
        match_cm = re.search(r'(?:top|summit)[^0-9]*?(\d+)\s*cm\b', html, re.IGNORECASE)
        if match_cm:
            depth_cm = float(match_cm.group(1))
            return round(depth_cm, 1), "snow-forecast.com"

        # Fallback: look for any depth value in meters on the page
        match_any = re.search(r'(\d+\.?\d*)\s*m\s+(?:top|at top)', html, re.IGNORECASE)
        if match_any:
            depth_m = float(match_any.group(1))
            depth_cm = depth_m * 100
            return round(depth_cm, 1), "snow-forecast.com"

        logger.debug(f"snow-forecast.com: No depth pattern found for {resort_name}")
        return None, None

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.debug(f"snow-forecast.com HTTP error for {resort_name}: {e}")
        return None, None
    except Exception as e:
        logger.debug(f"snow-forecast.com error for {resort_name}: {e}")
        return None, None


def scrape_tenki_jp(resort_name: str) -> tuple[float | None, str | None]:
    """Scrape tenki.jp for Japanese ski resort snow depth.

    tenki.jp uses server-rendered HTML (no JS needed) with consistent structure.
    Updates twice daily at 9:00 and 15:00 JST.

    Args:
        resort_name: Name of the resort (must be in TENKI_JP_RESORTS)

    Returns:
        (depth_cm, source) tuple, or (None, None) if scraping fails
    """
    resort_path = TENKI_JP_RESORTS.get(resort_name)
    if not resort_path:
        return None, None

    url = f"https://tenki.jp/season/ski/{resort_path}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "ja,en;q=0.9",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')

        # Look for 積雪 (sekisetsu = snow depth) patterns
        # Primary: <div class="depth">積雪<br><span>135cm</span></div>
        match = re.search(r'class="depth">積雪<br><span>(\d+)cm</span>', html)
        if match:
            depth_cm = float(match.group(1))
            if 0 < depth_cm < 600:
                return depth_cm, "tenki.jp"
            else:
                logger.warning(f"tenki.jp: Rejecting implausible depth {depth_cm}cm for {resort_name}")
                return None, None

        # Alternative: 積雪<br>XXXcm (without span)
        match_alt = re.search(r'積雪<br>(\d+)cm', html)
        if match_alt:
            depth_cm = float(match_alt.group(1))
            if 0 < depth_cm < 600:
                return depth_cm, "tenki.jp"

        # Fallback: 積雪 followed by number somewhere
        match_fallback = re.search(r'積雪[^0-9]*(\d+)\s*cm', html)
        if match_fallback:
            depth_cm = float(match_fallback.group(1))
            if 0 < depth_cm < 600:
                return depth_cm, "tenki.jp"

        logger.debug(f"tenki.jp: No snow depth found for {resort_name}")
        return None, None

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.debug(f"tenki.jp HTTP error for {resort_name}: {e}")
        return None, None
    except (TimeoutError, OSError) as e:
        logger.debug(f"tenki.jp timeout for {resort_name}: {e}")
        return None, None
    except Exception as e:
        logger.debug(f"tenki.jp error for {resort_name}: {e}")
        return None, None


def get_snotel_snow_depth_single(station_id, station_name):
    """Get current snow depth from a single SNOTEL station via metloom."""
    # Import at module level would be better, but keeping here for isolation
    try:
        from metloom.pointdata import SnotelPointData
        from metloom.variables import SnotelVariables
    except ImportError:
        logger.error("metloom library not installed - run: pip install metloom")
        return None

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        station = SnotelPointData(station_id, station_name)
        df = station.get_daily_data(start_date, end_date, [SnotelVariables.SNOWDEPTH])

        if df is None or df.empty:
            return None

        if 'SNOWDEPTH' in df.columns:
            values = df['SNOWDEPTH'].dropna()
            if not values.empty:
                snow_depth_inches = float(values.iloc[-1])
                return round(snow_depth_inches * 2.54, 1)  # Convert to cm

        return None

    except Exception as e:
        logger.warning(f"SNOTEL error for {station_name}: {e}")
        return None


def get_snotel_multi_station(stations_list):
    """Get weighted average snow depth from multiple SNOTEL stations.

    Args:
        stations_list: List of (station_id, station_name, weight) tuples

    Returns:
        (depth_cm, source_description) tuple
    """
    if not stations_list:
        return None, None

    depths = []
    sources = []

    for station_tuple in stations_list:
        # Validate tuple has 3 elements
        if len(station_tuple) != 3:
            logger.warning(f"Invalid station tuple: {station_tuple}")
            continue
        station_id, station_name, weight = station_tuple

        depth = get_snotel_snow_depth_single(station_id, station_name)
        if depth is not None and weight > 0:
            depths.append((depth, weight))
            sources.append(station_name)

    if not depths:
        return None, None

    # Weighted average (safe - we only add if weight > 0)
    total = sum(d * w for d, w in depths)
    total_weight = sum(w for _, w in depths)
    avg_depth = round(total / total_weight, 1)

    # Build source description
    if len(sources) == 1:
        source_desc = sources[0]
    else:
        source_desc = f"{sources[0]} +{len(sources)-1}"  # e.g. "Snowbird +1"

    return avg_depth, source_desc


def get_geosphere_snow_depth(station_id, station_name):
    """Get current snow depth from GeoSphere Austria TAWES station."""
    url = (
        f"https://dataset.api.hub.geosphere.at/v1/station/current/tawes-v1-10min"
        f"?parameters=SCHNEE&station_ids={station_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read())
            features = data.get("features", [])
            if features:
                params = features[0].get("properties", {}).get("parameters", {})
                schnee = params.get("SCHNEE", {}).get("data", [])
                if schnee and schnee[0] is not None:
                    return round(schnee[0], 1), station_name
        return None, None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning(f"GeoSphere rate limited for {station_name}")
        else:
            logger.warning(f"GeoSphere HTTP error for {station_name}: {e}")
        return None, None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"GeoSphere error for {station_name}: {e}")
        return None, None


def get_openmeteo_snow_depth(lat, lon):
    """Get current snow depth from Open-Meteo model (fallback)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=snow_depth"
        f"&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read())
            snow_m = data.get("current", {}).get("snow_depth")
            if snow_m is not None:
                return round(snow_m * 100, 1)
        return None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        return None


def get_austria_snow_depth(station_id, station_name, lat, lon):
    """Get snow depth for Austria: try GeoSphere first, fall back to Open-Meteo."""
    # Try GeoSphere TAWES first (real station data)
    depth, source = get_geosphere_snow_depth(station_id, station_name)
    if depth is not None:
        return depth, source

    # Fall back to Open-Meteo model
    depth = get_openmeteo_snow_depth(lat, lon)
    if depth is not None:
        return depth, "Open-Meteo (modeled)"

    return None, None


def get_slf_snow_depth(station_id, station_name):
    """Get current snow depth from SLF IMIS station (Switzerland)."""
    url = "https://measurement-api.slf.ch/public/api/imis/measurements"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read())
            # Find latest measurement for this station
            station_data = [d for d in data if d.get("station_code") == station_id]
            if station_data:
                # Get most recent measurement with HS (snow height)
                latest = max(station_data, key=lambda x: x.get("measure_date", ""))
                hs = latest.get("HS")
                if hs is not None:
                    return round(hs, 1)
        return None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"SLF error for {station_name}: {e}")
        return None


def fetch_open_meteo(lat: float, lon: float, retries: int = 3) -> dict | None:
    """Fetch forecast from Open-Meteo API with retry logic.

    Uses GFS Seamless model which combines:
    - HRRR (3km, hourly) for 0-48 hour forecasts
    - GFS (13-25km) for extended forecasts
    """
    url = (
        f"https://api.open-meteo.com/v1/gfs?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,snow_depth"
        f"&daily=snowfall_sum,precipitation_sum"
        f"&timezone=auto"
        f"&forecast_days=7"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            logger.warning(f"Open-Meteo error after {retries} retries: {e}")
            return None


def process_us_region(ski_areas):
    """Process US ski areas with OnTheSnow scraping + SNOTEL fallback.

    Args:
        ski_areas: List of (name, lat, lon, region, elev, stations_list) tuples
    """
    resorts = []
    success_count = 0

    for name, lat, lon, region, elev, stations_list in ski_areas:
        logger.info(f"Processing {name} (USA)...")

        # Try OnTheSnow scraping first (resort-reported depths)
        base_depth_cm, source = scrape_onthesnow(name)
        if base_depth_cm is not None:
            logger.info(f"  OnTheSnow: {base_depth_cm}cm")
            success_count += 1
        else:
            # Fall back to SNOTEL (watershed snow sensors)
            base_depth_cm, snotel_source = get_snotel_multi_station(stations_list)
            if base_depth_cm is not None:
                source = f"SNOTEL {snotel_source}"
                logger.info(f"  SNOTEL ({snotel_source}): {base_depth_cm}cm")
                success_count += 1
            else:
                source = None
                snotel_name = stations_list[0][1] if stations_list else "Unknown"
                logger.warning(f"  No data (tried OnTheSnow + SNOTEL {snotel_name})")

        # Get forecast from Open-Meteo
        forecast_data = fetch_open_meteo(lat, lon)

        forecast = []
        total_snow = 0.0
        temp_c = None

        if forecast_data:
            temp_c = forecast_data.get("current", {}).get("temperature_2m")
            daily = forecast_data.get("daily", {})
            dates = daily.get("time", [])
            snowfall = daily.get("snowfall_sum", [])

            for i, (date_str, snow_cm) in enumerate(zip(dates, snowfall)):
                snow_val = snow_cm if snow_cm else 0.0
                total_snow += snow_val
                forecast.append({
                    "day": i,
                    "date": date_str,
                    "new_snow_cm": round(snow_val, 1),
                    "source": "open-meteo",
                })

        # Calculate summit elevation and temperature
        summit_elev = get_summit_elevation(name, elev)
        summit_temp_c = None
        if temp_c is not None:
            summit_temp_c = apply_lapse_rate(temp_c, elev, summit_elev)

        # Calculate 7-day summit snowfall using lapse rate physics
        summit_7day_total = 0.0
        if summit_temp_c is not None:
            for day_forecast in forecast:
                base_snow = day_forecast.get("new_snow_cm", 0)
                has_precip = base_snow > 0
                precip_type = get_precip_type(summit_temp_c, has_precip)
                summit_snow = adjust_new_snow(base_snow, summit_temp_c, precip_type)
                summit_7day_total += summit_snow
        else:
            # If no temp data, summit snow equals base snow
            summit_7day_total = total_snow

        resort = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": "USA",
            "region": region,
            "elevation_m": elev,
            "summit_elevation_m": summit_elev,
            "base_depth_cm": base_depth_cm,
            "base_depth_source": source,
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "summit_temp_c": round(summit_temp_c, 1) if summit_temp_c is not None else None,
            "forecast": forecast,
            "seven_day_total_cm": round(total_snow, 1),
            "summit_seven_day_total_cm": round(summit_7day_total, 1),
        }

        resorts.append(resort)
        time.sleep(0.3)  # Be nice to APIs

    return resorts, success_count


def process_region(ski_areas, country, snow_depth_func, use_austria_fallback=False):
    """Process a region's ski areas and return resort data (Austria/Switzerland).

    Args:
        ski_areas: List of (name, lat, lon, region, elev, station_id, station_name) tuples
        country: Country name
        snow_depth_func: Function to get snow depth
        use_austria_fallback: If True, use Austria's fallback logic (GeoSphere -> Open-Meteo)
    """
    resorts = []
    success_count = 0

    for name, lat, lon, region, elev, station_id, station_name in ski_areas:
        logger.info(f"Processing {name} ({country})...")

        # Get base depth from regional snow sensor network
        if use_austria_fallback:
            base_depth_cm, actual_source = snow_depth_func(station_id, station_name, lat, lon)
            if actual_source:
                station_name = actual_source  # Update source name for display
        else:
            base_depth_cm = snow_depth_func(station_id, station_name)
        if base_depth_cm is not None:
            logger.info(f"  Station ({station_name}): {base_depth_cm}cm")
            success_count += 1
        else:
            logger.warning(f"  Station ({station_name}): No data")

        # Get forecast from Open-Meteo (works globally)
        forecast_data = fetch_open_meteo(lat, lon)

        forecast = []
        total_snow = 0.0
        temp_c = None

        if forecast_data:
            temp_c = forecast_data.get("current", {}).get("temperature_2m")
            daily = forecast_data.get("daily", {})
            dates = daily.get("time", [])
            snowfall = daily.get("snowfall_sum", [])

            for i, (date_str, snow_cm) in enumerate(zip(dates, snowfall)):
                snow_val = snow_cm if snow_cm else 0.0
                total_snow += snow_val
                forecast.append({
                    "day": i,
                    "date": date_str,
                    "new_snow_cm": round(snow_val, 1),
                    "source": "open-meteo",
                })

        # Calculate summit elevation and temperature
        summit_elev = get_summit_elevation(name, elev)
        summit_temp_c = None
        if temp_c is not None:
            summit_temp_c = apply_lapse_rate(temp_c, elev, summit_elev)

        # Calculate 7-day summit snowfall using lapse rate physics
        summit_7day_total = 0.0
        if summit_temp_c is not None:
            for day_forecast in forecast:
                base_snow = day_forecast.get("new_snow_cm", 0)
                has_precip = base_snow > 0
                precip_type = get_precip_type(summit_temp_c, has_precip)
                summit_snow = adjust_new_snow(base_snow, summit_temp_c, precip_type)
                summit_7day_total += summit_snow
        else:
            # If no temp data, summit snow equals base snow
            summit_7day_total = total_snow

        resort = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": country,
            "region": region,
            "elevation_m": elev,
            "summit_elevation_m": summit_elev,
            "base_depth_cm": base_depth_cm,
            "base_depth_source": station_name if base_depth_cm is not None else None,
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "summit_temp_c": round(summit_temp_c, 1) if summit_temp_c is not None else None,
            "forecast": forecast,
            "seven_day_total_cm": round(total_snow, 1),
            "summit_seven_day_total_cm": round(summit_7day_total, 1),
        }

        resorts.append(resort)
        time.sleep(0.3)  # Be nice to APIs

    return resorts, success_count


def process_japan_region():
    """Process Japan ski areas using OnTheSnow for depth + Open-Meteo for forecasts."""
    resorts = []
    success_count = 0

    for name, lat, lon, region, elev in JAPAN_SKI_AREAS:
        logger.info(f"Processing {name} (Japan)...")

        # Get forecast from Open-Meteo first (we need it anyway)
        forecast_data = fetch_open_meteo(lat, lon)

        # Get base depth - try sources in order of reliability:
        # 1. OnTheSnow (resort-reported)
        # 2. OnTheSnow via alias (e.g., Happo-One → Hakuba Valley)
        # 3. tenki.jp (Japanese weather service)
        # Note: Open-Meteo modeled snow_depth is unreliable for mountains (7-8x errors)
        base_depth_cm, source = scrape_onthesnow(name)
        if base_depth_cm is not None:
            logger.info(f"  OnTheSnow: {base_depth_cm}cm")
            success_count += 1
        else:
            # Try alias resort (e.g., Happo-One → Hakuba Valley)
            alias = JAPAN_RESORT_ALIASES.get(name)
            if alias:
                base_depth_cm, _ = scrape_onthesnow(alias)
                if base_depth_cm is not None:
                    source = f"OnTheSnow ({alias})"
                    logger.info(f"  OnTheSnow (via {alias}): {base_depth_cm}cm")
                    success_count += 1

            # Fallback to tenki.jp (Japanese weather service)
            if base_depth_cm is None:
                base_depth_cm, source = scrape_tenki_jp(name)
                if base_depth_cm is not None:
                    logger.info(f"  tenki.jp: {base_depth_cm}cm")
                    success_count += 1
                else:
                    logger.warning(f"  No snow depth data for {name}")

        forecast = []
        total_snow = 0.0
        temp_c = None

        if forecast_data:
            temp_c = forecast_data.get("current", {}).get("temperature_2m")
            daily = forecast_data.get("daily", {})
            dates = daily.get("time", [])
            snowfall = daily.get("snowfall_sum", [])

            for i, (date_str, snow_cm) in enumerate(zip(dates, snowfall)):
                snow_val = snow_cm if snow_cm else 0.0
                total_snow += snow_val
                forecast.append({
                    "day": i,
                    "date": date_str,
                    "new_snow_cm": round(snow_val, 1),
                    "source": "open-meteo",
                })

        # Calculate summit elevation and temperature
        summit_elev = get_summit_elevation(name, elev)
        summit_temp_c = None
        if temp_c is not None:
            summit_temp_c = apply_lapse_rate(temp_c, elev, summit_elev)

        # Calculate 7-day summit snowfall using lapse rate physics
        summit_7day_total = 0.0
        if summit_temp_c is not None:
            for day_forecast in forecast:
                base_snow = day_forecast.get("new_snow_cm", 0)
                has_precip = base_snow > 0
                precip_type = get_precip_type(summit_temp_c, has_precip)
                summit_snow = adjust_new_snow(base_snow, summit_temp_c, precip_type)
                summit_7day_total += summit_snow
        else:
            summit_7day_total = total_snow

        resort = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": "Japan",
            "region": region,
            "elevation_m": elev,
            "summit_elevation_m": summit_elev,
            "base_depth_cm": base_depth_cm,
            "base_depth_source": source if base_depth_cm else None,
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "summit_temp_c": round(summit_temp_c, 1) if summit_temp_c is not None else None,
            "forecast": forecast,
            "seven_day_total_cm": round(total_snow, 1),
            "summit_seven_day_total_cm": round(summit_7day_total, 1),
        }

        resorts.append(resort)
        time.sleep(0.3)  # Be nice to APIs

    return resorts, success_count


def main():
    """Fetch all data and output JSON."""
    logger.info("=" * 50)
    logger.info("FETCH DATA V2 - Multi-Region Snow Data")
    logger.info("=" * 50)

    all_resorts = []
    stats = {}

    # Process USA (SNOTEL with multi-station averaging)
    logger.info("\n--- USA (SNOTEL multi-station) ---")
    resorts, success = process_us_region(US_SKI_AREAS)
    all_resorts.extend(resorts)
    stats["USA"] = {"total": len(US_SKI_AREAS), "success": success}

    # Process Austria (GeoSphere TAWES with Open-Meteo fallback)
    logger.info("\n--- Austria (GeoSphere TAWES + fallback) ---")
    resorts, success = process_region(AUSTRIA_SKI_AREAS, "Austria", get_austria_snow_depth, use_austria_fallback=True)
    all_resorts.extend(resorts)
    stats["Austria"] = {"total": len(AUSTRIA_SKI_AREAS), "success": success}

    # Process Switzerland (SLF IMIS)
    logger.info("\n--- Switzerland (SLF IMIS) ---")
    resorts, success = process_region(SWISS_SKI_AREAS, "Switzerland", get_slf_snow_depth)
    all_resorts.extend(resorts)
    stats["Switzerland"] = {"total": len(SWISS_SKI_AREAS), "success": success}

    # Process Japan (OnTheSnow + Open-Meteo)
    logger.info("\n--- Japan (OnTheSnow + Open-Meteo) ---")
    resorts, success = process_japan_region()
    all_resorts.extend(resorts)
    stats["Japan"] = {"total": len(JAPAN_SKI_AREAS), "success": success}

    # Build output
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "base_depth": {
                "USA": "OnTheSnow (onthesnow.com) + SNOTEL fallback",
                "Austria": "GeoSphere ZAMG (geosphere.at)",
                "Switzerland": "SLF IMIS (slf.ch)",
            },
            "forecast": "Open-Meteo (open-meteo.com)",
        },
        "resorts": all_resorts,
    }

    # Write to data directory
    out_path = Path(__file__).parent.parent / "data" / "forecasts_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nOutput written to: {out_path}")
    logger.info(f"Total resorts: {len(all_resorts)}")
    for country, s in stats.items():
        logger.info(f"  {country}: {s['success']}/{s['total']} stations reporting")
    avg_snow = sum(r['seven_day_total_cm'] for r in all_resorts) / len(all_resorts)
    logger.info(f"Average 7-day snowfall: {avg_snow:.1f}cm")
    logger.info("Done!")


if __name__ == "__main__":
    main()
