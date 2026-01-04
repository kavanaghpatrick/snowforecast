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
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# USA - SNOTEL stations (pre-computed mappings)
# Format: (name, lat, lon, region, elevation_m, station_id, station_name)
# =============================================================================
US_SKI_AREAS = [
    ("Stevens Pass", 47.7448, -121.089, "Washington", 1241, "791:WA:SNTL", "Stevens Pass"),
    ("Crystal Mountain", 46.9282, -121.5045, "Washington", 2134, "679:WA:SNTL", "Morse Lake"),
    ("Mt. Baker", 48.857, -121.6695, "Washington", 1524, "909:WA:SNTL", "Wells Creek"),
    ("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 1067, "817:WA:SNTL", "Stampede Pass"),
    ("Mt. Hood Meadows", 45.3311, -121.6647, "Oregon", 1829, "651:OR:SNTL", "Mt Hood Test Site"),
    ("Mt. Bachelor", 43.9792, -121.6886, "Oregon", 2743, "729:OR:SNTL", "Santiam Jct"),
    ("Timberline", 45.3309, -121.7109, "Oregon", 1829, "651:OR:SNTL", "Mt Hood Test Site"),
    ("Mammoth Mountain", 37.6308, -119.0326, "California", 3369, "574:CA:SNTL", "Mammoth Pass"),
    ("Squaw Valley", 39.1969, -120.2358, "California", 2500, "784:CA:SNTL", "Squaw Valley G.C."),
    ("Heavenly", 38.9353, -119.9396, "California", 3060, "473:CA:SNTL", "Heavenly Valley"),
    ("Kirkwood", 38.6848, -120.0655, "California", 2377, "518:CA:SNTL", "Kirkwood"),
    ("Vail", 39.6403, -106.3742, "Colorado", 3527, "842:CO:SNTL", "Vail Mountain"),
    ("Breckenridge", 39.4817, -106.0384, "Colorado", 3914, "415:CO:SNTL", "Copper Mountain"),
    ("Aspen Snowmass", 39.2084, -106.949, "Colorado", 3813, "505:CO:SNTL", "Independence Pass"),
    ("Telluride", 37.9375, -107.8123, "Colorado", 3831, "797:CO:SNTL", "Telluride"),
    ("Park City", 40.6514, -111.508, "Utah", 3049, "628:UT:SNTL", "Mill D North"),
    ("Snowbird", 40.583, -111.6508, "Utah", 3353, "766:UT:SNTL", "Snowbird"),
    ("Alta", 40.5884, -111.6386, "Utah", 3215, "332:UT:SNTL", "Alta"),
    ("Big Sky", 45.2618, -111.4018, "Montana", 3403, "609:MT:SNTL", "Shower Falls"),
    ("Whitefish", 48.4833, -114.355, "Montana", 2133, "656:MT:SNTL", "Noisy Basin"),
    ("Jackson Hole", 43.5875, -110.8281, "Wyoming", 3185, "481:WY:SNTL", "Granite Creek"),
    ("Sun Valley", 43.6804, -114.4075, "Idaho", 2789, "489:ID:SNTL", "Hyndman"),
]

# =============================================================================
# AUSTRIA - GeoSphere TAWES mountain stations (using zamg library)
# Format: (name, lat, lon, region, elevation_m, station_id, station_name)
# IMPORTANT: Use MOUNTAIN station IDs, not town stations!
# =============================================================================
AUSTRIA_SKI_AREAS = [
    ("St. Anton", 47.1297, 10.2303, "Tyrol", 2079, "11110", "Galzig"),
    ("Sölden", 46.9128, 10.8617, "Tyrol", 3437, "11318", "Brunnenkogel"),
    ("Obergurgl", 46.8669, 11.0244, "Tyrol", 1941, "11127", "Obergurgl"),
    ("Ischgl", 46.9681, 10.1856, "Tyrol", 1587, "11312", "Galtür"),
    ("Kitzbühel", 47.4183, 12.3592, "Tyrol", 1794, "8989044", "Hahnenkamm"),
    ("Lech am Arlberg", 47.1575, 10.2128, "Vorarlberg", 2805, "11308", "Warth"),
    ("Pitztal Glacier", 46.9269, 10.8792, "Tyrol", 2864, "11316", "Pitztaler Gletscher"),
    ("Obertauern", 47.2489, 13.5597, "Salzburg", 1437, "11222", "Flattnitz"),
    ("Zell am See", 47.3286, 12.7381, "Salzburg", 1956, "11340", "Schmittenhöhe"),
    ("Schladming", 47.4678, 13.6264, "Styria", 2520, "11268", "Dachstein-Schladminger Gletscher"),
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
    ("Davos", 46.8131, 9.8439, "Graubünden", 1563, "SLF2", "Davos Stilli"),
    ("Laax", 46.8356, 9.2317, "Graubünden", 2325, "CMA2", "La Fuorcla"),
    ("Lenzerheide", 46.7267, 9.5542, "Graubünden", 2429, "PMA2", "Colms da Parsonz"),
    ("Mürren", 46.5573, 7.8352, "Bern", 2332, "SCH2", "Türliboden"),
    ("Andermatt", 46.6544, 8.6106, "Uri", 2209, "GOS3", "Gütsch"),
    ("Arosa", 46.7594, 9.6669, "Graubünden", 2495, "ROT3", "Plang Bi"),
    ("Engelberg", 46.7711, 8.4306, "Uri", 2149, "TIT2", "Titlisboden"),
]


def get_snotel_snow_depth(station_id, station_name):
    """Get current snow depth from a SNOTEL station via metloom."""
    try:
        from metloom.pointdata import SnotelPointData
        from metloom.variables import SnotelVariables

        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        station = SnotelPointData(station_id, station_name)
        df = station.get_daily_data(start_date, end_date, [SnotelVariables.SNOWDEPTH])

        if df is None or df.empty:
            return None

        # Get most recent non-null value - metloom returns inches, convert to cm
        for col in df.columns:
            if 'snow' in col.lower() or 'depth' in col.lower():
                values = df[col].dropna()
                if not values.empty:
                    snow_depth_inches = values.iloc[-1]
                    return round(snow_depth_inches * 2.54, 1)
        return None
    except Exception as e:
        logger.warning(f"SNOTEL error for {station_name}: {e}")
        return None


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
    """Fetch forecast from Open-Meteo API with retry logic."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m"
        f"&daily=snowfall_sum"
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


def process_region(ski_areas, country, snow_depth_func, use_austria_fallback=False):
    """Process a region's ski areas and return resort data.

    Args:
        ski_areas: List of resort tuples
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

        resort = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": country,
            "region": region,
            "elevation_m": elev,
            "base_depth_cm": base_depth_cm,
            "base_depth_source": station_name if base_depth_cm is not None else None,
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "forecast": forecast,
            "seven_day_total_cm": round(total_snow, 1),
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

    # Process USA (SNOTEL)
    logger.info("\n--- USA (SNOTEL) ---")
    resorts, success = process_region(US_SKI_AREAS, "USA", get_snotel_snow_depth)
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

    # Build output
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "base_depth": {
                "USA": "SNOTEL (nrcs.usda.gov)",
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
