#!/usr/bin/env python3
"""Hybrid SNOTEL + Open-Meteo forecast fetcher for Dashboard V2.

Uses SNOTEL for real base depth measurements + Open-Meteo for 7-day forecasts.
SNOTEL station mappings are pre-computed to avoid unreliable station-list API.

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

# Ski areas with their nearest SNOTEL station (pre-computed)
# Format: (name, lat, lon, state, elevation_m, snotel_id, snotel_name)
SKI_AREAS = [
    ("Stevens Pass", 47.7448, -121.089, "Washington", 1241, "791:WA:SNTL", "Stevens Pass"),
    ("Crystal Mountain", 46.9282, -121.5045, "Washington", 2134, "679:WA:SNTL", "Morse Lake"),
    ("Mt. Baker", 48.857, -121.6695, "Washington", 1524, "909:WA:SNTL", "Wells Creek"),
    ("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 1067, "778:WA:SNTL", "Snoqualmie Pass"),
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
    ("Big Sky", 45.2618, -111.4018, "Montana", 3403, "561:MT:SNTL", "Lone Mountain"),
    ("Whitefish", 48.4833, -114.355, "Montana", 2133, "656:MT:SNTL", "Noisy Basin"),
    ("Jackson Hole", 43.5875, -110.8281, "Wyoming", 3185, "481:WY:SNTL", "Granite Creek"),
    ("Sun Valley", 43.6804, -114.4075, "Idaho", 2789, "440:ID:SNTL", "Dollarhide Summit"),
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


def fetch_open_meteo(lat: float, lon: float) -> dict | None:
    """Fetch forecast from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m"
        f"&daily=snowfall_sum"
        f"&timezone=auto"
        f"&forecast_days=7"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning(f"Open-Meteo error: {e}")
        return None


def main():
    """Fetch all data and output JSON."""
    logger.info("=" * 50)
    logger.info("FETCH DATA V2 - SNOTEL + Open-Meteo Hybrid")
    logger.info("=" * 50)

    resorts = []
    snotel_success = 0

    for name, lat, lon, state, elev, snotel_id, snotel_name in SKI_AREAS:
        logger.info(f"Processing {name}...")

        # Get base depth from SNOTEL
        base_depth_cm = get_snotel_snow_depth(snotel_id, snotel_name)
        if base_depth_cm:
            logger.info(f"  SNOTEL ({snotel_name}): {base_depth_cm}cm")
            snotel_success += 1
        else:
            logger.warning(f"  SNOTEL ({snotel_name}): No data")

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

        resort = {
            "name": name,
            "lat": lat,
            "lon": lon,
            "state": state,
            "elevation_m": elev,
            "base_depth_cm": base_depth_cm,
            "base_depth_source": snotel_name if base_depth_cm else None,
            "temp_c": round(temp_c, 1) if temp_c is not None else None,
            "forecast": forecast,
            "seven_day_total_cm": round(total_snow, 1),
        }

        resorts.append(resort)
        time.sleep(0.3)  # Be nice to APIs

    # Build output
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "base_depth": "SNOTEL (nrcs.usda.gov)",
            "forecast": "Open-Meteo (open-meteo.com)",
        },
        "resorts": resorts,
    }

    # Write to data directory
    out_path = Path(__file__).parent.parent / "data" / "forecasts_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Output written to: {out_path}")
    logger.info(f"SNOTEL success: {snotel_success}/{len(SKI_AREAS)}")
    logger.info(f"Average 7-day snowfall: {sum(r['seven_day_total_cm'] for r in resorts)/len(resorts):.1f}cm")
    logger.info("Done!")


if __name__ == "__main__":
    main()
