#!/usr/bin/env python3
"""Minimal Open-Meteo forecast fetcher for Dashboard V2.

Fetches snow data for 22 ski areas and outputs JSON.
No database, no GRIB files - just REST API calls.

Usage:
    python scripts/fetch_data_v2.py
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All 22 ski areas: (name, lat, lon, state, elevation_m)
SKI_AREAS = [
    ("Stevens Pass", 47.7448, -121.089, "Washington", 1241),
    ("Crystal Mountain", 46.9282, -121.5045, "Washington", 2134),
    ("Mt. Baker", 48.857, -121.6695, "Washington", 1524),
    ("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 1067),
    ("Mt. Hood Meadows", 45.3311, -121.6647, "Oregon", 1829),
    ("Mt. Bachelor", 43.9792, -121.6886, "Oregon", 2743),
    ("Timberline", 45.3309, -121.7109, "Oregon", 1829),
    ("Mammoth Mountain", 37.6308, -119.0326, "California", 3369),
    ("Squaw Valley", 39.1969, -120.2358, "California", 2500),
    ("Heavenly", 38.9353, -119.9396, "California", 3060),
    ("Kirkwood", 38.6848, -120.0655, "California", 2377),
    ("Vail", 39.6403, -106.3742, "Colorado", 3527),
    ("Breckenridge", 39.4817, -106.0384, "Colorado", 3914),
    ("Aspen Snowmass", 39.2084, -106.949, "Colorado", 3813),
    ("Telluride", 37.9375, -107.8123, "Colorado", 3831),
    ("Park City", 40.6514, -111.508, "Utah", 3049),
    ("Snowbird", 40.583, -111.6508, "Utah", 3353),
    ("Alta", 40.5884, -111.6386, "Utah", 3215),
    ("Big Sky", 45.2618, -111.4018, "Montana", 3403),
    ("Whitefish", 48.4833, -114.355, "Montana", 2133),
    ("Jackson Hole", 43.5875, -110.8281, "Wyoming", 3185),
    ("Sun Valley", 43.6804, -114.4075, "Idaho", 2789),
]


def fetch_open_meteo(lat: float, lon: float) -> dict | None:
    """Fetch forecast from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,snow_depth"
        f"&daily=snowfall_sum"
        f"&timezone=auto"
        f"&forecast_days=7"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning(f"  API error: {e}")
        return None


def main():
    """Fetch all data and output JSON."""
    logger.info("=" * 50)
    logger.info("FETCH DATA V2 - Open-Meteo Fetcher")
    logger.info("=" * 50)

    resorts = []

    for name, lat, lon, state, elev in SKI_AREAS:
        logger.info(f"Fetching {name}...")

        data = fetch_open_meteo(lat, lon)

        if data is None:
            logger.warning(f"  Failed to fetch {name}, using defaults")
            resort = {
                "name": name,
                "lat": lat,
                "lon": lon,
                "state": state,
                "elevation_m": elev,
                "base_depth_cm": None,
                "temp_c": None,
                "forecast": [],
                "seven_day_total_cm": 0.0,
            }
        else:
            # Extract current conditions
            current = data.get("current", {})
            snow_depth_cm = current.get("snow_depth", 0) or 0
            temp_c = current.get("temperature_2m")

            # Extract daily forecasts
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            snowfall = daily.get("snowfall_sum", [])

            # Build forecast list
            forecast = []
            total_snow = 0.0
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
                "base_depth_cm": round(snow_depth_cm, 1) if snow_depth_cm else None,
                "temp_c": round(temp_c, 1) if temp_c is not None else None,
                "forecast": forecast,
                "seven_day_total_cm": round(total_snow, 1),
            }

            logger.info(f"  7-day total: {total_snow:.1f}cm")

        resorts.append(resort)
        time.sleep(0.1)  # Be nice to the API

    # Build output
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Open-Meteo (open-meteo.com)",
        "resorts": resorts,
    }

    # Write to data directory
    out_path = Path(__file__).parent.parent / "data" / "forecasts_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Output written to: {out_path}")
    logger.info(f"Resorts: {len(resorts)}")

    # Summary
    total_avg = sum(r["seven_day_total_cm"] for r in resorts) / len(resorts)
    logger.info(f"Average 7-day snowfall: {total_avg:.1f}cm")
    logger.info("Done!")


if __name__ == "__main__":
    main()
