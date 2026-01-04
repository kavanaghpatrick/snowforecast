#!/usr/bin/env python3
"""Minimal HRRR/NBM forecast fetcher for Dashboard V2.

Fetches snow data for 22 ski areas and outputs JSON.
No database, no cache classes - just data in, JSON out.

Usage:
    python scripts/fetch_data_v2.py
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np

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


def extract_points(ds, points):
    """Extract values at lat/lon points from xarray dataset."""
    results = {}
    data = list(ds.data_vars.values())[0]  # Get first data variable
    lats = ds.latitude.values
    lons = ds.longitude.values

    for name, lat, lon, *_ in points:
        try:
            lon_adj = lon + 360 if lon < 0 else lon
            dist = np.sqrt((lats - lat) ** 2 + (lons - lon_adj) ** 2)
            y, x = np.unravel_index(dist.argmin(), dist.shape)
            results[(lat, lon)] = float(data.values[y, x])
        except Exception as e:
            logger.warning(f"  Failed {name}: {e}")
            results[(lat, lon)] = None
    return results


def fetch_hrrr(day_offset):
    """Fetch HRRR data for a specific day (0=today, 1=tomorrow)."""
    from herbie import Herbie

    today = date.today()
    target_date = today + timedelta(days=day_offset)
    fxx = 12 + (day_offset * 24)  # 12h for today, 36h for tomorrow

    logger.info(f"HRRR: Fetching day {day_offset} ({target_date})")

    for run_offset in [0, 1]:
        run_date = datetime.combine(today - timedelta(days=run_offset), datetime.min.time())
        adj_fxx = fxx + (run_offset * 24)

        if adj_fxx > 48:
            continue

        try:
            H = Herbie(run_date, model="hrrr", product="sfc", fxx=adj_fxx)
            if not H.grib:
                continue

            logger.info(f"  Using run {run_date.date()} fxx={adj_fxx}")

            # Get snow depth (base depth) and temperature
            snow_depth = {}
            temp = {}

            try:
                ds = H.xarray(":SNOD:")
                snow_depth = extract_points(ds, SKI_AREAS)
            except Exception as e:
                logger.warning(f"  SNOD failed: {e}")

            try:
                ds = H.xarray(":TMP:2 m")
                temp = extract_points(ds, SKI_AREAS)
            except Exception as e:
                logger.warning(f"  TMP failed: {e}")

            return {
                "date": str(target_date),
                "snow_depth": snow_depth,
                "temp": temp,
                "source": "hrrr",
            }

        except Exception as e:
            logger.warning(f"  Run failed: {e}")

    return None


def fetch_nbm(day_offset):
    """Fetch NBM data for extended forecast (days 2-6)."""
    from herbie import Herbie

    today = date.today()
    target_date = today + timedelta(days=day_offset)
    fxx = (day_offset * 24) + 12  # Noon each day

    logger.info(f"NBM: Fetching day {day_offset} ({target_date})")

    for run_offset in [0, 1, 2]:
        run_date = datetime.combine(today - timedelta(days=run_offset), datetime.min.time())
        adj_fxx = fxx + (run_offset * 24)

        if adj_fxx > 264:
            continue

        try:
            H = Herbie(run_date, model="nbm", product="co", fxx=adj_fxx)
            if not H.grib:
                continue

            logger.info(f"  Using run {run_date.date()} fxx={adj_fxx}")

            # Get accumulated snow (new snow)
            new_snow = {}
            try:
                ds = H.xarray(":ASNOW:")
                new_snow = extract_points(ds, SKI_AREAS)
            except Exception as e:
                logger.warning(f"  ASNOW failed: {e}")

            return {
                "date": str(target_date),
                "new_snow": new_snow,
                "source": "nbm",
            }

        except Exception as e:
            logger.warning(f"  Run failed: {e}")

    return None


def main():
    """Fetch all data and output JSON."""
    logger.info("=" * 50)
    logger.info("FETCH DATA V2 - Minimal HRRR/NBM Fetcher")
    logger.info("=" * 50)

    today = date.today()
    resorts = []

    # Build resort data structure
    for name, lat, lon, state, elev in SKI_AREAS:
        resorts.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "state": state,
            "elevation_m": elev,
            "base_depth_cm": None,
            "temp_c": None,
            "forecast": [],
            "seven_day_total_cm": 0.0,
        })

    # Fetch HRRR for days 0-1
    hrrr_data = {}
    for day in range(2):
        result = fetch_hrrr(day)
        if result:
            hrrr_data[day] = result

    # Fetch NBM for days 2-6
    nbm_data = {}
    for day in range(2, 7):
        result = fetch_nbm(day)
        if result:
            nbm_data[day] = result

    # Populate resort data
    for resort in resorts:
        lat, lon = resort["lat"], resort["lon"]
        total_snow = 0.0

        # Day 0: Get base depth and temp from HRRR
        if 0 in hrrr_data:
            snow_depth = hrrr_data[0]["snow_depth"].get((lat, lon))
            temp_k = hrrr_data[0]["temp"].get((lat, lon))
            if snow_depth is not None:
                resort["base_depth_cm"] = round(snow_depth * 100, 1)
            if temp_k is not None:
                resort["temp_c"] = round(temp_k - 273.15, 1)

        # Days 0-1: HRRR new snow (estimated from depth change)
        for day in range(2):
            target_date = today + timedelta(days=day)
            new_snow_cm = 0.0

            if day in hrrr_data:
                # For HRRR, we don't have direct new snow - estimate small amounts
                # This is a simplification; HRRR SNOD is cumulative base depth
                new_snow_cm = 0.0  # Will be updated with diff if we have consecutive days

            resort["forecast"].append({
                "day": day,
                "date": str(target_date),
                "new_snow_cm": new_snow_cm,
                "source": "hrrr",
            })

        # Days 2-6: NBM new snow
        for day in range(2, 7):
            target_date = today + timedelta(days=day)
            new_snow_cm = 0.0

            if day in nbm_data:
                new_snow = nbm_data[day]["new_snow"].get((lat, lon))
                if new_snow is not None:
                    new_snow_cm = round(new_snow * 100, 1)  # m to cm
                    total_snow += new_snow_cm

            resort["forecast"].append({
                "day": day,
                "date": str(target_date),
                "new_snow_cm": new_snow_cm,
                "source": "nbm",
            })

        resort["seven_day_total_cm"] = round(total_snow, 1)

    # Build output
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resorts": resorts,
    }

    # Write to data directory
    out_path = Path(__file__).parent.parent / "data" / "forecasts_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Output written to: {out_path}")
    logger.info(f"Resorts: {len(resorts)}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
