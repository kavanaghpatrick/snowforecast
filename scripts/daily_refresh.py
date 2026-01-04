#!/usr/bin/env python3
"""Daily cache refresh script for GitHub Actions.

Downloads HRRR/NBM forecast data and extracts values for all 22 ski areas.
Optimized to download each GRIB file once and extract multiple points.

Usage:
    python scripts/daily_refresh.py
"""

import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snowforecast.cache.database import CacheDatabase, DEFAULT_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# All 22 ski areas with coordinates
SKI_AREAS = [
    ("Stevens Pass", 47.7448, -121.089),
    ("Crystal Mountain", 46.9282, -121.5045),
    ("Mt. Baker", 48.857, -121.6695),
    ("Snoqualmie Pass", 47.4204, -121.4138),
    ("Mt. Hood Meadows", 45.3311, -121.6647),
    ("Mt. Bachelor", 43.9792, -121.6886),
    ("Timberline", 45.3309, -121.7109),
    ("Mammoth Mountain", 37.6308, -119.0326),
    ("Squaw Valley", 39.1969, -120.2358),
    ("Heavenly", 38.9353, -119.9396),
    ("Kirkwood", 38.6848, -120.0655),
    ("Vail", 39.6403, -106.3742),
    ("Breckenridge", 39.4817, -106.0384),
    ("Aspen Snowmass", 39.2084, -106.949),
    ("Telluride", 37.9375, -107.8123),
    ("Park City", 40.6514, -111.508),
    ("Snowbird", 40.583, -111.6508),
    ("Alta", 40.5884, -111.6386),
    ("Big Sky", 45.2618, -111.4018),
    ("Whitefish", 48.4833, -114.355),
    ("Jackson Hole", 43.5875, -110.8281),
    ("Sun Valley", 43.6804, -114.4075),
]


def extract_points_from_xarray(ds, variable_name, points):
    """Extract values at multiple lat/lon points from an xarray dataset.

    HRRR uses Lambert Conformal projection with 2D lat/lon arrays.
    We find the nearest grid cell by computing distances.

    Args:
        ds: xarray Dataset from herbie
        variable_name: Name of variable in dataset
        points: List of (name, lat, lon) tuples

    Returns:
        Dict mapping (lat, lon) -> value
    """
    import numpy as np

    results = {}

    # Get the data variable (herbie returns with specific names)
    if variable_name in ds:
        data = ds[variable_name]
    else:
        # Try to find the variable
        for var in ds.data_vars:
            data = ds[var]
            break

    # Get 2D lat/lon arrays
    lats = ds.latitude.values  # Shape: (y, x)
    lons = ds.longitude.values  # Shape: (y, x)

    for name, lat, lon in points:
        try:
            # Handle longitude convention (HRRR uses 0-360, we use -180 to 180)
            lon_adjusted = lon + 360 if lon < 0 else lon

            # Find nearest grid cell by distance
            dist = np.sqrt((lats - lat) ** 2 + (lons - lon_adjusted) ** 2)
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

            # Extract value at that grid cell
            value = float(data.values[y_idx, x_idx])
            results[(lat, lon)] = value
        except Exception as e:
            logger.warning(f"  Failed to extract {name} ({lat}, {lon}): {e}")
            results[(lat, lon)] = None

    return results


def refresh_hrrr_batch(db, target_date, fxx):
    """Refresh HRRR data for all ski areas using batch extraction.

    Downloads ONE GRIB file and extracts ALL 22 points from it.
    """
    from herbie import Herbie

    logger.info(f"HRRR batch refresh for {target_date} (fxx={fxx}h)")

    # Try today's run, then yesterday's
    for run_offset in [0, 1]:
        run_date = datetime.combine(target_date, datetime.min.time()) - timedelta(days=run_offset)
        adjusted_fxx = fxx + (run_offset * 24)

        if adjusted_fxx > 48:
            continue  # HRRR only goes to 48h

        try:
            H = Herbie(
                run_date,
                model="hrrr",
                product="sfc",
                fxx=adjusted_fxx,
            )

            if not H.grib:
                logger.warning(f"  No GRIB found for {run_date.date()} fxx={adjusted_fxx}")
                continue

            logger.info(f"  Using run {run_date.date()} fxx={adjusted_fxx}")

            # Download and extract each variable for ALL points at once
            # Snow depth
            try:
                ds_snow = H.xarray(":SNOD:")
                snow_values = extract_points_from_xarray(ds_snow, "sd", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Snow depth extraction failed: {e}")
                snow_values = {}

            # Temperature at 2m
            try:
                ds_temp = H.xarray(":TMP:2 m")
                temp_values = extract_points_from_xarray(ds_temp, "t2m", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Temperature extraction failed: {e}")
                temp_values = {}

            # Precipitation
            try:
                ds_precip = H.xarray(":APCP:")
                precip_values = extract_points_from_xarray(ds_precip, "tp", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Precipitation extraction failed: {e}")
                precip_values = {}

            # Categorical snow
            try:
                ds_csnow = H.xarray(":CSNOW:")
                csnow_values = extract_points_from_xarray(ds_csnow, "csnow", SKI_AREAS)
            except Exception as e:
                csnow_values = {}

            # Store all results
            valid_time = run_date + timedelta(hours=adjusted_fxx)
            stored = 0

            for name, lat, lon in SKI_AREAS:
                snow = snow_values.get((lat, lon))
                temp = temp_values.get((lat, lon))
                precip = precip_values.get((lat, lon))
                csnow = csnow_values.get((lat, lon), 0)

                if snow is not None:
                    db.store_forecast(
                        lat=lat,
                        lon=lon,
                        run_time=run_date,
                        forecast_hour=adjusted_fxx,
                        snow_depth_m=snow,
                        temp_k=temp or 273.0,
                        precip_mm=precip or 0.0,
                        categorical_snow=csnow or 0.0,
                    )
                    stored += 1
                    logger.info(f"    {name}: {snow*100:.1f}cm")

            logger.info(f"  Stored {stored}/{len(SKI_AREAS)} forecasts")
            return True

        except Exception as e:
            logger.warning(f"  Run {run_date.date()} failed: {e}")
            continue

    logger.error(f"  No HRRR data available for {target_date}")
    return False


def refresh_nbm_batch(db, target_date, fxx):
    """Refresh NBM data for all ski areas using batch extraction.

    NBM (National Blend of Models) provides extended forecasts up to 264h (11 days).
    """
    from herbie import Herbie

    logger.info(f"NBM batch refresh for {target_date} (fxx={fxx}h)")

    # Try recent model runs
    for run_offset in [0, 1, 2]:
        run_date = datetime.combine(date.today(), datetime.min.time()) - timedelta(days=run_offset)
        adjusted_fxx = fxx + (run_offset * 24)

        if adjusted_fxx > 264:
            continue  # NBM goes to 264h

        try:
            H = Herbie(
                run_date,
                model="nbm",
                product="co",
                fxx=adjusted_fxx,
            )

            if not H.grib:
                logger.warning(f"  No GRIB found for NBM {run_date.date()} fxx={adjusted_fxx}")
                continue

            logger.info(f"  Using NBM run {run_date.date()} fxx={adjusted_fxx}")

            # NBM variables are slightly different
            # Accumulated snow
            try:
                ds_snow = H.xarray(":ASNOW:")
                snow_values = extract_points_from_xarray(ds_snow, "asnow", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Snow extraction failed: {e}")
                snow_values = {}

            # Temperature
            try:
                ds_temp = H.xarray(":TMP:2 m")
                temp_values = extract_points_from_xarray(ds_temp, "t2m", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Temperature extraction failed: {e}")
                temp_values = {}

            # Precipitation
            try:
                ds_precip = H.xarray(":APCP:")
                precip_values = extract_points_from_xarray(ds_precip, "tp", SKI_AREAS)
            except Exception as e:
                logger.warning(f"  Precipitation extraction failed: {e}")
                precip_values = {}

            # Store all results
            valid_time = run_date + timedelta(hours=adjusted_fxx)
            stored = 0

            for name, lat, lon in SKI_AREAS:
                snow = snow_values.get((lat, lon))
                temp = temp_values.get((lat, lon))
                precip = precip_values.get((lat, lon))

                if snow is not None or temp is not None:
                    db.store_forecast(
                        lat=lat,
                        lon=lon,
                        run_time=run_date,
                        forecast_hour=adjusted_fxx,
                        snow_depth_m=snow or 0.0,
                        temp_k=temp or 273.0,
                        precip_mm=precip or 0.0,
                        categorical_snow=1.0 if (snow or 0) > 0.01 else 0.0,
                    )
                    stored += 1
                    logger.info(f"    {name}: {(snow or 0)*100:.1f}cm snow")

            logger.info(f"  Stored {stored}/{len(SKI_AREAS)} forecasts")
            return True

        except Exception as e:
            logger.warning(f"  NBM run {run_date.date()} failed: {e}")
            continue

    logger.error(f"  No NBM data available for {target_date}")
    return False


def main():
    """Main refresh routine - populates 7-day forecasts for all ski areas."""
    logger.info("=" * 60)
    logger.info("DAILY CACHE REFRESH")
    logger.info("=" * 60)

    db = CacheDatabase(DEFAULT_DB_PATH)
    today = date.today()

    logger.info(f"Date: {today}")
    logger.info(f"Ski areas: {len(SKI_AREAS)}")
    logger.info(f"Database: {DEFAULT_DB_PATH}")
    logger.info("")

    success_count = 0

    # Days 0-1: Use HRRR (48h range, 3km resolution)
    logger.info("--- HRRR Forecasts (Days 0-1) ---")
    for day_offset in range(2):
        target = today + timedelta(days=day_offset)
        # Forecast hour: 12h for today, 36h for tomorrow
        fxx = 12 + (day_offset * 24)

        if refresh_hrrr_batch(db, target, fxx):
            success_count += 1

    logger.info("")

    # Days 2-6: Use NBM (264h range, ~2.5km resolution)
    logger.info("--- NBM Forecasts (Days 2-6) ---")
    for day_offset in range(2, 7):
        target = today + timedelta(days=day_offset)
        # Forecast hour from today: 48h, 72h, 96h, 120h, 144h
        fxx = day_offset * 24

        if refresh_nbm_batch(db, target, fxx):
            success_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info("=" * 60)

    stats = db.get_stats()
    logger.info(f"Total forecasts in cache: {stats['forecast_count']}")
    logger.info(f"Terrain entries: {stats['terrain_count']}")
    logger.info(f"Latest run time: {stats['latest_run_time']}")
    logger.info(f"Days refreshed: {success_count}/7")

    db.close()

    # Exit with error if less than half succeeded
    if success_count < 4:
        logger.error("Too many days failed to refresh")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
