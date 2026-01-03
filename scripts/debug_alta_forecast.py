#!/usr/bin/env python3
"""Debug script to trace the data flow for Alta forecast.

This script demonstrates the root cause of zeros in the UI:
The valid_time window in database.get_forecast() is too narrow (+/- 1 hour).

When the current time falls between stored valid_times, the query returns None,
causing the predictor to return zeros.

Usage:
    python3 scripts/debug_alta_forecast.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta

import duckdb

# Alta coordinates
ALTA_LAT = 40.5884
ALTA_LON = -111.6386


def main():
    print("=" * 70)
    print("ALTA FORECAST DEBUG SCRIPT")
    print("=" * 70)

    # Connect to database
    db_path = Path(__file__).parent.parent / "data/cache/snowforecast.duckdb"
    print(f"\nDatabase: {db_path}")
    print(f"Database exists: {db_path.exists()}")

    if not db_path.exists():
        print("ERROR: Database not found!")
        return

    conn = duckdb.connect(str(db_path), read_only=True)

    # Check what's in the database for Alta
    print("\n" + "=" * 70)
    print("STEP 1: What's in the database for Alta?")
    print("=" * 70)

    result = conn.execute("""
        SELECT run_time, forecast_hour, valid_time, snow_depth_m, temp_k, precip_mm
        FROM hrrr_forecasts
        WHERE lat = ? AND lon = ?
        ORDER BY valid_time DESC
    """, [ALTA_LAT, ALTA_LON]).fetchall()

    print(f"\nFound {len(result)} forecasts for Alta ({ALTA_LAT}, {ALTA_LON}):")
    for row in result[:10]:
        snow_cm = row[3] * 100 if row[3] else 0
        print(f"  valid={row[2]}, run={row[0]}, fxx={row[1]}, snow={snow_cm:.1f}cm")

    # Current time
    now = datetime.utcnow()
    print(f"\n" + "=" * 70)
    print("STEP 2: What time is it now?")
    print("=" * 70)
    print(f"\nCurrent UTC time: {now}")

    # The query from database.get_forecast()
    print("\n" + "=" * 70)
    print("STEP 3: Simulate database.get_forecast() query")
    print("=" * 70)

    valid_time = now
    max_age_hours = 2
    min_run_time = now - timedelta(hours=max_age_hours)
    window_start = valid_time - timedelta(hours=1)
    window_end = valid_time + timedelta(hours=1)

    print(f"\nQuery parameters:")
    print(f"  lat: {ALTA_LAT}")
    print(f"  lon: {ALTA_LON}")
    print(f"  valid_time: {valid_time}")
    print(f"  window: {window_start} to {window_end}")
    print(f"  min_run_time (max_age={max_age_hours}h): {min_run_time}")

    # Run the exact query from database.py
    result = conn.execute("""
        SELECT
            lat, lon, run_time, forecast_hour, valid_time,
            snow_depth_m, temp_k, precip_mm, categorical_snow, fetch_time
        FROM hrrr_forecasts
        WHERE lat = ? AND lon = ?
          AND valid_time BETWEEN ? AND ?
          AND run_time >= ?
        ORDER BY run_time DESC, ABS(EXTRACT(EPOCH FROM (valid_time - ?)))
        LIMIT 1
    """, [ALTA_LAT, ALTA_LON, window_start, window_end, min_run_time, valid_time]).fetchone()

    print(f"\nQuery result: {result}")

    if result is None:
        print("\n>>> CACHE MISS - This is why we get zeros! <<<")

        # Find nearest valid_time
        nearest = conn.execute("""
            SELECT valid_time, snow_depth_m * 100 as snow_cm
            FROM hrrr_forecasts
            WHERE lat = ? AND lon = ?
            ORDER BY ABS(EXTRACT(EPOCH FROM (valid_time - ?)))
            LIMIT 1
        """, [ALTA_LAT, ALTA_LON, valid_time]).fetchone()

        if nearest:
            time_diff = (nearest[0] - valid_time).total_seconds() / 60
            print(f"\nNearest stored forecast: valid_time={nearest[0]} ({time_diff:.0f} min away)")
            print(f"Snow depth: {nearest[1]:.1f} cm")
    else:
        print(f"\nCache HIT - snow_depth={result[5]*100:.1f}cm")

    # Root cause analysis
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print("""
The problem is in database.get_forecast() (database.py lines 280-300):

1. The query filters valid_time with a +/- 1 HOUR window
2. HRRR forecasts are stored for specific hours (00:00, 06:00, 12:00, etc.)
3. When current time is between stored hours, the window misses ALL stored data

Example:
  - Current time: 22:26 UTC
  - Query window: 21:26 to 23:26
  - Stored valid_times: 21:00, 00:00 (next day)
  - 21:00 < 21:26 (start) --> MISS!
  - 00:00 > 23:26 (end)   --> MISS!

SOLUTION OPTIONS:

1. Widen the valid_time window (e.g., +/- 3 hours or +/- 6 hours)

2. Use NEAREST valid_time instead of range filter:
   ORDER BY ABS(valid_time - target) LIMIT 1

3. Store more forecast hours in cache (every hour instead of every 6 hours)

The simplest fix is option 2: find the nearest valid_time regardless of range.
""")

    # Test with wider window
    print("\n" + "=" * 70)
    print("TEST: Query with 6-hour window")
    print("=" * 70)

    window_start_wide = valid_time - timedelta(hours=6)
    window_end_wide = valid_time + timedelta(hours=6)

    result_wide = conn.execute("""
        SELECT
            lat, lon, run_time, forecast_hour, valid_time,
            snow_depth_m, temp_k, precip_mm
        FROM hrrr_forecasts
        WHERE lat = ? AND lon = ?
          AND valid_time BETWEEN ? AND ?
        ORDER BY ABS(EXTRACT(EPOCH FROM (valid_time - ?)))
        LIMIT 1
    """, [ALTA_LAT, ALTA_LON, window_start_wide, window_end_wide, valid_time]).fetchone()

    if result_wide:
        print(f"Result with 6-hour window: snow_depth={result_wide[5]*100:.1f}cm")
        print(f"  valid_time: {result_wide[4]}")
        print(f"  run_time: {result_wide[2]}")
    else:
        print("Still no result with 6-hour window")

    print("\n" + "=" * 70)
    print("TEST: Query for NEAREST valid_time (no range filter)")
    print("=" * 70)

    result_nearest = conn.execute("""
        SELECT
            lat, lon, run_time, forecast_hour, valid_time,
            snow_depth_m, temp_k, precip_mm
        FROM hrrr_forecasts
        WHERE lat = ? AND lon = ?
        ORDER BY ABS(EXTRACT(EPOCH FROM (valid_time - ?)))
        LIMIT 1
    """, [ALTA_LAT, ALTA_LON, valid_time]).fetchone()

    if result_nearest:
        print(f"Nearest result: snow_depth={result_nearest[5]*100:.1f}cm")
        print(f"  valid_time: {result_nearest[4]}")
        diff_mins = abs((result_nearest[4] - valid_time).total_seconds()) / 60
        print(f"  time difference: {diff_mins:.0f} minutes")

    conn.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
