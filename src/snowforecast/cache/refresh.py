"""Background refresh for cache pre-warming.

Refreshes HRRR forecasts and terrain data for all 22 ski areas.
Run hourly via cron to keep cache fresh:

    # Every hour at :50 (before new HRRR run available)
    50 * * * * python -m snowforecast.cache.refresh

Usage:
    python -m snowforecast.cache.refresh          # Refresh all (HRRR + terrain)
    python -m snowforecast.cache.refresh --hrrr   # Refresh HRRR only
    python -m snowforecast.cache.refresh --terrain # Refresh terrain only
    python -m snowforecast.cache.refresh --status  # Show cache status
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from snowforecast.cache.database import CacheDatabase, DEFAULT_DB_PATH
from snowforecast.cache.hrrr import HRRRCache, CACHE_VALIDITY_HOURS
from snowforecast.cache.models import SKI_AREAS_DATA, SkiArea
from snowforecast.cache.terrain import TerrainCache

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    """Result of a refresh operation."""

    total: int
    success: int
    failed: int
    skipped: int
    duration_ms: int

    @property
    def success_rate(self) -> float:
        """Percentage of successful refreshes."""
        if self.total == 0:
            return 0.0
        return (self.success / self.total) * 100

    def __str__(self) -> str:
        return (
            f"Refresh complete: {self.success}/{self.total} successful, "
            f"{self.failed} failed, {self.skipped} skipped "
            f"({self.duration_ms}ms)"
        )


def refresh_hrrr_for_ski_areas(
    db: CacheDatabase,
    ski_areas: Optional[list[SkiArea]] = None,
    force: bool = False,
) -> RefreshResult:
    """Refresh HRRR forecasts for all ski areas.

    Checks which ski areas have stale/missing HRRR data and fetches fresh
    forecasts for those areas.

    Args:
        db: CacheDatabase instance
        ski_areas: List of ski areas to refresh. Defaults to all 22.
        force: Force refresh even if cache is fresh

    Returns:
        RefreshResult with counts of successful/failed refreshes
    """
    if ski_areas is None:
        ski_areas = SKI_AREAS_DATA

    hrrr_cache = HRRRCache(db)
    start_time = time.time()

    total = len(ski_areas)
    success = 0
    failed = 0
    skipped = 0

    logger.info(f"Starting HRRR refresh for {total} ski areas...")

    for i, area in enumerate(ski_areas, 1):
        try:
            # Check if cache is fresh (within validity window)
            valid_time = datetime.utcnow()
            cached = hrrr_cache.get(
                area.lat, area.lon, valid_time, max_age_hours=CACHE_VALIDITY_HOURS
            )

            if cached is not None and not force:
                logger.debug(
                    f"[{i}/{total}] {area.name}: cache fresh "
                    f"(run_time={cached.run_time})"
                )
                skipped += 1
                continue

            # Fetch and cache new data
            logger.info(f"[{i}/{total}] {area.name}: fetching HRRR...")
            result = hrrr_cache.fetch_and_cache(area.lat, area.lon, valid_time)

            if result is not None:
                logger.info(
                    f"[{i}/{total}] {area.name}: cached "
                    f"(snow_depth={result.snow_depth_cm:.1f}cm)"
                )
                success += 1
            else:
                logger.warning(f"[{i}/{total}] {area.name}: fetch returned None")
                failed += 1

        except Exception as e:
            logger.error(f"[{i}/{total}] {area.name}: failed - {e}")
            failed += 1

    duration_ms = int((time.time() - start_time) * 1000)

    result = RefreshResult(
        total=total,
        success=success,
        failed=failed,
        skipped=skipped,
        duration_ms=duration_ms,
    )

    logger.info(str(result))
    return result


def refresh_terrain_for_ski_areas(
    db: CacheDatabase,
    ski_areas: Optional[list[SkiArea]] = None,
) -> RefreshResult:
    """Ensure terrain is cached for all ski areas.

    Terrain is static and never expires, so this only fetches for
    locations not already cached.

    Args:
        db: CacheDatabase instance
        ski_areas: List of ski areas to check. Defaults to all 22.

    Returns:
        RefreshResult with counts
    """
    if ski_areas is None:
        ski_areas = SKI_AREAS_DATA

    terrain_cache = TerrainCache(db)
    start_time = time.time()

    total = len(ski_areas)
    success = 0
    failed = 0
    skipped = 0

    logger.info(f"Starting terrain refresh for {total} ski areas...")

    for i, area in enumerate(ski_areas, 1):
        try:
            # Check if already cached
            cached = terrain_cache.get(area.lat, area.lon)

            if cached is not None:
                logger.debug(
                    f"[{i}/{total}] {area.name}: already cached "
                    f"(elevation={cached.elevation}m)"
                )
                skipped += 1
                continue

            # Fetch and cache
            logger.info(f"[{i}/{total}] {area.name}: fetching terrain...")
            result = terrain_cache.fetch_and_cache(area.lat, area.lon)

            if result is not None:
                logger.info(
                    f"[{i}/{total}] {area.name}: cached "
                    f"(elevation={result.elevation}m)"
                )
                success += 1
            else:
                logger.warning(f"[{i}/{total}] {area.name}: fetch returned None")
                failed += 1

        except Exception as e:
            logger.error(f"[{i}/{total}] {area.name}: failed - {e}")
            failed += 1

    duration_ms = int((time.time() - start_time) * 1000)

    result = RefreshResult(
        total=total,
        success=success,
        failed=failed,
        skipped=skipped,
        duration_ms=duration_ms,
    )

    logger.info(str(result))
    return result


def refresh_all_ski_areas(
    db_path: Optional[Path] = None,
    force_hrrr: bool = False,
) -> tuple[RefreshResult, RefreshResult]:
    """Refresh both HRRR and terrain for all ski areas.

    This is the main entry point for background refresh.

    Args:
        db_path: Path to DuckDB file. Uses default if not specified.
        force_hrrr: Force HRRR refresh even if cache is fresh

    Returns:
        Tuple of (HRRR RefreshResult, Terrain RefreshResult)
    """
    db = CacheDatabase(db_path or DEFAULT_DB_PATH)

    try:
        logger.info("=" * 60)
        logger.info("Starting full cache refresh...")
        logger.info(f"Database: {db.db_path}")
        logger.info(f"Ski areas: {len(SKI_AREAS_DATA)}")
        logger.info("=" * 60)

        # Refresh terrain first (faster, less likely to fail)
        terrain_result = refresh_terrain_for_ski_areas(db)

        # Then refresh HRRR (slower, more likely to have issues)
        hrrr_result = refresh_hrrr_for_ski_areas(db, force=force_hrrr)

        logger.info("=" * 60)
        logger.info("Cache refresh complete:")
        logger.info(f"  Terrain: {terrain_result}")
        logger.info(f"  HRRR:    {hrrr_result}")
        logger.info("=" * 60)

        return hrrr_result, terrain_result

    finally:
        db.close()


def get_cache_status(db_path: Optional[Path] = None) -> dict:
    """Get current cache status.

    Args:
        db_path: Path to DuckDB file. Uses default if not specified.

    Returns:
        Dict with cache statistics and per-ski-area status
    """
    db = CacheDatabase(db_path or DEFAULT_DB_PATH)

    try:
        stats = db.get_stats()
        hrrr_cache = HRRRCache(db)
        terrain_cache = TerrainCache(db)

        # Check each ski area
        ski_area_status = []
        valid_time = datetime.utcnow()

        for area in SKI_AREAS_DATA:
            hrrr_cached = hrrr_cache.get(
                area.lat, area.lon, valid_time, max_age_hours=CACHE_VALIDITY_HOURS
            )
            terrain_cached = terrain_cache.get(area.lat, area.lon)

            ski_area_status.append({
                "name": area.name,
                "state": area.state,
                "hrrr_cached": hrrr_cached is not None,
                "hrrr_run_time": hrrr_cached.run_time if hrrr_cached else None,
                "terrain_cached": terrain_cached is not None,
                "elevation": terrain_cached.elevation if terrain_cached else None,
            })

        hrrr_cached_count = sum(1 for s in ski_area_status if s["hrrr_cached"])
        terrain_cached_count = sum(1 for s in ski_area_status if s["terrain_cached"])

        return {
            "db_path": str(db.db_path),
            "total_ski_areas": len(SKI_AREAS_DATA),
            "hrrr_cached": hrrr_cached_count,
            "terrain_cached": terrain_cached_count,
            "forecast_count": stats["forecast_count"],
            "terrain_count": stats["terrain_count"],
            "latest_run_time": stats["latest_run_time"],
            "ski_areas": ski_area_status,
        }

    finally:
        db.close()


def print_status(status: dict) -> None:
    """Print cache status in human-readable format."""
    print()
    print("=" * 60)
    print("Snow Forecast Cache Status")
    print("=" * 60)
    print(f"Database: {status['db_path']}")
    print(f"Total ski areas: {status['total_ski_areas']}")
    print()
    print(f"HRRR cached (fresh): {status['hrrr_cached']}/{status['total_ski_areas']}")
    print(f"Terrain cached: {status['terrain_cached']}/{status['total_ski_areas']}")
    print(f"Total forecast records: {status['forecast_count']}")
    print(f"Total terrain records: {status['terrain_count']}")

    if status["latest_run_time"]:
        print(f"Latest HRRR run: {status['latest_run_time']}")

    print()
    print("Ski Area Status:")
    print("-" * 60)

    for area in status["ski_areas"]:
        hrrr_status = "OK" if area["hrrr_cached"] else "STALE"
        terrain_status = "OK" if area["terrain_cached"] else "MISSING"
        elev = f"{area['elevation']:.0f}m" if area["elevation"] else "N/A"

        print(f"  {area['name']:<25} HRRR:{hrrr_status:<6} Terrain:{terrain_status} ({elev})")

    print("=" * 60)


def main():
    """CLI entry point for background refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh snow forecast cache for all ski areas",
        epilog="""
Examples:
  python -m snowforecast.cache.refresh          # Refresh all
  python -m snowforecast.cache.refresh --hrrr   # HRRR only
  python -m snowforecast.cache.refresh --status # Show status

Cron setup (refresh at :50 each hour before new HRRR run):
  50 * * * * cd /path/to/snowforecast && python -m snowforecast.cache.refresh >> /var/log/snowforecast-refresh.log 2>&1
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--hrrr",
        action="store_true",
        help="Refresh HRRR forecasts only",
    )
    parser.add_argument(
        "--terrain",
        action="store_true",
        help="Refresh terrain data only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if cache is fresh",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current cache status",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"Database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handle status command
    if args.status:
        status = get_cache_status(args.db)
        print_status(status)
        return 0

    # Determine what to refresh
    db = CacheDatabase(args.db or DEFAULT_DB_PATH)

    try:
        exit_code = 0

        if args.hrrr and not args.terrain:
            # HRRR only
            result = refresh_hrrr_for_ski_areas(db, force=args.force)
            if result.failed > 0:
                exit_code = 1

        elif args.terrain and not args.hrrr:
            # Terrain only
            result = refresh_terrain_for_ski_areas(db)
            if result.failed > 0:
                exit_code = 1

        else:
            # Both (default)
            hrrr_result, terrain_result = refresh_all_ski_areas(
                args.db, force_hrrr=args.force
            )
            if hrrr_result.failed > 0 or terrain_result.failed > 0:
                exit_code = 1

        return exit_code

    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
