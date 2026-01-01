"""Terrain caching layer for snowforecast.

Wraps DEM terrain fetching with permanent caching.
Terrain is static - never expires.
"""

import logging
import time
from typing import Optional

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.models import CachedTerrain, SKI_AREAS_DATA

logger = logging.getLogger(__name__)


class TerrainCache:
    """Cache layer for terrain data from Copernicus DEM.

    Terrain is static and never expires - one-time fetch per location.
    Cache lookup is ~1ms vs ~2s for DEM fetch.

    Example:
        >>> from snowforecast.cache.database import CacheDatabase
        >>> db = CacheDatabase()
        >>> cache = TerrainCache(db)
        >>> terrain = cache.fetch_and_cache(47.74, -121.09)
        >>> terrain.elevation
        1257.0
    """

    def __init__(self, db: CacheDatabase):
        """Initialize terrain cache.

        Args:
            db: CacheDatabase instance for storage
        """
        self.db = db
        self._predictor = None

    def _get_predictor(self):
        """Lazy load RealPredictor for DEM fetching."""
        if self._predictor is None:
            from snowforecast.api.predictor import RealPredictor
            self._predictor = RealPredictor()
            logger.info("RealPredictor initialized for terrain fetching")
        return self._predictor

    def get(self, lat: float, lon: float) -> Optional[CachedTerrain]:
        """Get cached terrain data for location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            CachedTerrain if found in cache, None otherwise
        """
        return self.db.get_terrain(lat, lon)

    def store(
        self,
        lat: float,
        lon: float,
        elevation: float,
        slope: float,
        aspect: float,
        roughness: float,
        tpi: float,
    ) -> None:
        """Store terrain data in cache (permanent).

        Args:
            lat: Latitude
            lon: Longitude
            elevation: Elevation in meters
            slope: Slope in degrees
            aspect: Aspect in degrees (0-360)
            roughness: Terrain roughness
            tpi: Topographic Position Index
        """
        self.db.store_terrain(
            lat=lat,
            lon=lon,
            elevation=elevation,
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tpi=tpi,
        )
        logger.debug(f"Stored terrain for ({lat}, {lon}): elev={elevation}m")

    def fetch_and_cache(self, lat: float, lon: float) -> CachedTerrain:
        """Fetch terrain from DEM if not cached, then store permanently.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            CachedTerrain with terrain features
        """
        # Check cache first
        cached = self.get(lat, lon)
        if cached is not None:
            logger.debug(f"Cache hit for terrain at ({lat}, {lon})")
            return cached

        # Cache miss - fetch from DEM
        logger.info(f"Cache miss for terrain at ({lat}, {lon}), fetching from DEM...")
        start_time = time.time()

        predictor = self._get_predictor()
        features = predictor.get_terrain_features(lat, lon)

        duration_ms = int((time.time() - start_time) * 1000)

        # Store in cache
        self.store(
            lat=lat,
            lon=lon,
            elevation=features.get("elevation", 0.0),
            slope=features.get("slope", 0.0),
            aspect=features.get("aspect", 0.0),
            roughness=features.get("roughness", 0.0),
            tpi=features.get("tpi", 0.0),
        )

        # Log the fetch
        self.db.log_fetch(
            source="dem",
            status="success",
            records_added=1,
            duration_ms=duration_ms,
        )

        logger.info(f"Fetched terrain for ({lat}, {lon}) in {duration_ms}ms")

        # Return the cached version
        return self.get(lat, lon)

    def prefetch_all_ski_areas(self) -> int:
        """Cache terrain for all 22 ski areas.

        Returns:
            Number of ski areas cached (should be 22 if all successful)
        """
        cached_count = 0
        total = len(SKI_AREAS_DATA)

        logger.info(f"Prefetching terrain for {total} ski areas...")

        for i, area in enumerate(SKI_AREAS_DATA, 1):
            try:
                # Check if already cached
                existing = self.get(area.lat, area.lon)
                if existing is not None:
                    logger.debug(f"[{i}/{total}] {area.name}: already cached")
                    cached_count += 1
                    continue

                # Fetch and cache
                self.fetch_and_cache(area.lat, area.lon)
                cached_count += 1
                logger.info(f"[{i}/{total}] {area.name}: cached terrain")

            except Exception as e:
                logger.error(f"[{i}/{total}] {area.name}: failed - {e}")
                # Log the error
                self.db.log_fetch(
                    source="dem",
                    status="error",
                    records_added=0,
                    duration_ms=0,
                    error_message=str(e),
                )

        logger.info(f"Prefetched terrain for {cached_count}/{total} ski areas")
        return cached_count

    def get_cached_count(self) -> int:
        """Get number of cached terrain entries.

        Returns:
            Count of cached terrain locations
        """
        return self.db.get_stats()["terrain_count"]

    def is_ski_area_cached(self, name: str) -> bool:
        """Check if a ski area's terrain is cached.

        Args:
            name: Ski area name

        Returns:
            True if terrain is cached, False otherwise
        """
        area = self.db.get_ski_area(name)
        if area is None:
            return False
        return self.get(area.lat, area.lon) is not None
