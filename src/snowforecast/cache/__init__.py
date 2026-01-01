"""Data caching layer for snowforecast.

Provides persistent caching of HRRR forecasts and terrain data using DuckDB.

Background refresh can be run via:
    python -m snowforecast.cache.refresh

Or scheduled via cron:
    # Every hour at :50 (before new HRRR run available)
    50 * * * * python -m snowforecast.cache.refresh
"""

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.hrrr import HRRRCache
from snowforecast.cache.models import CachedForecast, CachedTerrain, SkiArea
from snowforecast.cache.predictor import CachedPredictor
from snowforecast.cache.refresh import (
    RefreshResult,
    get_cache_status,
    refresh_all_ski_areas,
    refresh_hrrr_for_ski_areas,
    refresh_terrain_for_ski_areas,
)
from snowforecast.cache.terrain import TerrainCache

__all__ = [
    "CacheDatabase",
    "CachedForecast",
    "CachedPredictor",
    "CachedTerrain",
    "HRRRCache",
    "RefreshResult",
    "SkiArea",
    "TerrainCache",
    "get_cache_status",
    "refresh_all_ski_areas",
    "refresh_hrrr_for_ski_areas",
    "refresh_terrain_for_ski_areas",
]
