"""Data caching layer for snowforecast.

Provides persistent caching of HRRR forecasts and terrain data using DuckDB.

Background refresh can be run via:
    python -m snowforecast.cache.refresh

Or scheduled via cron:
    # Every hour at :50 (before new HRRR run available)
    50 * * * * python -m snowforecast.cache.refresh
"""

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.elevation_bands import (
    ElevationBandForecast,
    ElevationBandResult,
    PrecipType,
    compute_elevation_bands,
    get_snow_line,
    get_summit_elevation,
)
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
    "ElevationBandForecast",
    "ElevationBandResult",
    "HRRRCache",
    "PrecipType",
    "RefreshResult",
    "SkiArea",
    "TerrainCache",
    "compute_elevation_bands",
    "get_cache_status",
    "get_snow_line",
    "get_summit_elevation",
    "refresh_all_ski_areas",
    "refresh_hrrr_for_ski_areas",
    "refresh_terrain_for_ski_areas",
]
