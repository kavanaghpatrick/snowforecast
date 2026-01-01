"""Data caching layer for snowforecast.

Provides persistent caching of HRRR forecasts and terrain data using DuckDB.
"""

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.hrrr import HRRRCache
from snowforecast.cache.models import CachedForecast, CachedTerrain, SkiArea
from snowforecast.cache.predictor import CachedPredictor
from snowforecast.cache.terrain import TerrainCache

__all__ = [
    "CacheDatabase",
    "CachedForecast",
    "CachedPredictor",
    "CachedTerrain",
    "HRRRCache",
    "SkiArea",
    "TerrainCache",
]
