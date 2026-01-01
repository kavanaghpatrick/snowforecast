"""Data caching layer for snowforecast.

Provides persistent caching of HRRR forecasts and terrain data using DuckDB.
"""

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.models import CachedForecast, CachedTerrain, SkiArea

__all__ = [
    "CacheDatabase",
    "CachedForecast",
    "CachedTerrain",
    "SkiArea",
]
