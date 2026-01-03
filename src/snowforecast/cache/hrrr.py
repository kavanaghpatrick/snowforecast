"""HRRR forecast caching layer.

Wraps HRRR fetching with DuckDB caching to reduce API calls.
Cache validity is 2 hours (HRRR runs hourly).
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.models import CachedForecast

logger = logging.getLogger(__name__)

# Cache validity in hours (HRRR runs hourly, 2-hour window is reasonable)
CACHE_VALIDITY_HOURS = 2


class HRRRCache:
    """Cache layer for HRRR forecast data.

    Wraps the RealPredictor's HRRR fetching with DuckDB caching.
    Cache is valid if run_time is within the last 2 hours.
    All fetched data is stored for 7 days for historical analysis.

    Example:
        >>> db = CacheDatabase()
        >>> cache = HRRRCache(db)
        >>> forecast = cache.get(47.74, -121.09, datetime.now())
        >>> if forecast is None:
        ...     forecast = cache.fetch_and_cache(47.74, -121.09, datetime.now())
    """

    def __init__(self, db: CacheDatabase):
        """Initialize HRRR cache.

        Args:
            db: CacheDatabase instance for persistence
        """
        self.db = db
        self._predictor = None

    def _get_predictor(self):
        """Lazy load RealPredictor to avoid import issues."""
        if self._predictor is None:
            from snowforecast.api.predictor import RealPredictor
            self._predictor = RealPredictor()
        return self._predictor

    def get(
        self,
        lat: float,
        lon: float,
        valid_time: datetime,
        max_age_hours: int = CACHE_VALIDITY_HOURS,
    ) -> Optional[CachedForecast]:
        """Get cached forecast if fresh.

        Args:
            lat: Latitude
            lon: Longitude
            valid_time: Target forecast valid time
            max_age_hours: Maximum age of cached data in hours (default 2)

        Returns:
            CachedForecast if found and fresh, None otherwise
        """
        forecast = self.db.get_forecast(lat, lon, valid_time, max_age_hours)
        if forecast is not None:
            logger.debug(
                f"Cache HIT for ({lat}, {lon}) at {valid_time} "
                f"(run_time={forecast.run_time})"
            )
        else:
            logger.debug(f"Cache MISS for ({lat}, {lon}) at {valid_time}")
        return forecast

    def store(
        self,
        lat: float,
        lon: float,
        run_time: datetime,
        forecast_hour: int,
        snow_depth_m: float,
        temp_k: float,
        precip_mm: float,
        categorical_snow: float,
    ) -> CachedForecast:
        """Store forecast data in cache.

        Args:
            lat: Latitude
            lon: Longitude
            run_time: HRRR model run time
            forecast_hour: Forecast hour offset (0-48)
            snow_depth_m: Snow depth in meters
            temp_k: Temperature in Kelvin
            precip_mm: Precipitation in mm
            categorical_snow: Categorical snow flag (0 or 1)

        Returns:
            CachedForecast object representing stored data
        """
        self.db.store_forecast(
            lat=lat,
            lon=lon,
            run_time=run_time,
            forecast_hour=forecast_hour,
            snow_depth_m=snow_depth_m,
            temp_k=temp_k,
            precip_mm=precip_mm,
            categorical_snow=categorical_snow,
        )

        valid_time = run_time + timedelta(hours=forecast_hour)
        logger.info(
            f"Cached HRRR forecast for ({lat}, {lon}) "
            f"run_time={run_time}, fxx={forecast_hour}"
        )

        return CachedForecast(
            lat=lat,
            lon=lon,
            run_time=run_time,
            forecast_hour=forecast_hour,
            valid_time=valid_time,
            snow_depth_m=snow_depth_m,
            temp_k=temp_k,
            precip_mm=precip_mm,
            categorical_snow=categorical_snow,
            fetch_time=datetime.utcnow(),
        )

    def fetch_and_cache(
        self,
        lat: float,
        lon: float,
        valid_time: datetime,
        forecast_hours: int = 24,
    ) -> Optional[CachedForecast]:
        """Fetch from NOAA if not cached, cache result, and return.

        This method:
        1. Checks cache first (returns if fresh)
        2. Fetches from HRRR via RealPredictor if cache miss
        3. Stores result in cache
        4. Logs the fetch operation

        Args:
            lat: Latitude
            lon: Longitude
            valid_time: Target forecast valid time
            forecast_hours: Forecast horizon in hours (default 24)

        Returns:
            CachedForecast if successful, None if fetch failed
        """
        # Check cache first
        cached = self.get(lat, lon, valid_time)
        if cached is not None:
            return cached

        # On Streamlit Cloud, don't attempt live fetches (herbie not installed)
        # The pre-populated cache should have all needed data
        import os
        from pathlib import Path
        is_streamlit_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or Path("/mount/src").exists()
        if is_streamlit_cloud:
            logger.info(f"Streamlit Cloud: skipping live HRRR fetch (cache miss for {lat}, {lon})")
            return None

        # Fetch from HRRR
        start_time = time.time()
        predictor = self._get_predictor()

        # Convert valid_time to date for HRRR fetch
        if isinstance(valid_time, datetime):
            target_date = valid_time.date()
        else:
            target_date = valid_time

        try:
            hrrr_data = predictor.fetch_hrrr_forecast(
                lat, lon, target_date, forecast_hours
            )
            duration_ms = int((time.time() - start_time) * 1000)

            if hrrr_data is None:
                self.db.log_fetch(
                    source="hrrr",
                    status="error",
                    records_added=0,
                    duration_ms=duration_ms,
                    error_message="HRRR fetch returned None",
                )
                return None

            # Calculate run_time based on current UTC hour
            # HRRR runs hourly (00Z, 01Z, ..., 23Z)
            # Data typically available ~45 mins after run time
            now_utc = datetime.utcnow()
            run_time = datetime(now_utc.year, now_utc.month, now_utc.day, now_utc.hour)

            # Calculate forecast hour from valid_time relative to run_time
            if isinstance(valid_time, datetime):
                hours_diff = (valid_time - run_time).total_seconds() / 3600
                forecast_hour = max(0, min(48, int(hours_diff)))
            else:
                forecast_hour = forecast_hours

            # Store in cache
            cached = self.store(
                lat=lat,
                lon=lon,
                run_time=run_time,
                forecast_hour=forecast_hour,
                snow_depth_m=hrrr_data.get("snow_depth_m", 0.0),
                temp_k=hrrr_data.get("temp_k", 273.0),
                precip_mm=hrrr_data.get("precip_mm", 0.0),
                categorical_snow=hrrr_data.get("categorical_snow", 0.0),
            )

            # Log successful fetch
            self.db.log_fetch(
                source="hrrr",
                status="success",
                records_added=1,
                duration_ms=duration_ms,
            )

            return cached

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"HRRR fetch failed: {error_msg}")
            self.db.log_fetch(
                source="hrrr",
                status="error",
                records_added=0,
                duration_ms=duration_ms,
                error_message=error_msg[:500],  # Truncate long errors
            )
            return None

    def get_or_fetch(
        self,
        lat: float,
        lon: float,
        valid_time: datetime,
        forecast_hours: int = 24,
    ) -> Optional[CachedForecast]:
        """Convenience alias for fetch_and_cache.

        Args:
            lat: Latitude
            lon: Longitude
            valid_time: Target forecast valid time
            forecast_hours: Forecast horizon in hours

        Returns:
            CachedForecast if successful, None otherwise
        """
        return self.fetch_and_cache(lat, lon, valid_time, forecast_hours)

    def cleanup_old_data(self, keep_days: int = 7) -> int:
        """Remove forecasts older than keep_days.

        Args:
            keep_days: Number of days to keep (default 7)

        Returns:
            Number of records deleted
        """
        return self.db.cleanup_old_forecasts(keep_days)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with forecast_count, latest_run_time, etc.
        """
        return self.db.get_stats()
