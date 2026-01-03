"""Cached predictor using DuckDB-backed HRRR and terrain caches.

Provides drop-in replacement for RealPredictor with persistent caching.
HRRR forecasts cached for 2 hours, terrain data cached permanently.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Import directly from schemas to avoid loading api.__init__ which requires FastAPI
from snowforecast.api.schemas import ConfidenceInterval, ForecastResult
from snowforecast.cache.database import DEFAULT_DB_PATH, CacheDatabase
from snowforecast.cache.hrrr import HRRRCache
from snowforecast.cache.terrain import TerrainCache

logger = logging.getLogger(__name__)


class CachedPredictor:
    """Production predictor with persistent DuckDB caching.

    Drop-in replacement for RealPredictor that caches:
    - HRRR forecasts: 2-hour validity, 7-day retention
    - Terrain data: Permanent (terrain is static)

    First request for a location fetches from NOAA/Copernicus.
    Subsequent requests within cache window return instantly from cache.

    Example:
        >>> predictor = CachedPredictor()
        >>> forecast, ci = predictor.predict(47.74, -121.09, datetime.now())
        >>> # Second call is instant (from cache)
        >>> forecast2, ci2 = predictor.predict(47.74, -121.09, datetime.now())
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize predictor with cache database.

        Args:
            db_path: Path to DuckDB file. Uses default if not specified.
        """
        self.db = CacheDatabase(db_path or DEFAULT_DB_PATH)
        self.hrrr_cache = HRRRCache(self.db)
        self.terrain_cache = TerrainCache(self.db)
        logger.info(f"CachedPredictor initialized with db at {self.db.db_path}")

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation from cached terrain or DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters
        """
        terrain = self.terrain_cache.fetch_and_cache(lat, lon)
        if terrain is not None:
            return terrain.elevation
        return 2000.0  # Fallback

    def get_terrain_features(self, lat: float, lon: float) -> dict:
        """Get terrain features from cached terrain or DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with elevation, slope, aspect, etc.
        """
        terrain = self.terrain_cache.fetch_and_cache(lat, lon)
        if terrain is not None:
            return {
                "elevation": terrain.elevation,
                "slope": terrain.slope,
                "aspect": terrain.aspect,
                "roughness": terrain.roughness,
                "tpi": terrain.tpi,
            }
        # Fallback
        return {
            "elevation": 2000.0,
            "slope": 15.0,
            "aspect": 180.0,
            "roughness": 50.0,
            "tpi": 0.0,
        }

    def _get_real_predictor(self):
        """Get RealPredictor instance for NBM fetching."""
        if not hasattr(self, '_real_predictor') or self._real_predictor is None:
            from snowforecast.api.predictor import RealPredictor
            self._real_predictor = RealPredictor()
        return self._real_predictor

    def _fetch_nbm_forecast(
        self,
        lat: float,
        lon: float,
        target_date: date,
        forecast_hours: int = 24,
    ) -> Optional[dict]:
        """Fetch NBM forecast data for extended range forecasts.

        Delegates to RealPredictor.fetch_nbm_forecast().
        NBM provides forecasts up to 264 hours (11 days).

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Date to forecast
            forecast_hours: Hours ahead

        Returns:
            Dict with forecast variables or None if unavailable
        """
        # On Streamlit Cloud, don't attempt live fetches (herbie not installed)
        import os
        from pathlib import Path
        is_streamlit_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or Path("/mount/src").exists()
        if is_streamlit_cloud:
            logger.info(f"Streamlit Cloud: skipping NBM fetch (herbie unavailable)")
            return None

        try:
            predictor = self._get_real_predictor()
            return predictor.fetch_nbm_forecast(lat, lon, target_date, forecast_hours)
        except Exception as e:
            logger.error(f"NBM fetch via RealPredictor failed: {e}")
            return None


    def fetch_hrrr_forecast(
        self,
        lat: float,
        lon: float,
        target_date: date,
        forecast_hours: int = 24,
    ) -> Optional[dict]:
        """Fetch HRRR forecast from cache or NOAA.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Date to forecast
            forecast_hours: Hours ahead

        Returns:
            Dict with forecast variables or None if unavailable
        """
        # Convert date to datetime for cache
        if isinstance(target_date, date) and not isinstance(target_date, datetime):
            valid_time = datetime.combine(target_date, datetime.min.time())
            valid_time += timedelta(hours=forecast_hours)
        else:
            valid_time = target_date

        cached = self.hrrr_cache.fetch_and_cache(lat, lon, valid_time, forecast_hours)

        if cached is None:
            return None

        return {
            "snow_depth_m": cached.snow_depth_m,
            "snow_water_equiv_m": 0,
            "temp_k": cached.temp_k,
            "precip_rate": cached.precip_mm / (forecast_hours * 3600) if forecast_hours > 0 else 0,
            "categorical_snow": cached.categorical_snow,
            "precip_mm": cached.precip_mm,
        }

    def predict(
        self,
        lat: float,
        lon: float,
        target_date: datetime,
        forecast_hours: int = 24,
    ) -> tuple[ForecastResult, ConfidenceInterval]:
        """Generate prediction using cached data.

        Same interface as RealPredictor.predict().

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Target datetime
            forecast_hours: Forecast horizon in hours

        Returns:
            Tuple of (ForecastResult, ConfidenceInterval)
        """
        # Get terrain features (cached permanently)
        terrain = self.get_terrain_features(lat, lon)

        # Try to get HRRR forecast (cached for 2 hours)
        if isinstance(target_date, datetime):
            forecast_date = target_date.date()
        else:
            forecast_date = target_date

        hrrr_data = self.fetch_hrrr_forecast(lat, lon, forecast_date, forecast_hours)

        if hrrr_data is not None:
            # Use real HRRR data
            snow_depth_cm = hrrr_data["snow_depth_m"] * 100  # m to cm
            temp_c = hrrr_data["temp_k"] - 273.15

            # Estimate new snow from precip rate and temperature
            precip_rate = hrrr_data["precip_rate"]
            if temp_c < 0 and precip_rate > 0:
                # Cold enough for snow - estimate accumulation
                snow_ratio = 10 + max(0, -temp_c)
                new_snow_cm = precip_rate * 3600 * forecast_hours * snow_ratio / 10
            else:
                new_snow_cm = 0.0

            # Probability based on categorical snow flag and temperature
            if hrrr_data["categorical_snow"] > 0.5:
                snowfall_prob = 0.9
            elif temp_c < -2 and precip_rate > 0:
                snowfall_prob = 0.7
            elif temp_c < 2 and precip_rate > 0:
                snowfall_prob = 0.4
            else:
                snowfall_prob = 0.1

            ci_width = max(2.0, new_snow_cm * 0.3)
            logger.info(f"HRRR: depth={snow_depth_cm:.1f}cm, new={new_snow_cm:.1f}cm")

        else:
            # HRRR unavailable - try NBM for extended forecasts (days 3-7)
            logger.info("HRRR unavailable, trying NBM for extended forecast")

            nbm_data = self._fetch_nbm_forecast(lat, lon, forecast_date, forecast_hours)

            if nbm_data is not None:
                # Use real NBM data (National Blend of Models)
                snow_depth_cm = nbm_data["snow_depth_m"] * 100  # m to cm
                new_snow_cm = nbm_data["new_snow_m"] * 100  # NBM provides accumulated snow
                temp_c = nbm_data["temp_k"] - 273.15

                # Probability based on forecast snow amount
                if new_snow_cm > 5:
                    snowfall_prob = 0.85
                elif new_snow_cm > 1:
                    snowfall_prob = 0.6
                elif new_snow_cm > 0:
                    snowfall_prob = 0.3
                else:
                    snowfall_prob = 0.1

                # Wider CI for extended forecasts
                ci_width = max(3.0, new_snow_cm * 0.4)
                logger.info(f"NBM: depth={snow_depth_cm:.1f}cm, new={new_snow_cm:.1f}cm")

            else:
                # Both HRRR and NBM unavailable - this should be rare
                logger.warning("Both HRRR and NBM unavailable - no forecast data")
                snow_depth_cm = 0.0
                new_snow_cm = 0.0
                snowfall_prob = 0.0
                ci_width = 0.0

        # Apply terrain adjustments
        slope = terrain.get("slope", 15)
        if slope > 30:
            new_snow_cm *= 0.9

        aspect = terrain.get("aspect", 180)
        if 315 <= aspect or aspect <= 45:
            snow_depth_cm *= 1.05

        forecast = ForecastResult(
            snow_depth_cm=round(max(0, snow_depth_cm), 1),
            new_snow_cm=round(max(0, new_snow_cm), 1),
            snowfall_probability=round(min(0.99, max(0.01, snowfall_prob)), 2),
        )

        confidence = ConfidenceInterval(
            lower=round(max(0, new_snow_cm - ci_width), 1),
            upper=round(new_snow_cm + ci_width, 1),
        )

        return forecast, confidence

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with forecast_count, terrain_count, latest_run_time, db_path
        """
        return self.db.get_stats()

    def prefetch_terrain_for_ski_areas(self) -> int:
        """Pre-cache terrain for all 22 ski areas.

        Returns:
            Number of ski areas cached
        """
        return self.terrain_cache.prefetch_all_ski_areas()

    def cleanup_old_forecasts(self, keep_days: int = 7) -> int:
        """Remove old forecasts from cache.

        Args:
            keep_days: Number of days to keep

        Returns:
            Number of records deleted
        """
        return self.hrrr_cache.cleanup_old_data(keep_days)

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
