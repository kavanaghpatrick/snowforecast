"""Real predictor using HRRR forecasts and DEM terrain data.

This module provides production-ready predictions by:
1. Fetching real HRRR weather forecasts from NOAA
2. Getting terrain features from Copernicus DEM
3. Combining into snow predictions
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from snowforecast.api.schemas import ConfidenceInterval, ForecastResult

logger = logging.getLogger(__name__)


class RealPredictor:
    """Production predictor using real HRRR and DEM data.

    Uses:
    - HRRR: 3km resolution weather forecasts (SNOD, WEASD, TMP, PRATE)
    - DEM: 30m terrain data (elevation, slope, aspect)

    Example:
        >>> predictor = RealPredictor()
        >>> forecast, ci = predictor.predict(47.74, -121.09, datetime.now())
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize predictor with optional cache directory."""
        self._hrrr_pipeline = None
        self._dem_pipeline = None
        self._cache_dir = cache_dir or Path("data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_hrrr(self):
        """Lazy load HRRR pipeline."""
        if self._hrrr_pipeline is None:
            try:
                from snowforecast.pipelines.hrrr import HRRRPipeline
                self._hrrr_pipeline = HRRRPipeline()
                logger.info("HRRR pipeline initialized")
            except ImportError as e:
                logger.warning(f"HRRR pipeline not available: {e}")
                return None
        return self._hrrr_pipeline

    def _get_dem(self):
        """Lazy load DEM pipeline."""
        if self._dem_pipeline is None:
            try:
                from snowforecast.pipelines.dem import DEMPipeline
                self._dem_pipeline = DEMPipeline()
                logger.info("DEM pipeline initialized")
            except ImportError as e:
                logger.warning(f"DEM pipeline not available: {e}")
                return None
        return self._dem_pipeline

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation from DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters, or estimate if DEM unavailable
        """
        dem = self._get_dem()
        if dem is not None:
            try:
                elev = dem.get_elevation(lat, lon)
                if elev is not None and not np.isnan(elev):
                    return float(elev)
            except Exception as e:
                logger.warning(f"DEM lookup failed: {e}")

        # Fallback: rough elevation estimate based on location
        # Higher latitudes and certain longitudes tend to be higher
        base = 1500 + (lat - 35) * 80
        return max(500, min(4000, base))

    def get_terrain_features(self, lat: float, lon: float) -> dict:
        """Get terrain features from DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with elevation, slope, aspect, etc.
        """
        dem = self._get_dem()
        if dem is not None:
            try:
                features = dem.get_terrain_features(lat, lon)
                if features is not None:
                    # Convert dataclass to dict if needed
                    if hasattr(features, '__dict__'):
                        return {
                            "elevation": getattr(features, 'elevation', 2000),
                            "slope": getattr(features, 'slope', 15),
                            "aspect": getattr(features, 'aspect', 180),
                            "roughness": getattr(features, 'roughness', 50),
                            "tpi": getattr(features, 'tpi', 0),
                        }
                    return features
            except Exception as e:
                logger.warning(f"Terrain features failed: {e}")

        # Fallback
        return {
            "elevation": self.get_elevation(lat, lon),
            "slope": 15.0,
            "aspect": 180.0,
            "roughness": 50.0,
            "tpi": 0.0,
        }

    def _find_nearest_hrrr_point(self, ds, lat: float, lon: float) -> tuple:
        """Find nearest y,x indices for lat/lon in HRRR Lambert grid.

        HRRR uses Lambert Conformal projection, so lat/lon are 2D arrays.
        We need to find the grid point closest to our target.
        """
        import numpy as np

        # Get lat/lon arrays
        lats = ds.latitude.values
        lons = ds.longitude.values

        # Convert target lon to 0-360 range like HRRR uses
        target_lon = lon + 360 if lon < 0 else lon

        # Calculate distance to each grid point
        dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)

        # Find minimum
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx  # (y_idx, x_idx)

    def fetch_hrrr_forecast(
        self,
        lat: float,
        lon: float,
        target_date: date,
        forecast_hours: int = 24,
    ) -> Optional[dict]:
        """Fetch HRRR forecast data for location using Herbie directly.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Date to forecast
            forecast_hours: Hours ahead (used as fxx for today's run)

        Returns:
            Dict with forecast variables or None if unavailable
        """
        try:
            from datetime import date as date_type

            from herbie import Herbie

            # HRRR provides forecasts from the current run time
            # Use today's run with appropriate forecast hour (fxx)
            # HRRR only forecasts up to 48 hours ahead
            today = date_type.today()

            # Calculate forecast hour offset from today
            if hasattr(target_date, 'date'):
                target = target_date.date()
            else:
                target = target_date

            days_ahead = (target - today).days
            hours_ahead = days_ahead * 24 + forecast_hours

            # HRRR only provides 48-hour forecasts - return None for beyond
            if hours_ahead > 48:
                logger.info(f"Target date {target} is {hours_ahead}h ahead, beyond HRRR 48h limit")
                return None

            fxx = max(0, hours_ahead)

            # Get the HRRR forecast from today's run
            h = Herbie(
                today,
                model="hrrr",
                product="sfc",
                fxx=fxx,
            )

            snow_depth = 0.0
            temp_k = 273.0
            precip = 0.0
            cat_snow = 0.0

            # Download snow depth
            try:
                ds = h.xarray(":SNOD:surface", remove_grib=True)
                y_idx, x_idx = self._find_nearest_hrrr_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                snow_depth = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"HRRR snow depth at ({y_idx},{x_idx}): {snow_depth:.3f}m")
            except Exception as e:
                logger.warning(f"HRRR snow depth failed: {e}")

            # Download temperature
            try:
                ds = h.xarray(":TMP:2 m above", remove_grib=True)
                y_idx, x_idx = self._find_nearest_hrrr_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                temp_k = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"HRRR temp: {temp_k:.1f}K ({temp_k-273.15:.1f}C)")
            except Exception as e:
                logger.warning(f"HRRR temp failed: {e}")

            # Download accumulated precip
            try:
                ds = h.xarray(":APCP:surface", remove_grib=True)
                y_idx, x_idx = self._find_nearest_hrrr_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                precip = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"HRRR precip: {precip:.2f}mm")
            except Exception as e:
                logger.warning(f"HRRR precip failed: {e}")

            # Download categorical snow flag
            try:
                ds = h.xarray(":CSNOW:surface", remove_grib=True)
                y_idx, x_idx = self._find_nearest_hrrr_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                cat_snow = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
            except Exception as e:
                logger.debug(f"HRRR categorical snow failed: {e}")

            return {
                "snow_depth_m": snow_depth,
                "snow_water_equiv_m": 0,
                "temp_k": temp_k,
                "precip_rate": precip / (forecast_hours * 3600) if forecast_hours > 0 else 0,
                "categorical_snow": cat_snow,
                "precip_mm": precip,
            }

        except Exception as e:
            logger.error(f"HRRR fetch failed: {e}")
            return None

    def _find_nearest_nbm_point(self, ds, lat: float, lon: float) -> tuple:
        """Find nearest y,x indices for lat/lon in NBM grid.

        NBM uses Lambert Conformal projection similar to HRRR.
        """
        import numpy as np

        # Get lat/lon arrays
        lats = ds.latitude.values
        lons = ds.longitude.values

        # Convert target lon to match NBM convention
        target_lon = lon + 360 if lon < 0 else lon

        # Calculate distance to each grid point
        dist = np.sqrt((lats - lat)**2 + (lons - target_lon)**2)

        # Find minimum
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return idx  # (y_idx, x_idx)

    def fetch_nbm_forecast(
        self,
        lat: float,
        lon: float,
        target_date: date,
        forecast_hours: int = 24,
    ) -> Optional[dict]:
        """Fetch NBM (National Blend of Models) forecast for extended range.

        NBM provides forecasts from 1-264 hours (up to 11 days).
        Use for forecasts beyond HRRR's 48-hour limit.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Date to forecast
            forecast_hours: Hours ahead

        Returns:
            Dict with forecast variables or None if unavailable
        """
        try:
            from datetime import date as date_type

            from herbie import Herbie

            today = date_type.today()

            # Calculate forecast hour offset from today
            if hasattr(target_date, 'date'):
                target = target_date.date()
            else:
                target = target_date

            days_ahead = (target - today).days
            hours_ahead = days_ahead * 24 + forecast_hours

            # NBM provides forecasts up to 264 hours (11 days)
            if hours_ahead > 264:
                logger.info(f"Target {target} is {hours_ahead}h ahead, beyond NBM 264h limit")
                return None

            # NBM forecast hours are in specific intervals
            # Round to nearest available forecast hour (every 3h for days 1-3, every 6h after)
            if hours_ahead <= 72:
                fxx = round(hours_ahead / 3) * 3  # Round to nearest 3h
            else:
                fxx = round(hours_ahead / 6) * 6  # Round to nearest 6h

            fxx = max(1, min(fxx, 264))

            logger.info(f"Fetching NBM forecast for {target}, fxx={fxx}h")

            # Get NBM forecast
            h = Herbie(
                today,
                model="nbm",
                product="co",  # CONUS core product
                fxx=fxx,
            )

            snow_depth = 0.0
            temp_k = 273.0
            precip = 0.0
            asnow = 0.0  # Accumulated snow

            # Download accumulated snow (NBM's primary snow variable)
            try:
                ds = h.xarray(":ASNOW:", remove_grib=True)
                y_idx, x_idx = self._find_nearest_nbm_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                # ASNOW is in meters, accumulated over forecast period
                asnow = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"NBM accumulated snow: {asnow:.3f}m ({asnow*100:.1f}cm)")
            except Exception as e:
                logger.warning(f"NBM ASNOW failed: {e}")

            # Download temperature
            try:
                ds = h.xarray(":TMP:2 m above", remove_grib=True)
                y_idx, x_idx = self._find_nearest_nbm_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                temp_k = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"NBM temp: {temp_k:.1f}K ({temp_k-273.15:.1f}C)")
            except Exception as e:
                logger.warning(f"NBM temp failed: {e}")

            # Download total precipitation
            try:
                ds = h.xarray(":APCP:", remove_grib=True)
                y_idx, x_idx = self._find_nearest_nbm_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                precip = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
                logger.info(f"NBM precip: {precip:.2f}mm")
            except Exception as e:
                logger.debug(f"NBM precip failed: {e}")

            # Try to get snow depth if available
            try:
                ds = h.xarray(":SNOD:", remove_grib=True)
                y_idx, x_idx = self._find_nearest_nbm_point(ds, lat, lon)
                var_name = list(ds.data_vars)[0]
                snow_depth = float(ds[var_name].isel(y=y_idx, x=x_idx).values)
            except Exception as e:
                # Snow depth not always available in NBM, estimate from ASNOW
                snow_depth = asnow  # Use accumulated snow as proxy
                logger.debug(f"NBM SNOD not available, using ASNOW: {e}")

            return {
                "snow_depth_m": snow_depth,
                "new_snow_m": asnow,  # NBM provides accumulated snow directly!
                "temp_k": temp_k,
                "precip_mm": precip,
                "source": "nbm",
            }

        except Exception as e:
            logger.error(f"NBM fetch failed: {e}")
            return None

    def predict(
        self,
        lat: float,
        lon: float,
        target_date: datetime,
        forecast_hours: int = 24,
    ) -> tuple[ForecastResult, ConfidenceInterval]:
        """Generate prediction using real data.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Target datetime
            forecast_hours: Forecast horizon in hours

        Returns:
            Tuple of (ForecastResult, ConfidenceInterval)
        """
        # Get terrain features
        terrain = self.get_terrain_features(lat, lon)

        # Try to get HRRR forecast
        if isinstance(target_date, datetime):
            forecast_date = target_date.date()
        else:
            forecast_date = target_date

        # Try HRRR first (0-48h), then NBM (49-168h)
        hrrr_data = self.fetch_hrrr_forecast(lat, lon, forecast_date, forecast_hours)

        if hrrr_data is not None:
            # Use real HRRR data (highest resolution, best for 0-48h)
            snow_depth_cm = hrrr_data["snow_depth_m"] * 100  # m to cm
            temp_c = hrrr_data["temp_k"] - 273.15

            # Estimate new snow from precip rate and temperature
            precip_rate = hrrr_data["precip_rate"]
            if temp_c < 0 and precip_rate > 0:
                # Cold enough for snow - estimate accumulation
                snow_ratio = 10 + max(0, -temp_c)  # Higher ratio when colder
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
            # Try NBM for extended forecasts (49-168h)
            nbm_data = self.fetch_nbm_forecast(lat, lon, forecast_date, forecast_hours)

            if nbm_data is not None:
                # Use real NBM data
                snow_depth_cm = nbm_data["snow_depth_m"] * 100  # m to cm
                new_snow_cm = nbm_data["new_snow_m"] * 100  # NBM provides accumulated snow directly
                temp_c = nbm_data["temp_k"] - 273.15

                # Probability based on whether there's snow in the forecast
                if new_snow_cm > 5:
                    snowfall_prob = 0.85
                elif new_snow_cm > 1:
                    snowfall_prob = 0.6
                elif new_snow_cm > 0:
                    snowfall_prob = 0.3
                else:
                    snowfall_prob = 0.1

                # Wider confidence interval for extended forecasts
                ci_width = max(3.0, new_snow_cm * 0.4)
                logger.info(f"NBM: depth={snow_depth_cm:.1f}cm, new={new_snow_cm:.1f}cm")

            else:
                # Last resort: no forecast data available
                # This should be rare - only if both HRRR and NBM fail
                logger.warning("Both HRRR and NBM unavailable - no forecast data")
                snow_depth_cm = 0.0
                new_snow_cm = 0.0
                snowfall_prob = 0.0
                ci_width = 0.0

        # Apply terrain adjustments
        # Higher slopes = more wind-loading potential
        slope = terrain.get("slope", 15)
        if slope > 30:
            new_snow_cm *= 0.9  # Some wind scouring on steep slopes

        # North-facing slopes hold snow better
        aspect = terrain.get("aspect", 180)
        if 315 <= aspect or aspect <= 45:  # North-ish
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


class HybridPredictor:
    """Hybrid predictor that uses real data when available, falls back to mock.

    This is useful during development/testing when real data sources
    may not always be available.
    """

    def __init__(self):
        self._real = None
        self._mock = None

    def _get_real(self):
        if self._real is None:
            self._real = RealPredictor()
        return self._real

    def _get_mock(self):
        if self._mock is None:
            from snowforecast.api.app import MockPredictor
            self._mock = MockPredictor()
        return self._mock

    def predict(
        self,
        lat: float,
        lon: float,
        target_date: datetime,
        forecast_hours: int = 24,
    ) -> tuple[ForecastResult, ConfidenceInterval]:
        """Try real predictor, fall back to mock."""
        try:
            return self._get_real().predict(lat, lon, target_date, forecast_hours)
        except Exception as e:
            logger.warning(f"Real predictor failed, using mock: {e}")
            return self._get_mock().predict(lat, lon, target_date, forecast_hours)

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation, trying real DEM first."""
        try:
            return self._get_real().get_elevation(lat, lon)
        except Exception:
            return 1500 + (lat - 35) * 80
