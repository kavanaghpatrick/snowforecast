"""Elevation band forecasting for ski resorts.

Computes snow forecasts at multiple elevation bands (Base/Mid/Summit) using
atmospheric lapse rate adjustments. Similar to Mountain-Forecast.com display.

Uses CachedPredictor for pre-computed data, NOT on-demand calculation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

# Standard atmospheric lapse rate: temperature decreases ~6.5C per 1000m
LAPSE_RATE_C_PER_KM = -6.5

# Temperature thresholds for precipitation type
SNOW_THRESHOLD_C = -1.0  # Below this: snow
RAIN_THRESHOLD_C = 2.0   # Above this: rain (between is mixed)


class PrecipType(Enum):
    """Precipitation type based on temperature."""
    SNOW = "snow"
    MIXED = "mixed"
    RAIN = "rain"
    NONE = "none"


@dataclass
class ElevationBandForecast:
    """Forecast for a single elevation band.

    Attributes:
        name: Band name (Base, Mid, Summit)
        elevation_m: Elevation in meters
        temp_c: Temperature in Celsius (adjusted by lapse rate)
        new_snow_cm: Predicted new snow in cm (0 if above freezing)
        snow_depth_cm: Existing snow depth in cm
        precip_type: Type of precipitation at this elevation
        snowfall_probability: Probability of snowfall (0-1)
    """
    name: str
    elevation_m: float
    temp_c: float
    new_snow_cm: float
    snow_depth_cm: float
    precip_type: PrecipType
    snowfall_probability: float

    @property
    def precip_icon(self) -> str:
        """Get emoji icon for precipitation type."""
        if self.precip_type == PrecipType.SNOW:
            return "snow"
        elif self.precip_type == PrecipType.MIXED:
            return "mixed"
        elif self.precip_type == PrecipType.RAIN:
            return "rain"
        return "none"

    @property
    def temp_display(self) -> str:
        """Temperature formatted for display."""
        return f"{self.temp_c:.0f}C"


@dataclass
class ElevationBandResult:
    """Complete elevation band forecast for a resort.

    Attributes:
        ski_area: Name of the ski area
        forecast_time: When forecast was generated
        base: Forecast at base elevation
        mid: Forecast at mid elevation
        summit: Forecast at summit elevation
        snow_line_m: Elevation where rain/snow transition occurs
    """
    ski_area: str
    forecast_time: datetime
    base: ElevationBandForecast
    mid: ElevationBandForecast
    summit: ElevationBandForecast
    snow_line_m: float

    @property
    def bands(self) -> list[ElevationBandForecast]:
        """Return bands in display order (Summit first)."""
        return [self.summit, self.mid, self.base]


def apply_lapse_rate(base_temp_c: float, base_elev_m: float, target_elev_m: float) -> float:
    """Apply atmospheric lapse rate to calculate temperature at different elevation.

    Args:
        base_temp_c: Temperature at base elevation in Celsius
        base_elev_m: Base elevation in meters
        target_elev_m: Target elevation in meters

    Returns:
        Adjusted temperature at target elevation in Celsius
    """
    elevation_diff_km = (target_elev_m - base_elev_m) / 1000.0
    temp_adjustment = LAPSE_RATE_C_PER_KM * elevation_diff_km
    return base_temp_c + temp_adjustment


def get_precip_type(temp_c: float, has_precip: bool) -> PrecipType:
    """Determine precipitation type based on temperature.

    Args:
        temp_c: Temperature in Celsius
        has_precip: Whether precipitation is expected

    Returns:
        PrecipType enum value
    """
    if not has_precip:
        return PrecipType.NONE

    if temp_c <= SNOW_THRESHOLD_C:
        return PrecipType.SNOW
    elif temp_c >= RAIN_THRESHOLD_C:
        return PrecipType.RAIN
    else:
        return PrecipType.MIXED


def adjust_new_snow(base_new_snow_cm: float, temp_c: float, precip_type: PrecipType) -> float:
    """Adjust new snow amount based on temperature and precip type.

    Args:
        base_new_snow_cm: Base new snow prediction
        temp_c: Temperature at this elevation
        precip_type: Type of precipitation

    Returns:
        Adjusted new snow in cm
    """
    if precip_type == PrecipType.RAIN:
        return 0.0
    elif precip_type == PrecipType.MIXED:
        # Mixed precip - reduce snow amount
        return base_new_snow_cm * 0.5
    elif precip_type == PrecipType.SNOW:
        # Colder = fluffier snow (higher snow-to-liquid ratio)
        if temp_c < -10:
            return base_new_snow_cm * 1.3  # Very cold = fluffy
        elif temp_c < -5:
            return base_new_snow_cm * 1.1  # Cold = slightly fluffy
        else:
            return base_new_snow_cm
    return 0.0


def get_snow_line(
    base_temp_c: float,
    base_elev_m: float,
    summit_elev_m: float,
) -> float:
    """Calculate the elevation where rain transitions to snow.

    Uses the freezing level (0C) adjusted by snow threshold.

    Args:
        base_temp_c: Temperature at base elevation in Celsius
        base_elev_m: Base elevation in meters
        summit_elev_m: Summit elevation in meters (upper bound)

    Returns:
        Snow line elevation in meters
    """
    # Find elevation where temperature equals snow threshold
    # temp_at_elevation = base_temp + lapse_rate * (elev - base_elev) / 1000
    # SNOW_THRESHOLD_C = base_temp_c + LAPSE_RATE_C_PER_KM * (snow_line - base_elev) / 1000
    # Solving for snow_line:
    if base_temp_c <= SNOW_THRESHOLD_C:
        # Already snowing at base
        return base_elev_m

    temp_diff = SNOW_THRESHOLD_C - base_temp_c
    elev_diff_km = temp_diff / LAPSE_RATE_C_PER_KM
    snow_line = base_elev_m + (elev_diff_km * 1000)

    # Clamp to valid range
    return max(base_elev_m, min(summit_elev_m, snow_line))


def compute_elevation_bands(
    predictor,
    lat: float,
    lon: float,
    base_elev_m: float,
    summit_elev_m: float,
    ski_area_name: str,
    target_datetime: Optional[datetime] = None,
    forecast_hours: int = 24,
) -> ElevationBandResult:
    """Compute elevation band forecasts for a ski area.

    Uses CachedPredictor for base forecast, then applies lapse rate
    to calculate conditions at different elevations.

    Args:
        predictor: CachedPredictor instance
        lat: Latitude of ski area
        lon: Longitude of ski area
        base_elev_m: Base elevation in meters
        summit_elev_m: Summit elevation in meters
        ski_area_name: Name of the ski area
        target_datetime: Target datetime for forecast (default: now)
        forecast_hours: Hours ahead to forecast

    Returns:
        ElevationBandResult with forecasts at base, mid, and summit
    """
    if target_datetime is None:
        target_datetime = datetime.now()

    # Get base forecast from CachedPredictor
    forecast, confidence = predictor.predict(lat, lon, target_datetime, forecast_hours)

    # Get HRRR data for temperature (use the cache)
    hrrr_data = predictor.fetch_hrrr_forecast(lat, lon, target_datetime.date(), forecast_hours)

    if hrrr_data is not None:
        base_temp_c = hrrr_data["temp_k"] - 273.15
        has_precip = hrrr_data.get("precip_mm", 0) > 0 or forecast.snowfall_probability > 0.3
    else:
        # Fallback - estimate from climatology
        month = target_datetime.month
        if month in (12, 1, 2):
            base_temp_c = -5.0 + (2000 - base_elev_m) * 0.0065  # Adjusted for elevation
        elif month in (3, 11):
            base_temp_c = 0.0 + (2000 - base_elev_m) * 0.0065
        else:
            base_temp_c = 5.0 + (2000 - base_elev_m) * 0.0065
        has_precip = forecast.snowfall_probability > 0.3

    # Calculate mid elevation (midpoint between base and summit)
    mid_elev_m = (base_elev_m + summit_elev_m) / 2

    # Calculate temperatures at each band
    summit_temp_c = apply_lapse_rate(base_temp_c, base_elev_m, summit_elev_m)
    mid_temp_c = apply_lapse_rate(base_temp_c, base_elev_m, mid_elev_m)

    # Determine precip types
    base_precip_type = get_precip_type(base_temp_c, has_precip)
    mid_precip_type = get_precip_type(mid_temp_c, has_precip)
    summit_precip_type = get_precip_type(summit_temp_c, has_precip)

    # Adjust new snow for each band
    base_new_snow = adjust_new_snow(forecast.new_snow_cm, base_temp_c, base_precip_type)
    mid_new_snow = adjust_new_snow(forecast.new_snow_cm, mid_temp_c, mid_precip_type)
    summit_new_snow = adjust_new_snow(forecast.new_snow_cm, summit_temp_c, summit_precip_type)

    # Adjust snow depth (more at higher elevations)
    base_depth = forecast.snow_depth_cm
    mid_depth = forecast.snow_depth_cm * 1.2  # 20% more at mid
    summit_depth = forecast.snow_depth_cm * 1.5  # 50% more at summit

    # Adjust probability based on temperature
    base_prob = forecast.snowfall_probability if base_precip_type == PrecipType.SNOW else (
        forecast.snowfall_probability * 0.5 if base_precip_type == PrecipType.MIXED else 0.1
    )
    mid_prob = forecast.snowfall_probability if mid_precip_type in (PrecipType.SNOW, PrecipType.MIXED) else 0.1
    summit_prob = forecast.snowfall_probability  # Always use full prob at summit

    # Calculate snow line
    snow_line = get_snow_line(base_temp_c, base_elev_m, summit_elev_m)

    # Create band forecasts
    base_band = ElevationBandForecast(
        name="Base",
        elevation_m=base_elev_m,
        temp_c=round(base_temp_c, 1),
        new_snow_cm=round(base_new_snow, 1),
        snow_depth_cm=round(base_depth, 1),
        precip_type=base_precip_type,
        snowfall_probability=round(base_prob, 2),
    )

    mid_band = ElevationBandForecast(
        name="Mid",
        elevation_m=mid_elev_m,
        temp_c=round(mid_temp_c, 1),
        new_snow_cm=round(mid_new_snow, 1),
        snow_depth_cm=round(mid_depth, 1),
        precip_type=mid_precip_type,
        snowfall_probability=round(mid_prob, 2),
    )

    summit_band = ElevationBandForecast(
        name="Summit",
        elevation_m=summit_elev_m,
        temp_c=round(summit_temp_c, 1),
        new_snow_cm=round(summit_new_snow, 1),
        snow_depth_cm=round(summit_depth, 1),
        precip_type=summit_precip_type,
        snowfall_probability=round(summit_prob, 2),
    )

    return ElevationBandResult(
        ski_area=ski_area_name,
        forecast_time=target_datetime,
        base=base_band,
        mid=mid_band,
        summit=summit_band,
        snow_line_m=round(snow_line, 0),
    )


# Typical ski resort vertical drops for elevation band calculation
# Base elevation is from SKI_AREAS_DATA, vertical is added to get summit
RESORT_VERTICAL_M = {
    # Washington
    "Stevens Pass": 823,  # 2064m summit
    "Crystal Mountain": 884,  # 2225m summit
    "Mt. Baker": 457,  # 1737m summit
    "Snoqualmie Pass": 610,  # 1531m summit
    # Oregon
    "Mt. Hood Meadows": 884,  # 2408m summit
    "Mt. Bachelor": 1070,  # 2990m summit
    "Timberline": 1067,  # 2896m summit
    # California
    "Mammoth Mountain": 945,  # 3369m summit
    "Squaw Valley": 869,  # 2759m summit
    "Heavenly": 1067,  # 3068m summit
    "Kirkwood": 610,  # 2987m summit
    # Colorado
    "Vail": 1064,  # 3540m summit
    "Breckenridge": 884,  # 3810m summit
    "Aspen Snowmass": 1040,  # 3513m summit
    "Telluride": 1100,  # 3759m summit
    # Utah
    "Park City": 945,  # 3048m summit
    "Snowbird": 960,  # 3325m summit
    "Alta": 671,  # 3271m summit
    # Montana
    "Big Sky": 1330,  # 3402m summit
    "Whitefish": 713,  # 2176m summit
    # Wyoming
    "Jackson Hole": 1261,  # 3185m summit
    # Idaho
    "Sun Valley": 1036,  # 2788m summit
}

# Default vertical if resort not in list
DEFAULT_VERTICAL_M = 800


def get_summit_elevation(ski_area_name: str, base_elev_m: float) -> float:
    """Get summit elevation for a ski area.

    Args:
        ski_area_name: Name of the ski area
        base_elev_m: Base elevation in meters

    Returns:
        Summit elevation in meters
    """
    vertical = RESORT_VERTICAL_M.get(ski_area_name, DEFAULT_VERTICAL_M)
    return base_elev_m + vertical
