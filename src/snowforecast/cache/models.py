"""Data models for cache layer."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class CachedForecast:
    """Cached HRRR forecast data."""

    lat: float
    lon: float
    run_time: datetime
    forecast_hour: int
    valid_time: datetime
    snow_depth_m: float
    temp_k: float
    precip_mm: float
    categorical_snow: float
    fetch_time: datetime

    @property
    def snow_depth_cm(self) -> float:
        """Snow depth in centimeters."""
        return self.snow_depth_m * 100

    @property
    def temp_c(self) -> float:
        """Temperature in Celsius."""
        return self.temp_k - 273.15


@dataclass
class CachedTerrain:
    """Cached terrain/DEM data."""

    lat: float
    lon: float
    elevation: float
    slope: float
    aspect: float
    roughness: float
    tpi: float
    fetch_time: datetime


@dataclass
class SkiArea:
    """Ski area reference data."""

    name: str
    lat: float
    lon: float
    state: str
    base_elevation: float


@dataclass
class FetchLog:
    """Log entry for data fetches."""

    source: str  # 'hrrr', 'dem'
    timestamp: datetime
    status: str  # 'success', 'error'
    records_added: int
    duration_ms: int
    error_message: Optional[str] = None


# Western US Ski Areas - canonical list
SKI_AREAS_DATA = [
    # Washington
    SkiArea("Stevens Pass", 47.7448, -121.0890, "Washington", 1241),
    SkiArea("Crystal Mountain", 46.9282, -121.5045, "Washington", 1341),
    SkiArea("Mt. Baker", 48.8570, -121.6695, "Washington", 1280),
    SkiArea("Snoqualmie Pass", 47.4204, -121.4138, "Washington", 921),
    # Oregon
    SkiArea("Mt. Hood Meadows", 45.3311, -121.6647, "Oregon", 1524),
    SkiArea("Mt. Bachelor", 43.9792, -121.6886, "Oregon", 1920),
    SkiArea("Timberline", 45.3311, -121.7110, "Oregon", 1829),
    # California
    SkiArea("Mammoth Mountain", 37.6308, -119.0326, "California", 2424),
    SkiArea("Squaw Valley", 39.1969, -120.2358, "California", 1890),
    SkiArea("Heavenly", 38.9353, -119.9400, "California", 2001),
    SkiArea("Kirkwood", 38.6850, -120.0652, "California", 2377),
    # Colorado
    SkiArea("Vail", 39.6403, -106.3742, "Colorado", 2476),
    SkiArea("Breckenridge", 39.4817, -106.0384, "Colorado", 2926),
    SkiArea("Aspen Snowmass", 39.2084, -106.9490, "Colorado", 2473),
    SkiArea("Telluride", 37.9375, -107.8123, "Colorado", 2659),
    # Utah
    SkiArea("Park City", 40.6514, -111.5080, "Utah", 2103),
    SkiArea("Snowbird", 40.5830, -111.6538, "Utah", 2365),
    SkiArea("Alta", 40.5884, -111.6386, "Utah", 2600),
    # Montana
    SkiArea("Big Sky", 45.2618, -111.4018, "Montana", 2072),
    SkiArea("Whitefish", 48.4820, -114.3556, "Montana", 1463),
    # Wyoming
    SkiArea("Jackson Hole", 43.5875, -110.8279, "Wyoming", 1924),
    # Idaho
    SkiArea("Sun Valley", 43.6804, -114.4075, "Idaho", 1752),
]
