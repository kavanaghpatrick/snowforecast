"""Terrain feature engineering for snow prediction.

This module extends the base terrain features from DEMPipeline with additional
derived features useful for snowfall prediction models.
"""

import numpy as np
import pandas as pd

from snowforecast.utils.geo import haversine

# Pacific coast approximation (simplified polyline for Western US)
# These are lon, lat pairs forming a simplified Pacific coastline
PACIFIC_COAST_POINTS = [
    (-124.7, 48.5),   # Washington/Canada border
    (-124.2, 47.0),   # Olympic Peninsula
    (-124.0, 46.0),   # Washington coast
    (-124.0, 44.0),   # Oregon coast
    (-124.3, 42.0),   # Southern Oregon
    (-124.4, 40.5),   # Northern California
    (-123.0, 38.0),   # Bay Area
    (-121.5, 36.0),   # Central California
    (-120.5, 35.0),   # Southern Central California
    (-118.5, 34.0),   # Los Angeles area
    (-117.5, 33.0),   # San Diego area
    (-117.0, 32.5),   # Mexico border
]

# Elevation band thresholds (meters)
ELEVATION_BANDS = {
    "low": (0, 2000),
    "mid": (2000, 2500),
    "high": (2500, 3000),
    "alpine": (3000, float("inf")),
}

# Slope categories (degrees)
SLOPE_CATEGORIES = {
    "flat": (0, 5),
    "gentle": (5, 15),
    "moderate": (15, 30),
    "steep": (30, 45),
    "extreme": (45, float("inf")),
}

# Cardinal directions for aspect
ASPECT_CARDINALS = {
    "N": (337.5, 22.5),
    "NE": (22.5, 67.5),
    "E": (67.5, 112.5),
    "SE": (112.5, 157.5),
    "S": (157.5, 202.5),
    "SW": (202.5, 247.5),
    "W": (247.5, 292.5),
    "NW": (292.5, 337.5),
}

# Western US latitude range for normalization
WESTERN_US_LAT_MIN = 31.0
WESTERN_US_LAT_MAX = 49.0


def _distance_to_coast(lat: float, lon: float) -> float:
    """Calculate minimum distance to Pacific coast in km.

    Uses simplified coastline approximation for efficiency.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Distance to nearest coastline point in kilometers
    """
    min_distance = float("inf")

    # Check distance to each segment of the coastline
    for i in range(len(PACIFIC_COAST_POINTS) - 1):
        lon1, lat1 = PACIFIC_COAST_POINTS[i]
        lon2, lat2 = PACIFIC_COAST_POINTS[i + 1]

        # Project point onto line segment
        # Using simple perpendicular distance approximation
        dx = lon2 - lon1
        dy = lat2 - lat1
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            # Segment is a point
            dist = haversine(lat, lon, lat1, lon1)
        else:
            # Project point onto line, clamped to segment
            t = max(0, min(1, ((lon - lon1) * dx + (lat - lat1) * dy) / length_sq))
            proj_lon = lon1 + t * dx
            proj_lat = lat1 + t * dy
            dist = haversine(lat, lon, proj_lat, proj_lon)

        min_distance = min(min_distance, dist)

    return min_distance


def _get_elevation_band(elevation_m: float) -> str:
    """Categorize elevation into bands.

    Args:
        elevation_m: Elevation in meters

    Returns:
        Band name: 'low', 'mid', 'high', or 'alpine'
    """
    if pd.isna(elevation_m):
        return "unknown"

    for band_name, (low, high) in ELEVATION_BANDS.items():
        if low <= elevation_m < high:
            return band_name

    return "alpine"  # Default for very high elevations


def _get_slope_category(slope_deg: float) -> str:
    """Categorize slope into descriptive categories.

    Args:
        slope_deg: Slope in degrees

    Returns:
        Category name: 'flat', 'gentle', 'moderate', 'steep', or 'extreme'
    """
    if pd.isna(slope_deg):
        return "unknown"

    for cat_name, (low, high) in SLOPE_CATEGORIES.items():
        if low <= slope_deg < high:
            return cat_name

    return "extreme"  # Default for very steep slopes


def _get_aspect_cardinal(aspect_deg: float) -> str:
    """Convert aspect degrees to cardinal direction.

    Args:
        aspect_deg: Aspect in degrees (0=N, 90=E, 180=S, 270=W)

    Returns:
        Cardinal direction: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'
    """
    if pd.isna(aspect_deg):
        return "unknown"

    # Normalize to 0-360 range
    aspect_deg = aspect_deg % 360

    # North spans 337.5-360 and 0-22.5, handle specially
    if aspect_deg >= 337.5 or aspect_deg < 22.5:
        return "N"

    for cardinal, (low, high) in ASPECT_CARDINALS.items():
        if cardinal == "N":
            continue  # Already handled above
        if low <= aspect_deg < high:
            return cardinal

    return "N"  # Default fallback


def _is_north_facing(aspect_deg: float) -> int:
    """Determine if aspect is north-facing (315-45 degrees).

    North-facing slopes receive less direct sunlight and tend to
    retain snow longer.

    Args:
        aspect_deg: Aspect in degrees

    Returns:
        1 if north-facing, 0 otherwise
    """
    if pd.isna(aspect_deg):
        return 0

    aspect_deg = aspect_deg % 360
    return 1 if aspect_deg >= 315 or aspect_deg <= 45 else 0


def _compute_solar_exposure(aspect_deg: float, slope_deg: float) -> float:
    """Compute solar exposure index based on aspect and slope.

    South-facing slopes with steeper angles receive more direct sunlight.
    Returns a value from 0 (north-facing, minimal sun) to 1 (south-facing, maximum sun).

    Args:
        aspect_deg: Aspect in degrees (0=N, 180=S)
        slope_deg: Slope in degrees

    Returns:
        Solar exposure index (0-1)
    """
    if pd.isna(aspect_deg) or pd.isna(slope_deg):
        return 0.5  # Neutral default

    # Convert aspect to southness factor (1 = south, 0 = north)
    # cos(aspect) gives north=1, south=-1, so we negate and normalize to 0-1
    aspect_rad = np.radians(aspect_deg)
    southness = (1 - np.cos(aspect_rad)) / 2  # 0=north, 1=south

    # Steeper slopes amplify the effect
    slope_factor = min(slope_deg / 45.0, 1.0)  # Cap at 45 degrees

    # Combine: south-facing steep slopes get high exposure
    # Flat terrain gets 0.5 regardless of aspect
    exposure = 0.5 + (southness - 0.5) * slope_factor

    return float(exposure)


def _compute_wind_exposure(tpi: float) -> float:
    """Compute wind exposure from Topographic Position Index.

    Positive TPI indicates ridges (exposed to wind),
    negative TPI indicates valleys (sheltered).

    Args:
        tpi: Topographic Position Index value

    Returns:
        Wind exposure index normalized to roughly 0-1 range
    """
    if pd.isna(tpi):
        return 0.5  # Neutral default

    # Normalize TPI to roughly 0-1 range
    # TPI typically ranges from -50 to +50 meters
    exposure = 0.5 + tpi / 100.0
    return float(max(0, min(1, exposure)))


class TerrainFeatureEngineer:
    """Engineer terrain features for snow prediction.

    This class extends base terrain features from DEMPipeline with
    additional derived features useful for snowfall prediction,
    including elevation bands, slope categories, aspect encoding,
    exposure indices, and geographic features.
    """

    def __init__(self, dem_pipeline=None):
        """Initialize with optional DEM pipeline instance.

        Args:
            dem_pipeline: Optional DEMPipeline instance for extracting
                base terrain features. If None, expects input data
                to already contain base features.
        """
        self.dem_pipeline = dem_pipeline

    def get_base_terrain(self, lat: float, lon: float) -> dict:
        """Get base terrain features from DEM pipeline.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Dictionary with base terrain features

        Raises:
            ValueError: If no DEM pipeline is configured or point is outside data
        """
        if self.dem_pipeline is None:
            raise ValueError(
                "No DEM pipeline configured. Either provide one at init "
                "or use input data that already contains base terrain features."
            )

        features = self.dem_pipeline.get_terrain_features(lat, lon)
        return {
            "elevation": features.elevation,
            "slope": features.slope,
            "aspect": features.aspect,
            "aspect_sin": features.aspect_sin,
            "aspect_cos": features.aspect_cos,
            "roughness": features.roughness,
            "tpi": features.tpi,
        }

    def compute_elevation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute elevation-derived features.

        Features computed:
        - elevation_m: Elevation in meters (passthrough or from 'elevation')
        - elevation_km: Elevation in kilometers
        - elevation_band: Categorical (low/mid/high/alpine)

        Args:
            df: DataFrame with 'elevation' or 'elevation_m' column

        Returns:
            DataFrame with elevation features added
        """
        result = df.copy()

        # Get elevation column (handle both naming conventions)
        if "elevation_m" in result.columns:
            elev_col = "elevation_m"
        elif "elevation" in result.columns:
            elev_col = "elevation"
            result["elevation_m"] = result["elevation"]
        else:
            raise ValueError("DataFrame must contain 'elevation' or 'elevation_m' column")

        # Compute derived features
        result["elevation_km"] = result[elev_col] / 1000.0
        result["elevation_band"] = result[elev_col].apply(_get_elevation_band)

        return result

    def compute_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute slope-derived features.

        Features computed:
        - slope_deg: Slope in degrees (passthrough or from 'slope')
        - slope_category: flat/gentle/moderate/steep/extreme

        Args:
            df: DataFrame with 'slope' or 'slope_deg' column

        Returns:
            DataFrame with slope features added
        """
        result = df.copy()

        # Get slope column (handle both naming conventions)
        if "slope_deg" in result.columns:
            slope_col = "slope_deg"
        elif "slope" in result.columns:
            slope_col = "slope"
            result["slope_deg"] = result["slope"]
        else:
            raise ValueError("DataFrame must contain 'slope' or 'slope_deg' column")

        # Compute derived features
        result["slope_category"] = result[slope_col].apply(_get_slope_category)

        return result

    def compute_aspect_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aspect-derived features.

        Features computed:
        - aspect_deg: Aspect in degrees (passthrough or from 'aspect')
        - aspect_sin: sin(aspect) for cyclical encoding
        - aspect_cos: cos(aspect) for cyclical encoding
        - aspect_cardinal: N/NE/E/SE/S/SW/W/NW
        - north_facing: 1 if aspect 315-45, else 0

        Args:
            df: DataFrame with 'aspect' or 'aspect_deg' column

        Returns:
            DataFrame with aspect features added
        """
        result = df.copy()

        # Get aspect column (handle both naming conventions)
        if "aspect_deg" in result.columns:
            aspect_col = "aspect_deg"
        elif "aspect" in result.columns:
            aspect_col = "aspect"
            result["aspect_deg"] = result["aspect"]
        else:
            raise ValueError("DataFrame must contain 'aspect' or 'aspect_deg' column")

        # Compute cyclical encoding if not already present
        if "aspect_sin" not in result.columns:
            result["aspect_sin"] = np.sin(np.radians(result[aspect_col]))
        if "aspect_cos" not in result.columns:
            result["aspect_cos"] = np.cos(np.radians(result[aspect_col]))

        # Compute categorical and indicator features
        result["aspect_cardinal"] = result[aspect_col].apply(_get_aspect_cardinal)
        result["north_facing"] = result[aspect_col].apply(_is_north_facing)

        return result

    def compute_exposure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute exposure-derived features.

        Features computed:
        - wind_exposure: Based on TPI (positive = exposed ridge)
        - solar_exposure: Based on aspect and slope (south-facing = more sun)
        - terrain_roughness: Standard deviation of nearby elevations (passthrough)

        Args:
            df: DataFrame with 'tpi', 'aspect', and 'slope' columns

        Returns:
            DataFrame with exposure features added
        """
        result = df.copy()

        # Wind exposure from TPI
        if "tpi" in result.columns:
            result["wind_exposure"] = result["tpi"].apply(_compute_wind_exposure)
        else:
            result["wind_exposure"] = 0.5  # Neutral default

        # Solar exposure from aspect and slope
        aspect_col = "aspect_deg" if "aspect_deg" in result.columns else "aspect"
        slope_col = "slope_deg" if "slope_deg" in result.columns else "slope"

        if aspect_col in result.columns and slope_col in result.columns:
            result["solar_exposure"] = result.apply(
                lambda row: _compute_solar_exposure(row[aspect_col], row[slope_col]),
                axis=1
            )
        else:
            result["solar_exposure"] = 0.5  # Neutral default

        # Terrain roughness (passthrough if present)
        if "roughness" in result.columns:
            result["terrain_roughness"] = result["roughness"]
        elif "terrain_roughness" not in result.columns:
            result["terrain_roughness"] = 0.0

        return result

    def compute_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute geographic features.

        Features computed:
        - distance_to_coast_km: Distance to Pacific coast (maritime influence)
        - latitude_normalized: Latitude scaled 0-1 for Western US

        Args:
            df: DataFrame with 'lat' and 'lon' columns

        Returns:
            DataFrame with geographic features added
        """
        result = df.copy()

        if "lat" not in result.columns or "lon" not in result.columns:
            raise ValueError("DataFrame must contain 'lat' and 'lon' columns")

        # Distance to Pacific coast
        result["distance_to_coast_km"] = result.apply(
            lambda row: _distance_to_coast(row["lat"], row["lon"]),
            axis=1
        )

        # Normalized latitude for Western US
        result["latitude_normalized"] = (
            (result["lat"] - WESTERN_US_LAT_MIN)
            / (WESTERN_US_LAT_MAX - WESTERN_US_LAT_MIN)
        ).clip(0, 1)

        return result

    def compute_all(self, locations: pd.DataFrame) -> pd.DataFrame:
        """Compute all terrain features for locations.

        If a DEM pipeline is configured, base terrain features will be
        extracted first. Otherwise, the input must contain base features.

        Args:
            locations: DataFrame with lat, lon columns (and optionally
                base terrain features if no DEM pipeline is configured)

        Returns:
            DataFrame with all terrain features added
        """
        result = locations.copy()

        # Get base terrain features if we have a DEM pipeline
        if self.dem_pipeline is not None:
            # Fetch terrain features for each location
            base_features = []
            for _, row in result.iterrows():
                try:
                    features = self.get_base_terrain(row["lat"], row["lon"])
                    base_features.append(features)
                except (ValueError, RuntimeError):
                    # No data for this location
                    base_features.append({
                        "elevation": np.nan,
                        "slope": np.nan,
                        "aspect": np.nan,
                        "aspect_sin": np.nan,
                        "aspect_cos": np.nan,
                        "roughness": np.nan,
                        "tpi": np.nan,
                    })

            base_df = pd.DataFrame(base_features)
            for col in base_df.columns:
                result[col] = base_df[col].values

        # Apply all feature computations
        result = self.compute_elevation_features(result)
        result = self.compute_slope_features(result)
        result = self.compute_aspect_features(result)
        result = self.compute_exposure_features(result)
        result = self.compute_geographic_features(result)

        return result

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names computed by this class.

        Returns:
            List of feature column names
        """
        return [
            # Elevation features
            "elevation_m",
            "elevation_km",
            "elevation_band",
            # Slope features
            "slope_deg",
            "slope_category",
            # Aspect features
            "aspect_deg",
            "aspect_sin",
            "aspect_cos",
            "aspect_cardinal",
            "north_facing",
            # Exposure features
            "wind_exposure",
            "solar_exposure",
            "terrain_roughness",
            # Geographic features
            "distance_to_coast_km",
            "latitude_normalized",
        ]
