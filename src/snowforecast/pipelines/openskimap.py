"""OpenSkiMap ski resort location extraction pipeline.

This module extracts ski resort locations from OpenSkiMap, which provides
GeoJSON exports of ski areas and lifts based on OpenStreetMap data.

Data Sources:
- Ski areas: https://tiles.openskimap.org/geojson/ski_areas.geojson
- Lifts: https://tiles.openskimap.org/geojson/lifts.geojson
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import geojson
import pandas as pd
import requests
from shapely.geometry import shape

from snowforecast.utils import (
    StaticPipeline,
    ValidationResult,
    WESTERN_US_BBOX,
)
from snowforecast.utils.geo import haversine
from snowforecast.utils.io import get_data_path


# OpenSkiMap GeoJSON download URLs
SKI_AREAS_URL = "https://tiles.openskimap.org/geojson/ski_areas.geojson"
LIFTS_URL = "https://tiles.openskimap.org/geojson/lifts.geojson"

# Request timeout in seconds
REQUEST_TIMEOUT = 120


@dataclass
class SkiResort:
    """Ski resort location and elevation data.

    Attributes:
        name: Resort name
        lat: Representative latitude (centroid of ski area)
        lon: Representative longitude (centroid of ski area)
        base_elevation: Base area elevation in meters
        summit_elevation: Summit elevation in meters
        vertical_drop: Vertical drop (summit - base) in meters
        country: Country code (e.g., 'US', 'CA')
        state: State/province code for US/Canada (e.g., 'CO', 'UT')
        nearest_snotel: Nearest SNOTEL station ID (if found)
    """

    name: str
    lat: float
    lon: float
    base_elevation: float | None
    summit_elevation: float | None
    vertical_drop: float | None
    country: str
    state: str
    nearest_snotel: str | None = None


class OpenSkiMapPipeline(StaticPipeline):
    """Ski resort location extraction pipeline from OpenSkiMap.

    This pipeline downloads ski area and lift GeoJSON data from OpenSkiMap,
    extracts resort locations with elevations, and filters to the Western US.

    Example:
        >>> pipeline = OpenSkiMapPipeline()
        >>> resorts_df, validation = pipeline.run()
        >>> print(f"Found {len(resorts_df)} Western US ski resorts")
    """

    def __init__(self) -> None:
        """Initialize the pipeline with data paths."""
        self.raw_path = get_data_path("openskimap", "raw")
        self.processed_path = get_data_path("openskimap", "processed")

    def download(self, **kwargs) -> Path:
        """Download ski areas and lifts GeoJSON files.

        Returns:
            Path to the raw data directory containing downloaded files.
        """
        self.download_ski_areas()
        self.download_lifts()
        return self.raw_path

    def download_ski_areas(self) -> Path:
        """Download ski areas GeoJSON from OpenSkiMap.

        Returns:
            Path to the downloaded ski_areas.geojson file.

        Raises:
            requests.HTTPError: If download fails.
        """
        output_path = self.raw_path / "ski_areas.geojson"

        response = requests.get(SKI_AREAS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        with open(output_path, "w") as f:
            f.write(response.text)

        return output_path

    def download_lifts(self) -> Path:
        """Download lifts GeoJSON from OpenSkiMap.

        Returns:
            Path to the downloaded lifts.geojson file.

        Raises:
            requests.HTTPError: If download fails.
        """
        output_path = self.raw_path / "lifts.geojson"

        response = requests.get(LIFTS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        with open(output_path, "w") as f:
            f.write(response.text)

        return output_path

    def process(self, raw_path: Path) -> pd.DataFrame:
        """Process raw GeoJSON data into ski resort DataFrame.

        Args:
            raw_path: Path to raw data directory.

        Returns:
            DataFrame with all Western US ski resorts.
        """
        resorts = self.get_western_us_resorts()
        return self.export_to_dataframe(resorts)

    def parse_ski_areas(self, geojson_path: Path) -> list[dict]:
        """Parse ski area features from GeoJSON file.

        Args:
            geojson_path: Path to ski_areas.geojson file.

        Returns:
            List of ski area feature dictionaries.
        """
        with open(geojson_path) as f:
            data = geojson.load(f)

        features = []
        for feature in data.get("features", []):
            if feature.get("geometry") is None:
                continue

            properties = feature.get("properties", {})
            geometry = feature.get("geometry")

            # Extract name from properties
            name = properties.get("name")
            if not name:
                continue

            # Get centroid for representative lat/lon
            try:
                shp = shape(geometry)
                centroid = shp.centroid
                lat, lon = centroid.y, centroid.x
            except Exception:
                continue

            features.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "geometry": geometry,
                "properties": properties,
            })

        return features

    def extract_elevations_from_lifts(
        self,
        ski_area_name: str,
        lifts_data: dict,
    ) -> tuple[float | None, float | None]:
        """Extract base and summit elevation from lift endpoints.

        Lifts often have 3D coordinates [lon, lat, elevation] that we can use
        to determine the min (base) and max (summit) elevations for a ski area.

        Args:
            ski_area_name: Name of the ski area to match.
            lifts_data: Parsed GeoJSON data for lifts.

        Returns:
            Tuple of (base_elevation, summit_elevation) in meters, or (None, None)
            if no elevation data is available.
        """
        elevations = []

        for feature in lifts_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            # Check if this lift belongs to the ski area
            lift_ski_areas = properties.get("ski_areas", [])
            if isinstance(lift_ski_areas, list):
                # Check if ski_area_name matches any of the associated ski areas
                lift_names = [
                    sa.get("properties", {}).get("name", "")
                    for sa in lift_ski_areas
                    if isinstance(sa, dict)
                ]
                if ski_area_name not in lift_names:
                    continue
            else:
                continue

            # Extract elevations from coordinates
            coords = geometry.get("coordinates", [])
            for coord in coords:
                # Coordinates may be [lon, lat] or [lon, lat, elevation]
                if isinstance(coord, (list, tuple)) and len(coord) >= 3:
                    elevations.append(coord[2])

        if elevations:
            return min(elevations), max(elevations)
        return None, None

    def filter_region(
        self,
        resorts: list[SkiResort],
        bbox: dict | None = None,
        countries: list[str] | None = None,
    ) -> list[SkiResort]:
        """Filter resorts by geographic region.

        Args:
            resorts: List of ski resorts to filter.
            bbox: Bounding box dict with 'west', 'east', 'south', 'north' keys.
            countries: List of country codes to include (e.g., ['US', 'CA']).

        Returns:
            Filtered list of ski resorts.
        """
        filtered = []

        for resort in resorts:
            # Filter by bounding box
            if bbox is not None:
                if not (
                    bbox["west"] <= resort.lon <= bbox["east"]
                    and bbox["south"] <= resort.lat <= bbox["north"]
                ):
                    continue

            # Filter by country
            if countries is not None:
                if resort.country not in countries:
                    continue

            filtered.append(resort)

        return filtered

    def find_nearest_snotel(
        self,
        resort: SkiResort,
        snotel_stations: pd.DataFrame,
        max_distance_km: float = 50.0,
    ) -> str | None:
        """Find nearest SNOTEL station to a ski resort.

        Args:
            resort: Ski resort to find nearest station for.
            snotel_stations: DataFrame with SNOTEL stations, must have
                'station_id', 'lat', 'lon' columns.
            max_distance_km: Maximum distance in km to consider.

        Returns:
            Station ID of nearest SNOTEL station, or None if none within range.
        """
        if snotel_stations.empty:
            return None

        min_distance = float("inf")
        nearest_id = None

        for _, station in snotel_stations.iterrows():
            distance = haversine(
                resort.lat, resort.lon,
                station["lat"], station["lon"]
            )
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest_id = station["station_id"]

        return nearest_id

    def get_western_us_resorts(self) -> list[SkiResort]:
        """Get all ski resorts in the Western US.

        Downloads data if not already present, parses ski areas and lifts,
        extracts elevations, and filters to Western US bounding box.

        Returns:
            List of SkiResort objects for Western US ski areas.
        """
        ski_areas_path = self.raw_path / "ski_areas.geojson"
        lifts_path = self.raw_path / "lifts.geojson"

        # Download if files don't exist
        if not ski_areas_path.exists():
            self.download_ski_areas()
        if not lifts_path.exists():
            self.download_lifts()

        # Parse ski areas
        ski_areas = self.parse_ski_areas(ski_areas_path)

        # Load lifts data for elevation extraction
        with open(lifts_path) as f:
            lifts_data = geojson.load(f)

        resorts = []
        for area in ski_areas:
            # Extract elevations from associated lifts
            base_elev, summit_elev = self.extract_elevations_from_lifts(
                area["name"], lifts_data
            )

            # Calculate vertical drop if both elevations available
            vertical_drop = None
            if base_elev is not None and summit_elev is not None:
                vertical_drop = summit_elev - base_elev

            # Extract country and state from properties
            properties = area.get("properties", {})
            location = properties.get("location", {})

            # Try to get country/state from location object
            country = ""
            state = ""
            if isinstance(location, dict):
                locales = location.get("localized", {}).get("en", {})
                if isinstance(locales, dict):
                    country = locales.get("country", {}).get("name", "")
                    state = locales.get("region", {}).get("name", "")

            # Fallback: try to extract from ISO3166-2 or address
            if not country:
                iso = properties.get("sources", [])
                if isinstance(iso, list) and iso:
                    first_source = iso[0] if isinstance(iso[0], dict) else {}
                    country = first_source.get("country", "")

            resort = SkiResort(
                name=area["name"],
                lat=area["lat"],
                lon=area["lon"],
                base_elevation=base_elev,
                summit_elevation=summit_elev,
                vertical_drop=vertical_drop,
                country=country,
                state=state,
                nearest_snotel=None,
            )
            resorts.append(resort)

        # Filter to Western US bounding box
        western_us_bbox = WESTERN_US_BBOX.to_dict()
        filtered_resorts = self.filter_region(resorts, bbox=western_us_bbox)

        return filtered_resorts

    def export_to_dataframe(self, resorts: list[SkiResort]) -> pd.DataFrame:
        """Convert list of SkiResort objects to DataFrame.

        Args:
            resorts: List of SkiResort objects.

        Returns:
            DataFrame with resort data, ready for export to parquet.
        """
        if not resorts:
            return pd.DataFrame(columns=[
                "name", "lat", "lon", "base_elevation", "summit_elevation",
                "vertical_drop", "country", "state", "nearest_snotel"
            ])

        records = [asdict(resort) for resort in resorts]
        return pd.DataFrame.from_records(records)

    def save_to_parquet(self, df: pd.DataFrame) -> Path:
        """Save resort DataFrame to parquet file.

        Args:
            df: DataFrame with resort data.

        Returns:
            Path to saved parquet file.
        """
        output_path = self.processed_path / "resorts.parquet"
        df.to_parquet(output_path, index=False)
        return output_path

    def validate(self, data: Any) -> ValidationResult:
        """Validate the processed resort data.

        Args:
            data: DataFrame of ski resorts.

        Returns:
            ValidationResult with quality metrics.
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["Data is not a DataFrame"],
            )

        issues = []
        total_rows = len(data)

        if total_rows == 0:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["No resorts found"],
            )

        # Check for required columns
        required_cols = ["name", "lat", "lon"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        # Calculate missing percentage for key fields
        name_missing = data["name"].isna().sum() if "name" in data.columns else total_rows
        lat_missing = data["lat"].isna().sum() if "lat" in data.columns else total_rows
        lon_missing = data["lon"].isna().sum() if "lon" in data.columns else total_rows
        elev_missing = (
            data["base_elevation"].isna().sum() + data["summit_elevation"].isna().sum()
        ) / 2 if "base_elevation" in data.columns else total_rows

        total_missing = name_missing + lat_missing + lon_missing
        missing_pct = (total_missing / (total_rows * 3)) * 100

        # Validate coordinate ranges
        if "lat" in data.columns and "lon" in data.columns:
            invalid_lat = ((data["lat"] < -90) | (data["lat"] > 90)).sum()
            invalid_lon = ((data["lon"] < -180) | (data["lon"] > 180)).sum()
            if invalid_lat > 0:
                issues.append(f"{invalid_lat} resorts have invalid latitude")
            if invalid_lon > 0:
                issues.append(f"{invalid_lon} resorts have invalid longitude")

        # Calculate elevation statistics
        elev_stats = {}
        if "base_elevation" in data.columns:
            valid_base = data["base_elevation"].dropna()
            if len(valid_base) > 0:
                elev_stats["base_elevation_min"] = valid_base.min()
                elev_stats["base_elevation_max"] = valid_base.max()
                elev_stats["base_elevation_mean"] = valid_base.mean()

        if "summit_elevation" in data.columns:
            valid_summit = data["summit_elevation"].dropna()
            if len(valid_summit) > 0:
                elev_stats["summit_elevation_min"] = valid_summit.min()
                elev_stats["summit_elevation_max"] = valid_summit.max()
                elev_stats["summit_elevation_mean"] = valid_summit.mean()

        # Count outliers (negative elevations or extreme values)
        outliers_count = 0
        if "base_elevation" in data.columns:
            outliers_count += ((data["base_elevation"] < 0) | (data["base_elevation"] > 5000)).sum()
        if "summit_elevation" in data.columns:
            outliers_count += ((data["summit_elevation"] < 0) | (data["summit_elevation"] > 6000)).sum()

        valid = len(issues) == 0 and total_rows > 0

        return ValidationResult(
            valid=valid,
            total_rows=total_rows,
            missing_pct=missing_pct,
            outliers_count=int(outliers_count),
            issues=issues,
            stats={
                "resorts_with_elevation": int(total_rows - elev_missing),
                **elev_stats,
            },
        )

    def run(
        self,
        raise_on_invalid: bool = True,
        save_parquet: bool = True,
        **kwargs
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Run the full pipeline: download, process, validate, and optionally save.

        Args:
            raise_on_invalid: If True, raise ValueError when validation fails.
            save_parquet: If True, save processed data to parquet file.
            **kwargs: Additional parameters passed to download().

        Returns:
            Tuple of (processed DataFrame, validation result).

        Raises:
            ValueError: If raise_on_invalid=True and validation fails.
        """
        raw_path = self.download(**kwargs)
        df = self.process(raw_path)
        validation = self.validate(df)

        if save_parquet and validation.valid:
            self.save_to_parquet(df)

        if raise_on_invalid and not validation.valid:
            raise ValueError(
                f"Data validation failed: {validation.issues}. "
                f"Missing: {validation.missing_pct:.1f}%, "
                f"Outliers: {validation.outliers_count}"
            )

        return df, validation
