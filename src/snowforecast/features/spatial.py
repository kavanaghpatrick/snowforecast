"""Spatial alignment utilities for aligning gridded data to point locations.

This module provides the SpatialAligner class for extracting values from
xarray Datasets at specific lat/lon points using nearest-neighbor or
bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import xarray as xr

import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = None
    HAS_XARRAY = False


InterpolationMethod = Literal["nearest", "bilinear"]


@dataclass
class ExtractionResult:
    """Result of extracting values at points from a gridded dataset.

    Attributes:
        data: DataFrame with extracted values
        points_extracted: Number of points successfully extracted
        points_failed: Number of points that failed extraction
        variables: List of variables extracted
        method: Interpolation method used
    """

    data: pd.DataFrame
    points_extracted: int
    points_failed: int
    variables: list[str]
    method: str


class SpatialAligner:
    """Aligns gridded data to point locations.

    Extracts values from xarray Datasets at specific lat/lon points using
    nearest-neighbor or bilinear interpolation. Handles coordinate name
    variations (lat/lon vs latitude/longitude).

    Attributes:
        interpolation_method: Default interpolation method ("nearest" or "bilinear")

    Example:
        >>> aligner = SpatialAligner(interpolation_method="bilinear")
        >>> points = [(39.5, -105.5), (40.0, -106.0)]
        >>> df = aligner.extract_at_points(era5_ds, points)
    """

    # Coordinate name mappings for standardization
    LAT_NAMES = {"lat", "latitude", "Lat", "LAT", "Latitude", "y"}
    LON_NAMES = {"lon", "longitude", "Lon", "LON", "Longitude", "x"}

    def __init__(self, interpolation_method: InterpolationMethod = "nearest"):
        """Initialize the SpatialAligner.

        Args:
            interpolation_method: Default interpolation method.
                - "nearest": Nearest-neighbor (faster, discrete values)
                - "bilinear": Bilinear interpolation (smoother, continuous values)
        """
        if not HAS_XARRAY:
            raise ImportError(
                "xarray is required for SpatialAligner. "
                "Install with: pip install xarray"
            )

        if interpolation_method not in ("nearest", "bilinear"):
            raise ValueError(
                f"Invalid interpolation method: {interpolation_method}. "
                "Must be 'nearest' or 'bilinear'."
            )

        self.interpolation_method = interpolation_method

    def _find_coord_name(
        self, ds: "xr.Dataset", candidates: set[str]
    ) -> str | None:
        """Find the coordinate name in a dataset from a set of candidates.

        Args:
            ds: xarray Dataset to search
            candidates: Set of possible coordinate names

        Returns:
            The matching coordinate name, or None if not found
        """
        for name in candidates:
            if name in ds.coords:
                return name
        return None

    def _get_lat_lon_names(self, ds: "xr.Dataset") -> tuple[str, str]:
        """Get the latitude and longitude coordinate names from a dataset.

        Args:
            ds: xarray Dataset

        Returns:
            Tuple of (lat_name, lon_name)

        Raises:
            ValueError: If latitude or longitude coordinates cannot be found
        """
        lat_name = self._find_coord_name(ds, self.LAT_NAMES)
        lon_name = self._find_coord_name(ds, self.LON_NAMES)

        if lat_name is None:
            raise ValueError(
                f"Cannot find latitude coordinate. "
                f"Looked for: {self.LAT_NAMES}. "
                f"Available coords: {list(ds.coords.keys())}"
            )

        if lon_name is None:
            raise ValueError(
                f"Cannot find longitude coordinate. "
                f"Looked for: {self.LON_NAMES}. "
                f"Available coords: {list(ds.coords.keys())}"
            )

        return lat_name, lon_name

    def find_nearest_grid_cell(
        self, ds: "xr.Dataset", lat: float, lon: float
    ) -> tuple[int, int]:
        """Find the nearest grid cell indices for a point.

        Args:
            ds: xarray Dataset with lat/lon coordinates
            lat: Target latitude
            lon: Target longitude

        Returns:
            Tuple of (lat_index, lon_index) for the nearest grid cell

        Raises:
            ValueError: If coordinates cannot be found or point is out of bounds
        """
        lat_name, lon_name = self._get_lat_lon_names(ds)

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        # Find nearest indices
        lat_idx = int(np.abs(lats - lat).argmin())
        lon_idx = int(np.abs(lons - lon).argmin())

        return lat_idx, lon_idx

    def _extract_single_point(
        self,
        ds: "xr.Dataset",
        lat: float,
        lon: float,
        lat_name: str,
        lon_name: str,
        method: InterpolationMethod,
    ) -> "xr.Dataset":
        """Extract data at a single point from a dataset.

        Args:
            ds: Source dataset
            lat: Target latitude
            lon: Target longitude
            lat_name: Name of latitude coordinate
            lon_name: Name of longitude coordinate
            method: Interpolation method

        Returns:
            Dataset with values at the specified point
        """
        if method == "nearest":
            return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        else:  # bilinear
            return ds.interp({lat_name: lat, lon_name: lon}, method="linear")

    def extract_at_points(
        self,
        ds: "xr.Dataset",
        points: list[tuple[float, float]],
        method: InterpolationMethod | None = None,
        prefix: str = "",
    ) -> pd.DataFrame:
        """Extract values from a dataset at specific lat/lon points.

        Args:
            ds: xarray Dataset with gridded data
            points: List of (lat, lon) tuples
            method: Interpolation method (default: self.interpolation_method)
            prefix: Prefix to add to column names (e.g., "era5_")

        Returns:
            DataFrame with columns for each variable at each point.
            Includes point_id, point_lat, point_lon columns.

        Raises:
            ValueError: If dataset has no spatial coordinates
        """
        if method is None:
            method = self.interpolation_method

        # Handle empty points list
        if not points:
            return pd.DataFrame(columns=["point_id", "point_lat", "point_lon"])

        lat_name, lon_name = self._get_lat_lon_names(ds)

        records = []
        for i, (lat, lon) in enumerate(points):
            try:
                # Extract at single point
                point_ds = self._extract_single_point(
                    ds, lat, lon, lat_name, lon_name, method
                )

                # Convert to DataFrame
                point_df = point_ds.to_dataframe().reset_index()
                point_df["point_id"] = i
                point_df["point_lat"] = lat
                point_df["point_lon"] = lon

                records.append(point_df)
            except Exception:
                # Create empty record for failed extraction
                empty_record = {
                    "point_id": i,
                    "point_lat": lat,
                    "point_lon": lon,
                }
                # Add NaN for each data variable
                for var in ds.data_vars:
                    empty_record[var] = np.nan
                records.append(pd.DataFrame([empty_record]))

        result = pd.concat(records, ignore_index=True)

        # Add prefix to variable columns if specified
        if prefix:
            var_names = [str(v) for v in ds.data_vars]
            rename_map = {v: f"{prefix}{v}" for v in var_names if v in result.columns}
            result = result.rename(columns=rename_map)

        return result

    def extract_era5_at_points(
        self,
        era5_ds: "xr.Dataset",
        points: list[tuple[float, float]],
        method: InterpolationMethod | None = None,
    ) -> pd.DataFrame:
        """Extract ERA5 values at station/resort locations.

        Convenience method that adds "era5_" prefix to extracted columns.

        Args:
            era5_ds: ERA5 xarray Dataset
            points: List of (lat, lon) tuples

        Returns:
            DataFrame with era5-prefixed columns (e.g., era5_t2m, era5_tp)
        """
        return self.extract_at_points(era5_ds, points, method=method, prefix="era5_")

    def extract_hrrr_at_points(
        self,
        hrrr_ds: "xr.Dataset",
        points: list[tuple[float, float]],
        method: InterpolationMethod | None = None,
    ) -> pd.DataFrame:
        """Extract HRRR values at station/resort locations.

        Convenience method that adds "hrrr_" prefix to extracted columns.

        Args:
            hrrr_ds: HRRR xarray Dataset
            points: List of (lat, lon) tuples

        Returns:
            DataFrame with hrrr-prefixed columns (e.g., hrrr_tmp2m, hrrr_precip)
        """
        return self.extract_at_points(hrrr_ds, points, method=method, prefix="hrrr_")

    def extract_dem_at_points(
        self,
        dem_pipeline: Any,
        points: list[tuple[float, float]],
    ) -> pd.DataFrame:
        """Extract DEM terrain features at locations.

        Uses the DEMPipeline's get_terrain_features method to extract
        elevation, slope, aspect, and other terrain metrics.

        Args:
            dem_pipeline: DEMPipeline instance
            points: List of (lat, lon) tuples

        Returns:
            DataFrame with dem-prefixed columns:
            - dem_elevation: Elevation in meters
            - dem_slope: Slope in degrees
            - dem_aspect: Aspect in degrees (0=N, 90=E, 180=S, 270=W)
            - dem_aspect_sin: Sine of aspect (cyclical encoding)
            - dem_aspect_cos: Cosine of aspect (cyclical encoding)
            - dem_roughness: Terrain roughness index
            - dem_tpi: Topographic Position Index
        """
        records = []

        for i, (lat, lon) in enumerate(points):
            try:
                features = dem_pipeline.get_terrain_features(lat, lon)
                records.append({
                    "point_id": i,
                    "point_lat": lat,
                    "point_lon": lon,
                    "dem_elevation": features.elevation,
                    "dem_slope": features.slope,
                    "dem_aspect": features.aspect,
                    "dem_aspect_sin": features.aspect_sin,
                    "dem_aspect_cos": features.aspect_cos,
                    "dem_roughness": features.roughness,
                    "dem_tpi": features.tpi,
                })
            except (ValueError, Exception):
                # Point outside DEM coverage or other error
                records.append({
                    "point_id": i,
                    "point_lat": lat,
                    "point_lon": lon,
                    "dem_elevation": np.nan,
                    "dem_slope": np.nan,
                    "dem_aspect": np.nan,
                    "dem_aspect_sin": np.nan,
                    "dem_aspect_cos": np.nan,
                    "dem_roughness": np.nan,
                    "dem_tpi": np.nan,
                })

        return pd.DataFrame(records)

    def align_all_sources(
        self,
        points: list[tuple[float, float]],
        era5_ds: "xr.Dataset" | None = None,
        hrrr_ds: "xr.Dataset" | None = None,
        dem_pipeline: Any | None = None,
        method: InterpolationMethod | None = None,
    ) -> pd.DataFrame:
        """Align all data sources to common point locations.

        Extracts values from all provided data sources at the specified
        points and merges them into a single DataFrame.

        Args:
            points: List of (lat, lon) tuples for extraction
            era5_ds: Optional ERA5 xarray Dataset
            hrrr_ds: Optional HRRR xarray Dataset
            dem_pipeline: Optional DEMPipeline instance
            method: Interpolation method for gridded data

        Returns:
            DataFrame with all extracted features aligned to points.
            Columns are prefixed with source name (era5_, hrrr_, dem_).
        """
        # Start with point coordinates
        base_df = pd.DataFrame([
            {"point_id": i, "point_lat": lat, "point_lon": lon}
            for i, (lat, lon) in enumerate(points)
        ])

        # Extract from each source and merge
        if era5_ds is not None:
            era5_df = self.extract_era5_at_points(era5_ds, points, method=method)
            # Drop duplicate point columns before merge
            era5_df = era5_df.drop(
                columns=["point_lat", "point_lon"], errors="ignore"
            )
            base_df = base_df.merge(era5_df, on="point_id", how="left")

        if hrrr_ds is not None:
            hrrr_df = self.extract_hrrr_at_points(hrrr_ds, points, method=method)
            hrrr_df = hrrr_df.drop(
                columns=["point_lat", "point_lon"], errors="ignore"
            )
            base_df = base_df.merge(hrrr_df, on="point_id", how="left")

        if dem_pipeline is not None:
            dem_df = self.extract_dem_at_points(dem_pipeline, points)
            dem_df = dem_df.drop(
                columns=["point_lat", "point_lon"], errors="ignore"
            )
            base_df = base_df.merge(dem_df, on="point_id", how="left")

        return base_df

    def get_extraction_stats(
        self, ds: "xr.Dataset", points: list[tuple[float, float]]
    ) -> dict[str, Any]:
        """Get statistics about potential extraction from a dataset.

        Args:
            ds: xarray Dataset to analyze
            points: Points that would be extracted

        Returns:
            Dictionary with statistics about the extraction:
            - grid_resolution_lat: Approximate latitude grid spacing
            - grid_resolution_lon: Approximate longitude grid spacing
            - points_in_bounds: Number of points within grid bounds
            - points_out_of_bounds: Number of points outside grid bounds
            - variables: List of available variables
        """
        lat_name, lon_name = self._get_lat_lon_names(ds)

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        # Calculate grid resolution
        lat_res = float(np.abs(np.diff(lats)).mean()) if len(lats) > 1 else 0.0
        lon_res = float(np.abs(np.diff(lons)).mean()) if len(lons) > 1 else 0.0

        # Count points in bounds
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        in_bounds = sum(
            1 for lat, lon in points
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
        )
        out_of_bounds = len(points) - in_bounds

        return {
            "grid_resolution_lat": lat_res,
            "grid_resolution_lon": lon_res,
            "grid_lat_range": (lat_min, lat_max),
            "grid_lon_range": (lon_min, lon_max),
            "points_in_bounds": in_bounds,
            "points_out_of_bounds": out_of_bounds,
            "variables": list(ds.data_vars.keys()),
            "has_time_dim": "time" in ds.dims,
        }
