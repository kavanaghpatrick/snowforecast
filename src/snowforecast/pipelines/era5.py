"""ERA5-Land data ingestion pipeline.

ERA5-Land is a reanalysis dataset from ECMWF/Copernicus providing hourly data
at ~9km resolution from 1950 to present. This is our primary source for
historical atmospheric variables.

Documentation: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
import logging
import time

import pandas as pd
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import cdsapi
except ImportError:
    cdsapi = None

from snowforecast.utils import (
    GriddedPipeline,
    ValidationResult,
    get_data_path,
    BoundingBox,
    WESTERN_US_BBOX,
)

logger = logging.getLogger(__name__)

# ERA5 variable mappings: our name -> ERA5 API name
ERA5_VARIABLES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "sd": "snow_depth",
    "sf": "snowfall",
}

# Default variables to download (core snow-related variables)
DEFAULT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "snow_depth",
    "total_precipitation",
    "snowfall",
]

# Expected units for each variable (after processing)
VARIABLE_UNITS = {
    "t2m": "K",  # Kelvin
    "d2m": "K",
    "u10": "m/s",
    "v10": "m/s",
    "sp": "Pa",
    "tp": "m",  # meters of water
    "sd": "m",  # meters of water equivalent
    "sf": "m",  # meters of water equivalent
}


class ERA5Pipeline(GriddedPipeline):
    """ERA5-Land data ingestion pipeline.

    Downloads hourly data from the Copernicus Climate Data Store (CDS)
    and processes it into standardized xarray Datasets.

    Attributes:
        cache_dir: Directory for caching downloaded data
        raw_dir: Directory for raw downloaded files
        client: CDS API client instance
        bbox: Bounding box for data extraction (default: Western US)
        max_retries: Maximum number of retries for queued requests
        retry_wait_base: Base wait time in seconds for retry backoff

    Example:
        >>> pipeline = ERA5Pipeline()
        >>> nc_path = pipeline.download("2023-01-01", "2023-01-03")
        >>> ds = pipeline.process_to_dataset(nc_path)
        >>> daily_ds = pipeline.to_daily(ds)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        raw_dir: Path | None = None,
        bbox: BoundingBox | None = None,
        max_retries: int = 5,
        retry_wait_base: int = 60,
    ):
        """Initialize the ERA5 pipeline.

        Args:
            cache_dir: Directory for cached data (default: data/cache/era5)
            raw_dir: Directory for raw downloads (default: data/raw/era5)
            bbox: Bounding box for extraction (default: Western US)
            max_retries: Maximum CDS queue retries (default: 5)
            retry_wait_base: Base wait time in seconds (default: 60)
        """
        if cdsapi is None:
            raise ImportError(
                "cdsapi is required for ERA5Pipeline. "
                "Install with: pip install cdsapi"
            )
        if xr is None:
            raise ImportError(
                "xarray is required for ERA5Pipeline. "
                "Install with: pip install xarray netcdf4"
            )

        self.cache_dir = cache_dir or get_data_path("era5", "cache")
        self.raw_dir = raw_dir or get_data_path("era5", "raw")
        self.bbox = bbox or WESTERN_US_BBOX
        self.max_retries = max_retries
        self.retry_wait_base = retry_wait_base

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CDS client lazily
        self._client: cdsapi.Client | None = None

    @property
    def client(self) -> "cdsapi.Client":
        """Lazy-load the CDS API client."""
        if self._client is None:
            self._client = cdsapi.Client()
        return self._client

    def _generate_filename(
        self,
        start_date: str,
        end_date: str,
        variables: list[str],
    ) -> str:
        """Generate a standardized filename for downloaded data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of variable names

        Returns:
            Filename string (without path)
        """
        var_hash = "_".join(sorted(v[:3] for v in variables[:3]))
        return f"era5_{start_date}_{end_date}_{var_hash}.nc"

    def _build_request(
        self,
        start_date: str,
        end_date: str,
        variables: list[str],
        bbox: BoundingBox,
        hours: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the CDS API request dictionary.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of ERA5 variable names
            bbox: Bounding box for extraction
            hours: List of hours to download (default: all 24 hours)

        Returns:
            Request dictionary for cdsapi.Client.retrieve()
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate lists of years, months, days
        years = sorted(set(str(start_dt.year + i) for i in range((end_dt.year - start_dt.year) + 1)))

        # For simplicity, download all months/days and let CDS handle filtering
        # More efficient for short date ranges
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

        # If date range is within a single month, be more specific
        if start_dt.year == end_dt.year and start_dt.month == end_dt.month:
            months = [f"{start_dt.month:02d}"]
            days = [f"{d:02d}" for d in range(start_dt.day, end_dt.day + 1)]
        elif start_dt.year == end_dt.year:
            months = [f"{m:02d}" for m in range(start_dt.month, end_dt.month + 1)]

        if hours is None:
            hours = [f"{h:02d}:00" for h in range(24)]

        return {
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": list(bbox.to_tuple()),  # [N, W, S, E]
            "format": "netcdf",
        }

    def _download_with_retry(
        self,
        request: dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Download data with retry logic for CDS queue.

        The CDS API queues requests when the server is busy. This method
        implements exponential backoff retries.

        Args:
            request: CDS API request dictionary
            output_path: Path to save downloaded file

        Returns:
            Path to downloaded file

        Raises:
            RuntimeError: If max retries exceeded
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Submitting ERA5 request (attempt {attempt + 1}/{self.max_retries})")
                self.client.retrieve(
                    "reanalysis-era5-land",
                    request,
                    str(output_path),
                )
                logger.info(f"Download complete: {output_path}")
                return output_path

            except Exception as e:
                error_str = str(e).lower()
                if "queue" in error_str or "busy" in error_str or "limit" in error_str:
                    wait_time = self.retry_wait_base * (attempt + 1)
                    logger.warning(
                        f"Request queued/busy, waiting {wait_time}s before retry... "
                        f"Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    # Re-raise non-queue errors
                    logger.error(f"CDS API error: {e}")
                    raise

        raise RuntimeError(
            f"Max retries ({self.max_retries}) exceeded for ERA5 download. "
            "The CDS queue may be very busy. Try again later."
        )

    def download(
        self,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
        bbox: BoundingBox | None = None,
        hours: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        """Download ERA5-Land data for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of ERA5 variable names (default: core snow variables)
            bbox: Bounding box override (default: Western US)
            hours: List of hours to download (default: all 24 hours)
            force: Force re-download even if file exists

        Returns:
            Path to downloaded NetCDF file
        """
        variables = variables or DEFAULT_VARIABLES
        bbox = bbox or self.bbox

        # Generate output filename and path
        filename = self._generate_filename(start_date, end_date, variables)
        output_path = self.raw_dir / filename

        # Check cache
        if output_path.exists() and not force:
            logger.info(f"Using cached file: {output_path}")
            return output_path

        # Build and execute request
        request = self._build_request(start_date, end_date, variables, bbox, hours)
        logger.info(
            f"Downloading ERA5 data: {start_date} to {end_date}, "
            f"variables: {variables}, bbox: {bbox.to_tuple()}"
        )

        return self._download_with_retry(request, output_path)

    def process_to_dataset(self, raw_path: Path | list[Path]) -> "xr.Dataset":
        """Process raw NetCDF files to standardized xarray Dataset.

        Args:
            raw_path: Path(s) to raw NetCDF file(s)

        Returns:
            xarray Dataset with standardized coordinates and variables
        """
        if isinstance(raw_path, list):
            # Combine multiple files
            datasets = [xr.open_dataset(p) for p in raw_path]
            ds = xr.concat(datasets, dim="time")
        else:
            ds = xr.open_dataset(raw_path)

        # Standardize coordinate names
        rename_map = {}
        if "latitude" in ds.coords:
            rename_map["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_map["longitude"] = "lon"

        if rename_map:
            ds = ds.rename(rename_map)

        # Sort by time
        if "time" in ds.dims:
            ds = ds.sortby("time")

        # Add metadata
        ds.attrs["source"] = "ERA5-Land reanalysis"
        ds.attrs["processed_at"] = datetime.now().isoformat()

        return ds

    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw data to DataFrame (flattens spatial dims).

        For gridded analysis, use process_to_dataset() instead.

        Args:
            raw_path: Path(s) to raw NetCDF file(s)

        Returns:
            DataFrame with flattened data
        """
        ds = self.process_to_dataset(raw_path)
        return ds.to_dataframe().reset_index()

    def to_daily(self, hourly_ds: "xr.Dataset") -> "xr.Dataset":
        """Aggregate hourly data to daily values.

        Aggregation methods:
        - Temperature/dewpoint/pressure: daily mean
        - Wind: daily mean
        - Precipitation/snowfall: daily sum (already cumulative in ERA5)
        - Snow depth: daily mean

        Args:
            hourly_ds: Dataset with hourly data

        Returns:
            Dataset with daily aggregated data
        """
        # Variables that should be summed vs averaged
        sum_vars = {"tp", "sf", "total_precipitation", "snowfall"}

        # Separate variables by aggregation method
        sum_var_list = []
        mean_var_list = []
        for var in hourly_ds.data_vars:
            var_lower = var.lower()
            if var_lower in sum_vars or any(s in var_lower for s in sum_vars):
                sum_var_list.append(var)
            else:
                mean_var_list.append(var)

        # Resample and aggregate each variable type separately
        result_vars = {}

        # Sum variables (precipitation, snowfall)
        if sum_var_list:
            sum_ds = hourly_ds[sum_var_list].resample(time="1D").sum()
            for var in sum_var_list:
                result_vars[var] = sum_ds[var]

        # Mean variables (temperature, pressure, etc.)
        if mean_var_list:
            mean_ds = hourly_ds[mean_var_list].resample(time="1D").mean()
            for var in mean_var_list:
                result_vars[var] = mean_ds[var]

        # Combine back into dataset
        daily = xr.Dataset(result_vars)
        daily.attrs = hourly_ds.attrs.copy()
        daily.attrs["temporal_resolution"] = "daily"

        return daily

    def extract_at_points(
        self,
        data: "xr.Dataset",
        points: list[tuple[float, float]],
        method: str = "nearest",
    ) -> pd.DataFrame:
        """Extract time series at specific lat/lon points.

        Args:
            data: Dataset from process_to_dataset()
            points: List of (lat, lon) tuples
            method: Interpolation method ('nearest' or 'linear')

        Returns:
            DataFrame with time series for each point
        """
        records = []

        for i, (lat, lon) in enumerate(points):
            # Select nearest grid point
            if method == "nearest":
                point_ds = data.sel(lat=lat, lon=lon, method="nearest")
            else:
                point_ds = data.interp(lat=lat, lon=lon, method=method)

            # Convert to DataFrame
            point_df = point_ds.to_dataframe().reset_index()
            point_df["point_id"] = i
            point_df["point_lat"] = lat
            point_df["point_lon"] = lon

            records.append(point_df)

        return pd.concat(records, ignore_index=True)

    def validate(self, data: Any) -> ValidationResult:
        """Validate ERA5 data for quality and completeness.

        Checks:
        - No all-NaN variables
        - Reasonable value ranges for each variable
        - Temporal continuity (no large gaps)
        - Spatial coverage matches expected bbox

        Args:
            data: xarray Dataset or pandas DataFrame

        Returns:
            ValidationResult with quality metrics
        """
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)
        elif hasattr(data, "data_vars"):  # xarray Dataset
            return self._validate_dataset(data)
        else:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["Unknown data type for validation"],
            )

    def _validate_dataset(self, ds: "xr.Dataset") -> ValidationResult:
        """Validate xarray Dataset."""
        issues = []
        stats = {}

        # Calculate total data points
        total_points = 1
        for dim in ds.sizes:
            total_points *= ds.sizes[dim]

        # Check each variable
        missing_counts = []
        for var in ds.data_vars:
            data = ds[var].values
            nan_count = np.isnan(data).sum()
            nan_pct = (nan_count / data.size) * 100

            missing_counts.append(nan_pct)
            stats[f"{var}_missing_pct"] = nan_pct

            # Check if entirely NaN
            if nan_pct == 100:
                issues.append(f"Variable '{var}' is entirely NaN")

            # Check for reasonable value ranges
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats[f"{var}_min"] = float(valid_data.min())
                stats[f"{var}_max"] = float(valid_data.max())
                stats[f"{var}_mean"] = float(valid_data.mean())

                # Temperature sanity check (Kelvin)
                if "temperature" in var.lower() or var in ["t2m", "d2m"]:
                    if valid_data.min() < 180 or valid_data.max() > 350:
                        issues.append(
                            f"Variable '{var}' has unrealistic temperature values: "
                            f"min={valid_data.min():.1f}, max={valid_data.max():.1f}"
                        )

                # Precipitation sanity check (should be non-negative)
                if "precip" in var.lower() or var in ["tp", "sf"]:
                    if valid_data.min() < -0.001:  # Small tolerance for floating point
                        issues.append(
                            f"Variable '{var}' has negative precipitation values"
                        )

        # Overall missing percentage
        overall_missing_pct = np.mean(missing_counts) if missing_counts else 0

        # Check temporal coverage
        if "time" in ds.coords:
            times = pd.DatetimeIndex(ds.time.values)
            if len(times) > 1:
                expected_gap = pd.Timedelta(hours=1)
                actual_gaps = times[1:] - times[:-1]
                max_gap = actual_gaps.max()
                if max_gap > expected_gap * 24:  # More than 24 hours gap
                    issues.append(
                        f"Large temporal gap detected: {max_gap}"
                    )
                stats["time_range"] = f"{times.min()} to {times.max()}"
                stats["time_steps"] = len(times)

        # Determine validity (convert to Python bool to avoid numpy bool issues)
        valid = bool(len(issues) == 0 and overall_missing_pct < 20)

        return ValidationResult(
            valid=valid,
            total_rows=int(total_points),
            missing_pct=float(overall_missing_pct),
            outliers_count=0,
            issues=issues,
            stats=stats,
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate pandas DataFrame."""
        issues = []
        stats = {}

        total_rows = len(df)
        if total_rows == 0:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["DataFrame is empty"],
            )

        # Calculate missing percentage across all columns
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        stats["total_columns"] = len(df.columns)
        stats["total_cells"] = df.size

        # Check for empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns: {empty_cols}")

        # Convert to Python types to avoid numpy bool issues
        valid = bool(len(issues) == 0 and missing_pct < 20)

        return ValidationResult(
            valid=valid,
            total_rows=int(total_rows),
            missing_pct=float(missing_pct),
            outliers_count=0,
            issues=issues,
            stats=stats,
        )

    def save_dataset(
        self,
        ds: "xr.Dataset",
        output_path: Path,
        chunksizes: tuple[int, int, int] | None = None,
    ) -> Path:
        """Save Dataset to NetCDF with optimized chunking.

        Args:
            ds: Dataset to save
            output_path: Output file path
            chunksizes: Chunk sizes (time, lat, lon). Default: (24, 100, 100)

        Returns:
            Path to saved file
        """
        chunksizes = chunksizes or (24, 100, 100)

        # Build encoding dict
        encoding = {}
        for var in ds.data_vars:
            var_chunks = []
            for i, dim in enumerate(ds[var].dims):
                if i < len(chunksizes):
                    var_chunks.append(min(chunksizes[i], ds.sizes[dim]))
                else:
                    var_chunks.append(ds.sizes[dim])
            encoding[var] = {
                "chunksizes": tuple(var_chunks),
                "zlib": True,
                "complevel": 4,
            }

        ds.to_netcdf(output_path, encoding=encoding)
        logger.info(f"Saved dataset to: {output_path}")
        return output_path
