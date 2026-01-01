"""HRRR (High-Resolution Rapid Refresh) archive data ingestion pipeline.

HRRR is NOAA's high-resolution (3km) weather model, updated hourly.
The archive on AWS provides data from 2014 to present.

This pipeline downloads HRRR data using the herbie library.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import pandas as pd
import xarray as xr

from snowforecast.utils import (
    GriddedPipeline,
    ValidationResult,
    get_data_path,
    BoundingBox,
    WESTERN_US_BBOX,
)

logger = logging.getLogger(__name__)

# Default HRRR variables for snow forecasting
DEFAULT_VARIABLES = [
    "TMP:2 m above ground",  # 2m temperature
    "SNOD:surface",  # Snow depth (m)
    "WEASD:surface",  # Water equivalent of accumulated snow depth (kg/m^2)
    "PRATE:surface",  # Precipitation rate
    "CSNOW:surface",  # Categorical snow (is it snowing?)
]

# Extended variables including wind
EXTENDED_VARIABLES = DEFAULT_VARIABLES + [
    "UGRD:10 m above ground",  # U wind component
    "VGRD:10 m above ground",  # V wind component
]


class HRRRPipeline(GriddedPipeline):
    """HRRR archive data ingestion pipeline.

    Uses the herbie library to download HRRR data from AWS archive.
    Supports downloading analysis (f00) and forecast hours (f01-f48).

    Example:
        >>> pipeline = HRRRPipeline()
        >>> ds = pipeline.download_analysis("2023-01-15")
        >>> df = pipeline.extract_at_points(ds, [(40.0, -105.0), (39.5, -106.0)])
    """

    def __init__(
        self,
        product: str = "sfc",
        bbox: BoundingBox | None = None,
        max_workers: int = 4,
        save_format: str = "netcdf",
    ):
        """Initialize the HRRR pipeline.

        Args:
            product: HRRR product type ("sfc" for surface, "prs" for pressure levels)
            bbox: Bounding box to subset data. Defaults to Western US.
            max_workers: Maximum parallel download workers
            save_format: Output format ("netcdf" or "zarr")
        """
        self.product = product
        self.bbox = bbox or WESTERN_US_BBOX
        self.max_workers = max_workers
        self.save_format = save_format
        self._data_path = get_data_path("hrrr", "raw")

    def _get_herbie(self, date: str, fxx: int = 0) -> Any:
        """Create a Herbie object for the given date and forecast hour.

        Args:
            date: Date string in YYYY-MM-DD format
            fxx: Forecast hour (0 for analysis)

        Returns:
            Herbie object configured for HRRR
        """
        # Import here to allow the module to be imported without herbie installed
        from herbie import Herbie

        return Herbie(
            date,
            model="hrrr",
            product=self.product,
            fxx=fxx,
        )

    def _subset_kwargs(self) -> dict:
        """Get subset kwargs for herbie xarray call."""
        return {
            "lat": slice(self.bbox.south, self.bbox.north),
            "lon": slice(self.bbox.west, self.bbox.east),
        }

    def download_analysis(
        self,
        date: str,
        variables: list[str] | None = None,
        bbox: BoundingBox | None = None,
    ) -> xr.Dataset:
        """Download HRRR analysis (f00) for a specific date.

        Args:
            date: Date string in YYYY-MM-DD format
            variables: List of GRIB variable patterns. Defaults to DEFAULT_VARIABLES.
            bbox: Optional bounding box override

        Returns:
            xarray Dataset with requested variables
        """
        variables = variables or DEFAULT_VARIABLES
        use_bbox = bbox or self.bbox

        H = self._get_herbie(date, fxx=0)
        subset = {
            "lat": slice(use_bbox.south, use_bbox.north),
            "lon": slice(use_bbox.west, use_bbox.east),
        }

        # Download each variable and merge
        datasets = []
        for var in variables:
            try:
                ds = H.xarray(var, subset=subset)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to download {var} for {date}: {e}")

        if not datasets:
            raise ValueError(f"No variables downloaded for {date}")

        # Merge all variables into single dataset
        result = xr.merge(datasets)
        result.attrs["date"] = date
        result.attrs["product"] = self.product
        result.attrs["forecast_hour"] = 0

        return result

    def download_forecast(
        self,
        date: str,
        forecast_hours: list[int] | None = None,
        variables: list[str] | None = None,
    ) -> dict[int, xr.Dataset]:
        """Download HRRR forecasts for multiple lead times.

        Args:
            date: Date string in YYYY-MM-DD format
            forecast_hours: List of forecast hours (0-48). Defaults to [0, 6, 12, 18, 24].
            variables: List of GRIB variable patterns

        Returns:
            Dictionary mapping forecast hour to xarray Dataset
        """
        if forecast_hours is None:
            forecast_hours = [0, 6, 12, 18, 24]

        variables = variables or DEFAULT_VARIABLES
        results = {}

        for fxx in forecast_hours:
            try:
                H = self._get_herbie(date, fxx=fxx)
                subset = self._subset_kwargs()

                datasets = []
                for var in variables:
                    try:
                        ds = H.xarray(var, subset=subset)
                        datasets.append(ds)
                    except Exception as e:
                        logger.warning(f"Failed to download {var} for {date} f{fxx:02d}: {e}")

                if datasets:
                    merged = xr.merge(datasets)
                    merged.attrs["date"] = date
                    merged.attrs["forecast_hour"] = fxx
                    results[fxx] = merged

            except Exception as e:
                logger.warning(f"Failed to download forecast hour {fxx} for {date}: {e}")

        return results

    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
        parallel: bool = True,
        save: bool = True,
    ) -> list[Path]:
        """Download HRRR analysis data for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of GRIB variable patterns
            parallel: Whether to download in parallel
            save: Whether to save to disk

        Returns:
            List of paths to downloaded files
        """
        dates = pd.date_range(start_date, end_date)
        paths = []

        def download_single(date_ts: pd.Timestamp) -> Path | None:
            date_str = date_ts.strftime("%Y-%m-%d")
            try:
                ds = self.download_analysis(date_str, variables)
                if save:
                    path = self._save_dataset(ds, date_str, fxx=0)
                    return path
                return None
            except Exception as e:
                logger.error(f"Failed to download {date_str}: {e}")
                return None

        if parallel and len(dates) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(download_single, d): d for d in dates}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        paths.append(result)
        else:
            for date in dates:
                result = download_single(date)
                if result is not None:
                    paths.append(result)

        return sorted(paths)

    def _save_dataset(self, ds: xr.Dataset, date: str, fxx: int = 0) -> Path:
        """Save dataset to disk.

        Args:
            ds: Dataset to save
            date: Date string
            fxx: Forecast hour

        Returns:
            Path to saved file
        """
        # Parse date and create directory structure
        dt = datetime.strptime(date, "%Y-%m-%d")
        year_dir = self._data_path / str(dt.year)
        month_dir = year_dir / f"{dt.month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)

        filename = f"hrrr_{dt.strftime('%Y%m%d')}_f{fxx:02d}"

        if self.save_format == "zarr":
            path = month_dir / f"{filename}.zarr"
            ds.to_zarr(path, mode="w")
        else:
            path = month_dir / f"{filename}.nc"
            ds.to_netcdf(path)

        logger.info(f"Saved {path}")
        return path

    def extract_at_points(
        self,
        data: xr.Dataset,
        points: list[tuple[float, float]],
    ) -> pd.DataFrame:
        """Extract values at specific lat/lon points from a dataset.

        Args:
            data: xarray Dataset from download_analysis() or download_forecast()
            points: List of (lat, lon) tuples

        Returns:
            DataFrame with values for each point
        """
        records = []

        for lat, lon in points:
            # Select nearest grid point
            point_data = data.sel(latitude=lat, longitude=lon, method="nearest")

            record = {
                "lat": lat,
                "lon": lon,
                "actual_lat": float(point_data.latitude.values),
                "actual_lon": float(point_data.longitude.values),
            }

            # Add all data variables
            for var_name in data.data_vars:
                try:
                    value = float(point_data[var_name].values)
                    record[var_name] = value
                except (ValueError, TypeError):
                    # Handle multi-dimensional or missing data
                    record[var_name] = None

            # Add metadata
            record["date"] = data.attrs.get("date", "")
            record["forecast_hour"] = data.attrs.get("forecast_hour", 0)

            records.append(record)

        return pd.DataFrame(records)

    # Required abstract method implementations from GriddedPipeline

    def download(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Path | list[Path]:
        """Download raw HRRR data for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters (variables, parallel, etc.)

        Returns:
            List of paths to downloaded files
        """
        variables = kwargs.get("variables")
        parallel = kwargs.get("parallel", True)
        return self.download_date_range(start_date, end_date, variables, parallel)

    def process_to_dataset(self, raw_path: Path | list[Path]) -> xr.Dataset:
        """Load downloaded data as xarray Dataset.

        Args:
            raw_path: Path(s) to NetCDF or Zarr files

        Returns:
            xarray Dataset with all data
        """
        if isinstance(raw_path, Path):
            raw_path = [raw_path]

        datasets = []
        for path in raw_path:
            if path.suffix == ".zarr" or path.is_dir():
                ds = xr.open_zarr(path)
            else:
                ds = xr.open_dataset(path)
            datasets.append(ds)

        if len(datasets) == 1:
            return datasets[0]

        # Concatenate along time dimension
        return xr.concat(datasets, dim="time")

    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw data into DataFrame format.

        Args:
            raw_path: Path(s) to downloaded files

        Returns:
            DataFrame with flattened data
        """
        ds = self.process_to_dataset(raw_path)
        return ds.to_dataframe().reset_index()

    def validate(self, data: pd.DataFrame | xr.Dataset) -> ValidationResult:
        """Validate HRRR data quality.

        Args:
            data: DataFrame or Dataset to validate

        Returns:
            ValidationResult with quality metrics
        """
        issues = []
        stats = {}

        # Convert Dataset to DataFrame if needed
        if isinstance(data, xr.Dataset):
            df = data.to_dataframe().reset_index()
        else:
            df = data

        total_rows = len(df)

        if total_rows == 0:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["No data"],
            )

        # Calculate missing percentage across all columns
        missing_counts = df.isnull().sum()
        total_cells = df.size
        missing_cells = missing_counts.sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        # Check for data variables
        expected_vars = ["t2m", "sde", "sd", "unknown"]  # Common HRRR variable names
        has_data_vars = any(col in df.columns for col in df.columns if col not in ["latitude", "longitude", "time"])

        if not has_data_vars:
            issues.append("No data variables found")

        # Check temperature range if present (reasonable range check)
        temp_cols = [c for c in df.columns if "t2m" in c.lower() or "tmp" in c.lower()]
        for col in temp_cols:
            if col in df.columns:
                min_temp = df[col].min()
                max_temp = df[col].max()
                stats[f"{col}_min"] = min_temp
                stats[f"{col}_max"] = max_temp
                # Temperature in Kelvin should be reasonable (180K - 340K)
                if min_temp < 180 or max_temp > 340:
                    issues.append(f"Temperature values out of range: {min_temp:.1f}K - {max_temp:.1f}K")

        # Check for high missing percentage
        if missing_pct > 50:
            issues.append(f"High missing percentage: {missing_pct:.1f}%")

        valid = len(issues) == 0
        stats["columns"] = list(df.columns)

        return ValidationResult(
            valid=valid,
            total_rows=total_rows,
            missing_pct=missing_pct,
            outliers_count=0,
            issues=issues,
            stats=stats,
        )
