"""SNOTEL data ingestion pipeline.

SNOTEL (Snow Telemetry) is a network of 800+ automated stations in the Western US
mountains that measure snow depth, Snow Water Equivalent (SWE), and temperature.
This pipeline uses the metloom library to access SNOTEL data.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from snowforecast.utils import TemporalPipeline, ValidationResult
from snowforecast.utils.io import get_data_path

logger = logging.getLogger(__name__)


@dataclass
class StationMetadata:
    """Metadata for a SNOTEL station.

    Attributes:
        station_id: SNOTEL station ID (e.g., "1050:CO:SNTL")
        name: Station name
        lat: Latitude in degrees
        lon: Longitude in degrees
        elevation: Elevation in meters
        state: Two-letter state code
    """

    station_id: str
    name: str
    lat: float
    lon: float
    elevation: float
    state: str


class SnotelPipeline(TemporalPipeline):
    """SNOTEL data ingestion pipeline.

    Downloads snow observation data from SNOTEL stations using the metloom library,
    processes it into a standardized format, and validates data quality.

    Output schema (Parquet):
        - station_id: str - SNOTEL station ID
        - datetime: datetime64 - UTC timestamp
        - snow_depth_cm: float - Snow depth in centimeters
        - swe_mm: float - Snow Water Equivalent in millimeters
        - temp_avg_c: float - Average temperature in Celsius
        - quality_flag: str - Data quality flag

    Example:
        >>> pipeline = SnotelPipeline()
        >>> df, validation = pipeline.run("2023-01-01", "2023-01-31", states=["CO"])
        >>> print(df.head())
    """

    # Default variables to download
    DEFAULT_VARIABLES = ["SNOWDEPTH", "SWE", "TEMPAVG"]

    # Retry configuration for network timeouts
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds

    # Validation thresholds
    MAX_MISSING_PCT = 30.0  # Max acceptable missing data percentage
    MIN_ROWS = 1  # Minimum rows to be valid

    def __init__(self, raw_dir: Path | None = None, processed_dir: Path | None = None):
        """Initialize the SNOTEL pipeline.

        Args:
            raw_dir: Directory for raw data. Defaults to data/raw/snotel/
            processed_dir: Directory for processed data. Defaults to data/processed/snotel/
        """
        self.raw_dir = raw_dir or get_data_path("snotel", "raw")
        self.processed_dir = processed_dir or get_data_path("snotel", "processed")

    def _get_metloom_imports(self):
        """Lazily import metloom to avoid import errors when not installed."""
        try:
            from metloom.pointdata import SnotelPointData
            from metloom.variables import SnotelVariables

            return SnotelPointData, SnotelVariables
        except ImportError as e:
            raise ImportError(
                "metloom is required for the SNOTEL pipeline. "
                "Install with: pip install 'snowforecast[snotel]'"
            ) from e

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with retry logic for network errors.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.MAX_RETRIES} attempts failed: {e}")
        raise last_exception

    def get_station_metadata(self, state: str | None = None) -> list[StationMetadata]:
        """Get metadata for all SNOTEL stations.

        Args:
            state: Optional two-letter state code to filter stations

        Returns:
            List of StationMetadata objects

        Example:
            >>> pipeline = SnotelPipeline()
            >>> stations = pipeline.get_station_metadata(state="CO")
            >>> print(f"Found {len(stations)} stations in Colorado")
        """
        SnotelPointData, SnotelVariables = self._get_metloom_imports()

        # Create Western US bounding box for station query
        try:
            import geopandas as gpd
            from shapely.geometry import box

            # Western US: -125 to -102 lon, 31 to 49 lat
            west_us_bbox = gpd.GeoDataFrame(
                geometry=[box(-125, 31, -102, 49)], crs="EPSG:4326"
            )
        except ImportError:
            west_us_bbox = None

        # Get all SNOTEL points (API requires variables list)
        variables = [SnotelVariables.SWE]
        all_points = self._retry_with_backoff(
            SnotelPointData.points_from_geometry, west_us_bbox, variables
        )

        stations = []
        for point in all_points:
            # Parse state from station ID (format: "1050:CO:SNTL")
            parts = point.id.split(":")
            station_state = parts[1] if len(parts) > 1 else "UNKNOWN"

            # Filter by state if specified
            if state is not None and station_state != state:
                continue

            stations.append(
                StationMetadata(
                    station_id=point.id,
                    name=point.name,
                    lat=point.y,
                    lon=point.x,
                    elevation=getattr(point, "elevation", 0.0) or 0.0,
                    state=station_state,
                )
            )

        logger.info(f"Found {len(stations)} SNOTEL stations" + (f" in {state}" if state else ""))
        return stations

    def _parse_date(self, date_str: str) -> datetime:
        """Parse a date string to datetime.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            datetime object
        """
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _get_variable_mapping(self):
        """Get mapping from variable names to metloom variables."""
        _, SnotelVariables = self._get_metloom_imports()
        return {
            "SNOWDEPTH": SnotelVariables.SNOWDEPTH,
            "SWE": SnotelVariables.SWE,
            "TEMPAVG": SnotelVariables.TEMPAVG,
        }

    def download_station(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
    ) -> Path:
        """Download data for a single SNOTEL station.

        Args:
            station_id: SNOTEL station ID (e.g., "1050:CO:SNTL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to download. Defaults to SNOWDEPTH, SWE, TEMPAVG

        Returns:
            Path to the downloaded Parquet file

        Raises:
            ValueError: If station_id is invalid
            Exception: If download fails after retries
        """
        SnotelPointData, _ = self._get_metloom_imports()
        var_mapping = self._get_variable_mapping()

        variables = variables or self.DEFAULT_VARIABLES
        metloom_vars = [var_mapping[v] for v in variables if v in var_mapping]

        # Parse dates
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)

        # Create station point
        station = SnotelPointData(station_id, station_id.split(":")[0])

        # Download data with retry
        logger.info(f"Downloading data for station {station_id} ({start_date} to {end_date})")
        df = self._retry_with_backoff(
            station.get_daily_data, start_dt, end_dt, metloom_vars
        )

        if df is None or df.empty:
            logger.warning(f"No data returned for station {station_id}")
            df = pd.DataFrame()

        # Save to Parquet
        output_file = self.raw_dir / f"{station_id.replace(':', '_')}_{start_date}_{end_date}.parquet"
        df.to_parquet(output_file)
        logger.info(f"Saved raw data to {output_file}")

        return output_file

    def download_all_stations(
        self,
        start_date: str,
        end_date: str,
        states: list[str] | None = None,
    ) -> list[Path]:
        """Download data for all SNOTEL stations (optionally filtered by state).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            states: Optional list of state codes to filter stations

        Returns:
            List of paths to downloaded Parquet files
        """
        downloaded_paths = []

        # Get stations for each state (or all if states is None)
        if states:
            stations = []
            for state in states:
                stations.extend(self.get_station_metadata(state=state))
        else:
            stations = self.get_station_metadata()

        logger.info(f"Downloading data for {len(stations)} stations")

        for i, station in enumerate(stations):
            try:
                path = self.download_station(
                    station.station_id, start_date, end_date
                )
                downloaded_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to download station {station.station_id}: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(stations)} stations downloaded")

        logger.info(f"Downloaded {len(downloaded_paths)} of {len(stations)} stations")
        return downloaded_paths

    def download(
        self,
        start_date: str,
        end_date: str,
        states: list[str] | None = None,
        station_ids: list[str] | None = None,
        **kwargs,
    ) -> Path | list[Path]:
        """Download SNOTEL data for the specified date range.

        This is the main download method that satisfies the TemporalPipeline interface.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            states: Optional list of state codes to filter stations
            station_ids: Optional list of specific station IDs to download
            **kwargs: Additional arguments (ignored)

        Returns:
            List of paths to downloaded files
        """
        if station_ids:
            paths = []
            for sid in station_ids:
                try:
                    path = self.download_station(sid, start_date, end_date)
                    paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to download station {sid}: {e}")
            return paths
        else:
            return self.download_all_stations(start_date, end_date, states=states)

    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw SNOTEL data into standardized format.

        Args:
            raw_path: Path or list of paths to raw Parquet files

        Returns:
            DataFrame with standardized columns:
                - station_id: str
                - datetime: datetime64
                - snow_depth_cm: float
                - swe_mm: float
                - temp_avg_c: float
                - quality_flag: str
        """
        if isinstance(raw_path, Path):
            raw_paths = [raw_path]
        else:
            raw_paths = raw_path

        all_dfs = []
        for path in raw_paths:
            if not path.exists():
                logger.warning(f"Raw file not found: {path}")
                continue

            df = pd.read_parquet(path)
            if df.empty:
                continue

            # Extract station ID from filename
            filename = path.stem
            station_id = filename.split("_")[0].replace("_", ":")
            # Fix: the station_id parsing was wrong, let's extract it properly
            # Filename format: "1050_CO_SNTL_2023-01-01_2023-01-31"
            parts = filename.split("_")
            if len(parts) >= 3:
                station_id = f"{parts[0]}:{parts[1]}:{parts[2]}"

            processed_df = self._process_single_file(df, station_id)
            if not processed_df.empty:
                all_dfs.append(processed_df)

        if not all_dfs:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=["station_id", "datetime", "snow_depth_cm", "swe_mm", "temp_avg_c", "quality_flag"]
            )

        result = pd.concat(all_dfs, ignore_index=True)

        # Save processed data
        output_file = self.processed_dir / f"snotel_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        result.to_parquet(output_file)
        logger.info(f"Saved processed data to {output_file} ({len(result)} rows)")

        return result

    def _process_single_file(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Process a single raw DataFrame.

        Args:
            df: Raw DataFrame from metloom
            station_id: SNOTEL station ID

        Returns:
            Processed DataFrame with standardized columns
        """
        if df.empty:
            return pd.DataFrame()

        # metloom returns data with specific column names
        # We need to map them to our standard schema
        result = pd.DataFrame()

        # Handle datetime index
        if df.index.name == "datetime" or "datetime" in str(type(df.index)):
            result["datetime"] = df.index
        elif "datetime" in df.columns:
            result["datetime"] = df["datetime"]
        else:
            # Try to use the index as datetime
            result["datetime"] = pd.to_datetime(df.index)

        result["station_id"] = station_id

        # Map metloom column names to standard names
        # metloom uses format like "SNOWDEPTH_units" or just the variable name
        col_mappings = {
            "snow_depth_cm": ["SNOWDEPTH", "snow depth", "snowdepth"],
            "swe_mm": ["SWE", "swe", "snow water equivalent"],
            "temp_avg_c": ["TEMPAVG", "tempavg", "air temp avg", "air temperature average"],
        }

        df_cols_lower = {c.lower(): c for c in df.columns}

        for target_col, source_patterns in col_mappings.items():
            result[target_col] = None
            for pattern in source_patterns:
                # Check exact match (case-insensitive)
                if pattern.lower() in df_cols_lower:
                    result[target_col] = df[df_cols_lower[pattern.lower()]].values
                    break
                # Check if pattern is contained in any column name
                for col_lower, col_orig in df_cols_lower.items():
                    if pattern.lower() in col_lower:
                        result[target_col] = df[col_orig].values
                        break
                if result[target_col].any() if hasattr(result[target_col], 'any') else result[target_col] is not None:
                    break

        # Convert units if necessary
        result = self._convert_units(result)

        # Add quality flag based on data completeness
        result["quality_flag"] = result.apply(self._compute_quality_flag, axis=1)

        # Ensure datetime is in UTC
        result["datetime"] = pd.to_datetime(result["datetime"], utc=True)

        # Reorder columns
        return result[["station_id", "datetime", "snow_depth_cm", "swe_mm", "temp_avg_c", "quality_flag"]]

    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert units to standard metric.

        SNOTEL data may come in different units depending on the source.
        This method standardizes to:
            - snow_depth_cm: centimeters
            - swe_mm: millimeters
            - temp_avg_c: Celsius

        Args:
            df: DataFrame with raw values

        Returns:
            DataFrame with converted values
        """
        # metloom typically returns data in standard units already
        # Snow depth: inches -> cm (multiply by 2.54)
        # SWE: inches -> mm (multiply by 25.4)
        # Temperature: already in Celsius from most sources

        # Check if values seem to be in inches (typical range check)
        if "snow_depth_cm" in df.columns and df["snow_depth_cm"].notna().any():
            max_depth = df["snow_depth_cm"].max()
            # If max depth is less than 200, it's probably in inches
            if max_depth < 200 and max_depth > 0:
                df["snow_depth_cm"] = df["snow_depth_cm"] * 2.54

        if "swe_mm" in df.columns and df["swe_mm"].notna().any():
            max_swe = df["swe_mm"].max()
            # If max SWE is less than 80, it's probably in inches
            if max_swe < 80 and max_swe > 0:
                df["swe_mm"] = df["swe_mm"] * 25.4

        # Temperature conversion: Check if values seem to be Fahrenheit
        if "temp_avg_c" in df.columns and df["temp_avg_c"].notna().any():
            mean_temp = df["temp_avg_c"].mean()
            # If mean temp is above 50, it's probably Fahrenheit
            if mean_temp > 50:
                df["temp_avg_c"] = (df["temp_avg_c"] - 32) * 5 / 9

        return df

    def _compute_quality_flag(self, row: pd.Series) -> str:
        """Compute quality flag for a data row.

        Args:
            row: DataFrame row

        Returns:
            Quality flag string: 'good', 'partial', 'missing'
        """
        # Count non-null data values (excluding station_id, datetime, quality_flag)
        data_cols = ["snow_depth_cm", "swe_mm", "temp_avg_c"]
        non_null = sum(1 for col in data_cols if pd.notna(row.get(col)))

        if non_null == len(data_cols):
            return "good"
        elif non_null > 0:
            return "partial"
        else:
            return "missing"

    def _get_date_range(self, df: pd.DataFrame) -> tuple:
        """Get date range tuple from dataframe."""
        if "datetime" not in df.columns or df["datetime"].isna().all():
            return (None, None)
        return (
            df["datetime"].min().isoformat(),
            df["datetime"].max().isoformat(),
        )

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate processed SNOTEL data.

        Args:
            df: Processed DataFrame

        Returns:
            ValidationResult with quality metrics
        """
        if df.empty:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                outliers_count=0,
                issues=["No data available"],
                stats={},
            )

        total_rows = len(df)
        issues = []

        # Check required columns
        required_cols = ["station_id", "datetime", "snow_depth_cm", "swe_mm", "temp_avg_c", "quality_flag"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Calculate missing percentages
        data_cols = ["snow_depth_cm", "swe_mm", "temp_avg_c"]
        missing_counts = {col: df[col].isna().sum() for col in data_cols if col in df.columns}
        total_cells = len(df) * len(data_cols)
        missing_cells = sum(missing_counts.values())
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 100.0

        if missing_pct > self.MAX_MISSING_PCT:
            issues.append(f"High missing data: {missing_pct:.1f}% (threshold: {self.MAX_MISSING_PCT}%)")

        # Detect outliers
        outliers_count = 0

        # Snow depth outliers (negative or unreasonably high)
        if "snow_depth_cm" in df.columns:
            invalid_depth = (df["snow_depth_cm"] < 0) | (df["snow_depth_cm"] > 2000)
            outliers_count += invalid_depth.sum()
            if invalid_depth.any():
                issues.append(f"Invalid snow depth values: {invalid_depth.sum()}")

        # SWE outliers
        if "swe_mm" in df.columns:
            invalid_swe = (df["swe_mm"] < 0) | (df["swe_mm"] > 5000)
            outliers_count += invalid_swe.sum()
            if invalid_swe.any():
                issues.append(f"Invalid SWE values: {invalid_swe.sum()}")

        # Temperature outliers (extremely cold or hot)
        if "temp_avg_c" in df.columns:
            invalid_temp = (df["temp_avg_c"] < -60) | (df["temp_avg_c"] > 50)
            outliers_count += invalid_temp.sum()
            if invalid_temp.any():
                issues.append(f"Invalid temperature values: {invalid_temp.sum()}")

        # Compute summary statistics
        stats = {
            "stations": df["station_id"].nunique() if "station_id" in df.columns else 0,
            "date_range": self._get_date_range(df),
            "missing_by_column": missing_counts,
            "quality_distribution": df["quality_flag"].value_counts().to_dict() if "quality_flag" in df.columns else {},
        }

        # Determine overall validity
        valid = (
            len(missing_cols) == 0
            and total_rows >= self.MIN_ROWS
            and missing_pct <= self.MAX_MISSING_PCT
            and outliers_count == 0
        )

        # Convert numpy types to Python native types for ValidationResult
        return ValidationResult(
            valid=bool(valid),
            total_rows=int(total_rows),
            missing_pct=float(missing_pct),
            outliers_count=int(outliers_count),
            issues=issues,
            stats=stats,
        )
