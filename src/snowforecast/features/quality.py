"""Data quality control and cleaning for snowforecast training data.

This module implements comprehensive data quality control following strict
data integrity rules:

CRITICAL RULES:
- NEVER estimate or interpolate missing data
- NEVER fill gaps with assumed values
- Only ADD quality flag columns - never modify source values
- Filter to valid records, don't try to fix bad data

All quality operations are additive (flagging) or subtractive (filtering).
No values are ever modified or imputed.
"""

import logging
from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityFlag(IntFlag):
    """Bit flags for data quality issues.

    These are additive flags that can be combined with bitwise OR.
    A value of 0 (VALID) means no quality issues detected.
    """

    VALID = 0
    MISSING = auto()  # 1: Value is missing/NaN
    BELOW_PHYSICAL_LIMIT = auto()  # 2: Below physically possible range
    ABOVE_PHYSICAL_LIMIT = auto()  # 4: Above physically possible range
    STATISTICAL_OUTLIER = auto()  # 8: Statistical outlier (IQR or z-score)
    TEMPORAL_INCONSISTENT = auto()  # 16: Impossible temporal change


# Default physical limits for common meteorological variables
DEFAULT_PHYSICAL_LIMITS: dict[str, tuple[float | None, float | None]] = {
    # Temperature limits (Celsius)
    "temperature": (-60.0, 50.0),
    "temp": (-60.0, 50.0),
    "air_temperature": (-60.0, 50.0),
    "t2m": (-60.0, 50.0),
    # Precipitation (mm) - must be non-negative
    "precipitation": (0.0, None),
    "precip": (0.0, None),
    "prcp": (0.0, None),
    "tp": (0.0, None),
    # Snow depth (cm) - must be non-negative
    "snow_depth": (0.0, None),
    "snwd": (0.0, None),
    "swe": (0.0, None),  # Snow water equivalent
    # Wind speed (m/s) - must be non-negative
    "wind_speed": (0.0, 100.0),
    "wind": (0.0, 100.0),
    "u10": (-100.0, 100.0),  # U-component can be negative
    "v10": (-100.0, 100.0),  # V-component can be negative
    # Humidity (%) - 0 to 100
    "humidity": (0.0, 100.0),
    "relative_humidity": (0.0, 100.0),
    "rh": (0.0, 100.0),
    # Pressure (hPa)
    "pressure": (300.0, 1100.0),
    "msl": (300.0, 1100.0),
    "sp": (300.0, 1100.0),
    # Elevation (meters)
    "elevation": (0.0, 5000.0),
    "altitude": (0.0, 5000.0),
    # Latitude/Longitude
    "latitude": (-90.0, 90.0),
    "lat": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "lon": (-180.0, 180.0),
}

# Maximum physically plausible hourly changes for temporal consistency
DEFAULT_MAX_HOURLY_CHANGE: dict[str, float] = {
    "temperature": 15.0,  # Max 15C change per hour
    "temp": 15.0,
    "air_temperature": 15.0,
    "t2m": 15.0,
    "snow_depth": 30.0,  # Max 30cm change per hour (extreme snowfall or melting)
    "snwd": 30.0,
    "swe": 10.0,  # Max 10cm SWE change per hour
    "pressure": 10.0,  # Max 10 hPa change per hour
    "msl": 10.0,
}


@dataclass
class QualityReport:
    """Quality assessment report for a dataset.

    Attributes:
        total_records: Total number of records in the dataset
        valid_records: Number of records with no quality issues
        missing_pct: Percentage of records with missing values (0-100)
        outliers_detected: Number of statistical outliers detected
        physical_violations: Number of records outside physical limits
        temporal_violations: Number of records with temporal inconsistencies
        issues: List of human-readable quality issue descriptions
        column_stats: Per-column quality statistics
    """

    total_records: int
    valid_records: int
    missing_pct: float
    outliers_detected: int = 0
    physical_violations: int = 0
    temporal_violations: int = 0
    issues: list[str] = field(default_factory=list)
    column_stats: dict[str, dict] = field(default_factory=dict)

    @property
    def valid_pct(self) -> float:
        """Percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return 100.0 * self.valid_records / self.total_records

    @property
    def is_acceptable(self) -> bool:
        """Whether the data quality is acceptable (>=80% valid)."""
        return self.valid_pct >= 80.0

    def __str__(self) -> str:
        status = "ACCEPTABLE" if self.is_acceptable else "POOR"
        return (
            f"QualityReport({status}, "
            f"valid={self.valid_records}/{self.total_records} ({self.valid_pct:.1f}%), "
            f"missing={self.missing_pct:.1f}%, "
            f"outliers={self.outliers_detected}, "
            f"physical_violations={self.physical_violations}, "
            f"temporal_violations={self.temporal_violations})"
        )


class DataQualityController:
    """Quality control for training data.

    This class provides methods to:
    1. Check values against physical limits
    2. Detect statistical outliers
    3. Check temporal consistency
    4. Generate quality reports
    5. Apply quality flags (without modifying data)
    6. Filter to valid records (without interpolation)

    CRITICAL: This class NEVER modifies source data values. It only:
    - Adds quality flag columns
    - Filters rows based on quality criteria
    """

    def __init__(
        self,
        physical_limits: dict[str, tuple[float | None, float | None]] | None = None,
        max_hourly_change: dict[str, float] | None = None,
    ):
        """Initialize the quality controller.

        Args:
            physical_limits: Dictionary mapping column names to (min, max) tuples.
                None means no limit on that side. Uses DEFAULT_PHYSICAL_LIMITS
                if not specified.
            max_hourly_change: Dictionary mapping column names to max allowed
                hourly change. Uses DEFAULT_MAX_HOURLY_CHANGE if not specified.
        """
        self.physical_limits = {**DEFAULT_PHYSICAL_LIMITS}
        if physical_limits:
            self.physical_limits.update(physical_limits)

        self.max_hourly_change = {**DEFAULT_MAX_HOURLY_CHANGE}
        if max_hourly_change:
            self.max_hourly_change.update(max_hourly_change)

    def check_physical_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag values outside physical limits.

        Adds a column '{col}_physical_flag' for each column that has defined
        physical limits. The flag is:
        - 0: Within limits
        - QualityFlag.BELOW_PHYSICAL_LIMIT: Below minimum
        - QualityFlag.ABOVE_PHYSICAL_LIMIT: Above maximum
        - QualityFlag.MISSING: Value is NaN

        Args:
            df: Input DataFrame (NOT modified)

        Returns:
            New DataFrame with added flag columns
        """
        result = df.copy()

        for col in df.columns:
            col_lower = col.lower()
            if col_lower not in self.physical_limits:
                continue

            min_val, max_val = self.physical_limits[col_lower]
            flag_col = f"{col}_physical_flag"

            # Start with VALID (0) for all rows
            flags = pd.Series(int(QualityFlag.VALID), index=df.index)

            # Flag missing values
            is_missing = df[col].isna()
            flags = flags.where(~is_missing, int(QualityFlag.MISSING))

            # Flag values below minimum
            if min_val is not None:
                below_min = df[col] < min_val
                flags = flags.where(
                    ~below_min, flags | int(QualityFlag.BELOW_PHYSICAL_LIMIT)
                )

            # Flag values above maximum
            if max_val is not None:
                above_max = df[col] > max_val
                flags = flags.where(
                    ~above_max, flags | int(QualityFlag.ABOVE_PHYSICAL_LIMIT)
                )

            result[flag_col] = flags.astype(int)

        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Detect statistical outliers using IQR or z-score method.

        Adds a column '{col}_outlier_flag' for each specified column.
        The flag is:
        - 0: Not an outlier
        - QualityFlag.STATISTICAL_OUTLIER: Detected as outlier
        - QualityFlag.MISSING: Value is NaN

        Args:
            df: Input DataFrame (NOT modified)
            columns: Columns to check for outliers. If None, checks all
                numeric columns.
            method: Detection method - "iqr" (Interquartile Range) or
                "zscore" (standard deviations from mean)
            threshold: For IQR method, multiplier for IQR range (default 3.0
                means values beyond Q1-3*IQR or Q3+3*IQR are outliers).
                For zscore method, number of standard deviations.

        Returns:
            New DataFrame with added outlier flag columns
        """
        result = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            flag_col = f"{col}_outlier_flag"
            series = df[col]

            # Start with VALID (0) for all rows
            flags = pd.Series(int(QualityFlag.VALID), index=df.index)

            # Flag missing values
            is_missing = series.isna()
            flags = flags.where(~is_missing, int(QualityFlag.MISSING))

            # Get non-missing values for outlier detection
            valid_values = series.dropna()
            if len(valid_values) < 3:
                # Not enough data to detect outliers
                result[flag_col] = flags.astype(int)
                continue

            if method == "iqr":
                q1 = valid_values.quantile(0.25)
                q3 = valid_values.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                is_outlier = (series < lower_bound) | (series > upper_bound)

            elif method == "zscore":
                mean = valid_values.mean()
                std = valid_values.std()

                if std == 0:
                    # All values are the same, no outliers
                    is_outlier = pd.Series(False, index=df.index)
                else:
                    z_scores = (series - mean).abs() / std
                    is_outlier = z_scores > threshold

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            # Apply outlier flag (only for non-missing values)
            is_outlier = is_outlier & ~is_missing
            flags = flags.where(
                ~is_outlier, flags | int(QualityFlag.STATISTICAL_OUTLIER)
            )

            result[flag_col] = flags.astype(int)

        return result

    def check_temporal_consistency(
        self,
        df: pd.DataFrame,
        time_column: str = "datetime",
        group_by: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Check for impossible temporal changes.

        Flags values where the change from the previous time step exceeds
        the maximum physically plausible hourly change rate.

        Args:
            df: Input DataFrame with a datetime column. Must be sorted by time.
            time_column: Name of the datetime column
            group_by: Column(s) to group by before checking consistency
                (e.g., station_id for multi-station data)

        Returns:
            New DataFrame with added temporal flag columns
        """
        result = df.copy()

        if time_column not in df.columns:
            logger.warning(f"Time column '{time_column}' not found, skipping temporal check")
            return result

        # Ensure datetime type
        try:
            times = pd.to_datetime(df[time_column])
        except Exception as e:
            logger.warning(f"Could not parse time column: {e}")
            return result

        for col in df.columns:
            col_lower = col.lower()
            if col_lower not in self.max_hourly_change:
                continue

            max_change = self.max_hourly_change[col_lower]
            flag_col = f"{col}_temporal_flag"

            # Start with VALID (0) for all rows
            flags = pd.Series(int(QualityFlag.VALID), index=df.index)

            # Flag missing values
            is_missing = df[col].isna()
            flags = flags.where(~is_missing, int(QualityFlag.MISSING))

            if group_by is not None:
                # Check consistency within each group
                groups = df.groupby(group_by, sort=False)
                for _, group_df in groups:
                    self._check_group_temporal(
                        group_df, col, times.loc[group_df.index],
                        max_change, flags
                    )
            else:
                self._check_group_temporal(df, col, times, max_change, flags)

            result[flag_col] = flags.astype(int)

        return result

    def _check_group_temporal(
        self,
        df: pd.DataFrame,
        col: str,
        times: pd.Series,
        max_change: float,
        flags: pd.Series,
    ) -> None:
        """Helper to check temporal consistency within a group."""
        if len(df) < 2:
            return

        # Calculate time differences in hours
        time_diff = times.diff().dt.total_seconds() / 3600.0

        # Calculate value differences
        value_diff = df[col].diff().abs()

        # Calculate allowed change based on time elapsed
        allowed_change = time_diff.abs() * max_change

        # Flag where change exceeds allowed (only where we have valid time diff)
        exceeds_limit = (value_diff > allowed_change) & (time_diff > 0)

        # Apply flag (modify in place)
        for idx in df.index[exceeds_limit]:
            if not pd.isna(df.loc[idx, col]):
                flags.loc[idx] = flags.loc[idx] | int(QualityFlag.TEMPORAL_INCONSISTENT)

    def apply_quality_flags(
        self,
        df: pd.DataFrame,
        time_column: str = "datetime",
        group_by: str | list[str] | None = None,
        outlier_method: Literal["iqr", "zscore"] = "iqr",
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Apply all quality flags to the DataFrame.

        This runs all quality checks and adds flag columns. The original
        data values are NEVER modified.

        A combined 'quality_flag' column is also added, which is the bitwise
        OR of all individual flags for numeric columns.

        Args:
            df: Input DataFrame (NOT modified)
            time_column: Name of the datetime column for temporal checks
            group_by: Column(s) to group by for temporal consistency
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection

        Returns:
            New DataFrame with added quality flag columns
        """
        # Apply physical limits check
        result = self.check_physical_limits(df)

        # Apply outlier detection
        result = self.detect_outliers(
            result, method=outlier_method, threshold=outlier_threshold
        )

        # Apply temporal consistency check
        result = self.check_temporal_consistency(
            result, time_column=time_column, group_by=group_by
        )

        # Create combined quality flag
        flag_columns = [c for c in result.columns if c.endswith("_flag")]
        if flag_columns:
            combined = result[flag_columns].max(axis=1).astype(int)
            result["quality_flag"] = combined
        else:
            result["quality_flag"] = int(QualityFlag.VALID)

        return result

    def generate_report(
        self,
        df: pd.DataFrame,
        time_column: str = "datetime",
        group_by: str | list[str] | None = None,
    ) -> QualityReport:
        """Generate a quality report for the dataset.

        Args:
            df: Input DataFrame (will apply quality flags if not present)
            time_column: Name of the datetime column
            group_by: Column(s) to group by for temporal consistency

        Returns:
            QualityReport with quality statistics
        """
        # Apply flags if not already present
        if "quality_flag" not in df.columns:
            df = self.apply_quality_flags(df, time_column=time_column, group_by=group_by)

        total_records = len(df)

        # Count records with no quality issues
        valid_records = (df["quality_flag"] == int(QualityFlag.VALID)).sum()

        # Count missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude flag columns
        data_cols = [c for c in numeric_cols if not c.endswith("_flag")]
        if data_cols:
            missing_count = df[data_cols].isna().any(axis=1).sum()
            missing_pct = 100.0 * missing_count / total_records if total_records > 0 else 0.0
        else:
            missing_pct = 0.0

        # Count outliers
        outlier_cols = [c for c in df.columns if c.endswith("_outlier_flag")]
        outliers_detected = 0
        for col in outlier_cols:
            outliers_detected += (
                (df[col] & int(QualityFlag.STATISTICAL_OUTLIER)) != 0
            ).sum()

        # Count physical violations
        physical_cols = [c for c in df.columns if c.endswith("_physical_flag")]
        physical_violations = 0
        for col in physical_cols:
            violations = (
                (df[col] & int(QualityFlag.BELOW_PHYSICAL_LIMIT)) != 0
            ) | (
                (df[col] & int(QualityFlag.ABOVE_PHYSICAL_LIMIT)) != 0
            )
            physical_violations += violations.sum()

        # Count temporal violations
        temporal_cols = [c for c in df.columns if c.endswith("_temporal_flag")]
        temporal_violations = 0
        for col in temporal_cols:
            temporal_violations += (
                (df[col] & int(QualityFlag.TEMPORAL_INCONSISTENT)) != 0
            ).sum()

        # Generate issues list
        issues = []
        if missing_pct > 0:
            issues.append(f"Missing data: {missing_pct:.1f}% of records")
        if outliers_detected > 0:
            issues.append(f"Statistical outliers: {outliers_detected} detected")
        if physical_violations > 0:
            issues.append(f"Physical limit violations: {physical_violations}")
        if temporal_violations > 0:
            issues.append(f"Temporal inconsistencies: {temporal_violations}")

        # Per-column statistics
        column_stats = {}
        for col in data_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    column_stats[col] = {
                        "count": len(col_data),
                        "missing": df[col].isna().sum(),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }

        return QualityReport(
            total_records=total_records,
            valid_records=int(valid_records),
            missing_pct=missing_pct,
            outliers_detected=outliers_detected,
            physical_violations=physical_violations,
            temporal_violations=temporal_violations,
            issues=issues,
            column_stats=column_stats,
        )

    def filter_to_valid(
        self,
        df: pd.DataFrame,
        min_completeness: float = 0.8,
        exclude_outliers: bool = True,
        exclude_physical_violations: bool = True,
        exclude_temporal_violations: bool = True,
        time_column: str = "datetime",
        group_by: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Filter to rows meeting quality threshold.

        CRITICAL: This method NEVER interpolates or estimates missing values.
        It only REMOVES rows that don't meet quality criteria.

        Args:
            df: Input DataFrame (NOT modified)
            min_completeness: Minimum fraction of non-null values required
                for a row to be included (0.0 to 1.0)
            exclude_outliers: If True, exclude rows flagged as outliers
            exclude_physical_violations: If True, exclude rows with physical
                limit violations
            exclude_temporal_violations: If True, exclude rows with temporal
                inconsistencies
            time_column: Name of the datetime column
            group_by: Column(s) to group by for temporal consistency

        Returns:
            Filtered DataFrame with valid records only (NO interpolation)
        """
        # Apply flags if not already present
        if "quality_flag" not in df.columns:
            df = self.apply_quality_flags(df, time_column=time_column, group_by=group_by)

        # Start with all rows
        mask = pd.Series(True, index=df.index)

        # Check completeness
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_cols = [c for c in numeric_cols if not c.endswith("_flag")]
        if data_cols and min_completeness > 0:
            completeness = df[data_cols].notna().mean(axis=1)
            mask = mask & (completeness >= min_completeness)

        # Exclude based on quality flags
        excluded_flags = int(QualityFlag.MISSING)  # Always exclude missing

        if exclude_outliers:
            excluded_flags |= int(QualityFlag.STATISTICAL_OUTLIER)

        if exclude_physical_violations:
            excluded_flags |= int(QualityFlag.BELOW_PHYSICAL_LIMIT)
            excluded_flags |= int(QualityFlag.ABOVE_PHYSICAL_LIMIT)

        if exclude_temporal_violations:
            excluded_flags |= int(QualityFlag.TEMPORAL_INCONSISTENT)

        # Check combined quality flag
        mask = mask & ((df["quality_flag"] & excluded_flags) == 0)

        result = df[mask].copy()

        # Log filtering statistics
        filtered_count = len(df) - len(result)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} rows ({100*filtered_count/len(df):.1f}%) "
                f"due to quality issues. Remaining: {len(result)} rows."
            )

        return result
