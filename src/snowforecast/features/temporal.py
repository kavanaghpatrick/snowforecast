"""Temporal alignment and resampling utilities.

This module provides utilities for aligning data from different sources
to common temporal resolutions and timezones. Key functionality:

- Resample hourly data to daily with appropriate aggregations
- Handle timezone conversions to UTC
- Align data to common date ranges (filling missing dates with NaN)
- Create complete date indices

IMPORTANT: This module NEVER interpolates missing data. All gaps are
represented as NaN values.
"""

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


# Default aggregation methods for common meteorological variables
DEFAULT_AGGREGATIONS: dict[str, str | Callable] = {
    # Temperature: daily mean
    "temp": "mean",
    "temperature": "mean",
    "temp_avg": "mean",
    "temp_avg_c": "mean",
    "temp_max": "max",
    "temp_min": "min",
    "tmax": "max",
    "tmin": "min",
    # Precipitation: daily sum
    "precip": "sum",
    "precipitation": "sum",
    "prcp": "sum",
    # Snowfall: daily sum
    "snowfall": "sum",
    "snow": "sum",
    "snowfall_cm": "sum",
    # Snow depth: daily mean (could also use end-of-day/max)
    "snow_depth": "mean",
    "snow_depth_cm": "mean",
    "snwd": "mean",
    # SWE: daily mean
    "swe": "mean",
    "swe_mm": "mean",
    # Wind: daily mean magnitude
    "wind": "mean",
    "wind_speed": "mean",
    "wind_u": "mean",
    "wind_v": "mean",
    # Humidity: daily mean
    "humidity": "mean",
    "relative_humidity": "mean",
    # Pressure: daily mean
    "pressure": "mean",
    "surface_pressure": "mean",
}


class TemporalAligner:
    """Aligns data to common temporal resolution.

    This class provides methods to:
    - Resample hourly data to daily using appropriate aggregations
    - Convert timestamps to UTC
    - Align DataFrames to a common date range
    - Create complete date indices

    IMPORTANT: This class NEVER interpolates missing data. Gaps are always
    represented as NaN values.

    Example:
        >>> aligner = TemporalAligner(target_freq="1D", timezone="UTC")
        >>> # Resample hourly SNOTEL data to daily
        >>> daily_df = aligner.resample_hourly_to_daily(
        ...     hourly_df,
        ...     agg_methods={"temp_avg_c": "mean", "snow_depth_cm": "mean"}
        ... )
        >>> # Align to specific date range
        >>> aligned_df = aligner.align_to_date_range(daily_df, "2023-01-01", "2023-12-31")

    Attributes:
        target_freq: Target resampling frequency (e.g., "1D" for daily)
        timezone: Target timezone (default "UTC")
    """

    def __init__(self, target_freq: str = "1D", timezone: str = "UTC"):
        """Initialize with target frequency and timezone.

        Args:
            target_freq: Target resampling frequency. Common values:
                - "1D": Daily
                - "1h": Hourly
                - "1W": Weekly
            timezone: Target timezone for all output data. Default is "UTC".
        """
        self.target_freq = target_freq
        self.timezone = timezone

    def resample_hourly_to_daily(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        agg_methods: dict[str, str | Callable] | None = None,
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Resample hourly data to daily with appropriate aggregations.

        Different meteorological variables require different aggregation methods:
        - Temperature: mean (daily average)
        - Precipitation/Snowfall: sum (daily total)
        - Snow depth: mean or max
        - Wind: mean magnitude

        Args:
            df: DataFrame with hourly data. Must have a datetime column.
            datetime_col: Name of the datetime column. Default "datetime".
            agg_methods: Dictionary mapping column names to aggregation methods.
                Keys are column names, values are "mean", "sum", "max", "min",
                or callable functions. If None, uses DEFAULT_AGGREGATIONS.
            group_cols: Optional list of columns to group by before resampling
                (e.g., ["station_id"] to resample per station).

        Returns:
            DataFrame with daily data. Index is reset with date column.
            Missing days within the original range are NOT filled.

        Raises:
            ValueError: If datetime_col not found in DataFrame
            ValueError: If DataFrame is empty

        Example:
            >>> daily = aligner.resample_hourly_to_daily(
            ...     hourly_df,
            ...     agg_methods={"temp_c": "mean", "precip_mm": "sum"}
            ... )
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to resample_hourly_to_daily")
            return df.copy()

        if datetime_col not in df.columns and df.index.name != datetime_col:
            raise ValueError(
                f"datetime_col '{datetime_col}' not found in DataFrame columns: {list(df.columns)}"
            )

        df = df.copy()

        # Ensure datetime column is the index
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure index is timezone-aware (convert to UTC if needed)
        if df.index.tz is None:
            logger.warning("Datetime index is timezone-naive, assuming UTC")
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Build aggregation dictionary
        agg_dict = self._build_agg_dict(df.columns, agg_methods, group_cols)

        if not agg_dict:
            logger.warning("No numeric columns to aggregate")
            # Return structure with just grouping columns and date
            result = df.resample(self.target_freq).first()
            result = result.reset_index()
            result.rename(columns={result.columns[0]: "date"}, inplace=True)
            return result

        # Perform resampling
        if group_cols:
            # Group by specified columns, then resample
            resampled = (
                df.groupby(group_cols)
                .resample(self.target_freq)
                .agg(agg_dict)
            )
            # Flatten multi-index if created
            if isinstance(resampled.index, pd.MultiIndex):
                resampled = resampled.reset_index()
        else:
            resampled = df.resample(self.target_freq).agg(agg_dict)
            resampled = resampled.reset_index()

        # Rename datetime index to "date" for daily data
        if self.target_freq in ["1D", "D", "1d", "d"]:
            if datetime_col in resampled.columns:
                resampled = resampled.rename(columns={datetime_col: "date"})
            elif "index" in resampled.columns:
                resampled = resampled.rename(columns={"index": "date"})

        logger.info(
            f"Resampled {len(df)} hourly records to {len(resampled)} daily records"
        )

        return resampled

    def _build_agg_dict(
        self,
        columns: pd.Index,
        agg_methods: dict[str, str | Callable] | None,
        group_cols: list[str] | None,
    ) -> dict[str, str | Callable]:
        """Build aggregation dictionary for resampling.

        Args:
            columns: DataFrame columns
            agg_methods: User-provided aggregation methods
            group_cols: Columns to exclude from aggregation

        Returns:
            Dictionary mapping column names to aggregation methods
        """
        agg_dict = {}
        exclude_cols = set(group_cols or [])

        for col in columns:
            if col in exclude_cols:
                continue

            # Check user-provided methods first
            if agg_methods and col in agg_methods:
                agg_dict[col] = agg_methods[col]
                continue

            # Check default aggregations (case-insensitive)
            col_lower = col.lower()
            for pattern, method in DEFAULT_AGGREGATIONS.items():
                if pattern in col_lower:
                    agg_dict[col] = method
                    break
            else:
                # Default to mean for numeric columns
                agg_dict[col] = "mean"

        return agg_dict

    def align_to_date_range(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        date_col: str = "date",
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Ensure DataFrame covers full date range, filling missing dates with NaN.

        This method creates a complete date index from start_date to end_date
        and reindexes the DataFrame to include all dates. Missing dates are
        filled with NaN values - NO INTERPOLATION is performed.

        Args:
            df: DataFrame with date column
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            date_col: Name of the date column. Default "date".
            group_cols: Optional list of columns to group by when aligning
                (e.g., ["station_id"] to align per station). Each group will
                have the full date range.

        Returns:
            DataFrame with complete date range. Missing values are NaN.

        Raises:
            ValueError: If date_col not found in DataFrame

        Example:
            >>> # Align to full year, filling gaps with NaN
            >>> aligned = aligner.align_to_date_range(df, "2023-01-01", "2023-12-31")
        """
        if df.empty:
            # Return empty DataFrame with date index
            date_range = self.create_date_index(start_date, end_date)
            result = pd.DataFrame({date_col: date_range})
            return result

        if date_col not in df.columns:
            raise ValueError(
                f"date_col '{date_col}' not found in DataFrame columns: {list(df.columns)}"
            )

        df = df.copy()

        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Normalize to date (remove time component if present)
        if hasattr(df[date_col].dt, "normalize"):
            df[date_col] = df[date_col].dt.normalize()

        # Remove timezone info for date alignment (keep as dates only)
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_localize(None)

        # Create full date range
        full_dates = self.create_date_index(start_date, end_date)

        if group_cols:
            # Align each group separately
            aligned_dfs = []
            for group_vals, group_df in df.groupby(group_cols):
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)

                # Set date as index and reindex to full range
                group_aligned = group_df.set_index(date_col).reindex(full_dates)
                group_aligned = group_aligned.reset_index()
                group_aligned = group_aligned.rename(columns={"index": date_col})

                # Fill in group column values
                for col, val in zip(group_cols, group_vals):
                    group_aligned[col] = val

                aligned_dfs.append(group_aligned)

            if aligned_dfs:
                result = pd.concat(aligned_dfs, ignore_index=True)
            else:
                result = df
        else:
            # Simple case: no grouping
            df = df.set_index(date_col)

            # Handle duplicate dates by keeping first occurrence
            if df.index.duplicated().any():
                logger.warning(
                    f"Found {df.index.duplicated().sum()} duplicate dates, keeping first occurrence"
                )
                df = df[~df.index.duplicated(keep="first")]

            result = df.reindex(full_dates)
            result = result.reset_index()
            result = result.rename(columns={"index": date_col})

        # Log coverage statistics
        original_dates = len(df[date_col].unique()) if date_col in df.columns else len(df)
        total_dates = len(full_dates)
        coverage = (original_dates / total_dates * 100) if total_dates > 0 else 0

        logger.info(
            f"Aligned to date range {start_date} to {end_date}: "
            f"{original_dates}/{total_dates} dates ({coverage:.1f}% coverage)"
        )

        return result

    def localize_to_utc(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        source_tz: str | None = None,
    ) -> pd.DataFrame:
        """Convert timestamps to UTC.

        Handles three cases:
        1. Timezone-naive data with known source timezone: localize then convert
        2. Timezone-naive data with unknown source: assume UTC and localize
        3. Timezone-aware data: convert to UTC

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of the datetime column. Default "datetime".
            source_tz: Source timezone if known (e.g., "America/Denver", "US/Mountain").
                If None and data is timezone-naive, assumes UTC.

        Returns:
            DataFrame with datetime column in UTC

        Raises:
            ValueError: If datetime_col not found in DataFrame

        Example:
            >>> # Convert Mountain Time to UTC
            >>> utc_df = aligner.localize_to_utc(df, source_tz="America/Denver")
        """
        if df.empty:
            return df.copy()

        if datetime_col not in df.columns:
            raise ValueError(
                f"datetime_col '{datetime_col}' not found in DataFrame columns: {list(df.columns)}"
            )

        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Check if already timezone-aware
        is_tz_aware = df[datetime_col].dt.tz is not None

        if is_tz_aware:
            # Convert to UTC
            original_tz = str(df[datetime_col].dt.tz)
            df[datetime_col] = df[datetime_col].dt.tz_convert("UTC")
            logger.info(f"Converted {datetime_col} from {original_tz} to UTC")
        elif source_tz:
            # Localize to source timezone, then convert to UTC
            try:
                df[datetime_col] = df[datetime_col].dt.tz_localize(source_tz)
                df[datetime_col] = df[datetime_col].dt.tz_convert("UTC")
                logger.info(f"Localized {datetime_col} from {source_tz} to UTC")
            except Exception as e:
                # Handle ambiguous/nonexistent times during DST transitions
                logger.warning(
                    f"Error during timezone conversion: {e}. "
                    f"Using 'infer' for ambiguous times and 'shift_forward' for nonexistent."
                )
                df[datetime_col] = df[datetime_col].dt.tz_localize(
                    source_tz,
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
                df[datetime_col] = df[datetime_col].dt.tz_convert("UTC")
        else:
            # No source timezone provided - assume UTC
            logger.warning(
                "No source timezone provided for timezone-naive data. Assuming UTC."
            )
            df[datetime_col] = df[datetime_col].dt.tz_localize("UTC")

        return df

    def create_date_index(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DatetimeIndex:
        """Create complete daily date index.

        Creates a DatetimeIndex with daily frequency from start_date to end_date
        (inclusive).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DatetimeIndex with daily frequency

        Example:
            >>> dates = aligner.create_date_index("2023-01-01", "2023-01-31")
            >>> len(dates)
            31
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Create date range based on target frequency
        if self.target_freq in ["1D", "D", "1d", "d"]:
            date_index = pd.date_range(start=start, end=end, freq="D")
        else:
            date_index = pd.date_range(start=start, end=end, freq=self.target_freq)

        return date_index

    def merge_temporal_sources(
        self,
        dataframes: list[pd.DataFrame],
        date_col: str = "date",
        start_date: str | None = None,
        end_date: str | None = None,
        suffixes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Merge multiple temporal DataFrames on date.

        Utility method to merge data from different sources (e.g., SNOTEL, GHCN)
        that have been aligned to the same temporal resolution.

        Args:
            dataframes: List of DataFrames to merge
            date_col: Name of the date column to merge on
            start_date: Optional start date to filter merged result
            end_date: Optional end date to filter merged result
            suffixes: Optional list of suffixes for overlapping columns

        Returns:
            Merged DataFrame with all columns from input DataFrames

        Example:
            >>> merged = aligner.merge_temporal_sources(
            ...     [snotel_daily, ghcn_daily],
            ...     suffixes=["_snotel", "_ghcn"]
            ... )
        """
        if not dataframes:
            return pd.DataFrame()

        if len(dataframes) == 1:
            return dataframes[0].copy()

        # Start with first DataFrame
        result = dataframes[0].copy()

        # Merge remaining DataFrames
        for i, df in enumerate(dataframes[1:], start=1):
            suffix = suffixes[i] if suffixes and i < len(suffixes) else f"_{i}"
            prev_suffix = suffixes[i - 1] if suffixes and i - 1 < len(suffixes) else ""

            result = pd.merge(
                result,
                df,
                on=date_col,
                how="outer",
                suffixes=(prev_suffix, suffix),
            )

        # Sort by date
        result = result.sort_values(date_col).reset_index(drop=True)

        # Filter by date range if specified
        if start_date:
            result = result[result[date_col] >= start_date]
        if end_date:
            result = result[result[date_col] <= end_date]

        return result

    def get_temporal_coverage(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Calculate temporal coverage statistics for a DataFrame.

        Args:
            df: DataFrame with date column
            date_col: Name of the date column
            start_date: Optional start date for coverage calculation
            end_date: Optional end date for coverage calculation

        Returns:
            Dictionary with coverage statistics:
                - total_days: Total days in the range
                - present_days: Days with data
                - missing_days: Days without data
                - coverage_pct: Percentage of days with data
                - gaps: List of (gap_start, gap_end, gap_length) tuples
        """
        if df.empty:
            return {
                "total_days": 0,
                "present_days": 0,
                "missing_days": 0,
                "coverage_pct": 0.0,
                "gaps": [],
            }

        dates = pd.to_datetime(df[date_col]).dt.normalize()

        # Determine range
        if start_date is None:
            start_date = str(dates.min().date())
        if end_date is None:
            end_date = str(dates.max().date())

        full_range = self.create_date_index(start_date, end_date)
        total_days = len(full_range)

        # Count present dates
        present_dates = set(dates.dt.date)
        present_days = len(present_dates)
        missing_days = total_days - present_days

        # Find gaps
        gaps = []
        full_set = set(d.date() for d in full_range)
        missing_set = full_set - present_dates

        if missing_set:
            # Find contiguous gaps
            missing_sorted = sorted(missing_set)
            gap_start = missing_sorted[0]
            gap_prev = gap_start

            for d in missing_sorted[1:]:
                if (d - gap_prev).days > 1:
                    # End of gap
                    gap_length = (gap_prev - gap_start).days + 1
                    gaps.append((str(gap_start), str(gap_prev), gap_length))
                    gap_start = d
                gap_prev = d

            # Final gap
            gap_length = (gap_prev - gap_start).days + 1
            gaps.append((str(gap_start), str(gap_prev), gap_length))

        coverage_pct = (present_days / total_days * 100) if total_days > 0 else 0.0

        return {
            "total_days": total_days,
            "present_days": present_days,
            "missing_days": missing_days,
            "coverage_pct": coverage_pct,
            "gaps": gaps,
        }
