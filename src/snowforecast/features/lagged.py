"""Lagged and rolling window feature engineering."""

from typing import Any

import pandas as pd


class LaggedFeatures:
    """Engineer lagged and rolling window features.

    This class creates temporal features that capture historical patterns,
    trends, and variability in time-series data. All operations are grouped
    by station_id (or custom group column) to prevent data leakage between
    different stations.

    Features created:
    - Lagged values: Values from N days ago
    - Rolling means: N-day moving averages
    - Rolling std: N-day variability measures
    - Rolling min/max: N-day extremes
    - Trend features: Changes and percent changes over N days
    - Cumulative features: N-day cumulative sums

    Example:
        >>> lf = LaggedFeatures(default_lags=[1, 7], default_windows=[7, 14])
        >>> df = lf.create_lags(df, ["temperature"], group_col="station_id")
        >>> # Creates: temperature_lag_1d, temperature_lag_7d
    """

    def __init__(
        self,
        default_lags: list[int] | None = None,
        default_windows: list[int] | None = None,
    ):
        """Initialize with default lag and window sizes.

        Args:
            default_lags: Default lag periods in days. Defaults to [1, 2, 3, 7].
            default_windows: Default rolling window sizes in days.
                Defaults to [3, 7, 14].
        """
        self.default_lags = default_lags if default_lags is not None else [1, 2, 3, 7]
        self.default_windows = default_windows if default_windows is not None else [3, 7, 14]

    def _validate_dataframe(self, df: pd.DataFrame, variables: list[str]) -> None:
        """Validate that required variables exist in the DataFrame.

        Args:
            df: Input DataFrame
            variables: List of variable names to check

        Raises:
            ValueError: If any required variable is missing
        """
        missing = [var for var in variables if var not in df.columns]
        if missing:
            raise ValueError(f"Variables not found in DataFrame: {missing}")

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a datetime index or date column for proper ordering.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame sorted by date within groups
        """
        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()

        # Check for common date column names
        date_cols = ["date", "datetime", "timestamp", "time"]
        for col in date_cols:
            if col in df.columns:
                df = df.sort_values(col)
                return df

        # If no date column found, assume data is already sorted
        return df

    def compute_all(
        self,
        df: pd.DataFrame,
        variables: list[str],
        group_col: str = "station_id",
        lags: list[int] | None = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute all lagged and rolling features for specified variables.

        This is a convenience method that applies all feature engineering
        methods in sequence: lags, rolling means, rolling std, rolling min/max,
        trend features, and cumulative features.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to create features for
            group_col: Column to group by (default: "station_id")
            lags: Lag periods to use (default: self.default_lags)
            windows: Window sizes to use (default: self.default_windows)

        Returns:
            DataFrame with all original columns plus new feature columns
        """
        result = df.copy()

        result = self.create_lags(result, variables, lags, group_col)
        result = self.create_rolling_mean(result, variables, windows, group_col)
        result = self.create_rolling_std(result, variables, windows, group_col)
        result = self.create_rolling_min_max(result, variables, windows, group_col)
        result = self.create_trend_features(result, variables, lags, group_col)
        result = self.create_cumulative_features(result, variables, windows, group_col)

        return result

    def create_lags(
        self,
        df: pd.DataFrame,
        variables: list[str],
        lags: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create lagged features.

        For each variable and lag period, creates a new column with values
        from N days ago. For example, if temperature today is 5C and 7 days
        ago was 2C, then temp_lag_7d = 2.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to create lags for
            lags: Lag periods in days (default: self.default_lags)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new lag columns: {var}_lag_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        lags = lags if lags is not None else self.default_lags

        for var in variables:
            for lag in lags:
                col_name = f"{var}_lag_{lag}d"
                if group_col in result.columns:
                    result[col_name] = result.groupby(group_col)[var].shift(lag)
                else:
                    result[col_name] = result[var].shift(lag)

        return result

    def create_rolling_mean(
        self,
        df: pd.DataFrame,
        variables: list[str],
        windows: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create rolling mean features.

        For each variable and window size, creates a new column with the
        N-day rolling mean. This captures short-term averages.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to compute rolling means for
            windows: Window sizes in days (default: self.default_windows)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new columns: {var}_roll_mean_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        windows = windows if windows is not None else self.default_windows

        for var in variables:
            for window in windows:
                col_name = f"{var}_roll_mean_{window}d"
                if group_col in result.columns:
                    result[col_name] = result.groupby(group_col)[var].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                else:
                    result[col_name] = result[var].rolling(
                        window=window, min_periods=1
                    ).mean()

        return result

    def create_rolling_std(
        self,
        df: pd.DataFrame,
        variables: list[str],
        windows: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create rolling standard deviation (variability) features.

        For each variable and window size, creates a new column with the
        N-day rolling standard deviation. This captures volatility.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to compute rolling std for
            windows: Window sizes in days (default: self.default_windows)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new columns: {var}_roll_std_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        windows = windows if windows is not None else self.default_windows

        for var in variables:
            for window in windows:
                col_name = f"{var}_roll_std_{window}d"
                if group_col in result.columns:
                    result[col_name] = result.groupby(group_col)[var].transform(
                        lambda x: x.rolling(window=window, min_periods=2).std()
                    )
                else:
                    result[col_name] = result[var].rolling(
                        window=window, min_periods=2
                    ).std()

        return result

    def create_rolling_min_max(
        self,
        df: pd.DataFrame,
        variables: list[str],
        windows: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create rolling min and max features.

        For each variable and window size, creates two new columns with
        the N-day rolling minimum and maximum values.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to compute rolling min/max for
            windows: Window sizes in days (default: self.default_windows)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new columns: {var}_roll_min_{n}d, {var}_roll_max_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        windows = windows if windows is not None else self.default_windows

        for var in variables:
            for window in windows:
                min_col = f"{var}_roll_min_{window}d"
                max_col = f"{var}_roll_max_{window}d"
                if group_col in result.columns:
                    result[min_col] = result.groupby(group_col)[var].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                    result[max_col] = result.groupby(group_col)[var].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                else:
                    result[min_col] = result[var].rolling(
                        window=window, min_periods=1
                    ).min()
                    result[max_col] = result[var].rolling(
                        window=window, min_periods=1
                    ).max()

        return result

    def create_trend_features(
        self,
        df: pd.DataFrame,
        variables: list[str],
        periods: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create trend (change) features.

        For each variable and period, creates two new columns:
        - Absolute change: current value - value N days ago
        - Percent change: (current - previous) / previous * 100

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to compute trends for
            periods: Periods for trend calculation (default: self.default_lags)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new columns:
                {var}_change_{n}d, {var}_pct_change_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        periods = periods if periods is not None else self.default_lags

        for var in variables:
            for period in periods:
                change_col = f"{var}_change_{period}d"
                pct_change_col = f"{var}_pct_change_{period}d"

                if group_col in result.columns:
                    # Absolute change
                    lagged = result.groupby(group_col)[var].shift(period)
                    result[change_col] = result[var] - lagged
                    # Percent change (handle division by zero)
                    result[pct_change_col] = result[change_col] / lagged.replace(0, float("nan")) * 100
                else:
                    lagged = result[var].shift(period)
                    result[change_col] = result[var] - lagged
                    result[pct_change_col] = result[change_col] / lagged.replace(0, float("nan")) * 100

        return result

    def create_cumulative_features(
        self,
        df: pd.DataFrame,
        variables: list[str],
        windows: list[int] | None = None,
        group_col: str = "station_id",
    ) -> pd.DataFrame:
        """Create cumulative sum features.

        For each variable and window size, creates a new column with the
        N-day cumulative sum. This is particularly useful for precipitation
        to track total rainfall/snowfall over a period.

        Args:
            df: Input DataFrame with time-series data
            variables: List of column names to compute cumulative sums for
            windows: Window sizes in days (default: self.default_windows)
            group_col: Column to group by (default: "station_id")

        Returns:
            DataFrame with new columns: {var}_cumsum_{n}d

        Raises:
            ValueError: If variables are not found in DataFrame
        """
        self._validate_dataframe(df, variables)
        result = df.copy()
        result = self._ensure_datetime_index(result)

        windows = windows if windows is not None else self.default_windows

        for var in variables:
            for window in windows:
                col_name = f"{var}_cumsum_{window}d"
                if group_col in result.columns:
                    result[col_name] = result.groupby(group_col)[var].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum()
                    )
                else:
                    result[col_name] = result[var].rolling(
                        window=window, min_periods=1
                    ).sum()

        return result

    def get_feature_names(
        self,
        variables: list[str],
        lags: list[int] | None = None,
        windows: list[int] | None = None,
    ) -> dict[str, list[str]]:
        """Get all feature names that would be created for given variables.

        This is useful for understanding what columns will be added
        without actually computing the features.

        Args:
            variables: List of input variable names
            lags: Lag periods to use (default: self.default_lags)
            windows: Window sizes to use (default: self.default_windows)

        Returns:
            Dictionary mapping feature type to list of column names
        """
        lags = lags if lags is not None else self.default_lags
        windows = windows if windows is not None else self.default_windows

        feature_names: dict[str, list[str]] = {
            "lags": [],
            "rolling_mean": [],
            "rolling_std": [],
            "rolling_min": [],
            "rolling_max": [],
            "change": [],
            "pct_change": [],
            "cumsum": [],
        }

        for var in variables:
            for lag in lags:
                feature_names["lags"].append(f"{var}_lag_{lag}d")
                feature_names["change"].append(f"{var}_change_{lag}d")
                feature_names["pct_change"].append(f"{var}_pct_change_{lag}d")

            for window in windows:
                feature_names["rolling_mean"].append(f"{var}_roll_mean_{window}d")
                feature_names["rolling_std"].append(f"{var}_roll_std_{window}d")
                feature_names["rolling_min"].append(f"{var}_roll_min_{window}d")
                feature_names["rolling_max"].append(f"{var}_roll_max_{window}d")
                feature_names["cumsum"].append(f"{var}_cumsum_{window}d")

        return feature_names
