"""Temporal and cyclical feature engineering for snow forecasting.

This module provides features that capture seasonal and daily patterns
important for predicting snowfall events in mountain environments.
"""

import numpy as np
import pandas as pd


class TemporalFeatures:
    """Engineer temporal and cyclical features from datetime data.

    This class generates features that capture:
    - Cyclical patterns (day of year, month) using sin/cos encoding
    - Seasonal indicators (winter, snow season)
    - Week-based features (day of week, weekend)
    - Hydrological water year (Oct 1 - Sep 30)

    Sin/cos encoding is preferred over one-hot encoding for cyclical features
    because it preserves the continuous, circular nature of time. For example,
    December 31 and January 1 are encoded as nearby points in sin/cos space.

    Example:
        >>> tf = TemporalFeatures()
        >>> df = pd.DataFrame({"datetime": pd.date_range("2023-01-01", periods=365)})
        >>> df_features = tf.compute_all(df)
        >>> df_features.columns.tolist()[:5]
        ['datetime', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos', 'month']
    """

    def __init__(self, datetime_col: str = "datetime"):
        """Initialize TemporalFeatures.

        Args:
            datetime_col: Name of the datetime column in input DataFrames.
                         Defaults to "datetime".
        """
        self.datetime_col = datetime_col

    def compute_all(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute all temporal features.

        This method chains all individual feature computations and returns
        a DataFrame with all temporal features added.

        Args:
            df: Input DataFrame containing a datetime column.
            datetime_col: Override the datetime column name for this call.
                         If None, uses the instance's datetime_col.

        Returns:
            DataFrame with all temporal features added as new columns:
            - day_of_year, day_of_year_sin, day_of_year_cos
            - month, month_sin, month_cos
            - season, is_winter, is_snow_season
            - day_of_week, is_weekend, week_of_year
            - water_year, water_year_day, water_year_progress

        Raises:
            KeyError: If the datetime column is not found in the DataFrame.
            ValueError: If the datetime column cannot be converted to datetime.
        """
        col = datetime_col or self.datetime_col

        # Validate datetime column exists
        if col not in df.columns:
            raise KeyError(f"Datetime column '{col}' not found in DataFrame. "
                          f"Available columns: {list(df.columns)}")

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Ensure datetime column is datetime type
        result = self._ensure_datetime(result, col)

        # Compute all features
        result = self.compute_cyclical_day_of_year(result, col)
        result = self.compute_cyclical_month(result, col)
        result = self.compute_season(result, col)
        result = self.compute_week_features(result, col)
        result = self.compute_water_year(result, col)

        return result

    def compute_cyclical_day_of_year(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute day of year features with cyclical encoding.

        Features:
        - day_of_year: Raw day of year (1-365/366)
        - day_of_year_sin: sin(2*pi * day/365.25)
        - day_of_year_cos: cos(2*pi * day/365.25)

        The sin/cos encoding ensures that December 31 (day 365) and
        January 1 (day 1) are close in the encoded space.

        Args:
            df: Input DataFrame with datetime column.
            datetime_col: Override the datetime column name.

        Returns:
            DataFrame with day_of_year features added.
        """
        col = datetime_col or self.datetime_col
        result = df.copy()
        result = self._ensure_datetime(result, col)

        dt = result[col].dt
        result["day_of_year"] = dt.dayofyear

        # Use 365.25 as period to account for leap years
        sin_vals, cos_vals = self.cyclical_encode(
            result["day_of_year"].values, period=365.25
        )
        result["day_of_year_sin"] = sin_vals
        result["day_of_year_cos"] = cos_vals

        return result

    def compute_cyclical_month(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute month features with cyclical encoding.

        Features:
        - month: Raw month number (1-12)
        - month_sin: sin(2*pi * month/12)
        - month_cos: cos(2*pi * month/12)

        Args:
            df: Input DataFrame with datetime column.
            datetime_col: Override the datetime column name.

        Returns:
            DataFrame with month features added.
        """
        col = datetime_col or self.datetime_col
        result = df.copy()
        result = self._ensure_datetime(result, col)

        dt = result[col].dt
        result["month"] = dt.month

        sin_vals, cos_vals = self.cyclical_encode(
            result["month"].values, period=12
        )
        result["month_sin"] = sin_vals
        result["month_cos"] = cos_vals

        return result

    def compute_season(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute season-related features.

        Features:
        - season: Categorical season name (winter/spring/summer/fall)
        - is_winter: Binary indicator (1 if Dec/Jan/Feb, else 0)
        - is_snow_season: Binary indicator (1 if Nov-Apr, else 0)

        Season definitions:
        - Winter: December, January, February
        - Spring: March, April, May
        - Summer: June, July, August
        - Fall: September, October, November

        Snow season (Nov-Apr) is the typical period when significant
        snowfall occurs in Western US mountains.

        Args:
            df: Input DataFrame with datetime column.
            datetime_col: Override the datetime column name.

        Returns:
            DataFrame with season features added.
        """
        col = datetime_col or self.datetime_col
        result = df.copy()
        result = self._ensure_datetime(result, col)

        month = result[col].dt.month

        # Map months to seasons
        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        }
        result["season"] = month.map(season_map)

        # Binary winter indicator (Dec, Jan, Feb)
        result["is_winter"] = month.isin([12, 1, 2]).astype(int)

        # Binary snow season indicator (Nov through Apr)
        result["is_snow_season"] = month.isin([11, 12, 1, 2, 3, 4]).astype(int)

        return result

    def compute_week_features(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute week-related features.

        Features:
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Binary indicator (1 if Saturday/Sunday, else 0)
        - week_of_year: ISO week number (1-52/53)

        While snowfall is not affected by day of week, these features
        may be useful for operational forecasting (e.g., ski resort planning).

        Args:
            df: Input DataFrame with datetime column.
            datetime_col: Override the datetime column name.

        Returns:
            DataFrame with week features added.
        """
        col = datetime_col or self.datetime_col
        result = df.copy()
        result = self._ensure_datetime(result, col)

        dt = result[col].dt

        # Day of week (0=Monday, 6=Sunday)
        result["day_of_week"] = dt.dayofweek

        # Weekend indicator
        result["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)

        # ISO week of year (1-52/53)
        result["week_of_year"] = dt.isocalendar().week.astype(int)

        return result

    def compute_water_year(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None
    ) -> pd.DataFrame:
        """Compute water year features.

        The water year is the hydrological year used in hydrology studies,
        running from October 1 to September 30. This aligns better with
        the snow accumulation/melt cycle than the calendar year.

        Features:
        - water_year: The water year (e.g., WY2024 runs Oct 1, 2023 to Sep 30, 2024)
        - water_year_day: Day of water year (1-365/366), starting Oct 1
        - water_year_progress: Progress through water year (0.0 to 1.0)

        Args:
            df: Input DataFrame with datetime column.
            datetime_col: Override the datetime column name.

        Returns:
            DataFrame with water year features added.
        """
        col = datetime_col or self.datetime_col
        result = df.copy()
        result = self._ensure_datetime(result, col)

        dt = result[col]

        # Water year: year + 1 if month >= October
        result["water_year"] = dt.dt.year + (dt.dt.month >= 10).astype(int)

        # Water year start date for each row
        water_year_start = pd.to_datetime(
            (result["water_year"] - 1).astype(str) + "-10-01"
        )

        # Days since water year start
        result["water_year_day"] = (dt - water_year_start).dt.days + 1

        # Progress through water year (0 to 1)
        # Use 365.25 days average to account for leap years
        result["water_year_progress"] = (result["water_year_day"] - 1) / 365.25
        # Clip to [0, 1] to handle edge cases
        result["water_year_progress"] = result["water_year_progress"].clip(0, 1)

        return result

    @staticmethod
    def cyclical_encode(
        values: np.ndarray,
        period: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode cyclical feature as sin/cos pair.

        This encoding maps cyclical values to a continuous 2D space where
        values at the beginning and end of the cycle are close together.

        Args:
            values: Array of values to encode (e.g., day of year, month).
            period: The period of the cycle (e.g., 365.25 for days, 12 for months).

        Returns:
            Tuple of (sin_values, cos_values) arrays.

        Example:
            >>> sin_vals, cos_vals = TemporalFeatures.cyclical_encode(
            ...     np.array([1, 183, 365]), period=365
            ... )
            >>> # Day 1 and 365 have similar cos values (both near 1)
            >>> # Day 183 (mid-year) has cos near -1
        """
        values = np.asarray(values, dtype=np.float64)
        angle = 2 * np.pi * values / period
        return np.sin(angle), np.cos(angle)

    def _ensure_datetime(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Ensure the datetime column is of datetime type.

        Args:
            df: Input DataFrame.
            col: Name of the datetime column.

        Returns:
            DataFrame with the column converted to datetime if needed.

        Raises:
            ValueError: If the column cannot be converted to datetime.
        """
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df = df.copy()
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                raise ValueError(
                    f"Cannot convert column '{col}' to datetime: {e}"
                ) from e
        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names generated by compute_all.

        Returns:
            List of feature column names.
        """
        return [
            "day_of_year", "day_of_year_sin", "day_of_year_cos",
            "month", "month_sin", "month_cos",
            "season", "is_winter", "is_snow_season",
            "day_of_week", "is_weekend", "week_of_year",
            "water_year", "water_year_day", "water_year_progress",
        ]
