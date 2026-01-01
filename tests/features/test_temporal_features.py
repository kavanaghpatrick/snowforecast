"""Tests for temporal and cyclical feature engineering.

These tests verify that TemporalFeatures correctly computes temporal
features used for snow forecasting, including:
- Cyclical encoding of day of year and month
- Season and snow season indicators
- Week-based features
- Hydrological water year calculations
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.features.temporal_features import TemporalFeatures


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with datetime column."""
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=365, freq="D"),
        "value": np.random.randn(365),
    })


@pytest.fixture
def tf() -> TemporalFeatures:
    """Create a TemporalFeatures instance."""
    return TemporalFeatures()


# =============================================================================
# Test compute_all
# =============================================================================

class TestComputeAll:
    """Tests for the compute_all method."""

    def test_compute_all_adds_all_features(self, tf, sample_df):
        """Test that compute_all adds all expected features."""
        result = tf.compute_all(sample_df)

        expected_features = tf.get_feature_names()
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_compute_all_preserves_original_columns(self, tf, sample_df):
        """Test that compute_all preserves original columns."""
        result = tf.compute_all(sample_df)

        assert "datetime" in result.columns
        assert "value" in result.columns
        assert len(result) == len(sample_df)

    def test_compute_all_does_not_modify_input(self, tf, sample_df):
        """Test that compute_all does not modify the input DataFrame."""
        original_cols = list(sample_df.columns)
        _ = tf.compute_all(sample_df)

        assert list(sample_df.columns) == original_cols

    def test_compute_all_with_custom_datetime_col(self, tf):
        """Test compute_all with a custom datetime column name."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = tf.compute_all(df, datetime_col="timestamp")
        assert "day_of_year" in result.columns

    def test_compute_all_missing_column_raises(self, tf, sample_df):
        """Test that compute_all raises KeyError for missing datetime column."""
        with pytest.raises(KeyError, match="not found"):
            tf.compute_all(sample_df, datetime_col="nonexistent")


# =============================================================================
# Test cyclical encoding
# =============================================================================

class TestCyclicalEncoding:
    """Tests for cyclical sin/cos encoding."""

    def test_cyclical_encode_day_of_year(self, tf):
        """Test that day 1 and day 365 are close in sin/cos space."""
        # Day 1 and day 365 should be close (cyclical nature)
        sin_vals, cos_vals = tf.cyclical_encode(np.array([1, 365]), period=365)

        # Both days should have similar cosine values (both near 1.0)
        assert np.isclose(cos_vals[0], cos_vals[1], atol=0.02)

        # Sin values should be close but opposite sign
        # sin(2*pi*1/365) ~ 0.017, sin(2*pi*365/365) ~ 0.0
        assert np.isclose(sin_vals[0], -sin_vals[1], atol=0.02)

    def test_cyclical_encode_mid_year_opposite(self, tf):
        """Test that mid-year (day ~183) is opposite to year start/end."""
        sin_vals, cos_vals = tf.cyclical_encode(np.array([1, 183]), period=365)

        # Day 183 (mid-year) should have cosine near -1.0
        assert cos_vals[1] < -0.95

        # Day 1 should have cosine near 1.0
        assert cos_vals[0] > 0.95

    def test_cyclical_encode_months(self, tf):
        """Test cyclical encoding for months."""
        # January (1) and December (12) should be close
        sin_vals, cos_vals = tf.cyclical_encode(np.array([1, 12]), period=12)

        # Distance in sin/cos space should be small
        dist = np.sqrt((sin_vals[0] - sin_vals[1])**2 + (cos_vals[0] - cos_vals[1])**2)
        assert dist < 0.6  # Close in circular space

    def test_compute_cyclical_day_of_year(self, tf, sample_df):
        """Test compute_cyclical_day_of_year method."""
        result = tf.compute_cyclical_day_of_year(sample_df)

        assert "day_of_year" in result.columns
        assert "day_of_year_sin" in result.columns
        assert "day_of_year_cos" in result.columns

        # Check day_of_year values are in expected range
        assert result["day_of_year"].min() == 1
        assert result["day_of_year"].max() == 365

        # Check sin/cos values are in [-1, 1]
        assert result["day_of_year_sin"].between(-1, 1).all()
        assert result["day_of_year_cos"].between(-1, 1).all()

    def test_compute_cyclical_month(self, tf, sample_df):
        """Test compute_cyclical_month method."""
        result = tf.compute_cyclical_month(sample_df)

        assert "month" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

        # Check month values are in expected range
        assert result["month"].min() == 1
        assert result["month"].max() == 12

        # Check sin/cos values are in [-1, 1]
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()


# =============================================================================
# Test season features
# =============================================================================

class TestSeasonFeatures:
    """Tests for season-related features."""

    def test_compute_season_categorization(self, tf):
        """Test that seasons are correctly assigned."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-01-15"),  # Winter
                pd.Timestamp("2023-04-15"),  # Spring
                pd.Timestamp("2023-07-15"),  # Summer
                pd.Timestamp("2023-10-15"),  # Fall
                pd.Timestamp("2023-12-15"),  # Winter
            ]
        })

        result = tf.compute_season(df)

        assert result["season"].tolist() == [
            "winter", "spring", "summer", "fall", "winter"
        ]

    def test_is_winter_indicator(self, tf):
        """Test is_winter binary indicator."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-12-01"),  # Winter
                pd.Timestamp("2023-01-15"),  # Winter
                pd.Timestamp("2023-02-28"),  # Winter
                pd.Timestamp("2023-03-01"),  # Not winter
                pd.Timestamp("2023-06-15"),  # Not winter
            ]
        })

        result = tf.compute_season(df)

        assert result["is_winter"].tolist() == [1, 1, 1, 0, 0]

    def test_is_snow_season_indicator(self, tf):
        """Test is_snow_season binary indicator (Nov-Apr)."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-11-01"),  # Snow season
                pd.Timestamp("2023-12-15"),  # Snow season
                pd.Timestamp("2024-01-15"),  # Snow season
                pd.Timestamp("2024-02-15"),  # Snow season
                pd.Timestamp("2024-03-15"),  # Snow season
                pd.Timestamp("2024-04-30"),  # Snow season
                pd.Timestamp("2024-05-01"),  # Not snow season
                pd.Timestamp("2024-08-01"),  # Not snow season
                pd.Timestamp("2024-10-31"),  # Not snow season
            ]
        })

        result = tf.compute_season(df)

        expected = [1, 1, 1, 1, 1, 1, 0, 0, 0]
        assert result["is_snow_season"].tolist() == expected


# =============================================================================
# Test week features
# =============================================================================

class TestWeekFeatures:
    """Tests for week-related features."""

    def test_day_of_week(self, tf):
        """Test day_of_week calculation (0=Monday, 6=Sunday)."""
        # 2024-01-01 is a Monday
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=7, freq="D")
        })

        result = tf.compute_week_features(df)

        assert result["day_of_week"].tolist() == [0, 1, 2, 3, 4, 5, 6]

    def test_is_weekend(self, tf):
        """Test is_weekend indicator."""
        # 2024-01-01 is a Monday
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=7, freq="D")
        })

        result = tf.compute_week_features(df)

        # Saturday (5) and Sunday (6) are weekends
        assert result["is_weekend"].tolist() == [0, 0, 0, 0, 0, 1, 1]

    def test_week_of_year(self, tf):
        """Test week_of_year calculation."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-01"),  # Week 1
                pd.Timestamp("2024-06-15"),  # ~Week 24
                pd.Timestamp("2024-12-31"),  # Week 1 of next year (ISO)
            ]
        })

        result = tf.compute_week_features(df)

        # Check week values are in valid range
        assert result["week_of_year"].min() >= 1
        assert result["week_of_year"].max() <= 53


# =============================================================================
# Test water year features
# =============================================================================

class TestWaterYearFeatures:
    """Tests for hydrological water year features."""

    def test_water_year_calculation(self, tf):
        """Test water year is calculated correctly (Oct 1 - Sep 30)."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-09-30"),  # WY 2023
                pd.Timestamp("2023-10-01"),  # WY 2024
                pd.Timestamp("2024-09-30"),  # WY 2024
                pd.Timestamp("2024-10-01"),  # WY 2025
            ]
        })

        result = tf.compute_water_year(df)

        assert result["water_year"].tolist() == [2023, 2024, 2024, 2025]

    def test_water_year_day(self, tf):
        """Test water_year_day calculation (day 1 = Oct 1)."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-10-01"),  # Day 1 of WY 2024
                pd.Timestamp("2023-10-02"),  # Day 2 of WY 2024
                pd.Timestamp("2023-11-01"),  # Day 32 of WY 2024
                pd.Timestamp("2024-01-01"),  # Day 93 of WY 2024
            ]
        })

        result = tf.compute_water_year(df)

        assert result["water_year_day"].iloc[0] == 1
        assert result["water_year_day"].iloc[1] == 2
        assert result["water_year_day"].iloc[2] == 32
        assert result["water_year_day"].iloc[3] == 93

    def test_water_year_progress(self, tf):
        """Test water_year_progress is in [0, 1] range."""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-10-01"),  # Start of WY (progress ~0)
                pd.Timestamp("2024-04-01"),  # Mid WY (~0.5)
                pd.Timestamp("2024-09-30"),  # End of WY (progress ~1)
            ]
        })

        result = tf.compute_water_year(df)

        assert 0 <= result["water_year_progress"].iloc[0] <= 0.01
        assert 0.4 < result["water_year_progress"].iloc[1] < 0.6
        assert 0.99 <= result["water_year_progress"].iloc[2] <= 1.0

    def test_water_year_full_year(self, tf):
        """Test water year features for a full water year."""
        # Full water year 2024 (Oct 1, 2023 to Sep 30, 2024)
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-10-01", "2024-09-30", freq="D")
        })

        result = tf.compute_water_year(df)

        # All should be WY 2024
        assert (result["water_year"] == 2024).all()

        # Day should range from 1 to 366 (2024 is leap year)
        assert result["water_year_day"].min() == 1
        assert result["water_year_day"].max() == 366

        # Progress should be in [0, 1]
        assert result["water_year_progress"].min() >= 0
        assert result["water_year_progress"].max() <= 1


# =============================================================================
# Test edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_string_datetime_conversion(self, tf):
        """Test that string datetime columns are converted."""
        df = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-06-15", "2023-12-31"],
            "value": [1, 2, 3],
        })

        result = tf.compute_all(df)

        assert "day_of_year" in result.columns
        assert result["day_of_year"].tolist() == [1, 166, 365]

    def test_empty_dataframe(self, tf):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime([]),
            "value": [],
        })

        result = tf.compute_all(df)

        assert len(result) == 0
        assert "day_of_year" in result.columns

    def test_leap_year_handling(self, tf):
        """Test that leap years are handled correctly."""
        # 2024 is a leap year
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-02-29"),  # Leap day
                pd.Timestamp("2024-03-01"),  # Day after leap day
            ]
        })

        result = tf.compute_cyclical_day_of_year(df)

        assert result["day_of_year"].tolist() == [60, 61]

    def test_invalid_datetime_raises(self, tf):
        """Test that invalid datetime values raise appropriate error."""
        df = pd.DataFrame({
            "datetime": ["not-a-date", "also-not-a-date"],
            "value": [1, 2],
        })

        with pytest.raises(ValueError, match="Cannot convert"):
            tf.compute_all(df)

    def test_instance_datetime_col_config(self):
        """Test that instance datetime_col configuration works."""
        tf = TemporalFeatures(datetime_col="timestamp")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10),
            "value": range(10),
        })

        result = tf.compute_all(df)

        assert "day_of_year" in result.columns

    def test_get_feature_names(self, tf):
        """Test get_feature_names returns all feature names."""
        feature_names = tf.get_feature_names()

        assert len(feature_names) == 15
        assert "day_of_year" in feature_names
        assert "water_year_progress" in feature_names


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests for TemporalFeatures."""

    def test_multi_year_data(self, tf):
        """Test with multi-year data spanning multiple water years."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", "2023-12-31", freq="D"),
            "value": np.random.randn(1461),  # ~4 years
        })

        result = tf.compute_all(df)

        # Check water years span correctly
        assert result["water_year"].min() == 2020
        assert result["water_year"].max() == 2024  # Oct-Dec 2023 is WY 2024

        # Check all features are present
        for feature in tf.get_feature_names():
            assert feature in result.columns

    def test_hourly_data(self, tf):
        """Test with hourly data."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=24*7, freq="h"),
            "value": np.random.randn(24*7),
        })

        result = tf.compute_all(df)

        # Day of year should repeat for each hour
        assert (result["day_of_year"] <= 7).all()

        # Week features should work with hourly data
        assert "is_weekend" in result.columns
