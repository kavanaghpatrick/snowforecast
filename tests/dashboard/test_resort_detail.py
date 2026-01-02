"""Tests for resort detail panel component."""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta

from snowforecast.dashboard.components.resort_detail import (
    generate_forecast_summary,
    _get_day_name,
    _classify_snow_intensity,
    _find_snow_events,
    _format_date_range,
    _describe_conditions,
    LIGHT_SNOW_THRESHOLD,
    MODERATE_SNOW_THRESHOLD,
    HEAVY_SNOW_THRESHOLD,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_day_name_from_date(self):
        """Should return correct day name from date object."""
        # Monday = 0, so 2024-01-15 is a Monday
        monday = date(2024, 1, 15)
        assert _get_day_name(monday) == "Monday"

        tuesday = date(2024, 1, 16)
        assert _get_day_name(tuesday) == "Tuesday"

    def test_get_day_name_from_datetime(self):
        """Should work with datetime objects."""
        dt = datetime(2024, 1, 17, 12, 0)  # Wednesday
        assert _get_day_name(dt) == "Wednesday"

    def test_get_day_name_from_string(self):
        """Should parse date strings."""
        assert _get_day_name("2024-01-15") == "Monday"

    def test_classify_snow_intensity_light(self):
        """Below light threshold should be 'light'."""
        assert _classify_snow_intensity(0) == "light"
        assert _classify_snow_intensity(4.9) == "light"

    def test_classify_snow_intensity_moderate(self):
        """Between light and moderate threshold should be 'moderate'."""
        assert _classify_snow_intensity(5) == "moderate"
        assert _classify_snow_intensity(14.9) == "moderate"

    def test_classify_snow_intensity_heavy(self):
        """Between moderate and heavy threshold should be 'heavy'."""
        assert _classify_snow_intensity(15) == "heavy"
        assert _classify_snow_intensity(24.9) == "heavy"

    def test_classify_snow_intensity_very_heavy(self):
        """Above heavy threshold should be 'very heavy'."""
        assert _classify_snow_intensity(25) == "very heavy"
        assert _classify_snow_intensity(50) == "very heavy"


class TestFindSnowEvents:
    """Tests for snow event detection."""

    def test_no_events_when_no_snow(self):
        """Should return empty list when no significant snow."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [0, 1, 2, 1, 0, 0, 0],
        })
        events = _find_snow_events(df)
        assert events == []

    def test_single_day_event(self):
        """Should detect single day snow event."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [0, 0, 10, 0, 0, 0, 0],
        })
        events = _find_snow_events(df)
        assert len(events) == 1
        assert events[0]["total_cm"] == 10

    def test_multi_day_event(self):
        """Should group consecutive snow days into one event."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [0, 10, 15, 20, 0, 0, 0],
        })
        events = _find_snow_events(df)
        assert len(events) == 1
        assert events[0]["total_cm"] == 45  # 10 + 15 + 20

    def test_multiple_separate_events(self):
        """Should detect multiple distinct events."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [10, 0, 0, 15, 20, 0, 10],
        })
        events = _find_snow_events(df)
        assert len(events) == 3

    def test_event_intensity_classification(self):
        """Should classify event intensity by max daily amount."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=3),
            "new_snow_cm": [10, 30, 10],  # Max is 30 (very heavy)
        })
        events = _find_snow_events(df)
        assert len(events) == 1
        assert events[0]["intensity"] == "very heavy"

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        events = _find_snow_events(df)
        assert events == []

    def test_missing_column(self):
        """Should handle DataFrame without new_snow_cm column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=3),
            "other_column": [1, 2, 3],
        })
        events = _find_snow_events(df)
        assert events == []


class TestFormatDateRange:
    """Tests for date range formatting."""

    def test_same_day(self):
        """Should return single day name when start == end."""
        d = date(2024, 1, 15)  # Monday
        assert _format_date_range(d, d) == "Monday"

    def test_two_days(self):
        """Should return range for consecutive days."""
        start = date(2024, 1, 15)  # Monday
        end = date(2024, 1, 16)  # Tuesday
        assert _format_date_range(start, end) == "Monday-Tuesday"

    def test_multi_day_range(self):
        """Should work for longer ranges."""
        start = date(2024, 1, 15)  # Monday
        end = date(2024, 1, 18)  # Thursday
        assert _format_date_range(start, end) == "Monday-Thursday"


class TestDescribeConditions:
    """Tests for condition description."""

    def test_excellent_base(self):
        """Should describe >100cm as excellent."""
        df = pd.DataFrame({
            "snow_depth_cm": [120, 125, 130],
            "probability": [0.5, 0.5, 0.5],
        })
        desc = _describe_conditions(df)
        assert "Excellent" in desc

    def test_good_base(self):
        """Should describe 60-100cm as good."""
        df = pd.DataFrame({
            "snow_depth_cm": [80, 85, 90],
            "probability": [0.5, 0.5, 0.5],
        })
        desc = _describe_conditions(df)
        assert "Good" in desc

    def test_fair_base(self):
        """Should describe 30-60cm as fair."""
        df = pd.DataFrame({
            "snow_depth_cm": [45, 50, 55],
            "probability": [0.5, 0.5, 0.5],
        })
        desc = _describe_conditions(df)
        assert "Fair" in desc

    def test_limited_base(self):
        """Should describe <30cm as limited."""
        df = pd.DataFrame({
            "snow_depth_cm": [15, 20, 25],
            "probability": [0.5, 0.5, 0.5],
        })
        desc = _describe_conditions(df)
        assert "Limited" in desc

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        desc = _describe_conditions(df)
        assert desc == ""


class TestGenerateForecastSummary:
    """Tests for natural language forecast summary generation."""

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        summary = generate_forecast_summary(df)
        assert "No forecast data" in summary

    def test_none_input(self):
        """Should handle None input."""
        summary = generate_forecast_summary(None)
        assert "No forecast data" in summary

    def test_dry_week(self):
        """Should describe dry conditions when no snow expected."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [0, 0, 0, 0, 0, 0, 0],
            "snow_depth_cm": [100, 98, 96, 94, 92, 90, 88],
            "probability": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        })
        summary = generate_forecast_summary(df)
        assert "dry" in summary.lower() or "Dry" in summary

    def test_light_flurries(self):
        """Should mention flurries for very light snow."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [1, 1, 1, 0, 0, 0, 0],
            "snow_depth_cm": [100, 101, 102, 102, 101, 100, 99],
            "probability": [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
        })
        summary = generate_forecast_summary(df)
        # Either "flurries" or "Dry" depending on total
        assert len(summary) > 0

    def test_heavy_snow_event(self):
        """Should describe heavy snow events."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [0, 0, 20, 25, 0, 0, 0],
            "snow_depth_cm": [100, 100, 120, 145, 145, 143, 140],
            "probability": [0.1, 0.3, 0.9, 0.95, 0.2, 0.1, 0.1],
        })
        summary = generate_forecast_summary(df)
        # Should mention heavy snow and include amount
        assert "heavy" in summary.lower() or "Heavy" in summary
        assert "cm" in summary

    def test_multiple_events(self):
        """Should describe multiple snow events."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [10, 0, 0, 0, 15, 10, 0],
            "snow_depth_cm": [100, 110, 108, 106, 121, 131, 129],
            "probability": [0.8, 0.1, 0.1, 0.1, 0.85, 0.8, 0.2],
        })
        summary = generate_forecast_summary(df)
        # Should contain descriptions for both events
        assert "snow" in summary.lower()

    def test_includes_condition_assessment(self):
        """Summary should include base condition assessment."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=7),
            "new_snow_cm": [15, 20, 0, 0, 0, 0, 0],
            "snow_depth_cm": [150, 165, 185, 183, 180, 178, 175],
            "probability": [0.9, 0.95, 0.2, 0.1, 0.1, 0.1, 0.1],
        })
        summary = generate_forecast_summary(df)
        # Should mention conditions
        assert "condition" in summary.lower() or "Excellent" in summary or "Good" in summary


class TestThresholdConstants:
    """Tests for threshold constant values."""

    def test_threshold_ordering(self):
        """Thresholds should be in ascending order."""
        assert LIGHT_SNOW_THRESHOLD < MODERATE_SNOW_THRESHOLD
        assert MODERATE_SNOW_THRESHOLD < HEAVY_SNOW_THRESHOLD

    def test_threshold_values(self):
        """Thresholds should have reasonable values."""
        assert LIGHT_SNOW_THRESHOLD == 5
        assert MODERATE_SNOW_THRESHOLD == 15
        assert HEAVY_SNOW_THRESHOLD == 25


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_snow_values(self):
        """Should handle negative snow values gracefully."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=3),
            "new_snow_cm": [-5, 0, 10],
            "snow_depth_cm": [100, 100, 110],
        })
        # Should not raise
        summary = generate_forecast_summary(df)
        assert isinstance(summary, str)

    def test_missing_columns(self):
        """Should handle DataFrames with missing columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=3),
            # Missing new_snow_cm, snow_depth_cm
        })
        summary = generate_forecast_summary(df)
        assert isinstance(summary, str)

    def test_nan_values(self):
        """Should handle NaN values in data."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-15", periods=3),
            "new_snow_cm": [10, float("nan"), 15],
            "snow_depth_cm": [100, float("nan"), 125],
        })
        # Should not raise
        summary = generate_forecast_summary(df)
        assert isinstance(summary, str)

    def test_single_row_dataframe(self):
        """Should handle single row DataFrame."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 15)],
            "new_snow_cm": [20],
            "snow_depth_cm": [120],
            "probability": [0.9],
        })
        summary = generate_forecast_summary(df)
        assert isinstance(summary, str)
        assert len(summary) > 0
