"""Tests for confidence visualization component.

Tests the confidence level classification, badge generation,
color mapping, and formatting functions.
"""

import pytest
from snowforecast.api.schemas import ConfidenceInterval
from snowforecast.dashboard.components.confidence import (
    get_confidence_level,
    get_confidence_badge,
    get_confidence_color,
    format_forecast_with_ci,
)


class TestGetConfidenceLevel:
    """Tests for confidence level classification."""

    def test_high_confidence_narrow_ci_high_prob(self):
        """Narrow CI (<5cm) with high probability (>0.7) returns high."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)  # width = 4
        assert get_confidence_level(ci, 0.8) == "high"

    def test_high_confidence_boundary_ci_width(self):
        """CI width exactly at boundary (4.99) with high prob returns high."""
        ci = ConfidenceInterval(lower=10.0, upper=14.99)  # width = 4.99
        assert get_confidence_level(ci, 0.75) == "high"

    def test_high_confidence_boundary_probability(self):
        """High confidence requires probability > 0.7 (not >=)."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)  # width = 4
        # Probability exactly 0.7 should NOT be high
        assert get_confidence_level(ci, 0.7) == "medium"
        # Probability just above 0.7 should be high
        assert get_confidence_level(ci, 0.71) == "high"

    def test_medium_confidence_moderate_ci_moderate_prob(self):
        """Moderate CI (<15cm) with moderate probability (>0.4) returns medium."""
        ci = ConfidenceInterval(lower=5.0, upper=18.0)  # width = 13
        assert get_confidence_level(ci, 0.5) == "medium"

    def test_medium_confidence_boundary_ci_width(self):
        """CI width exactly at boundary (14.99) with moderate prob returns medium."""
        ci = ConfidenceInterval(lower=5.0, upper=19.99)  # width = 14.99
        assert get_confidence_level(ci, 0.45) == "medium"

    def test_medium_confidence_boundary_probability(self):
        """Medium confidence requires probability > 0.4 (not >=)."""
        ci = ConfidenceInterval(lower=5.0, upper=18.0)  # width = 13
        # Probability exactly 0.4 should NOT be medium
        assert get_confidence_level(ci, 0.4) == "low"
        # Probability just above 0.4 should be medium
        assert get_confidence_level(ci, 0.41) == "medium"

    def test_medium_confidence_narrow_ci_low_prob(self):
        """Narrow CI but low probability returns medium (not high)."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)  # width = 4
        # Low probability (0.5) but narrow CI - should be medium
        assert get_confidence_level(ci, 0.5) == "medium"

    def test_low_confidence_wide_ci(self):
        """Wide CI (>=15cm) returns low regardless of probability."""
        ci = ConfidenceInterval(lower=0.0, upper=15.0)  # width = 15
        assert get_confidence_level(ci, 0.9) == "low"

    def test_low_confidence_very_wide_ci(self):
        """Very wide CI returns low."""
        ci = ConfidenceInterval(lower=0.0, upper=50.0)  # width = 50
        assert get_confidence_level(ci, 0.95) == "low"

    def test_low_confidence_low_probability(self):
        """Very low probability returns low regardless of CI width."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)  # width = 4
        assert get_confidence_level(ci, 0.2) == "low"

    def test_zero_ci_width(self):
        """Zero CI width (lower == upper) with high prob returns high."""
        ci = ConfidenceInterval(lower=10.0, upper=10.0)  # width = 0
        assert get_confidence_level(ci, 0.9) == "high"

    def test_zero_probability(self):
        """Zero probability returns low."""
        ci = ConfidenceInterval(lower=10.0, upper=12.0)  # width = 2
        assert get_confidence_level(ci, 0.0) == "low"

    def test_one_hundred_percent_probability(self):
        """100% probability with narrow CI returns high."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)  # width = 4
        assert get_confidence_level(ci, 1.0) == "high"

    def test_negative_ci_values(self):
        """Negative CI values (edge case) still work based on width."""
        ci = ConfidenceInterval(lower=-5.0, upper=0.0)  # width = 5
        assert get_confidence_level(ci, 0.8) == "medium"  # width = 5 (not <5)


class TestGetConfidenceBadge:
    """Tests for badge text generation."""

    def test_high_confidence_badge(self):
        """High confidence returns green badge."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)
        badge = get_confidence_badge(ci, 0.8)
        assert badge == "🟢 High Confidence"

    def test_medium_confidence_badge(self):
        """Medium confidence returns yellow badge."""
        ci = ConfidenceInterval(lower=5.0, upper=18.0)
        badge = get_confidence_badge(ci, 0.5)
        assert badge == "🟡 Medium Confidence"

    def test_low_confidence_badge(self):
        """Low confidence returns red badge."""
        ci = ConfidenceInterval(lower=0.0, upper=30.0)
        badge = get_confidence_badge(ci, 0.3)
        assert badge == "🔴 Low Confidence"


class TestGetConfidenceColor:
    """Tests for CSS color mapping."""

    def test_high_color(self):
        """High level returns green color."""
        assert get_confidence_color("high") == "#22c55e"

    def test_medium_color(self):
        """Medium level returns yellow color."""
        assert get_confidence_color("medium") == "#eab308"

    def test_low_color(self):
        """Low level returns red color."""
        assert get_confidence_color("low") == "#ef4444"

    def test_unknown_level_returns_gray(self):
        """Unknown level returns gray fallback."""
        assert get_confidence_color("unknown") == "#6b7280"

    def test_empty_string_returns_gray(self):
        """Empty string returns gray fallback."""
        assert get_confidence_color("") == "#6b7280"


class TestFormatForecastWithCI:
    """Tests for forecast formatting with confidence interval."""

    def test_basic_formatting(self):
        """Basic formatting works correctly."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0)
        result = format_forecast_with_ci(15.0, ci)
        assert result == "15 +/- 5 cm"

    def test_rounding_value(self):
        """Forecast value is rounded to nearest integer."""
        ci = ConfidenceInterval(lower=10.0, upper=20.0)
        result = format_forecast_with_ci(15.7, ci)
        assert result == "16 +/- 5 cm"

    def test_rounding_margin(self):
        """Margin is rounded to nearest integer."""
        ci = ConfidenceInterval(lower=10.0, upper=17.0)  # margin = 3.5
        result = format_forecast_with_ci(13.5, ci)
        assert result == "14 +/- 4 cm"

    def test_zero_margin(self):
        """Zero margin (same lower/upper) formats correctly."""
        ci = ConfidenceInterval(lower=15.0, upper=15.0)
        result = format_forecast_with_ci(15.0, ci)
        assert result == "15 +/- 0 cm"

    def test_large_values(self):
        """Large forecast values format correctly."""
        ci = ConfidenceInterval(lower=80.0, upper=120.0)
        result = format_forecast_with_ci(100.0, ci)
        assert result == "100 +/- 20 cm"

    def test_zero_value(self):
        """Zero forecast value formats correctly."""
        ci = ConfidenceInterval(lower=0.0, upper=5.0)
        result = format_forecast_with_ci(0.0, ci)
        assert result == "0 +/- 2 cm"

    def test_asymmetric_ci(self):
        """Asymmetric CI still uses half-width as margin."""
        # Value 15, CI from 10-30 (asymmetric around value)
        ci = ConfidenceInterval(lower=10.0, upper=30.0)
        result = format_forecast_with_ci(15.0, ci)
        # Margin is (30-10)/2 = 10, regardless of where value sits
        assert result == "15 +/- 10 cm"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_ci_values(self):
        """Very large CI values are handled."""
        ci = ConfidenceInterval(lower=0.0, upper=500.0)
        level = get_confidence_level(ci, 0.5)
        assert level == "low"
        formatted = format_forecast_with_ci(250.0, ci)
        assert formatted == "250 +/- 250 cm"

    def test_small_ci_values(self):
        """Very small CI values are handled."""
        ci = ConfidenceInterval(lower=9.9, upper=10.1)  # width = 0.2
        level = get_confidence_level(ci, 0.99)
        assert level == "high"
        formatted = format_forecast_with_ci(10.0, ci)
        assert formatted == "10 +/- 0 cm"

    def test_probability_at_exactly_one(self):
        """Probability of exactly 1.0 is handled."""
        ci = ConfidenceInterval(lower=10.0, upper=14.0)
        level = get_confidence_level(ci, 1.0)
        assert level == "high"

    def test_ci_with_decimal_precision(self):
        """CI with many decimal places is handled."""
        ci = ConfidenceInterval(lower=10.12345, upper=14.98765)
        level = get_confidence_level(ci, 0.87654)
        assert level == "high"  # width ~4.86, prob > 0.7


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_complete_workflow_high(self):
        """Complete workflow for high confidence forecast."""
        ci = ConfidenceInterval(lower=12.0, upper=16.0)
        probability = 0.85

        level = get_confidence_level(ci, probability)
        badge = get_confidence_badge(ci, probability)
        color = get_confidence_color(level)
        formatted = format_forecast_with_ci(14.0, ci)

        assert level == "high"
        assert "High Confidence" in badge
        assert color == "#22c55e"
        assert formatted == "14 +/- 2 cm"

    def test_complete_workflow_medium(self):
        """Complete workflow for medium confidence forecast."""
        ci = ConfidenceInterval(lower=8.0, upper=20.0)
        probability = 0.55

        level = get_confidence_level(ci, probability)
        badge = get_confidence_badge(ci, probability)
        color = get_confidence_color(level)
        formatted = format_forecast_with_ci(14.0, ci)

        assert level == "medium"
        assert "Medium Confidence" in badge
        assert color == "#eab308"
        assert formatted == "14 +/- 6 cm"

    def test_complete_workflow_low(self):
        """Complete workflow for low confidence forecast."""
        ci = ConfidenceInterval(lower=0.0, upper=40.0)
        probability = 0.25

        level = get_confidence_level(ci, probability)
        badge = get_confidence_badge(ci, probability)
        color = get_confidence_color(level)
        formatted = format_forecast_with_ci(20.0, ci)

        assert level == "low"
        assert "Low Confidence" in badge
        assert color == "#ef4444"
        assert formatted == "20 +/- 20 cm"
