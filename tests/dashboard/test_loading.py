"""Tests for loading and error handling components.

Tests the decorators and utility functions without requiring Streamlit.
Streamlit-dependent rendering functions are tested with mocks.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestWithLoadingDecorator:
    """Tests for the with_loading decorator."""

    @patch("snowforecast.dashboard.components.loading.st")
    def test_returns_original_value(self, mock_st):
        """Decorator does not change the return value."""
        from snowforecast.dashboard.components.loading import with_loading

        @with_loading("Loading...")
        def add_numbers(a, b):
            return a + b

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        result = add_numbers(2, 3)
        assert result == 5

    @patch("snowforecast.dashboard.components.loading.st")
    def test_preserves_none_return(self, mock_st):
        """Decorator preserves None return value."""
        from snowforecast.dashboard.components.loading import with_loading

        @with_loading("Loading...")
        def returns_none():
            return None

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        result = returns_none()
        assert result is None

    @patch("snowforecast.dashboard.components.loading.st")
    def test_calls_spinner_with_message(self, mock_st):
        """Decorator calls st.spinner with provided message."""
        from snowforecast.dashboard.components.loading import with_loading

        @with_loading("Custom loading message")
        def some_function():
            return 42

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        some_function()
        mock_st.spinner.assert_called_once_with("Custom loading message")

    @patch("snowforecast.dashboard.components.loading.st")
    def test_preserves_function_metadata(self, mock_st):
        """Decorator preserves function name and docstring."""
        from snowforecast.dashboard.components.loading import with_loading

        @with_loading("Loading...")
        def documented_function():
            """This is a docstring."""
            return 1

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."

    @patch("snowforecast.dashboard.components.loading.st")
    def test_passes_args_and_kwargs(self, mock_st):
        """Decorator passes positional and keyword arguments correctly."""
        from snowforecast.dashboard.components.loading import with_loading

        @with_loading("Loading...")
        def function_with_args(a, b, c=10):
            return a + b + c

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        result = function_with_args(1, 2, c=20)
        assert result == 23


class TestWithErrorHandlingDecorator:
    """Tests for the with_error_handling decorator."""

    @patch("snowforecast.dashboard.components.loading.st")
    def test_returns_value_on_success(self, mock_st):
        """Returns function value when no exception."""
        from snowforecast.dashboard.components.loading import with_error_handling

        @with_error_handling("Error")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"
        mock_st.error.assert_not_called()

    @patch("snowforecast.dashboard.components.loading.st")
    def test_catches_exception_returns_none(self, mock_st):
        """Catches exception and returns None."""
        from snowforecast.dashboard.components.loading import with_error_handling

        @with_error_handling("Operation failed")
        def failing_function():
            raise ValueError("Something went wrong")

        result = failing_function()
        assert result is None

    @patch("snowforecast.dashboard.components.loading.st")
    def test_displays_error_message(self, mock_st):
        """Displays user-friendly error message via st.error."""
        from snowforecast.dashboard.components.loading import with_error_handling

        @with_error_handling("Custom error message")
        def failing_function():
            raise RuntimeError("Detailed error")

        failing_function()
        mock_st.error.assert_called_once()
        call_args = mock_st.error.call_args[0][0]
        assert "Custom error message" in call_args
        assert "Detailed error" in call_args

    @patch("snowforecast.dashboard.components.loading.st")
    def test_preserves_function_metadata(self, mock_st):
        """Decorator preserves function name and docstring."""
        from snowforecast.dashboard.components.loading import with_error_handling

        @with_error_handling("Error")
        def my_documented_function():
            """My docstring here."""
            return 1

        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "My docstring here."

    @patch("snowforecast.dashboard.components.loading.st")
    def test_catches_all_exception_types(self, mock_st):
        """Catches various exception types."""
        from snowforecast.dashboard.components.loading import with_error_handling

        @with_error_handling("Error")
        def raises_type_error():
            raise TypeError("Type error")

        @with_error_handling("Error")
        def raises_key_error():
            raise KeyError("key")

        result1 = raises_type_error()
        result2 = raises_key_error()

        assert result1 is None
        assert result2 is None
        assert mock_st.error.call_count == 2


class TestWithLoadingAndErrorHandling:
    """Tests for the combined decorator."""

    @patch("snowforecast.dashboard.components.loading.st")
    def test_shows_spinner_and_handles_error(self, mock_st):
        """Combined decorator shows spinner and handles errors."""
        from snowforecast.dashboard.components.loading import with_loading_and_error_handling

        @with_loading_and_error_handling(
            loading_message="Loading data...",
            error_message="Failed to load"
        )
        def risky_function():
            raise ValueError("Oops")

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        result = risky_function()

        assert result is None
        mock_st.spinner.assert_called_once_with("Loading data...")
        mock_st.error.assert_called_once()

    @patch("snowforecast.dashboard.components.loading.st")
    def test_returns_value_on_success(self, mock_st):
        """Combined decorator returns value on success."""
        from snowforecast.dashboard.components.loading import with_loading_and_error_handling

        @with_loading_and_error_handling(
            loading_message="Working...",
            error_message="Failed"
        )
        def successful_function():
            return {"data": "value"}

        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        result = successful_function()

        assert result == {"data": "value"}
        mock_st.error.assert_not_called()


class TestGetCacheFreshness:
    """Tests for get_cache_freshness function."""

    def test_fresh_cache(self):
        """Cache within 30 minutes is Fresh."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        recent = datetime.now() - timedelta(minutes=15)
        status, color = get_cache_freshness(recent)

        assert status == "Fresh"
        assert color == "#22c55e"  # Green

    def test_recent_cache(self):
        """Cache between 30min and 2 hours is Recent."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        recent = datetime.now() - timedelta(hours=1)
        status, color = get_cache_freshness(recent)

        assert status == "Recent"
        assert color == "#3b82f6"  # Blue

    def test_stale_cache(self):
        """Cache between 2 and 6 hours is Stale."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        stale = datetime.now() - timedelta(hours=4)
        status, color = get_cache_freshness(stale)

        assert status == "Stale"
        assert color == "#f59e0b"  # Amber

    def test_old_cache(self):
        """Cache older than 6 hours is Old."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        old = datetime.now() - timedelta(hours=8)
        status, color = get_cache_freshness(old)

        assert status == "Old"
        assert color == "#ef4444"  # Red

    def test_none_returns_unknown(self):
        """None timestamp returns Unknown status."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        status, color = get_cache_freshness(None)

        assert status == "Unknown"
        assert color == "#9ca3af"  # Gray

    def test_boundary_fresh_to_recent(self):
        """Exactly at 30 minute boundary is Recent (not Fresh)."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        at_boundary = datetime.now() - timedelta(minutes=30)
        status, _ = get_cache_freshness(at_boundary)

        # At exactly 30 min, it's no longer Fresh
        assert status == "Recent"

    def test_boundary_recent_to_stale(self):
        """At 2 hour boundary is Stale."""
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        at_boundary = datetime.now() - timedelta(hours=2)
        status, _ = get_cache_freshness(at_boundary)

        assert status == "Stale"


class TestFormatCacheAge:
    """Tests for format_cache_age function."""

    def test_just_now(self):
        """Less than 1 minute shows 'Just now'."""
        from snowforecast.dashboard.components.cache_status import format_cache_age

        recent = datetime.now() - timedelta(seconds=30)
        result = format_cache_age(recent)

        assert result == "Just now"

    def test_minutes_ago(self):
        """Minutes are formatted correctly."""
        from snowforecast.dashboard.components.cache_status import format_cache_age

        minutes_ago = datetime.now() - timedelta(minutes=5)
        result = format_cache_age(minutes_ago)

        assert result == "5 min ago"

    def test_hours_ago(self):
        """Hours are formatted with decimal."""
        from snowforecast.dashboard.components.cache_status import format_cache_age

        hours_ago = datetime.now() - timedelta(hours=2, minutes=30)
        result = format_cache_age(hours_ago)

        assert "hrs ago" in result
        assert "2.5" in result

    def test_days_ago(self):
        """Days are formatted with decimal."""
        from snowforecast.dashboard.components.cache_status import format_cache_age

        days_ago = datetime.now() - timedelta(days=1, hours=12)
        result = format_cache_age(days_ago)

        assert "days ago" in result
        assert "1.5" in result

    def test_none_returns_never(self):
        """None returns 'Never'."""
        from snowforecast.dashboard.components.cache_status import format_cache_age

        result = format_cache_age(None)

        assert result == "Never"


class TestShouldShowStaleWarning:
    """Tests for should_show_stale_warning function."""

    def test_fresh_data_no_warning(self):
        """Fresh data should not show warning."""
        from snowforecast.dashboard.components.cache_status import should_show_stale_warning

        fresh = datetime.now() - timedelta(minutes=30)
        assert should_show_stale_warning(fresh) is False

    def test_stale_data_shows_warning(self):
        """Stale data should show warning."""
        from snowforecast.dashboard.components.cache_status import should_show_stale_warning

        stale = datetime.now() - timedelta(hours=3)
        assert should_show_stale_warning(stale) is True

    def test_none_shows_warning(self):
        """None timestamp should show warning."""
        from snowforecast.dashboard.components.cache_status import should_show_stale_warning

        assert should_show_stale_warning(None) is True


class TestRenderRetryButton:
    """Tests for render_retry_button function."""

    @patch("snowforecast.dashboard.components.loading.st")
    def test_returns_false_when_not_clicked(self, mock_st):
        """Returns False when button not clicked."""
        from snowforecast.dashboard.components.loading import render_retry_button

        mock_st.button.return_value = False

        result = render_retry_button("test_key")

        assert result is False

    @patch("snowforecast.dashboard.components.loading.st")
    def test_returns_true_when_clicked(self, mock_st):
        """Returns True when button clicked."""
        from snowforecast.dashboard.components.loading import render_retry_button

        mock_st.button.return_value = True

        result = render_retry_button("test_key")

        assert result is True

    @patch("snowforecast.dashboard.components.loading.st")
    def test_calls_callback_when_clicked(self, mock_st):
        """Calls on_click callback when button clicked."""
        from snowforecast.dashboard.components.loading import render_retry_button

        mock_st.button.return_value = True
        callback = MagicMock()

        render_retry_button("test_key", on_click=callback)

        callback.assert_called_once()

    @patch("snowforecast.dashboard.components.loading.st")
    def test_no_callback_when_not_clicked(self, mock_st):
        """Does not call callback when button not clicked."""
        from snowforecast.dashboard.components.loading import render_retry_button

        mock_st.button.return_value = False
        callback = MagicMock()

        render_retry_button("test_key", on_click=callback)

        callback.assert_not_called()


class TestRenderCacheStatus:
    """Tests for render_cache_status function."""

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_handles_missing_method(self, mock_st):
        """Handles predictor without get_cache_stats method."""
        from snowforecast.dashboard.components.cache_status import render_cache_status

        mock_predictor = MagicMock(spec=[])  # No get_cache_stats

        render_cache_status(mock_predictor)

        mock_st.sidebar.warning.assert_called()

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_displays_stats(self, mock_st):
        """Displays cache statistics."""
        from snowforecast.dashboard.components.cache_status import render_cache_status

        mock_predictor = MagicMock()
        mock_predictor.get_cache_stats.return_value = {
            "forecast_count": 100,
            "terrain_count": 22,
            "latest_run_time": datetime.now() - timedelta(minutes=15),
        }

        render_cache_status(mock_predictor)

        # Should have called markdown for status display
        assert mock_st.sidebar.markdown.called


class TestRenderDataWarning:
    """Tests for render_data_warning function."""

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_no_warning_for_fresh_data(self, mock_st):
        """No warning displayed for fresh data."""
        from snowforecast.dashboard.components.cache_status import render_data_warning

        fresh = datetime.now() - timedelta(minutes=30)
        render_data_warning(fresh, threshold_hours=2.0)

        mock_st.warning.assert_not_called()

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_warning_for_stale_data(self, mock_st):
        """Warning displayed for stale data."""
        from snowforecast.dashboard.components.cache_status import render_data_warning

        stale = datetime.now() - timedelta(hours=3)
        render_data_warning(stale, threshold_hours=2.0)

        mock_st.warning.assert_called_once()
        call_args = mock_st.warning.call_args[0][0]
        assert "3.0" in call_args  # Should mention hours

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_warning_for_none_timestamp(self, mock_st):
        """Warning displayed when timestamp is None."""
        from snowforecast.dashboard.components.cache_status import render_data_warning

        render_data_warning(None)

        mock_st.warning.assert_called_once()


class TestRenderFallbackNotice:
    """Tests for render_fallback_notice function."""

    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_displays_info_with_reason(self, mock_st):
        """Displays info message with reason."""
        from snowforecast.dashboard.components.cache_status import render_fallback_notice

        render_fallback_notice("API timeout")

        mock_st.info.assert_called_once()
        call_args = mock_st.info.call_args[0][0]
        assert "cached data" in call_args.lower()
        assert "API timeout" in call_args


class TestFreshnessThresholds:
    """Tests for freshness threshold constants."""

    def test_thresholds_are_timedeltas(self):
        """Thresholds are timedelta objects."""
        from snowforecast.dashboard.components.cache_status import FRESHNESS_THRESHOLDS

        assert isinstance(FRESHNESS_THRESHOLDS["fresh"], timedelta)
        assert isinstance(FRESHNESS_THRESHOLDS["recent"], timedelta)
        assert isinstance(FRESHNESS_THRESHOLDS["stale"], timedelta)

    def test_thresholds_are_ordered(self):
        """Thresholds are in ascending order."""
        from snowforecast.dashboard.components.cache_status import FRESHNESS_THRESHOLDS

        assert FRESHNESS_THRESHOLDS["fresh"] < FRESHNESS_THRESHOLDS["recent"]
        assert FRESHNESS_THRESHOLDS["recent"] < FRESHNESS_THRESHOLDS["stale"]


class TestFreshnessColors:
    """Tests for freshness color constants."""

    def test_colors_are_hex_strings(self):
        """Colors are valid hex color strings."""
        from snowforecast.dashboard.components.cache_status import FRESHNESS_COLORS

        for key, color in FRESHNESS_COLORS.items():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format

    def test_all_statuses_have_colors(self):
        """All status types have corresponding colors."""
        from snowforecast.dashboard.components.cache_status import FRESHNESS_COLORS

        expected_keys = {"fresh", "recent", "stale", "old", "unknown"}
        assert set(FRESHNESS_COLORS.keys()) == expected_keys


class TestIntegrationScenarios:
    """Integration tests for typical usage patterns."""

    @patch("snowforecast.dashboard.components.loading.st")
    @patch("snowforecast.dashboard.components.cache_status.st")
    def test_fetch_with_loading_and_status_check(self, mock_cache_st, mock_loading_st):
        """Simulate fetching data with loading and checking cache status."""
        from snowforecast.dashboard.components.loading import with_loading
        from snowforecast.dashboard.components.cache_status import get_cache_freshness

        mock_loading_st.spinner.return_value.__enter__ = MagicMock()
        mock_loading_st.spinner.return_value.__exit__ = MagicMock()

        @with_loading("Fetching forecast...")
        def fetch_forecast():
            return {"snow_cm": 15, "probability": 0.8}

        # Fetch data
        result = fetch_forecast()
        assert result["snow_cm"] == 15

        # Check cache status
        cache_time = datetime.now() - timedelta(minutes=10)
        status, color = get_cache_freshness(cache_time)
        assert status == "Fresh"

    @patch("snowforecast.dashboard.components.loading.st")
    def test_error_handling_workflow(self, mock_st):
        """Simulate error handling workflow."""
        from snowforecast.dashboard.components.loading import (
            with_error_handling,
            render_retry_button,
        )

        @with_error_handling("Failed to load resort data")
        def fetch_resort_data():
            raise ConnectionError("Network unavailable")

        # First attempt fails
        result = fetch_resort_data()
        assert result is None
        mock_st.error.assert_called()

        # User sees retry button
        mock_st.button.return_value = True
        clicked = render_retry_button("resort_data")
        assert clicked is True
