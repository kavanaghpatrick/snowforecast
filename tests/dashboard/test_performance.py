"""Tests for performance optimization utilities.

Tests the timing, caching, lazy loading, and performance monitoring
components without requiring a running Streamlit server.
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from snowforecast.dashboard.components.performance import (
    TARGET_PAGE_LOAD,
    TARGET_RESORT_SELECT,
    TARGET_TIME_SWITCH,
    PerformanceTimer,
    check_performance_targets,
    clear_performance_metrics,
    get_cached_predictor,
    get_performance_metrics,
    lazy_load,
    prefetch_forecasts,
    timed,
)


class TestPerformanceTimer:
    """Tests for PerformanceTimer context manager."""

    def test_records_elapsed_time(self):
        """Timer records elapsed time accurately."""
        with PerformanceTimer("test_op", log=False) as timer:
            time.sleep(0.1)  # Sleep 100ms

        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should be close to 100ms

    def test_records_operation_name(self):
        """Timer stores operation name."""
        with PerformanceTimer("my_operation", log=False) as timer:
            pass

        assert timer.operation == "my_operation"

    def test_enter_returns_self(self):
        """__enter__ returns the timer instance."""
        timer = PerformanceTimer("test", log=False)
        result = timer.__enter__()
        timer.__exit__(None, None, None)

        assert result is timer

    def test_start_time_set_on_enter(self):
        """Start time is set when entering context."""
        timer = PerformanceTimer("test", log=False)
        before = time.time()
        timer.__enter__()
        after = time.time()
        timer.__exit__(None, None, None)

        assert before <= timer.start_time <= after

    @patch("snowforecast.dashboard.components.performance.logger")
    def test_logs_when_enabled(self, mock_logger):
        """Timer logs when log=True."""
        with PerformanceTimer("logged_op", log=True):
            pass

        mock_logger.info.assert_called_once()
        assert "logged_op" in str(mock_logger.info.call_args)

    @patch("snowforecast.dashboard.components.performance.logger")
    def test_no_log_when_disabled(self, mock_logger):
        """Timer does not log when log=False."""
        with PerformanceTimer("silent_op", log=False):
            pass

        mock_logger.info.assert_not_called()

    def test_stores_metric_in_session_state(self):
        """Timer stores metric in session state."""
        mock_session_state = {}

        with patch(
            "snowforecast.dashboard.components.performance._record_metric",
            side_effect=lambda op, elapsed: mock_session_state.setdefault("performance_metrics", []).append(
                {"operation": op, "elapsed": elapsed, "timestamp": datetime.now().isoformat()}
            ),
        ):
            with PerformanceTimer("stored_op", log=False):
                time.sleep(0.05)

        assert "performance_metrics" in mock_session_state
        metrics = mock_session_state["performance_metrics"]
        assert len(metrics) == 1
        assert metrics[0]["operation"] == "stored_op"
        assert metrics[0]["elapsed"] >= 0.05

    def test_metric_has_timestamp(self):
        """Stored metric includes timestamp."""
        mock_session_state = {}

        with patch(
            "snowforecast.dashboard.components.performance._record_metric",
            side_effect=lambda op, elapsed: mock_session_state.setdefault("performance_metrics", []).append(
                {"operation": op, "elapsed": elapsed, "timestamp": datetime.now().isoformat()}
            ),
        ):
            with PerformanceTimer("timestamped_op", log=False):
                pass

        metrics = mock_session_state["performance_metrics"]
        assert "timestamp" in metrics[0]
        # Should be ISO format
        datetime.fromisoformat(metrics[0]["timestamp"])

    def test_appends_to_existing_metrics(self):
        """Timer appends to existing metrics list."""
        mock_session_state = {
            "performance_metrics": [{"operation": "existing", "elapsed": 1.0, "timestamp": "2024-01-01T00:00:00"}]
        }

        with patch(
            "snowforecast.dashboard.components.performance._record_metric",
            side_effect=lambda op, elapsed: mock_session_state["performance_metrics"].append(
                {"operation": op, "elapsed": elapsed, "timestamp": datetime.now().isoformat()}
            ),
        ):
            with PerformanceTimer("new_op", log=False):
                pass

        metrics = mock_session_state["performance_metrics"]
        assert len(metrics) == 2
        assert metrics[0]["operation"] == "existing"
        assert metrics[1]["operation"] == "new_op"


class TestTimedDecorator:
    """Tests for @timed decorator."""

    def test_decorator_times_function(self):
        """@timed decorator records function execution time."""
        mock_session_state = {}

        with patch(
            "snowforecast.dashboard.components.performance._record_metric",
            side_effect=lambda op, elapsed: mock_session_state.setdefault("performance_metrics", []).append(
                {"operation": op, "elapsed": elapsed, "timestamp": datetime.now().isoformat()}
            ),
        ):

            @timed("decorated_func")
            def slow_function():
                time.sleep(0.05)
                return "result"

            result = slow_function()

        assert result == "result"
        metrics = mock_session_state["performance_metrics"]
        assert len(metrics) == 1
        assert metrics[0]["operation"] == "decorated_func"
        assert metrics[0]["elapsed"] >= 0.05

    def test_decorator_preserves_args(self):
        """@timed decorator preserves function arguments."""

        @timed("with_args")
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = func_with_args("x", "y", c="z")

        assert result == "x-y-z"

    def test_decorator_preserves_function_metadata(self):
        """@timed decorator preserves function name and docstring."""

        @timed("metadata_test")
        def documented_function():
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert "This is the docstring" in documented_function.__doc__

    def test_decorator_handles_exceptions(self):
        """@timed decorator still records time even if function raises."""
        mock_session_state = {}

        with patch(
            "snowforecast.dashboard.components.performance._record_metric",
            side_effect=lambda op, elapsed: mock_session_state.setdefault("performance_metrics", []).append(
                {"operation": op, "elapsed": elapsed, "timestamp": datetime.now().isoformat()}
            ),
        ):

            @timed("exception_func")
            def raising_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                raising_function()

        # Metric should still be recorded
        metrics = mock_session_state["performance_metrics"]
        assert len(metrics) == 1
        assert metrics[0]["operation"] == "exception_func"


class TestGetCachedPredictor:
    """Tests for get_cached_predictor singleton."""

    @patch("snowforecast.dashboard.components.performance.st")
    @patch("snowforecast.cache.CachedPredictor")
    def test_creates_predictor_if_not_exists(self, mock_predictor_class, mock_st):
        """Creates new predictor if not in session state."""
        mock_st.session_state = {}
        mock_instance = MagicMock()
        mock_predictor_class.return_value = mock_instance

        result = get_cached_predictor()

        mock_predictor_class.assert_called_once()
        assert mock_st.session_state["predictor"] == mock_instance
        assert result == mock_instance

    @patch("snowforecast.dashboard.components.performance.st")
    def test_returns_existing_predictor(self, mock_st):
        """Returns existing predictor if already in session state."""
        existing_predictor = MagicMock()
        mock_st.session_state = {"predictor": existing_predictor}

        result = get_cached_predictor()

        assert result is existing_predictor


class TestPrefetchForecasts:
    """Tests for prefetch_forecasts function."""

    @patch("snowforecast.dashboard.components.performance.get_cached_predictor")
    @patch("snowforecast.dashboard.components.performance.st")
    def test_fetches_forecasts_for_all_resorts(self, mock_st, mock_get_predictor):
        """Fetches forecasts for each resort in list."""
        mock_st.session_state = {}
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {"temp": 20}
        mock_get_predictor.return_value = mock_predictor

        resorts = [
            {"name": "Alta", "lat": 40.5, "lon": -111.6},
            {"name": "Snowbird", "lat": 40.6, "lon": -111.7},
        ]

        prefetch_forecasts(resorts)

        assert mock_predictor.predict.call_count == 2
        assert "all_forecasts" in mock_st.session_state
        assert "Alta" in mock_st.session_state["all_forecasts"]
        assert "Snowbird" in mock_st.session_state["all_forecasts"]

    @patch("snowforecast.dashboard.components.performance.get_cached_predictor")
    @patch("snowforecast.dashboard.components.performance.st")
    def test_skips_if_already_fetched(self, mock_st, mock_get_predictor):
        """Does not re-fetch if forecasts already in session state."""
        mock_st.session_state = {"all_forecasts": {"Alta": {"temp": 20}}}
        mock_predictor = MagicMock()
        mock_get_predictor.return_value = mock_predictor

        resorts = [{"name": "Alta", "lat": 40.5, "lon": -111.6}]

        prefetch_forecasts(resorts)

        mock_predictor.predict.assert_not_called()

    @patch("snowforecast.dashboard.components.performance.get_cached_predictor")
    @patch("snowforecast.dashboard.components.performance.st")
    def test_handles_forecast_errors(self, mock_st, mock_get_predictor):
        """Handles errors for individual resort forecasts."""
        mock_st.session_state = {}
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = [
            {"temp": 20},  # Alta succeeds
            Exception("Network error"),  # Snowbird fails
        ]
        mock_get_predictor.return_value = mock_predictor

        resorts = [
            {"name": "Alta", "lat": 40.5, "lon": -111.6},
            {"name": "Snowbird", "lat": 40.6, "lon": -111.7},
        ]

        prefetch_forecasts(resorts)  # Should not raise

        assert mock_st.session_state["all_forecasts"]["Alta"] == {"temp": 20}
        assert mock_st.session_state["all_forecasts"]["Snowbird"] is None


class TestLazyLoad:
    """Tests for @lazy_load decorator."""

    @patch("snowforecast.dashboard.components.performance.st")
    def test_renders_when_visible_true(self, mock_st):
        """Renders component when visibility is True."""
        mock_st.session_state = {"visible_my_component": True}

        @lazy_load("my_component")
        def render_component():
            return "rendered"

        result = render_component()

        assert result == "rendered"

    @patch("snowforecast.dashboard.components.performance.st")
    def test_renders_when_visibility_not_set(self, mock_st):
        """Renders component when visibility flag not set (default True)."""
        mock_st.session_state = {}

        @lazy_load("my_component")
        def render_component():
            return "rendered"

        result = render_component()

        assert result == "rendered"

    @patch("snowforecast.dashboard.components.performance.st")
    def test_skips_when_visible_false(self, mock_st):
        """Skips rendering when visibility is False."""
        mock_st.session_state = {"visible_my_component": False}

        @lazy_load("my_component")
        def render_component():
            return "rendered"

        result = render_component()

        assert result is None

    @patch("snowforecast.dashboard.components.performance.st")
    def test_preserves_function_args(self, mock_st):
        """Preserves function arguments when rendering."""
        mock_st.session_state = {"visible_chart": True}

        @lazy_load("chart")
        def render_chart(data, title="Default"):
            return f"{title}: {data}"

        result = render_chart([1, 2, 3], title="Sales")

        assert result == "Sales: [1, 2, 3]"


class TestClearPerformanceMetrics:
    """Tests for clear_performance_metrics function."""

    @patch("snowforecast.dashboard.components.performance.st")
    def test_clears_metrics(self, mock_st):
        """Clears all performance metrics."""
        mock_st.session_state = {
            "performance_metrics": [
                {"operation": "op1", "elapsed": 1.0, "timestamp": "2024-01-01T00:00:00"},
                {"operation": "op2", "elapsed": 2.0, "timestamp": "2024-01-01T00:01:00"},
            ]
        }

        clear_performance_metrics()

        assert mock_st.session_state["performance_metrics"] == []

    @patch("snowforecast.dashboard.components.performance.st")
    def test_clears_even_if_empty(self, mock_st):
        """Works even if metrics list is empty."""
        mock_st.session_state = {"performance_metrics": []}

        clear_performance_metrics()

        assert mock_st.session_state["performance_metrics"] == []


class TestGetPerformanceMetrics:
    """Tests for get_performance_metrics function."""

    @patch("snowforecast.dashboard.components.performance.st")
    def test_returns_metrics(self, mock_st):
        """Returns all collected metrics."""
        metrics = [
            {"operation": "op1", "elapsed": 1.0, "timestamp": "2024-01-01T00:00:00"},
            {"operation": "op2", "elapsed": 2.0, "timestamp": "2024-01-01T00:01:00"},
        ]
        mock_st.session_state = {"performance_metrics": metrics}

        result = get_performance_metrics()

        assert result == metrics

    @patch("snowforecast.dashboard.components.performance.st")
    def test_returns_empty_list_if_none(self, mock_st):
        """Returns empty list if no metrics collected."""
        mock_st.session_state = {}

        result = get_performance_metrics()

        assert result == []


class TestCheckPerformanceTargets:
    """Tests for check_performance_targets function."""

    @patch("snowforecast.dashboard.components.performance.st")
    def test_returns_empty_if_no_metrics(self, mock_st):
        """Returns empty dict if no metrics."""
        mock_st.session_state = {}

        result = check_performance_targets()

        assert result == {}

    @patch("snowforecast.dashboard.components.performance.st")
    def test_checks_page_load_target(self, mock_st):
        """Checks page load against target."""
        mock_st.session_state = {
            "performance_metrics": [{"operation": "page_load", "elapsed": 2.0, "timestamp": "2024-01-01T00:00:00"}]
        }

        result = check_performance_targets()

        assert result["page_load"] is True  # 2.0 < 3.0

    @patch("snowforecast.dashboard.components.performance.st")
    def test_fails_page_load_over_target(self, mock_st):
        """Fails page load check when over target."""
        mock_st.session_state = {
            "performance_metrics": [{"operation": "page_load", "elapsed": 4.0, "timestamp": "2024-01-01T00:00:00"}]
        }

        result = check_performance_targets()

        assert result["page_load"] is False  # 4.0 > 3.0

    @patch("snowforecast.dashboard.components.performance.st")
    def test_checks_time_switch_target(self, mock_st):
        """Checks time switch against target."""
        mock_st.session_state = {
            "performance_metrics": [{"operation": "time_switch", "elapsed": 0.5, "timestamp": "2024-01-01T00:00:00"}]
        }

        result = check_performance_targets()

        assert result["time_switch"] is True  # 0.5 < 1.0

    @patch("snowforecast.dashboard.components.performance.st")
    def test_checks_resort_select_target(self, mock_st):
        """Checks resort selection against target."""
        mock_st.session_state = {
            "performance_metrics": [{"operation": "resort_select", "elapsed": 0.3, "timestamp": "2024-01-01T00:00:00"}]
        }

        result = check_performance_targets()

        assert result["resort_select"] is True  # 0.3 < 0.5


class TestPerformanceTargetConstants:
    """Tests for performance target constants."""

    def test_page_load_target(self):
        """Page load target is 3 seconds."""
        assert TARGET_PAGE_LOAD == 3.0

    def test_time_switch_target(self):
        """Time switch target is 1 second."""
        assert TARGET_TIME_SWITCH == 1.0

    def test_resort_select_target(self):
        """Resort select target is 0.5 seconds."""
        assert TARGET_RESORT_SELECT == 0.5


class TestRenderPerformanceMetrics:
    """Tests for render_performance_metrics function."""

    @patch("snowforecast.dashboard.components.performance.st")
    def test_shows_info_when_no_metrics(self, mock_st):
        """Shows info message when no metrics collected."""
        mock_st.session_state = {}
        mock_st.info = MagicMock()

        from snowforecast.dashboard.components.performance import render_performance_metrics

        render_performance_metrics()

        mock_st.info.assert_called_once_with("No performance metrics collected yet")

    @patch("snowforecast.dashboard.components.performance.st")
    def test_renders_metrics_with_colors(self, mock_st):
        """Renders metrics with color coding based on elapsed time."""
        mock_st.session_state = {
            "performance_metrics": [
                {"operation": "fast_op", "elapsed": 0.5, "timestamp": "2024-01-01T00:00:00"},
                {"operation": "medium_op", "elapsed": 2.0, "timestamp": "2024-01-01T00:00:01"},
                {"operation": "slow_op", "elapsed": 4.0, "timestamp": "2024-01-01T00:00:02"},
            ]
        }
        mock_st.markdown = MagicMock()

        from snowforecast.dashboard.components.performance import render_performance_metrics

        render_performance_metrics()

        # Check markdown was called for each metric
        calls = mock_st.markdown.call_args_list
        assert len(calls) == 4  # Header + 3 metrics
        assert "Performance Metrics" in str(calls[0])
        assert "green" in str(calls[1])  # fast_op < 1s
        assert "orange" in str(calls[2])  # medium_op between 1-3s
        assert "red" in str(calls[3])  # slow_op > 3s

    @patch("snowforecast.dashboard.components.performance.st")
    def test_limits_to_last_10_metrics(self, mock_st):
        """Only shows last 10 metrics."""
        metrics = [{"operation": f"op_{i}", "elapsed": 0.5, "timestamp": f"2024-01-01T00:00:{i:02d}"} for i in range(15)]
        mock_st.session_state = {"performance_metrics": metrics}
        mock_st.markdown = MagicMock()

        from snowforecast.dashboard.components.performance import render_performance_metrics

        render_performance_metrics()

        # Header + 10 metrics = 11 calls
        assert mock_st.markdown.call_count == 11

    @patch("snowforecast.dashboard.components.performance.st")
    def test_uses_custom_container(self, mock_st):
        """Uses custom container when provided."""
        mock_st.session_state = {}
        custom_container = MagicMock()

        from snowforecast.dashboard.components.performance import render_performance_metrics

        render_performance_metrics(container=custom_container)

        custom_container.info.assert_called_once()
        mock_st.info.assert_not_called()
