"""Tests for time selector component.

Tests the TIME_STEPS constant, get_forecast_time() conversion logic,
and prefetch functionality. Since render functions require Streamlit context,
we test the pure functions that handle datetime calculations.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from snowforecast.dashboard.components.time_selector import (
    ALL_FORECASTS_KEY,
    SELECTED_TIME_STEP_KEY,
    TIME_STEPS,
    calculate_forecast_datetime,
    get_forecast_hours,
    get_forecast_time,
    parse_time_step,
)


class TestTimeStepsConstant:
    """Tests for TIME_STEPS constant."""

    def test_time_steps_has_9_elements(self):
        """TIME_STEPS should have exactly 9 elements."""
        assert len(TIME_STEPS) == 9

    def test_time_steps_starts_with_now(self):
        """First element should be 'Now'."""
        assert TIME_STEPS[0] == "Now"

    def test_time_steps_contains_tonight(self):
        """TIME_STEPS should include 'Tonight'."""
        assert "Tonight" in TIME_STEPS

    def test_time_steps_contains_tomorrow_am(self):
        """TIME_STEPS should include 'Tomorrow AM'."""
        assert "Tomorrow AM" in TIME_STEPS

    def test_time_steps_contains_tomorrow_pm(self):
        """TIME_STEPS should include 'Tomorrow PM'."""
        assert "Tomorrow PM" in TIME_STEPS

    def test_time_steps_contains_day_3_through_7(self):
        """TIME_STEPS should include Day 3 through Day 7."""
        for day in [3, 4, 5, 6, 7]:
            assert f"Day {day}" in TIME_STEPS

    def test_time_steps_ends_with_day_7(self):
        """Last element should be 'Day 7'."""
        assert TIME_STEPS[-1] == "Day 7"

    def test_time_steps_order(self):
        """TIME_STEPS should be in chronological order."""
        expected = [
            "Now",
            "Tonight",
            "Tomorrow AM",
            "Tomorrow PM",
            "Day 3",
            "Day 4",
            "Day 5",
            "Day 6",
            "Day 7",
        ]
        assert TIME_STEPS == expected

    def test_time_steps_is_list(self):
        """TIME_STEPS should be a list."""
        assert isinstance(TIME_STEPS, list)

    def test_time_steps_elements_are_strings(self):
        """All TIME_STEPS elements should be strings."""
        for step in TIME_STEPS:
            assert isinstance(step, str)


class TestGetForecastTime:
    """Tests for get_forecast_time() conversion logic."""

    def test_now_returns_base_time(self):
        """'Now' should return the base time unchanged."""
        base = datetime(2026, 1, 15, 14, 30)
        result = get_forecast_time("Now", base)
        assert result == base

    def test_tonight_returns_9pm_same_day(self):
        """'Tonight' should return 9 PM of the same day."""
        base = datetime(2026, 1, 15, 14, 0)  # 2 PM
        result = get_forecast_time("Tonight", base)
        assert result.hour == 21
        assert result.day == 15

    def test_tonight_when_past_9pm_returns_base(self):
        """'Tonight' when already past 9 PM should return base time."""
        base = datetime(2026, 1, 15, 22, 0)  # 10 PM
        result = get_forecast_time("Tonight", base)
        assert result == base

    def test_tomorrow_am_returns_9am_next_day(self):
        """'Tomorrow AM' should return 9 AM of the next day."""
        base = datetime(2026, 1, 15, 14, 0)
        result = get_forecast_time("Tomorrow AM", base)
        assert result.hour == 9
        assert result.day == 16
        assert result.month == 1

    def test_tomorrow_pm_returns_3pm_next_day(self):
        """'Tomorrow PM' should return 3 PM of the next day."""
        base = datetime(2026, 1, 15, 14, 0)
        result = get_forecast_time("Tomorrow PM", base)
        assert result.hour == 15
        assert result.day == 16

    def test_day_3_returns_noon_2_days_ahead(self):
        """'Day 3' should return noon 2 days ahead."""
        base = datetime(2026, 1, 15, 14, 0)
        result = get_forecast_time("Day 3", base)
        assert result.hour == 12
        assert result.day == 17

    def test_day_7_returns_noon_6_days_ahead(self):
        """'Day 7' should return noon 6 days ahead."""
        base = datetime(2026, 1, 15, 14, 0)
        result = get_forecast_time("Day 7", base)
        assert result.hour == 12
        assert result.day == 21

    def test_invalid_time_step_raises_value_error(self):
        """Invalid time step should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_forecast_time("Invalid", datetime.now())
        assert "Invalid time_step" in str(exc_info.value)

    def test_base_time_none_uses_current_time(self):
        """When base_time is None, should use datetime.now()."""
        result = get_forecast_time("Now")
        # Should be very close to now
        now = datetime.now()
        delta = abs((result - now).total_seconds())
        assert delta < 2  # Within 2 seconds

    def test_month_boundary_crossing(self):
        """Should handle month boundaries correctly."""
        base = datetime(2026, 1, 31, 14, 0)  # Jan 31
        result = get_forecast_time("Tomorrow AM", base)
        assert result.month == 2
        assert result.day == 1

    def test_year_boundary_crossing(self):
        """Should handle year boundaries correctly."""
        base = datetime(2025, 12, 31, 14, 0)  # Dec 31
        result = get_forecast_time("Tomorrow AM", base)
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 1


class TestGetForecastHours:
    """Tests for get_forecast_hours() calculation."""

    def test_now_returns_minimum_1_hour(self):
        """'Now' should return at least 1 hour."""
        base = datetime(2026, 1, 15, 14, 0)
        result = get_forecast_hours("Now", base)
        assert result >= 1

    def test_tonight_returns_hours_until_9pm(self):
        """'Tonight' should return hours from base to 9 PM."""
        base = datetime(2026, 1, 15, 14, 0)  # 2 PM
        result = get_forecast_hours("Tonight", base)
        assert result == 7  # 2 PM to 9 PM = 7 hours

    def test_tomorrow_am_returns_correct_hours(self):
        """'Tomorrow AM' should return correct hours ahead."""
        base = datetime(2026, 1, 15, 14, 0)  # 2 PM
        result = get_forecast_hours("Tomorrow AM", base)
        # 2 PM to 9 AM next day = 19 hours
        assert result == 19

    def test_day_7_returns_correct_hours(self):
        """'Day 7' should return approximately 144 hours (6 days)."""
        base = datetime(2026, 1, 15, 12, 0)  # noon
        result = get_forecast_hours("Day 7", base)
        # Noon to noon 6 days later = 144 hours
        assert result == 144


class TestParseTimeStep:
    """Tests for parse_time_step() pure function."""

    def test_parse_now(self):
        """'Now' should return (0, -1)."""
        days, hour = parse_time_step("Now")
        assert days == 0
        assert hour == -1  # -1 indicates use current hour

    def test_parse_tonight(self):
        """'Tonight' should return (0, 21)."""
        days, hour = parse_time_step("Tonight")
        assert days == 0
        assert hour == 21

    def test_parse_tomorrow_am(self):
        """'Tomorrow AM' should return (1, 9)."""
        days, hour = parse_time_step("Tomorrow AM")
        assert days == 1
        assert hour == 9

    def test_parse_tomorrow_pm(self):
        """'Tomorrow PM' should return (1, 15)."""
        days, hour = parse_time_step("Tomorrow PM")
        assert days == 1
        assert hour == 15

    def test_parse_day_3(self):
        """'Day 3' should return (2, 12)."""
        days, hour = parse_time_step("Day 3")
        assert days == 2
        assert hour == 12

    def test_parse_day_7(self):
        """'Day 7' should return (6, 12)."""
        days, hour = parse_time_step("Day 7")
        assert days == 6
        assert hour == 12

    def test_parse_invalid_raises_value_error(self):
        """Invalid time step should raise ValueError."""
        with pytest.raises(ValueError):
            parse_time_step("Invalid")

    def test_parse_all_time_steps_succeeds(self):
        """All TIME_STEPS should parse without error."""
        for step in TIME_STEPS:
            days, hour = parse_time_step(step)
            assert isinstance(days, int)
            assert isinstance(hour, int)


class TestCalculateForecastDatetime:
    """Tests for calculate_forecast_datetime() pure function."""

    def test_calculate_now(self):
        """'Now' should return base datetime."""
        year, month, day, hour = calculate_forecast_datetime(
            "Now", 2026, 1, 15, 14
        )
        assert year == 2026
        assert month == 1
        assert day == 15
        assert hour == 14

    def test_calculate_tonight(self):
        """'Tonight' should return 9 PM same day."""
        year, month, day, hour = calculate_forecast_datetime(
            "Tonight", 2026, 1, 15, 14
        )
        assert year == 2026
        assert month == 1
        assert day == 15
        assert hour == 21

    def test_calculate_tomorrow_am(self):
        """'Tomorrow AM' should return 9 AM next day."""
        year, month, day, hour = calculate_forecast_datetime(
            "Tomorrow AM", 2026, 1, 15, 14
        )
        assert year == 2026
        assert month == 1
        assert day == 16
        assert hour == 9

    def test_calculate_day_7(self):
        """'Day 7' should return noon 6 days later."""
        year, month, day, hour = calculate_forecast_datetime(
            "Day 7", 2026, 1, 15, 14
        )
        assert year == 2026
        assert month == 1
        assert day == 21
        assert hour == 12

    def test_calculate_handles_month_boundary(self):
        """Should handle month boundary correctly."""
        year, month, day, hour = calculate_forecast_datetime(
            "Tomorrow AM", 2026, 1, 31, 14
        )
        assert year == 2026
        assert month == 2
        assert day == 1


class TestPrefetchAllForecasts:
    """Tests for prefetch_all_forecasts() function."""

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_prefetch_returns_dict_with_all_time_steps(self, mock_st):
        """Should return dict with entry for each time step."""
        mock_st.session_state = {}

        mock_predictor = MagicMock()
        mock_forecast = MagicMock()
        mock_confidence = MagicMock()
        mock_predictor.predict.return_value = (mock_forecast, mock_confidence)

        from snowforecast.dashboard.components.time_selector import prefetch_all_forecasts

        result = prefetch_all_forecasts(mock_predictor, 47.0, -121.0)

        assert len(result) == 9
        for step in TIME_STEPS:
            assert step in result

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_prefetch_sets_session_state(self, mock_st):
        """Should set session state with forecasts."""
        mock_st.session_state = {}

        mock_predictor = MagicMock()
        mock_forecast = MagicMock()
        mock_confidence = MagicMock()
        mock_predictor.predict.return_value = (mock_forecast, mock_confidence)

        from snowforecast.dashboard.components.time_selector import prefetch_all_forecasts

        prefetch_all_forecasts(mock_predictor, 47.0, -121.0)

        assert ALL_FORECASTS_KEY in mock_st.session_state
        assert "forecast_lat" in mock_st.session_state
        assert "forecast_lon" in mock_st.session_state

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_prefetch_handles_predictor_error(self, mock_st):
        """Should handle predictor errors gracefully."""
        mock_st.session_state = {}

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = Exception("API error")

        from snowforecast.dashboard.components.time_selector import prefetch_all_forecasts

        result = prefetch_all_forecasts(mock_predictor, 47.0, -121.0)

        # Should still return dict with all time steps
        assert len(result) == 9
        # Each should have an error key
        for step in TIME_STEPS:
            assert "error" in result[step]

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_prefetch_calls_predictor_for_each_time_step(self, mock_st):
        """Should call predictor.predict() for each time step."""
        mock_st.session_state = {}

        mock_predictor = MagicMock()
        mock_forecast = MagicMock()
        mock_confidence = MagicMock()
        mock_predictor.predict.return_value = (mock_forecast, mock_confidence)

        from snowforecast.dashboard.components.time_selector import prefetch_all_forecasts

        prefetch_all_forecasts(mock_predictor, 47.0, -121.0)

        assert mock_predictor.predict.call_count == 9

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_prefetch_stores_forecast_time(self, mock_st):
        """Each forecast should include the target time."""
        mock_st.session_state = {}

        mock_predictor = MagicMock()
        mock_forecast = MagicMock()
        mock_confidence = MagicMock()
        mock_predictor.predict.return_value = (mock_forecast, mock_confidence)

        from snowforecast.dashboard.components.time_selector import prefetch_all_forecasts

        result = prefetch_all_forecasts(mock_predictor, 47.0, -121.0)

        for step in TIME_STEPS:
            assert "time" in result[step]
            assert isinstance(result[step]["time"], datetime)


class TestNeedsPrefetch:
    """Tests for needs_prefetch() function."""

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_needs_prefetch_when_no_cache(self, mock_st):
        """Should return True when no cache exists."""
        mock_st.session_state = {}

        from snowforecast.dashboard.components.time_selector import needs_prefetch

        assert needs_prefetch(47.0, -121.0) is True

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_needs_prefetch_when_location_changed(self, mock_st):
        """Should return True when location has changed."""
        mock_st.session_state = {
            ALL_FORECASTS_KEY: {},
            "forecast_lat": 47.0,
            "forecast_lon": -121.0,
        }

        from snowforecast.dashboard.components.time_selector import needs_prefetch

        # Different location
        assert needs_prefetch(48.0, -120.0) is True

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_no_prefetch_when_same_location(self, mock_st):
        """Should return False when location is the same."""
        mock_st.session_state = {
            ALL_FORECASTS_KEY: {},
            "forecast_lat": 47.0,
            "forecast_lon": -121.0,
        }

        from snowforecast.dashboard.components.time_selector import needs_prefetch

        # Same location (within tolerance)
        assert needs_prefetch(47.0, -121.0) is False
        assert needs_prefetch(47.0005, -121.0005) is False


class TestClearForecastCache:
    """Tests for clear_forecast_cache() function."""

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_clear_removes_all_cache_keys(self, mock_st):
        """Should remove all forecast cache keys from session state."""
        mock_st.session_state = {
            ALL_FORECASTS_KEY: {},
            "forecast_lat": 47.0,
            "forecast_lon": -121.0,
            "forecast_base_time": datetime.now(),
            "other_key": "should_remain",
        }

        from snowforecast.dashboard.components.time_selector import clear_forecast_cache

        clear_forecast_cache()

        assert ALL_FORECASTS_KEY not in mock_st.session_state
        assert "forecast_lat" not in mock_st.session_state
        assert "forecast_lon" not in mock_st.session_state
        assert "forecast_base_time" not in mock_st.session_state
        assert "other_key" in mock_st.session_state

    @patch("snowforecast.dashboard.components.time_selector.st")
    def test_clear_handles_empty_state(self, mock_st):
        """Should not raise error when session state is empty."""
        mock_st.session_state = {}

        from snowforecast.dashboard.components.time_selector import clear_forecast_cache

        # Should not raise
        clear_forecast_cache()


class TestSessionStateKeys:
    """Tests for session state key constants."""

    def test_selected_time_step_key_is_string(self):
        """SELECTED_TIME_STEP_KEY should be a string."""
        assert isinstance(SELECTED_TIME_STEP_KEY, str)

    def test_all_forecasts_key_is_string(self):
        """ALL_FORECASTS_KEY should be a string."""
        assert isinstance(ALL_FORECASTS_KEY, str)

    def test_keys_are_unique(self):
        """Session state keys should be unique."""
        assert SELECTED_TIME_STEP_KEY != ALL_FORECASTS_KEY
