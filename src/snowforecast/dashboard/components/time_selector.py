"""Time step selector component for forecast period navigation.

Provides 9 discrete time step buttons (not a slider) for selecting forecast
periods from "Now" through "Day 7". Designed for Streamlit+PyDeck dashboards
where smooth animation is not practical.

Usage:
    from snowforecast.dashboard.components.time_selector import (
        TIME_STEPS,
        render_time_selector,
        get_forecast_time,
        prefetch_all_forecasts,
    )

    # In Streamlit app
    selected = render_time_selector()
    forecast_time = get_forecast_time(selected)

    # Pre-fetch all forecasts on initial load
    forecasts = prefetch_all_forecasts(predictor, lat, lon)
"""

from datetime import datetime, timedelta
from typing import Optional

import streamlit as st

# Session state keys
SELECTED_TIME_STEP_KEY = "selected_time_step"
ALL_FORECASTS_KEY = "all_forecasts"

# 9 discrete time steps for forecast navigation
TIME_STEPS = [
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


def get_forecast_time(time_step: str, base_time: Optional[datetime] = None) -> datetime:
    """Convert time step string to datetime for forecast lookup.

    Maps user-friendly time step names to actual datetime values based
    on the current time or a provided base time.

    Args:
        time_step: One of TIME_STEPS (e.g., "Now", "Tomorrow AM", "Day 3")
        base_time: Reference time for calculations. If None, uses datetime.now()

    Returns:
        datetime representing the forecast target time

    Raises:
        ValueError: If time_step is not in TIME_STEPS

    Examples:
        >>> from datetime import datetime
        >>> base = datetime(2026, 1, 15, 14, 0)  # 2pm
        >>> get_forecast_time("Now", base)
        datetime.datetime(2026, 1, 15, 14, 0)
        >>> get_forecast_time("Tomorrow AM", base)
        datetime.datetime(2026, 1, 16, 9, 0)
    """
    if time_step not in TIME_STEPS:
        raise ValueError(f"Invalid time_step '{time_step}'. Must be one of {TIME_STEPS}")

    if base_time is None:
        base_time = datetime.now()

    # Get the start of today for date calculations
    today_start = base_time.replace(hour=0, minute=0, second=0, microsecond=0)

    if time_step == "Now":
        return base_time

    elif time_step == "Tonight":
        # Tonight at 9 PM
        tonight = today_start.replace(hour=21)
        # If it's already past 9 PM, return current time
        if base_time.hour >= 21:
            return base_time
        return tonight

    elif time_step == "Tomorrow AM":
        # Tomorrow at 9 AM
        tomorrow = today_start + timedelta(days=1)
        return tomorrow.replace(hour=9)

    elif time_step == "Tomorrow PM":
        # Tomorrow at 3 PM
        tomorrow = today_start + timedelta(days=1)
        return tomorrow.replace(hour=15)

    elif time_step.startswith("Day "):
        # Day 3 through Day 7
        day_num = int(time_step.split()[1])
        target_day = today_start + timedelta(days=day_num - 1)
        # Use noon for day forecasts
        return target_day.replace(hour=12)

    # Should never reach here due to validation above
    return base_time


def get_forecast_hours(time_step: str, base_time: Optional[datetime] = None) -> int:
    """Calculate forecast hours ahead for a given time step.

    Args:
        time_step: One of TIME_STEPS
        base_time: Reference time. If None, uses datetime.now()

    Returns:
        Hours ahead from base_time to the forecast time
    """
    if base_time is None:
        base_time = datetime.now()

    forecast_time = get_forecast_time(time_step, base_time)
    delta = forecast_time - base_time
    hours = max(1, int(delta.total_seconds() / 3600))
    return hours


def render_time_selector(container=None) -> str:
    """Render time step buttons for forecast selection.

    Displays 9 horizontal radio buttons for selecting forecast time periods.
    Updates st.session_state.selected_time_step on selection.

    Args:
        container: Streamlit container (st, column, etc.). If None, uses st.

    Returns:
        Currently selected time step string

    Side Effects:
        - Sets st.session_state.selected_time_step to the selected value
    """
    target = container if container is not None else st

    # Initialize session state if needed
    if SELECTED_TIME_STEP_KEY not in st.session_state:
        st.session_state[SELECTED_TIME_STEP_KEY] = TIME_STEPS[0]

    # Get current selection
    current = st.session_state[SELECTED_TIME_STEP_KEY]

    # Ensure current value is valid
    if current not in TIME_STEPS:
        current = TIME_STEPS[0]
        st.session_state[SELECTED_TIME_STEP_KEY] = current

    # Find current index
    current_index = TIME_STEPS.index(current)

    # Render horizontal radio buttons
    selected = target.radio(
        "Forecast Time",
        options=TIME_STEPS,
        index=current_index,
        horizontal=True,
        key="time_step_radio",
        help="Select forecast time period. Data is pre-fetched for instant switching.",
    )

    # Update session state if changed
    if selected != st.session_state[SELECTED_TIME_STEP_KEY]:
        st.session_state[SELECTED_TIME_STEP_KEY] = selected

    return selected


def prefetch_all_forecasts(
    predictor,
    lat: float,
    lon: float,
    base_time: Optional[datetime] = None,
) -> dict:
    """Pre-fetch 7-day forecast data for all time steps.

    Fetches forecasts for all TIME_STEPS at once and caches them in
    st.session_state.all_forecasts for instant switching between time periods.

    Args:
        predictor: Predictor instance with predict(lat, lon, target_date, forecast_hours)
        lat: Latitude for forecasts
        lon: Longitude for forecasts
        base_time: Reference time for calculations. If None, uses datetime.now()

    Returns:
        Dict mapping time_step strings to forecast data:
        {
            "Now": {"forecast": ForecastResult, "confidence": ConfidenceInterval, "time": datetime},
            "Tonight": {...},
            ...
        }

    Side Effects:
        - Sets st.session_state.all_forecasts to the returned dict
        - Sets st.session_state.forecast_lat to lat
        - Sets st.session_state.forecast_lon to lon
    """
    if base_time is None:
        base_time = datetime.now()

    forecasts = {}

    for time_step in TIME_STEPS:
        try:
            forecast_time = get_forecast_time(time_step, base_time)
            forecast_hours = get_forecast_hours(time_step, base_time)

            # Call predictor
            forecast_result, confidence = predictor.predict(
                lat=lat,
                lon=lon,
                target_date=forecast_time,
                forecast_hours=forecast_hours,
            )

            forecasts[time_step] = {
                "forecast": forecast_result,
                "confidence": confidence,
                "time": forecast_time,
                "hours_ahead": forecast_hours,
            }

        except Exception as e:
            # Store error state for this time step
            forecasts[time_step] = {
                "forecast": None,
                "confidence": None,
                "time": get_forecast_time(time_step, base_time),
                "hours_ahead": get_forecast_hours(time_step, base_time),
                "error": str(e),
            }

    # Cache in session state
    st.session_state[ALL_FORECASTS_KEY] = forecasts
    st.session_state["forecast_lat"] = lat
    st.session_state["forecast_lon"] = lon
    st.session_state["forecast_base_time"] = base_time

    return forecasts


def get_cached_forecast(time_step: str) -> Optional[dict]:
    """Get cached forecast for a specific time step.

    Args:
        time_step: One of TIME_STEPS

    Returns:
        Forecast dict with keys: forecast, confidence, time, hours_ahead
        Returns None if no cached data exists
    """
    if ALL_FORECASTS_KEY not in st.session_state:
        return None

    all_forecasts = st.session_state[ALL_FORECASTS_KEY]
    return all_forecasts.get(time_step)


def get_current_forecast() -> Optional[dict]:
    """Get forecast for currently selected time step.

    Returns:
        Forecast dict for the selected time step, or None if not cached
    """
    if SELECTED_TIME_STEP_KEY not in st.session_state:
        return None

    selected = st.session_state[SELECTED_TIME_STEP_KEY]
    return get_cached_forecast(selected)


def needs_prefetch(lat: float, lon: float, tolerance: float = 0.001) -> bool:
    """Check if forecasts need to be pre-fetched for new location.

    Args:
        lat: Latitude to check
        lon: Longitude to check
        tolerance: Coordinate tolerance for considering location same

    Returns:
        True if prefetch is needed (no cache or different location)
    """
    if ALL_FORECASTS_KEY not in st.session_state:
        return True

    cached_lat = st.session_state.get("forecast_lat")
    cached_lon = st.session_state.get("forecast_lon")

    if cached_lat is None or cached_lon is None:
        return True

    # Check if location has changed
    if abs(lat - cached_lat) > tolerance or abs(lon - cached_lon) > tolerance:
        return True

    return False


def clear_forecast_cache() -> None:
    """Clear all cached forecasts from session state.

    Call this when the user changes location or when data needs refresh.
    """
    keys_to_clear = [
        ALL_FORECASTS_KEY,
        "forecast_lat",
        "forecast_lon",
        "forecast_base_time",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# Pure functions for testing (no Streamlit dependency)


def parse_time_step(time_step: str) -> tuple[int, int]:
    """Parse time step into days offset and hour.

    Args:
        time_step: One of TIME_STEPS

    Returns:
        Tuple of (days_offset, hour)

    Raises:
        ValueError: If time_step is invalid
    """
    if time_step not in TIME_STEPS:
        raise ValueError(f"Invalid time_step: {time_step}")

    if time_step == "Now":
        return (0, -1)  # -1 indicates use current hour
    elif time_step == "Tonight":
        return (0, 21)
    elif time_step == "Tomorrow AM":
        return (1, 9)
    elif time_step == "Tomorrow PM":
        return (1, 15)
    elif time_step.startswith("Day "):
        day_num = int(time_step.split()[1])
        return (day_num - 1, 12)

    raise ValueError(f"Invalid time_step: {time_step}")


def calculate_forecast_datetime(
    time_step: str,
    base_year: int,
    base_month: int,
    base_day: int,
    base_hour: int,
) -> tuple[int, int, int, int]:
    """Calculate forecast datetime components.

    Pure function for testing datetime calculations without datetime objects.

    Args:
        time_step: One of TIME_STEPS
        base_year: Base year
        base_month: Base month (1-12)
        base_day: Base day (1-31)
        base_hour: Base hour (0-23)

    Returns:
        Tuple of (year, month, day, hour)
    """
    days_offset, target_hour = parse_time_step(time_step)

    # Use datetime for actual calculation
    base = datetime(base_year, base_month, base_day, base_hour)

    if target_hour == -1:
        # "Now" - use base time
        result = base
    else:
        # Calculate target date
        target_date = base.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date = target_date + timedelta(days=days_offset)
        result = target_date.replace(hour=target_hour)

    return (result.year, result.month, result.day, result.hour)
