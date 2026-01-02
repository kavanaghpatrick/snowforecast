"""Resort detail panel component with forecast summary and visualizations.

This module provides an enhanced resort detail view with:
- Natural language forecast summaries
- Color-coded 7-day forecast table with AM/PM/Night blocks
- Confidence intervals display
- Snow depth trend chart
"""


import pandas as pd

from snowforecast.visualization import snow_depth_category, snow_depth_to_hex

# Day name mappings
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Snow intensity thresholds (cm per day)
LIGHT_SNOW_THRESHOLD = 5
MODERATE_SNOW_THRESHOLD = 15
HEAVY_SNOW_THRESHOLD = 25


def _get_day_name(date_obj) -> str:
    """Get day name from a date object or datetime."""
    if isinstance(date_obj, str):
        date_obj = pd.to_datetime(date_obj)
    return DAY_NAMES[date_obj.weekday()]


def _classify_snow_intensity(new_snow_cm: float) -> str:
    """Classify snow intensity based on daily accumulation.

    Args:
        new_snow_cm: New snow amount in centimeters

    Returns:
        Intensity classification string
    """
    if new_snow_cm < LIGHT_SNOW_THRESHOLD:
        return "light"
    elif new_snow_cm < MODERATE_SNOW_THRESHOLD:
        return "moderate"
    elif new_snow_cm < HEAVY_SNOW_THRESHOLD:
        return "heavy"
    else:
        return "very heavy"


def _find_snow_events(forecasts: pd.DataFrame) -> list[dict]:
    """Identify distinct snow events from forecast data.

    A snow event is a consecutive period with >5cm new snow per day.

    Args:
        forecasts: DataFrame with 'date', 'new_snow_cm' columns

    Returns:
        List of event dicts with start_date, end_date, total_cm, intensity
    """
    if forecasts.empty or "new_snow_cm" not in forecasts.columns:
        return []

    events = []
    current_event = None

    for _, row in forecasts.iterrows():
        new_snow = row.get("new_snow_cm", 0)
        date = row.get("date")

        # Handle NaN/None values - treat as no snow
        try:
            if pd.isna(new_snow):
                new_snow = 0
        except (TypeError, ValueError):
            new_snow = 0

        if new_snow >= LIGHT_SNOW_THRESHOLD:
            if current_event is None:
                current_event = {
                    "start_date": date,
                    "end_date": date,
                    "total_cm": new_snow,
                    "daily_amounts": [new_snow],
                }
            else:
                current_event["end_date"] = date
                current_event["total_cm"] += new_snow
                current_event["daily_amounts"].append(new_snow)
        else:
            if current_event is not None:
                # Finalize the event
                max_daily = max(current_event["daily_amounts"])
                current_event["intensity"] = _classify_snow_intensity(max_daily)
                del current_event["daily_amounts"]
                events.append(current_event)
                current_event = None

    # Handle event at end of forecast period
    if current_event is not None:
        max_daily = max(current_event["daily_amounts"])
        current_event["intensity"] = _classify_snow_intensity(max_daily)
        del current_event["daily_amounts"]
        events.append(current_event)

    return events


def _format_date_range(start_date, end_date) -> str:
    """Format a date range for display.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Formatted string like "Tuesday-Wednesday" or "Tuesday"
    """
    start_day = _get_day_name(start_date)
    end_day = _get_day_name(end_date)

    if start_date == end_date or start_day == end_day:
        return start_day
    else:
        return f"{start_day}-{end_day}"


def _describe_conditions(forecasts: pd.DataFrame) -> str:
    """Describe general conditions based on forecast data.

    Args:
        forecasts: DataFrame with forecast data

    Returns:
        Condition description string
    """
    if forecasts.empty:
        return ""

    # Check average probability of snow
    avg_prob = forecasts.get("probability", pd.Series([0])).mean()

    # Check current snow depth
    if "snow_depth_cm" in forecasts.columns:
        current_depth = forecasts.iloc[0].get("snow_depth_cm", 0)
        if current_depth > 100:
            return "Excellent base conditions."
        elif current_depth > 60:
            return "Good base conditions."
        elif current_depth > 30:
            return "Fair base conditions."
        else:
            return "Limited base coverage."

    if avg_prob < 0.2:
        return "Dry conditions expected."
    elif avg_prob < 0.5:
        return "Mixed conditions expected."
    else:
        return "Active weather pattern."


def generate_forecast_summary(forecasts: pd.DataFrame) -> str:
    """Generate natural language forecast summary.

    Analyzes 7-day forecast data and generates human-readable summary.

    Args:
        forecasts: DataFrame with columns:
            - date: Date of forecast
            - new_snow_cm: Expected new snow in cm
            - snow_depth_cm: Predicted snow depth in cm
            - probability: Snow probability (0-1)
            - ci_lower, ci_upper: Confidence interval bounds

    Returns:
        Natural language summary string, e.g.:
        "Heavy snow expected Tuesday-Wednesday (15-25cm). Clearing Thursday."

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-15', periods=7),
        ...     'new_snow_cm': [0, 2, 20, 25, 3, 0, 0],
        ...     'snow_depth_cm': [100, 100, 120, 145, 148, 145, 142],
        ...     'probability': [0.1, 0.3, 0.9, 0.95, 0.4, 0.1, 0.1]
        ... })
        >>> summary = generate_forecast_summary(df)
        >>> 'snow' in summary.lower() or 'dry' in summary.lower()
        True
    """
    if forecasts is None or forecasts.empty:
        return "No forecast data available."

    # Find snow events
    events = _find_snow_events(forecasts)

    if not events:
        # No significant snow expected
        conditions = _describe_conditions(forecasts)
        total_expected = forecasts.get("new_snow_cm", pd.Series([0])).sum()

        if total_expected < 1:
            return f"Dry week ahead. {conditions}"
        else:
            return f"Light flurries possible ({total_expected:.0f}cm total). {conditions}"

    # Build summary from events
    parts = []

    for i, event in enumerate(events):
        date_range = _format_date_range(event["start_date"], event["end_date"])
        total_cm = event["total_cm"]
        intensity = event["intensity"]

        if intensity == "very heavy":
            desc = f"Very heavy snow expected {date_range} ({total_cm:.0f}cm)"
        elif intensity == "heavy":
            desc = f"Heavy snow expected {date_range} ({total_cm:.0f}cm)"
        elif intensity == "moderate":
            desc = f"Moderate snow {date_range} ({total_cm:.0f}cm)"
        else:
            desc = f"Light snow {date_range} ({total_cm:.0f}cm)"

        parts.append(desc)

    # Add overall conditions
    conditions = _describe_conditions(forecasts)

    summary = ". ".join(parts)
    if conditions:
        summary += f". {conditions}"

    return summary


def render_forecast_table(forecasts: pd.DataFrame, container=None) -> None:
    """Render 7-day forecast with color-coded snow amounts.

    Displays a table with:
    - Day name and date
    - Snow depth (color-coded by category)
    - New snow amount
    - Confidence intervals

    Args:
        forecasts: DataFrame with forecast data
        container: Streamlit container (st, column, etc.). If None, uses st.
    """
    import streamlit as st

    target = container if container is not None else st

    if forecasts is None or forecasts.empty:
        target.info("No forecast data available")
        return

    target.markdown("**7-Day Forecast**")

    # Build HTML table with color coding
    html_rows = []

    for _, row in forecasts.iterrows():
        date = row.get("date")
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            day_name = _get_day_name(date)
            date_str = date.strftime("%m/%d") if hasattr(date, "strftime") else str(date)[:5]
        else:
            day_name = "N/A"
            date_str = "N/A"

        snow_depth = row.get("snow_depth_cm", 0)
        new_snow = row.get("new_snow_cm", 0)
        ci_lower = row.get("ci_lower", 0)
        ci_upper = row.get("ci_upper", 0)
        probability = row.get("probability", 0)

        # Get color for snow depth
        depth_color = snow_depth_to_hex(snow_depth)
        snow_depth_category(snow_depth)

        # Build row HTML
        html_rows.append(f"""
            <tr>
                <td style="font-weight:bold;padding:8px;">{day_name}</td>
                <td style="padding:8px;color:#666;">{date_str}</td>
                <td style="padding:8px;background:{depth_color};text-align:center;border-radius:4px;">
                    {snow_depth:.0f}cm
                </td>
                <td style="padding:8px;text-align:center;">
                    {'+' if new_snow > 0 else ''}{new_snow:.1f}cm
                </td>
                <td style="padding:8px;text-align:center;color:#666;font-size:0.85em;">
                    {ci_lower:.0f}-{ci_upper:.0f}cm
                </td>
                <td style="padding:8px;text-align:center;">
                    {probability:.0%}
                </td>
            </tr>
        """)

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="border-bottom:2px solid #ddd;">
                <th style="padding:8px;text-align:left;">Day</th>
                <th style="padding:8px;text-align:left;">Date</th>
                <th style="padding:8px;text-align:center;">Base</th>
                <th style="padding:8px;text-align:center;">New</th>
                <th style="padding:8px;text-align:center;">CI (95%)</th>
                <th style="padding:8px;text-align:center;">Prob</th>
            </tr>
        </thead>
        <tbody>
            {''.join(html_rows)}
        </tbody>
    </table>
    """

    target.markdown(table_html, unsafe_allow_html=True)


def render_snow_chart(forecasts: pd.DataFrame, container=None) -> None:
    """Render snow depth trend chart.

    Args:
        forecasts: DataFrame with 'date' and 'snow_depth_cm' columns
        container: Streamlit container. If None, uses st.
    """
    import streamlit as st

    target = container if container is not None else st

    if forecasts is None or forecasts.empty:
        return

    if "date" not in forecasts.columns or "snow_depth_cm" not in forecasts.columns:
        return

    target.markdown("**Snow Depth Trend**")

    chart_data = forecasts[["date", "snow_depth_cm"]].copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"])
    chart_data = chart_data.set_index("date")
    chart_data.columns = ["Snow Depth (cm)"]

    target.line_chart(chart_data)


def render_resort_detail(
    resort: dict,
    forecasts: pd.DataFrame,
    container=None,
    show_chart: bool = True,
) -> None:
    """Render complete resort detail panel.

    Main component that combines:
    - Resort header with name, state, elevation
    - Natural language forecast summary
    - Color-coded forecast table
    - Snow depth trend chart

    Args:
        resort: Dict with resort info:
            - name: Resort name
            - state: State abbreviation or name
            - elevation: Elevation in meters
            - latitude, longitude: Coordinates (optional)
        forecasts: DataFrame with 7-day forecast data
        container: Streamlit container. If None, uses st.
        show_chart: Whether to show the snow depth chart

    Side Effects:
        Sets st.session_state.selected_resort to resort dict
    """
    import streamlit as st

    target = container if container is not None else st

    # Set session state
    st.session_state.selected_resort = resort

    # Header
    name = resort.get("name", "Unknown Resort")
    state = resort.get("state", "")
    elevation = resort.get("elevation", 0)

    target.subheader(f"{name}, {state}")

    # Elevation info
    if elevation > 0:
        elev_ft = elevation * 3.28084
        target.caption(f"Elevation: {elevation:.0f}m ({elev_ft:.0f}ft)")

    # Coordinates if available
    lat = resort.get("latitude")
    lon = resort.get("longitude")
    if lat is not None and lon is not None:
        target.caption(f"Location: {lat:.4f}N, {abs(lon):.4f}W")

    target.markdown("---")

    # Forecast summary
    summary = generate_forecast_summary(forecasts)
    target.markdown(f"**Forecast:** {summary}")

    target.markdown("")

    # Forecast table
    render_forecast_table(forecasts, target)

    target.markdown("")

    # Snow depth chart
    if show_chart:
        render_snow_chart(forecasts, target)
