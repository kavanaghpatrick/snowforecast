"""
Snow Forecast Dashboard v2 - Minimal Streamlit app for Western US ski resorts.

Reads forecast data from JSON and displays an interactive table with filtering.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

st.set_page_config(
    page_title="Snow Forecast",
    page_icon="❄️",
    layout="wide"
)

# Data path relative to this file's location
DATA_PATH = Path(__file__).parent.parent / "data" / "forecasts_v2.json"


def load_data() -> dict | None:
    """Load forecast JSON, return None if missing or invalid."""
    if not DATA_PATH.exists():
        return None
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def check_freshness(updated_str: str) -> tuple[bool, float]:
    """
    Check if data is stale (> 24 hours old).

    Args:
        updated_str: ISO format timestamp string

    Returns:
        Tuple of (is_stale, hours_old)
    """
    try:
        # Parse ISO format timestamp
        updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - updated
        hours_old = delta.total_seconds() / 3600
        is_stale = hours_old > 24
        return is_stale, hours_old
    except (ValueError, TypeError):
        # If we can't parse, assume stale
        return True, float("inf")


def find_best_day(daily_snow: list[dict]) -> str:
    """Find the day with the most snowfall from daily forecast data."""
    if not daily_snow:
        return "N/A"

    best_day = max(daily_snow, key=lambda d: d.get("new_snow_cm", 0))
    if best_day.get("new_snow_cm", 0) == 0:
        return "None expected"

    # Format the date nicely
    try:
        date = datetime.fromisoformat(best_day["date"])
        return date.strftime("%a %b %d")
    except (ValueError, KeyError):
        return best_day.get("date", "N/A")


def main():
    st.title("❄️ Snow Forecast - Western US")

    # Load data
    data = load_data()

    if data is None:
        st.error("No forecast data available. Please check that data/forecasts_v2.json exists.")
        st.info("Run the data fetcher to populate forecast data.")
        return

    # Check freshness and display status
    is_stale, hours = check_freshness(data.get("updated", ""))

    col1, col2 = st.columns([3, 1])
    with col1:
        if is_stale:
            if hours == float("inf"):
                st.warning("⚠️ Data freshness unknown - timestamp missing or invalid")
            else:
                st.warning(f"⚠️ Data is {hours:.1f} hours old - may be outdated")
        else:
            st.success(f"✅ Updated {hours:.1f} hours ago")

    with col2:
        st.caption(f"Last update: {data.get('updated', 'Unknown')[:19]}")

    # Build dataframe from resorts
    resorts = data.get("resorts", [])
    if not resorts:
        st.warning("No resort data found in forecast file.")
        return

    # Process resort data
    rows = []
    for resort in resorts:
        daily = resort.get("forecast", [])
        # Use pre-computed total, or sum from forecast data
        seven_day_total = resort.get("seven_day_total_cm", sum(d.get("new_snow_cm", 0) for d in daily[:7]))
        best_day = find_best_day(daily[:7])

        base_depth = resort.get("base_depth_cm")
        snotel_station = resort.get("base_depth_source", "")

        rows.append({
            "Resort": resort.get("name", "Unknown"),
            "State": resort.get("state", "??"),
            "Elevation (m)": resort.get("elevation_m", 0),
            "Base (cm)": base_depth if base_depth else None,
            "SNOTEL Station": snotel_station if snotel_station else "N/A",
            "7-Day Snow (cm)": round(seven_day_total, 1),
            "Peak Snowfall": best_day,
        })

    df = pd.DataFrame(rows)

    # State filter
    st.subheader("Resort Forecasts")

    states = ["All States"] + sorted(df["State"].unique().tolist())
    selected_state = st.selectbox("Filter by State", states)

    if selected_state != "All States":
        df_filtered = df[df["State"] == selected_state]
    else:
        df_filtered = df

    # Display count
    st.caption(f"Showing {len(df_filtered)} of {len(df)} resorts")

    # Display sortable table
    st.dataframe(
        df_filtered.sort_values("7-Day Snow (cm)", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Resort": st.column_config.TextColumn("Resort", width="medium"),
            "State": st.column_config.TextColumn("State", width="small"),
            "Elevation (m)": st.column_config.NumberColumn("Elev (m)", format="%d"),
            "Base (cm)": st.column_config.NumberColumn("Base (cm)", format="%.0f"),
            "SNOTEL Station": st.column_config.TextColumn("SNOTEL Station", width="medium",
                help="Nearby SNOTEL station used for base depth measurement"),
            "7-Day Snow (cm)": st.column_config.NumberColumn("7-Day (cm)", format="%.1f"),
            "Peak Snowfall": st.column_config.TextColumn("Peak Snowfall", width="small",
                help="Day with highest forecasted snowfall in the next 7 days"),
        }
    )

    # Bar chart of base depths
    st.subheader("Base Depth by Resort")

    chart_data = df_filtered.set_index("Resort")["Base (cm)"].dropna().sort_values(ascending=False)
    if len(chart_data) > 20:
        st.caption("Showing top 20 resorts by base depth")
        chart_data = chart_data.head(20)

    st.bar_chart(chart_data)

    # Footer
    st.divider()
    st.caption(
        "Data sources: SNOTEL (base depth from nearby stations), Open-Meteo (7-day forecasts). "
        "Base depths are from the nearest SNOTEL station, not necessarily at the resort. "
        "Always check official resort reports before traveling."
    )


if __name__ == "__main__":
    main()
