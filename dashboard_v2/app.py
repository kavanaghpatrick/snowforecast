"""
Snow Forecast Dashboard v2 - Multi-region ski resort forecasts.

Reads forecast data from JSON and displays an interactive table with filtering.
Supports USA, Austria, and Switzerland with regional snow depth sensors.
"""

import streamlit as st
import pandas as pd
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

st.set_page_config(
    page_title="Snow Forecast",
    page_icon="❄️",
    layout="wide"
)

# GitHub raw URL for live data (bypasses Streamlit Cloud's file caching)
GITHUB_DATA_URL = "https://raw.githubusercontent.com/kavanaghpatrick/snowforecast/main/data/forecasts_v2.json"
# Local fallback
DATA_PATH = Path(__file__).parent.parent / "data" / "forecasts_v2.json"


@st.cache_data(ttl=300)  # Cache for 5 minutes, then fetch fresh
def load_data() -> dict | None:
    """Load forecast JSON from GitHub (live) or local file (fallback)."""
    # Try GitHub first for latest data
    try:
        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        pass  # Fall back to local file

    # Local fallback
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
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
    st.title("❄️ Snow Forecast")

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
        station_name = resort.get("base_depth_source", "")

        # Handle both old (state) and new (country/region) formats
        country = resort.get("country", "USA")
        region = resort.get("region", resort.get("state", "??"))

        rows.append({
            "Resort": resort.get("name", "Unknown"),
            "Country": country,
            "Region": region,
            "Elevation (m)": resort.get("elevation_m", 0),
            "Base (cm)": base_depth if base_depth else None,
            "Station": station_name if station_name else "N/A",
            "7-Day Snow (cm)": round(seven_day_total, 1),
            "Peak Snowfall": best_day,
        })

    df = pd.DataFrame(rows)

    # Country and region filters
    st.subheader("Resort Forecasts")

    col1, col2 = st.columns(2)
    with col1:
        countries = ["All Countries"] + sorted(df["Country"].unique().tolist())
        selected_country = st.selectbox("Filter by Country", countries)

    with col2:
        if selected_country != "All Countries":
            available_regions = sorted(df[df["Country"] == selected_country]["Region"].unique().tolist())
        else:
            available_regions = sorted(df["Region"].unique().tolist())
        regions = ["All Regions"] + available_regions
        selected_region = st.selectbox("Filter by Region", regions)

    # Apply filters
    df_filtered = df
    if selected_country != "All Countries":
        df_filtered = df_filtered[df_filtered["Country"] == selected_country]
    if selected_region != "All Regions":
        df_filtered = df_filtered[df_filtered["Region"] == selected_region]

    # Display count
    st.caption(f"Showing {len(df_filtered)} of {len(df)} resorts")

    # Display sortable table
    st.dataframe(
        df_filtered.sort_values("7-Day Snow (cm)", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Resort": st.column_config.TextColumn("Resort", width="medium"),
            "Country": st.column_config.TextColumn("Country", width="small"),
            "Region": st.column_config.TextColumn("Region", width="small"),
            "Elevation (m)": st.column_config.NumberColumn("Elev (m)", format="%d"),
            "Base (cm)": st.column_config.NumberColumn("Base (cm)", format="%.0f"),
            "Station": st.column_config.TextColumn("Station", width="medium",
                help="Nearby snow monitoring station used for base depth measurement"),
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
        "Data sources: USA - SNOTEL (nrcs.usda.gov), Austria - GeoSphere TAWES (geosphere.at), "
        "Switzerland - SLF IMIS (slf.ch), Forecasts - Open-Meteo (open-meteo.com). "
        "Base depths are from nearby mountain monitoring stations, not at the resort. "
        "Always check official resort reports before traveling."
    )


if __name__ == "__main__":
    main()
