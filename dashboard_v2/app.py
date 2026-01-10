"""
Snow Forecast Dashboard v2 - Multi-region ski resort forecasts.

Displays forecast data with optimized UX based on Gemini UI audit.
"""

import streamlit as st
import pandas as pd
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

st.set_page_config(
    page_title="Snow Forecast",
    page_icon="❄️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Compact header */
    .block-container { padding-top: 1rem; }

    /* Better table row visibility */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        font-size: 14px;
    }

    /* Mobile-friendly title */
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem !important; }
        .stDataFrame { font-size: 12px; }
    }

    /* Status badge styling */
    .status-fresh {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        display: inline-block;
    }
    .status-stale {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# GitHub raw URL for live data
GITHUB_DATA_URL = "https://raw.githubusercontent.com/kavanaghpatrick/snowforecast/main/data/forecasts_v2.json"
DATA_PATH = Path(__file__).parent.parent / "data" / "forecasts_v2.json"


@st.cache_data(ttl=300)
def load_data() -> dict | None:
    """Load forecast JSON from GitHub (live) or local file (fallback)."""
    try:
        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        pass

    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def format_time_ago(hours: float) -> str:
    """Convert hours to human-readable time ago string."""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} min ago" if minutes != 1 else "1 min ago"
    elif hours < 24:
        h = int(hours)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    else:
        days = int(hours / 24)
        return f"{days} day{'s' if days != 1 else ''} ago"


def format_date_human(iso_str: str) -> str:
    """Convert ISO date to human readable format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %I:%M %p").replace(" 0", " ").replace("AM", "am").replace("PM", "pm")
    except (ValueError, TypeError):
        return "Unknown"


def check_freshness(updated_str: str) -> tuple[bool, float]:
    """Check if data is stale (> 24 hours old)."""
    try:
        updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_old = (now - updated).total_seconds() / 3600
        return hours_old > 24, hours_old
    except (ValueError, TypeError):
        return True, float("inf")


def find_best_day(daily_snow: list[dict]) -> tuple[str, float]:
    """Find the day with the most snowfall. Returns (formatted_date, amount)."""
    if not daily_snow:
        return "—", 0

    best_day = max(daily_snow, key=lambda d: d.get("new_snow_cm", 0))
    amount = best_day.get("new_snow_cm", 0)

    if amount == 0:
        return "—", 0

    try:
        date = datetime.fromisoformat(best_day["date"])
        return date.strftime("%a %d"), amount
    except (ValueError, KeyError):
        return best_day.get("date", "—"), amount


def get_snow_color(value: float, max_val: float) -> str:
    """Get background color based on snow amount (blue gradient)."""
    if value <= 0 or max_val <= 0:
        return ""
    ratio = min(value / max_val, 1.0)
    # Light blue to deep blue gradient
    r = int(240 - (ratio * 100))
    g = int(248 - (ratio * 80))
    b = int(255 - (ratio * 20))
    return f"background-color: rgb({r}, {g}, {b})"


def main():
    # Compact header with inline status
    col_title, col_status = st.columns([3, 2])

    with col_title:
        st.markdown("# ❄️ Snow Forecast")

    # Load data
    data = load_data()

    if data is None:
        st.error("No forecast data available. Run the data fetcher to populate.")
        return

    # Check freshness and show compact status
    is_stale, hours = check_freshness(data.get("updated", ""))

    with col_status:
        if hours == float("inf"):
            st.markdown('<span class="status-stale">⚠️ Unknown freshness</span>', unsafe_allow_html=True)
        elif is_stale:
            st.markdown(f'<span class="status-stale">⚠️ {format_time_ago(hours)}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-fresh">✓ {format_time_ago(hours)}</span>', unsafe_allow_html=True)

    # Build dataframe from resorts
    resorts = data.get("resorts", [])
    if not resorts:
        st.warning("No resort data found.")
        return

    # Process resort data with optimized column order
    rows = []
    for resort in resorts:
        daily = resort.get("forecast", [])
        base_total = resort.get("seven_day_total_cm", sum(d.get("new_snow_cm", 0) for d in daily[:7]))
        summit_total = resort.get("summit_seven_day_total_cm", base_total)
        best_day, best_amount = find_best_day(daily[:7])

        # Extract daily snowfall array for sparkline (7 days)
        daily_snow = [d.get("new_snow_cm", 0) for d in daily[:7]]
        # Pad to 7 days if needed
        while len(daily_snow) < 7:
            daily_snow.append(0)

        base_depth = resort.get("base_depth_cm")
        station_name = resort.get("base_depth_source", "")

        # Combine location
        country = resort.get("country", "USA")
        region = resort.get("region", resort.get("state", ""))
        location = f"{region}, {country}" if region else country

        rows.append({
            "Resort": resort.get("name", "Unknown"),
            "Forecast": daily_snow,  # Sparkline data
            "Total": round(summit_total, 0),  # Numeric total for sorting
            "Best Day": best_day,
            "Snowpack": round(base_depth, 0) if base_depth else None,
            "Location": location,
            "_station": station_name,  # Hidden, for tooltip
            "_country": country,  # Hidden, for filtering
            "_region": region,  # Hidden, for filtering
        })

    df = pd.DataFrame(rows)

    # Filters in a more compact layout
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        countries = ["All"] + sorted(df["_country"].unique().tolist())
        selected_country = st.selectbox("Country", countries, label_visibility="collapsed")

    with col2:
        if selected_country != "All":
            available_regions = sorted(df[df["_country"] == selected_country]["_region"].unique().tolist())
        else:
            available_regions = sorted(df["_region"].unique().tolist())
        regions = ["All Regions"] + available_regions
        selected_region = st.selectbox("Region", regions, label_visibility="collapsed")

    with col3:
        st.caption(f"Showing {len(df)} resorts across 3 countries")

    # Apply filters
    df_display = df.copy()
    if selected_country != "All":
        df_display = df_display[df_display["_country"] == selected_country]
    if selected_region != "All Regions":
        df_display = df_display[df_display["_region"] == selected_region]

    # Sort by total snowfall
    df_display = df_display.sort_values("Total", ascending=False)

    # Calculate max for sparkline scaling (at least 10cm per day to show small amounts)
    all_daily = [v for row in df_display["Forecast"] for v in row]
    max_daily = max(max(all_daily) if all_daily else 10, 10)

    # Display table with sparkline visualization
    st.dataframe(
        df_display[["Resort", "Forecast", "Total", "Best Day", "Snowpack", "Location"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Resort": st.column_config.TextColumn("Resort", width="medium"),
            "Forecast": st.column_config.LineChartColumn(
                "7-Day Forecast",
                help="Daily snowfall forecast for next 7 days (cm). Line shows snow timing.",
                y_min=0,
                y_max=max_daily,
                width="medium",
            ),
            "Total": st.column_config.NumberColumn(
                "Total",
                help="Total forecasted snowfall (cm)",
                format="%.0f cm",
                width="small",
            ),
            "Best Day": st.column_config.TextColumn(
                "Peak",
                help="Day with highest forecasted snowfall",
                width="small"
            ),
            "Snowpack": st.column_config.NumberColumn(
                "Snowpack",
                help="Natural snow depth from nearby monitoring station (cm). USA: SNOTEL. Europe: GeoSphere/SLF.",
                format="%.0f cm",
                width="small",
            ),
            "Location": st.column_config.TextColumn("Location", width="medium"),
        },
        height=450,
    )

    # Compact chart section
    with st.expander("📊 Snowpack Comparison", expanded=False):
        chart_data = df_display.set_index("Resort")["Snowpack"].dropna().sort_values(ascending=False)
        if len(chart_data) > 15:
            chart_data = chart_data.head(15)
            st.caption("Top 15 resorts by snowpack")
        st.bar_chart(chart_data)

    # Compact footer
    st.divider()
    st.caption(
        "Snowpack: SNOTEL (USA), GeoSphere (Austria), SLF (Switzerland) - natural snow depth, excludes snowmaking. "
        "Forecasts: Open-Meteo. May differ from resort reports."
    )


if __name__ == "__main__":
    main()
