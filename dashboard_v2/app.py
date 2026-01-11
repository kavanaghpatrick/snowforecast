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

# Custom CSS for polished styling
st.markdown("""
<style>
    /* ===== TYPOGRAPHY & BASE ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ===== LAYOUT ===== */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }

    /* ===== HEADER ===== */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #0f172a !important;
        font-size: 2rem !important;
    }

    /* Header row alignment */
    .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    /* ===== STATUS BADGES (Pill Style) ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .status-fresh {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #047857;
        border: 1px solid #a7f3d0;
        box-shadow: 0 1px 2px rgba(4, 120, 87, 0.08);
    }

    .status-stale {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        color: #b45309;
        border: 1px solid #fcd34d;
        box-shadow: 0 1px 2px rgba(180, 83, 9, 0.08);
    }

    /* ===== FILTERS ===== */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #94a3b8 !important;
    }

    .filter-caption {
        color: #64748b;
        font-size: 13px;
        padding-top: 8px;
    }

    /* ===== DATA TABLE ===== */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    [data-testid="stDataFrame"] th {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        font-weight: 600 !important;
        color: #334155 !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #e2e8f0 !important;
    }

    /* Alternating row colors */
    [data-testid="stDataFrame"] tr:nth-child(even) {
        background: #f8fafc !important;
    }

    [data-testid="stDataFrame"] tr:hover {
        background: #f1f5f9 !important;
    }

    /* Sparkline column emphasis */
    [data-testid="stDataFrame"] td {
        vertical-align: middle !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #334155 !important;
        background: #f8fafc !important;
        border-radius: 8px !important;
    }

    /* ===== FOOTER ===== */
    .footer-text {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.6;
    }

    /* ===== MOBILE RESPONSIVE ===== */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        .status-badge {
            font-size: 12px;
            padding: 5px 10px;
        }

        /* Stack filters on mobile */
        [data-testid="column"] {
            min-width: 100% !important;
        }

        [data-testid="stDataFrame"] {
            font-size: 12px;
        }

        /* Hide location column on very small screens */
        @media (max-width: 480px) {
            [data-testid="stDataFrame"] td:last-child,
            [data-testid="stDataFrame"] th:last-child {
                display: none;
            }
        }
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        st.write("")  # Vertical spacer to align with title
        if hours == float("inf"):
            st.markdown('<span class="status-badge status-stale">⚠️ Unknown</span>', unsafe_allow_html=True)
        elif is_stale:
            st.markdown(f'<span class="status-badge status-stale">⚠️ {format_time_ago(hours)}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge status-fresh">✓ {format_time_ago(hours)}</span>', unsafe_allow_html=True)

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

    # Filters with clean layout
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        countries = ["All Countries"] + sorted(df["_country"].unique().tolist())
        selected_country = st.selectbox("Country", countries, label_visibility="collapsed")

    with col2:
        if selected_country != "All Countries":
            available_regions = sorted(df[df["_country"] == selected_country]["_region"].unique().tolist())
        else:
            available_regions = sorted(df["_region"].unique().tolist())
        regions = ["All Regions"] + available_regions
        selected_region = st.selectbox("Region", regions, label_visibility="collapsed")

    with col3:
        st.markdown(
            f'<p class="filter-caption" style="text-align: right; margin: 0; padding-top: 10px;">'
            f'{len(df)} resorts · {df["_country"].nunique()} countries</p>',
            unsafe_allow_html=True
        )

    # Apply filters
    df_display = df.copy()
    if selected_country != "All Countries":
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

    # Footer with data sources
    st.markdown("---")
    st.markdown(
        '<p class="footer-text">'
        '<strong>Data sources:</strong> '
        'SNOTEL (USA) · GeoSphere (Austria) · SLF (Switzerland) · tenki.jp & OnTheSnow (Japan)<br>'
        'Forecasts via Open-Meteo. Natural snow depth only—excludes snowmaking. May differ from resort reports.'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
