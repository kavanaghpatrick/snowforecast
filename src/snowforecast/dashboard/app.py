"""Streamlit dashboard for Snow Forecast visualization.

Run with: streamlit run src/snowforecast/dashboard/app.py
"""

import sys
from pathlib import Path

# Add src directory to path for Streamlit Cloud compatibility
# This ensures 'import snowforecast' works even if package isn't pip-installed
_src_path = Path(__file__).parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from datetime import date, datetime, timedelta

import pandas as pd

# Try to import streamlit, provide helpful error if missing
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

# Page configuration
st.set_page_config(
    page_title="Snow Forecast Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Western US Ski Areas with coordinates
SKI_AREAS = {
    # Washington
    "Stevens Pass": (47.7448, -121.0890, "Washington", 1241),
    "Crystal Mountain": (46.9282, -121.5045, "Washington", 1341),
    "Mt. Baker": (48.8570, -121.6695, "Washington", 1280),
    "Snoqualmie Pass": (47.4204, -121.4138, "Washington", 921),
    # Oregon
    "Mt. Hood Meadows": (45.3311, -121.6647, "Oregon", 1524),
    "Mt. Bachelor": (43.9792, -121.6886, "Oregon", 1920),
    "Timberline": (45.3311, -121.7110, "Oregon", 1829),
    # California
    "Mammoth Mountain": (37.6308, -119.0326, "California", 2424),
    "Squaw Valley": (39.1969, -120.2358, "California", 1890),
    "Heavenly": (38.9353, -119.9400, "California", 2001),
    "Kirkwood": (38.6850, -120.0652, "California", 2377),
    # Colorado
    "Vail": (39.6403, -106.3742, "Colorado", 2476),
    "Breckenridge": (39.4817, -106.0384, "Colorado", 2926),
    "Aspen Snowmass": (39.2084, -106.9490, "Colorado", 2473),
    "Telluride": (37.9375, -107.8123, "Colorado", 2659),
    # Utah
    "Park City": (40.6514, -111.5080, "Utah", 2103),
    "Snowbird": (40.5830, -111.6538, "Utah", 2365),
    "Alta": (40.5884, -111.6386, "Utah", 2600),
    # Montana
    "Big Sky": (45.2618, -111.4018, "Montana", 2072),
    "Whitefish": (48.4820, -114.3556, "Montana", 1463),
    # Wyoming
    "Jackson Hole": (43.5875, -110.8279, "Wyoming", 1924),
    # Idaho
    "Sun Valley": (43.6804, -114.4075, "Idaho", 1752),
}


@st.cache_resource
def get_predictor():
    """Get cached predictor instance.

    Uses CachedPredictor with DuckDB backend for persistent caching.
    HRRR forecasts cached for 2 hours, terrain data cached permanently.
    """
    try:
        from snowforecast.cache import CachedPredictor
        return CachedPredictor()
    except ImportError as e:
        st.error(f"Could not load CachedPredictor: {e}")
        # Fall back to RealPredictor if cache module not available
        try:
            from snowforecast.api.predictor import RealPredictor
            st.warning("Using RealPredictor (cache module not available)")
            return RealPredictor()
        except ImportError:
            st.error("No predictor available")
            return None


def get_cache_status() -> dict:
    """Get cache status information for display.

    Returns:
        Dict with is_fresh, freshness_label, last_updated, forecast_count, terrain_count
    """
    predictor = get_predictor()
    if predictor is None or not hasattr(predictor, 'get_cache_stats'):
        return {
            "is_fresh": False,
            "freshness_label": "Unknown",
            "freshness_color": "gray",
            "last_updated": None,
            "forecast_count": 0,
            "terrain_count": 0,
        }

    stats = predictor.get_cache_stats()
    latest_run = stats.get("latest_run_time")

    if latest_run is None:
        return {
            "is_fresh": False,
            "freshness_label": "No Data",
            "freshness_color": "gray",
            "last_updated": None,
            "forecast_count": stats.get("forecast_count", 0),
            "terrain_count": stats.get("terrain_count", 0),
        }

    # Calculate age - ensure we're comparing with current UTC time
    now = datetime.utcnow()
    # Handle both datetime and pd.Timestamp
    if hasattr(latest_run, 'to_pydatetime'):
        latest_run = latest_run.to_pydatetime()
    if latest_run.tzinfo is not None:
        latest_run = latest_run.replace(tzinfo=None)

    age_hours = (now - latest_run).total_seconds() / 3600

    # Determine freshness status
    if age_hours < 1:
        freshness_label = "Fresh"
        freshness_color = "green"
        is_fresh = True
    elif age_hours < 2:
        freshness_label = "Recent"
        freshness_color = "orange"
        is_fresh = True
    else:
        freshness_label = "Stale"
        freshness_color = "red"
        is_fresh = False

    return {
        "is_fresh": is_fresh,
        "freshness_label": freshness_label,
        "freshness_color": freshness_color,
        "last_updated": latest_run,
        "age_hours": round(age_hours, 1),
        "forecast_count": stats.get("forecast_count", 0),
        "terrain_count": stats.get("terrain_count", 0),
    }


def render_cache_status_indicator():
    """Render the cache status indicator in the sidebar."""
    status = get_cache_status()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cache Status**")

    # Freshness indicator with colored emoji
    if status["freshness_color"] == "green":
        freshness_icon = ":green_circle:"
    elif status["freshness_color"] == "orange":
        freshness_icon = ":orange_circle:"
    elif status["freshness_color"] == "red":
        freshness_icon = ":red_circle:"
    else:
        freshness_icon = ":white_circle:"

    st.sidebar.markdown(f"{freshness_icon} **{status['freshness_label']}**")

    # Last updated timestamp
    if status["last_updated"]:
        # Convert to local time for display
        last_updated_str = status["last_updated"].strftime("%Y-%m-%d %H:%M UTC")
        st.sidebar.caption(f"Last updated: {last_updated_str}")
        if "age_hours" in status:
            if status["age_hours"] < 1:
                age_str = f"{int(status['age_hours'] * 60)} min ago"
            else:
                age_str = f"{status['age_hours']:.1f} hrs ago"
            st.sidebar.caption(f"({age_str})")
    else:
        st.sidebar.caption("No cached data yet")

    # Cache stats
    st.sidebar.caption(
        f"Forecasts: {status['forecast_count']} | Terrain: {status['terrain_count']}"
    )


def fetch_ski_area_forecast(
    name: str, lat: float, lon: float, days: int = 7, progress_container=None
) -> pd.DataFrame:
    """Fetch forecast for a ski area.

    Args:
        name: Ski area name
        lat: Latitude
        lon: Longitude
        days: Number of days to forecast
        progress_container: Optional Streamlit container for progress

    Returns:
        DataFrame with daily forecasts
    """
    predictor = get_predictor()
    if predictor is None:
        return pd.DataFrame()

    records = []
    today = date.today()

    # Get terrain once
    terrain = predictor.get_terrain_features(lat, lon)

    for day in range(days):
        if progress_container:
            progress = (day + 1) / days
            day_name = "Today" if day == 0 else f"Day +{day}"
            progress_container.progress(progress, text=f"Fetching {day_name} forecast...")

        target_date = today + timedelta(days=day)
        try:
            forecast, confidence = predictor.predict(
                lat, lon,
                datetime.combine(target_date, datetime.min.time()),
                forecast_hours=24
            )
            records.append({
                "date": target_date,
                "day": day,
                "snow_depth_cm": forecast.snow_depth_cm,
                "new_snow_cm": forecast.new_snow_cm,
                "probability": forecast.snowfall_probability,
                "ci_lower": confidence.lower,
                "ci_upper": confidence.upper,
            })
        except Exception as e:
            st.warning(f"Forecast error for day {day}: {e}")

    if progress_container:
        progress_container.empty()

    df = pd.DataFrame(records)
    df["ski_area"] = name
    df["latitude"] = lat
    df["longitude"] = lon
    df["elevation"] = terrain.get("elevation", 2000)
    df["slope"] = terrain.get("slope", 15)
    df["aspect"] = terrain.get("aspect", 180)

    return df


def fetch_all_current_conditions(progress_container=None) -> pd.DataFrame:
    """Fetch current conditions for all ski areas.

    Args:
        progress_container: Optional Streamlit container for progress updates
    """
    predictor = get_predictor()
    if predictor is None:
        return pd.DataFrame()

    records = []
    today = datetime.now()
    ski_areas_list = list(SKI_AREAS.items())
    total = len(ski_areas_list)

    for i, (name, (lat, lon, state, base_elev)) in enumerate(ski_areas_list):
        # Update progress
        if progress_container:
            progress = (i + 1) / total
            progress_container.progress(progress, text=f"Fetching {name} ({i+1}/{total})...")

        try:
            terrain = predictor.get_terrain_features(lat, lon)
            forecast, _ = predictor.predict(lat, lon, today, forecast_hours=24)

            records.append({
                "ski_area": name,
                "state": state,
                "latitude": lat,
                "longitude": lon,
                "elevation": terrain.get("elevation", base_elev),
                "snow_depth_cm": forecast.snow_depth_cm,
                "new_snow_cm": forecast.new_snow_cm,
                "probability": forecast.snowfall_probability,
            })
        except Exception:
            # Use fallback values on error
            records.append({
                "ski_area": name,
                "state": state,
                "latitude": lat,
                "longitude": lon,
                "elevation": base_elev,
                "snow_depth_cm": 0,
                "new_snow_cm": 0,
                "probability": 0,
            })

    if progress_container:
        progress_container.empty()

    return pd.DataFrame(records)


def create_current_conditions_cards(df: pd.DataFrame, selected_area: str):
    """Display current conditions for selected ski area."""
    if selected_area not in df["ski_area"].values:
        return

    row = df[df["ski_area"] == selected_area].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Snow Base",
            value=f"{row['snow_depth_cm']:.0f} cm",
            delta=f"{row['snow_depth_cm']/2.54:.0f} inches",
        )

    with col2:
        st.metric(
            label="24hr New Snow",
            value=f"{row['new_snow_cm']:.1f} cm",
            delta=f"{row['new_snow_cm']/2.54:.1f} inches",
        )

    with col3:
        st.metric(
            label="Snow Probability",
            value=f"{row['probability']:.0%}",
        )

    with col4:
        st.metric(
            label="Elevation",
            value=f"{row['elevation']:.0f} m",
            delta=f"{row['elevation']*3.28084:.0f} ft",
        )


def create_forecast_chart(forecast_df: pd.DataFrame):
    """Create 7-day forecast chart."""
    if forecast_df.empty:
        st.info("No forecast data available")
        return

    st.subheader("7-Day Snow Forecast")

    # Prepare data for chart
    chart_data = forecast_df[["date", "snow_depth_cm", "new_snow_cm"]].copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"])
    chart_data = chart_data.set_index("date")
    chart_data.columns = ["Base Depth (cm)", "New Snow (cm)"]

    # Show new snow as bar chart
    st.bar_chart(chart_data["New Snow (cm)"])

    # Show base depth as line
    st.line_chart(chart_data["Base Depth (cm)"])


def create_forecast_table(forecast_df: pd.DataFrame):
    """Create detailed forecast table."""
    if forecast_df.empty:
        return

    st.subheader("Detailed Forecast")

    display_df = forecast_df[["date", "snow_depth_cm", "new_snow_cm", "probability", "ci_lower", "ci_upper"]].copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%a %m/%d")
    display_df.columns = ["Date", "Base (cm)", "New (cm)", "Prob", "CI Low", "CI High"]

    # Format columns
    display_df["Base (cm)"] = display_df["Base (cm)"].apply(lambda x: f"{x:.1f}")
    display_df["New (cm)"] = display_df["New (cm)"].apply(lambda x: f"{x:.1f}")
    display_df["Prob"] = display_df["Prob"].apply(lambda x: f"{x:.0%}")
    display_df["CI Low"] = display_df["CI Low"].apply(lambda x: f"{x:.1f}")
    display_df["CI High"] = display_df["CI High"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def create_regional_map(conditions_df: pd.DataFrame, selected_area: str = None):
    """Create map of all ski areas with current conditions."""
    st.subheader("Regional Snow Conditions")

    if conditions_df.empty:
        st.info("Loading conditions...")
        return

    # Prepare map data
    map_data = conditions_df[["latitude", "longitude", "snow_depth_cm", "ski_area"]].copy()
    map_data.columns = ["lat", "lon", "size", "name"]

    # Scale size for visibility (min 50, max 500)
    map_data["size"] = map_data["size"].clip(lower=10) * 3

    st.map(map_data, size="size", color="#1E90FF")

    st.caption("Circle size indicates snow depth")


def create_regional_table(conditions_df: pd.DataFrame):
    """Create sortable table of all ski areas."""
    st.subheader("All Ski Areas")

    if conditions_df.empty:
        return

    display_df = conditions_df[["ski_area", "state", "elevation", "snow_depth_cm", "new_snow_cm"]].copy()
    display_df.columns = ["Ski Area", "State", "Elevation (m)", "Base (cm)", "New 24hr (cm)"]
    display_df = display_df.sort_values("Base (cm)", ascending=False)

    # Format
    display_df["Base (cm)"] = display_df["Base (cm)"].apply(lambda x: f"{x:.0f}")
    display_df["New 24hr (cm)"] = display_df["New 24hr (cm)"].apply(lambda x: f"{x:.1f}")
    display_df["Elevation (m)"] = display_df["Elevation (m)"].apply(lambda x: f"{x:.0f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def create_sidebar() -> tuple:
    """Create sidebar with ski area selector.

    Returns:
        Tuple of (selected_area, selected_state)
    """
    st.sidebar.header("Select Location")

    # State filter
    states = ["All States"] + sorted(set(info[2] for info in SKI_AREAS.values()))
    selected_state = st.sidebar.selectbox("State", states)

    # Filter ski areas by state
    if selected_state != "All States":
        available_areas = [name for name, info in SKI_AREAS.items() if info[2] == selected_state]
    else:
        available_areas = list(SKI_AREAS.keys())

    selected_area = st.sidebar.selectbox("Ski Area", sorted(available_areas))

    # Show location info
    if selected_area:
        lat, lon, state, elev = SKI_AREAS[selected_area]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{selected_area}**")
        st.sidebar.markdown(f"📍 {lat:.4f}°N, {abs(lon):.4f}°W")
        st.sidebar.markdown(f"🏔️ Base: {elev}m ({elev*3.28084:.0f}ft)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown("- NOAA HRRR (3km)")
    st.sidebar.markdown("- Copernicus DEM (30m)")

    # Add cache status indicator
    render_cache_status_indicator()

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Data"):
        # Clear all cached data from session state
        for key in list(st.session_state.keys()):
            if key.startswith("conditions_") or key.startswith("forecast_") or key.startswith("single_"):
                del st.session_state[key]
        st.rerun()

    return selected_area, selected_state


def fetch_single_resort(name: str) -> dict:
    """Fetch conditions for a single resort (fast path)."""
    predictor = get_predictor()
    if predictor is None:
        return None

    lat, lon, state, base_elev = SKI_AREAS[name]
    today = datetime.now()

    try:
        terrain = predictor.get_terrain_features(lat, lon)
        forecast, _ = predictor.predict(lat, lon, today, forecast_hours=24)
        return {
            "ski_area": name,
            "state": state,
            "latitude": lat,
            "longitude": lon,
            "elevation": terrain.get("elevation", base_elev),
            "snow_depth_cm": forecast.snow_depth_cm,
            "new_snow_cm": forecast.new_snow_cm,
            "probability": forecast.snowfall_probability,
        }
    except Exception:
        return {
            "ski_area": name,
            "state": state,
            "latitude": lat,
            "longitude": lon,
            "elevation": base_elev,
            "snow_depth_cm": 0,
            "new_snow_cm": 0,
            "probability": 0,
        }


def main():
    """Main dashboard function."""
    st.title("❄️ Snow Forecast Dashboard")
    st.markdown("Real-time snow conditions from NOAA HRRR + Copernicus DEM")

    # Sidebar
    selected_area, selected_state = create_sidebar()

    # Fast path: fetch only selected resort first
    single_cache_key = f"single_{selected_area}_{datetime.now().strftime('%Y%m%d%H')}"
    if single_cache_key not in st.session_state:
        with st.spinner(f"Fetching {selected_area} from NOAA HRRR..."):
            result = fetch_single_resort(selected_area)
            st.session_state[single_cache_key] = result

    selected_data = st.session_state.get(single_cache_key)

    # Show selected resort conditions immediately (from single fetch)
    if selected_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Snow Base", f"{selected_data['snow_depth_cm']:.0f} cm",
                     f"{selected_data['snow_depth_cm']/2.54:.0f} inches")
        with col2:
            st.metric("24hr New Snow", f"{selected_data['new_snow_cm']:.1f} cm",
                     f"{selected_data['new_snow_cm']/2.54:.1f} inches")
        with col3:
            st.metric("Snow Probability", f"{selected_data['probability']:.0%}")
        with col4:
            st.metric("Elevation", f"{selected_data['elevation']:.0f} m",
                     f"{selected_data['elevation']*3.28084:.0f} ft")

    st.markdown("---")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # 7-day forecast for selected resort
        if selected_area:
            lat, lon, _, _ = SKI_AREAS[selected_area]
            forecast_cache_key = f"forecast_{selected_area}_{datetime.now().strftime('%Y%m%d%H')}"
            if forecast_cache_key not in st.session_state:
                forecast_progress = st.empty()
                forecast_df = fetch_ski_area_forecast(
                    selected_area, lat, lon, days=7, progress_container=forecast_progress
                )
                st.session_state[forecast_cache_key] = forecast_df
            else:
                forecast_df = st.session_state[forecast_cache_key]
            create_forecast_chart(forecast_df)
            create_forecast_table(forecast_df)

    with col2:
        # Regional comparison - load lazily with expander
        with st.expander("🗺️ Compare All Resorts (loads 22 resorts)", expanded=False):
            cache_key = f"conditions_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key not in st.session_state:
                progress_bar = st.empty()
                st.caption("Loading all resort data from NOAA HRRR...")
                conditions_df = fetch_all_current_conditions(progress_bar)
                st.session_state[cache_key] = conditions_df
            else:
                conditions_df = st.session_state[cache_key]

            if not conditions_df.empty:
                create_regional_map(conditions_df, selected_area)
                create_regional_table(conditions_df)

    # Footer
    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )
    with col_f2:
        st.markdown(
            "*Forecasts beyond 48 hours use climatological estimates*"
        )


def run_dashboard():
    """Entry point for running dashboard."""
    main()


if __name__ == "__main__":
    main()
