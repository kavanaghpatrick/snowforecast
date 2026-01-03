"""Streamlit dashboard for Snow Forecast visualization.

Phase 1: Core Layout - Interactive map, time selector, detail panel.
Phase 2: Enhanced Metrics - Snow quality, confidence, elevation bands, SNOTEL.
Phase 3: Polish & Mobile - Favorites, responsive layout, cache status, error handling.
Phase 4: Advanced Viz - 3D terrain, forecast overlay heatmap.
Run with: streamlit run src/snowforecast/dashboard/app.py
"""

import sys
from pathlib import Path

# Add src directory to path for Streamlit Cloud compatibility
_app_file = Path(__file__).resolve()
_src_path = _app_file.parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

# Import dashboard components
from snowforecast.dashboard.components import (
    # Map
    render_resort_map,
    # Time selector
    render_time_selector,
    get_current_forecast,
    clear_forecast_cache,
    # Resort detail
    generate_forecast_summary,
    render_forecast_table,
    # Loading states
    render_loading_skeleton,
    render_empty_state,
    # Cache status
    render_cache_status_badge,
    should_show_stale_warning,
    render_data_warning,
    # Phase 2: Snow Quality
    create_quality_metrics,
    render_snow_quality_badge,
    # Phase 2: Confidence
    render_confidence_badge,
    render_confidence_explanation,
    # Phase 2: Elevation Bands
    render_elevation_bands,
    # Phase 2: SNOTEL
    render_snotel_section,
    # Phase 3: Responsive
    inject_responsive_css,
    get_breakpoint,
    # Phase 3: Favorites
    render_favorite_toggle,
    get_favorites,
    # Phase 4: 3D Terrain
    create_terrain_deck,
    create_resort_layer,
    render_terrain_controls,
    # Phase 4: Forecast Overlay
    create_forecast_overlay,
    render_overlay_toggle,
)
# Phase 2: Elevation bands computation
from snowforecast.cache.elevation_bands import compute_elevation_bands, get_summit_elevation

# Page configuration
st.set_page_config(
    page_title="Snow Forecast Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Phase 3: Inject responsive CSS for mobile/tablet support
inject_responsive_css()

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
    """Get cached predictor instance."""
    try:
        from snowforecast.cache import CachedPredictor
        return CachedPredictor()
    except ImportError as e:
        st.error(f"Could not load CachedPredictor: {e}")
        return None


@st.cache_data(ttl=3600, max_entries=5, show_spinner="Loading resort conditions...")
def fetch_all_conditions() -> pd.DataFrame:
    """Fetch current conditions for all ski areas (cached 1 hour).

    Memory optimization: max_entries=5 limits cache to 5 versions.
    """
    predictor = get_predictor()
    if predictor is None:
        return pd.DataFrame()

    records = []
    today = datetime.now(timezone.utc).replace(tzinfo=None)

    for name, (lat, lon, state, base_elev) in SKI_AREAS.items():
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

    df = pd.DataFrame(records)
    # Optimize memory: use float32 instead of float64
    for col in ["latitude", "longitude", "elevation", "snow_depth_cm", "new_snow_cm", "probability"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["state"] = df["state"].astype("category")
    return df


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading 7-day forecast...")
def fetch_resort_7day_forecast(name: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch 7-day forecast for a resort (cached 1 hour).

    Memory optimization: max_entries=10 limits to 10 resort forecasts.
    """
    predictor = get_predictor()
    if predictor is None:
        return pd.DataFrame()

    records = []
    today = date.today()
    terrain = predictor.get_terrain_features(lat, lon)

    for day in range(7):
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
        except Exception:
            records.append({
                "date": target_date,
                "day": day,
                "snow_depth_cm": 0,
                "new_snow_cm": 0,
                "probability": 0,
                "ci_lower": 0,
                "ci_upper": 0,
            })

    df = pd.DataFrame(records)
    df["ski_area"] = name
    df["elevation"] = terrain.get("elevation", 2000)
    # Optimize memory: use float32 instead of float64
    for col in ["snow_depth_cm", "new_snow_cm", "probability", "ci_lower", "ci_upper", "elevation"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["day"] = df["day"].astype("int8")
    return df


def create_sidebar() -> tuple[str, str]:
    """Create sidebar with resort selector and data info."""
    st.sidebar.header("Select Location")

    # Phase 3: Get favorites for quick access
    favorites = get_favorites()

    # State filter
    states = ["All States"] + sorted(set(info[2] for info in SKI_AREAS.values()))
    selected_state = st.sidebar.selectbox("State", states)

    # Filter ski areas by state
    if selected_state != "All States":
        available_areas = [name for name, info in SKI_AREAS.items() if info[2] == selected_state]
    else:
        available_areas = list(SKI_AREAS.keys())

    # Phase 3: Show favorites at top of list if any exist
    if favorites:
        # Move favorites to top of the list
        favorite_areas = [f for f in favorites if f in available_areas]
        other_areas = [a for a in available_areas if a not in favorites]
        sorted_areas = favorite_areas + sorted(other_areas)
    else:
        sorted_areas = sorted(available_areas)

    selected_area = st.sidebar.selectbox("Ski Area", sorted_areas)

    # Show location info
    if selected_area:
        lat, lon, state, elev = SKI_AREAS[selected_area]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{selected_area}**")
        st.sidebar.markdown(f"📍 {lat:.4f}°N, {abs(lon):.4f}°W")
        st.sidebar.markdown(f"🏔️ Base: {elev}m ({elev*3.28084:.0f}ft)")

        # Phase 3: Favorite toggle
        render_favorite_toggle(selected_area, container=st.sidebar)

    # Data sources
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown("- NOAA HRRR (3km)")
    st.sidebar.markdown("- Copernicus DEM (30m)")

    # Cache status
    predictor = get_predictor()
    if predictor:
        render_cache_status_badge(predictor, container=st.sidebar)
        # Phase 3: Stale data warning if cache is old
        if should_show_stale_warning(predictor):
            render_data_warning(
                "Cache data may be stale",
                suggestion="Click 'Refresh Data' for latest forecasts",
                container=st.sidebar,
            )

    # Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        clear_forecast_cache()
        st.cache_data.clear()  # Clear native Streamlit cache
        st.rerun()

    return selected_area, selected_state


def main():
    """Main dashboard function with all phases integrated."""
    # Phase 3: Check breakpoint for responsive layout
    breakpoint = get_breakpoint()
    is_mobile = breakpoint == "mobile"

    # Header
    st.title("❄️ Snow Forecast Dashboard")
    st.markdown("Real-time snow conditions from NOAA HRRR + Copernicus DEM")

    # Sidebar
    selected_area, selected_state = create_sidebar()

    # Safety check for empty selection
    if not selected_area or selected_area not in SKI_AREAS:
        st.warning("Please select a ski area from the sidebar.")
        return

    lat, lon, state, base_elev = SKI_AREAS[selected_area]

    # Time selector (horizontal)
    st.markdown("---")
    selected_time = render_time_selector()

    # Memory optimization: Removed aggressive prefetching of all 9 time steps.
    # Forecasts are now loaded on-demand via get_current_forecast() which
    # uses the DuckDB cache. This significantly reduces memory usage on
    # Streamlit Cloud (1GB limit).

    # Main content: Map | Detail Panel (stacked vertically on mobile)
    if is_mobile:
        # Mobile: Full width, stacked layout
        col_map = st.container()
        col_detail = st.container()
    else:
        # Desktop: Side by side columns
        col_map, col_detail = st.columns([1, 1])

    # Left column: Interactive Map
    with col_map:
        st.subheader("Regional Overview")

        # Phase 4: Map visualization controls
        if not is_mobile:
            ctrl_col1, ctrl_col2 = st.columns(2)
            with ctrl_col1:
                terrain_controls = render_terrain_controls(st)
            with ctrl_col2:
                show_overlay = render_overlay_toggle()
        else:
            # Simplified controls on mobile (3D can be heavy)
            terrain_controls = {"enabled_3d": False, "pitch": 0, "use_texture": True}
            show_overlay = False

        # Load all resort conditions (cached via @st.cache_data)
        conditions_df = fetch_all_conditions()

        # Render interactive PyDeck map
        if not conditions_df.empty:
            try:
                if terrain_controls["enabled_3d"]:
                    # Phase 4: 3D terrain mode
                    # Create resort marker layer to overlay on terrain
                    resort_layer = create_resort_layer(conditions_df)

                    # Build layers list
                    layers = [resort_layer]

                    # Add forecast overlay if enabled
                    if show_overlay:
                        # Create overlay from conditions data
                        overlay_df = conditions_df[["latitude", "longitude", "new_snow_cm"]].copy()
                        overlay_df.columns = ["lat", "lon", "new_snow_cm"]
                        overlay_layer = create_forecast_overlay(overlay_df, radius=20000, opacity=0.4)
                        layers.insert(0, overlay_layer)  # Insert below resort markers

                    # Create 3D terrain deck with resort markers
                    deck = create_terrain_deck(
                        lat=lat,
                        lon=lon,
                        zoom=6,
                        pitch=terrain_controls["pitch"],
                        bearing=0,
                        use_texture=terrain_controls["use_texture"],
                        additional_layers=layers,
                    )
                    st.pydeck_chart(deck, use_container_width=True)
                    st.caption("3D Terrain | Circle color = snow depth | Circle size = new snow")
                else:
                    # Standard 2D map mode
                    if show_overlay:
                        # Create map with overlay
                        from snowforecast.dashboard.components import create_base_view
                        import pydeck as pdk

                        overlay_df = conditions_df[["latitude", "longitude", "new_snow_cm"]].copy()
                        overlay_df.columns = ["lat", "lon", "new_snow_cm"]
                        overlay_layer = create_forecast_overlay(overlay_df, radius=20000, opacity=0.4)
                        resort_layer = create_resort_layer(conditions_df)

                        view = create_base_view(lat, lon, zoom=5)
                        deck = pdk.Deck(
                            layers=[overlay_layer, resort_layer],
                            initial_view_state=view,
                            map_style="mapbox://styles/mapbox/dark-v10",
                        )
                        st.pydeck_chart(deck, use_container_width=True)
                        st.caption("Forecast overlay | Circle color = snow depth | Circle size = new snow")
                    else:
                        # Standard map without overlay
                        deck = render_resort_map(conditions_df)
                        st.pydeck_chart(deck, use_container_width=True)
                        st.caption("Circle color = snow depth | Circle size = new snow")
            except Exception as e:
                st.error(f"Map error: {e}")
                # Fallback to basic map
                st.map(conditions_df[["latitude", "longitude"]])
        else:
            render_empty_state("No resort data available", suggestion="Check your connection")

    # Right column: Resort Detail Panel
    with col_detail:
        # Get current forecast for selected time step
        current_forecast = get_current_forecast()

        if current_forecast and current_forecast.get("forecast"):
            forecast = current_forecast["forecast"]
            confidence = current_forecast["confidence"]
            forecast_time = current_forecast["time"]

            # Resort info dict for detail panel
            resort_info = {
                "name": selected_area,
                "state": state,
                "elevation": base_elev,
                "latitude": lat,
                "longitude": lon,
            }

            # Display time context
            time_str = forecast_time.strftime("%A %I:%M %p") if forecast_time else "Now"
            st.caption(f"Forecast for: {time_str}")

            # Quick metrics row (Phase 1)
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Snow Base", f"{forecast.snow_depth_cm:.0f} cm",
                         f"{forecast.snow_depth_cm/2.54:.0f} in")
            with metric_cols[1]:
                st.metric("New Snow", f"{forecast.new_snow_cm:.1f} cm",
                         f"{forecast.new_snow_cm/2.54:.1f} in")
            with metric_cols[2]:
                st.metric("Probability", f"{forecast.snowfall_probability:.0%}")
            with metric_cols[3]:
                # Terrain elevation from predictor
                predictor = get_predictor()
                if predictor:
                    terrain = predictor.get_terrain_features(lat, lon)
                    elev = terrain.get("elevation", base_elev)
                else:
                    elev = base_elev
                st.metric("Elevation", f"{elev:.0f} m", f"{elev*3.28084:.0f} ft")

            # Phase 2: Snow Quality Badge
            try:
                # Estimate temperature from elevation (winter approximation)
                temp_c = -5.0 - (elev - 2000) * 0.0065
                quality_metrics = create_quality_metrics(
                    snow_depth_change=forecast.new_snow_cm,
                    precip_mm=forecast.new_snow_cm / 10.0,  # Approximate
                    temp_c=temp_c,
                )
                col_quality, col_confidence = st.columns(2)
                with col_quality:
                    render_snow_quality_badge(quality_metrics)
                with col_confidence:
                    render_confidence_badge(confidence, forecast.snowfall_probability)
                render_confidence_explanation()
            except Exception:
                pass  # Gracefully degrade if snow quality calc fails

            st.markdown("---")

            # Load 7-day forecast for table (cached via @st.cache_data)
            forecast_df = fetch_resort_7day_forecast(selected_area, lat, lon)

            # Natural language forecast summary
            if not forecast_df.empty:
                summary = generate_forecast_summary(forecast_df)
                st.markdown(f"**Forecast:** {summary}")
                st.markdown("")

            # Forecast table
            render_forecast_table(forecast_df)

            # Phase 2: Elevation Bands
            with st.expander("Forecast by Elevation", expanded=False):
                try:
                    predictor = get_predictor()
                    if predictor:
                        summit_elev = get_summit_elevation(selected_area, base_elev)
                        elev_result = compute_elevation_bands(
                            predictor=predictor,
                            lat=lat,
                            lon=lon,
                            base_elev_m=base_elev,
                            summit_elev_m=summit_elev,
                            ski_area_name=selected_area,
                        )
                        render_elevation_bands(elev_result)
                except Exception as e:
                    st.caption(f"Elevation data unavailable: {e}")

        else:
            # No forecast data - show loading or error
            render_loading_skeleton(height=300, skeleton_type="card")
            st.caption("Loading forecast data...")

    # Bottom section: Tabs for additional views (Phase 2: added SNOTEL)
    st.markdown("---")
    tab_chart, tab_snotel, tab_table = st.tabs(["📈 Forecast Chart", "📡 SNOTEL Stations", "📊 All Resorts"])

    with tab_chart:
        # Use cached forecast data
        forecast_df = fetch_resort_7day_forecast(selected_area, lat, lon)
        if not forecast_df.empty:
            st.subheader("7-Day Snow Forecast")

            # Prepare chart data
            chart_data = forecast_df[["date", "snow_depth_cm", "new_snow_cm"]].copy()
            chart_data["date"] = pd.to_datetime(chart_data["date"])
            chart_data = chart_data.set_index("date")

            col_bar, col_line = st.columns(2)
            with col_bar:
                st.markdown("**New Snow (cm)**")
                st.bar_chart(chart_data["new_snow_cm"])
            with col_line:
                st.markdown("**Base Depth (cm)**")
                st.line_chart(chart_data["snow_depth_cm"])

    with tab_snotel:
        # Phase 2: SNOTEL station observations
        st.subheader("SNOTEL Observations")
        st.caption("Compare model forecasts with real-world SNOTEL observations")
        try:
            render_snotel_section(lat, lon, radius_km=50)
        except Exception as e:
            st.info(f"SNOTEL data unavailable: {e}")

    with tab_table:
        st.subheader("All Ski Resorts")
        # Use cached conditions data
        if not conditions_df.empty:
            # Format for display
            display_df = conditions_df[[
                "ski_area", "state", "elevation", "snow_depth_cm", "new_snow_cm", "probability"
            ]].copy()
            display_df.columns = ["Resort", "State", "Elevation (m)", "Base (cm)", "New (cm)", "Prob"]
            display_df = display_df.sort_values("Base (cm)", ascending=False)

            # Format columns
            display_df["Base (cm)"] = display_df["Base (cm)"].apply(lambda x: f"{x:.0f}")
            display_df["New (cm)"] = display_df["New (cm)"].apply(lambda x: f"{x:.1f}")
            display_df["Elevation (m)"] = display_df["Elevation (m)"].apply(lambda x: f"{x:.0f}")
            display_df["Prob"] = display_df["Prob"].apply(lambda x: f"{x:.0%}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with col_f2:
        st.caption("Forecasts use NOAA HRRR model data")


def run_dashboard():
    """Entry point for running dashboard."""
    main()


if __name__ == "__main__":
    main()
