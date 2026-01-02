"""SNOTEL observations display component for Snowforecast dashboard.

This module provides components for displaying SNOTEL station data:
- Current snow water equivalent (SWE) and snow depth
- Percentage of normal snowpack comparison
- Nearby station discovery
- Map layer for station markers
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pydeck as pdk

from snowforecast.utils.geo import haversine


@dataclass
class SnotelStation:
    """SNOTEL station with current observations.

    Attributes:
        station_id: SNOTEL station ID (e.g., "322")
        name: Station name
        lat: Latitude in degrees
        lon: Longitude in degrees
        elevation_m: Elevation in meters
        current_swe_mm: Current snow water equivalent in millimeters
        current_depth_cm: Current snow depth in centimeters
        normal_swe_mm: 30-year average SWE for this date
        pct_of_normal: Percentage of normal snowpack
        last_updated: Timestamp of last observation
    """

    station_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    current_swe_mm: float
    current_depth_cm: float
    normal_swe_mm: float
    pct_of_normal: float
    last_updated: datetime


# Mock SNOTEL data for development (real pipeline may not be connected)
MOCK_SNOTEL_STATIONS = [
    SnotelStation(
        "322", "Alta Guard Station", 40.58, -111.63, 2661,
        450, 120, 400, 112.5, datetime.now()
    ),
    SnotelStation(
        "628", "Brighton", 40.60, -111.58, 2670,
        380, 100, 350, 108.6, datetime.now()
    ),
    SnotelStation(
        "766", "Snowbird", 40.56, -111.65, 2850,
        520, 140, 480, 108.3, datetime.now()
    ),
    SnotelStation(
        "842", "Vail Mountain", 39.64, -106.37, 3100,
        350, 95, 420, 83.3, datetime.now()
    ),
    SnotelStation(
        "1050", "Berthoud Summit", 39.80, -105.78, 3450,
        480, 130, 500, 96.0, datetime.now()
    ),
    SnotelStation(
        "978", "Park City", 40.65, -111.51, 2560,
        280, 75, 320, 87.5, datetime.now()
    ),
    SnotelStation(
        "556", "Mammoth Pass", 37.61, -119.03, 2835,
        620, 165, 550, 112.7, datetime.now()
    ),
    SnotelStation(
        "774", "Squaw Valley", 39.19, -120.24, 2530,
        420, 110, 480, 87.5, datetime.now()
    ),
    SnotelStation(
        "1032", "Jackson Hole", 43.59, -110.85, 2970,
        550, 145, 600, 91.7, datetime.now()
    ),
    SnotelStation(
        "890", "Telluride", 37.94, -107.85, 3050,
        300, 80, 380, 78.9, datetime.now()
    ),
    SnotelStation(
        "1105", "Aspen", 39.10, -106.82, 2920,
        320, 85, 400, 80.0, datetime.now()
    ),
    SnotelStation(
        "669", "Steamboat", 40.45, -106.74, 2730,
        410, 108, 450, 91.1, datetime.now()
    ),
]


def calculate_pct_of_normal(current_swe: float, normal_swe: float) -> float:
    """Calculate percentage of normal snowpack.

    Args:
        current_swe: Current SWE in any unit
        normal_swe: Normal (30-year average) SWE in same unit

    Returns:
        Percentage of normal as a float (e.g., 112.5 for 112.5%)
        Returns 0.0 if normal_swe is zero or negative

    Examples:
        >>> calculate_pct_of_normal(450, 400)
        112.5
        >>> calculate_pct_of_normal(300, 400)
        75.0
        >>> calculate_pct_of_normal(100, 0)
        0.0
    """
    if normal_swe <= 0:
        return 0.0
    return (current_swe / normal_swe) * 100


def get_snowpack_status(pct_of_normal: float) -> tuple[str, str]:
    """Return status text and color based on percentage of normal.

    Args:
        pct_of_normal: Percentage of normal snowpack (e.g., 112.5)

    Returns:
        Tuple of (status_text, hex_color):
        - >120%: "Above Normal" (blue #3B82F6)
        - 90-120%: "Normal" (green #22C55E)
        - 70-90%: "Below Normal" (yellow #EAB308)
        - <70%: "Well Below Normal" (red #EF4444)

    Examples:
        >>> get_snowpack_status(125.0)
        ('Above Normal', '#3B82F6')
        >>> get_snowpack_status(100.0)
        ('Normal', '#22C55E')
        >>> get_snowpack_status(80.0)
        ('Below Normal', '#EAB308')
        >>> get_snowpack_status(50.0)
        ('Well Below Normal', '#EF4444')
    """
    if pct_of_normal > 120:
        return ("Above Normal", "#3B82F6")  # Blue
    elif pct_of_normal >= 90:
        return ("Normal", "#22C55E")  # Green
    elif pct_of_normal >= 70:
        return ("Below Normal", "#EAB308")  # Yellow
    else:
        return ("Well Below Normal", "#EF4444")  # Red


def get_nearby_snotel_stations(
    lat: float,
    lon: float,
    radius_km: float = 50,
    stations: Optional[list[SnotelStation]] = None,
) -> list[SnotelStation]:
    """Get SNOTEL stations within radius of a point.

    Uses Haversine distance calculation to find nearby stations.

    Args:
        lat: Latitude of the center point
        lon: Longitude of the center point
        radius_km: Search radius in kilometers (default 50)
        stations: Optional list of stations to search. If None, uses mock data.

    Returns:
        List of SnotelStation objects within the radius, sorted by distance

    Examples:
        >>> stations = get_nearby_snotel_stations(40.58, -111.63, radius_km=20)
        >>> len(stations) >= 1
        True
    """
    if stations is None:
        stations = MOCK_SNOTEL_STATIONS

    nearby = []
    for station in stations:
        distance = haversine(lat, lon, station.lat, station.lon)
        if distance <= radius_km:
            nearby.append((distance, station))

    # Sort by distance
    nearby.sort(key=lambda x: x[0])

    return [station for _, station in nearby]


def _hex_to_rgb(hex_color: str) -> list[int]:
    """Convert hex color to RGB list."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def create_snotel_map_layer(
    stations: list[SnotelStation],
    icon_size: int = 15,
) -> pdk.Layer:
    """Create PyDeck layer for SNOTEL station markers.

    Uses diamond-shaped icons with colors based on snowpack status.
    Different from resort markers (circles) for visual distinction.

    Args:
        stations: List of SnotelStation objects to display
        icon_size: Base icon size in pixels (default 15)

    Returns:
        PyDeck ScatterplotLayer configured for SNOTEL stations
    """
    # Build data with colors based on snowpack status
    data = []
    for station in stations:
        _, color_hex = get_snowpack_status(station.pct_of_normal)
        color_rgb = _hex_to_rgb(color_hex)

        data.append({
            "station_id": station.station_id,
            "name": station.name,
            "latitude": station.lat,
            "longitude": station.lon,
            "elevation_m": station.elevation_m,
            "current_swe_mm": station.current_swe_mm,
            "current_depth_cm": station.current_depth_cm,
            "pct_of_normal": station.pct_of_normal,
            "color": color_rgb + [200],  # Add alpha
        })

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=icon_size * 50,  # Scale for visibility
        radius_min_pixels=8,
        radius_max_pixels=20,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=2,
    )


def render_station_card(station: SnotelStation, container=None) -> None:
    """Render a single SNOTEL station card.

    Args:
        station: SnotelStation to display
        container: Streamlit container. If None, uses st.
    """
    import streamlit as st

    target = container if container is not None else st

    status_text, status_color = get_snowpack_status(station.pct_of_normal)

    card_html = f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        background: white;
    ">
        <div style="font-weight: bold; margin-bottom: 4px;">
            {station.name}
        </div>
        <div style="font-size: 0.85em; color: #666; margin-bottom: 8px;">
            {station.elevation_m:.0f}m elevation
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <div style="font-size: 0.75em; color: #888;">SWE</div>
                <div style="font-weight: bold;">{station.current_swe_mm:.0f}mm</div>
            </div>
            <div>
                <div style="font-size: 0.75em; color: #888;">Depth</div>
                <div style="font-weight: bold;">{station.current_depth_cm:.0f}cm</div>
            </div>
            <div>
                <div style="font-size: 0.75em; color: #888;">% Normal</div>
                <div style="font-weight: bold; color: {status_color};">{station.pct_of_normal:.0f}%</div>
            </div>
        </div>
        <div style="
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background: {status_color}20;
            color: {status_color};
            font-size: 0.85em;
            font-weight: 500;
        ">
            {status_text}
        </div>
    </div>
    """

    target.markdown(card_html, unsafe_allow_html=True)


def render_snotel_section(
    lat: float,
    lon: float,
    radius_km: float = 50,
    container=None,
    stations: Optional[list[SnotelStation]] = None,
) -> None:
    """Render SNOTEL observations section in detail panel.

    Shows nearby SNOTEL stations with current conditions including:
    - Current SWE and snow depth
    - Percentage of normal with color-coded status
    - Mini historical comparison chart

    Args:
        lat: Latitude of the center point (typically resort location)
        lon: Longitude of the center point
        radius_km: Search radius in kilometers (default 50)
        container: Streamlit container. If None, uses st.
        stations: Optional list of stations. If None, uses mock data.
    """
    import streamlit as st

    target = container if container is not None else st

    # Get nearby stations
    nearby = get_nearby_snotel_stations(lat, lon, radius_km, stations)

    if not nearby:
        target.info(f"No SNOTEL stations found within {radius_km}km")
        return

    target.markdown("**Nearby SNOTEL Stations**")
    target.caption(f"{len(nearby)} stations within {radius_km}km")

    # Calculate regional average
    if nearby:
        avg_pct = sum(s.pct_of_normal for s in nearby) / len(nearby)
        status_text, status_color = get_snowpack_status(avg_pct)

        target.markdown(
            f"<div style='padding: 8px; background: {status_color}15; "
            f"border-left: 4px solid {status_color}; margin-bottom: 12px;'>"
            f"<b>Regional Average:</b> {avg_pct:.0f}% of normal "
            f"<span style='color: {status_color};'>({status_text})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Render individual station cards
    for station in nearby[:5]:  # Show top 5 nearest
        render_station_card(station, target)

    if len(nearby) > 5:
        target.caption(f"+ {len(nearby) - 5} more stations...")

    # Mini chart showing historical comparison (simplified)
    if nearby:
        target.markdown("**Snowpack Comparison**")

        # Create bar chart data
        chart_data = pd.DataFrame([
            {
                "Station": s.name[:15] + "..." if len(s.name) > 15 else s.name,
                "% of Normal": s.pct_of_normal,
            }
            for s in nearby[:5]
        ])
        chart_data = chart_data.set_index("Station")

        target.bar_chart(chart_data)
