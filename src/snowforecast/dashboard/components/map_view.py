"""Interactive resort map component for Snowforecast dashboard.

Provides a PyDeck-based map visualization showing all ski resorts
with color-coded markers based on snow depth.

Example usage:
    >>> import pandas as pd
    >>> from snowforecast.dashboard.components import render_resort_map
    >>>
    >>> # Sample resort data
    >>> resort_data = pd.DataFrame({
    ...     'ski_area': ['Vail', 'Park City'],
    ...     'state': ['Colorado', 'Utah'],
    ...     'latitude': [39.64, 40.65],
    ...     'longitude': [-106.37, -111.51],
    ...     'snow_depth_cm': [150, 100],
    ...     'new_snow_cm': [15, 8],
    ...     'probability': [0.75, 0.50],
    ... })
    >>> deck = render_resort_map(resort_data)
"""

import pandas as pd
import pydeck as pdk

from snowforecast.visualization import snow_depth_to_rgb


def create_resort_layer(resort_data: pd.DataFrame) -> pdk.Layer:
    """Create ScatterplotLayer for ski resorts.

    Args:
        resort_data: DataFrame with columns:
            - latitude: float
            - longitude: float
            - snow_depth_cm: float (for marker color)
            - new_snow_cm: float (for marker size)

    Returns:
        PyDeck ScatterplotLayer with color-coded markers

    Note:
        - Marker color is based on snow_depth_cm using the standard
          snow depth color scale from snowforecast.visualization
        - Marker radius scales with new_snow_cm (min 8px, max 50px)
        - Markers are pickable for tooltip interaction
    """
    # Copy to avoid modifying original data
    data = resort_data.copy()

    # Apply color function to each row
    data['color'] = data['snow_depth_cm'].apply(snow_depth_to_rgb)

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=['longitude', 'latitude'],
        get_fill_color='color',
        get_radius='new_snow_cm * 50 + 500',
        radius_min_pixels=8,
        radius_max_pixels=50,
        pickable=True,
    )


def create_base_view(
    center_lat: float = None,
    center_lon: float = None,
    zoom: float = None
) -> pdk.ViewState:
    """Create map view, optionally centered on a specific resort.

    Args:
        center_lat: Latitude to center on (default: Western US overview)
        center_lon: Longitude to center on (default: Western US overview)
        zoom: Zoom level (default: 5.5 for overview, 9 for single resort)

    Returns:
        PyDeck ViewState centered on specified location or Western US overview.

    Note:
        When no coordinates provided, defaults to Western US overview:
        - Latitude 40.0 (Utah/Colorado border)
        - Longitude -111.0 (Salt Lake City area)
        - Zoom 5.5 (shows WA, OR, CA, UT, CO, MT, WY, ID)
    """
    if center_lat is not None and center_lon is not None:
        # Center on specific resort with closer zoom
        return pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom if zoom is not None else 9,
            pitch=0,
        )
    else:
        # Default: Western US overview
        return pdk.ViewState(
            latitude=40.0,
            longitude=-111.0,
            zoom=zoom if zoom is not None else 5.5,
            pitch=0,
        )


def render_resort_map(
    resort_data: pd.DataFrame,
    center_lat: float = None,
    center_lon: float = None,
    zoom: float = None
) -> pdk.Deck:
    """Render the full interactive resort map.

    Args:
        resort_data: DataFrame with columns:
            - ski_area: str (resort name)
            - state: str (state name)
            - latitude: float
            - longitude: float
            - snow_depth_cm: float
            - new_snow_cm: float
            - probability: float (snowfall probability, 0-1)
        center_lat: Optional latitude to center map on
        center_lon: Optional longitude to center map on
        zoom: Optional zoom level (default: 9 for single resort, 5.5 for overview)

    Returns:
        PyDeck Deck object ready for display with st.pydeck_chart()

    Features:
        - Color-coded markers based on snow depth
        - Size scaled by new snow in 24 hours
        - Tooltips showing resort name, state, and conditions
        - Free CartoDB basemap (no API key required)
        - Loads in <3 seconds for 22 resorts
        - Centers on selected resort when coordinates provided

    Example:
        >>> deck = render_resort_map(resort_data, center_lat=37.63, center_lon=-119.03)
        >>> st.pydeck_chart(deck)
    """
    tooltip = {
        "html": """
            <b>{ski_area}</b><br/>
            {state}<br/>
            Snow Depth: {snow_depth_cm:.0f} cm<br/>
            New Snow: {new_snow_cm:.1f} cm<br/>
            Probability: {probability_pct}
        """,
        "style": {
            "backgroundColor": "#1a1a2e",
            "color": "white",
            "padding": "8px",
            "borderRadius": "4px",
        }
    }

    return pdk.Deck(
        layers=[create_resort_layer(resort_data)],
        initial_view_state=create_base_view(center_lat, center_lon, zoom),
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
