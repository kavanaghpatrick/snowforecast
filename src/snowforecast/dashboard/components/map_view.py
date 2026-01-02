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

import pydeck as pdk
import pandas as pd

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


def create_base_view() -> pdk.ViewState:
    """Create Western US view centered on ski country.

    Returns:
        PyDeck ViewState centered on Utah/Colorado ski areas,
        zoomed to show all Western US resorts.

    Note:
        View is centered at approximately:
        - Latitude 40.0 (Utah/Colorado border)
        - Longitude -111.0 (Salt Lake City area)
        - Zoom 5.5 (shows WA, OR, CA, UT, CO, MT, WY, ID)
    """
    return pdk.ViewState(
        latitude=40.0,
        longitude=-111.0,
        zoom=5.5,
        pitch=0,
    )


def render_resort_map(resort_data: pd.DataFrame) -> pdk.Deck:
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

    Returns:
        PyDeck Deck object ready for display with st.pydeck_chart()

    Features:
        - Color-coded markers based on snow depth
        - Size scaled by new snow in 24 hours
        - Tooltips showing resort name, state, and conditions
        - Free CartoDB basemap (no API key required)
        - Loads in <3 seconds for 22 resorts

    Example:
        >>> deck = render_resort_map(resort_data)
        >>> st.pydeck_chart(deck)
    """
    tooltip = {
        "html": """
            <b>{ski_area}</b><br/>
            {state}<br/>
            Snow Depth: {snow_depth_cm} cm<br/>
            New Snow: {new_snow_cm} cm<br/>
            Probability: {probability:.0%}
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
        initial_view_state=create_base_view(),
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
