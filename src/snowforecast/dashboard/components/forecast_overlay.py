"""Forecast overlay component for Snowforecast dashboard.

Provides a HexagonLayer heatmap overlay showing predicted snowfall intensity
across the map. The overlay uses the same color scale as snow depth markers
for visual consistency.

Example usage:
    >>> import pandas as pd
    >>> from snowforecast.dashboard.components import (
    ...     create_forecast_overlay,
    ...     generate_grid_points,
    ...     render_overlay_toggle,
    ... )
    >>>
    >>> # Generate grid points for forecast
    >>> grid = generate_grid_points(
    ...     lat_min=37.0, lat_max=49.0,
    ...     lon_min=-125.0, lon_max=-102.0,
    ...     resolution=0.5,
    ... )
    >>> # Add forecast data (would come from predictor)
    >>> grid['new_snow_cm'] = [10, 20, 30, ...]  # predicted snowfall
    >>>
    >>> # Create overlay layer
    >>> overlay = create_forecast_overlay(grid)
    >>>
    >>> # Add to existing deck
    >>> deck.layers.append(overlay)
"""

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from typing import Optional

from snowforecast.visualization import SNOW_DEPTH_SCALE


# Color range matching snow depth scale (Trace to Extreme)
# RGB values extracted from SNOW_DEPTH_SCALE hex colors
OVERLAY_COLOR_RANGE = [
    [230, 243, 255],  # Trace - very light blue (#E6F3FF)
    [173, 216, 230],  # Light - light blue (#ADD8E6)
    [100, 149, 237],  # Moderate - cornflower (#6495ED)
    [65, 105, 225],   # Heavy - royal blue (#4169E1)
    [0, 0, 205],      # Very heavy - medium blue (#0000CD)
    [138, 43, 226],   # Extreme - purple (#8A2BE2)
]


def create_forecast_overlay(
    forecast_points: pd.DataFrame,
    radius: int = 5000,  # 5km hexagons
    opacity: float = 0.5,
    extruded: bool = False,
) -> pdk.Layer:
    """Create HexagonLayer for forecast heatmap overlay.

    Args:
        forecast_points: DataFrame with columns:
            - lon: float (longitude)
            - lat: float (latitude)
            - new_snow_cm: float (predicted snowfall in cm)
        radius: Hexagon radius in meters (default 5000 = 5km)
        opacity: Layer opacity 0-1 (default 0.5)
        extruded: Whether to show as 3D hexagons (default False)

    Returns:
        PyDeck HexagonLayer with color gradient based on snowfall intensity

    Note:
        - Colors match the snow depth scale from snowforecast.visualization
        - Hexagons aggregate points within their radius
        - Elevation weight is based on new_snow_cm values
    """
    return pdk.Layer(
        "HexagonLayer",
        data=forecast_points,
        get_position=['lon', 'lat'],
        get_elevation_weight='new_snow_cm',
        elevation_scale=50,
        radius=radius,
        coverage=0.8,
        color_range=OVERLAY_COLOR_RANGE,
        pickable=True,
        extruded=extruded,
        opacity=opacity,
    )


def generate_grid_points(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: float = 0.1,  # ~10km grid
) -> pd.DataFrame:
    """Generate a grid of points for forecast sampling.

    Creates a uniform grid of lat/lon points covering the specified bounding
    box. Use this to create sampling points for the forecast predictor.

    Args:
        lat_min: Southern boundary latitude
        lat_max: Northern boundary latitude
        lon_min: Western boundary longitude
        lon_max: Eastern boundary longitude
        resolution: Grid spacing in degrees (default 0.1 ~ 10km)

    Returns:
        DataFrame with columns:
            - lat: float (latitude)
            - lon: float (longitude)

    Example:
        >>> grid = generate_grid_points(37.0, 40.0, -110.0, -105.0, resolution=0.5)
        >>> len(grid)
        77  # 7 lat points x 11 lon points
    """
    # Calculate number of points to ensure we stay within bounds
    # Using linspace for floating point precision
    num_lats = int(round((lat_max - lat_min) / resolution)) + 1
    num_lons = int(round((lon_max - lon_min) / resolution)) + 1

    lats = np.linspace(lat_min, lat_max, num_lats)
    lons = np.linspace(lon_min, lon_max, num_lons)

    # Create meshgrid and flatten
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    return pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
    })


def render_overlay_toggle(container=None) -> bool:
    """Render 'Show forecast overlay' checkbox.

    Provides a toggle for users to show/hide the forecast heatmap overlay
    on the map.

    Args:
        container: Streamlit container (st, st.sidebar, st.columns()[0], etc.)
                   If None, uses st directly.

    Returns:
        bool: True if overlay should be shown, False otherwise

    Example:
        >>> show_overlay = render_overlay_toggle(st.sidebar)
        >>> if show_overlay:
        ...     deck.layers.append(create_forecast_overlay(grid))
    """
    target = container if container is not None else st

    return target.checkbox(
        "Show forecast overlay",
        key="show_forecast_overlay",
        help="Display snowfall prediction heatmap across the map",
    )
