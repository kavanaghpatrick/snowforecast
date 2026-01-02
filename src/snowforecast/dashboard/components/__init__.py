"""Dashboard components for Snowforecast visualization.

This module provides reusable components for the Streamlit dashboard:

- render_resort_map: Interactive PyDeck map with ski resort markers
- create_resort_layer: ScatterplotLayer for resort visualization
- create_base_view: Western US view state centered on ski country
"""

from snowforecast.dashboard.components.map_view import (
    render_resort_map,
    create_resort_layer,
    create_base_view,
)

__all__ = [
    "render_resort_map",
    "create_resort_layer",
    "create_base_view",
]
