"""Dashboard UI components for Snowforecast.

This module provides reusable Streamlit components:

- Map: render_resort_map, create_resort_layer, create_base_view
- Detail Panel: generate_forecast_summary, render_forecast_table, render_resort_detail
- Terrain: create_terrain_layer, create_3d_view, create_terrain_deck, render_terrain_controls
"""

from snowforecast.dashboard.components.map_view import (
    render_resort_map,
    create_resort_layer,
    create_base_view,
)

from .resort_detail import (
    generate_forecast_summary,
    render_forecast_table,
    render_resort_detail,
)

from snowforecast.dashboard.components.terrain_layer import (
    create_terrain_layer,
    create_3d_view,
    create_terrain_deck,
    render_terrain_controls,
    TERRAIN_IMAGE,
    ELEVATION_DECODER,
)

__all__ = [
    # Map components
    "render_resort_map",
    "create_resort_layer",
    "create_base_view",
    # Detail panel
    "generate_forecast_summary",
    "render_forecast_table",
    "render_resort_detail",
    # Terrain 3D
    "create_terrain_layer",
    "create_3d_view",
    "create_terrain_deck",
    "render_terrain_controls",
    "TERRAIN_IMAGE",
    "ELEVATION_DECODER",
]
