"""Dashboard components for Snow Forecast visualization.

This module provides reusable UI components:
- terrain_layer: 3D terrain visualization with PyDeck
"""

from snowforecast.dashboard.components.terrain_layer import (
    create_terrain_layer,
    create_3d_view,
    create_terrain_deck,
    render_terrain_controls,
    TERRAIN_IMAGE,
    ELEVATION_DECODER,
)

__all__ = [
    "create_terrain_layer",
    "create_3d_view",
    "create_terrain_deck",
    "render_terrain_controls",
    "TERRAIN_IMAGE",
    "ELEVATION_DECODER",
]
