"""Dashboard UI components for Snowforecast.

This module provides reusable Streamlit components:

- Map: render_resort_map, create_resort_layer, create_base_view
- Detail Panel: generate_forecast_summary, render_forecast_table, render_resort_detail
- Terrain: create_terrain_layer, create_3d_view, create_terrain_deck, render_terrain_controls
- Elevation: render_elevation_bands
- Favorites: get_favorites, add_favorite, remove_favorite, is_favorite, render_favorite_toggle
- Responsive: get_breakpoint, is_mobile, is_tablet, is_desktop, render_responsive_columns
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

from snowforecast.dashboard.components.elevation_bands import render_elevation_bands

from snowforecast.dashboard.components.favorites import (
    get_favorites,
    save_favorites,
    add_favorite,
    remove_favorite,
    is_favorite,
    render_favorite_toggle,
    render_favorites_filter,
    FAVORITES_KEY,
)

from snowforecast.dashboard.components.responsive import (
    get_viewport_width,
    get_breakpoint,
    is_mobile,
    is_tablet,
    is_desktop,
    get_column_ratio,
    should_show_3d,
    get_touch_target_size,
    render_responsive_columns,
    inject_responsive_css,
    set_viewport_width,
    inject_viewport_detector,
    MOBILE_MAX,
    TABLET_MAX,
    Breakpoint,
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
    # Elevation bands
    "render_elevation_bands",
    # Favorites
    "get_favorites",
    "save_favorites",
    "add_favorite",
    "remove_favorite",
    "is_favorite",
    "render_favorite_toggle",
    "render_favorites_filter",
    "FAVORITES_KEY",
    # Responsive layout
    "get_viewport_width",
    "get_breakpoint",
    "is_mobile",
    "is_tablet",
    "is_desktop",
    "get_column_ratio",
    "should_show_3d",
    "get_touch_target_size",
    "render_responsive_columns",
    "inject_responsive_css",
    "set_viewport_width",
    "inject_viewport_detector",
    "MOBILE_MAX",
    "TABLET_MAX",
    "Breakpoint",
]
