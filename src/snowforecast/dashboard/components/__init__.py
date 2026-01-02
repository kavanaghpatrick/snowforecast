"""Dashboard UI components for Snowforecast.

This module provides reusable Streamlit components:

- Map: render_resort_map, create_resort_layer, create_base_view
- Detail Panel: generate_forecast_summary, render_forecast_table, render_resort_detail
- Terrain: create_terrain_layer, create_3d_view, create_terrain_deck, render_terrain_controls
- Elevation: render_elevation_bands
- Favorites: get_favorites, add_favorite, remove_favorite, is_favorite, render_favorite_toggle
- Time Selector: TIME_STEPS, render_time_selector, get_forecast_time, prefetch_all_forecasts
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

from snowforecast.dashboard.components.time_selector import (
    TIME_STEPS,
    SELECTED_TIME_STEP_KEY,
    ALL_FORECASTS_KEY,
    render_time_selector,
    get_forecast_time,
    get_forecast_hours,
    prefetch_all_forecasts,
    get_cached_forecast,
    get_current_forecast,
    needs_prefetch,
    clear_forecast_cache,
    parse_time_step,
    calculate_forecast_datetime,
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
    # Time selector
    "TIME_STEPS",
    "SELECTED_TIME_STEP_KEY",
    "ALL_FORECASTS_KEY",
    "render_time_selector",
    "get_forecast_time",
    "get_forecast_hours",
    "prefetch_all_forecasts",
    "get_cached_forecast",
    "get_current_forecast",
    "needs_prefetch",
    "clear_forecast_cache",
    "parse_time_step",
    "calculate_forecast_datetime",
]
