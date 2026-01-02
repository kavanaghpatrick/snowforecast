"""Dashboard UI components for Snowforecast.

This module provides reusable Streamlit components:

- Map: render_resort_map, create_resort_layer, create_base_view
- Detail Panel: generate_forecast_summary, render_forecast_table, render_resort_detail
- Terrain: create_terrain_layer, create_3d_view, create_terrain_deck, render_terrain_controls
- Elevation: render_elevation_bands
- Favorites: get_favorites, add_favorite, remove_favorite, is_favorite, render_favorite_toggle
- Loading: with_loading, with_error_handling, render_loading_skeleton, render_retry_button
- Cache Status: get_cache_freshness, render_cache_status, render_data_warning, render_fallback_notice
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

from snowforecast.dashboard.components.loading import (
    with_loading,
    with_error_handling,
    with_loading_and_error_handling,
    render_loading_skeleton,
    render_retry_button,
    render_error_with_retry,
    render_empty_state,
)

from snowforecast.dashboard.components.cache_status import (
    get_cache_freshness,
    get_cache_freshness_emoji,
    format_cache_age,
    render_cache_status,
    render_cache_status_badge,
    render_data_warning,
    render_fallback_notice,
    render_data_source_indicator,
    should_show_stale_warning,
    FRESHNESS_THRESHOLDS,
    FRESHNESS_COLORS,
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
    # Loading and error handling
    "with_loading",
    "with_error_handling",
    "with_loading_and_error_handling",
    "render_loading_skeleton",
    "render_retry_button",
    "render_error_with_retry",
    "render_empty_state",
    # Cache status
    "get_cache_freshness",
    "get_cache_freshness_emoji",
    "format_cache_age",
    "render_cache_status",
    "render_cache_status_badge",
    "render_data_warning",
    "render_fallback_notice",
    "render_data_source_indicator",
    "should_show_stale_warning",
    "FRESHNESS_THRESHOLDS",
    "FRESHNESS_COLORS",
]
