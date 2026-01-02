"""Dashboard UI components for Snowforecast.

This module provides reusable Streamlit components:

- Map: render_resort_map, create_resort_layer, create_base_view
- Detail Panel: generate_forecast_summary, render_forecast_table, render_resort_detail
- Terrain: create_terrain_layer, create_3d_view, create_terrain_deck, render_terrain_controls
- Elevation: render_elevation_bands
- Favorites: get_favorites, add_favorite, remove_favorite, is_favorite, render_favorite_toggle
- Time Selector: TIME_STEPS, render_time_selector, get_forecast_time, prefetch_all_forecasts
- Forecast Overlay: create_forecast_overlay, generate_grid_points, render_overlay_toggle
- Confidence: get_confidence_level, get_confidence_badge, get_confidence_color, format_forecast_with_ci
- SNOTEL: SnotelStation, get_nearby_snotel_stations, render_snotel_section, create_snotel_map_layer
- Snow Quality: calculate_slr, classify_snow_quality, render_snow_quality_badge
- Responsive: get_breakpoint, is_mobile, is_tablet, is_desktop, render_responsive_columns
- Performance: PerformanceTimer, timed, lazy_load, get_cached_predictor, prefetch_forecasts
- Loading: with_loading, with_error_handling, render_loading_skeleton, render_retry_button
- Cache Status: get_cache_freshness, render_cache_status, render_data_warning, render_fallback_notice
"""

from snowforecast.dashboard.components.cache_status import (
    FRESHNESS_COLORS,
    FRESHNESS_THRESHOLDS,
    format_cache_age,
    get_cache_freshness,
    get_cache_freshness_emoji,
    render_cache_status,
    render_cache_status_badge,
    render_data_source_indicator,
    render_data_warning,
    render_fallback_notice,
    should_show_stale_warning,
)
from snowforecast.dashboard.components.confidence import (
    format_forecast_with_ci,
    get_confidence_badge,
    get_confidence_color,
    get_confidence_level,
    render_confidence_badge,
    render_confidence_explanation,
    render_forecast_with_confidence,
)
from snowforecast.dashboard.components.elevation_bands import render_elevation_bands
from snowforecast.dashboard.components.favorites import (
    FAVORITES_KEY,
    add_favorite,
    get_favorites,
    is_favorite,
    remove_favorite,
    render_favorite_toggle,
    render_favorites_filter,
    save_favorites,
)
from snowforecast.dashboard.components.forecast_overlay import (
    OVERLAY_COLOR_RANGE,
    create_forecast_overlay,
    generate_grid_points,
    render_overlay_toggle,
)
from snowforecast.dashboard.components.loading import (
    render_empty_state,
    render_error_with_retry,
    render_loading_skeleton,
    render_retry_button,
    with_error_handling,
    with_loading,
    with_loading_and_error_handling,
)
from snowforecast.dashboard.components.map_view import (
    create_base_view,
    create_resort_layer,
    render_resort_map,
)
from snowforecast.dashboard.components.performance import (
    TARGET_PAGE_LOAD,
    TARGET_RESORT_SELECT,
    TARGET_TIME_SWITCH,
    PerformanceTimer,
    check_performance_targets,
    clear_performance_metrics,
    get_cached_predictor,
    get_performance_metrics,
    lazy_load,
    prefetch_forecasts,
    render_performance_metrics,
    timed,
)
from snowforecast.dashboard.components.responsive import (
    MOBILE_MAX,
    TABLET_MAX,
    Breakpoint,
    get_breakpoint,
    get_column_ratio,
    get_touch_target_size,
    get_viewport_width,
    inject_responsive_css,
    inject_viewport_detector,
    is_desktop,
    is_mobile,
    is_tablet,
    render_responsive_columns,
    set_viewport_width,
    should_show_3d,
)
from snowforecast.dashboard.components.snotel_display import (
    MOCK_SNOTEL_STATIONS,
    SnotelStation,
    calculate_pct_of_normal,
    create_snotel_map_layer,
    get_nearby_snotel_stations,
    get_snowpack_status,
    render_snotel_section,
    render_station_card,
)
from snowforecast.dashboard.components.snow_quality import (
    QualityMetrics,
    SnowQuality,
    calculate_slr,
    classify_snow_quality,
    create_quality_metrics,
    get_quality_badge,
    get_quality_explanation,
    get_slr_description,
    get_temp_trend,
    render_snow_quality_badge,
    render_snow_quality_compact,
    render_snow_quality_details,
)
from snowforecast.dashboard.components.terrain_layer import (
    ELEVATION_DECODER,
    TERRAIN_IMAGE,
    create_3d_view,
    create_terrain_deck,
    create_terrain_layer,
    render_terrain_controls,
)
from snowforecast.dashboard.components.time_selector import (
    ALL_FORECASTS_KEY,
    SELECTED_TIME_STEP_KEY,
    TIME_STEPS,
    calculate_forecast_datetime,
    clear_forecast_cache,
    get_cached_forecast,
    get_current_forecast,
    get_forecast_hours,
    get_forecast_time,
    needs_prefetch,
    parse_time_step,
    prefetch_all_forecasts,
    render_time_selector,
)

from .resort_detail import (
    generate_forecast_summary,
    render_forecast_table,
    render_resort_detail,
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
    # Forecast overlay
    "create_forecast_overlay",
    "generate_grid_points",
    "render_overlay_toggle",
    "OVERLAY_COLOR_RANGE",
    # Confidence visualization
    "get_confidence_level",
    "get_confidence_badge",
    "get_confidence_color",
    "format_forecast_with_ci",
    "render_confidence_badge",
    "render_confidence_explanation",
    "render_forecast_with_confidence",
    # SNOTEL
    "SnotelStation",
    "MOCK_SNOTEL_STATIONS",
    "calculate_pct_of_normal",
    "get_snowpack_status",
    "get_nearby_snotel_stations",
    "create_snotel_map_layer",
    "render_station_card",
    "render_snotel_section",
    # Snow Quality
    "SnowQuality",
    "QualityMetrics",
    "calculate_slr",
    "get_temp_trend",
    "classify_snow_quality",
    "get_quality_badge",
    "get_slr_description",
    "get_quality_explanation",
    "render_snow_quality_badge",
    "render_snow_quality_details",
    "render_snow_quality_compact",
    "create_quality_metrics",
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
    # Performance
    "PerformanceTimer",
    "timed",
    "lazy_load",
    "get_cached_predictor",
    "prefetch_forecasts",
    "render_performance_metrics",
    "clear_performance_metrics",
    "get_performance_metrics",
    "check_performance_targets",
    "TARGET_PAGE_LOAD",
    "TARGET_TIME_SWITCH",
    "TARGET_RESORT_SELECT",
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
