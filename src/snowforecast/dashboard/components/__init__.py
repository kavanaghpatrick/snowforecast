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

from snowforecast.dashboard.components.forecast_overlay import (
    create_forecast_overlay,
    generate_grid_points,
    render_overlay_toggle,
    OVERLAY_COLOR_RANGE,
)

from snowforecast.dashboard.components.confidence import (
    get_confidence_level,
    get_confidence_badge,
    get_confidence_color,
    format_forecast_with_ci,
    render_confidence_badge,
    render_confidence_explanation,
    render_forecast_with_confidence,
)

from snowforecast.dashboard.components.snotel_display import (
    SnotelStation,
    MOCK_SNOTEL_STATIONS,
    calculate_pct_of_normal,
    get_snowpack_status,
    get_nearby_snotel_stations,
    create_snotel_map_layer,
    render_station_card,
    render_snotel_section,
)

from snowforecast.dashboard.components.snow_quality import (
    SnowQuality,
    QualityMetrics,
    calculate_slr,
    get_temp_trend,
    classify_snow_quality,
    get_quality_badge,
    get_slr_description,
    get_quality_explanation,
    render_snow_quality_badge,
    render_snow_quality_details,
    render_snow_quality_compact,
    create_quality_metrics,
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

from snowforecast.dashboard.components.performance import (
    PerformanceTimer,
    timed,
    lazy_load,
    get_cached_predictor,
    prefetch_forecasts,
    render_performance_metrics,
    clear_performance_metrics,
    get_performance_metrics,
    check_performance_targets,
    TARGET_PAGE_LOAD,
    TARGET_TIME_SWITCH,
    TARGET_RESORT_SELECT,
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
