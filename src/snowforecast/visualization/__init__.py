"""Visualization utilities for Snowforecast dashboard.

This module provides shared color scales and legend components
used across all visual components (maps, charts, tables).
"""

from .colors import (
    # Elevation colors
    ELEVATION_SCALE,
    # Snow depth colors
    SNOW_DEPTH_SCALE,
    elevation_to_rgb,
    render_elevation_legend,
    # Legends
    render_snow_legend,
    snow_depth_category,
    snow_depth_to_hex,
    snow_depth_to_rgb,
)

__all__ = [
    "SNOW_DEPTH_SCALE",
    "snow_depth_to_hex",
    "snow_depth_to_rgb",
    "snow_depth_category",
    "ELEVATION_SCALE",
    "elevation_to_rgb",
    "render_snow_legend",
    "render_elevation_legend",
]
