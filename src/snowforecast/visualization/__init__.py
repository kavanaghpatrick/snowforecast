"""Visualization utilities for Snowforecast dashboard.

This module provides shared color scales and legend components
used across all visual components (maps, charts, tables).
"""

from .colors import (
    # Snow depth colors
    SNOW_DEPTH_SCALE,
    snow_depth_to_hex,
    snow_depth_to_rgb,
    snow_depth_category,
    # Elevation colors
    ELEVATION_SCALE,
    elevation_to_rgb,
    # Legends
    render_snow_legend,
    render_elevation_legend,
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
