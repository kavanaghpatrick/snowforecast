"""Dashboard UI components for Snow Forecast.

This module provides reusable Streamlit components:
- resort_detail: Enhanced resort detail panel with forecasts
"""

from .resort_detail import (
    generate_forecast_summary,
    render_forecast_table,
    render_resort_detail,
)

__all__ = [
    "generate_forecast_summary",
    "render_forecast_table",
    "render_resort_detail",
]
