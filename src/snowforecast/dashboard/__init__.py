"""Visualization dashboard for Snow Forecast.

This module provides:

- run_dashboard: Launch Streamlit dashboard
- create_map_view: Map visualization of predictions
- create_time_series: Station time series plots
- create_metrics_cards: Performance metric displays
- render_elevation_bands: Elevation band forecast component
"""

from snowforecast.dashboard.app import run_dashboard
from snowforecast.dashboard.components import render_elevation_bands

__all__ = ["render_elevation_bands", "run_dashboard"]
