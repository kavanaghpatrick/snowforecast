"""Visualization dashboard for Snow Forecast.

This module provides:

- run_dashboard: Launch Streamlit dashboard
- create_map_view: Map visualization of predictions
- create_time_series: Station time series plots
- create_metrics_cards: Performance metric displays
"""

from snowforecast.dashboard.app import run_dashboard

__all__ = ["run_dashboard"]
