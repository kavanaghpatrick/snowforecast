# Agent Task: Resort Detail Panel (#36)

## Prerequisites
- Issue #48 (Color Scale) complete
- Issue #35 (Resort Map) in progress

## Your Mission
Create enhanced resort detail panel with forecast summary and visualizations.

## Files to Create

### `src/snowforecast/dashboard/components/resort_detail.py`
```python
import streamlit as st
import pandas as pd
from snowforecast.visualization import snow_depth_to_hex, render_snow_legend

def generate_forecast_summary(forecasts: list) -> str:
    """Generate natural language forecast summary."""
    # Analyze 7-day forecast
    # Return: "Heavy snow expected Tuesday-Wednesday (15-25cm). Clearing Thursday."
    ...

def render_forecast_table(forecasts: pd.DataFrame) -> None:
    """Render 7-day forecast with AM/PM/Night blocks."""
    # Use color scale for snow amounts
    # Show confidence intervals
    ...

def render_resort_detail(resort: dict, forecasts: pd.DataFrame) -> None:
    """Main detail panel component."""
    st.subheader(f"{resort['name']}, {resort['state']}")
    st.caption(f"Elevation: {resort['elevation']}m")
    
    # Summary
    st.markdown(generate_forecast_summary(forecasts))
    
    # Forecast table
    render_forecast_table(forecasts)
    
    # Snow depth chart
    st.line_chart(forecasts[['date', 'snow_depth_cm']].set_index('date'))
```

## Natural Language Summary
Generate human-readable forecast like:
- "Heavy snow expected Tuesday night through Wednesday (15-25cm)"
- "Clearing Thursday with cold temperatures"
- "Dry weekend with good conditions"

## Worktree
Work in: `/Users/patrickkavanagh/snowforecast-worktrees/detail-panel`
Branch: `phase6/36-detail-panel`
