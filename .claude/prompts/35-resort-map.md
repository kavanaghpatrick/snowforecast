# Agent Task: Interactive Resort Map (#35)

## Prerequisites
- Issue #48 (Color Scale) must be complete
- Import colors from: `from snowforecast.visualization import snow_depth_to_rgb`

## Your Mission
Create an interactive PyDeck map showing all 22 ski resorts.

## Files to Create

### `src/snowforecast/dashboard/components/__init__.py`

### `src/snowforecast/dashboard/components/map_view.py`
```python
import pydeck as pdk
import pandas as pd
import streamlit as st
from snowforecast.visualization import snow_depth_to_rgb

def create_resort_layer(resort_data: pd.DataFrame) -> pdk.Layer:
    """Create ScatterplotLayer for resorts."""
    # Add color column based on snow depth
    resort_data['color'] = resort_data['snow_depth_cm'].apply(snow_depth_to_rgb)
    
    return pdk.Layer(
        "ScatterplotLayer",
        data=resort_data,
        get_position=['longitude', 'latitude'],
        get_fill_color='color',
        get_radius='new_snow_cm * 50 + 500',
        radius_min_pixels=8,
        radius_max_pixels=50,
        pickable=True,
    )

def create_base_view() -> pdk.ViewState:
    """Western US view centered on ski country."""
    return pdk.ViewState(
        latitude=40.0,
        longitude=-111.0,
        zoom=5.5,
        pitch=0,
    )

def render_resort_map(resort_data: pd.DataFrame) -> pdk.Deck:
    """Render the full resort map."""
    tooltip = {
        "html": """
            <b>{ski_area}</b><br/>
            {state}<br/>
            Snow Depth: {snow_depth_cm} cm<br/>
            New Snow: {new_snow_cm} cm<br/>
            Probability: {probability:.0%}
        """,
        "style": {"backgroundColor": "#1a1a2e", "color": "white"}
    }
    
    return pdk.Deck(
        layers=[create_resort_layer(resort_data)],
        initial_view_state=create_base_view(),
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
```

## Integration
Update `src/snowforecast/dashboard/app.py` to use the new map component.

## Acceptance Criteria
- [ ] Map shows all 22 resorts
- [ ] Colors match snow depth scale
- [ ] Tooltips work on hover
- [ ] Uses CartoDB (free, no API key)
- [ ] Loads in <3 seconds

## Worktree
Work in: `/Users/patrickkavanagh/snowforecast-worktrees/resort-map`
Branch: `phase6/35-resort-map`
