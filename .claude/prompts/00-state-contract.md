# State Contract for Dashboard V2

## CRITICAL: All agents must follow this contract

This document defines the shared `st.session_state` variables used for communication between dashboard components. All agents MUST use these exact variable names.

## Session State Variables

### Resort Selection
```python
# Selected resort name (string or None)
st.session_state.selected_resort: Optional[str] = None

# Example: "Alta", "Snowbird", None
```

### Time Selection
```python
# Selected forecast time step (string)
st.session_state.selected_time_step: str = "Now"

# Valid values: "Now", "Tonight", "Tomorrow AM", "Tomorrow PM",
#               "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"
```

### View Mode
```python
# 2D or 3D map view
st.session_state.view_mode: str = "2D"

# Valid values: "2D", "3D"
```

### Favorites Filter
```python
# Show only favorites
st.session_state.filter_favorites: bool = False
```

### Cached Data
```python
# All resort forecasts (cached on load)
st.session_state.all_forecasts: dict[str, list[ForecastResult]] = {}

# Resort data for map
st.session_state.resort_data: pd.DataFrame = None
```

## How to Use

### Reading State (any component)
```python
selected = st.session_state.get('selected_resort')
if selected:
    show_detail_panel(selected)
```

### Writing State (map component)
```python
def on_resort_click(resort_name: str):
    st.session_state.selected_resort = resort_name
    st.rerun()
```

### Initializing State (app.py)
```python
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'selected_resort': None,
        'selected_time_step': 'Now',
        'view_mode': '2D',
        'filter_favorites': False,
        'all_forecasts': {},
        'resort_data': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

## File Ownership

To avoid merge conflicts:

| Component | Owner | Files |
|-----------|-------|-------|
| State init | app.py | `init_session_state()` |
| Resort selection | map_view.py | Writes `selected_resort` |
| Time selection | timeline.py | Writes `selected_time_step` |
| View mode | terrain_layer.py | Writes `view_mode` |
| Favorites filter | favorites.py | Writes `filter_favorites` |
| Detail panel | resort_detail.py | Reads `selected_resort` |

## Integration Point

All components are integrated in `app.py`:

```python
# src/snowforecast/dashboard/app.py

def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        render_favorites_filter()  # favorites.py
        render_view_toggle()       # terrain_layer.py
        render_time_selector()     # timeline.py

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        render_resort_map()        # map_view.py

    with col2:
        if st.session_state.selected_resort:
            render_resort_detail() # resort_detail.py
        else:
            st.info("Select a resort on the map")
```

## IMPORTANT FOR ALL AGENTS

1. **DO NOT** create new session state variables without documenting here
2. **DO NOT** modify another component's session state writes
3. **DO** read from session state as needed
4. **DO** use `st.rerun()` after writing state to trigger re-render
