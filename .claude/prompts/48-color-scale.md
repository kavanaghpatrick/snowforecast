# Agent Task: Color Scale Foundation (#48)

## BLOCKING DEPENDENCY
This task MUST complete before any other Phase 6 work can begin.

## Your Mission
Create the shared color scale utilities that will be used by ALL visual components.

## Files to Create

### 1. `src/snowforecast/visualization/__init__.py`
```python
from .colors import (
    snow_depth_to_hex,
    snow_depth_to_rgb,
    elevation_to_rgb,
    render_snow_legend,
    SNOW_DEPTH_SCALE,
    ELEVATION_SCALE,
)
```

### 2. `src/snowforecast/visualization/colors.py`
Implement:
- `SNOW_DEPTH_SCALE` - List of (threshold_cm, hex_color) tuples
- `ELEVATION_SCALE` - List of (threshold_m, rgb_list) tuples
- `snow_depth_to_hex(depth_cm)` - Returns hex color string
- `snow_depth_to_rgb(depth_cm, alpha=200)` - Returns [R,G,B,A] list for PyDeck
- `elevation_to_rgb(elevation_m, alpha=200)` - Returns [R,G,B,A] list
- `render_snow_legend(container)` - Renders legend in Streamlit container

### 3. `tests/visualization/test_colors.py`
Test all functions with edge cases (0, negative, very large values).

## Color Scales

### Snow Depth (Blue-Purple)
| Threshold | Color | Hex |
|-----------|-------|-----|
| 0-10 cm | Very light blue | #E6F3FF |
| 10-30 cm | Light blue | #ADD8E6 |
| 30-60 cm | Cornflower | #6495ED |
| 60-100 cm | Royal blue | #4169E1 |
| 100-150 cm | Medium blue | #0000CD |
| >150 cm | Purple | #8A2BE2 |

### Elevation (Green-Brown-White)
| Threshold | Color | RGB |
|-----------|-------|-----|
| <1500m | Forest green | [34, 139, 34] |
| 1500-2000m | Sage | [143, 188, 143] |
| 2000-2500m | Tan | [210, 180, 140] |
| 2500-3000m | Brown | [139, 119, 101] |
| 3000-3500m | Gray | [169, 169, 169] |
| >3500m | White | [255, 255, 255] |

## Acceptance Criteria
- [ ] All files created
- [ ] Tests pass: `pytest tests/visualization/test_colors.py -v`
- [ ] Can import: `from snowforecast.visualization import snow_depth_to_rgb`

## When Done
1. Run tests
2. Commit: `git commit -m "Add color scale foundation for dashboard v2"`
3. Push to develop
4. Mark issue #48 as complete
