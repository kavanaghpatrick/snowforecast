# Agent Task: Elevation Band Forecasts (#39)

## Your Mission
Show forecasts at multiple elevation bands (Base / Mid / Summit) like Mountain-Forecast.com.

## Key Requirement
PRE-COMPUTE in batch pipeline, NOT on-demand calculation.

## Dependencies
- Issue #48 (Color Scale) must be complete
- Uses existing CachedPredictor

## Files to Create

### `src/snowforecast/cache/elevation_bands.py`
```python
"""Compute forecasts at multiple elevation bands."""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from snowforecast.cache import CachedPredictor


@dataclass
class ElevationBandForecast:
    """Forecast for a single elevation band."""
    elevation_m: float
    temp_c: float
    snow_depth_cm: float
    new_snow_cm: float
    precip_type: str  # "snow", "rain", "mixed"


@dataclass
class ElevationBandResult:
    """Complete elevation band forecast for a resort."""
    base: ElevationBandForecast
    mid: ElevationBandForecast
    summit: ElevationBandForecast
    snow_line_m: float  # Elevation where temp = 0°C


# Temperature lapse rate: ~6.5°C per 1000m elevation gain
LAPSE_RATE_C_PER_KM = -6.5


def compute_elevation_bands(
    lat: float,
    lon: float,
    base_elev: float,
    summit_elev: float,
    target_date: datetime,
    predictor: Optional[CachedPredictor] = None,
) -> ElevationBandResult:
    """
    Compute forecasts for 3 elevation levels.

    Args:
        lat: Latitude
        lon: Longitude
        base_elev: Resort base elevation in meters
        summit_elev: Resort summit elevation in meters
        target_date: Target forecast datetime
        predictor: CachedPredictor instance (creates one if None)

    Returns:
        ElevationBandResult with base, mid, summit forecasts
    """
    if predictor is None:
        predictor = CachedPredictor()

    mid_elev = (base_elev + summit_elev) / 2

    # Get base forecast from predictor
    base_forecast, base_ci = predictor.predict(lat, lon, target_date, forecast_hours=24)

    # Estimate base temperature (if not directly available, use a reasonable default)
    # In reality, we'd extract this from HRRR/NBM data
    base_temp_c = -5.0  # Placeholder - should come from forecast data

    bands = {}
    for name, elev in [("base", base_elev), ("mid", mid_elev), ("summit", summit_elev)]:
        # Apply lapse rate adjustment
        elev_diff_km = (elev - base_elev) / 1000
        temp_c = base_temp_c + (elev_diff_km * LAPSE_RATE_C_PER_KM)

        # Determine precipitation type based on temperature
        if temp_c < -2:
            precip_type = "snow"
        elif temp_c > 2:
            precip_type = "rain"
        else:
            precip_type = "mixed"

        # Adjust snow amounts based on elevation (higher = more snow)
        elev_factor = 1 + (elev - base_elev) / 2000 * 0.2  # +20% per 2000m

        bands[name] = ElevationBandForecast(
            elevation_m=elev,
            temp_c=round(temp_c, 1),
            snow_depth_cm=round(base_forecast.snow_depth_cm * elev_factor, 1),
            new_snow_cm=round(base_forecast.new_snow_cm * elev_factor, 1) if temp_c < 0 else 0,
            precip_type=precip_type,
        )

    # Calculate snow line (elevation where temp = 0°C)
    # snow_line = base_elev + (base_temp / -LAPSE_RATE) * 1000
    snow_line_m = base_elev + (base_temp_c / -LAPSE_RATE_C_PER_KM) * 1000
    snow_line_m = max(base_elev, min(summit_elev, snow_line_m))

    return ElevationBandResult(
        base=bands["base"],
        mid=bands["mid"],
        summit=bands["summit"],
        snow_line_m=round(snow_line_m, 0),
    )


def get_snow_line(base_temp_c: float, base_elev: float) -> float:
    """
    Calculate elevation where temperature = 0°C (rain/snow boundary).

    Args:
        base_temp_c: Temperature at base elevation in Celsius
        base_elev: Base elevation in meters

    Returns:
        Snow line elevation in meters
    """
    if base_temp_c <= 0:
        return base_elev  # Already snowing at base

    # How many km up until temp = 0?
    km_to_freezing = base_temp_c / -LAPSE_RATE_C_PER_KM
    return base_elev + (km_to_freezing * 1000)
```

### `src/snowforecast/dashboard/components/elevation_bands.py`
```python
"""UI component for elevation band display."""

import streamlit as st
from snowforecast.cache.elevation_bands import ElevationBandResult


def render_elevation_bands(result: ElevationBandResult) -> None:
    """Render elevation band forecast table."""

    st.markdown("### Forecast by Elevation")
    st.caption(f"Snow line: {result.snow_line_m:.0f}m")

    # Create table data
    data = []
    for name, band in [("Summit", result.summit), ("Mid", result.mid), ("Base", result.base)]:
        # Precip icon
        if band.precip_type == "snow":
            precip_icon = "❄️"
        elif band.precip_type == "rain":
            precip_icon = "🌧️"
        else:
            precip_icon = "🌨️"

        data.append({
            "Elevation": f"{name} ({band.elevation_m:.0f}m)",
            "Temp": f"{band.temp_c:+.1f}°C",
            "New Snow": f"{band.new_snow_cm:.0f}cm" if band.new_snow_cm > 0 else "-",
            "Type": precip_icon,
        })

    st.table(data)
```

## Display Format (like Mountain-Forecast.com)
| Elevation | Temp | New Snow | Type |
|-----------|------|----------|------|
| Summit (3200m) | -8°C | 15cm | ❄️ |
| Mid (2600m) | -4°C | 12cm | ❄️ |
| Base (2000m) | 0°C | - | 🌨️ |

Snow line: 2400m

## Acceptance Criteria
- [ ] 3 elevation bands per resort (base, mid, summit)
- [ ] Lapse rate correctly applied (~6.5°C/1000m)
- [ ] Snow line calculated and displayed
- [ ] Precip type indicator (snow/rain/mixed)
- [ ] Pre-computed (uses CachedPredictor)

## Worktree
Work in: `/Users/patrickkavanagh/snowforecast-worktrees/elevation-bands`
Branch: `phase6/39-elevation-bands`
