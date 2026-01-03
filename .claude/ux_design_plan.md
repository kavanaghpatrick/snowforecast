# Snow Forecast Dashboard - UX Design & Integration Plan

## Executive Summary

We have 15 sophisticated UI components that were implemented but never integrated into the main dashboard. This document outlines the UX design and integration plan to transform the basic dashboard into a feature-rich snow forecasting application.

---

## Current State vs Target State

### Current Dashboard (Basic)
- Simple ski area dropdown selector
- 4 basic metrics (Snow Base, New Snow, Probability, Elevation)
- Basic bar/line charts
- Static regional map
- No interactivity beyond resort selection

### Target Dashboard (Enhanced)
- Interactive PyDeck map with color-coded markers
- Time-based forecast navigation (Now → Day 7)
- Natural language forecast summaries
- Snow quality indicators (Powder/Good/Wet/Icy)
- Confidence visualization with CI ranges
- SNOTEL station integration
- Elevation band forecasts (Base/Mid/Summit)
- Favorites system
- Responsive mobile/tablet support
- Loading skeletons and error handling

---

## Available Components Audit

| Component | File | Purpose | Integration Priority |
|-----------|------|---------|---------------------|
| Resort Map | `map_view.py` | Interactive PyDeck map | HIGH |
| Resort Detail | `resort_detail.py` | Natural language summaries | HIGH |
| Time Selector | `time_selector.py` | 9-step forecast navigation | HIGH |
| Confidence | `confidence.py` | CI badges and explanations | MEDIUM |
| Snow Quality | `snow_quality.py` | SLR, powder classification | MEDIUM |
| Elevation Bands | `elevation_bands.py` | Base/Mid/Summit forecasts | MEDIUM |
| SNOTEL Display | `snotel_display.py` | Nearby observation stations | MEDIUM |
| Favorites | `favorites.py` | Save favorite resorts | LOW |
| Responsive | `responsive.py` | Mobile/tablet breakpoints | LOW |
| Performance | `performance.py` | Prefetching, timing | LOW |
| Loading | `loading.py` | Skeletons, error handling | LOW |
| Cache Status | `cache_status.py` | Data freshness indicators | LOW |
| Terrain Layer | `terrain_layer.py` | 3D terrain visualization | DEFER |
| Forecast Overlay | `forecast_overlay.py` | Heatmap overlay | DEFER |

---

## UX Design: Page Layout

### Desktop Layout (>1024px)

```
┌────────────────────────────────────────────────────────────────────┐
│  HEADER: ❄️ Snow Forecast Dashboard                    [Favorites ★]│
├────────────────────────────────────────────────────────────────────┤
│  TIME SELECTOR: [Now] [Tonight] [Tomorrow AM] [Tomorrow PM] ...    │
├──────────────────────────────┬─────────────────────────────────────┤
│                              │                                     │
│   INTERACTIVE MAP            │   RESORT DETAIL PANEL              │
│   (PyDeck with color-coded   │   ┌─────────────────────────────┐  │
│    markers, tooltips)        │   │ Resort Name, State           │  │
│                              │   │ Elevation: 2600m (8530ft)    │  │
│   [Color legend]             │   ├─────────────────────────────┤  │
│   [SNOTEL station toggle]    │   │ FORECAST SUMMARY             │  │
│                              │   │ "Heavy snow expected Tuesday │  │
│                              │   │  (15-25cm). Good conditions."│  │
│                              │   ├─────────────────────────────┤  │
│                              │   │ METRICS ROW                  │  │
│                              │   │ [Base: 0cm] [New: 15cm]      │  │
│                              │   │ [Prob: 75%] [Quality: Powder]│  │
│                              │   ├─────────────────────────────┤  │
│                              │   │ CONFIDENCE BADGE             │  │
│                              │   │ 🟢 High Confidence (±5cm)    │  │
│                              │   ├─────────────────────────────┤  │
│                              │   │ ELEVATION BANDS              │  │
│                              │   │ Summit: -8°C, 20cm snow      │  │
│                              │   │ Mid:    -5°C, 15cm snow      │  │
│                              │   │ Base:   -2°C, 10cm mixed     │  │
├──────────────────────────────┴─────────────────────────────────────┤
│  TABS: [Forecast Chart] [SNOTEL Stations] [All Resorts Table]      │
├────────────────────────────────────────────────────────────────────┤
│  TAB CONTENT AREA                                                  │
│  (7-day forecast table / SNOTEL cards / Sortable resort table)     │
└────────────────────────────────────────────────────────────────────┘
```

### Mobile Layout (<768px)

```
┌─────────────────────────────┐
│ ❄️ Snow Forecast    [★] [☰] │
├─────────────────────────────┤
│ TIME: [Now ▼]               │
├─────────────────────────────┤
│ RESORT: [Stevens Pass ▼]    │
├─────────────────────────────┤
│ QUICK STATS                 │
│ Base: 120cm | New: 15cm     │
│ 🟢 High Confidence | Powder │
├─────────────────────────────┤
│ FORECAST SUMMARY            │
│ "Heavy snow tonight..."     │
├─────────────────────────────┤
│ ELEVATION BANDS (collapsed) │
├─────────────────────────────┤
│ [View Map] [View Table]     │
└─────────────────────────────┘
```

---

## User Flow

### Primary Flow: Check Resort Forecast
1. User lands on page → Sees map with all resorts color-coded
2. User clicks resort on map OR selects from dropdown
3. Detail panel updates with:
   - Natural language forecast summary
   - Metrics with confidence badges
   - Snow quality indicator
   - Elevation band forecasts
4. User can switch time periods using time selector
5. All data updates instantly (prefetched)

### Secondary Flow: Compare Conditions
1. User expands "All Resorts" tab
2. Sees sortable table of all resorts
3. Can sort by: Snow Depth, New Snow, State
4. Can filter by state
5. Click row to select that resort

### Tertiary Flow: Check SNOTEL Validation
1. User expands "SNOTEL Stations" tab
2. Sees nearby observation stations
3. Can compare model forecast vs actual observations
4. Sees regional snowpack % of normal

---

## Integration Plan

### Phase 1: Core Layout (HIGH priority)
**Files to modify:** `app.py`
**Components to integrate:**
1. `render_resort_map()` - Replace basic st.map
2. `render_time_selector()` - Add above main content
3. `render_resort_detail()` - Replace basic metrics
4. `generate_forecast_summary()` - Add NL summaries

### Phase 2: Enhanced Metrics (MEDIUM priority)
**Components to integrate:**
1. `render_snow_quality_badge()` - Add to metrics row
2. `render_confidence_badge()` - Add below metrics
3. `render_elevation_bands()` - Add elevation section
4. `render_snotel_section()` - Add in tabs

### Phase 3: Polish (LOW priority)
**Components to integrate:**
1. `render_favorite_toggle()` - Add to header
2. `inject_responsive_css()` - Add for mobile
3. `render_loading_skeleton()` - Add during data fetch
4. `render_cache_status_badge()` - Add to sidebar

### Phase 4: Advanced (DEFER)
**Components to defer:**
1. `create_terrain_layer()` - 3D terrain (complex)
2. `create_forecast_overlay()` - Heatmap (performance concerns)

---

## Component Integration Details

### 1. Resort Map Integration

```python
# BEFORE (basic)
st.map(map_data, size="size", color="#1E90FF")

# AFTER (enhanced)
from snowforecast.dashboard.components import render_resort_map
deck = render_resort_map(conditions_df)
st.pydeck_chart(deck, use_container_width=True)
```

### 2. Time Selector Integration

```python
from snowforecast.dashboard.components import (
    render_time_selector,
    prefetch_all_forecasts,
    get_current_forecast,
    needs_prefetch,
)

# At top of main content
selected_time = render_time_selector()

# Prefetch on location change
if needs_prefetch(lat, lon):
    with st.spinner("Loading forecasts..."):
        prefetch_all_forecasts(predictor, lat, lon)

# Get forecast for selected time (instant)
forecast_data = get_current_forecast()
```

### 3. Snow Quality Integration

```python
from snowforecast.dashboard.components import (
    create_quality_metrics,
    render_snow_quality_badge,
)

# Calculate quality from forecast data
metrics = create_quality_metrics(
    snow_depth_change=forecast.new_snow_cm,
    precip_mm=forecast.precip_mm,
    temp_c=forecast.temp_c,
    hourly_temps=hourly_temps,
)

# Display badge
render_snow_quality_badge(metrics)
```

### 4. Confidence Integration

```python
from snowforecast.dashboard.components import (
    render_confidence_badge,
    render_forecast_with_confidence,
)

# Show forecast with CI
render_forecast_with_confidence(
    value=forecast.new_snow_cm,
    ci=confidence,
    probability=forecast.probability,
    label="New Snow",
)
```

---

## Data Flow Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Action    │────>│  Session State   │────>│  UI Components  │
│  (select resort,│     │  - selected_area │     │  - Map          │
│   change time)  │     │  - all_forecasts │     │  - Detail panel │
└─────────────────┘     │  - forecast_lat  │     │  - Charts       │
                        │  - forecast_lon  │     └─────────────────┘
                        └──────────────────┘
                               │
                               │ Cache miss
                               ▼
                        ┌──────────────────┐
                        │  CachedPredictor │
                        │  - DuckDB cache  │
                        │  - HRRR API      │
                        │  - Terrain API   │
                        └──────────────────┘
```

---

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `selected_time_step` | str | Current time selector value |
| `all_forecasts` | dict | Prefetched forecasts for all time steps |
| `forecast_lat` | float | Latitude of cached forecasts |
| `forecast_lon` | float | Longitude of cached forecasts |
| `selected_resort` | dict | Currently selected resort info |
| `favorites` | list | User's favorite resorts |

---

## Performance Considerations

1. **Prefetch Strategy**: On resort selection, fetch all 9 time steps at once
2. **Map Rendering**: PyDeck loads in <3s for 22 resorts
3. **Session Caching**: Use `st.session_state` to avoid refetching
4. **Lazy Loading**: SNOTEL and All Resorts tabs load on expand

---

## Open Questions for Review

1. Should the map be the primary focus, or the detail panel?
2. Is 9 time steps (Now through Day 7) too many options?
3. Should SNOTEL be prominent or hidden in a tab?
4. Mobile-first or desktop-first approach?
5. Should we show all 22 resorts on initial load or lazy load?

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyDeck rendering slow | Medium | High | Test on Streamlit Cloud first |
| Too many API calls | Medium | Medium | Aggressive prefetching |
| Complex state management | High | Medium | Clear session state structure |
| Mobile UX poor | Medium | Medium | Defer responsive to Phase 3 |
| Component import errors | Low | High | Test all imports before merge |

---

## Success Metrics

1. **Page Load**: <5s to interactive state
2. **Time Switch**: <500ms (from cache)
3. **Resort Switch**: <3s (includes prefetch)
4. **No Crashes**: 0 TypeError/ImportError on Streamlit Cloud
5. **Feature Visibility**: All HIGH priority components visible

---

## Next Steps

1. Review this plan with Gemini (architecture/UX)
2. Review with Codex (implementation feasibility)
3. Get user approval
4. Implement Phase 1
5. Deploy and test
6. Iterate on Phases 2-3
