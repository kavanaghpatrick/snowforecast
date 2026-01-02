# Product Requirements Document: Snowforecast Dashboard V2

> **Version**: 2.1
> **Created**: January 2, 2026
> **Status**: Approved (with modifications)
> **Reviewed By**: Gemini, Technical Feasibility Agent

---

## Review Summary

### Critical Changes from Review

1. **Timeline Animation Descoped** - Streamlit+PyDeck cannot achieve smooth animation (<100ms/frame). Changed to discrete time-step buttons instead of slider.

2. **3D Terrain Texture** - Must use OpenTopoMap tiles (not Mapbox) for free tier compliance. Requires validation spike.

3. **Color Scale as Dependency** - P1.2 (Color Scale) must complete BEFORE parallel work begins (used by all visual tasks).

4. **Pre-compute Elevation Bands** - Move elevation interpolation to batch pipeline, not on-demand calculation.

5. **Added Missing Features**:
   - Natural language forecast summaries (LLM-generated)
   - Freezing level / rain-snow line visualization
   - Local storage favorites (no account needed)
   - AM/PM/Night time blocks (not hourly)

---

## Executive Summary

Redesign the Snowforecast dashboard with interactive mapping, 3D terrain visualization, and professional weather app UX patterns. All features must use **free/open data sources only** (no paid APIs).

### Goals
1. Add interactive map with ski resort markers and snow depth visualization
2. Implement 3D terrain view for mountain elevation context
3. Improve forecast display with elevation-band predictions
4. Add timeline/animation controls for multi-day forecasts
5. Mobile-responsive design

### Non-Goals
- Real-time streaming data (batch updates are sufficient)
- User accounts or personalization
- Paid API integrations (Mapbox, Google Maps, etc.)

---

## Technical Architecture

### Stack
- **Frontend**: Streamlit (existing)
- **Mapping**: PyDeck (primary), Folium (secondary)
- **Data**: NOAA HRRR/NBM via Herbie, SNOTEL, Copernicus DEM
- **Tiles**: OpenTopoMap (basemap), AWS Terrain Tiles (elevation)
- **Cache**: DuckDB (existing)

### Free Data Sources

| Component | Source | Cost |
|-----------|--------|------|
| Basemap tiles | OpenTopoMap / CartoDB | Free |
| Terrain elevation | AWS Terrain Tiles | Free |
| Weather forecasts | NOAA HRRR/NBM | Free |
| Snow observations | SNOTEL | Free |
| Elevation data | Copernicus DEM | Free |
| Ski area locations | OpenSkiMap | Free |

---

## Feature Specifications

### Phase 1: Interactive Resort Map

#### P1.1: Base Map with Resort Markers
**Priority**: P0 (Must Have)

Display all 22 ski resorts on an interactive map with:
- OpenTopoMap basemap (topographic contours)
- Clickable markers for each resort
- Color-coded by current snow depth (blue-purple gradient)
- Size proportional to new snow forecast
- Tooltip on hover: resort name, snow depth, new snow, probability

**Implementation**:
```python
# PyDeck ScatterplotLayer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=resort_data,
    get_position=['lon', 'lat'],
    get_fill_color='snow_color',  # Blue-purple gradient
    get_radius='new_snow_cm * 50',
    pickable=True,
)
```

**Acceptance Criteria**:
- [ ] Map loads in <2 seconds
- [ ] All 22 resorts visible with correct positions
- [ ] Colors accurately reflect snow depth categories
- [ ] Tooltips display on hover
- [ ] Click selects resort and updates detail panel

#### P1.2: Snow Depth Color Scale
**Priority**: P0

Implement consistent color scale across all visualizations:

| Category | Depth (cm) | Color | Hex |
|----------|------------|-------|-----|
| Trace | 0-10 | Very light blue | #E6F3FF |
| Light | 10-30 | Light blue | #ADD8E6 |
| Moderate | 30-60 | Cornflower | #6495ED |
| Heavy | 60-100 | Royal blue | #4169E1 |
| Very Heavy | 100-150 | Medium blue | #0000CD |
| Extreme | >150 | Purple | #8A2BE2 |

**Acceptance Criteria**:
- [ ] Color scale implemented as reusable function
- [ ] Legend displayed on map
- [ ] Same colors used in tables and charts

#### P1.3: Resort Detail Panel
**Priority**: P0

When a resort is selected, show detailed forecast panel:
- Resort name and elevation
- 7-day forecast table (existing, improved styling)
- Mini snow depth chart
- Current conditions summary

**Acceptance Criteria**:
- [ ] Panel updates when resort selected on map
- [ ] Smooth transition animation
- [ ] Mobile-friendly layout

---

### Phase 2: 3D Terrain Visualization

#### P2.1: Terrain Layer
**Priority**: P1 (Should Have)

Add 3D terrain visualization using AWS Terrain Tiles:

```python
terrain_layer = pdk.Layer(
    "TerrainLayer",
    elevation_data="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    elevation_decoder={
        "rScaler": 256,
        "gScaler": 1,
        "bScaler": 1/256,
        "offset": -32768
    },
    texture=None,  # Or satellite imagery
)
```

**Features**:
- Toggle between 2D and 3D view
- Adjustable pitch and bearing
- Exaggeration control for terrain relief

**Acceptance Criteria**:
- [ ] 3D terrain renders correctly for Western US
- [ ] Performance: <5 second initial load
- [ ] Smooth rotation and zoom
- [ ] Toggle button switches between 2D/3D

#### P2.2: Elevation-Based Forecast Display
**Priority**: P1

Show forecasts at multiple elevation bands (like Mountain-Forecast.com):
- Base elevation (resort base)
- Mid-mountain
- Summit elevation

**Data Source**: Use Copernicus DEM for actual elevations, interpolate HRRR/NBM forecasts.

**Acceptance Criteria**:
- [ ] Three elevation bands displayed per resort
- [ ] Temperature lapse rate applied correctly (~6.5°C/1000m)
- [ ] Snow line elevation shown

---

### Phase 3: Forecast Timeline & Animation

#### P3.1: Timeline Slider
**Priority**: P1

Horizontal timeline control for navigating forecast hours:
- 7-day range (168 hours)
- 3-hour increments for days 1-2
- 6-hour increments for days 3-7
- Play/pause animation button

**UI Reference**: Windy.com bottom timeline

**Acceptance Criteria**:
- [ ] Slider updates map colors in real-time
- [ ] Current time indicator
- [ ] Animation plays at 1 frame/second
- [ ] Performance: <100ms per frame update

#### P3.2: Forecast Heatmap Layer
**Priority**: P2 (Nice to Have)

Overlay showing predicted snowfall as color gradient across the map:
- Interpolated from resort point forecasts
- Semi-transparent to show terrain beneath
- Updates with timeline slider

**Implementation**: PyDeck HexagonLayer or custom GridLayer

**Acceptance Criteria**:
- [ ] Smooth gradient across mountain regions
- [ ] Transparency allows terrain visibility
- [ ] Performance acceptable with animation

---

### Phase 4: Enhanced Data Display

#### P4.1: Snow Probability Visualization
**Priority**: P1

Display forecast uncertainty:
- Confidence intervals shown as error bars
- Color intensity reflects probability
- "High confidence" vs "Low confidence" badges

**Acceptance Criteria**:
- [ ] CI shown in forecast tables
- [ ] Visual indicator for low-confidence forecasts
- [ ] Tooltip explains confidence level

#### P4.2: Historical Comparison
**Priority**: P2

Show current conditions vs historical averages:
- "X% of normal snowpack"
- Comparison chart for season-to-date
- Data source: SNOTEL historical

**Acceptance Criteria**:
- [ ] Historical data fetched from SNOTEL
- [ ] Percentage calculation accurate
- [ ] Chart shows YTD snowfall vs average

#### P4.3: Snow Quality Indicators
**Priority**: P2

Display snow quality metrics:
- Snow-to-liquid ratio (SLR) from HRRR
- Temperature trend (warming/cooling)
- "Powder" vs "Packed" classification

**Acceptance Criteria**:
- [ ] SLR displayed where available
- [ ] Quality badge on each resort
- [ ] Explanation tooltip

---

### Phase 5: Mobile & Polish

#### P5.1: Responsive Layout
**Priority**: P1

Ensure dashboard works on mobile devices:
- Single-column layout on small screens
- Touch-friendly map controls
- Collapsible sections

**Acceptance Criteria**:
- [ ] Usable on 375px width screens
- [ ] No horizontal scrolling required
- [ ] Touch gestures work for map

#### P5.2: Performance Optimization
**Priority**: P1

Optimize for fast loading:
- Lazy load map layers
- Cache rendered tiles
- Minimize API calls

**Acceptance Criteria**:
- [ ] Initial load <3 seconds
- [ ] Subsequent navigations <1 second
- [ ] No UI freezing during data fetch

#### P5.3: Error States & Loading
**Priority**: P1

Handle errors gracefully:
- Loading spinners during data fetch
- Error messages for failed requests
- Fallback display when NOAA unavailable

**Acceptance Criteria**:
- [ ] Loading indicator visible during fetch
- [ ] Clear error message if data unavailable
- [ ] Retry button for failed requests

---

## Implementation Phases

### Phase 1: Core Map (Issues #35-38)
- P1.1: Base map with markers
- P1.2: Color scale implementation
- P1.3: Resort detail panel
- Integration and testing

**Estimated Effort**: 4 parallel tasks, 1-2 days each

### Phase 2: 3D Terrain (Issues #39-40)
- P2.1: Terrain layer
- P2.2: Elevation-band forecasts

**Estimated Effort**: 2 parallel tasks, 2-3 days each

### Phase 3: Timeline (Issues #41-42)
- P3.1: Timeline slider
- P3.2: Forecast heatmap

**Estimated Effort**: 2 parallel tasks, 2-3 days each

### Phase 4: Data Enhancements (Issues #43-45)
- P4.1: Probability visualization
- P4.2: Historical comparison
- P4.3: Snow quality indicators

**Estimated Effort**: 3 parallel tasks, 1-2 days each

### Phase 5: Polish (Issues #46-48)
- P5.1: Responsive layout
- P5.2: Performance optimization
- P5.3: Error states

**Estimated Effort**: 3 parallel tasks, 1-2 days each

---

## File Structure

```
src/snowforecast/
├── dashboard/
│   ├── app.py                 # Main Streamlit app (existing)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── map_view.py        # PyDeck map component
│   │   ├── terrain_layer.py   # 3D terrain
│   │   ├── timeline.py        # Forecast timeline
│   │   ├── resort_detail.py   # Detail panel
│   │   └── color_scales.py    # Shared color utilities
│   └── pages/
│       ├── overview.py        # Main dashboard
│       ├── map.py             # Full-screen map view
│       └── resort.py          # Single resort detail
├── visualization/
│   ├── __init__.py
│   ├── pydeck_layers.py       # PyDeck layer factories
│   ├── folium_maps.py         # Folium map builders
│   └── colors.py              # Color scale definitions
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Page load time | <3 seconds | Lighthouse |
| Map interaction FPS | >30 FPS | Chrome DevTools |
| Mobile usability | Score >90 | Lighthouse |
| Data accuracy | Matches NOAA | Manual verification |
| User engagement | >2 min avg session | Analytics |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AWS Terrain Tiles slow | Poor 3D performance | Cache tiles, lazy load |
| NOAA API downtime | No forecast data | Show cached data with timestamp |
| PyDeck Streamlit bugs | Map not rendering | Fallback to Folium |
| Mobile performance | Unusable on phones | Disable 3D on mobile |

---

## Dependencies

- `pydeck>=0.8.0` - Map visualization
- `streamlit>=1.28.0` - Dashboard framework
- `streamlit-folium>=0.15.0` - Folium integration (fallback)
- `branca>=0.6.0` - Color scales for Folium

---

## Review Checklist

- [ ] Grok review for critical issues
- [ ] Gemini review for architecture
- [ ] Codex review for implementation feasibility
- [ ] Team sign-off on scope
- [ ] GitHub issues created
- [ ] Parallel development setup complete
