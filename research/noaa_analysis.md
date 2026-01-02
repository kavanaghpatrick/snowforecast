# NOAA Weather Visualization Analysis

> Research conducted January 2, 2026 using Playwright to capture screenshots from weather.gov, NOAA, and related visualization tools.

## Executive Summary

This document analyzes NOAA/NWS visualization approaches for snow forecasts, including color scales, uncertainty representation, map projections, and data overlay techniques. The findings will inform the design of visualization components for the snowforecast project.

---

## 1. Key NOAA Snow Forecast Products

### 1.1 Weather Prediction Center (WPC)
**URL**: https://www.wpc.ncep.noaa.gov/wwd/winter_wx.shtml

The WPC is the authoritative source for official U.S. snow forecasts. Key products include:

| Product | Description | Update Frequency |
|---------|-------------|------------------|
| Day 1-3 Snow Forecasts | Deterministic accumulation maps | Twice daily (09Z, 21Z) |
| Probabilistic Winter Precip | Probability of exceeding thresholds (4", 8", 12") | Twice daily |
| Winter Storm Severity Index | Impact-based severity classification | As needed |
| QPF (Quantitative Precip Forecast) | Liquid equivalent precipitation | Multiple times daily |

**Screenshot**: `screenshots/noaa/wpc_snow_forecast.png`

### 1.2 NOHRSC National Snow Analyses
**URL**: https://www.nohrsc.noaa.gov/nsa/

The National Operational Hydrologic Remote Sensing Center provides:

- **Snow Depth Analysis** - Current observed snow depth (inches)
- **Snow Water Equivalent (SWE)** - Water content in snowpack (inches)
- **24-hour Snowfall** - Recent snowfall accumulation
- **Snow Melt** - Rate of snowpack decrease
- **Mean Snowpack Temperature** - Thermal state of snow

**Screenshot**: `screenshots/noaa/nohrsc_snow_analysis.png`, `screenshots/noaa/nohrsc_interactive.png`

### 1.3 National Digital Forecast Database (NDFD)
**URL**: https://graphical.weather.gov/

Provides gridded forecast data including:
- Snow amount forecasts (6-hour and total)
- Probability of precipitation
- Weather type forecasts

**Screenshot**: `screenshots/noaa/nws_graphical.png`, `screenshots/noaa/nws_ndfd_conus.png`

### 1.4 National Blend of Models (NBM)
**URL**: https://www.weather.gov/mdl/nbm_text

Statistical post-processing of multiple model outputs to create consensus forecasts with uncertainty quantification.

**Screenshot**: `screenshots/noaa/nbm_snow.png`

---

## 2. Official Color Scales for Snow Amounts

### 2.1 WPC Probabilistic Snow - Threshold Categories

The WPC uses a three-tier probability classification system:

| Risk Level | Probability Range | Color | Usage |
|------------|-------------------|-------|-------|
| **Slight (SLGT)** | 10-39% | Yellow/Light Orange | Low confidence of threshold exceedance |
| **Moderate (MDT)** | 40-69% | Orange/Red | Moderate confidence |
| **High** | 70-100% | Dark Red/Magenta | High confidence of significant snow |

Thresholds applied: >= 4", >= 8", >= 12" in 24 hours

### 2.2 NOHRSC Snow Depth Color Scale

Based on the NOHRSC interactive viewer, snow depth uses a blue-to-purple progression:

| Depth Range (inches) | Color Description |
|---------------------|-------------------|
| 0 - 0.39 | Very light blue / near-white |
| 0.4 - 0.8 | Light blue |
| 0.8 - 1.6 | Sky blue |
| 1.6 - 3.1 | Medium blue |
| 3.1 - 6.3 | Darker blue |
| 6.3 - 12 | Deep blue |
| 12 - 24 | Blue-purple transition |
| 24 - 49 | Purple |
| 49 - 98 | Deep purple |
| 98 - 197 | Dark purple / magenta |
| 197 - 394 | Very dark purple |
| 394+ | Near black / deep magenta |

**Key observation**: NOHRSC uses a logarithmic-like scale where color bands expand at higher depths, appropriate for the wide range of snow depths (0 to 400+ inches).

### 2.3 NWS Temperature Color Scale (Reference)

The NWS graphical forecasts use a rainbow spectrum for temperature:

| Temperature | Color |
|-------------|-------|
| Very Cold (< 0F) | Deep purple |
| Cold (0-20F) | Purple to blue |
| Cool (20-40F) | Blue to cyan |
| Mild (40-60F) | Green to yellow |
| Warm (60-80F) | Yellow to orange |
| Hot (80-100F) | Orange to red |
| Very Hot (>100F) | Deep red to magenta |

### 2.4 Precipitation Amount Color Scale (QPF)

Standard NWS liquid precipitation colors:

| Amount (inches) | Color |
|-----------------|-------|
| Trace - 0.10 | Light green |
| 0.10 - 0.25 | Green |
| 0.25 - 0.50 | Dark green |
| 0.50 - 0.75 | Yellow |
| 0.75 - 1.00 | Light orange |
| 1.00 - 1.50 | Orange |
| 1.50 - 2.00 | Red |
| 2.00 - 3.00 | Dark red |
| 3.00+ | Purple/Magenta |

---

## 3. Forecast Uncertainty Display Methods

### 3.1 Probabilistic Forecasts (Primary Method)

The WPC's Probabilistic Winter Precipitation Forecast (PWPF) is the gold standard:

**Methodology**:
1. Multi-model ensemble (61+ members) including:
   - NCEP GEFS (10 members)
   - ECMWF ensemble (25 members)
   - Canadian ensemble (10 members)
   - High-resolution models (HRRR, NAM, WRF, FV3LAM)
   - Deterministic models (GFS, ECMWF operational)
   - WPC manual forecast (as distribution mode)

2. Statistical distribution fitting using binormal PDF allowing skewness

3. Output products:
   - **Probability of Exceedance Maps**: P(snow >= X inches)
   - **Percentile Accumulation Maps**: Amount at 10th, 25th, 50th, 75th, 90th percentile

**Visualization**:
- Filled contour maps with probability percentages
- Color intensity increases with probability
- Clear threshold labels (4", 8", 12")

**Screenshot**: `screenshots/noaa/wpc_5day_snow.png`

### 3.2 Categorical Risk Levels

The SLGT/MDT/HIGH system simplifies probabilistic information:

```
SLIGHT:   10-39% chance  ->  "Snow possible but not guaranteed"
MODERATE: 40-69% chance  ->  "Snow likely"
HIGH:     70-100% chance ->  "Snow very likely to expected"
```

### 3.3 Percentile Spreads

NBM and ensemble products show uncertainty through percentile ranges:

- **10th percentile**: Low-end scenario (90% chance of exceeding)
- **50th percentile**: Median/most likely
- **90th percentile**: High-end scenario (10% chance of exceeding)

The spread between percentiles indicates forecast confidence:
- Narrow spread = high confidence
- Wide spread = low confidence / high uncertainty

### 3.4 Time-Based Uncertainty

Uncertainty visualization changes by forecast lead time:
- **Day 1**: Detailed deterministic maps with probability overlays
- **Day 2-3**: Probabilistic maps with broader categories
- **Day 4-7**: Extended outlooks with large uncertainty bounds

---

## 4. Map Projection and Styling

### 4.1 Common Projections Used

| Product | Projection | Notes |
|---------|------------|-------|
| WPC Products | Lambert Conformal Conic | Standard for CONUS |
| NOHRSC Maps | Lambert Conformal Conic | Same as WPC |
| NDFD Graphics | Lambert Conformal Conic | Matches NWS standards |
| ArcGIS Services | GCS_Sphere_EMEP | 6,371,200m radius sphere |

**Lambert Conformal Conic Parameters (CONUS)**:
- Standard parallels: 25N and 45N (typical)
- Central meridian: -95W (typical)
- Preserves shape well for mid-latitude CONUS

### 4.2 Map Styling Elements

**Base Map Features**:
- State boundaries (thin gray lines)
- Country boundaries (thicker gray/black)
- Major water bodies (blue fill, usually light)
- Coastlines emphasized
- No topographic shading on forecast maps

**Overlay Styling**:
- Semi-transparent color fills (60-80% opacity)
- Contour lines at major thresholds
- Clean legend placement (typically right side or bottom)
- Time stamp prominently displayed

### 4.3 Color Accessibility Considerations

NOAA maps generally:
- Use sequential color scales (light to dark)
- Avoid red-green only schemes (colorblind-safe)
- Provide numeric labels on contours
- Include clear legends with value ranges

---

## 5. Data Overlay Techniques

### 5.1 WPC Map Services

**REST Endpoints**:
```
Base URL: https://mapservices.weather.noaa.gov/vector/rest/services/

Winter Precipitation:
precip/wpc_prob_winter_precip/MapServer

Layers available:
- Day 1-3 Snow Accumulation (4", 8", 12" thresholds)
- Day 1-3 Ice Accumulation (0.25" threshold)
```

**Geographic Extent**:
- Longitude: -129.6 to -62.3 W
- Latitude: 24.9 to 52.8 N
- Max image size: 4096 x 4096 pixels

**Update Schedule**: Twice daily at 0900Z and 2100Z

### 5.2 NOHRSC Map Services

**REST Endpoint**:
```
https://mapservices.weather.noaa.gov/raster/rest/services/snow/NOHRSC_Snow_Analysis/MapServer
```

**Available Layers**:
- Snow Depth (Layer 3)
- Snow Water Equivalent (Layer 7)

**Format Support**: PNG, JPG, TIFF, PDF, GIF, SVG

### 5.3 GeoJSON/Vector Data

WPC services support:
- JSON query responses
- GeoJSON export
- PBF (Protocol Buffer) tiles

### 5.4 Integration Approaches

**Option 1: Direct Image Overlay**
- Fetch pre-rendered PNG images
- Overlay on web map with geo-referencing
- Pros: Simple, fast
- Cons: Limited interactivity

**Option 2: WMS/WMTS Tile Services**
- Use OGC-compliant tile services
- Integrate with Leaflet/OpenLayers/MapboxGL
- Pros: Standard approach, good caching
- Cons: May need custom styling

**Option 3: Feature Service Queries**
- Query ArcGIS REST services for vector data
- Render locally with custom styling
- Pros: Full control over appearance
- Cons: More complex implementation

---

## 6. Design Recommendations for Snowforecast Project

### 6.1 Color Scale Recommendations

**For Snow Depth/Accumulation**:
```python
# Recommended color scale (inches to hex)
SNOW_DEPTH_COLORS = {
    0: "#FFFFFF",      # White (no snow)
    1: "#E0F0FF",      # Very light blue
    2: "#B0D8FF",      # Light blue
    4: "#80C0FF",      # Sky blue
    6: "#50A8FF",      # Medium blue
    8: "#2090FF",      # Blue
    12: "#0078E0",     # Deep blue
    18: "#6050C0",     # Blue-purple
    24: "#8040A0",     # Purple
    36: "#A03080",     # Deep purple
    48: "#C02060",     # Magenta
    72: "#800040",     # Dark magenta
}
```

**For Probability**:
```python
PROBABILITY_COLORS = {
    "slight": "#FFE066",   # Yellow (10-39%)
    "moderate": "#FF8C00", # Orange (40-69%)
    "high": "#DC143C",     # Crimson (70-100%)
}
```

### 6.2 Uncertainty Visualization

1. **Primary**: Use filled probability contours for Day 1-3
2. **Secondary**: Show percentile spreads (10th, 50th, 90th)
3. **Tertiary**: Display ensemble spread indicator (narrow=confident)

### 6.3 Map Projection

Use **Lambert Conformal Conic** for Western US focus:
- Standard parallels: 33N and 45N
- Central meridian: -114W
- Appropriate for mountain regions

### 6.4 Styling Guidelines

- Use 70% opacity for filled contours
- Add thin black contour lines at major thresholds
- Include clear time stamps (UTC and local)
- Legend on right side with units clearly labeled
- State boundaries in thin gray (#808080)

---

## 7. Screenshots Captured

| Filename | Source | Content |
|----------|--------|---------|
| `wpc_snow_forecast.png` | WPC | Winter weather forecast overview |
| `wpc_5day_snow.png` | WPC | Probabilistic winter precipitation guidance |
| `wpc_qpf.png` | WPC | Quantitative precipitation forecasts |
| `nohrsc_snow_analysis.png` | NOHRSC | National snow analyses main page |
| `nohrsc_interactive.png` | NOHRSC | Interactive snow information viewer |
| `nohrsc_snow_depth_layer.png` | NOHRSC | Snow depth layer visualization |
| `nws_graphical.png` | NWS | Graphical forecast main page |
| `nws_ndfd_conus.png` | NWS | NDFD CONUS forecasts |
| `nws_national_snow.png` | NWS | National forecast maps |
| `nws_digital_services.png` | NWS | Digital services graphical interface |
| `nbm_snow.png` | NWS/MDL | National Blend of Models text products |
| `ncei_climate.png` | NCEI | Climate at a Glance interface |
| `wpc_winter_legends_scroll1.png` | WPC | Snowfall probability forecasts |
| `wpc_winter_legends_scroll2.png` | WPC | Extended forecasts and legends |

**Total**: 37 screenshots captured (9.9 MB)

---

## 8. API/Service Reference

### Primary Data Sources

| Service | URL | Format |
|---------|-----|--------|
| WPC Winter Precip MapServer | `mapservices.weather.noaa.gov/vector/rest/services/precip/wpc_prob_winter_precip/MapServer` | REST/JSON |
| NOHRSC Snow Analysis | `mapservices.weather.noaa.gov/raster/rest/services/snow/NOHRSC_Snow_Analysis/MapServer` | REST/Raster |
| NDFD WMS | `idpgis.ncep.noaa.gov/arcgis/services/NWS_Forecasts_Guidance_Warnings/ndfd/MapServer/WMSServer` | WMS |
| NBM Data | `nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/` | GRIB2 |

### Legend Endpoints

```
# WPC Probability legend (JSON)
https://mapservices.weather.noaa.gov/vector/rest/services/precip/wpc_prob_winter_precip/MapServer/legend?f=json

# NOHRSC Snow Analysis legend (JSON)
https://mapservices.weather.noaa.gov/raster/rest/services/snow/NOHRSC_Snow_Analysis/MapServer/legend?f=json
```

---

## 9. Key Takeaways

1. **Color Consistency**: NOAA uses consistent color scales across products - blue for snow amounts, yellow-orange-red for probability/risk levels

2. **Uncertainty is Standard**: All modern NOAA products include probabilistic information, not just single deterministic values

3. **Threshold-Based Display**: Snow forecasts commonly use exceedance thresholds (4", 8", 12") rather than continuous scales

4. **Multi-Model Ensembles**: The PWPF uses 61+ ensemble members for robust uncertainty quantification

5. **Accessible Design**: Maps include legends, numeric labels, and colorblind-friendly palettes

6. **Standard Services**: NOAA provides REST/WMS services enabling integration with modern web mapping libraries

---

## References

- [WPC Winter Weather Forecasts](https://www.wpc.ncep.noaa.gov/wwd/winter_wx.shtml)
- [NOHRSC National Snow Analyses](https://www.nohrsc.noaa.gov/nsa/)
- [NWS Graphical Forecasts](https://graphical.weather.gov/)
- [WPC Probabilistic Winter Precipitation Info](https://www.wpc.ncep.noaa.gov/pwpf/about_pwpf_productsbody.html)
- [NOAA Map Services](https://mapservices.weather.noaa.gov/)
- [NCEI Daily Snow Data](https://www.ncei.noaa.gov/access/monitoring/daily-snow/)
