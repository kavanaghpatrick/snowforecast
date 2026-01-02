# Mountain-Forecast.com Analysis

**Research Date**: 2026-01-01
**Screenshots Location**: `/Users/patrickkavanagh/snowforecast/research/screenshots/mountain-forecast/`

## Overview

Mountain-Forecast.com is a specialized mountain weather forecasting service that provides detailed weather predictions for peaks and ski resorts worldwide. The service focuses on climbers, mountaineers, and outdoor enthusiasts who need altitude-specific forecasts.

---

## Key Features Analyzed

### 1. Elevation-Based Forecast System

Mountain-Forecast.com provides distinct forecasts for **multiple elevations** on each mountain. For Mount Rainier, the available elevations are:

| Level | Elevation | Description |
|-------|-----------|-------------|
| Peak | 4392m | Summit level |
| High | 3500m | Upper slopes |
| Mid | 2500m | Mid-mountain |
| Base | 1500m | Lower elevations |

**URL Structure**: `/peaks/{Mountain-Name}/forecasts/{elevation}`
- Example: `/peaks/Mount-Rainier/forecasts/4392` (summit)
- Example: `/peaks/Mount-Rainier/forecasts/2500` (mid elevation)

**Technical Implementation**:
- Elevation data is stored in JavaScript: `window.FCGON = {"level":"4392","levels":["4392","3500","2500","1500"]}`
- CSS variable for active elevation styling: `--active-elevation-color: hsl(201, 20%, 23%)`
- Tabs allow quick switching between elevation levels without page reload

### 2. Forecast Table Layout (6-Day Multi-Day View)

The main forecast is presented in a **horizontal table spanning 6 days**, with each day divided into 3 time periods:

| Time Period | Description |
|-------------|-------------|
| AM | Morning hours |
| PM | Afternoon hours |
| night | Evening/overnight hours |

**This creates 18 forecast columns (6 days x 3 periods)**.

#### Table Row Structure (Top to Bottom):

1. **Weather Icons Row**: Visual icons showing conditions (cloudy, snow showers, heavy snow, etc.)
2. **Weather Description Row**: Text labels like "mod. snow", "snow shwrs", "cloudy", "heavy snow"
3. **Wind Speed Row (km/h)**: Numerical values (e.g., 20, 45, 65, 90)
4. **Wind Direction Arrows**: Visual arrow indicators showing wind direction
5. **Snow Amount Row (cm)**: Predicted snowfall for each period (e.g., 7, 29, 5, 25)
6. **Rain Amount Row (mm)**: Rainfall predictions (shown as "---" when all snow)
7. **Max Temperature Row (C)**: Maximum temperatures (e.g., -11, -17, -19)
8. **Min Temperature Row (C)**: Minimum temperatures (e.g., -11, -13, -21)
9. **Wind Chill Row (C)**: Feels-like temperatures (e.g., -20, -27, -40)
10. **Freezing Level Row (m)**: Elevation where temp hits 0C (e.g., 2250, 1800, 1050)
11. **Cloud Base Row (m)**: Cloud base altitude (e.g., 2850, 1500, 1450)
12. **Multi-Elevation Temperature Grid**: Temperature at sea level through 5000m
13. **Sunrise/Sunset Rows**: Times for each day

### 3. Weather Summary Sections

Two distinct summary panels appear above the table:

**Days 1-3 Weather Summary**:
> "A heavy fall of snow, heaviest during Sat night. Extremely cold (max -11C on Thu morning, min -19C on Sat morning). Winds increasing (moderate winds from the SSW on Thu morning, severe gales from the SSW by Sat night)."

**Days 4-6 Weather Summary**:
> "A heavy fall of snow, heaviest during Tue night. Extremely cold (max -17C on Mon night, min -23C on Tue night). Winds increasing (strong winds from the NW on Mon afternoon, stormy winds from the WSW by Tue night)."

### 4. Color Coding System

**Temperature Visualization**:
- Based on the screenshots, the forecast table uses a color gradient system
- Colder temperatures appear in cooler blue/purple tones
- The multi-elevation temperature grid at the bottom uses a clear color gradient

**Snow Amount Highlighting**:
- Higher snow amounts appear to be emphasized
- Values of 25-29cm are visually prominent in the table

**Wind Speed Coloring**:
- Wind speeds appear to be color-coded by intensity
- Higher winds (70-90 km/h) use different coloring than moderate winds (20-50 km/h)

### 5. Additional Data Sections

#### Live Weather Station Data

Below the forecast table, Mountain-Forecast displays **live conditions from nearby weather stations**:

| Station | Distance | Elevation | Data Points |
|---------|----------|-----------|-------------|
| Pierce County Airport | 48 km WNW | 164m | Temp, weather, wind, visibility, cloud layers |
| Black Diamond | 51 km NW | 192m | Limited data |
| Stampede Pass Airport | 58 km NE | 628m | Temp, weather, wind |
| Tacoma/McChord AFB | 61 km W | 217m | Temp, weather, wind, visibility |

**Data Fields Available**:
- Temperature (C)
- Current weather conditions (Drizzle, Light snow, Light rain)
- Wind speed and direction
- Gusts
- Visibility (km)
- Cloud cover by level (Low/Mid/High: few/scattered/overcast)

#### Mountain Information Section
- Latitude/Longitude: 46.85 N 121.76 W
- Mountain Range: Cascade Range
- Parent Range: Pacific Coast Ranges
- Country: United States
- Photos section with user-submitted images

---

## Mobile vs Desktop Differences

### Desktop View (1920x1080)
- Full horizontal forecast table visible
- All 18 forecast columns (6 days x 3 periods) displayed side-by-side
- Elevation tabs prominently displayed on the left sidebar
- Weather station data shown in a detailed table format
- Photo gallery displayed horizontally

### Mobile View (390x844 iPhone viewport)
- **Privacy consent dialog dominates initial viewport** (significant UX issue)
- Forecast table requires horizontal scrolling
- Content is vertically stacked
- Elevation selector becomes a dropdown or compact element
- Weather summaries displayed prominently before the table
- Same data available but reorganized for touch interaction

**Key Mobile Observations**:
- The consent/privacy dialog completely obscures content on first load
- Users must scroll horizontally to view full 6-day forecast
- Data density is reduced compared to desktop

---

## Data Source Information

From the page metadata:
- **Forecast Update Frequency**: Updates every ~2-3 hours (countdown timer shown)
- **Forecast Issuance**: "Issued: 9 am Thu 01 Jan Local Time"
- **Update Countdown**: "Updates in: 2hr 27min 51s"
- **Data URL**: `/peaks/Mount-Rainier/forecasts/data` (JSON API endpoint)

---

## Technical Implementation Notes

### JavaScript Data Structure
```javascript
window.FCGON = {
  "dataUrl": "/peaks/Mount-Rainier/forecasts/data",
  "level": "4392",
  "levels": ["4392", "3500", "2500", "1500"],
  "forecast_update_ts": 1767317353,
  "server_time": 1767308477,
  "maps": [{
    "lat": 46.8529,
    "lng": -121.7601,
    "topoUrl": "https://jura.snow-forecast.com/osm_tiles/{z}/{x}/{y}.png"
  }]
}
```

### CSS Classes Used
- `.forecast-content` - Main container
- `.forecast-content__main` - Primary content area
- `.forecast-content__sidebar` - Elevation selector sidebar
- `.forecast-content__issued` - Issuance timestamp display

### Units Configuration
```javascript
let height_units = 'm'
let imperial = "true"
```

---

## Key Design Patterns for Snowforecast Project

### Applicable to Our ML Model Dashboard:

1. **Multi-Elevation Forecasts**: Display predictions at multiple elevation bands (e.g., base, mid, summit) - directly applicable to our ski resort predictions

2. **6-Day Table Layout**: Horizontal scrolling table with AM/PM/night periods is an effective way to show temporal forecasts

3. **Summary Text**: Natural language summaries above detailed tables help users quickly understand conditions

4. **Color Coding**: Temperature and snow amount color gradients improve scannability

5. **Live Station Data**: Showing nearby weather station observations adds credibility and allows comparison

6. **Freezing Level Display**: Critical for snow sports - indicates where rain turns to snow

7. **Wind Chill**: Important safety metric for mountain activities

### Differences from Our Approach:

| Mountain-Forecast | Our ML Model |
|-------------------|--------------|
| Point forecasts for peaks | Ski resort-focused predictions |
| Multiple elevations per peak | Single location predictions |
| 6-day hourly breakdown | 24-hour snowfall focus |
| Traditional NWP model | ML ensemble approach |
| Global coverage | Western US focus |

---

## Screenshot Inventory

| File | Description |
|------|-------------|
| `01_mount_rainier_full_desktop.png` | Full page desktop view of Mount Rainier forecast |
| `02_mount_rainier_above_fold.png` | Desktop above-the-fold view |
| `03_forecast_table_element.png` | Isolated forecast table element |
| `05_mount_rainier_mobile_full.png` | Full page mobile view |
| `06_mount_rainier_mobile_viewport.png` | Mobile viewport (shows consent dialog) |
| `07_whistler_full_desktop.png` | Whistler Mountain forecast page |
| `08_homepage.png` | Mountain-Forecast homepage |
| `09_rainier_summit_4392m.png` | Summit elevation (4392m) forecast |
| `10_rainier_base_2000m.png` | Base elevation (2000m) forecast |
| `11_rainier_mid_3000m.png` | Mid elevation (3000m) forecast |
| `page_structure.html` | Raw HTML for analysis |
| `visible_text.txt` | Extracted visible text content |

---

## Recommendations for Snowforecast Dashboard

Based on this analysis, consider implementing:

1. **Elevation-band forecasts**: Show predictions for base, mid-mountain, and summit elevations
2. **Confidence indicators**: Add uncertainty bands (Mountain-Forecast doesn't show this)
3. **Color-coded snow amounts**: Use intuitive color gradients for snowfall predictions
4. **Summary text generation**: Auto-generate natural language summaries from model output
5. **Comparison with observations**: Show how predictions compare to actual station data
6. **Responsive design**: Ensure mobile users can access forecasts easily
7. **Update timestamps**: Clearly show when predictions were generated and when they update

---

## API Potential

Mountain-Forecast appears to have a data API endpoint at:
- `/peaks/{Mountain-Name}/forecasts/data`

This could potentially be used for:
- Comparison/validation of our ML model predictions
- Baseline forecasts for evaluation
- Historical accuracy analysis (if data is archived)

**Note**: Terms of service should be reviewed before any API usage.
