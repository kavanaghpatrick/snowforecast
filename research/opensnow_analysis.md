# OpenSnow UI/UX Analysis

**Research Date:** January 2, 2026
**Website:** https://opensnow.com
**Purpose:** Competitive analysis for snowforecast project UI design reference

---

## Executive Summary

OpenSnow is the leading ski weather application providing snow forecasts, resort conditions, and weather maps for ski resorts globally. The platform combines automated weather model data with local "Daily Snow" expert analysis to deliver actionable forecasts for skiers and snowboarders.

---

## Pages Analyzed

Screenshots captured and stored in: `/Users/patrickkavanagh/snowforecast/research/screenshots/opensnow/`

| Screenshot | Page | Description |
|------------|------|-------------|
| `01_homepage.png` | Homepage | Landing page with value proposition |
| `01b_homepage_scrolled.png` | Homepage (scrolled) | Key features section |
| `04_parkcity.png` | Resort Detail | Park City resort forecast page |
| `05_utah_10day.png` | Daily Snow | Utah regional 10-day forecast |
| `07_map.png` | Interactive Map | Weather/snow forecast map |
| `10_mammoth.png` | Resort Detail | Mammoth Mountain full page |
| `10b_mammoth_10day.png` | Hourly Forecast | Detailed hourly data table |

---

## Key UI Components

### 1. Navigation Header
- **Design:** Dark slate blue (#3C5A78) with white text
- **Logo:** Orange snowflake icon with "OPENSNOW" text
- **Navigation items:** Pricing, Help, Log In, Create Free Account (CTA button)
- **CTA styling:** Bright blue button (#2196F3) with white text

### 2. Resort Detail Page (Primary Interface)

**Header Section:**
- Resort name with location (State, Country)
- Elevation badge showing summit/base elevations
- Favorite star and alert bell icons
- Tab navigation: Weather, Snow Summary, Snow Report, Cams, Weather Stations, Daily Snows, Avalanche Forecast, Trail Maps, Info

**Snow Summary Card:**
- Time period tabs: Prev 1-15 Days, Prev 8-15 Days, Prev 1-2 Days, Last 24 Hours, Next 1-5 Days, Next 6-10 Days, Next 11-15 Days
- Large numeric display for snow totals (e.g., "0"" with superscript inches)
- Calendar grid showing daily values with color coding
- "Powered by PILLAR" attribution

**Right Now Section:**
- Current temperature with "feels like" temperature
- Wind speed and direction
- Timestamp for data freshness

**Hourly Forecast Table:**
| Row | Data Displayed |
|-----|----------------|
| Precip Chance | Percentage with colored bars |
| Precipitation | Inches (locked for premium) |
| Snowfall | Inches (locked for premium) |
| Snow Ratio SLR | Ratio value |
| Snow Level | Elevation in feet |
| Weather | Icon representation |
| Temperature | Degrees F |
| Feels Like | Degrees F (colored: blue for cold) |
| Rel. Humidity | Percentage |
| Wind Speed | MPH with direction |
| Wind Gust | MPH |
| Cloud Cover | Percentage |

**Daily Forecast Section:**
- Toggle: Daily/Hourly view
- Columns: Day, Temp range, Weather icon, Snow amount, Precip Chance, Precip Amount, Feels Like, Relative Humidity, Wind (Speed/Gust), Cloud Cover
- Lock icons indicate premium-only data

### 3. Daily Snow (Regional Forecast)

**Layout:**
- Hero section with forecaster byline and date
- Headline: "3 Potential Storms Over Next 8 Days"
- Summary and Short Term Forecast sections
- Sidebar: "Nearby 5-Day Forecasts" with resort list

**Resort List Cards:**
- Resort name with icon
- 5-day snow total preview
- Temperature indicator
- Quick-glance formatting

**Forecaster Profile:**
- Photo, name, title
- Bio text explaining credentials and philosophy

### 4. Interactive Map

**Map Technology:** Mapbox GL JS

**Map Controls (Right Side):**
- North indicator (N button)
- 3D toggle button
- Zoom +/- controls
- Geolocation button
- Layer controls
- Fullscreen toggle

**Layer Categories (from page structure analysis):**
1. **Storm Forecasts:** Global Storm Forecast
2. **Radar:** Super-Res Radar + StormNet, Current/Forecast US Radar, Global Radar, Japan Radar
3. **Weather:** Lightning Risk, Hail Size, Temperature, Cloud Cover, Wind Gust
4. **Snow Forecasts:** Total Snowfall, 6/12/24-Hour Snowfall
5. **Snow Data:** Avalanche Forecast, Snow Depth, Snowfall (24hr/Season)
6. **Other:** Public Lands, Land Ownership, Air Quality, Active Fires, Smoke layers

**Premium Banner:** "Upgrade to Access Additional Map Layers" with orange CTA

---

## Color Scheme Analysis

### Primary Colors
| Color | Hex (Approximate) | Usage |
|-------|-------------------|-------|
| Dark Slate Blue | #3C5A78 | Header, footer, cards |
| Orange/Amber | #F5A623 | CTAs, premium badges, snow icons |
| White | #FFFFFF | Background, text on dark |
| Light Gray | #F5F5F5 | Page backgrounds |
| Text Dark | #333333 | Body text |

### Snow Amount Color Coding
Based on the calendar grid visualization:
| Amount | Color |
|--------|-------|
| 0" | Light gray/white |
| 1-3" | Light blue |
| 4-6" | Medium blue |
| 7-12" | Dark blue |
| 12"+ | Deep blue/purple |

### Temperature Color Coding
| Temp Range | Color |
|------------|-------|
| Below freezing | Blue tones |
| Above freezing | Yellow/Orange tones |

### Precipitation Chance Bars
- Uses gradient from light to dark blue based on percentage
- Higher percentages = darker blue fill

---

## Map Interaction Patterns

### Navigation
- Pan: Click and drag
- Zoom: Scroll wheel, +/- buttons
- Rotate: Right-click drag or two-finger rotate
- Tilt: Available in 3D mode

### Layer Selection
- Drawer panel slides up from bottom
- Organized by category with icons
- Toggle switches for each layer
- Premium layers show lock icon

### Location Selection
- Click on map to get forecast for point
- Resort markers are clickable
- Weather station markers available

---

## Elevation Data Display

### Resort Page
- **Summit elevation:** Displayed in header (e.g., "11,053 ft")
- **Base elevation:** Displayed alongside summit
- **Snow Level:** Shown in hourly forecast (elevation where snow changes to rain)

### Snow Level Forecasting
- Displayed as feet (e.g., "6,900", "6,700", "6,600")
- Critical for understanding if precipitation will be snow or rain at different elevations
- Updated hourly in forecast table

---

## Key Features to Consider

### 1. Multi-Model Comparison
- OpenSnow advertises comparing multiple weather models
- Allows users to see consensus or divergence in forecasts

### 2. Snow-to-Liquid Ratio (SLR)
- Displayed in hourly forecasts
- Indicates snow density (higher = fluffier powder)

### 3. Local Expert Analysis
- "Daily Snow" regional forecasts written by meteorologists
- Combines model data with local knowledge
- Published regularly with forecaster attribution

### 4. Freemium Model
- Basic forecasts free
- Premium features: Extended forecasts, detailed hourly data, advanced map layers
- Lock icons clearly indicate premium content

### 5. Historical Context
- "History" toggle on forecast tables
- Allows comparison of current conditions to historical data

---

## Design Patterns Worth Adopting

1. **Time Period Tabs:** Easy switching between forecast windows (24hr, 5-day, 10-day)
2. **Snow Calendar Grid:** Visual representation of daily snow totals over time
3. **Hourly Data Tables:** Comprehensive but scannable forecast data
4. **Premium Locking:** Clear visual indication of paid features without disrupting free experience
5. **Map Layer Organization:** Categorized layers with toggle switches
6. **Mobile-First Design:** Responsive cards and collapsible sections
7. **Data Attribution:** "How to read this data" help links

---

## Technical Observations

### Frontend Framework
- Uses Tailwind CSS (tw- class prefixes observed)
- React-based component architecture (Card__body patterns)
- Mapbox GL JS for interactive maps

### Data Sources (Inferred)
- Multiple weather models (HRRR, GFS, ECMWF likely)
- Weather station networks
- Radar imagery (NEXRAD)
- Satellite data

### Performance
- Lazy loading for map tiles
- Progressive data loading with loading states
- Caching evident (quick page transitions)

---

## Recommendations for Snowforecast Project

### UI Design
1. Adopt similar time-period tab navigation for forecast ranges
2. Implement snow calendar grid for visualizing predictions vs actuals
3. Use consistent color coding for snow amounts
4. Design clear hourly/daily forecast tables

### Data Display
1. Always show elevation context (snow level vs resort elevation)
2. Include snow-to-liquid ratio when available
3. Display confidence/uncertainty in predictions
4. Show data timestamps for freshness indication

### Map Features
1. Use Mapbox GL JS for interactive snow maps
2. Implement layer toggles for different data types
3. Allow point-click to get forecast for any location

### Differentiation Opportunities
1. ML-based prediction confidence visualization
2. Automated anomaly detection (unusual snow events)
3. Comparison with historical patterns
4. Skill score display (model accuracy metrics)

---

## Appendix: Screenshot Inventory

```
/Users/patrickkavanagh/snowforecast/research/screenshots/opensnow/
├── 01_homepage.png              # Landing page
├── 01b_homepage_scrolled.png    # Features section
├── 03_alta_resort.png           # 404 error (URL changed)
├── 04_parkcity.png              # Park City resort page
├── 05_utah_10day.png            # Utah Daily Snow regional
├── 06_colorado_state.png        # 404 error (URL changed)
├── 07_map.png                   # Interactive map
├── 08_powder.png                # 404 error (feature removed?)
├── 09_browse.png                # 404 error (URL changed)
├── 10_mammoth.png               # Mammoth resort full page
├── 10b_mammoth_10day.png        # Hourly forecast detail
├── 11_wyoming.png               # 404 error (URL changed)
└── findings.json                # Automated structure analysis
```

**Note:** Several URLs returned 404 errors with a "Yard Sale!" error page, indicating recent URL structure changes on the OpenSnow platform.
