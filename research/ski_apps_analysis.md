# Ski Weather Apps UI/UX Analysis

**Research Date:** January 2, 2026
**Purpose:** Identify design patterns for snow depth displays, resort comparisons, and forecast visualizations to inform snowforecast dashboard design.

---

## Sites Researched

| Site | URL | Focus | Quality |
|------|-----|-------|---------|
| **OpenSnow** | opensnow.com | Daily forecasts, resort conditions | Excellent |
| **Snow-Forecast.com** | snow-forecast.com | 6-day forecasts, global coverage | Good |
| **OnTheSnow** | onthesnow.com | Resort comparisons, trip planning | Good |
| **SnowBrains** | snowbrains.com | News, snow reports | Fair |
| **Powder Project** | powderproject.com | (Site unavailable during research) | N/A |

---

## Key Design Patterns

### 1. Snow Depth Displays

#### OpenSnow Pattern (Best in Class)
- **Primary metric**: Large, bold snowfall number (e.g., "15"") with unit clearly labeled
- **Color coding**: Blue gradient intensity correlates with snow amount
- **Elevation splits**: Base/Mid/Summit readings side by side
- **Time windows**: 24h, 48h, 5-day, 7-day tabs
- **Visual hierarchy**: Current conditions > Near-term forecast > Extended forecast

#### Snow-Forecast.com Pattern
- **Matrix layout**: Days as columns, metrics as rows
- **Color bands**: Temperature gradient (blue cold, red warm)
- **Snow icons**: Snowflake symbols with cm amounts inside
- **Freeze level**: Shown as elevation line (critical for rain/snow determination)
- **Three elevations**: Bot/Mid/Top with separate forecasts

#### OnTheSnow Pattern
- **Table format**: Resort rows with sortable columns
- **Key columns**: 24h, 48h, Base depth, Summit depth, % Open
- **Snowfall icons**: Blue snowflake badges with amounts
- **Quick filters**: "Most Snow 24h", "Most Snow 48h", "Deepest Base"

### 2. Resort Comparison Views

#### OpenSnow State View (opensnow_state_co)
```
Layout Structure:
+------------------------------------------+
| State Header (Colorado)                   |
| [All Resorts] [Ski Forecasts] [Powder] [Alerts] |
+------------------------------------------+
| Daily Snow Summary (expert forecast)      |
| "Three storms on tap - significant snow"  |
+------------------------------------------+
| Resort List (scrollable)                  |
| +--------------------------------------+ |
| | Resort Name | Current | 5-Day Forecast | |
| | [Icon] Vail |  0"     | [bar chart]    | |
| | [Icon] Breck|  0"     | [bar chart]    | |
| +--------------------------------------+ |
+------------------------------------------+
```

**Key Features:**
- Horizontal 5-day mini bar charts per resort
- Current snow amount prominently displayed
- Resort icons/logos for quick recognition
- Sticky header with filter tabs
- "Hourly 5-Day Forecasts" as expandable detail

#### OnTheSnow Comparison Table
```
| Resort          | 24h  | 48h  | Base | Summit | Open |
|-----------------|------|------|------|--------|------|
| Breckenridge    | 3"   | 5"   | 48"  | 72"    | 95%  |
| Vail            | 2"   | 4"   | 42"  | 68"    | 92%  |
| Aspen           | 4"   | 7"   | 52"  | 78"    | 100% |
```

**Key Features:**
- Sortable columns (click header to sort)
- Color-coded cells (more snow = darker blue)
- Row hover highlighting
- Quick-filter chips above table
- Export/share functionality

### 3. Historical vs Forecast Data

#### OpenSnow Approach
- **Separation**: Clear visual distinction between "Reported" and "Forecast"
- **Reported data**: Gray/white background, "as of [time]" timestamp
- **Forecast data**: Blue background, confidence indicators
- **Verification**: Links to "Forecast Verification" pages showing accuracy

#### Snow-Forecast.com Approach
- **Snow History section**: "Bluebird Powder" / "Powder days" counts
- **Season totals**: YTD snowfall vs average
- **Historical comparison**: Current year line vs historical range band
- **Time series**: Small sparkline showing season progression

### 4. Mobile-Friendly Designs

#### OpenSnow Mobile (Best Mobile UX)
```
+------------------+
| [Logo]    [Menu] |
+------------------+
| Colorado Daily   |
| Snow             |
+------------------+
| Expert Summary   |
| "Three storms    |
| on tap..."       |
+------------------+
| [Prev] [Refresh] |
+------------------+
| About Forecaster |
| [Avatar] Joel    |
| Gratz            |
+------------------+
| Recent News      |
| [Card] [Card]    |
+------------------+
```

**Mobile Patterns:**
- Single-column layout
- Large touch targets (44px minimum)
- Collapsible sections for detailed data
- Bottom navigation bar for key actions
- Swipe gestures for day navigation
- Pull-to-refresh for updates

#### OnTheSnow Mobile
- Card-based layout for resorts
- Horizontal scroll for multi-day forecasts
- Fixed search bar at top
- "Most Reported Snowfall" as quick-access list
- Featured webcams with live thumbnails

### 5. Weekly/Seasonal Charts

#### OpenSnow 5-Day Forecast Visualization
```
     Mon   Tue   Wed   Thu   Fri
     ---   ---   ---   ---   ---
15" |     |#####|#####|     |     |
10" |     |#####|#####|#####|     |
 5" |#####|#####|#####|#####|     |
 0" |_____|_____|_____|_____|_____|
```
- Stacked bar chart per day
- Color indicates snow type (powder vs packed)
- Hover shows exact amounts + confidence

#### Snow-Forecast.com 6-Day Matrix
```
|        | AM  | PM  | Night |
|--------|-----|-----|-------|
| Mon    | [*] | [*] | [**]  |
| Tue    | [**]| [*] | [*]   |
```
- Three periods per day (AM/PM/Night)
- Snowflake icons with amounts
- Temperature bars below
- Wind arrows with speed labels

---

## Color Palettes

### Snow Amount Color Coding
| Amount | Color | Hex |
|--------|-------|-----|
| 0" | Light gray | #E0E0E0 |
| 1-3" | Light blue | #B3D9FF |
| 4-6" | Medium blue | #66B3FF |
| 7-12" | Dark blue | #3399FF |
| 12"+ | Deep blue/purple | #0066CC |

### Temperature Color Coding
| Temp | Color | Hex |
|------|-------|-----|
| <20F | Deep blue | #0066CC |
| 20-32F | Light blue | #66B3FF |
| 32-40F | Green | #66CC66 |
| 40F+ | Yellow/Orange | #FFCC00 |

### Alert/Confidence Colors
| Status | Color | Use |
|--------|-------|-----|
| High confidence | Green | #28A745 |
| Medium confidence | Yellow | #FFC107 |
| Low confidence | Red | #DC3545 |
| Storm watch | Orange | #FD7E14 |
| Powder alert | Purple | #6F42C1 |

---

## Typography Patterns

### Headlines
- **Font**: Sans-serif (Inter, Roboto, or system fonts)
- **Size**: 24-32px for page titles
- **Weight**: Bold (700)

### Data Numbers
- **Font**: Monospace or tabular figures for alignment
- **Size**: 36-48px for hero metrics, 14-18px for table data
- **Weight**: Bold for emphasis

### Body Text
- **Font**: System sans-serif
- **Size**: 14-16px
- **Line height**: 1.5

---

## Recommendations for Snowforecast Dashboard

### Must-Have Features

1. **Multi-Resort Comparison Table**
   - Sortable columns for 24h, 48h, 5-day snowfall
   - Base/summit depth columns
   - Row click expands to detailed view
   - Filter by region/state

2. **5-Day Forecast Bar Charts**
   - Per-resort mini charts in comparison view
   - Full-width chart on resort detail page
   - Show confidence bands/ranges

3. **Current Conditions Card**
   - Large hero number for latest snowfall
   - Last updated timestamp
   - Source attribution (SNOTEL, resort report)

4. **Mobile-First Design**
   - Single-column layout
   - Card-based components
   - Bottom navigation
   - Swipe for time periods

### Nice-to-Have Features

1. **Historical Comparison**
   - Season-to-date vs average
   - Year-over-year comparison
   - "Best snow" rankings

2. **Alert System**
   - Powder day notifications
   - Storm watch badges
   - Customizable thresholds

3. **Interactive Map**
   - Color-coded resort markers
   - Click for quick stats popup
   - Regional weather overlay

### Avoid These Anti-Patterns

1. **Information overload** - Don't show all metrics at once
2. **Tiny touch targets** - Minimum 44px for mobile
3. **No loading states** - Always show skeleton/spinner
4. **Stale data indicators** - Always show "last updated" time
5. **Modal overuse** - Use inline expansion instead

---

## Screenshot Reference

All screenshots saved to: `/Users/patrickkavanagh/snowforecast/research/screenshots/ski-apps/`

### Key Screenshots for Reference

| Screenshot | What It Shows |
|------------|---------------|
| `opensnow_state_co_full.png` | Best resort comparison list |
| `onthesnow_colorado_v2_full.png` | Table-based resort comparison |
| `snowforecast_mammoth_full.png` | 6-day matrix forecast |
| `opensnow_mobile_daily_full.png` | Mobile daily snow format |
| `onthesnow_mobile_home_full.png` | Mobile resort cards |
| `snowforecast_map_usa_full.png` | Interactive snow map |
| `opensnow_colorado_full.png` | Expert forecast + resort list |

---

## Implementation Priority

### Phase 1: Core Display
1. Resort comparison table (sortable)
2. Current conditions card
3. 5-day forecast mini-charts
4. Responsive mobile layout

### Phase 2: Enhanced Features
1. Historical data visualization
2. Interactive map view
3. Alert/notification system
4. Export/share functionality

### Phase 3: Polish
1. Animations and transitions
2. Offline support (PWA)
3. Push notifications
4. Social features (compare with friends)

---

## Technical Notes

### Data Refresh Patterns
- **OpenSnow**: Real-time for reported, hourly for forecasts
- **Snow-Forecast.com**: Updates every 6 hours
- **OnTheSnow**: Resort-reported, varies by resort

### API Considerations
- Most sites use client-side rendering with REST APIs
- Weather data typically JSON format
- Image assets served via CDN
- Consider caching strategy (5-15 min for forecasts)

---

*Analysis complete. Screenshots available for detailed UI reference.*
