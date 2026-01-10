# PRD: 7-Day Snowfall Sparklines

## Problem
The current progress bar shows only the **total** 7-day snowfall, hiding crucial timing information. Skiers need to know **when** the snow is hitting:
- 20cm on Friday = powder day trip opportunity
- 3cm/day spread = mediocre conditions all week
- Storm arriving Day 6-7 = plan for next weekend

## Solution
Replace the progress bar with inline sparkline charts showing daily snowfall distribution.

## User Stories
1. As a skier, I want to see which day has the most snow so I can plan my trip timing
2. As a weekend warrior, I want to quickly identify resorts with Saturday/Sunday snow
3. As a storm chaser, I want to spot incoming storms (back-loaded forecasts)

## Requirements

### Must Have
- [x] Show 7-day daily snowfall as mini bar chart per resort row
- [x] Display numeric total alongside the sparkline
- [x] Maintain sorting by total snowfall
- [x] Work on mobile (compact display)

### Nice to Have
- [ ] Highlight peak day visually
- [ ] Show day labels on hover

## Technical Approach

### Option A: Streamlit's Built-in BarChartColumn (Recommended)
```python
"7-Day": st.column_config.BarChartColumn(
    "7-Day Snow",
    help="Daily snowfall forecast (cm)",
    y_min=0,
    y_max=max_daily_snow,
)
```
- Pros: Native, fast, no dependencies
- Cons: Limited customization

### Option B: Custom Altair Sparklines
- Pros: Full control over styling
- Cons: More complex, may not fit in table cell

### Decision: Option A
Use `BarChartColumn` - it's built-in, fast, and sufficient for our needs.

## Data Transformation
Current data structure already has what we need:
```json
{
  "forecast": [
    {"day": 0, "date": "2026-01-10", "new_snow_cm": 0.0},
    {"day": 1, "date": "2026-01-11", "new_snow_cm": 2.6},
    ...
  ]
}
```

Transform to: `[0.0, 2.6, 0.0, 0.0, 0.0, 0.0, 0.0]` for the sparkline column.

## UI Mockup
```
Resort          | 7-Day Snow      | Peak Day | Snowpack | Location
----------------|-----------------|----------|----------|----------
Mt. Baker       | ▁▂▇▃▁▁▁ 27cm   | Tue 14   | 213 cm   | WA, USA
Stevens Pass    | ▁▃▁▁▁▁▁  3cm   | Sun 12   | 173 cm   | WA, USA
Vail            | ▁▁▁▁▁▁▁  0cm   | —        | 76 cm    | CO, USA
```

## Implementation Steps
1. Extract daily forecast array for each resort
2. Add sparkline data column to dataframe
3. Replace ProgressColumn with BarChartColumn
4. Add total as suffix or separate column
5. Test on mobile

## Success Metrics
- Users can identify peak snow days at a glance
- No increase in page load time
- Works on mobile without horizontal scroll

## Timeline
- Implementation: 30 minutes
- Testing: 15 minutes
