# PRD: Add Japanese Ski Areas

## Summary
Add 10-12 Japanese ski resorts to the snow forecast dashboard using existing data sources.

## Feasibility Assessment: HIGH

| Requirement | Status | Notes |
|-------------|--------|-------|
| Open-Meteo forecasts | ✅ Works | Tested Niseko: returns 7-day snowfall data |
| Snow depth source | ✅ OnTheSnow | Same scraper we use for USA |
| Code changes | Minimal | Add resort list + region processor |

## Data Sources

### Forecasts: Open-Meteo (same as USA/Europe)
- JMA model available at 5km resolution
- Tested API call for Niseko returns valid snowfall data
- No code changes needed - existing `fetch_open_meteo()` works globally

### Snow Depth: OnTheSnow Japan
Confirmed coverage for 11 resorts:
- Hokkaido: Niseko United, Furano, Rusutsu, Kiroro
- Nagano: Hakuba Valley, Happo-One, Nozawa Onsen, Shiga Kogen
- Other: Myoko Kogen

URL pattern: `https://www.onthesnow.com/{region}/{resort-slug}/skireport`

## Resorts to Add (10 total)

### Hokkaido (4)
| Resort | Lat | Lon | Base Elev | OnTheSnow Slug |
|--------|-----|-----|-----------|----------------|
| Niseko United | 42.8635 | 140.6981 | 260m | hokkaido/niseko-united |
| Furano | 43.2818 | 142.4735 | 245m | hokkaido/furano |
| Rusutsu | 42.75 | 140.88 | 400m | hokkaido/rusutsu |
| Kiroro | 43.0701 | 140.9891 | 520m | hokkaido/kiroro |

### Nagano (4)
| Resort | Lat | Lon | Base Elev | OnTheSnow Slug |
|--------|-----|-----|-----------|----------------|
| Hakuba Valley | 36.70 | 137.83 | 760m | nagano/hakuba-valley |
| Nozawa Onsen | 36.9228 | 138.4406 | 565m | nagano/nozawa-onsen |
| Shiga Kogen | 36.70 | 138.50 | 1300m | nagano/shiga-kogen |
| Myoko Kogen | 36.88 | 138.12 | 731m | niigata/myoko-kogen |

### Niigata/Other (2)
| Resort | Lat | Lon | Base Elev | OnTheSnow Slug |
|--------|-----|-----|-----------|----------------|
| Happo-One | 36.70 | 137.827 | 760m | nagano/happo-one |
| Zao Onsen | 38.16 | 140.40 | 780m | yamagata/zao-onsen |

## Implementation

### Step 1: Add Japan resort data to fetch_data_v2.py
```python
JAPAN_SKI_AREAS = [
    # Hokkaido
    ("Niseko United", 42.8635, 140.6981, "Hokkaido", 260, "hokkaido/niseko-united"),
    ("Furano", 43.2818, 142.4735, "Hokkaido", 245, "hokkaido/furano"),
    ...
]
```

### Step 2: Add process_japan_region() function
- Similar to `process_us_region()` but simpler (no SNOTEL)
- Use OnTheSnow for snow depth (same as USA fallback)
- Use Open-Meteo for forecasts

### Step 3: Update main() to include Japan
```python
# Japan
logger.info("\n--- Japan (OnTheSnow + Open-Meteo) ---")
japan_resorts = process_japan_region()
all_resorts.extend(japan_resorts)
```

### Step 4: Add vertical drops for summit calculations
```python
# In elevation_bands.py
RESORT_VERTICAL_M = {
    ...
    # Japan - Hokkaido
    "Niseko United": 1048,  # 260m base + 1048m = 1308m summit
    "Furano": 829,          # 245m base + 829m = 1074m summit
    ...
}
```

## Files to Modify
1. `scripts/fetch_data_v2.py` - Add JAPAN_SKI_AREAS, process_japan_region()
2. `src/snowforecast/cache/elevation_bands.py` - Add Japan vertical drops

## Estimated Effort
- ~50 lines of code
- 30 minutes implementation
- Uses existing infrastructure

## Success Criteria
- 10 Japanese resorts appear in dashboard
- Forecasts match OpenSnow/snow-forecast.com (±20%)
- Snow depth from OnTheSnow displays correctly
