# PRD: Resort Data Scraper (Revised)

## Problem

SNOTEL-based snowpack data shows 30-90% variance from resort-reported base depths.

## Solution (Revised after AI Review)

Scrape OnTheSnow.com using simple HTTP requests (no Playwright needed - data is in SSR JSON).
Integrate into existing `fetch_data_v2.py` to avoid race conditions.

## Key Findings from AI Review

| Reviewer | Finding | Action |
|----------|---------|--------|
| **Gemini** | Data overwrite bug - separate workflow would lose data | Integrate into existing workflow |
| **Gemini** | Check if SSR before using Playwright | Confirmed: OnTheSnow uses `__NEXT_DATA__` JSON |
| **Grok** | Git conflicts from concurrent workflows | Single workflow, single commit |
| **Grok** | Playwright feasible but heavy | Use `requests` instead - 10s vs 10min |

## Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                   fetch_data_v2.py                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ For each resort:                                     │   │
│  │   1. Try scrape_onthesnow() → base_depth             │   │
│  │   2. If fail, fallback to SNOTEL/GeoSphere/SLF       │   │
│  │   3. Always get Open-Meteo forecast                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│                   forecasts_v2.json                         │
└─────────────────────────────────────────────────────────────┘
```

## Technical Design

### OnTheSnow Data Extraction

OnTheSnow uses Next.js SSR. Snow data is in `<script id="__NEXT_DATA__">` JSON:

```python
def scrape_onthesnow(resort_slug: str) -> dict | None:
    """Scrape OnTheSnow for resort snow data using simple HTTP request."""
    url = f"https://www.onthesnow.com/united-states/{resort_slug}/skireport"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            html = response.read().decode('utf-8')

        # Extract __NEXT_DATA__ JSON
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', html)
        if not match:
            return None

        data = json.loads(match.group(1))
        resort = data['props']['pageProps']['fullResort']
        snow = resort.get('snow', {})

        return {
            'base_depth_cm': snow.get('middle'),  # Mid-mountain depth in cm
            'new_snow_24h_cm': snow.get('last24'),
            'new_snow_48h_cm': snow.get('last48'),
            'season_total_cm': snow.get('ytd'),
            'source': 'onthesnow.com',
        }
    except Exception as e:
        logger.warning(f"OnTheSnow scrape failed for {resort_slug}: {e}")
        return None
```

### Resort Slug Mapping

Add to `fetch_data_v2.py`:

```python
ONTHESNOW_SLUGS = {
    # USA
    "Stevens Pass": "washington/stevens-pass-resort",
    "Crystal Mountain": "washington/crystal-mountain-wa",
    "Mt. Baker": "washington/mt-baker",
    "Snoqualmie Pass": "washington/the-summit-at-snoqualmie",
    "Mt. Hood Meadows": "oregon/mt-hood-meadows",
    "Mt. Bachelor": "oregon/mt-bachelor",
    "Timberline": "oregon/timberline-ski-area",
    "Mammoth Mountain": "california/mammoth-mountain",
    "Palisades Tahoe": "california/palisades-tahoe",
    "Heavenly": "california/heavenly",
    "Kirkwood": "california/kirkwood",
    "Vail": "colorado/vail",
    "Breckenridge": "colorado/breckenridge",
    "Aspen Snowmass": "colorado/aspen-snowmass",
    "Telluride": "colorado/telluride",
    "Park City": "utah/park-city-mountain-resort",
    "Snowbird": "utah/snowbird",
    "Alta": "utah/alta",
    "Big Sky": "montana/big-sky-resort",
    "Whitefish": "montana/whitefish-mountain-resort",
    "Jackson Hole": "wyoming/jackson-hole-mountain-resort",
    "Sun Valley": "idaho/sun-valley",
    # Austria
    "St. Anton": "austria/st-anton",
    "Sölden": "austria/soelden",
    # ... etc
}
```

### Integration into fetch_data_v2.py

```python
def process_us_region():
    """Process all US ski areas with OnTheSnow scraping + SNOTEL fallback."""
    results = []

    for name, lat, lon, region, elev, stations_list in US_SKI_AREAS:
        # Try OnTheSnow first
        slug = ONTHESNOW_SLUGS.get(name)
        scraped = scrape_onthesnow(slug) if slug else None

        if scraped and scraped.get('base_depth_cm'):
            base_depth = scraped['base_depth_cm']
            source = f"onthesnow.com"
        else:
            # Fallback to SNOTEL
            base_depth, source = get_snotel_multi_station(stations_list)
            source = f"SNOTEL {source}" if source else "SNOTEL"

        # Get forecast from Open-Meteo (always)
        forecast = get_openmeteo_forecast(lat, lon)

        results.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": "USA",
            "region": region,
            "elevation_m": elev,
            "base_depth_cm": base_depth,
            "base_depth_source": source,
            "forecast": forecast,
            # ... etc
        })

    return results
```

### Workflow Changes

Keep existing schedule (3x/day is sufficient - resorts update 1-2x daily):

```yaml
name: Refresh Dashboard V2 Data

on:
  schedule:
    - cron: '0 6,12,18 * * *'  # 3x daily (unchanged)
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 15  # Increased for scraping
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install metloom pandas geopandas shapely

      - name: Fetch forecast data
        run: python scripts/fetch_data_v2.py
        continue-on-error: true

      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -f data/forecasts_v2.json
          git diff --staged --quiet || git commit -m "Update forecast data [skip ci]"
          git push
```

## Data Schema

No schema changes needed - just the source field value changes:

```json
{
  "name": "Stevens Pass",
  "base_depth_cm": 172.7,
  "base_depth_source": "onthesnow.com",  // Was: "SNOTEL Stevens Pass"
  ...
}
```

## Implementation Plan

### Phase 1: USA Resorts (Day 1)
1. Add `scrape_onthesnow()` function to `fetch_data_v2.py`
2. Add `ONTHESNOW_SLUGS` mapping for 22 US resorts
3. Update `process_us_region()` to try scraping first
4. Test locally
5. Keep existing 3x/day schedule (no workflow changes needed)

### Phase 2: European Resorts (Day 2)
1. Verify OnTheSnow has European resorts (Austria, Switzerland)
2. If yes, add slugs and scrape them too
3. If no, keep GeoSphere/SLF as primary (they're already accurate)

### Phase 3: Monitoring (Day 3)
1. Add logging to track scrape success rate
2. Monitor for first week
3. Adjust if needed

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OnTheSnow changes JSON structure | Low | Parse defensively, fallback to SNOTEL |
| Rate limiting | Low | 4 requests/day per resort is minimal |
| Site blocks our UA | Low | Use standard browser UA |

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Avg variance vs resort-reported | 40% | <15% |
| Data source | SNOTEL (watershed) | OnTheSnow (resort-reported) |
| Update frequency | 3x/day | 3x/day (unchanged) |

## Estimated Effort

| Task | Time |
|------|------|
| Add scraping function | 1 hour |
| Map all resort slugs | 2 hours |
| Test and debug | 2 hours |
| Update workflow | 30 min |
| **Total** | **~6 hours** |

## Decision: Playwright vs Requests

**Requests wins.** OnTheSnow serves data in SSR JSON (`__NEXT_DATA__`).

| Factor | Playwright | Requests |
|--------|------------|----------|
| Runtime | ~10 minutes | ~30 seconds |
| Dependencies | Heavy (Chromium) | None (stdlib) |
| GitHub Actions | Needs browser install | Just Python |
| Reliability | Browser can crash | Simple HTTP |
| Complexity | High | Low |
