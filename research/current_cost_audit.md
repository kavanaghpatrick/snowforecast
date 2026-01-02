# Snowforecast Cost Audit

**Date**: 2026-01-02
**Branch**: phase5/cache-integration

---

## Executive Summary

The snowforecast project currently uses **100% free data sources**. All APIs and services are either:
- Open data on AWS (no account required)
- Free government data portals (NOAA, USDA)
- Open source mapping data (OpenStreetMap-based)
- Copernicus open data (free registration required)

**Estimated Monthly Cost: $0**

---

## 1. Data Source APIs

### 1.1 NOAA HRRR (High-Resolution Rapid Refresh)

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/hrrr.py`, `src/snowforecast/cache/hrrr.py` |
| **Access Method** | AWS Open Data Registry via `herbie-data` library |
| **Cost** | **FREE** |
| **Authentication** | None required |
| **Rate Limits** | None (AWS S3 public bucket) |
| **Data Source** | `s3://noaa-hrrr-bdp-pds/` |

The Herbie library downloads HRRR GRIB2 files directly from AWS Open Data. This is completely free with no API key required.

### 1.2 ERA5-Land (Copernicus Climate Data Store)

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/era5.py` |
| **Access Method** | CDS API (`cdsapi` library) |
| **Cost** | **FREE** (registration required) |
| **Authentication** | `CDS_API_KEY` environment variable |
| **Rate Limits** | Queue-based (can be slow during peak times) |
| **Registration** | https://cds.climate.copernicus.eu |

ERA5-Land is freely available from Copernicus. Users must register for a free account and obtain an API key. The pipeline includes retry logic for queue delays.

**Note**: The `CDS_API_KEY` referenced in `claude.md` is for user setup, not a paid service.

### 1.3 Copernicus DEM (Digital Elevation Model)

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/dem.py`, `src/snowforecast/cache/terrain.py` |
| **Access Method** | AWS S3 public bucket (direct HTTPS) |
| **Cost** | **FREE** |
| **Authentication** | None required |
| **Data Source** | `https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/` |

GLO-30 DEM tiles are accessed via rasterio's GDAL/HTTPS support directly from AWS. No authentication needed.

### 1.4 SNOTEL (Snow Telemetry)

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/snotel.py` |
| **Access Method** | `metloom` library (USDA NRCS API) |
| **Cost** | **FREE** |
| **Authentication** | None required |
| **Rate Limits** | Reasonable (includes retry with backoff) |

SNOTEL data is provided free by the USDA Natural Resources Conservation Service.

### 1.5 GHCN-Daily (Global Historical Climatology Network)

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/ghcn.py` |
| **Access Method** | Direct HTTPS download from NCEI |
| **Cost** | **FREE** |
| **Authentication** | None required |
| **Base URL** | `https://www.ncei.noaa.gov/pub/data/ghcn/daily/` |

NOAA's National Centers for Environmental Information provides this data freely.

### 1.6 OpenSkiMap

| Attribute | Value |
|-----------|-------|
| **Used In** | `src/snowforecast/pipelines/openskimap.py` |
| **Access Method** | GeoJSON file download |
| **Cost** | **FREE** |
| **Authentication** | None required |
| **Data Source** | `https://tiles.openskimap.org/geojson/ski_areas.geojson` |

OpenSkiMap is based on OpenStreetMap and is freely available under open licenses.

---

## 2. Dashboard External Resources

### 2.1 Map Tiles

| Current Usage | Status |
|---------------|--------|
| **Streamlit Built-in Map** | Uses OpenStreetMap tiles (FREE) |
| **st.map()** | Default Mapbox-light style, included with Streamlit (FREE) |

The dashboard (`src/snowforecast/dashboard/app.py`) uses Streamlit's built-in `st.map()` which includes free map tiles.

### 2.2 External Libraries (CDN)

**Currently None**. The Streamlit dashboard uses:
- Python libraries installed locally (no CDN JavaScript)
- No external CSS or JS frameworks

### 2.3 Images/Icons

| Source | Usage | Cost |
|--------|-------|------|
| Emoji | Page icons, weather indicators | FREE (Unicode) |

---

## 3. Research Phase (Not in Production)

The `research/` directory contains analysis of potential future integrations. These are **NOT currently used in the codebase**:

### 3.1 Mapbox GL JS (Research Only)

Referenced in: `research/mapping_libraries.md`, `research/opensnow_analysis.md`

- Would require `MAPBOX_API_KEY` if implemented
- Free tier: 50,000 map loads/month, 100,000 geocoding requests/month
- **Status**: Research only, not integrated

### 3.2 OpenWeatherMap (Research Only)

Referenced in: `research/mapping_libraries.md`

- Would require API key if implemented
- Free tier: 1,000 calls/day
- **Status**: Research only, not integrated

---

## 4. Dependencies Cost Summary

### Python Packages (pyproject.toml)

All packages are **open source and free**:

| Category | Packages | License |
|----------|----------|---------|
| Core | numpy, pandas, xarray, pyarrow | BSD/Apache |
| Weather | metloom, herbie-data, cdsapi | MIT/Apache |
| Geospatial | rasterio, rioxarray, shapely, geojson | BSD |
| ML | lightgbm, xgboost, scikit-learn, torch | MIT/Apache/BSD |
| API | fastapi, pydantic, uvicorn | MIT |
| Dashboard | streamlit | Apache 2.0 |
| Database | duckdb | MIT |

---

## 5. Infrastructure Costs

### 5.1 Current (Development)

| Component | Cost |
|-----------|------|
| Local development | $0 |
| Git repository | $0 (GitHub free tier) |
| CI/CD | $0 (GitHub Actions free tier for public repos) |

### 5.2 Potential Production Costs

If deployed to production, consider:

| Service | Estimated Cost | Notes |
|---------|----------------|-------|
| Cloud VM (e.g., AWS t3.medium) | ~$30-50/month | For API/dashboard hosting |
| Cloud storage (S3 for cache) | ~$5-10/month | DuckDB currently uses local storage |
| Domain name | ~$12/year | Optional |

---

## 6. Rate Limits and Quotas

| Service | Rate Limit | Notes |
|---------|------------|-------|
| HRRR (AWS) | Unlimited | S3 public bucket |
| ERA5 (CDS) | Queue-based | May wait during peak times |
| SNOTEL | Reasonable | Built-in retry logic |
| GHCN | Reasonable | Per-station file downloads |
| Copernicus DEM | Unlimited | S3 public bucket |
| OpenSkiMap | Reasonable | Small file downloads |

---

## 7. Environment Variables

| Variable | Required For | Status |
|----------|--------------|--------|
| `CDS_API_KEY` | ERA5 pipeline | Optional (only if using ERA5) |
| `GROK_API_KEY` | Code review (development) | Not for production |

No paid API keys are required for the core prediction functionality.

---

## 8. Recommendations

### Current State
1. All production data sources are free
2. No paid services integrated
3. Cache layer reduces external API calls

### Future Considerations
1. **If implementing Mapbox**: Use free tier (50K loads/month) or consider open alternatives like MapLibre
2. **If implementing OpenWeatherMap**: Free tier is 1K calls/day (sufficient for MVP)
3. **Production deployment**: Budget ~$50-100/month for cloud hosting

---

## 9. Summary Table

| Category | Item | Cost | Status |
|----------|------|------|--------|
| Weather Data | NOAA HRRR | FREE | Active |
| Weather Data | ERA5-Land | FREE | Active |
| Terrain Data | Copernicus DEM | FREE | Active |
| Ground Truth | SNOTEL | FREE | Active |
| Ground Truth | GHCN-Daily | FREE | Active |
| Locations | OpenSkiMap | FREE | Active |
| Dashboard | Streamlit + OSM tiles | FREE | Active |
| Database | DuckDB (local) | FREE | Active |
| **TOTAL** | | **$0/month** | |

---

*Last updated: 2026-01-02*
