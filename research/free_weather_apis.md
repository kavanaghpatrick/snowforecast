# Free Weather and Snow Data APIs Research

*Last updated: January 2, 2026*

This document catalogs all free weather and snow data APIs for the snowforecast project, organized by cost model and suitability for our mountain snowfall prediction use case.

---

## Table of Contents

1. [Summary Comparison Table](#summary-comparison-table)
2. [Completely Free (US Government)](#completely-free-us-government)
3. [Completely Free (No API Key)](#completely-free-no-api-key)
4. [Free Tier (API Key Required)](#free-tier-api-key-required)
5. [Regional/Country-Specific Free APIs](#regionalcountry-specific-free-apis)
6. [Python Library Support](#python-library-support)
7. [Recommendations for Snowforecast](#recommendations-for-snowforecast)

---

## Summary Comparison Table

| API | Completely Free? | API Key Required? | Rate Limit | Snow Variables | Best For |
|-----|------------------|-------------------|------------|----------------|----------|
| **NOAA (weather.gov)** | Yes | No | Generous (unpublished) | Precipitation forecasts | US forecast data |
| **NOAA NCEI CDO** | Yes | Yes (free) | 10K/day, 5/sec | SNOW, SNWD historical | Historical analysis |
| **NOAA HRRR/NBM** | Yes | No | Unlimited (bulk) | Full snow suite | ML training data |
| **Open-Meteo** | Yes (non-commercial) | No | 10K/day | Snowfall, snow depth | Quick prototyping |
| **Pirate Weather** | Yes (10K/month) | Yes (free) | 10K/month | Extensive snow fields | Dark Sky replacement |
| **OpenWeatherMap** | Free tier | Yes | 1M/month, 60/min | Basic snow | Wide coverage |
| **Tomorrow.io** | Free tier | Yes | ~500/day | Snow intensity | Minute-level forecasts |
| **Visual Crossing** | Free tier | Yes | 1K records/day | Snow, snow depth | Historical + forecast |
| **WeatherAPI.com** | Free tier | Yes | 1M/month | Basic precip | Simple integration |
| **Bright Sky (DWD)** | Yes | No | High (unpublished) | Limited | Germany only |
| **Environment Canada** | Yes | No | Unlimited | Snow layer | Canada only |

---

## Completely Free (US Government)

### 1. NOAA Weather.gov API

**URL**: https://api.weather.gov/

**Cost Model**: Completely free, US government public data

**API Key**: NOT required

**Rate Limits**:
- Not publicly disclosed but "generous for typical use"
- Rate limit errors can be retried after ~5 seconds
- Recommends User-Agent header with contact info

**Snow/Precipitation Data**:
- 7-day forecasts at 2.5km grid resolution
- Precipitation probability and amounts
- Weather conditions including snow
- Hourly and 12-hour period forecasts

**Endpoints**:
```
GET /gridpoints/{office}/{gridX},{gridY}/forecast
GET /gridpoints/{office}/{gridX},{gridY}/forecast/hourly
GET /alerts/active
GET /stations/{stationId}/observations
```

**Coverage**: United States only

**Python Support**:
- `noaa-sdk` package
- Direct HTTP requests (simple REST API)

**Pros**:
- No authentication
- Official government data
- Real-time updates
- Free for any purpose including commercial

**Cons**:
- US only
- Grid-based (need to convert lat/lon to grid)
- Raw model output not directly available

---

### 2. NOAA NCEI Climate Data Online (CDO) API

**URL**: https://www.ncei.noaa.gov/cdo-web/api/v2/

**Cost Model**: Completely free

**API Key**: Required (free registration)

**Rate Limits**:
- 5 requests per second
- 10,000 requests per day

**Snow Variables**:
- `SNOW` - Snowfall in mm
- `SNWD` - Snow depth in mm
- Historical daily summaries
- Monthly and yearly aggregations

**Endpoints**:
```
GET /datasets - List available datasets
GET /data - Fetch actual weather data
GET /stations - List weather stations
GET /datatypes - Available data types
```

**Available Datasets**:
- `GHCND` - Global Historical Climatology Network Daily
- `GSOM` - Global Summary of the Month
- `GSOY` - Global Summary of the Year

**Coverage**: Global (historical data)

**Python Support**:
- `noaa-cdo` package
- Direct HTTP requests

**Best For**: Historical analysis, training data

---

### 3. NOAA HRRR via Herbie

**URL**: NOMADS/AWS/Google Cloud (bulk access)

**Cost Model**: Completely free (open government data)

**API Key**: NOT required

**Rate Limits**: None (bulk file access)

**Snow Variables** (GRIB2 fields):
- `SNOD` - Snow depth (m)
- `WEASD` - Water equivalent of accumulated snow depth (kg/m^2)
- `ASNOW` - Accumulated snow (kg/m^2)
- `SNOWC` - Snow cover (%)
- Freezing level height
- Precipitation type

**Access Methods**:
```python
from herbie import Herbie

H = Herbie("2025-01-01", model="hrrr", product="sfc", fxx=6)
ds = H.xarray("SNOD")  # Snow depth
```

**Resolution**: 3km (HRRR), 13km (RAP), 0.25deg (GFS)

**Coverage**:
- HRRR: CONUS only
- GFS: Global

**Python Support**: `herbie-data` package (excellent)

**Best For**: ML training, high-resolution forecasts

---

### 4. NOAA National Blend of Models (NBM)

**URL**: AWS S3 / NOMADS

**Cost Model**: Completely free

**API Key**: NOT required

**Rate Limits**: None (bulk file access)

**Snow Variables** (as of NBM v4.3, May 2025):
- Snow accumulation probabilistic guidance
- Snow level (wet-bulb zero level)
- Ice/freezing rain amounts
- Precipitation type probabilities

**Resolution**:
- CONUS: 2.5 km
- Alaska: 3.0 km

**Access**:
```python
from herbie import Herbie
H = Herbie("2025-01-01", model="nbm", fxx=12)
```

**Best For**: Calibrated probabilistic snow forecasts

---

### 5. NOAA GFS (Global Forecast System)

**URL**: NOMADS / AWS / Open-Meteo

**Cost Model**: Completely free

**API Key**: NOT required

**Snow Variables**:
- Total precipitation
- Snow depth
- Categorical precipitation type
- Freezing level

**Resolution**: 0.25 degree (~25km)

**Forecast Range**: 16 days

**Python Support**: `herbie-data`, `cfgrib`, `xarray`

---

## Completely Free (No API Key)

### 1. Open-Meteo

**URL**: https://open-meteo.com/

**Cost Model**: Free for non-commercial use (CC-BY 4.0)

**API Key**: NOT required (optional for commercial)

**Rate Limits**:
- 10,000 calls/day
- 5,000 calls/hour
- 600 calls/minute
- Fractional counting (2 weeks data = 1.5 calls)

**Snow Variables**:

*Hourly:*
- `snowfall` - Precipitation in cm/hour
- `snow_depth` - Ground accumulation in meters
- `freezing_level_height` - 0C altitude in meters

*15-Minute:*
- `snowfall` - 15-min sum in cm
- `snowfall_height` - Instant measurement

*Daily:*
- `snowfall_sum` - Daily total in cm
- `snowfall_water_equivalent_sum`

**Example Request**:
```
https://api.open-meteo.com/v1/forecast?
  latitude=39.1911&longitude=-106.8175&
  hourly=snowfall,snow_depth&
  daily=snowfall_sum
```

**Weather Models Available**:
- NOAA GFS + HRRR
- ECMWF IFS (9km, CC-BY 4.0 since Oct 2025)
- DWD ICON
- MeteoFrance Arome/Arpege
- JMA MSM/GSM
- GEM HRDPS (Canada)
- MET Norway

**Python Support**: `openmeteo-requests`, `open-meteo` packages

**Pros**:
- No API key needed
- Multiple weather models
- Historical data available
- JSON responses
- CORS enabled

**Cons**:
- Non-commercial only for free tier
- Rate limited

**Best For**: Rapid prototyping, non-commercial projects

---

### 2. Bright Sky (DWD Germany)

**URL**: https://brightsky.dev/

**Cost Model**: Completely free

**API Key**: NOT required

**Rate Limits**: Not specified (handles 2M+ requests/day)

**Snow Variables**: Limited - primarily precipitation data

**Coverage**: Germany only (DWD data)

**Python Support**: Direct HTTP requests

**Best For**: German weather data only

---

### 3. Environment Canada (MSC GeoMet)

**URL**: https://api.weather.gc.ca/

**Cost Model**: Completely free

**API Key**: NOT required

**Rate Limits**: None specified

**Data Available**:
- Real-time and archived weather
- Numerical weather forecasts (GRIB2/NetCDF)
- Radar imagery including precipitation
- Snow on ground satellite layers

**Python Support**: `env-canada` package

**Coverage**: Canada only

**Best For**: Canadian weather/snow data

---

## Free Tier (API Key Required)

### 1. Pirate Weather (Dark Sky Replacement)

**URL**: https://api.pirateweather.net/

**Cost Model**: Free tier + optional donation

**API Key**: Required (free at https://pirate-weather.apiable.io/)

**Rate Limits**:
- Free: 10,000 calls/month
- $2/month donation: 20,000 calls/month

**Snow Variables** (version 2+):
- `snowAccumulation` - Expected snow in cm
- `snowIntensity` - Current intensity cm/hr
- `snowIntensityMax` - Maximum intensity
- `currentDaySnow` - Since midnight accumulation
- `iceAccumulation` - Ice/freezing rain
- `liquidAccumulation` - Rain portion
- `precipType` - "snow", "rain", "sleet", etc.

**Weather Models Used**:
- HRRR (primary for US)
- NBM
- GFS
- GEFS
- ERA5 (historical)
- ECMWF IFS
- DWD MOSMIX

**Python Support**: `pirateweather` package

**API Compatibility**: Drop-in Dark Sky replacement

**Best For**: Detailed snow breakdown, Dark Sky migration

---

### 2. OpenWeatherMap

**URL**: https://openweathermap.org/api

**Cost Model**: Free tier with paid upgrades

**API Key**: Required (free registration)

**Rate Limits (Free Tier)**:
- 60 calls/minute
- 1,000,000 calls/month

**Free Tier Includes**:
- Current weather API
- 3-hour forecast for 5 days
- Weather maps (5 layers)
- Air Pollution API
- Geocoding API

**Snow Data**:
- `snow.1h` - Snow volume last hour (mm)
- `snow.3h` - Snow volume last 3 hours (mm)
- Weather condition codes for snow types

**Coverage**: Global

**Python Support**: `pyowm` package

**Pros**:
- High call limit
- Global coverage
- Mature ecosystem

**Cons**:
- Limited snow variables
- Credit card required for One Call 3.0

**Best For**: Global coverage, simple integration

---

### 3. Tomorrow.io (formerly Climacell)

**URL**: https://api.tomorrow.io/

**Cost Model**: Free tier with paid upgrades

**API Key**: Required (free registration)

**Rate Limits (Free Tier)**:
- ~500 calls/day (varies)
- 25 calls/hour
- 3 calls/second

**Snow Variables**:
- `snowIntensity` - mm/hr
- `sleetIntensity` - mm/hr
- `freezingRainIntensity` - mm/hr
- `precipitationProbability`
- `precipitationType`

**Forecast Range**:
- Hourly: 120 hours (5 days)
- Daily: 5 days
- Minute-by-minute: Premium only

**Python Support**: Direct HTTP or `tomorrow-io` community packages

**Best For**: Minute-level nowcasting (premium), 80+ data layers

---

### 4. Visual Crossing

**URL**: https://www.visualcrossing.com/weather-api/

**Cost Model**: Free tier with paid upgrades

**API Key**: Required (free registration)

**Rate Limits (Free Tier)**:
- 1,000 records/day
- Pay-as-you-go: $0.0001/record after

**Snow Variables**:
- `snow` - New snowfall amount
- `snowdepth` - Current ground accumulation
- Precipitation types: rain, snow, freezingrain, ice

**Features**:
- Historical data access
- 15-day forecasts
- Bulk data downloads

**Python Support**: Direct HTTP requests

**Best For**: Historical snow analysis, mixed historical/forecast

---

### 5. WeatherAPI.com

**URL**: https://www.weatherapi.com/

**Cost Model**: Free tier with paid upgrades

**API Key**: Required (free registration)

**Rate Limits (Free Tier)**:
- 1,000,000 calls/month
- No daily limit specified

**Free Tier Includes**:
- Realtime weather
- 3-day forecast
- Historical: past 7 days only
- Basic weather alerts

**Snow Data**:
- Precipitation totals
- Weather conditions
- Limited snow-specific fields

**Python Support**: Direct HTTP requests

**Best For**: High volume, simple use cases

---

## Regional/Country-Specific Free APIs

### MeteoSwiss (Switzerland)

**URL**: https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html

**Status**: Open Data since May 2025, API coming Q2 2026

**Models**: ICON-CH1-EPS, ICON-CH2-EPS, COSMO

**Access**: File downloads currently, API later

**Alternative**: Open-Meteo provides MeteoSwiss ICON data

---

### Meteo France (AROME/ARPEGE)

**URL**: https://portail-api.meteofrance.fr/

**Cost Model**: Free under Open License 2.0

**API Key**: Required (free, valid 3 years)

**Models**:
- AROME: 2.5km France, 42-hour forecast
- ARPEGE: 0.25deg regional, 4-day forecast

**Alternative**: Open-Meteo includes MeteoFrance models

---

### JMA Japan

**Status**: No official API

**Alternative**: Open-Meteo JMA endpoint (limited data due to licensing)

**Community**: `jma-client` Python package for scraping

---

## Python Library Support

| Library | Models/APIs | Install | Notes |
|---------|-------------|---------|-------|
| `herbie-data` | HRRR, GFS, NBM, RAP, ECMWF | `pip install herbie-data` | Best for NOAA GRIB2 |
| `openmeteo-requests` | Open-Meteo | `pip install openmeteo-requests` | Official client |
| `pirateweather` | Pirate Weather | `pip install pirateweather` | Dark Sky compatible |
| `pyowm` | OpenWeatherMap | `pip install pyowm` | Mature, well-documented |
| `noaa-sdk` | weather.gov | `pip install noaa-sdk` | Simple NWS API wrapper |
| `env-canada` | Environment Canada | `pip install env-canada` | Canadian data |
| `cfgrib` | Any GRIB2 | `pip install cfgrib` | Low-level GRIB reading |
| `xarray` | Any NetCDF/GRIB | `pip install xarray` | Data analysis |

---

## Recommendations for Snowforecast

### Already Using (Keep)

1. **HRRR via Herbie** - 3km US forecasts, excellent snow variables
2. **NBM via Herbie** - Calibrated probabilistic forecasts

### Add for Enhanced Coverage

1. **Open-Meteo** (Primary recommendation)
   - No API key needed
   - Access to multiple models (ECMWF, DWD, MeteoFrance)
   - Good snow variables
   - Free for non-commercial/research
   - Use for: Quick validation, international coverage

2. **NOAA CDO API** (Historical)
   - Station-based snow measurements
   - Ground truth validation
   - Use for: Training data validation

3. **Pirate Weather** (Optional)
   - Good snow variable breakdown
   - 10K/month free
   - Use for: Additional ensemble member, Dark Sky format compatibility

### Not Recommended

- **OpenWeatherMap**: Limited snow variables, requires credit card for best API
- **Tomorrow.io**: Low free tier limits
- **Bright Sky**: Germany only
- **Visual Crossing**: Record-based pricing not suitable for ML

### Data Pipeline Priority

```
1. HRRR (Herbie)      -> Primary 3km US forecasts
2. NBM (Herbie)       -> Probabilistic guidance
3. GFS (Herbie)       -> Extended range (16 day)
4. Open-Meteo         -> Multi-model ensemble, validation
5. NCEI CDO           -> Historical ground truth
```

---

## Sources

- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [Open-Meteo Pricing](https://open-meteo.com/en/pricing)
- [Weather.gov API Documentation](https://www.weather.gov/documentation/services-web-api)
- [NOAA NCEI CDO API](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)
- [NOAA Open Data Dissemination](https://www.noaa.gov/nodd/datasets)
- [Herbie Documentation](https://herbie.readthedocs.io/)
- [OpenWeatherMap Pricing](https://openweathermap.org/price)
- [Pirate Weather API Docs](https://docs.pirateweather.net/en/latest/API/)
- [Tomorrow.io Free Plan Limits](https://support.tomorrow.io/hc/en-us/articles/20273728362644-Free-API-Plan-Rate-Limits)
- [Visual Crossing Pricing](https://www.visualcrossing.com/weather-data-editions/)
- [WeatherAPI.com Pricing](https://www.weatherapi.com/pricing.aspx)
- [Bright Sky](https://brightsky.dev/)
- [Environment Canada MSC GeoMet](https://api.weather.gc.ca/)
- [NBM on AWS](https://registry.opendata.aws/noaa-nbm/)
- [Meteo France on AWS](https://registry.opendata.aws/meteo-france-models/)
- [MeteoSwiss Open Data](https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html)
