# Free Snow Observation Data Sources

Research summary for mountain snowfall prediction model. All sources listed are **completely free** for non-commercial use unless otherwise noted.

---

## 1. SNOTEL (SNOw TELemetry) - USDA NRCS

**Status**: Completely Free (Public Domain)

### Overview
SNOTEL is an automated data collection network operated by the USDA Natural Resources Conservation Service (NRCS). Stations are located in remote, high-elevation mountain watersheds throughout the western United States.

### Station Count
- **900+ stations** across 11 western states including Alaska
- ~829 active stations with complete daily records
- Stations located in high-elevation mountain watersheds (ideal for ski/avalanche prediction)

### Variables Available
| Variable Code | Description | Temporal Resolution |
|--------------|-------------|---------------------|
| `SNWD_D` / `SNWD_H` | Snow depth | Daily / Hourly |
| `WTEQ_D` / `WTEQ_H` | Snow water equivalent (SWE) | Daily / Hourly |
| `TAVG_D` | Average temperature | Daily |
| `TMAX_D` / `TMIN_D` | Max/Min temperature | Daily |
| `TOBS_D` | Observed temperature | Daily |
| `PRCP_*` / `PREC_*` | Precipitation (various accumulations) | Various |
| Soil moisture | At various depths (enhanced sites) | Daily |
| Soil temperature | At various depths (enhanced sites) | Daily |
| Solar radiation | Enhanced sites only | Hourly |
| Wind speed | Enhanced sites only | Hourly |
| Relative humidity | Enhanced sites only | Hourly |

### API Access

**AWDB SOAP Web Service** (Primary API):
- Endpoint: `https://wcc.sc.egov.usda.gov/awdbWebService/services`
- User Guide: https://www.nrcs.usda.gov/sites/default/files/2023-03/AWDB%20Web%20Service%20User%20Guide.pdf
- No authentication required
- Returns XML

**CUAHSI HydroPortal** (Alternative):
- WSDL: `https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL`
- 923 sites, 114 variables
- ~877 million values (as of Nov 2025)

**Python Libraries**:
- `climata` - Convenience functions for SNOTEL queries
- `ulmo` - Maintained by UW APL
- `nrcs-api` - GitHub: EntilZha/nrcs-api

### Data License
- **Public Domain (us-pd)** - No restrictions

### Key URLs
- Interactive Map: https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-water-interactive-map
- Station List: https://wcc.sc.egov.usda.gov/nwcc/sitelist.jsp
- Data Catalog: https://catalog.data.gov/dataset/snowpack-telemetry-network-snotel

---

## 2. GHCN-Daily (Global Historical Climatology Network)

**Status**: Completely Free

### Overview
GHCN-Daily is NOAA's comprehensive daily climate observation database, integrating data from ~30 different sources worldwide. It provides the longest historical records available.

### Station Count
- **100,000+ stations** in 180 countries
- **24,000+ stations** report snowfall and/or snow depth (mostly North America)
- **20,000 stations** actively reporting in any 30-day period
- Records date back to 1832

### Variables Available
| Variable | Description | Coverage |
|----------|-------------|----------|
| SNOW | Snowfall (mm, tenths of mm) | 24,000+ stations |
| SNWD | Snow depth (mm) | 24,000+ stations |
| PRCP | Precipitation (mm) | ~50% of all stations |
| TMAX / TMIN | Max/Min temperature (tenths of degrees C) | 25,000+ stations |
| TOBS | Temperature at observation time | Many stations |

### API Access

**NOAA NCEI Web Services**:
- Base URL: https://www.ncei.noaa.gov/cdo-web/api/v2/
- Requires free API token (register at https://www.ncdc.noaa.gov/cdo-web/token)
- Rate limit: 5 requests/second, 1000 requests/day

**AWS Open Data**:
- S3 Bucket: `s3://noaa-ghcn-pds/`
- No authentication needed
- CSV format, bulk downloads available
- Registry: https://registry.opendata.aws/noaa-ghcn/

**FTP Access**:
- Host: ftp.ncei.noaa.gov
- Path: /pub/data/ghcn/daily/

### Data Format
- CSV with station metadata files
- Daily values with quality flags
- Comprehensive station history files

### Data License
- **Public Domain** - NOAA data

### Key URLs
- Documentation: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Daily Snow Access: https://www.ncei.noaa.gov/access/monitoring/daily-snow/
- AWS Registry: https://registry.opendata.aws/noaa-ghcn/

---

## 3. NOHRSC / SNODAS (National Operational Hydrologic Remote Sensing Center)

**Status**: Completely Free

### Overview
NOHRSC provides modeled and assimilated snow products combining ground-based, airborne, and satellite observations. SNODAS (Snow Data Assimilation System) is their flagship product.

### Products Available
| Product | Resolution | Coverage |
|---------|-----------|----------|
| Snow water equivalent | 1 km | CONUS |
| Snow depth | 1 km | CONUS |
| Snow pack temperature | 1 km | CONUS |
| Snow sublimation/evaporation | 1 km | CONUS |
| Blowing snow estimates | 1 km | CONUS |
| Snow cover (satellite-derived) | 1 km | CONUS |

### Temporal Coverage
- SNODAS available from **October 1, 2003** to present
- Updated **daily** (24-hour temporal resolution)
- Hourly model runs behind the scenes

### API Access

**NOAA Map Service (REST)**:
- Endpoint: https://mapservices.weather.noaa.gov/raster/rest/services/snow/NOHRSC_Snow_Analysis/MapServer
- Returns GeoJSON, image exports
- Two layers: SWE and Snow Depth

**NSIDC Data Access**:
- SNODAS at NSIDC: https://nsidc.org/data/g02158/versions/1
- Gridded NetCDF/GeoTIFF format
- Bulk download available

**Direct FTP**:
- Archived data: https://www.nohrsc.noaa.gov/archived_data/

### Data License
- **Public Domain** - NOAA/NWS data

### Key URLs
- Main Site: https://www.nohrsc.noaa.gov/
- National Snow Analyses: https://www.nohrsc.noaa.gov/nsa/
- SNODAS at NSIDC: https://nsidc.org/data/g02158/versions/1

---

## 4. Ski Resort Snow Reports

### Free APIs Available

**Open-Meteo** (Recommended):
- Status: **Completely Free** (non-commercial), no API key required
- Snow variables in forecast data
- 80+ years of historical data via ERA5
- Based on NOAA GFS, HRRR, ECMWF, and other models
- URL: https://open-meteo.com/
- License: CC BY 4.0

**Weather Unlocked Ski Resort API**:
- Free tier available with registration
- Mountain forecasts for base/mid/upper elevations
- 4 updates daily
- Snowfall predictions (24h, 48h, 72h, weekly)
- URL: https://developer.weatherunlocked.com/skiresort

**SnoCountry API**:
- Free demo key available (`SnoCountry.example`)
- Returns up to 3 resorts per request with demo key
- Full access requires paid subscription
- URL: http://feeds.snocountry.net/

**MyWeather2**:
- Free weather XML/JSON feeds
- Ski snow reports and forecasts
- Global resort coverage
- URL: https://www.myweather2.com/developer/

### Paid APIs (for reference)
- **Mountain News Partner API (OnTheSnow)**: Fee-based, comprehensive resort data
- **Ski API (skiapi.com)**: Lift status and snow conditions

### Scraping Considerations
- **Not Recommended**: Most ski resort websites prohibit scraping in their ToS
- **Legal Risk**: Potential CFAA violations, cease-and-desist letters
- **Better Approach**: Use official APIs listed above

---

## 5. MesoWest / Synoptic Data

**Status**: Free tier available, commercial options

### Overview
MesoWest aggregates data from 170,000+ weather stations across 320+ networks worldwide. In 2025, significant changes occurred with Synoptic Data taking over API services.

### 2025 Changes
- Legacy MesoWest user accounts retired
- Interactive Data Download Tool limited
- Core web services (maps, graphs) still available
- API services now through Synoptic Data

### Free Access Options

**Synoptic Data Free Tier**:
- 3 API requests per day
- Access to 170,000+ stations
- 160+ weather variables
- Registration required
- URL: https://synopticdata.com/

**MesoWest Direct**:
- Real-time CSV: https://mesowest.utah.edu/data/mesowest.dat.gz
- Updated every 15 minutes
- Bulk data access limited

### Variables (Snow-Related)
- Not all stations report snow
- Temperature, precipitation, wind common
- Snow depth available at select stations

### Key URLs
- MesoWest: https://mesowest.utah.edu/
- Synoptic API Docs: https://docs.synopticdata.com/
- Synoptic Viewer: https://viewer.synopticdata.com/

---

## 6. RAWS (Remote Automatic Weather Stations)

**Status**: Free

### Overview
RAWS stations are primarily designed for fire weather monitoring, operated by the National Interagency Fire Center (NIFC). About 2,800 units across the US, with 1,700 managed by NIFC.

### Standard Variables
- Wind speed and direction
- Air temperature
- Precipitation
- Relative humidity
- Solar radiation
- Fuel moisture

### Snow Data Availability
**Limited**: Snow depth sensors are **optional add-ons**, not standard equipment. Most RAWS stations do not report snow. For snow-specific data, SNOTEL is preferred.

### API Access

**Western Regional Climate Center Archive**:
- URL: https://raws.dri.edu/
- Still under construction
- Historical data available

**NIFC Open Data**:
- URL: https://data-nifc.opendata.arcgis.com/datasets/29185087b4594a35abe059cbdbf97ee4_1/about
- Station locations as GeoJSON
- Metadata only (not observations)

**RAWSmet R Package**:
- GitHub: MazamaScience/RAWSmet
- Funded by USFS AirFire
- Access to metadata and timeseries
- CEFA and DRI endpoints

### Key URLs
- RAWS Home: https://raws.nifc.gov/
- RAWS Archive: https://raws.dri.edu/
- NIFC Open Data: https://data-nifc.opendata.arcgis.com/

---

## 7. Citizen Weather Networks

### CoCoRaHS (Community Collaborative Rain, Hail and Snow Network)

**Status**: Completely Free

**Overview**:
- 27,500+ active volunteer observers
- Coverage: US, Canada, Puerto Rico, Virgin Islands, Guam, Bahamas
- Daily precipitation, snowfall, and snow depth reports
- 69+ million daily reports in database

**Data Access**:
- Data Export Tool: https://data.cocorahs.org/cocorahs/export/exportmanager.aspx
- Data Explorer (DEx): https://dex.cocorahs.org/
- Python package: `CoCoRaHS-Download-Tool` (PyPI)
- API Endpoint: `http://data.cocorahs.org/cocorahs/export/exportreports.aspx`
- Formats: CSV, XML

**Note**: Stations with 100+ observations are also in GHCN-Daily

**URLs**:
- Main: https://www.cocorahs.org/
- View Data: https://www.cocorahs.org/ViewData/

### CWOP (Citizen Weather Observer Program)

**Status**: Free

**Overview**:
- 7,000+ North American stations
- 50,000-75,000 observations per hour
- Data flows into MADIS

**Snow Data**: Some stations include snow observations. MADIS provides dedicated snow dataset.

**URLs**:
- Main: http://www.wxqa.com/
- Join: https://www.weather.gov/pub/JoinCWOP

### Weather Underground PWS Network

**Status**: Free for PWS contributors

**Overview**:
- 250,000+ personal weather stations globally
- Quality controlled observations
- Updates as often as every 2.5 seconds

**API Access**:
- Free API for PWS owners/contributors
- Includes: current obs, 5-day forecast, PWS historical data
- Rate limit: 1,500 calls/day, 30/minute
- Registration: https://www.wunderground.com/signup
- Paid tier: $200/month for higher volumes (non-contributors)

**URLs**:
- PWS Overview: https://www.wunderground.com/pws/overview

---

## 8. MADIS (Meteorological Assimilation Data Ingest System)

**Status**: Free with application

### Overview
MADIS is NOAA's comprehensive observational database aggregating 26,000+ weather stations from multiple networks including CWOP, MesoWest, and others.

### Snow Dataset
MADIS maintains a dedicated **Snow dataset** separate from atmospheric and hydrological data:
- Snow depth
- Snow water equivalent

### Access Requirements
- **Application Required**: https://madis.ncep.noaa.gov/data_application.shtml
- Data available from July 2001 to present
- Some datasets have restrictions

### Access Methods
- FTP bulk download (recommended for continuous feeds)
- LDM (Local Data Manager) for real-time
- Text/XML Viewer for on-demand queries
- API with NetCDF files

### Key URLs
- Main: https://madis.ncep.noaa.gov/
- Data Portal: https://madis-data.ncep.noaa.gov/
- Data Application: https://madis.ncep.noaa.gov/data_application.shtml

---

## Summary Comparison

| Source | Stations | Snow Depth | SWE | Historical | API | Best For |
|--------|----------|------------|-----|------------|-----|----------|
| **SNOTEL** | 900+ | Yes | Yes | 30+ years | SOAP, REST | Mountain SWE/depth |
| **GHCN-Daily** | 24,000+ | Yes | No | 175+ years | REST, S3 | Long-term historical |
| **SNODAS** | Gridded | Yes | Yes | 2003+ | REST, FTP | Spatial coverage |
| **Open-Meteo** | Model | Yes* | No | 80+ years | REST | Free forecasts |
| **MesoWest/Synoptic** | 170,000+ | Limited | No | Varies | REST | Dense coverage |
| **CoCoRaHS** | 27,500+ | Yes | No | 1998+ | REST | Citizen reports |
| **MADIS** | 26,000+ | Yes | Yes | 2001+ | FTP, API | Aggregated data |
| **RAWS** | 2,800 | Rare | No | Varies | REST | Fire weather |

*Model-derived, not observed

---

## Recommended Data Pipeline for Snowforecast

### Primary Sources (High Priority)
1. **SNOTEL** - Best for mountain snow conditions, SWE, hourly data
2. **GHCN-Daily** - Best for historical training data, broad coverage
3. **SNODAS** - Best for gridded spatial analysis

### Secondary Sources (Enhancement)
4. **CoCoRaHS** - Additional point observations
5. **Open-Meteo** - Forecast integration, model data

### Optional (If Needed)
6. **MesoWest/Synoptic** - Dense station network (limited snow)
7. **MADIS** - If broader aggregation needed

---

## References

- [SNOTEL Interactive Map](https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-water-interactive-map)
- [AWDB Web Service User Guide](https://www.nrcs.usda.gov/sites/default/files/2023-03/AWDB%20Web%20Service%20User%20Guide.pdf)
- [GHCN-Daily Documentation](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
- [GHCN-Daily on AWS](https://registry.opendata.aws/noaa-ghcn/)
- [NOHRSC Main Site](https://www.nohrsc.noaa.gov/)
- [SNODAS at NSIDC](https://nsidc.org/data/g02158/versions/1)
- [Open-Meteo API](https://open-meteo.com/)
- [MesoWest 2025 Changes](https://mesowest.utah.edu/product_change_notice_2025.html)
- [Synoptic Data](https://synopticdata.com/)
- [CoCoRaHS Data Explorer](https://dex.cocorahs.org/)
- [CWOP Information](https://www.weather.gov/pub/JoinCWOP)
- [Weather Underground PWS](https://www.wunderground.com/pws/overview)
- [MADIS Portal](https://madis.ncep.noaa.gov/)
- [RAWS Home](https://raws.nifc.gov/)
