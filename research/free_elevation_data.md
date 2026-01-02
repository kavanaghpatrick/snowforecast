# Free Elevation and DEM Data Sources

**Research Date:** January 2026
**Purpose:** Comprehensive guide to free digital elevation model (DEM) data sources for the Snowforecast project

---

## Quick Reference Table

| Dataset | Resolution | Coverage | License | Best For |
|---------|------------|----------|---------|----------|
| **Copernicus DEM GLO-30** | 30m | Global | Free/Open | Primary choice - best quality |
| **Copernicus DEM GLO-90** | 90m | Global | Free/Open | Large-area analysis |
| **USGS 3DEP 1m** | 1m | USA | Public Domain | High-detail US terrain |
| **USGS 3DEP 10m** | ~10m | USA | Public Domain | Good US coverage |
| **AWS Terrain Tiles** | Multi-res (3m-1km) | Global | Free (attribution) | Web mapping, visualization |
| **SRTM** | 30m/90m | 56S-60N | Public Domain | Legacy compatibility |
| **ASTER GDEM V3** | 30m | 83S-83N | Free | Alternative global DEM |
| **ALOS World 3D** | 30m | Global | Free (registration) | High-precision DSM |
| **GMTED2010** | 250m-1km | Global | Public Domain | Coarse global analysis |
| **EU-DEM** | 30m | Europe | Free/Open | European applications |

---

## 1. Copernicus DEM (Recommended Primary Source)

### Overview
The Copernicus DEM is a Digital Surface Model (DSM) derived from the TanDEM-X mission, representing Earth's surface including buildings, infrastructure, and vegetation. It is currently the best freely available global DEM.

### Available Products

| Product | Resolution | Coverage | Access |
|---------|------------|----------|--------|
| **GLO-30** | 30m (1 arc-second) | Global (some restrictions) | Free, open |
| **GLO-90** | 90m (3 arc-seconds) | Global | Free, open |
| **EEA-10** | 10m | Europe only | Restricted access |

### Access Details
- **Cost:** Free for GLO-30 and GLO-90
- **Registration:** Not required for basic access
- **Format:** Cloud Optimized GeoTIFF (COG)
- **Coordinate System:** WGS84 (EPSG:4326)

### Download Methods

#### 1. Direct HTTPS (No Login Required)
```bash
# GLO-30 and GLO-90 available via direct HTTPS
# Example tile: N46_00_W122_00
curl -O https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif
```

#### 2. AWS Registry (Recommended for programmatic access)
```python
# Using rasterio with AWS S3
import rasterio

# GLO-30 tiles on AWS
url = "s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N46_00_W122_00_DEM/Copernicus_DSM_COG_10_N46_00_W122_00_DEM.tif"
with rasterio.open(url) as src:
    data = src.read(1)
```

#### 3. OpenTopography API
```python
import requests

api_key = "YOUR_API_KEY"
params = {
    "demtype": "COP30",
    "south": 45.0,
    "north": 49.0,
    "west": -123.0,
    "east": -120.0,
    "outputFormat": "GTiff",
    "API_Key": api_key
}
response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)
```

#### 4. PySTAC Client
```python
# pip install stactools-cop-dem pystac-client
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox=[-123, 45, -120, 49]
)
items = list(search.get_items())
```

### Attribution
```
Contains modified Copernicus DEM data 2021-present
```

### Limitations
- GLO-30 has some restricted tiles (certain countries not publicly released)
- 10m resolution (EEA-10) restricted to eligible users only
- Represents surface elevation (DSM), not bare earth (DTM)

### Sources
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
- [AWS Registry - Copernicus DEM](https://registry.opendata.aws/copernicus-dem/)
- [OpenTopography - Copernicus GLO-30](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3)

---

## 2. USGS 3DEP (3D Elevation Program)

### Overview
The USGS 3D Elevation Program (3DEP) provides the highest-quality elevation data for the United States, derived primarily from LiDAR. All 3DEP products are **free of charge and without use restrictions**.

### Available Products

| Product | Resolution | Coverage | Notes |
|---------|------------|----------|-------|
| **S1M (Seamless 1m)** | 1m | Growing US coverage | Best resolution, production ongoing |
| **1/3 arc-second** | ~10m | Full CONUS, Alaska, Hawaii | Medium resolution, complete |
| **1 arc-second** | ~30m | Full CONUS | Lower resolution, complete |

### Access Methods

#### 1. OpenTopography API (Recommended)
```python
import requests

# Free API key from OpenTopography account
api_key = "YOUR_API_KEY"
params = {
    "demtype": "USGS10m",  # or "USGS30m", "USGS1m"
    "south": 45.0,
    "north": 46.0,
    "west": -122.0,
    "east": -121.0,
    "outputFormat": "GTiff",
    "API_Key": api_key
}
response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)
```

**Rate Limits:**
- Academic users: 300 calls per 24 hours
- Non-academic: 100 calls per 24 hours
- Area limits: 225,000 km2 (30m), 25,000 km2 (10m), 250 km2 (1m)

#### 2. USGS National Map API
```python
import requests

# Elevation point query
params = {
    "x": -122.0,
    "y": 46.0,
    "units": "Meters",
    "output": "json"
}
response = requests.get(
    "https://epqs.nationalmap.gov/v1/json",
    params=params
)
elevation = response.json()["value"]
```

#### 3. 3DEP Elevation ImageServer (WMS/WCS)
```python
# WCS endpoint for dynamic access
wcs_url = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"
# Supports hillshade, aspect, slope, contour functions
```

#### 4. AWS Cloud Access (LiDAR Point Clouds)
```bash
# USGS 3DEP LiDAR on AWS (Requester Pays for large volume)
aws s3 ls s3://usgs-lidar-public/
```

#### 5. Google Earth Engine
```python
import ee
ee.Initialize()

dem = ee.Image("USGS/3DEP/1m")
# or ee.Image("USGS/3DEP/10m")
```

### Download Tools
- [National Map Downloader](https://apps.nationalmap.gov/downloader/)
- [LidarExplorer](https://apps.nationalmap.gov/lidar-explorer/)
- [EarthExplorer](https://earthexplorer.usgs.gov/)

### Sources
- [USGS 3DEP Program](https://www.usgs.gov/3d-elevation-program)
- [OpenTopography 3DEP API](https://opentopography.org/news/api-access-usgs-3dep-rasters-now-available)
- [USGS National Map API](https://www.usgs.gov/faqs/there-api-accessing-national-map-data)

---

## 3. AWS Terrain Tiles (Mapzen)

### Overview
Terrain Tiles on AWS is a public dataset providing global DEMs aggregated from multiple sources. Originally created by Mapzen, now maintained on AWS. **Free to access** with attribution requirements.

### Data Sources (by resolution)
- **3-10m:** USGS 3DEP (NED) - United States
- **30m:** SRTM - Global
- **250m-1km:** GMTED2010 - Global
- **Bathymetry:** ETOPO1

### Available Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **Terrarium PNG** | 24-bit elevation encoding | Web visualization |
| **Normal PNG** | Surface direction + elevation | Hillshading |
| **GeoTIFF** | Raw 16-bit integer meters | Analysis |
| **Skadi HGT** | SRTM-style format | Compatibility |

### Access Endpoints
```bash
# Base URL: s3://elevation-tiles-prod/
# Formats available at zoom levels 0-15

# Terrarium (PNG)
https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

# GeoTIFF
https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif

# Skadi (HGT)
https://s3.amazonaws.com/elevation-tiles-prod/skadi/{N|S}{lat}/{N|S}{lat}{E|W}{lon}.hgt.gz
```

### Python Access
```python
# Using elevatr-style access
import requests

def get_terrain_tile(z, x, y, format="geotiff"):
    if format == "geotiff":
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
    elif format == "terrarium":
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    return requests.get(url).content

# Decode Terrarium PNG
def decode_terrarium(r, g, b):
    """Convert RGB to elevation in meters"""
    return (r * 256 + g + b / 256) - 32768
```

### Cost and Limits
- **Dataset Access:** Free
- **Transfer Costs:** Standard AWS S3 egress (requester pays for large volumes)
- **Rate Limits:** No documented hard limits
- **Total Size:** ~51.5 TB (4.6 billion tiles)

### Attribution Required
```
Terrain data from Mapzen Terrain Tiles, containing:
- 3DEP content courtesy of the U.S. Geological Survey
- SRTM and GMTED2010 content courtesy of the U.S. Geological Survey
- ETOPO1 content courtesy of U.S. National Oceanic and Atmospheric Administration
```

### Sources
- [AWS Registry - Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
- [Mapzen Terrain Tiles Blog](https://aws.amazon.com/blogs/publicsector/announcing-terrain-tiles-on-aws-a-qa-with-mapzen/)

---

## 4. SRTM (Shuttle Radar Topography Mission)

### Overview
SRTM data was collected during an 11-day Space Shuttle mission in February 2000. While older than newer alternatives, it remains widely used for its global coverage and compatibility.

### Specifications
- **Resolution:** 1 arc-second (~30m) and 3 arc-seconds (~90m)
- **Coverage:** 56S to 60N latitude (~80% of Earth's land)
- **Vertical Accuracy:** <16m absolute
- **Format:** HGT (SRTM native), GeoTIFF available

### Products Available

| Product | Resolution | Coverage |
|---------|------------|----------|
| SRTM GL1 | 30m | Near-global |
| SRTM GL3 | 90m | Near-global |
| SRTM Non-Void Filled | Various | Raw with voids |
| SRTM Void Filled | Various | Interpolated voids |

### Download Methods

#### 1. USGS EarthExplorer
```
https://earthexplorer.usgs.gov/
Digital Elevation > SRTM > Select version
```

#### 2. OpenTopography API
```python
import requests

params = {
    "demtype": "SRTMGL1",  # or "SRTMGL3"
    "south": 45.0,
    "north": 46.0,
    "west": -122.0,
    "east": -121.0,
    "outputFormat": "GTiff",
    "API_Key": "YOUR_KEY"
}
response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)
```

#### 3. Direct HGT Download
```bash
# 30m tiles from Derek Watkins mirror
curl -O https://dwtkns.com/srtm30m/data/N46W122.hgt.zip
```

### License
Public Domain - No restrictions on use.

### Sources
- [NASA Earthdata - SRTM](https://www.earthdata.nasa.gov/data/instruments/srtm)
- [USGS EROS Archive - SRTM](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1)
- [OpenTopography - SRTM](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042013.4326.1)

---

## 5. ASTER GDEM V3

### Overview
ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer) Global DEM Version 3 provides global elevation data derived from stereo imagery. **Free with registration.**

**Note:** The ASTER instrument was decommissioned August 5, 2025, but existing data remains available.

### Specifications
- **Resolution:** 1 arc-second (~30m)
- **Coverage:** 83N to 83S latitude
- **Tile Size:** 1 x 1
- **Format:** Cloud Optimized GeoTIFF (COG), NetCDF4, standard GeoTIFF

### Download Methods

#### 1. NASA Earthdata (Recommended)
```bash
# Requires free NASA Earthdata account
# https://urs.earthdata.nasa.gov/

# Using wget with .netrc authentication
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
     --keep-session-cookies \
     "https://e4ftl01.cr.usgs.gov/ASTER_B/ASTT/ASTGTMV003.003/2000.03.01/ASTGTMV003_N46W122.zip"
```

#### 2. USGS EarthExplorer
```
https://earthexplorer.usgs.gov/
Digital Elevation > ASTER
```

### License
Free for all users - requires NASA Earthdata account.

### Sources
- [NASA Earthdata - ASTER GDEM V3](https://lpdaac.usgs.gov/products/astgtmv003/)
- [ASTER GDEM Home](https://asterweb.jpl.nasa.gov/gdem.asp)

---

## 6. ALOS World 3D (AW3D30)

### Overview
ALOS World 3D is a 30-meter resolution Digital Surface Model from JAXA (Japan Aerospace Exploration Agency), derived from the PRISM sensor on the ALOS satellite. Considered one of the most precise global DSMs available.

### Specifications
- **Resolution:** 30m (1 arc-second)
- **Coverage:** Global
- **Type:** Digital Surface Model (DSM)
- **Accuracy:** High precision (better than SRTM in many areas)

### Download Methods

#### 1. JAXA ALOS Portal (Primary)
```
https://www.eorc.jaxa.jp/ALOS/en/aw3d30/
- Requires free registration
- Download tiles by region
```

#### 2. OpenTopography API
```python
import requests

params = {
    "demtype": "AW3D30",
    "south": 45.0,
    "north": 46.0,
    "west": -122.0,
    "east": -121.0,
    "outputFormat": "GTiff",
    "API_Key": "YOUR_KEY"
}
response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)
```

### License
Free for research and non-commercial use. Registration required.

### Sources
- [JAXA ALOS Portal](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/)
- [GIS Geography - Free DEM Sources](https://gisgeography.com/free-global-dem-data-sources/)

---

## 7. GMTED2010

### Overview
Global Multi-resolution Terrain Elevation Data 2010 (GMTED2010) replaces GTOPO30 as the standard for global coarse-resolution elevation data. Best for continental/global scale applications.

### Specifications

| Resolution | Ground Spacing | Best Use |
|------------|----------------|----------|
| 30 arc-seconds | ~1 km | Global overview |
| 15 arc-seconds | ~500 m | Regional analysis |
| 7.5 arc-seconds | ~250 m | Detailed continental |

### Coverage
- 84N to 56S (most products)
- 84N to 90S (some products)
- Gaps in Greenland and Antarctica at higher resolutions

### Download Methods

#### 1. USGS EarthExplorer
```
https://earthexplorer.usgs.gov/
Digital Elevation > GMTED2010
```

#### 2. Google Earth Engine
```python
import ee
ee.Initialize()

gmted = ee.Image("USGS/GMTED2010_FULL")
```

#### 3. TEMIS (NetCDF format)
```
https://www.temis.nl/data/gmted2010/
- Pre-processed NetCDF files
- Multiple resolutions available
```

### License
Public Domain - No restrictions.

### Sources
- [USGS GMTED2010](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation)
- [Google Earth Engine - GMTED2010](https://developers.google.com/earth-engine/datasets/catalog/USGS_GMTED2010_FULL)

---

## 8. EU-DEM (European Digital Elevation Model)

### Overview
EU-DEM is a hybrid DEM for Europe based on SRTM and ASTER GDEM data. **Note:** EU-DEM is no longer maintained; Copernicus DEM is recommended as the replacement.

### Specifications
- **Resolution:** 30m (1 arc-second)
- **Coverage:** 39 EEA member countries
- **Accuracy:** 2.9m RMSE vertical
- **Bias:** -0.56m

### Download
```
https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/eu-dem
```

### License
Free under Copernicus data policy (Regulation EU No 1159/2013).

### Sources
- [Eurostat EU-DEM](https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/eu-dem)
- [EEA Data Hub](https://www.eea.europa.eu/en/datahub/datahubitem-view/d08852bc-7b5f-4835-a776-08362e2fbf4b)

---

## 9. Hillshade and Terrain Visualization Tiles

### Free Hillshade Sources

#### MapTiler (Free Tier)
- **Hillshade tiles:** Zoom levels 0-12
- **Terrain RGB:** ~30m resolution
- **Access:** Free MapTiler Cloud account required
- **URL:** https://www.maptiler.com/terrain/

#### Stamen/Stadia Maps
- **Terrain tiles:** Hill shading with natural vegetation colors
- **Flavors:** Standard terrain, labels, lines, background
- **Access:** Served by Stadia Maps (as of July 2023)
- **URL:** https://maps.stamen.com/

#### OpenMapTiles
- **Open-source styles:** Including Stamen Toner port
- **Terrain GL Style:** Hill shading and contour lines
- **License:** Free and open-source (attribution required)
- **URL:** https://openmaptiles.org/styles/

### Example Usage (MapTiler)
```python
import requests

# MapTiler terrain RGB tiles
api_key = "YOUR_MAPTILER_KEY"
z, x, y = 10, 164, 395
url = f"https://api.maptiler.com/tiles/terrain-rgb/{z}/{x}/{y}.png?key={api_key}"
response = requests.get(url)
```

---

## 10. OpenTopography - Unified Access Portal

### Overview
OpenTopography provides a unified API for accessing multiple global DEM datasets. Highly recommended for programmatic access.

### Available Datasets via API

| Dataset Code | Description |
|--------------|-------------|
| `SRTMGL3` | SRTM GL3 90m |
| `SRTMGL1` | SRTM GL1 30m |
| `AW3D30` | ALOS World 3D 30m |
| `NASADEM` | NASA DEM |
| `COP30` | Copernicus GLO-30 |
| `COP90` | Copernicus GLO-90 |
| `USGS10m` | USGS 3DEP 10m |
| `USGS30m` | USGS 3DEP 30m |
| `USGS1m` | USGS 3DEP 1m (academic only) |

### API Access
```python
import requests

def download_dem(demtype, bbox, api_key, output_path):
    """
    Download DEM from OpenTopography.

    demtype: One of the dataset codes above
    bbox: (south, north, west, east)
    """
    south, north, west, east = bbox
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    response = requests.get(
        "https://portal.opentopography.org/API/globaldem",
        params=params
    )

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    return False
```

### Rate Limits
- **Academic:** 300 requests/day
- **Non-academic:** 100 requests/day
- **1m data:** Academic users only

### Get API Key
1. Create free account at https://opentopography.org/
2. Go to MyOpenTopo Dashboard
3. Click "Request API Key"

### Sources
- [OpenTopography Developers](https://opentopography.org/developers)
- [OpenTopography API Docs](https://portal.opentopography.org/apidocs/)

---

## Recommendations for Snowforecast Project

### Primary: Copernicus DEM GLO-30
- **Why:** Best quality global 30m DEM, free, cloud-optimized, no rate limits
- **Access:** AWS S3 or OpenTopography API
- **Coverage:** Complete Western US

### Fallback: USGS 3DEP 10m
- **Why:** Higher resolution for US, excellent for mountain terrain
- **Access:** OpenTopography API (300 calls/day academic)
- **Limitation:** US only

### For Visualization: AWS Terrain Tiles
- **Why:** Pre-rendered, multiple formats, free
- **Access:** Direct S3 URLs

### Implementation Priority

```python
# Recommended approach for Snowforecast
DEM_SOURCES = [
    ("COP30", "Copernicus GLO-30"),    # Primary - 30m global
    ("USGS10m", "USGS 3DEP 10m"),      # US fallback - 10m
    ("SRTMGL1", "SRTM 30m"),           # Legacy fallback - 30m
]
```

---

## Summary

| Use Case | Recommended Source |
|----------|-------------------|
| Global coverage, production | Copernicus DEM GLO-30 |
| Highest US resolution | USGS 3DEP 1m (academic) |
| Good US resolution | USGS 3DEP 10m |
| Web map visualization | AWS Terrain Tiles |
| Large area analysis | GMTED2010 or Copernicus GLO-90 |
| Unified API access | OpenTopography |
| Europe-specific | Copernicus EEA-10 (restricted) |

All sources listed are **free** for research and most commercial uses, though some require registration or have attribution requirements.
