# Free Mapping Tile Providers - Research Summary

> **Purpose**: Replace paid Mapbox with free alternatives for the snowforecast dashboard
> **Date**: January 2026
> **Status**: Research Complete

---

## Quick Recommendation

For the snowforecast project, I recommend this combination:

| Layer Type | Provider | Why |
|------------|----------|-----|
| **Basemap** | OpenStreetMap (via xyzservices) | Completely free, no API key |
| **Terrain** | AWS Terrain Tiles | Free S3 public dataset, no key |
| **Satellite** | Sentinel-2 Cloudless (EOX) | Free for non-commercial, high quality |
| **Fallback** | Stadia Maps (Stamen styles) | 2,500 free credits/month |

---

## 1. Free Basemap Tiles

### Completely Free (No API Key Required)

#### OpenStreetMap (OSM)
- **URL**: `https://tile.openstreetmap.org/{z}/{x}/{y}.png`
- **Cost**: Free
- **API Key**: Not required
- **Limits**: Fair use only - do NOT use for heavy commercial apps
- **Quality**: Good general purpose
- **Attribution**: Required - "OpenStreetMap contributors"

> **Warning**: OpenStreetMap.org tiles are volunteer-run. For production, use a commercial provider that serves OSM tiles.

**Usage with Folium:**
```python
import folium
m = folium.Map(location=[40.0, -111.0], zoom_start=8, tiles='OpenStreetMap')
```

**Usage with PyDeck:**
```python
import pydeck as pdk
# PyDeck requires a basemap provider or custom tile layer
# See "PyDeck with Custom Tiles" section below
```

---

#### OpenTopoMap
- **URL**: `https://tile.opentopomap.org/{z}/{x}/{y}.png`
- **Cost**: Free
- **API Key**: Not required
- **Limits**: Zoom levels 0-19
- **Quality**: Excellent for terrain visualization (topographic contours)
- **Status**: Not actively updated as of 2024, but tiles still available
- **License**: CC-BY-SA

> **Best for**: Mountain/ski applications - shows elevation contours naturally

**Usage with Folium:**
```python
import folium
m = folium.Map(
    location=[40.0, -111.0],
    zoom_start=10,
    tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='OpenTopoMap (CC-BY-SA)'
)
```

---

### Free Tier (API Key Required)

#### Stadia Maps (Stamen Tiles)
- **URL**: `https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png?api_key=YOUR_KEY`
- **Cost**: Free tier available
- **API Key**: Required (sign up at stadiamaps.com)
- **Limits**: 2,500 credits/month free (1 raster tile = 1 credit)
- **Quality**: Excellent - includes famous Stamen styles (Terrain, Toner, Watercolor)
- **Commercial**: Free for non-commercial; $20/month for commercial

**Available Stamen Styles:**
- `stamen_terrain` - Terrain with hillshading (best for mountains)
- `stamen_toner` - High contrast B&W
- `stamen_watercolor` - Artistic watercolor effect
- `stamen_terrain_background` - Terrain without labels

**Usage with Folium:**
```python
import folium

# Note: Stadia requires API key since 2023
STADIA_KEY = "your_api_key"
stadia_url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain/{{z}}/{{x}}/{{y}}.png?api_key={STADIA_KEY}"

m = folium.Map(
    location=[40.0, -111.0],
    zoom_start=10,
    tiles=stadia_url,
    attr='Stadia Maps / Stamen Design'
)
```

---

#### CARTO (CartoDB)
- **URL**: `https://{s}.basemaps.cartocdn.com/{style}/{z}/{x}/{y}.png`
- **Cost**: Free for grantees; Enterprise license for commercial
- **API Key**: Required for API access
- **Styles**: positron (light), dark_matter (dark), voyager (colorful)
- **Quality**: Clean, modern design

**Usage with Folium:**
```python
import folium

# Built-in support
m = folium.Map(location=[40.0, -111.0], zoom_start=10, tiles='CartoDB positron')
# or
m = folium.Map(location=[40.0, -111.0], zoom_start=10, tiles='CartoDB dark_matter')
```

---

#### Thunderforest
- **URL**: `https://tile.thunderforest.com/{style}/{z}/{x}/{y}.png?apikey=YOUR_KEY`
- **Cost**: Free "Hobby Project" tier
- **API Key**: Required (sign up at thunderforest.com)
- **Limits**: Limited requests/month (check current limits)
- **Styles**: outdoors, landscape, cycle, transport, atlas

> **Note**: Static map requests count as 10 tile requests

---

#### Geoapify
- **URL**: `https://maps.geoapify.com/v1/tile/{style}/{z}/{x}/{y}.png?apiKey=YOUR_KEY`
- **Cost**: Free tier
- **API Key**: Required
- **Limits**: 3,000 credits/day (~12,000 tiles/day at 0.25 credits/tile)
- **Commercial**: Limited commercial use allowed on free tier
- **Quality**: Good, OSM-based

---

#### MapTiler
- **URL**: `https://api.maptiler.com/maps/{style}/{z}/{x}/{y}.png?key=YOUR_KEY`
- **Cost**: Free tier
- **API Key**: Required
- **Limits**: Up to 100,000 tile loads/month
- **Quality**: Excellent vector and raster tiles
- **Note**: Service pauses when limit hit (no overage charges)

---

## 2. Free Terrain/Elevation Tiles

### AWS Terrain Tiles (Recommended)
- **S3 Bucket**: `s3://elevation-tiles-prod/`
- **Cost**: Completely free (AWS Public Dataset)
- **API Key**: Not required for direct S3 access
- **Resolution**: 30m globally, 10m in US, 3m in select areas
- **Formats**: Terrarium PNG, Normal PNG, GeoTIFF, Skadi HGT

**Available Formats:**
| Format | URL Pattern | Use Case |
|--------|-------------|----------|
| Terrarium | `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png` | RGB elevation encoding |
| Normal | `https://s3.amazonaws.com/elevation-tiles-prod/normal/{z}/{x}/{y}.png` | Surface normals for lighting |
| GeoTIFF | `https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif` | Raw elevation analysis |
| Skadi | `https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat}/{lon}.hgt` | SRTM HGT format |

**Data Sources:**
- 3DEP (US): 10m outside Alaska, 3m in select areas
- SRTM: 30m globally (land areas)
- GMTED: Coarser resolutions for global coverage

**Attribution Required**: Yes

**Usage Example:**
```python
import requests
import numpy as np
from PIL import Image
import io

def get_elevation_terrarium(z, x, y):
    """Get elevation from AWS Terrain Tiles (Terrarium format)"""
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)

    # Decode Terrarium format: elevation = (R * 256 + G + B / 256) - 32768
    elevation = (arr[:,:,0] * 256 + arr[:,:,1] + arr[:,:,2] / 256) - 32768
    return elevation
```

---

### MapTiler Terrain RGB
- **URL**: `https://api.maptiler.com/tiles/terrain-rgb/{z}/{x}/{y}.webp?key=YOUR_KEY`
- **Cost**: Free tier (part of 100k tiles/month limit)
- **API Key**: Required
- **Resolution**: 30m globally
- **Features**: Also offers hillshade, contour lines, 3D Cesium tiles

---

### OpenTopoMap (Combined)
- **URL**: `https://tile.opentopomap.org/{z}/{x}/{y}.png`
- **Note**: Not raw elevation data, but pre-rendered topographic visualization
- **Best for**: Direct display without processing

---

## 3. Free Satellite Imagery

### Sentinel-2 Cloudless by EOX (Recommended)
- **WMTS**: `https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml`
- **WMS**: `https://tiles.maps.eox.at/wms?service=wms&request=getcapabilities`
- **Cost**: Free for non-commercial use
- **API Key**: Not required
- **Resolution**: 10m
- **Coverage**: Global, cloudless composite (2016-2024 versions)
- **Updates**: Annual updates

**Licensing:**
- 2016 version: CC-BY 4.0 (commercial allowed)
- 2018-2024: CC-BY-NC-SA 4.0 (non-commercial only)

**Attribution Required:**
```
Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH
(Contains modified Copernicus Sentinel data 2024)
```

**Usage with Folium:**
```python
import folium

# Using WMS
m = folium.Map(location=[40.0, -111.0], zoom_start=10)
folium.raster_layers.WmsTileLayer(
    url='https://tiles.maps.eox.at/wms?',
    layers='s2cloudless-2024',
    name='Sentinel-2 Cloudless',
    fmt='image/png',
    attr='Sentinel-2 cloudless by EOX'
).add_to(m)
```

---

### SentinelMap Beta
- **URL**: `https://api.sentinelmap.eu/...` (requires signup)
- **Cost**: Free beta
- **API Key**: Required (sign up with GitHub)
- **Limits**: 50,000 tiles/month during beta
- **Quality**: Cloud-free Sentinel-2 tiles

---

### USGS National Map Imagery
- **URL**: `https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer`
- **Cost**: Free (public domain)
- **API Key**: Not required
- **Coverage**: US only
- **Resolution**: Varies - Landsat for medium scales, NAIP for larger scales
- **Formats**: REST, WMS, WMTS

**Best for**: US-focused applications needing high-res NAIP imagery

**Usage with Folium:**
```python
import folium

m = folium.Map(location=[40.0, -111.0], zoom_start=10)
folium.TileLayer(
    tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
    attr='USGS National Map',
    name='USGS Imagery'
).add_to(m)
```

---

### ESRI World Imagery
- **URL**: `https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer`
- **Cost**: Requires ArcGIS license for most uses
- **Exception**: Free for OSM mapping
- **Limits**: 100,000-150,000 tile export limit
- **Quality**: High resolution

> **Warning**: Not truly free for general use - requires ArcGIS subscription

---

## 4. Python Integration

### Using xyzservices Package

The `xyzservices` package provides 200+ tile providers in a unified interface:

```python
import xyzservices.providers as xyz

# List all providers
print(xyz.keys())

# Access specific providers
osm = xyz.OpenStreetMap.Mapnik
carto_light = xyz.CartoDB.Positron
carto_dark = xyz.CartoDB.DarkMatter
stadia_terrain = xyz.Stadia.StamenTerrain  # Requires API key

# Get URL and attribution
print(xyz.OpenStreetMap.Mapnik.url)
print(xyz.OpenStreetMap.Mapnik.attribution)
```

### Folium Integration

```python
import folium
import xyzservices.providers as xyz

# Method 1: Built-in tiles
m = folium.Map(location=[40.0, -111.0], zoom_start=10, tiles='OpenStreetMap')

# Method 2: xyzservices TileProvider
m = folium.Map(
    location=[40.0, -111.0],
    zoom_start=10,
    tiles=xyz.CartoDB.Positron
)

# Method 3: Custom URL
m = folium.Map(
    location=[40.0, -111.0],
    zoom_start=10,
    tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='OpenTopoMap'
)

# Add layer control for multiple basemaps
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('CartoDB positron').add_to(m)
folium.TileLayer(
    tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
    attr='OpenTopoMap',
    name='OpenTopoMap'
).add_to(m)
folium.LayerControl().add_to(m)
```

### PyDeck with Custom Tiles

PyDeck doesn't have native support for custom XYZ tiles, but there are workarounds:

```python
import pydeck as pdk

# Option 1: Use built-in providers (requires API key)
# Set environment variable: MAPBOX_API_KEY, CARTO_API_KEY, or GOOGLE_MAPS_API_KEY

# Option 2: Use CARTO basemap (free for non-commercial)
deck = pdk.Deck(
    map_provider='carto',
    map_style='light',  # or 'dark', 'road', 'satellite'
    initial_view_state=pdk.ViewState(
        latitude=40.0,
        longitude=-111.0,
        zoom=8
    ),
    layers=[...]
)

# Option 3: No basemap (render your own data only)
deck = pdk.Deck(
    map_provider=None,
    initial_view_state=pdk.ViewState(...),
    layers=[...]
)

# Option 4: Custom TileLayer (requires custom JS library)
# See: https://github.com/agressin/pydeck_myTileLayer
pdk.settings.custom_libraries = [
    {
        "libraryName": "MyTileLayerLibrary",
        "resourceUri": "https://cdn.jsdelivr.net/gh/agressin/pydeck_myTileLayer@master/dist/bundle.js",
    }
]

osm_layer = pdk.Layer(
    "MyTileLayer",
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
)
```

### Using leafmap (Recommended for PyDeck)

```python
import leafmap.deckgl as leafmap

# leafmap provides easier basemap integration
m = leafmap.Map(center=[40.0, -111.0], zoom=8)
m.add_basemap('OpenTopoMap')
m.add_basemap('CartoDB.Positron')
```

---

## 5. Comparison Table

### Basemaps

| Provider | Cost | API Key | Limits | Best For |
|----------|------|---------|--------|----------|
| OpenStreetMap | Free | No | Fair use | General purpose |
| OpenTopoMap | Free | No | None | Mountains/terrain |
| Stadia (Stamen) | Free tier | Yes | 2,500/mo | Styled maps |
| CARTO | Free (grants) | Yes | Varies | Clean design |
| Geoapify | Free tier | Yes | 12k tiles/day | Prototypes |
| MapTiler | Free tier | Yes | 100k/mo | Production apps |

### Terrain/Elevation

| Provider | Cost | API Key | Resolution | Format |
|----------|------|---------|------------|--------|
| AWS Terrain Tiles | Free | No | 10-30m | PNG/GeoTIFF |
| MapTiler Terrain | Free tier | Yes | 30m | RGB/WMTS |
| OpenTopoMap | Free | No | N/A | Pre-rendered |

### Satellite

| Provider | Cost | API Key | Resolution | Coverage |
|----------|------|---------|------------|----------|
| Sentinel-2 (EOX) | Free (NC) | No | 10m | Global |
| SentinelMap Beta | Free | Yes | 10m | Global |
| USGS National Map | Free | No | Varies | US only |
| ESRI World Imagery | License | Yes | High | Global |

---

## 6. Implementation Recommendations for Snowforecast

### Recommended Stack

```python
# requirements.txt additions
folium>=0.14.0
xyzservices>=2024.1.0
leafmap>=0.30.0  # For PyDeck integration

# No API keys needed for basic setup!
```

### Dashboard Implementation

```python
import folium
from folium.plugins import Fullscreen

def create_ski_resort_map(resorts_df):
    """Create interactive map with free tile layers."""

    # Center on Western US mountains
    m = folium.Map(
        location=[40.5, -111.5],
        zoom_start=7,
        tiles=None  # Start with no tiles, add custom
    )

    # Add multiple free basemap options
    folium.TileLayer(
        tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap (CC-BY-SA)',
        name='Topographic',
        overlay=False
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False
    ).add_to(m)

    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light',
        overlay=False
    ).add_to(m)

    # Add satellite layer (Sentinel-2)
    folium.raster_layers.WmsTileLayer(
        url='https://tiles.maps.eox.at/wms?',
        layers='s2cloudless-2024',
        name='Satellite',
        fmt='image/png',
        attr='Sentinel-2 cloudless by EOX',
        overlay=False
    ).add_to(m)

    # Add USGS imagery for US
    folium.TileLayer(
        tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
        attr='USGS',
        name='USGS Imagery',
        overlay=False
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    Fullscreen().add_to(m)

    return m
```

---

## Sources

### Basemap Providers
- [OpenMapTiles](https://openmaptiles.org/)
- [Stadia Maps Pricing](https://stadiamaps.com/pricing)
- [Stadia Maps Stamen Partnership](https://stadiamaps.com/stamen/)
- [CARTO Basemaps](https://carto.com/basemaps)
- [Thunderforest API Keys](https://www.thunderforest.com/docs/apikeys/)
- [Geoapify Pricing](https://www.geoapify.com/pricing/)
- [MapTiler Pricing](https://www.maptiler.com/cloud/pricing/)
- [7 Free Map APIs vs Google Maps](https://felt.com/blog/7-free-map-apis-compared-to-google-maps)

### Terrain Data
- [AWS Terrain Tiles - Registry of Open Data](https://registry.opendata.aws/terrain-tiles/)
- [MapTiler Terrain](https://www.maptiler.com/terrain/)
- [OpenTopoMap Wiki](https://wiki.openstreetmap.org/wiki/OpenTopoMap)

### Satellite Imagery
- [Sentinel-2 Cloudless by EOX](https://s2maps.eu/)
- [SentinelMap](https://www.sentinelmap.eu/)
- [USGS National Map Services](https://www.usgs.gov/faqs/what-are-base-map-services-or-urls-used-national-map)
- [ESRI World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)

### Python Libraries
- [Folium Tiles Documentation](https://python-visualization.github.io/folium/latest/user_guide/raster_layers/tiles.html)
- [xyzservices GitHub](https://github.com/geopandas/xyzservices)
- [PyDeck Custom Layers](https://deckgl.readthedocs.io/en/latest/custom_layers.html)
- [pydeck_myTileLayer GitHub](https://github.com/agressin/pydeck_myTileLayer)
- [leafmap Documentation](https://leafmap.org/)
