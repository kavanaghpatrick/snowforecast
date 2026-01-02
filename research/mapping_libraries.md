# JavaScript Mapping Libraries for Snow/Weather Visualization with Streamlit

> Research conducted: January 2026
> Focus: Terrain visualization, elevation shading, snow depth color scales, Streamlit integration

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Library Comparison Matrix](#library-comparison-matrix)
3. [PyDeck (Recommended for Streamlit)](#pydeck-recommended-for-streamlit)
4. [Folium](#folium)
5. [Mapbox GL JS](#mapbox-gl-js)
6. [Deck.gl](#deckgl)
7. [Leaflet](#leaflet)
8. [Color Scales for Snow Visualization](#color-scales-for-snow-visualization)
9. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

For the Snowforecast project, **PyDeck** is the recommended primary library due to:
- Native Streamlit integration via `st.pydeck_chart`
- 3D terrain visualization with TerrainLayer
- Excellent color scale support for data-heavy visualizations
- No additional components required

**Folium** is recommended as a secondary option for:
- Simpler 2D maps with markers
- HeatMap overlays
- Easier setup with `streamlit-folium` component

---

## Library Comparison Matrix

| Feature | PyDeck | Folium | Mapbox GL JS | Deck.gl | Leaflet |
|---------|--------|--------|--------------|---------|---------|
| **Streamlit Integration** | Native | Component | Component | Via PyDeck | Via Folium |
| **3D Terrain** | Excellent | None | Excellent | Excellent | Limited |
| **Color Gradients** | Excellent | Good | Excellent | Excellent | Good (plugins) |
| **Tooltips** | Built-in | Built-in | Built-in | Built-in | Built-in |
| **Performance (large data)** | Excellent | Moderate | Excellent | Excellent | Moderate |
| **Learning Curve** | Moderate | Easy | Steep | Steep | Moderate |
| **API Key Required** | Optional* | No | Yes | No | No |

*PyDeck uses Carto tiles by default; Mapbox requires API key for terrain tiles

---

## PyDeck (Recommended for Streamlit)

### Overview
PyDeck is the Python binding for deck.gl, providing powerful WebGL-based visualization directly in Streamlit.

**Installation:**
```bash
pip install pydeck streamlit
```

### Key Layers for Snow Visualization

#### 1. TerrainLayer - 3D Elevation Visualization

```python
import pydeck as pdk
import streamlit as st
import os

# AWS Open Data Terrain Tiles (free, no API key)
TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Elevation decoder for Terrarium format
ELEVATION_DECODER = {
    "rScaler": 256,
    "gScaler": 1,
    "bScaler": 1 / 256,
    "offset": -32768
}

# Optional: Mapbox satellite imagery (requires API key)
MAPBOX_API_KEY = os.environ.get("MAPBOX_API_KEY", "")
SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX_API_KEY}"

terrain_layer = pdk.Layer(
    "TerrainLayer",
    elevation_decoder=ELEVATION_DECODER,
    texture=SURFACE_IMAGE if MAPBOX_API_KEY else None,
    elevation_data=TERRAIN_IMAGE,
)

# Colorado Rockies view
view_state = pdk.ViewState(
    latitude=39.5,
    longitude=-106.0,
    zoom=10,
    bearing=45,
    pitch=60
)

deck = pdk.Deck(
    layers=[terrain_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/satellite-v9" if MAPBOX_API_KEY else None
)

st.pydeck_chart(deck)
```

#### 2. ScatterplotLayer - Snow Depth at Station Points

```python
import pydeck as pdk
import pandas as pd
import streamlit as st

# Sample snow station data
data = pd.DataFrame({
    'name': ['Station A', 'Station B', 'Station C', 'Station D'],
    'lat': [39.5, 39.7, 39.3, 39.6],
    'lon': [-106.0, -106.2, -105.8, -106.1],
    'snow_depth_cm': [45, 120, 80, 200],
    'elevation_m': [2800, 3200, 2600, 3500]
})

# Color function based on snow depth
def get_snow_color(depth):
    """Return RGBA color based on snow depth (cm)"""
    if depth < 30:
        return [173, 216, 230, 200]  # Light blue
    elif depth < 60:
        return [100, 149, 237, 200]  # Cornflower blue
    elif depth < 100:
        return [65, 105, 225, 200]   # Royal blue
    elif depth < 150:
        return [0, 0, 205, 200]      # Medium blue
    else:
        return [138, 43, 226, 200]   # Purple (deep snow)

data['color'] = data['snow_depth_cm'].apply(get_snow_color)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position=['lon', 'lat'],
    get_fill_color='color',
    get_radius='snow_depth_cm * 50',  # Radius proportional to depth
    radius_min_pixels=10,
    radius_max_pixels=100,
    pickable=True,
    opacity=0.8,
)

view_state = pdk.ViewState(
    latitude=39.5,
    longitude=-106.0,
    zoom=9,
    pitch=45,
)

# Tooltip configuration
tooltip = {
    "html": """
        <b>{name}</b><br/>
        Snow Depth: {snow_depth_cm} cm<br/>
        Elevation: {elevation_m} m
    """,
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
)

st.pydeck_chart(deck)
```

#### 3. HexagonLayer - Aggregated Snow Data

```python
import pydeck as pdk
import pandas as pd
import streamlit as st

# Sample observation data
observations = pd.DataFrame({
    'lat': [39.5, 39.51, 39.52, 39.7, 39.71, 39.3],
    'lon': [-106.0, -106.01, -106.02, -106.2, -106.21, -105.8],
    'snow_amount': [45, 50, 48, 120, 115, 80]
})

layer = pdk.Layer(
    "HexagonLayer",
    data=observations,
    get_position=['lon', 'lat'],
    get_elevation_weight='snow_amount',
    elevation_scale=100,
    elevation_range=[0, 3000],
    extruded=True,
    radius=1000,
    coverage=0.8,
    # Custom color range: light blue -> dark blue -> purple
    color_range=[
        [173, 216, 230],  # Light blue
        [100, 149, 237],  # Cornflower
        [65, 105, 225],   # Royal blue
        [0, 0, 205],      # Medium blue
        [75, 0, 130],     # Indigo
        [138, 43, 226],   # Purple
    ],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=39.5,
    longitude=-106.0,
    zoom=9,
    pitch=50,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
```

#### 4. ColumnLayer - 3D Snow Depth Bars

```python
import pydeck as pdk
import pandas as pd
import streamlit as st

data = pd.DataFrame({
    'name': ['Alta', 'Snowbird', 'Park City', 'Vail'],
    'lat': [40.59, 40.58, 40.65, 39.64],
    'lon': [-111.64, -111.66, -111.51, -106.37],
    'snow_depth_cm': [150, 140, 90, 120],
})

# Normalize snow depth to color (0-255 scale)
max_depth = data['snow_depth_cm'].max()
data['color_intensity'] = (data['snow_depth_cm'] / max_depth * 255).astype(int)

layer = pdk.Layer(
    "ColumnLayer",
    data=data,
    get_position=['lon', 'lat'],
    get_elevation='snow_depth_cm * 50',  # Exaggerate for visibility
    elevation_scale=1,
    radius=2000,
    get_fill_color=['255 - color_intensity', '255 - color_intensity', 255, 200],
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=40.0,
    longitude=-110.0,
    zoom=7,
    pitch=60,
    bearing=30
)

tooltip = {"html": "<b>{name}</b><br/>Snow: {snow_depth_cm} cm"}

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
))
```

### Sources
- [Streamlit PyDeck Documentation](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
- [PyDeck Layer Documentation](https://deckgl.readthedocs.io/en/latest/layer.html)
- [PyDeck TerrainLayer](https://deckgl.readthedocs.io/en/latest/gallery/terrain_layer.html)
- [PyDeck ScatterplotLayer](https://deckgl.readthedocs.io/en/latest/gallery/scatterplot_layer.html)

---

## Folium

### Overview
Folium is a Python wrapper for Leaflet.js, providing simple 2D map creation with excellent marker and popup support.

**Installation:**
```bash
pip install folium streamlit-folium branca
```

### Basic Snow Station Map

```python
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import branca.colormap as cm

# Create base map centered on Colorado
m = folium.Map(
    location=[39.5, -106.0],
    zoom_start=9,
    tiles='OpenTopoMap'  # Shows terrain
)

# Snow station data
stations = [
    {'name': 'Alta', 'lat': 40.59, 'lon': -111.64, 'snow_cm': 150, 'elev': 3200},
    {'name': 'Snowbird', 'lat': 40.58, 'lon': -111.66, 'snow_cm': 140, 'elev': 3100},
    {'name': 'Park City', 'lat': 40.65, 'lon': -111.51, 'snow_cm': 90, 'elev': 2100},
]

# Create colormap for snow depth
colormap = cm.LinearColormap(
    colors=['#ADD8E6', '#6495ED', '#4169E1', '#0000CD', '#8A2BE2'],
    vmin=0,
    vmax=200,
    caption='Snow Depth (cm)'
)

# Add markers with color based on snow depth
for station in stations:
    color = colormap(station['snow_cm'])

    folium.CircleMarker(
        location=[station['lat'], station['lon']],
        radius=station['snow_cm'] / 10,  # Size proportional to snow
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"""
            <b>{station['name']}</b><br/>
            Snow Depth: {station['snow_cm']} cm<br/>
            Elevation: {station['elev']} m
            """,
            max_width=200
        ),
        tooltip=f"{station['name']}: {station['snow_cm']} cm"
    ).add_to(m)

# Add colormap legend to map
m.add_child(colormap)

# Display in Streamlit
st_data = st_folium(m, width=700, height=500)
```

### HeatMap for Snow Density

```python
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

m = folium.Map(location=[39.5, -106.0], zoom_start=8)

# Sample observation points with snow amounts
heat_data = [
    [39.5, -106.0, 0.8],   # [lat, lon, intensity 0-1]
    [39.51, -106.01, 0.85],
    [39.7, -106.2, 1.0],   # High snow
    [39.3, -105.8, 0.6],
]

# Custom gradient: light blue -> dark blue -> purple
gradient = {
    0.0: '#E6F3FF',   # Very light blue
    0.3: '#ADD8E6',   # Light blue
    0.5: '#6495ED',   # Cornflower blue
    0.7: '#4169E1',   # Royal blue
    0.85: '#0000CD',  # Medium blue
    1.0: '#8A2BE2',   # Purple (deep snow)
}

HeatMap(
    heat_data,
    radius=25,
    blur=15,
    gradient=gradient,
    max_val=1.0
).add_to(m)

st_folium(m, width=700, height=500)
```

### StepColormap for Discrete Snow Categories

```python
import folium
import branca.colormap as cm
import streamlit as st
from streamlit_folium import st_folium

# Create discrete color categories
colormap = cm.StepColormap(
    colors=['#E6F3FF', '#ADD8E6', '#6495ED', '#4169E1', '#0000CD', '#8A2BE2'],
    vmin=0,
    vmax=200,
    index=[0, 30, 60, 100, 150, 200],
    caption='Snow Depth Categories (cm)'
)

m = folium.Map(location=[39.5, -106.0], zoom_start=8)

# Add stations with categorical coloring
stations = [
    {'name': 'Light Snow Station', 'lat': 39.5, 'lon': -106.0, 'snow_cm': 25},
    {'name': 'Moderate Station', 'lat': 39.6, 'lon': -106.1, 'snow_cm': 75},
    {'name': 'Heavy Snow Station', 'lat': 39.7, 'lon': -106.2, 'snow_cm': 160},
]

for station in stations:
    folium.CircleMarker(
        location=[station['lat'], station['lon']],
        radius=12,
        color=colormap(station['snow_cm']),
        fill=True,
        fill_color=colormap(station['snow_cm']),
        fill_opacity=0.8,
        tooltip=f"{station['name']}: {station['snow_cm']} cm"
    ).add_to(m)

m.add_child(colormap)
st_folium(m, width=700, height=500)
```

### Sources
- [streamlit-folium Documentation](https://folium.streamlit.app/)
- [GitHub: streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
- [Folium Colormaps Guide](https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html)
- [Folium CircleMarker](https://python-visualization.github.io/folium/latest/user_guide/vector_layers/circle_and_circle_marker.html)

---

## Mapbox GL JS

### Overview
Mapbox GL JS is a powerful JavaScript library with excellent 3D terrain, dynamic weather effects, and professional-grade styling.

**Key Features:**
- Native 3D terrain with `setTerrain()`
- Snow and rain particle effects (v3.9+)
- Fog/atmosphere effects
- Client-side elevation queries
- Professional styling options

**API Key Required:** Yes (free tier available at mapbox.com)

### Streamlit Integration Options

#### 1. Via PyDeck (Recommended)

```python
import pydeck as pdk
import os

MAPBOX_API_KEY = os.environ.get("MAPBOX_API_KEY")

deck = pdk.Deck(
    api_keys={"mapbox": MAPBOX_API_KEY},
    map_style="mapbox://styles/mapbox/outdoors-v12",
    initial_view_state=pdk.ViewState(
        latitude=39.5,
        longitude=-106.0,
        zoom=10,
        pitch=60,
    ),
    layers=[...]
)

st.pydeck_chart(deck)
```

#### 2. Via streamlit-mapbox Component

```bash
pip install streamlit-mapbox
```

```python
# Basic usage (limited functionality)
from streamlit_mapbox import streamlit_mapbox

# Note: This component has limited features compared to PyDeck
```

#### 3. Custom HTML Component (Full Features)

```python
import streamlit as st
import streamlit.components.v1 as components

MAPBOX_TOKEN = "your_token_here"

mapbox_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src='https://api.mapbox.com/mapbox-gl-js/v3.9.0/mapbox-gl.js'></script>
    <link href='https://api.mapbox.com/mapbox-gl-js/v3.9.0/mapbox-gl.css' rel='stylesheet' />
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 500px; }}
    </style>
</head>
<body>
    <div id='map'></div>
    <script>
        mapboxgl.accessToken = '{MAPBOX_TOKEN}';
        const map = new mapboxgl.Map({{
            container: 'map',
            style: 'mapbox://styles/mapbox/outdoors-v12',
            center: [-106.0, 39.5],
            zoom: 10,
            pitch: 60
        }});

        map.on('load', () => {{
            // Add 3D terrain
            map.addSource('mapbox-dem', {{
                'type': 'raster-dem',
                'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
                'tileSize': 512,
                'maxzoom': 14
            }});
            map.setTerrain({{ 'source': 'mapbox-dem', 'exaggeration': 1.5 }});

            // Add snow effect (v3.9+)
            map.setSnow({{
                density: 0.5,
                intensity: 0.5
            }});

            // Add fog for depth
            map.setFog({{
                range: [0.5, 10],
                color: 'white',
                'horizon-blend': 0.1
            }});
        }});
    </script>
</body>
</html>
"""

components.html(mapbox_html, height=520)
```

### Mapbox Features for Snow Visualization

1. **3D Terrain**: `map.setTerrain()` with exaggeration control
2. **Snow Effects**: `map.setSnow()` for animated snow particles
3. **Elevation Queries**: `map.queryTerrainElevation(lnglat)` for point elevation
4. **Fog/Atmosphere**: `map.setFog()` for depth perception

### Sources
- [Mapbox GL JS Documentation](https://docs.mapbox.com/mapbox-gl-js/guides/)
- [Add 3D Terrain Example](https://docs.mapbox.com/mapbox-gl-js/example/add-terrain/)
- [Snow Effect Example](https://docs.mapbox.com/mapbox-gl-js/example/snow/)
- [Rain and Snow Playground](https://docs.mapbox.com/playground/rain-and-snow/)
- [Query Terrain Elevation](https://docs.mapbox.com/mapbox-gl-js/example/query-terrain-elevation/)

---

## Deck.gl

### Overview
Deck.gl is the underlying library for PyDeck, providing WebGL-powered visualization. Direct usage offers more control but requires JavaScript.

**Key Features:**
- High-performance rendering of millions of data points
- Extensive layer catalog (50+ layer types)
- GPU-accelerated data filtering
- Custom shader support

### Relevant Layers for Snow Visualization

| Layer | Use Case | Key Properties |
|-------|----------|----------------|
| `TerrainLayer` | 3D elevation models | `elevationData`, `texture`, `elevationDecoder` |
| `ScatterplotLayer` | Station points | `getPosition`, `getFillColor`, `getRadius` |
| `HexagonLayer` | Aggregated data | `colorRange`, `elevationRange`, `radius` |
| `GridLayer` | Grid aggregation | `colorDomain`, `colorRange`, `colorScaleType` |
| `ColumnLayer` | 3D bars | `getElevation`, `getFillColor` |
| `GeoJsonLayer` | Boundaries/regions | `getFillColor`, `getLineColor`, `extruded` |

### Color Scale Configuration

```javascript
// deck.gl color scale types
const colorScaleTypes = {
    'linear': 'Continuous interpolation across colorRange',
    'quantize': 'Equal segments mapped to discrete colors',
    'quantile': 'Equal-size groups mapped to discrete colors'
};

// Example HexagonLayer with color scale
new HexagonLayer({
    data: snowObservations,
    getPosition: d => [d.lon, d.lat],
    getElevationWeight: d => d.snow_amount,
    colorDomain: [0, 200],  // Snow depth range in cm
    colorRange: [
        [173, 216, 230],  // Light blue (0-40cm)
        [100, 149, 237],  // Cornflower (40-80cm)
        [65, 105, 225],   // Royal blue (80-120cm)
        [0, 0, 205],      // Medium blue (120-160cm)
        [138, 43, 226],   // Purple (160-200cm)
    ],
    colorScaleType: 'quantize',
    elevationScale: 100,
    extruded: true
});
```

### Sources
- [deck.gl Documentation](https://deck.gl/docs)
- [HexagonLayer](https://deck.gl/docs/api-reference/aggregation-layers/hexagon-layer)
- [GridLayer](https://deck.gl/docs/api-reference/aggregation-layers/grid-layer)
- [TerrainLayer](https://deck.gl/docs/api-reference/geo-layers/terrain-layer)

---

## Leaflet

### Overview
Leaflet is a lightweight JavaScript mapping library with extensive plugin ecosystem. Best accessed via Folium in Python.

### Weather Plugins

#### leaflet-openweathermap
Provides access to OpenWeatherMap tile layers including snow coverage.

```javascript
// Available layers: Clouds, Precipitation, Rain, Snow, Temperature, Wind
L.OWM.snow({
    showLegend: true,
    opacity: 0.5,
    appId: 'your_openweathermap_api_key'
}).addTo(map);
```

**Python (via Folium):**
```python
import folium

m = folium.Map(location=[39.5, -106.0], zoom_start=8)

# Add OpenWeatherMap snow layer
folium.TileLayer(
    tiles='https://tile.openweathermap.org/map/snow/{z}/{x}/{y}.png?appid=YOUR_API_KEY',
    attr='OpenWeatherMap',
    name='Snow Coverage',
    overlay=True
).add_to(m)

folium.LayerControl().add_to(m)
```

#### WeatherLayers Service
Commercial service supporting multiple mapping libraries including Leaflet.

**Supported Data:**
- Wind, Temperature, Humidity, Pressure
- Precipitation, Snow, Clouds
- Radar, Solar, CAPE

### Sources
- [leaflet-openweathermap GitHub](https://github.com/buche/leaflet-openweathermap)
- [Leaflet Plugins Directory](https://leafletjs.com/plugins.html)
- [WeatherLayers](https://weatherlayers.com/)
- [OpenWeatherMap Weather Maps API](https://openweathermap.org/api/weather-map-2)

---

## Color Scales for Snow Visualization

### Elevation Color Scale (Green -> Brown -> White)

Traditional hypsometric tints for terrain:

```python
# RGB values for elevation coloring
ELEVATION_COLORS = {
    'low_valley': (34, 139, 34),      # Forest green (< 1500m)
    'foothills': (143, 188, 143),     # Sage green (1500-2000m)
    'mid_mountain': (210, 180, 140),  # Tan (2000-2500m)
    'high_mountain': (139, 119, 101), # Brown (2500-3000m)
    'alpine': (169, 169, 169),        # Gray (3000-3500m)
    'snow_peak': (255, 255, 255),     # White (> 3500m)
}

# Linear interpolation function
def elevation_to_color(elevation_m):
    """Return RGB color for elevation in meters"""
    if elevation_m < 1500:
        return [34, 139, 34]
    elif elevation_m < 2000:
        t = (elevation_m - 1500) / 500
        return [
            int(34 + t * (143 - 34)),
            int(139 + t * (188 - 139)),
            int(34 + t * (143 - 34))
        ]
    elif elevation_m < 2500:
        t = (elevation_m - 2000) / 500
        return [
            int(143 + t * (210 - 143)),
            int(188 + t * (180 - 188)),
            int(143 + t * (140 - 143))
        ]
    elif elevation_m < 3000:
        t = (elevation_m - 2500) / 500
        return [
            int(210 + t * (139 - 210)),
            int(180 + t * (119 - 180)),
            int(140 + t * (101 - 140))
        ]
    elif elevation_m < 3500:
        t = (elevation_m - 3000) / 500
        return [
            int(139 + t * (169 - 139)),
            int(119 + t * (169 - 119)),
            int(101 + t * (169 - 101))
        ]
    else:
        return [255, 255, 255]
```

### Snow Depth Color Scale (Light Blue -> Purple)

```python
# Snow depth color scale
SNOW_COLORS = {
    'trace': '#E6F3FF',        # Very light blue (< 10cm)
    'light': '#ADD8E6',        # Light blue (10-30cm)
    'moderate': '#6495ED',     # Cornflower blue (30-60cm)
    'heavy': '#4169E1',        # Royal blue (60-100cm)
    'very_heavy': '#0000CD',   # Medium blue (100-150cm)
    'extreme': '#8A2BE2',      # Blue-violet (> 150cm)
}

def snow_depth_to_color(depth_cm):
    """Return hex color for snow depth in cm"""
    if depth_cm < 10:
        return '#E6F3FF'
    elif depth_cm < 30:
        return '#ADD8E6'
    elif depth_cm < 60:
        return '#6495ED'
    elif depth_cm < 100:
        return '#4169E1'
    elif depth_cm < 150:
        return '#0000CD'
    else:
        return '#8A2BE2'

# As RGB list for PyDeck
def snow_depth_to_rgb(depth_cm, alpha=200):
    """Return RGBA list for snow depth"""
    colors = {
        10: [230, 243, 255, alpha],    # Trace
        30: [173, 216, 230, alpha],    # Light
        60: [100, 149, 237, alpha],    # Moderate
        100: [65, 105, 225, alpha],    # Heavy
        150: [0, 0, 205, alpha],       # Very heavy
        999: [138, 43, 226, alpha],    # Extreme
    }
    for threshold, color in colors.items():
        if depth_cm < threshold:
            return color
    return colors[999]
```

### Combined Visualization (PyDeck)

```python
import pydeck as pdk
import pandas as pd
import streamlit as st

# Station data with both elevation and snow
data = pd.DataFrame({
    'name': ['Alta Basin', 'Snowbird Peak', 'Park City Base', 'Vail Summit'],
    'lat': [40.59, 40.58, 40.65, 39.64],
    'lon': [-111.64, -111.66, -111.51, -106.37],
    'snow_depth_cm': [150, 140, 50, 120],
    'elevation_m': [3200, 3500, 2100, 3400]
})

# Apply color functions
data['snow_color'] = data['snow_depth_cm'].apply(
    lambda d: snow_depth_to_rgb(d)
)
data['elevation_color'] = data['elevation_m'].apply(
    lambda e: elevation_to_color(e) + [200]
)

# Snow depth layer
snow_layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position=['lon', 'lat'],
    get_fill_color='snow_color',
    get_radius='snow_depth_cm * 30',
    radius_min_pixels=15,
    pickable=True,
    opacity=0.8,
)

tooltip = {
    "html": """
        <b>{name}</b><br/>
        Elevation: {elevation_m}m<br/>
        Snow Depth: {snow_depth_cm}cm
    """,
    "style": {"backgroundColor": "#1a1a2e", "color": "white"}
}

view_state = pdk.ViewState(
    latitude=40.0, longitude=-108.0, zoom=6, pitch=45
)

st.pydeck_chart(pdk.Deck(
    layers=[snow_layer],
    initial_view_state=view_state,
    tooltip=tooltip
))
```

### Sources
- [Creative Color Schemes for Elevation Maps](https://www.maplibrary.org/1354/creative-color-schemes-for-elevation-maps/)
- [Colour-Ramp for Elevation Mapping](https://mapscaping.com/colour-ramp-for-elevation-mapping/)
- [NASA Visualizing Elevation](https://svs.gsfc.nasa.gov/11734/)
- [MATLAB demcmap](https://www.mathworks.com/help/map/ref/demcmap.html)

---

## Implementation Recommendations

### For Snowforecast Dashboard

#### Primary Recommendation: PyDeck

1. **Station Map with Snow Depth**
   - Use `ScatterplotLayer` for SNOTEL/GHCN stations
   - Color by snow depth (blue-purple scale)
   - Size by snow amount
   - Tooltip with station details

2. **3D Terrain Visualization**
   - Use `TerrainLayer` with AWS elevation tiles
   - Optional Mapbox satellite imagery
   - Pitch and bearing for 3D perspective

3. **Forecast Heatmap**
   - Use `HexagonLayer` for aggregated predictions
   - Elevation represents predicted snow amount
   - Color intensity for confidence

#### Secondary: Folium for Simple Maps

1. **Quick Overview Maps**
   - OpenTopoMap base layer
   - CircleMarker for stations
   - LinearColormap legend

2. **Weather Overlays**
   - OpenWeatherMap snow layer
   - Layer toggle controls

### Code Structure

```
src/snowforecast/visualization/
    __init__.py
    map_utils.py        # Color scales, data prep
    pydeck_maps.py      # PyDeck layer factories
    folium_maps.py      # Folium map builders

dashboard/
    app.py              # Main Streamlit app
    pages/
        station_map.py  # Interactive station map
        forecast_map.py # Prediction visualization
```

### Sample Integration

```python
# dashboard/pages/station_map.py
import streamlit as st
import pydeck as pdk
from snowforecast.visualization import pydeck_maps, map_utils

st.title("Snow Station Map")

# Load data
stations = load_station_data()

# Sidebar controls
color_by = st.sidebar.selectbox(
    "Color by",
    ["Snow Depth", "Elevation", "24h Change"]
)

show_3d = st.sidebar.checkbox("3D Terrain", value=True)

# Build layers
layers = []

if show_3d:
    layers.append(pydeck_maps.create_terrain_layer())

layers.append(pydeck_maps.create_station_layer(
    stations,
    color_column=color_by.lower().replace(" ", "_")
))

# Display
deck = pdk.Deck(
    layers=layers,
    initial_view_state=pydeck_maps.COLORADO_VIEW,
    tooltip=pydeck_maps.STATION_TOOLTIP
)

st.pydeck_chart(deck)
```

---

## Additional Resources

### Documentation
- [Streamlit st.pydeck_chart](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
- [PyDeck Gallery](https://deckgl.readthedocs.io/)
- [Folium Documentation](https://python-visualization.github.io/folium/latest/)
- [streamlit-folium](https://folium.streamlit.app/)

### Tutorials
- [3D Geospatial Dashboard with Streamlit and PyDeck](https://medium.com/@agiraldoal/how-to-create-a-3d-geospatial-dashboard-with-python-streamlit-and-pydeck-c1f2cc3c2cf4)
- [Earth Engine with PyDeck](https://blog.gishub.org/earth-engine-tutorial-37-how-to-use-earth-engine-with-pydeck-for-3d-terrain-visualization)
- [Streamlit Demo PyDeck Maps](https://github.com/streamlit/demo-pydeck-maps)

### Color Tools
- ColorBrewer: https://colorbrewer2.org/
- Coolors: https://coolors.co/
