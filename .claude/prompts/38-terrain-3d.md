# Agent Task: 3D Terrain Layer (#38)

## Prerequisites
- Issue #48 (Color Scale) complete
- Issue #35 (Resort Map) complete or in progress

## Your Mission
Add 3D terrain visualization using PyDeck TerrainLayer with FREE AWS Terrain Tiles.

## IMPORTANT: FREE TILES ONLY
Do NOT use Mapbox tiles. Use only free sources:
- Elevation: AWS Terrain Tiles (free)
- Texture: OpenTopoMap (free) - MUST VALIDATE THIS WORKS

## Files to Create

### `src/snowforecast/dashboard/components/terrain_layer.py`
```python
import pydeck as pdk

# FREE AWS Terrain Tiles
TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

ELEVATION_DECODER = {
    "rScaler": 256,
    "gScaler": 1,
    "bScaler": 1 / 256,
    "offset": -32768
}

# FREE texture - OpenTopoMap (validate this works!)
TEXTURE_IMAGE = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"

def create_terrain_layer(use_texture: bool = True) -> pdk.Layer:
    """Create 3D terrain layer with free tiles."""
    return pdk.Layer(
        "TerrainLayer",
        elevation_data=TERRAIN_IMAGE,
        elevation_decoder=ELEVATION_DECODER,
        texture=TEXTURE_IMAGE if use_texture else None,
        wireframe=not use_texture,
    )

def create_3d_view(lat: float = 40.0, lon: float = -111.0) -> pdk.ViewState:
    """Create 3D view state."""
    return pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=9,
        pitch=60,
        bearing=30,
    )
```

## SPIKE: Validate OpenTopoMap Texture
Before full implementation, test that OpenTopoMap works as TerrainLayer texture:
```python
import pydeck as pdk
deck = pdk.Deck(layers=[create_terrain_layer()], ...)
# If texture fails, use wireframe fallback
```

## UI Controls
- Toggle: "2D / 3D View"
- If 3D: pitch slider (0-60)
- Disable 3D on mobile (detect viewport)

## Worktree
Work in: `/Users/patrickkavanagh/snowforecast-worktrees/terrain-3d`
Branch: `phase6/38-terrain-3d`
