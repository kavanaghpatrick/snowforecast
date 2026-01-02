#!/usr/bin/env python3
"""Validate that free terrain tile sources are accessible and working.

This spike script tests:
1. AWS Terrain Tiles (elevation data)
2. OpenTopoMap (texture overlay)

Run with: python scripts/validate_terrain_tiles.py
"""

import sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Test coordinates: Snowbird, Utah (40.5830, -111.6538)
# At zoom 10, tile: x=201, y=388
TEST_ZOOM = 10
TEST_X = 201
TEST_Y = 388

# Tile sources to validate
TILE_SOURCES = {
    "AWS Terrain Tiles (elevation)": f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{TEST_ZOOM}/{TEST_X}/{TEST_Y}.png",
    "OpenTopoMap (texture)": f"https://a.tile.opentopomap.org/{TEST_ZOOM}/{TEST_X}/{TEST_Y}.png",
    "OpenStreetMap (fallback texture)": f"https://a.tile.openstreetmap.org/{TEST_ZOOM}/{TEST_X}/{TEST_Y}.png",
}

# User agent to avoid 403 errors (some tile servers require it)
HEADERS = {
    "User-Agent": "SnowForecast/0.1 (terrain validation script)"
}


def check_tile(name: str, url: str) -> tuple[bool, str]:
    """Check if a tile is accessible.

    Returns:
        (success, message) tuple
    """
    try:
        request = Request(url, headers=HEADERS)
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            content_length = len(response.read())

            if response.status == 200:
                # Verify it's actually an image
                if "image" in content_type or content_length > 1000:
                    return True, f"OK - {content_length} bytes, {content_type}"
                else:
                    return False, f"Unexpected content: {content_type}, {content_length} bytes"
            else:
                return False, f"HTTP {response.status}"

    except HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}"
    except URLError as e:
        return False, f"URL Error: {e.reason}"
    except TimeoutError:
        return False, "Timeout (>10s)"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run tile validation."""
    print("=" * 60)
    print("Terrain Tile Source Validation")
    print("=" * 60)
    print(f"Test tile: z={TEST_ZOOM}, x={TEST_X}, y={TEST_Y}")
    print(f"Location: Snowbird, Utah area")
    print("-" * 60)

    all_passed = True
    results = {}

    for name, url in TILE_SOURCES.items():
        print(f"\nChecking: {name}")
        print(f"  URL: {url}")

        success, message = check_tile(name, url)
        results[name] = (success, message)

        if success:
            print(f"  Status: PASS - {message}")
        else:
            print(f"  Status: FAIL - {message}")
            if "elevation" not in name:
                # Texture failure is concerning but not critical
                print("  (texture failure - wireframe fallback available)")
            else:
                all_passed = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Critical check: elevation tiles
    elevation_ok, _ = results.get("AWS Terrain Tiles (elevation)", (False, ""))
    if elevation_ok:
        print("Elevation data: AVAILABLE (AWS Terrain Tiles)")
    else:
        print("Elevation data: UNAVAILABLE - 3D terrain will not work!")
        all_passed = False

    # Texture check
    opentopomap_ok, _ = results.get("OpenTopoMap (texture)", (False, ""))
    osm_ok, _ = results.get("OpenStreetMap (fallback texture)", (False, ""))

    if opentopomap_ok:
        print("Texture: OpenTopoMap AVAILABLE (primary)")
    elif osm_ok:
        print("Texture: OpenStreetMap AVAILABLE (fallback)")
        print("  Recommendation: Use OSM texture instead of OpenTopoMap")
    else:
        print("Texture: UNAVAILABLE - use wireframe mode")

    print("\n" + "-" * 60)
    if all_passed and (opentopomap_ok or osm_ok):
        print("Result: All critical tiles accessible")
        print("Ready for production use")
        return 0
    elif all_passed:
        print("Result: Elevation available, texture unavailable")
        print("Use wireframe mode: create_terrain_layer(wireframe=True)")
        return 0
    else:
        print("Result: CRITICAL FAILURE - elevation tiles inaccessible")
        return 1


if __name__ == "__main__":
    sys.exit(main())
