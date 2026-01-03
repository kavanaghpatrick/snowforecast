#!/usr/bin/env python3
"""
TEST 3: Map Rendering Test for Snowforecast Dashboard

Tests PyDeck map component rendering, resort markers, console errors, and performance.
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Configuration
BASE_URLS = [
    "https://snowforecast.streamlit.app",
    "https://kavanaghpatrick-snowforecast.streamlit.app"
]
SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots")
LOAD_TIMEOUT = 60000  # 60 seconds for Streamlit apps


async def test_map_rendering():
    """Test 3: Map Rendering Test"""

    results = {
        "test_name": "Map Rendering Test",
        "timestamp": datetime.now().isoformat(),
        "urls_tested": [],
        "console_errors": [],
        "console_warnings": [],
        "page_errors": [],
        "map_found": False,
        "map_element_type": None,
        "map_load_time": None,
        "resort_markers_visible": False,
        "screenshot_path": None,
        "success": False,
        "issues": []
    }

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Set up console and error listeners
        def handle_console(msg):
            msg_type = msg.type
            msg_text = msg.text
            entry = {"type": msg_type, "text": msg_text}
            if msg_type == "error":
                results["console_errors"].append(entry)
                print(f"Console ERROR: {msg_text}")
            elif msg_type == "warning":
                results["console_warnings"].append(entry)
                print(f"Console WARNING: {msg_text}")

        def handle_pageerror(err):
            error_text = str(err)
            results["page_errors"].append(error_text)
            print(f"Page ERROR: {error_text}")

        page.on("console", handle_console)
        page.on("pageerror", handle_pageerror)

        # Try each URL
        for base_url in BASE_URLS:
            print(f"\n{'='*60}")
            print(f"Testing: {base_url}")
            print('='*60)

            try:
                start_time = time.time()

                # Navigate to the page
                print("Navigating to page...")
                await page.goto(base_url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT)
                results["urls_tested"].append(base_url)

                # Wait for Streamlit to fully load
                print("Waiting for Streamlit to load...")
                await page.wait_for_load_state("networkidle", timeout=LOAD_TIMEOUT)

                # Additional wait for dynamic content
                await asyncio.sleep(5)

                page_load_time = time.time() - start_time
                print(f"Page loaded in {page_load_time:.2f}s")

                # Look for map elements
                print("\nSearching for map component...")
                map_start_time = time.time()

                # Selectors for PyDeck map components in Streamlit
                map_selectors = [
                    '[data-testid="stDeckGlJsonChart"]',  # Streamlit's PyDeck wrapper
                    'iframe[title*="deck"]',              # deck.gl iframe
                    'iframe[src*="deck"]',                # deck.gl iframe by src
                    '.stDeckGlJsonChart',                 # Alternative class selector
                    'canvas',                              # Canvas element (deck.gl renders to canvas)
                    'iframe',                              # Any iframe (PyDeck might use iframe)
                    '.mapboxgl-map',                       # Mapbox GL map
                    '.folium-map',                         # Folium map
                    '[data-testid="stPydeckChart"]',       # Alternative Streamlit selector
                ]

                map_element = None
                map_element_type = None

                for selector in map_selectors:
                    try:
                        element = page.locator(selector).first
                        if await element.count() > 0:
                            is_visible = await element.is_visible()
                            if is_visible:
                                map_element = element
                                map_element_type = selector
                                print(f"  Found map element: {selector}")
                                break
                    except Exception as e:
                        continue

                map_search_time = time.time() - map_start_time

                if map_element:
                    results["map_found"] = True
                    results["map_element_type"] = map_element_type
                    results["map_load_time"] = map_search_time
                    print(f"  Map element type: {map_element_type}")
                    print(f"  Map located in {map_search_time:.2f}s")

                    # Get bounding box of map element
                    try:
                        bbox = await map_element.bounding_box()
                        if bbox:
                            print(f"  Map dimensions: {bbox['width']:.0f}x{bbox['height']:.0f}")
                    except:
                        pass

                    # Check for resort markers
                    print("\nChecking for resort markers...")

                    # Look for marker indicators in the page content
                    page_content = await page.content()
                    marker_indicators = [
                        "ScatterplotLayer",
                        "IconLayer",
                        "ski_area",
                        "resort",
                        "snow_depth",
                        "marker"
                    ]

                    found_indicators = []
                    for indicator in marker_indicators:
                        if indicator.lower() in page_content.lower():
                            found_indicators.append(indicator)

                    if found_indicators:
                        results["resort_markers_visible"] = True
                        print(f"  Found marker indicators: {', '.join(found_indicators)}")
                    else:
                        print("  No marker indicators found in page content")
                        results["issues"].append("No resort marker indicators found")

                    # Take screenshot of map area
                    print("\nTaking screenshots...")

                    # Full page screenshot
                    full_screenshot_path = SCREENSHOT_DIR / f"map_test_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=str(full_screenshot_path), full_page=True)
                    print(f"  Full page: {full_screenshot_path}")

                    # Map element screenshot
                    try:
                        map_screenshot_path = SCREENSHOT_DIR / f"map_component_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        await map_element.screenshot(path=str(map_screenshot_path))
                        results["screenshot_path"] = str(map_screenshot_path)
                        print(f"  Map component: {map_screenshot_path}")
                    except Exception as e:
                        print(f"  Could not capture map element: {e}")
                        # Take viewport screenshot instead
                        viewport_path = SCREENSHOT_DIR / f"map_viewport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        await page.screenshot(path=str(viewport_path))
                        results["screenshot_path"] = str(viewport_path)
                        print(f"  Viewport: {viewport_path}")

                    results["success"] = True
                    break  # Success, don't try other URLs

                else:
                    print("  No map element found!")
                    results["issues"].append(f"No map element found at {base_url}")

                    # Take screenshot anyway for debugging
                    debug_screenshot_path = SCREENSHOT_DIR / f"debug_no_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=str(debug_screenshot_path), full_page=True)
                    results["screenshot_path"] = str(debug_screenshot_path)
                    print(f"  Debug screenshot: {debug_screenshot_path}")

            except PlaywrightTimeout as e:
                print(f"Timeout: {e}")
                results["issues"].append(f"Timeout at {base_url}: {str(e)}")
            except Exception as e:
                print(f"Error: {e}")
                results["issues"].append(f"Error at {base_url}: {str(e)}")

        await browser.close()

    return results


def print_report(results):
    """Print a formatted test report."""
    print("\n" + "="*70)
    print("TEST 3: MAP RENDERING TEST REPORT")
    print("="*70)

    print(f"\nTimestamp: {results['timestamp']}")
    print(f"URLs Tested: {', '.join(results['urls_tested'])}")

    print(f"\n--- MAP STATUS ---")
    print(f"Map Found: {'YES' if results['map_found'] else 'NO'}")
    print(f"Map Element Type: {results['map_element_type'] or 'N/A'}")
    print(f"Map Load Time: {results['map_load_time']:.2f}s" if results['map_load_time'] else "Map Load Time: N/A")
    print(f"Resort Markers Visible: {'YES' if results['resort_markers_visible'] else 'NO'}")

    print(f"\n--- CONSOLE ERRORS ({len(results['console_errors'])}) ---")
    if results['console_errors']:
        for i, err in enumerate(results['console_errors'][:10], 1):  # Limit to 10
            print(f"  {i}. {err['text'][:100]}...")
    else:
        print("  None")

    print(f"\n--- CONSOLE WARNINGS ({len(results['console_warnings'])}) ---")
    if results['console_warnings']:
        # Filter for PyDeck/memory related warnings
        relevant_warnings = [w for w in results['console_warnings']
                          if any(term in w['text'].lower() for term in ['pydeck', 'deck', 'memory', 'gl', 'canvas', 'map'])]
        if relevant_warnings:
            for i, warn in enumerate(relevant_warnings[:5], 1):
                print(f"  {i}. {warn['text'][:100]}...")
        else:
            print(f"  {len(results['console_warnings'])} warnings (none related to PyDeck/memory)")
    else:
        print("  None")

    print(f"\n--- PAGE ERRORS ({len(results['page_errors'])}) ---")
    if results['page_errors']:
        for i, err in enumerate(results['page_errors'][:5], 1):
            print(f"  {i}. {err[:150]}...")
    else:
        print("  None")

    print(f"\n--- ISSUES ---")
    if results['issues']:
        for issue in results['issues']:
            print(f"  - {issue}")
    else:
        print("  None")

    print(f"\n--- SCREENSHOTS ---")
    print(f"  Path: {results['screenshot_path'] or 'None'}")

    print(f"\n--- OVERALL RESULT ---")
    if results['success']:
        print("  PASS: Map renders correctly")
    else:
        print("  FAIL: Map rendering issues detected")

    # Summary of PyDeck/memory related issues
    pydeck_errors = [e for e in results['console_errors']
                    if any(term in e['text'].lower() for term in ['pydeck', 'deck', 'gl', 'webgl', 'memory'])]
    if pydeck_errors:
        print(f"\n  WARNING: {len(pydeck_errors)} PyDeck/WebGL related errors found!")

    print("\n" + "="*70)


async def main():
    """Run the map rendering test."""
    results = await test_map_rendering()
    print_report(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
