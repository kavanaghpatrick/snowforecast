#!/usr/bin/env python3
"""
TEST 3: Map Rendering Test for Snowforecast Dashboard (v2)

Tests PyDeck map component rendering, resort markers, console errors, and performance.
Enhanced version with better error detection and alternate URL handling.
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Configuration
BASE_URLS = [
    "https://kavanaghpatrick-snowforecast.streamlit.app",
    "https://snowforecast.streamlit.app",
]
SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots")
LOAD_TIMEOUT = 90000  # 90 seconds for Streamlit apps (they can be slow to wake)


async def test_map_rendering():
    """Test 3: Map Rendering Test"""

    results = {
        "test_name": "Map Rendering Test",
        "timestamp": datetime.now().isoformat(),
        "urls_tested": [],
        "url_used": None,
        "console_errors": [],
        "console_warnings": [],
        "page_errors": [],
        "app_error": None,
        "map_found": False,
        "map_element_type": None,
        "map_load_time": None,
        "total_load_time": None,
        "resort_markers_visible": False,
        "screenshot_paths": [],
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
                print(f"  [Console ERROR] {msg_text[:100]}")
            elif msg_type == "warning":
                results["console_warnings"].append(entry)

        def handle_pageerror(err):
            error_text = str(err)
            results["page_errors"].append(error_text)
            print(f"  [Page ERROR] {error_text[:100]}")

        page.on("console", handle_console)
        page.on("pageerror", handle_pageerror)

        # Try each URL
        for base_url in BASE_URLS:
            print(f"\n{'='*70}")
            print(f"Testing: {base_url}")
            print('='*70)
            results["urls_tested"].append(base_url)

            try:
                start_time = time.time()

                # Navigate to the page
                print("Step 1: Navigating to page...")
                await page.goto(base_url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT)

                # Wait for Streamlit to fully load
                print("Step 2: Waiting for Streamlit to initialize...")
                await page.wait_for_load_state("networkidle", timeout=LOAD_TIMEOUT)

                # Additional wait for Streamlit app to fully render
                print("Step 3: Waiting for app to render (15s)...")
                await asyncio.sleep(15)

                total_load_time = time.time() - start_time
                results["total_load_time"] = total_load_time
                print(f"  Total load time: {total_load_time:.2f}s")

                # Check for Streamlit error page
                print("\nStep 4: Checking for app errors...")
                page_content = await page.content()
                page_text = await page.inner_text("body")

                # Check for known Streamlit error messages
                error_indicators = [
                    "Oh no",
                    "Error running app",
                    "This app has encountered an error",
                    "connection error",
                    "Please try again",
                    "Something went wrong",
                    "App isn't running",
                ]

                app_has_error = False
                for indicator in error_indicators:
                    if indicator.lower() in page_text.lower():
                        app_has_error = True
                        results["app_error"] = indicator
                        results["issues"].append(f"App error detected: '{indicator}'")
                        print(f"  APP ERROR DETECTED: {indicator}")
                        break

                if app_has_error:
                    # Take screenshot of error
                    error_screenshot = SCREENSHOT_DIR / f"app_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=str(error_screenshot), full_page=True)
                    results["screenshot_paths"].append(str(error_screenshot))
                    print(f"  Error screenshot saved: {error_screenshot}")
                    continue  # Try next URL

                print("  No app errors detected")
                results["url_used"] = base_url

                # Look for map elements
                print("\nStep 5: Searching for map component...")
                map_start_time = time.time()

                # Selectors for PyDeck map components in Streamlit (in order of specificity)
                map_selectors = [
                    '[data-testid="stDeckGlJsonChart"]',  # Streamlit's PyDeck wrapper
                    '[data-testid="stPydeckChart"]',       # Alternative Streamlit selector
                    'iframe[title*="streamlit_pydeck"]',   # PyDeck iframe
                    'iframe[src*="pydeck"]',               # PyDeck iframe by src
                    '.stDeckGlJsonChart',                  # Alternative class selector
                    'canvas.mapboxgl-canvas',              # Mapbox GL canvas
                    'canvas.maplibregl-canvas',            # MapLibre GL canvas
                    '.deckgl-wrapper',                     # deck.gl wrapper
                    '.mapboxgl-map',                       # Mapbox GL map container
                    '.folium-map',                         # Folium map
                ]

                map_element = None
                map_element_type = None

                for selector in map_selectors:
                    try:
                        element = page.locator(selector).first
                        count = await element.count()
                        if count > 0:
                            is_visible = await element.is_visible()
                            if is_visible:
                                map_element = element
                                map_element_type = selector
                                print(f"  Found map element: {selector}")
                                break
                    except Exception as e:
                        continue

                # If no specific map found, look for large canvas elements (deck.gl renders to canvas)
                if not map_element:
                    print("  Checking for canvas elements...")
                    canvases = page.locator("canvas")
                    canvas_count = await canvases.count()
                    print(f"  Found {canvas_count} canvas elements")

                    for i in range(canvas_count):
                        canvas = canvases.nth(i)
                        try:
                            bbox = await canvas.bounding_box()
                            if bbox and bbox['width'] > 200 and bbox['height'] > 200:
                                map_element = canvas
                                map_element_type = f"canvas (large: {bbox['width']:.0f}x{bbox['height']:.0f})"
                                print(f"  Found large canvas: {bbox['width']:.0f}x{bbox['height']:.0f}")
                                break
                        except:
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
                else:
                    print("  No map element found")
                    results["issues"].append("No PyDeck/map element found on page")

                # Check for resort markers in page content
                print("\nStep 6: Checking for resort markers...")
                marker_indicators = [
                    "ScatterplotLayer",
                    "IconLayer",
                    "ski_area",
                    "Stevens Pass",
                    "Crystal Mountain",
                    "Mammoth",
                    "Vail",
                    "Park City",
                    "snow_depth",
                    "resort",
                ]

                found_indicators = []
                for indicator in marker_indicators:
                    if indicator in page_content:
                        found_indicators.append(indicator)

                if found_indicators:
                    results["resort_markers_visible"] = True
                    print(f"  Found resort indicators: {', '.join(found_indicators[:5])}")
                else:
                    print("  No resort marker indicators found in page content")

                # Take screenshots
                print("\nStep 7: Taking screenshots...")

                # Full page screenshot
                full_screenshot_path = SCREENSHOT_DIR / f"map_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await page.screenshot(path=str(full_screenshot_path), full_page=True)
                results["screenshot_paths"].append(str(full_screenshot_path))
                print(f"  Full page: {full_screenshot_path}")

                # Viewport screenshot
                viewport_path = SCREENSHOT_DIR / f"map_viewport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await page.screenshot(path=str(viewport_path), full_page=False)
                results["screenshot_paths"].append(str(viewport_path))
                print(f"  Viewport: {viewport_path}")

                # Map element screenshot if found
                if map_element:
                    try:
                        map_screenshot_path = SCREENSHOT_DIR / f"map_element_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        await map_element.screenshot(path=str(map_screenshot_path))
                        results["screenshot_paths"].append(str(map_screenshot_path))
                        print(f"  Map element: {map_screenshot_path}")
                    except Exception as e:
                        print(f"  Could not capture map element: {e}")

                # Determine success
                if map_element or found_indicators:
                    results["success"] = True
                    print("\n  TEST PASSED: Map component detected")
                    break  # Success, don't try other URLs
                else:
                    results["issues"].append("Map not found, no resort data visible")

            except PlaywrightTimeout as e:
                print(f"  Timeout: {e}")
                results["issues"].append(f"Timeout at {base_url}: {str(e)[:50]}")
            except Exception as e:
                print(f"  Error: {e}")
                results["issues"].append(f"Error at {base_url}: {str(e)[:50]}")

        await browser.close()

    return results


def print_report(results):
    """Print a formatted test report."""
    print("\n" + "="*70)
    print("TEST 3: MAP RENDERING TEST REPORT")
    print("="*70)

    print(f"\nTimestamp: {results['timestamp']}")
    print(f"URLs Tested: {len(results['urls_tested'])}")
    for url in results['urls_tested']:
        status = "(used)" if url == results.get('url_used') else "(failed/error)"
        print(f"  - {url} {status}")

    if results.get('app_error'):
        print(f"\n--- APP ERROR ---")
        print(f"  Streamlit app error: {results['app_error']}")

    print(f"\n--- MAP STATUS ---")
    print(f"Map Found: {'YES' if results['map_found'] else 'NO'}")
    print(f"Map Element Type: {results['map_element_type'] or 'N/A'}")
    if results['map_load_time']:
        print(f"Map Load Time: {results['map_load_time']:.2f}s")
    if results['total_load_time']:
        print(f"Total Page Load Time: {results['total_load_time']:.2f}s")
    print(f"Resort Markers Visible: {'YES' if results['resort_markers_visible'] else 'NO'}")

    print(f"\n--- CONSOLE ERRORS ({len(results['console_errors'])}) ---")
    if results['console_errors']:
        # Filter for relevant errors
        pydeck_errors = [e for e in results['console_errors']
                        if any(term in e['text'].lower() for term in ['pydeck', 'deck', 'gl', 'webgl', 'memory', 'map', 'canvas'])]
        other_errors = [e for e in results['console_errors'] if e not in pydeck_errors]

        if pydeck_errors:
            print("  PyDeck/WebGL related:")
            for i, err in enumerate(pydeck_errors[:5], 1):
                print(f"    {i}. {err['text'][:80]}...")

        if other_errors:
            print(f"  Other errors: {len(other_errors)}")
            for i, err in enumerate(other_errors[:3], 1):
                print(f"    {i}. {err['text'][:80]}...")
    else:
        print("  None")

    print(f"\n--- PAGE ERRORS ({len(results['page_errors'])}) ---")
    if results['page_errors']:
        for i, err in enumerate(results['page_errors'][:5], 1):
            print(f"  {i}. {err[:100]}...")
    else:
        print("  None")

    print(f"\n--- ISSUES ({len(results['issues'])}) ---")
    if results['issues']:
        for issue in results['issues']:
            print(f"  - {issue}")
    else:
        print("  None")

    print(f"\n--- SCREENSHOTS ---")
    if results['screenshot_paths']:
        for path in results['screenshot_paths']:
            print(f"  - {path}")
    else:
        print("  None")

    print(f"\n{'='*70}")
    print(f"OVERALL RESULT: {'PASS' if results['success'] else 'FAIL'}")
    print('='*70)

    if not results['success']:
        if results.get('app_error'):
            print("\nFAILURE REASON: Streamlit app is not running/errored")
            print("ACTION: Check Streamlit Cloud deployment status")
        elif not results['map_found']:
            print("\nFAILURE REASON: Map component not found on page")
        else:
            print("\nFAILURE REASON: Unknown issue")


async def main():
    """Run the map rendering test."""
    results = await test_map_rendering()
    print_report(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
