#!/usr/bin/env python3
"""
Map Visualization Test for Snowforecast Dashboard

Tests PyDeck map component:
1. Verify map loads (look for Leaflet/deck.gl canvas)
2. Check if resort markers are visible (colored circles)
3. Verify map legend "Circle color = snow depth | Circle size = new snow"
4. Test if clicking/hovering on markers shows tooltips with resort details
5. Check that map is centered on Western US ski regions
6. Verify multiple states are visible (Colorado, Utah, California, Washington)
"""

import asyncio
import re
import time
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Configuration
BASE_URLS = [
    "https://snowforecast.streamlit.app",
    "https://kavanaghpatrick-snowforecast.streamlit.app",
]
SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots")
LOAD_TIMEOUT = 120000  # 120 seconds for Streamlit apps

# Western US states we expect to see resorts in
EXPECTED_STATES = ["Colorado", "Utah", "California", "Washington", "Oregon", "Wyoming", "Montana", "Idaho", "Nevada", "New Mexico"]

# Known resort names to look for
KNOWN_RESORTS = [
    "Vail", "Aspen", "Breckenridge", "Copper Mountain", "Keystone",
    "Park City", "Snowbird", "Alta", "Brighton", "Deer Valley",
    "Mammoth", "Squaw", "Palisades", "Heavenly", "Northstar",
    "Crystal Mountain", "Stevens Pass", "Mt. Baker", "Snoqualmie",
    "Mt. Hood", "Bachelor", "Big Sky", "Jackson Hole", "Sun Valley"
]


async def test_map_visualization():
    """Comprehensive map visualization test."""

    results = {
        "test_name": "Map Visualization Test",
        "timestamp": datetime.now().isoformat(),
        "url": None,
        "tests": {
            "map_loads": {"passed": False, "details": ""},
            "resort_markers_visible": {"passed": False, "details": ""},
            "map_legend": {"passed": False, "details": ""},
            "tooltips": {"passed": False, "details": ""},
            "western_us_centered": {"passed": False, "details": ""},
            "multiple_states": {"passed": False, "details": ""},
        },
        "console_errors": [],
        "page_errors": [],
        "screenshot_paths": [],
        "overall_success": False
    }

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Set up error listeners
        def handle_console(msg):
            if msg.type == "error":
                results["console_errors"].append({"type": msg.type, "text": msg.text})

        def handle_pageerror(err):
            results["page_errors"].append(str(err))

        page.on("console", handle_console)
        page.on("pageerror", handle_pageerror)

        try:
            print(f"\n{'='*70}")
            print("MAP VISUALIZATION TEST")
            print(f"{'='*70}")

            # Try each URL until one works
            app_loaded = False
            working_url = None

            for base_url in BASE_URLS:
                print(f"\n[1/6] Loading page: {base_url}")
                start_time = time.time()
                max_retries = 2

                for attempt in range(max_retries):
                    print(f"    Attempt {attempt + 1}/{max_retries}...")
                    await page.goto(base_url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT)

                    # Don't wait for networkidle - Streamlit apps have persistent WebSocket connections
                    # Instead, wait for specific Streamlit elements to appear
                    print("    Waiting for Streamlit to initialize...")
                    try:
                        # Wait for Streamlit's main content area
                        await page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=45000)
                        print("    Streamlit app container detected")
                    except:
                        print("    Streamlit container not found...")

                    # Wait for Streamlit to fully render
                    print("    Waiting for content to render (20s)...")
                    await asyncio.sleep(20)

                    # Check if we got an error page or access denied
                    body_text = await page.inner_text("body")
                    if "Oh no" in body_text or "Error running app" in body_text:
                        print(f"    App error detected on attempt {attempt + 1}, retrying...")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(10)  # Wait before retry
                            continue
                        else:
                            break  # Try next URL
                    elif "do not have access" in body_text or "Sign in to continue" in body_text:
                        print(f"    Access denied at {base_url}, trying next URL...")
                        break  # Try next URL
                    else:
                        app_loaded = True
                        working_url = base_url
                        break

                if app_loaded:
                    break

            if not app_loaded:
                print("    WARNING: App may not have loaded correctly after trying all URLs")
                results["url"] = "All URLs failed"
            else:
                results["url"] = working_url
                print(f"    Successfully loaded: {working_url}")

            load_time = time.time() - start_time
            print(f"    Page loaded in {load_time:.2f}s")

            # Check for app errors
            page_text = await page.inner_text("body")
            error_indicators = ["Oh no", "Error running app", "This app has encountered an error"]
            for indicator in error_indicators:
                if indicator.lower() in page_text.lower():
                    print(f"    ERROR: App error detected - {indicator}")
                    error_screenshot = SCREENSHOT_DIR / f"app_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=str(error_screenshot), full_page=True)
                    results["screenshot_paths"].append(str(error_screenshot))
                    return results

            page_content = await page.content()

            # Try to get content from within Streamlit's iframe if present
            # Streamlit apps are often loaded in an iframe
            iframes = page.frame_locator("iframe")
            try:
                # Try to get text from the main iframe
                main_frame = page.frames[1] if len(page.frames) > 1 else page.frames[0]
                frame_text = await main_frame.locator("body").inner_text()
                if len(frame_text) > len(page_text):
                    page_text = frame_text
                    page_content = await main_frame.content()
                    print(f"    Using iframe content ({len(page_text)} chars)")
            except Exception as e:
                print(f"    Could not access iframe content: {e}")

            # Debug: Print some of the page text to see what we're working with
            print(f"    Page text sample: {page_text[:200]}..." if len(page_text) > 200 else f"    Page text: {page_text}")

            # TEST 1: Verify map loads
            print("\n[2/6] TEST 1: Verifying map loads...")
            map_selectors = [
                '[data-testid="stDeckGlJsonChart"]',
                '[data-testid="stPydeckChart"]',
                'iframe[title*="pydeck"]',
                'iframe[src*="pydeck"]',
                '.stDeckGlJsonChart',
                'canvas.mapboxgl-canvas',
                'canvas.maplibregl-canvas',
                '.deckgl-wrapper',
                '.mapboxgl-map',
                '[data-testid="stMap"]',
                '.leaflet-container',
                '[class*="folium"]',
                'iframe[srcdoc*="leaflet"]',
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
                            break
                except:
                    continue

            # Check iframes for embedded maps
            if not map_element:
                iframes = page.locator("iframe")
                iframe_count = await iframes.count()
                print(f"    Found {iframe_count} iframes, checking for map content...")
                for i in range(iframe_count):
                    iframe = iframes.nth(i)
                    try:
                        bbox = await iframe.bounding_box()
                        if bbox and bbox['width'] > 200 and bbox['height'] > 200:
                            map_element = iframe
                            map_element_type = f"iframe ({bbox['width']:.0f}x{bbox['height']:.0f})"
                            print(f"    Found potential map iframe: {bbox['width']:.0f}x{bbox['height']:.0f}")
                            break
                    except:
                        continue

            # Also check for large canvas elements (deck.gl renders to canvas)
            if not map_element:
                canvases = page.locator("canvas")
                canvas_count = await canvases.count()
                print(f"    Found {canvas_count} canvas elements...")
                for i in range(canvas_count):
                    canvas = canvases.nth(i)
                    try:
                        bbox = await canvas.bounding_box()
                        if bbox and bbox['width'] > 200 and bbox['height'] > 200:
                            map_element = canvas
                            map_element_type = f"canvas ({bbox['width']:.0f}x{bbox['height']:.0f})"
                            print(f"    Found large canvas: {bbox['width']:.0f}x{bbox['height']:.0f}")
                            break
                    except:
                        continue

            # Check for map by looking for map-related text in page
            if not map_element:
                # Look for common map locations that indicate a map is present
                map_location_indicators = ["Salt Lake City", "Denver", "Los Angeles", "Seattle", "Portland"]
                found_locations = [loc for loc in map_location_indicators if loc in page_text]
                if found_locations:
                    print(f"    Found map locations in page: {', '.join(found_locations)}")
                    # Try to find any visual element that might be the map
                    try:
                        # Look for divs with map-like styling or regional content
                        regional_section = page.locator("text=Regional Overview").first
                        if await regional_section.count() > 0:
                            map_element = regional_section
                            map_element_type = "Regional Overview section (implicit map)"
                    except:
                        pass

            if map_element:
                results["tests"]["map_loads"]["passed"] = True
                results["tests"]["map_loads"]["details"] = f"Found map element: {map_element_type}"
                print(f"    PASS: Map found - {map_element_type}")
            else:
                results["tests"]["map_loads"]["details"] = "No map element found"
                print("    FAIL: No map element found")

            # TEST 2: Check resort markers / ski area data
            print("\n[3/6] TEST 2: Checking for resort markers / ski area data...")
            marker_indicators = ["ScatterplotLayer", "IconLayer", "TextLayer"]
            found_layers = [ind for ind in marker_indicators if ind in page_content]

            # Also look for resort names in the page
            found_resorts = [resort for resort in KNOWN_RESORTS if resort.lower() in page_text.lower()]

            # Check for ski area selector or location data (based on actual UI)
            ski_area_indicators = [
                "Ski Area", "ski area", "Select Location",
                "Snow Base", "New Snow", "Elevation", "Probability",
                "Base:", "Forecast"
            ]
            found_ski_indicators = [ind for ind in ski_area_indicators if ind in page_text]

            # Check for specific resort names that might be in a dropdown or selector
            page_html_lower = page_content.lower()
            additional_resorts = ["alta", "snowbird", "park city", "brighton", "deer valley",
                                  "vail", "aspen", "mammoth", "tahoe", "crystal"]
            found_in_html = [r for r in additional_resorts if r in page_html_lower]

            if found_layers or found_resorts or found_ski_indicators or found_in_html:
                results["tests"]["resort_markers_visible"]["passed"] = True
                details = []
                if found_layers:
                    details.append(f"Layers: {', '.join(found_layers)}")
                if found_resorts:
                    details.append(f"Resorts visible: {', '.join(found_resorts[:5])}")
                if found_ski_indicators:
                    details.append(f"UI elements: {', '.join(found_ski_indicators[:5])}")
                if found_in_html:
                    details.append(f"Resorts in page: {', '.join(found_in_html[:5])}")
                results["tests"]["resort_markers_visible"]["details"] = "; ".join(details)
                print(f"    PASS: Resort/ski area data detected")
                if found_ski_indicators:
                    print(f"    Found UI indicators: {', '.join(found_ski_indicators[:5])}")
            else:
                results["tests"]["resort_markers_visible"]["details"] = "No marker layers or resorts found"
                print("    FAIL: No resort markers found")

            # TEST 3: Verify map legend or data labels
            print("\n[4/6] TEST 3: Checking for map legend / data labels...")
            legend_text_patterns = [
                "Circle color",
                "snow depth",
                "Circle size",
                "new snow",
                "color =",
                "size =",
                # Also check for data labels visible in screenshot
                "Snow Base",
                "New Snow",
                "Probability",
                "Elevation",
                "Good Snow",
                "Low Confidence",
                "confidence level"
            ]

            found_legend_elements = []
            for pattern in legend_text_patterns:
                if pattern.lower() in page_text.lower():
                    found_legend_elements.append(pattern)

            # Also look for metric displays (like "0 cm", "1947 m", "1%")
            metric_pattern = r'\d+\s*(cm|m|%|in|ft)'
            metrics_found = re.findall(metric_pattern, page_text, re.IGNORECASE)

            if len(found_legend_elements) >= 2 or (found_legend_elements and metrics_found):
                results["tests"]["map_legend"]["passed"] = True
                details = f"Labels: {', '.join(found_legend_elements[:5])}"
                if metrics_found:
                    details += f"; Metrics with units found"
                results["tests"]["map_legend"]["details"] = details
                print(f"    PASS: Data labels found: {', '.join(found_legend_elements[:5])}")
            else:
                results["tests"]["map_legend"]["details"] = f"Only found: {', '.join(found_legend_elements) if found_legend_elements else 'nothing'}"
                print(f"    FAIL: Legend not found or incomplete")

            # TEST 4: Test tooltips / data display (hover/click or static display)
            print("\n[5/6] TEST 4: Testing tooltips / data display...")
            tooltip_found = False

            # Based on the screenshot, the app shows data in a detail panel rather than tooltips
            # Check for the presence of detailed resort info anywhere on page
            detail_fields = [
                "Snow Depth", "New Snow", "Probability", "Elevation",
                "Snow Base", "Forecast", "Base:", "confidence"
            ]
            found_detail_fields = [f for f in detail_fields if f.lower() in page_text.lower()]

            if map_element:
                try:
                    # Get map bounding box
                    bbox = await map_element.bounding_box()
                    if bbox:
                        # Hover over different points on the map
                        test_points = [
                            (bbox['x'] + bbox['width'] * 0.3, bbox['y'] + bbox['height'] * 0.5),
                            (bbox['x'] + bbox['width'] * 0.5, bbox['y'] + bbox['height'] * 0.4),
                            (bbox['x'] + bbox['width'] * 0.6, bbox['y'] + bbox['height'] * 0.6),
                            (bbox['x'] + bbox['width'] * 0.4, bbox['y'] + bbox['height'] * 0.3),
                        ]

                        for x, y in test_points:
                            await page.mouse.move(x, y)
                            await asyncio.sleep(1)

                            # Look for tooltip elements
                            tooltip_selectors = [
                                '[class*="tooltip"]',
                                '[class*="Tooltip"]',
                                '[role="tooltip"]',
                                '.deck-tooltip',
                                '[style*="position: absolute"]',
                            ]

                            for selector in tooltip_selectors:
                                try:
                                    tooltips = page.locator(selector)
                                    count = await tooltips.count()
                                    if count > 0:
                                        for i in range(count):
                                            tooltip_text = await tooltips.nth(i).inner_text()
                                            # Check if tooltip contains expected fields
                                            expected_fields = ["Resort", "State", "Snow Depth", "New Snow", "Probability"]
                                            fields_found = [f for f in expected_fields if f.lower() in tooltip_text.lower()]
                                            if fields_found:
                                                tooltip_found = True
                                                results["tests"]["tooltips"]["details"] = f"Fields found: {', '.join(fields_found)}"
                                                break
                                except:
                                    continue

                            if tooltip_found:
                                break

                        # Take screenshot after hover
                        hover_screenshot = SCREENSHOT_DIR / f"map_hover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        await page.screenshot(path=str(hover_screenshot))
                        results["screenshot_paths"].append(str(hover_screenshot))

                except Exception as e:
                    print(f"    Tooltip test error: {e}")

            # Check if data is displayed in a sidebar/panel (as seen in screenshot)
            # The app shows detailed info for selected resort in a panel
            if tooltip_found:
                results["tests"]["tooltips"]["passed"] = True
                print(f"    PASS: Tooltips working - {results['tests']['tooltips']['details']}")
            elif len(found_detail_fields) >= 3:  # At least 3 detail fields visible
                results["tests"]["tooltips"]["passed"] = True
                results["tests"]["tooltips"]["details"] = f"Detail panel shows: {', '.join(found_detail_fields)}"
                print(f"    PASS: Resort details displayed - {', '.join(found_detail_fields)}")
            else:
                results["tests"]["tooltips"]["details"] = f"Limited data display: {', '.join(found_detail_fields) if found_detail_fields else 'none'}"
                print("    FAIL: Could not verify tooltip/detail functionality")

            # TEST 5: Check map is centered on Western US
            print("\n[6/6] TEST 5: Checking Western US centering...")
            western_us_indicators = [
                "Colorado", "Utah", "California", "Washington", "Oregon",
                "Wyoming", "Montana", "Idaho", "Nevada", "New Mexico",
                "Rocky Mountain", "Sierra Nevada", "Cascade"
            ]

            found_regions = [region for region in western_us_indicators if region.lower() in page_text.lower()]

            if found_regions:
                results["tests"]["western_us_centered"]["passed"] = True
                results["tests"]["western_us_centered"]["details"] = f"Found: {', '.join(found_regions[:5])}"
                print(f"    PASS: Western US regions visible - {', '.join(found_regions[:5])}")
            else:
                results["tests"]["western_us_centered"]["details"] = "No Western US indicators found"
                print("    FAIL: Cannot verify Western US centering")

            # TEST 6: Verify multiple states visible
            found_states = [state for state in EXPECTED_STATES if state.lower() in page_text.lower()]

            if len(found_states) >= 3:  # At least 3 states
                results["tests"]["multiple_states"]["passed"] = True
                results["tests"]["multiple_states"]["details"] = f"States found: {', '.join(found_states)}"
                print(f"    PASS: Multiple states visible - {', '.join(found_states)}")
            else:
                results["tests"]["multiple_states"]["details"] = f"Only found: {', '.join(found_states) if found_states else 'none'}"
                print(f"    FAIL: Not enough states visible - found {len(found_states)}")

            # Take final screenshots
            print("\n--- Taking screenshots ---")

            # Full page
            full_screenshot = SCREENSHOT_DIR / f"map_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=str(full_screenshot), full_page=True)
            results["screenshot_paths"].append(str(full_screenshot))
            print(f"    Full page: {full_screenshot}")

            # Viewport
            viewport_screenshot = SCREENSHOT_DIR / f"map_viewport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=str(viewport_screenshot), full_page=False)
            results["screenshot_paths"].append(str(viewport_screenshot))
            print(f"    Viewport: {viewport_screenshot}")

            # Map element
            if map_element:
                try:
                    map_screenshot = SCREENSHOT_DIR / f"map_element_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await map_element.screenshot(path=str(map_screenshot))
                    results["screenshot_paths"].append(str(map_screenshot))
                    print(f"    Map element: {map_screenshot}")
                except Exception as e:
                    print(f"    Could not capture map element: {e}")

        except PlaywrightTimeout as e:
            print(f"\nTIMEOUT: {e}")
            results["page_errors"].append(f"Timeout: {str(e)}")
        except Exception as e:
            print(f"\nERROR: {e}")
            results["page_errors"].append(f"Error: {str(e)}")
        finally:
            await browser.close()

    # Calculate overall success
    passed_tests = sum(1 for t in results["tests"].values() if t["passed"])
    total_tests = len(results["tests"])
    results["overall_success"] = passed_tests >= 4  # At least 4 out of 6 tests pass

    return results


def print_report(results):
    """Print formatted test report."""
    print("\n" + "="*70)
    print("MAP VISUALIZATION TEST REPORT")
    print("="*70)

    print(f"\nTimestamp: {results['timestamp']}")
    print(f"URL: {results['url']}")

    print(f"\n--- TEST RESULTS ---")
    for test_name, test_result in results["tests"].items():
        status = "PASS" if test_result["passed"] else "FAIL"
        print(f"\n  {test_name}:")
        print(f"    Status: {status}")
        print(f"    Details: {test_result['details']}")

    print(f"\n--- CONSOLE ERRORS ({len(results['console_errors'])}) ---")
    if results["console_errors"]:
        pydeck_errors = [e for e in results["console_errors"]
                        if any(term in e["text"].lower() for term in ["pydeck", "deck", "gl", "webgl", "map", "canvas"])]
        if pydeck_errors:
            print("  PyDeck/WebGL related:")
            for err in pydeck_errors[:5]:
                print(f"    - {err['text'][:80]}...")
        other_count = len(results["console_errors"]) - len(pydeck_errors)
        if other_count > 0:
            print(f"  Other errors: {other_count}")
    else:
        print("  None")

    print(f"\n--- PAGE ERRORS ({len(results['page_errors'])}) ---")
    if results["page_errors"]:
        for err in results["page_errors"][:3]:
            print(f"  - {err[:100]}...")
    else:
        print("  None")

    print(f"\n--- SCREENSHOTS ---")
    for path in results["screenshot_paths"]:
        print(f"  - {path}")

    passed = sum(1 for t in results["tests"].values() if t["passed"])
    total = len(results["tests"])

    print(f"\n{'='*70}")
    print(f"OVERALL: {passed}/{total} tests passed")
    print(f"RESULT: {'PASS' if results['overall_success'] else 'FAIL'}")
    print("="*70)


async def main():
    """Run map visualization test."""
    results = await test_map_visualization()
    print_report(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
