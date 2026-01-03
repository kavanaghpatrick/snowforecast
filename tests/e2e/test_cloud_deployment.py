"""Playwright test for Streamlit Cloud deployment verification.

Tests that:
1. The app loads successfully (no error messages)
2. The map displays with ski resort markers
3. Snow depth values are shown (not zero)
4. The detail panel shows forecast data (not "Loading forecast data...")
5. Cache status badge shows valid data (not "Unknown" or "Never")

Run with: python tests/e2e/test_cloud_deployment.py
"""

import time
import os
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Streamlit Cloud URL
BASE_URL = "https://snowforecast.streamlit.app"

# Timeouts (cloud can be slower)
LOAD_TIMEOUT = 120000  # 120 seconds for initial load
ELEMENT_TIMEOUT = 60000  # 60 seconds for elements
NETWORK_IDLE_TIMEOUT = 90000  # 90 seconds for network idle

# Screenshot directory
SCREENSHOT_DIR = "/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots/cloud"


def ensure_screenshot_dir():
    """Create screenshot directory if it doesn't exist."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def save_screenshot(page, name):
    """Save a screenshot with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{SCREENSHOT_DIR}/{name}_{timestamp}.png"
    page.screenshot(path=filepath, full_page=True)
    print(f"  Screenshot saved: {filepath}")
    return filepath


def run_cloud_deployment_test():
    """Run verification tests on Streamlit Cloud deployment."""
    ensure_screenshot_dir()

    results = {
        "app_loads": {"status": "not_tested", "details": ""},
        "map_displays": {"status": "not_tested", "details": ""},
        "snow_depth_values": {"status": "not_tested", "details": "", "values_found": []},
        "detail_panel_forecast": {"status": "not_tested", "details": ""},
        "cache_status": {"status": "not_tested", "details": ""},
        "screenshots": []
    }

    with sync_playwright() as p:
        print("=" * 70)
        print("SNOWFORECAST STREAMLIT CLOUD DEPLOYMENT VERIFICATION")
        print("=" * 70)
        print(f"Testing URL: {BASE_URL}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        # =========================================================
        # TEST 1: App Loads Successfully
        # =========================================================
        print(f"\n[TEST 1] App Loads Successfully")
        print("-" * 50)

        try:
            print(f"  Navigating to {BASE_URL}...")
            response = page.goto(BASE_URL, timeout=LOAD_TIMEOUT, wait_until="domcontentloaded")

            if not response:
                results["app_loads"] = {"status": "FAIL", "details": "No response from server"}
                print(f"  FAIL: No response from server")
                results["screenshots"].append(save_screenshot(page, "01_no_response"))
                browser.close()
                return results

            if not response.ok:
                results["app_loads"] = {"status": "FAIL", "details": f"HTTP {response.status}"}
                print(f"  FAIL: HTTP {response.status}")
                results["screenshots"].append(save_screenshot(page, "01_http_error"))
                browser.close()
                return results

            print(f"  HTTP Status: {response.status}")

            # Wait for network idle
            print("  Waiting for network idle...")
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)
            except PlaywrightTimeout:
                print("  Warning: Network didn't become idle within timeout, continuing...")

            # Wait for Streamlit app container
            print("  Waiting for Streamlit app to initialize...")
            page.wait_for_selector('[data-testid="stApp"]', timeout=ELEMENT_TIMEOUT)

            # Check for error messages
            error_indicators = [
                'text="Connection error"',
                'text="This app has encountered an error"',
                'text="Error"',
                'text="Something went wrong"',
                'text="Please contact the app developer"',
                '[data-testid="stException"]',
                '.stException',
            ]

            errors_found = []
            for selector in error_indicators:
                try:
                    error_el = page.locator(selector)
                    if error_el.count() > 0 and error_el.first.is_visible():
                        errors_found.append(selector)
                except:
                    pass

            if errors_found:
                results["app_loads"] = {"status": "FAIL", "details": f"Error messages found: {errors_found}"}
                print(f"  FAIL: Error messages found: {errors_found}")
                results["screenshots"].append(save_screenshot(page, "01_app_error"))
            else:
                # Wait for title
                print("  Waiting for app title...")
                try:
                    page.wait_for_selector('h1:has-text("Snow Forecast")', timeout=30000)
                    title_found = True
                except:
                    title_found = False

                # Give extra time for Streamlit to render
                print("  Waiting for rendering to complete...")
                time.sleep(10)

                results["app_loads"] = {"status": "PASS", "details": f"App loaded successfully, title found: {title_found}"}
                print(f"  PASS: App loaded successfully")

            results["screenshots"].append(save_screenshot(page, "01_initial_load"))

        except PlaywrightTimeout as e:
            results["app_loads"] = {"status": "FAIL", "details": f"Timeout: {str(e)}"}
            print(f"  FAIL: Timeout - {e}")
            results["screenshots"].append(save_screenshot(page, "01_timeout"))
            browser.close()
            return results
        except Exception as e:
            results["app_loads"] = {"status": "FAIL", "details": f"Error: {str(e)}"}
            print(f"  FAIL: {e}")
            results["screenshots"].append(save_screenshot(page, "01_error"))
            browser.close()
            return results

        # =========================================================
        # TEST 2: Map Displays with Ski Resort Markers
        # =========================================================
        print(f"\n[TEST 2] Map Displays with Ski Resort Markers")
        print("-" * 50)

        try:
            content = page.content()

            # Look for map-related elements
            map_selectors = [
                'iframe[title*="map" i]',
                'iframe[title*="folium" i]',
                'iframe[src*="map"]',
                '[data-testid="stIFrame"]',
                'iframe',
            ]

            map_iframe = None
            for selector in map_selectors:
                map_el = page.locator(selector)
                if map_el.count() > 0:
                    print(f"  Found map element with selector: {selector}")
                    map_iframe = map_el.first
                    break

            if map_iframe:
                # Check if iframe is visible
                is_visible = map_iframe.is_visible()
                print(f"  Map iframe visible: {is_visible}")

                # Get iframe dimensions
                try:
                    bbox = map_iframe.bounding_box()
                    if bbox:
                        print(f"  Map dimensions: {bbox['width']}x{bbox['height']}")
                except:
                    pass

                # Try to access iframe content for markers
                try:
                    # Get all frames
                    frames = page.frames
                    print(f"  Found {len(frames)} frames")

                    markers_found = False
                    for frame in frames:
                        frame_content = frame.content()
                        # Look for marker indicators in Folium/Leaflet
                        marker_indicators = ['marker', 'leaflet', 'CircleMarker', 'L.marker', 'data-lat', 'data-lng']
                        for indicator in marker_indicators:
                            if indicator.lower() in frame_content.lower():
                                markers_found = True
                                print(f"  Found marker indicator: {indicator}")
                                break
                        if markers_found:
                            break

                    if markers_found:
                        results["map_displays"] = {"status": "PASS", "details": "Map iframe visible with marker indicators"}
                        print(f"  PASS: Map displays with markers")
                    else:
                        results["map_displays"] = {"status": "PARTIAL", "details": "Map iframe visible but no marker indicators found"}
                        print(f"  PARTIAL: Map visible but markers not confirmed")

                except Exception as e:
                    results["map_displays"] = {"status": "PARTIAL", "details": f"Map visible but couldn't inspect iframe: {e}"}
                    print(f"  PARTIAL: Map visible, iframe inspection failed")
            else:
                # Check for pydeck or other map types
                pydeck = page.locator('[class*="deck"]')
                if pydeck.count() > 0:
                    results["map_displays"] = {"status": "PASS", "details": "PyDeck map found"}
                    print(f"  PASS: PyDeck map found")
                else:
                    results["map_displays"] = {"status": "FAIL", "details": "No map element found"}
                    print(f"  FAIL: No map found")

            results["screenshots"].append(save_screenshot(page, "02_map"))

        except Exception as e:
            results["map_displays"] = {"status": "FAIL", "details": f"Error: {str(e)}"}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 3: Snow Depth Values Shown (Not Zero)
        # =========================================================
        print(f"\n[TEST 3] Snow Depth Values Shown (Not Zero)")
        print("-" * 50)

        try:
            content = page.content()

            # Look for snow depth patterns
            import re

            # Common patterns for snow depth values
            snow_patterns = [
                r'(\d+)\s*(?:cm|inches|in|")',  # Number + unit
                r'Snow\s*(?:Depth|depth)[:=]\s*(\d+)',  # "Snow Depth: 123"
                r'(\d{2,3})\s*cm',  # Two-three digit + cm
                r'depth[:\s]+(\d+)',  # depth: 123
            ]

            values_found = []
            for pattern in snow_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        val = int(match) if isinstance(match, str) else int(match[0])
                        if val > 0:  # Only count non-zero values
                            values_found.append(val)
                    except:
                        pass

            # Also look in all frames
            for frame in page.frames:
                try:
                    frame_content = frame.content()
                    for pattern in snow_patterns:
                        matches = re.findall(pattern, frame_content, re.IGNORECASE)
                        for match in matches:
                            try:
                                val = int(match) if isinstance(match, str) else int(match[0])
                                if val > 0:
                                    values_found.append(val)
                            except:
                                pass
                except:
                    pass

            # Remove duplicates and filter reasonable values
            values_found = list(set([v for v in values_found if 0 < v < 5000]))  # Reasonable snow depth range in cm

            print(f"  Snow depth values found: {values_found[:10]}...")  # Show first 10

            if values_found:
                # Check if any are actually snow depths (reasonable range)
                reasonable_values = [v for v in values_found if 10 < v < 1000]  # More specific range
                if reasonable_values:
                    results["snow_depth_values"] = {
                        "status": "PASS",
                        "details": f"Found {len(values_found)} snow depth values",
                        "values_found": sorted(values_found)[:20]  # Store first 20
                    }
                    print(f"  PASS: Found snow depth values: {sorted(reasonable_values)[:5]}")
                else:
                    results["snow_depth_values"] = {
                        "status": "PARTIAL",
                        "details": f"Found values but none in typical snow depth range",
                        "values_found": values_found[:10]
                    }
                    print(f"  PARTIAL: Values found but uncertain if snow depths")
            else:
                # Look for specific text patterns
                has_snow_text = "snow" in content.lower() and ("cm" in content.lower() or "inch" in content.lower())
                if has_snow_text:
                    results["snow_depth_values"] = {
                        "status": "PARTIAL",
                        "details": "Snow-related text found but couldn't extract numeric values",
                        "values_found": []
                    }
                    print(f"  PARTIAL: Snow text found but no numeric values extracted")
                else:
                    results["snow_depth_values"] = {
                        "status": "FAIL",
                        "details": "No snow depth values found",
                        "values_found": []
                    }
                    print(f"  FAIL: No snow depth values found")

            results["screenshots"].append(save_screenshot(page, "03_snow_values"))

        except Exception as e:
            results["snow_depth_values"] = {"status": "FAIL", "details": f"Error: {str(e)}", "values_found": []}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 4: Detail Panel Shows Forecast Data
        # =========================================================
        print(f"\n[TEST 4] Detail Panel Shows Forecast Data")
        print("-" * 50)

        try:
            content = page.content()

            # Check for "Loading forecast data..." which indicates incomplete load
            loading_text = "Loading forecast data"
            has_loading_text = loading_text.lower() in content.lower()

            if has_loading_text:
                print(f"  Warning: 'Loading forecast data...' text found")
                results["detail_panel_forecast"] = {
                    "status": "FAIL",
                    "details": "Detail panel still showing 'Loading forecast data...'"
                }
                print(f"  FAIL: Forecast still loading")
            else:
                # Look for forecast indicators
                forecast_indicators = [
                    "7-day forecast",
                    "forecast",
                    "24h snowfall",
                    "snowfall",
                    "temperature",
                    "prediction",
                    "expected",
                ]

                found_indicators = []
                for indicator in forecast_indicators:
                    if indicator.lower() in content.lower():
                        found_indicators.append(indicator)

                # Also check for data tables or charts
                has_chart = page.locator('[data-testid="stVegaLiteChart"], canvas, svg').count() > 0
                has_dataframe = page.locator('[data-testid="stDataFrame"], table').count() > 0

                print(f"  Found indicators: {found_indicators}")
                print(f"  Chart present: {has_chart}, DataFrame present: {has_dataframe}")

                if found_indicators or has_chart or has_dataframe:
                    results["detail_panel_forecast"] = {
                        "status": "PASS",
                        "details": f"Forecast data present. Indicators: {found_indicators}, Chart: {has_chart}, Table: {has_dataframe}"
                    }
                    print(f"  PASS: Forecast data is displayed")
                else:
                    results["detail_panel_forecast"] = {
                        "status": "PARTIAL",
                        "details": "No 'Loading' text but forecast indicators not found"
                    }
                    print(f"  PARTIAL: No loading text but couldn't confirm forecast data")

            results["screenshots"].append(save_screenshot(page, "04_detail_panel"))

        except Exception as e:
            results["detail_panel_forecast"] = {"status": "FAIL", "details": f"Error: {str(e)}"}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 5: Cache Status Badge Shows Valid Data
        # =========================================================
        print(f"\n[TEST 5] Cache Status Badge Shows Valid Data")
        print("-" * 50)

        try:
            content = page.content()

            # Check for invalid cache states
            invalid_states = ["unknown", "never"]
            has_invalid = False
            invalid_found = []

            for state in invalid_states:
                # Look for cache-related context with invalid states
                cache_patterns = [
                    f'cache.*{state}',
                    f'{state}.*cache',
                    f'last.*updated.*{state}',
                    f'data.*age.*{state}',
                ]
                for pattern in cache_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        has_invalid = True
                        invalid_found.append(state)

            # Look for cache status elements
            cache_elements = page.locator('text=/cache/i, text=/last updated/i, text=/data age/i, text=/fresh/i, text=/stale/i')
            cache_text = ""

            if cache_elements.count() > 0:
                for i in range(min(cache_elements.count(), 5)):
                    try:
                        text = cache_elements.nth(i).text_content()
                        if text:
                            cache_text += f" {text}"
                    except:
                        pass

            print(f"  Cache-related text found: {cache_text[:200] if cache_text else 'None'}")

            # Look for time-based indicators (minutes ago, hours ago)
            time_pattern = r'(\d+)\s*(minutes?|hours?|mins?|hrs?)\s*ago'
            time_matches = re.findall(time_pattern, content, re.IGNORECASE)

            if time_matches:
                print(f"  Time indicators found: {time_matches}")

            if has_invalid:
                results["cache_status"] = {
                    "status": "FAIL",
                    "details": f"Invalid cache states found: {invalid_found}"
                }
                print(f"  FAIL: Invalid cache states: {invalid_found}")
            elif time_matches or "fresh" in content.lower() or "updated" in content.lower():
                results["cache_status"] = {
                    "status": "PASS",
                    "details": f"Valid cache indicators found. Time matches: {time_matches}"
                }
                print(f"  PASS: Cache status shows valid data")
            elif cache_text:
                results["cache_status"] = {
                    "status": "PARTIAL",
                    "details": f"Cache elements found but status unclear: {cache_text[:100]}"
                }
                print(f"  PARTIAL: Cache elements found but status unclear")
            else:
                results["cache_status"] = {
                    "status": "PARTIAL",
                    "details": "No explicit cache status found on page"
                }
                print(f"  PARTIAL: No cache status badge found")

            results["screenshots"].append(save_screenshot(page, "05_cache_status"))

        except Exception as e:
            results["cache_status"] = {"status": "FAIL", "details": f"Error: {str(e)}"}
            print(f"  FAIL: {e}")

        # Take final full-page screenshot
        print(f"\n[FINAL] Taking full-page screenshot...")
        results["screenshots"].append(save_screenshot(page, "06_final_state"))

        # Cleanup
        browser.close()

    return results


def print_summary(results):
    """Print a summary of test results."""
    print("\n" + "=" * 70)
    print("DEPLOYMENT VERIFICATION RESULTS")
    print("=" * 70)

    status_symbols = {
        "PASS": "[PASS]",
        "PARTIAL": "[PARTIAL]",
        "FAIL": "[FAIL]",
        "not_tested": "[NOT TESTED]"
    }

    for test_name, result in results.items():
        if test_name == "screenshots":
            continue

        status = result.get("status", "unknown")
        details = result.get("details", "")
        values = result.get("values_found", [])

        symbol = status_symbols.get(status, f"[{status}]")

        print(f"\n{symbol} {test_name.replace('_', ' ').title()}")
        print(f"  Details: {details}")

        if values:
            print(f"  Values: {values[:10]}{'...' if len(values) > 10 else ''}")

    print("\n" + "-" * 70)
    print("SCREENSHOTS:")
    for ss in results.get("screenshots", []):
        print(f"  - {ss}")

    # Calculate summary
    tests = [k for k in results.keys() if k != "screenshots"]
    passed = len([k for k in tests if results[k].get("status") == "PASS"])
    partial = len([k for k in tests if results[k].get("status") == "PARTIAL"])
    failed = len([k for k in tests if results[k].get("status") == "FAIL"])
    total = len(tests)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed}/{total} PASS, {partial}/{total} PARTIAL, {failed}/{total} FAIL")

    if passed == total:
        print("ALL TESTS PASSED - Deployment is fully functional")
    elif failed == 0:
        print("NO CRITICAL FAILURES - Deployment is partially functional")
    else:
        print("FAILURES DETECTED - Deployment has issues")

    print("=" * 70)

    return passed, partial, failed, total


if __name__ == "__main__":
    results = run_cloud_deployment_test()
    print_summary(results)
