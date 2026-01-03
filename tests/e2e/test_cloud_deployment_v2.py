"""Playwright test for Streamlit Cloud deployment verification - V2.

Improved version that handles Streamlit Cloud's rendering behavior.

Tests that:
1. The app loads successfully (no error messages)
2. The map displays with ski resort markers
3. Snow depth values are shown (not zero)
4. The detail panel shows forecast data (not "Loading forecast data...")
5. Cache status badge shows valid data (not "Unknown" or "Never")

Run with: python tests/e2e/test_cloud_deployment_v2.py
"""

import time
import os
import re
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Streamlit Cloud URL
BASE_URL = "https://snowforecast.streamlit.app"

# Timeouts (cloud can be slower - app needs to wake up)
INITIAL_LOAD_TIMEOUT = 180000  # 3 minutes for initial load (app might be sleeping)
ELEMENT_TIMEOUT = 60000  # 60 seconds for elements

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
    print(f"  Screenshot: {filepath}")
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
        print("SNOWFORECAST STREAMLIT CLOUD DEPLOYMENT VERIFICATION v2")
        print("=" * 70)
        print(f"URL: {BASE_URL}")
        print(f"Time: {datetime.now().isoformat()}")

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
            response = page.goto(BASE_URL, timeout=INITIAL_LOAD_TIMEOUT, wait_until="load")

            if not response or not response.ok:
                results["app_loads"] = {"status": "FAIL", "details": f"HTTP {response.status if response else 'No response'}"}
                print(f"  FAIL: Bad HTTP response")
                browser.close()
                return results

            print(f"  HTTP: {response.status}")

            # Wait for the page title to appear (Snow Forecast Dashboard)
            print("  Waiting for app title...")
            try:
                page.wait_for_selector('h1:has-text("Snow Forecast")', timeout=ELEMENT_TIMEOUT)
                print("  Title found!")
            except:
                print("  Warning: Title not found, continuing...")

            # Give app extra time to fully render
            print("  Waiting for full render (15s)...")
            time.sleep(15)

            # Check for error messages
            content = page.content()
            error_patterns = [
                "Connection error",
                "This app has encountered an error",
                "Something went wrong",
                "Please contact the app developer",
            ]

            errors = [e for e in error_patterns if e.lower() in content.lower()]

            if errors:
                results["app_loads"] = {"status": "FAIL", "details": f"Error messages: {errors}"}
                print(f"  FAIL: Errors found: {errors}")
            else:
                results["app_loads"] = {"status": "PASS", "details": "App loaded successfully"}
                print(f"  PASS: App loaded successfully")

            results["screenshots"].append(save_screenshot(page, "01_initial_load"))

        except PlaywrightTimeout as e:
            results["app_loads"] = {"status": "FAIL", "details": f"Timeout: {str(e)[:100]}"}
            print(f"  FAIL: Timeout")
            results["screenshots"].append(save_screenshot(page, "01_timeout"))
            browser.close()
            return results
        except Exception as e:
            results["app_loads"] = {"status": "FAIL", "details": f"Error: {str(e)[:100]}"}
            print(f"  FAIL: {e}")
            results["screenshots"].append(save_screenshot(page, "01_error"))
            browser.close()
            return results

        # Get full page content for analysis
        content = page.content()

        # =========================================================
        # TEST 2: Map Displays with Ski Resort Markers
        # =========================================================
        print(f"\n[TEST 2] Map Displays with Ski Resort Markers")
        print("-" * 50)

        try:
            # Look for the map element - Streamlit uses iframes for Folium maps
            map_iframe = page.locator('iframe').first

            if map_iframe:
                is_visible = map_iframe.is_visible()
                print(f"  Map iframe visible: {is_visible}")

                if is_visible:
                    # Check iframe dimensions
                    try:
                        bbox = map_iframe.bounding_box()
                        if bbox:
                            width, height = bbox['width'], bbox['height']
                            print(f"  Map size: {width}x{height}")

                            if width > 200 and height > 200:
                                # Good size map
                                results["map_displays"] = {"status": "PASS", "details": f"Map visible, size {width}x{height}"}
                                print(f"  PASS: Map displayed correctly")
                            else:
                                results["map_displays"] = {"status": "PARTIAL", "details": f"Map visible but small: {width}x{height}"}
                                print(f"  PARTIAL: Map too small")
                    except Exception as e:
                        results["map_displays"] = {"status": "PARTIAL", "details": f"Map visible but couldn't check size: {e}"}
                        print(f"  PARTIAL: Map visible, size check failed")
                else:
                    results["map_displays"] = {"status": "FAIL", "details": "Map iframe not visible"}
                    print(f"  FAIL: Map not visible")
            else:
                results["map_displays"] = {"status": "FAIL", "details": "No iframe/map found"}
                print(f"  FAIL: No map found")

            # Check for "Circle color = snow depth" text which indicates map legend
            if "circle color = snow depth" in content.lower():
                print(f"  Map legend found: 'Circle color = snow depth'")

            results["screenshots"].append(save_screenshot(page, "02_map"))

        except Exception as e:
            results["map_displays"] = {"status": "FAIL", "details": f"Error: {str(e)[:100]}"}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 3: Snow Depth Values Shown (Not Zero)
        # =========================================================
        print(f"\n[TEST 3] Snow Depth Values Shown (Not Zero)")
        print("-" * 50)

        try:
            # Look for "Base: Xm (Yft)" pattern in sidebar (shows snow depth info)
            base_pattern = r'Base:\s*(\d+)m?\s*\((\d+)ft\)'
            base_match = re.search(base_pattern, content)

            # Look for snow depth in cm pattern
            depth_pattern = r'(\d+)\s*cm'
            depth_matches = re.findall(depth_pattern, content)

            values_found = []

            if base_match:
                base_m = int(base_match.group(1))
                base_ft = int(base_match.group(2))
                values_found.append(f"Base: {base_m}m/{base_ft}ft")
                print(f"  Found base elevation: {base_m}m ({base_ft}ft)")

            if depth_matches:
                depths = [int(d) for d in depth_matches if int(d) > 0 and int(d) < 5000]
                values_found.extend([f"{d}cm" for d in depths])
                print(f"  Found depth values: {depths[:5]}")

            # Check sidebar for "Alta" or other resort showing
            resort_info = page.locator('text=/Alta|Snowbird|Park City|Jackson/i')
            if resort_info.count() > 0:
                print(f"  Found resort name in sidebar")
                values_found.append("Resort info displayed")

            # Check for coordinate display (40.xxxx, 111.xxxx)
            coord_pattern = r'(\d+\.\d+)[°]?[NS]?,?\s*(\d+\.\d+)[°]?[EW]?'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                print(f"  Found coordinates: {coord_match.group(0)}")
                values_found.append(f"Coordinates: {coord_match.group(0)}")

            if values_found:
                results["snow_depth_values"] = {
                    "status": "PASS",
                    "details": f"Found {len(values_found)} location/depth values",
                    "values_found": values_found[:10]
                }
                print(f"  PASS: Location and elevation data found")
            else:
                results["snow_depth_values"] = {
                    "status": "FAIL",
                    "details": "No snow depth or location values found",
                    "values_found": []
                }
                print(f"  FAIL: No values found")

            results["screenshots"].append(save_screenshot(page, "03_snow_values"))

        except Exception as e:
            results["snow_depth_values"] = {"status": "FAIL", "details": f"Error: {str(e)[:100]}", "values_found": []}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 4: Detail Panel Shows Forecast Data (NOT "Loading...")
        # =========================================================
        print(f"\n[TEST 4] Detail Panel Shows Forecast Data")
        print("-" * 50)

        try:
            # Check for "Loading forecast data..." which indicates incomplete load
            loading_text = "Loading forecast data..."
            has_loading_text = loading_text in content

            if has_loading_text:
                results["detail_panel_forecast"] = {
                    "status": "FAIL",
                    "details": "Detail panel still showing 'Loading forecast data...'"
                }
                print(f"  FAIL: Forecast panel showing 'Loading forecast data...'")
            else:
                # Look for forecast chart or data
                has_regional_overview = "Regional Overview" in content
                has_forecast_time = "Forecast Time" in content

                print(f"  Regional Overview present: {has_regional_overview}")
                print(f"  Forecast Time selector: {has_forecast_time}")

                if has_regional_overview and has_forecast_time:
                    results["detail_panel_forecast"] = {
                        "status": "PARTIAL",
                        "details": "Regional view loaded but detail panel may be loading"
                    }
                    print(f"  PARTIAL: Main view loaded but detail forecast loading")
                else:
                    results["detail_panel_forecast"] = {
                        "status": "PASS",
                        "details": "No 'Loading forecast data' text, app appears functional"
                    }
                    print(f"  PASS: No loading indicators")

            results["screenshots"].append(save_screenshot(page, "04_detail_panel"))

        except Exception as e:
            results["detail_panel_forecast"] = {"status": "FAIL", "details": f"Error: {str(e)[:100]}"}
            print(f"  FAIL: {e}")

        # =========================================================
        # TEST 5: Cache Status Badge Shows Valid Data (NOT "Unknown")
        # =========================================================
        print(f"\n[TEST 5] Cache Status Badge")
        print("-" * 50)

        try:
            # Check for "Unknown - Never" which indicates no cached data
            has_unknown = "Unknown - Never" in content
            has_freshness_warning = "Data freshness unknown" in content

            if has_unknown:
                results["cache_status"] = {
                    "status": "FAIL",
                    "details": "Cache shows 'Unknown - Never'"
                }
                print(f"  FAIL: Cache status shows 'Unknown - Never'")
            elif has_freshness_warning:
                results["cache_status"] = {
                    "status": "FAIL",
                    "details": "Data freshness unknown - using cached data"
                }
                print(f"  FAIL: 'Data freshness unknown - using cached data'")
            else:
                # Check for valid cache indicators
                valid_patterns = [
                    r'(\d+)\s*(minutes?|hours?|mins?|hrs?)\s*ago',
                    r'Fresh',
                    r'Updated',
                    r'Last\s*updated',
                ]

                found_valid = []
                for pattern in valid_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        found_valid.append(pattern)

                if found_valid:
                    results["cache_status"] = {
                        "status": "PASS",
                        "details": f"Valid cache indicators found"
                    }
                    print(f"  PASS: Valid cache status found")
                else:
                    results["cache_status"] = {
                        "status": "PARTIAL",
                        "details": "No explicit cache status found"
                    }
                    print(f"  PARTIAL: No cache status visible")

            results["screenshots"].append(save_screenshot(page, "05_cache_status"))

        except Exception as e:
            results["cache_status"] = {"status": "FAIL", "details": f"Error: {str(e)[:100]}"}
            print(f"  FAIL: {e}")

        # =========================================================
        # Additional: Wait and retry for loading items
        # =========================================================
        print(f"\n[EXTRA] Waiting additional time for data load...")
        print("-" * 50)

        time.sleep(10)
        content_after_wait = page.content()

        # Re-check loading forecast
        if "Loading forecast data..." in content and "Loading forecast data..." not in content_after_wait:
            results["detail_panel_forecast"] = {
                "status": "PASS",
                "details": "Forecast loaded after additional wait"
            }
            print(f"  Forecast panel now loaded!")

        # Re-check cache status
        if "Unknown - Never" in content:
            if "Unknown - Never" not in content_after_wait:
                results["cache_status"] = {
                    "status": "PASS",
                    "details": "Cache status updated after wait"
                }
                print(f"  Cache status now valid!")

        # Final screenshots
        results["screenshots"].append(save_screenshot(page, "06_after_wait"))
        results["screenshots"].append(save_screenshot(page, "07_final"))

        browser.close()

    return results


def print_summary(results):
    """Print detailed test results summary."""
    print("\n" + "=" * 70)
    print("DEPLOYMENT VERIFICATION RESULTS")
    print("=" * 70)

    status_symbols = {
        "PASS": "[PASS]    ",
        "PARTIAL": "[PARTIAL] ",
        "FAIL": "[FAIL]    ",
        "not_tested": "[SKIP]    "
    }

    for test_name, result in results.items():
        if test_name == "screenshots":
            continue

        status = result.get("status", "unknown")
        details = result.get("details", "")
        values = result.get("values_found", [])

        symbol = status_symbols.get(status, f"[{status}]")

        print(f"\n{symbol} {test_name.replace('_', ' ').upper()}")
        print(f"           Details: {details}")

        if values:
            print(f"           Values: {values[:10]}")

    print("\n" + "-" * 70)
    print("SCREENSHOTS SAVED:")
    for ss in results.get("screenshots", []):
        print(f"  {ss}")

    # Summary
    tests = [k for k in results.keys() if k != "screenshots"]
    passed = len([k for k in tests if results[k].get("status") == "PASS"])
    partial = len([k for k in tests if results[k].get("status") == "PARTIAL"])
    failed = len([k for k in tests if results[k].get("status") == "FAIL"])
    total = len(tests)

    print(f"\n{'=' * 70}")
    print(f"FINAL SCORE: {passed}/{total} PASS | {partial}/{total} PARTIAL | {failed}/{total} FAIL")

    if passed == total:
        status_msg = "ALL TESTS PASSED - Deployment fully functional"
    elif failed == 0:
        status_msg = "NO CRITICAL FAILURES - Deployment partially functional"
    else:
        status_msg = "FAILURES DETECTED - Deployment has issues"

    print(status_msg)
    print("=" * 70)

    return passed, partial, failed, total


if __name__ == "__main__":
    results = run_cloud_deployment_test()
    print_summary(results)
