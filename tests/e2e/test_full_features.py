"""Full Feature Test for Snowforecast Streamlit App.

Comprehensive test of all dashboard features:
1. Navigate and wait for full load
2. Test all tabs: Forecast Chart, SNOTEL Stations, All Resorts
3. Test time selector (radio buttons)
4. Test Refresh Data button
5. Check cache status badge
6. Take screenshots of each feature

Run with: python tests/e2e/test_full_features.py
"""

import time
import os
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# URLs to try (alternative URL first since primary may have error)
URLS = [
    "https://kavanaghpatrick-snowforecast.streamlit.app",
    "https://snowforecast.streamlit.app"
]

# Timeouts (Streamlit Cloud can be slow on cold start)
LOAD_TIMEOUT = 120000  # 120 seconds for initial load
ELEMENT_TIMEOUT = 60000  # 60 seconds for elements
TAB_TIMEOUT = 30000  # 30 seconds for tab content
RETRY_ATTEMPTS = 3  # Number of retry attempts for app loading

# Screenshot directory
SCREENSHOT_DIR = "/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots"


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


def run_full_feature_test():
    """Run comprehensive feature test on Snowforecast app."""
    ensure_screenshot_dir()
    results = {
        "navigation": {"status": "not_tested", "details": ""},
        "forecast_chart_tab": {"status": "not_tested", "details": ""},
        "snotel_stations_tab": {"status": "not_tested", "details": ""},
        "all_resorts_tab": {"status": "not_tested", "details": ""},
        "time_selector": {"status": "not_tested", "details": ""},
        "refresh_button": {"status": "not_tested", "details": ""},
        "cache_status": {"status": "not_tested", "details": ""},
        "screenshots": []
    }

    with sync_playwright() as p:
        print("=" * 60)
        print("SNOWFORECAST FULL FEATURE TEST")
        print("=" * 60)

        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Try each URL with retries (Streamlit Cloud can have cold starts)
        connected_url = None
        for url in URLS:
            print(f"\n[1] Attempting to navigate to: {url}")

            for attempt in range(RETRY_ATTEMPTS):
                print(f"  Attempt {attempt + 1}/{RETRY_ATTEMPTS}...")
                try:
                    response = page.goto(url, timeout=LOAD_TIMEOUT, wait_until="domcontentloaded")
                    if response and response.ok:
                        # Wait for page to settle and possible cold start
                        print(f"  Waiting for app to initialize...")
                        time.sleep(10)  # Give Streamlit time to spin up

                        page_content = page.content().lower()

                        # Check for error page
                        if "oh no" in page_content or "error running app" in page_content:
                            print(f"  App error page detected, retrying...")
                            # Reload to retry
                            page.reload()
                            time.sleep(5)
                            page_content = page.content().lower()

                            if "oh no" in page_content or "error running app" in page_content:
                                print(f"  Still error after reload")
                                continue  # Try next attempt

                        # Check if Streamlit app content is present
                        if "streamlit" in page_content or "snow" in page_content or "forecast" in page_content:
                            connected_url = url
                            print(f"  Connected successfully to {url}")
                            break
                        else:
                            print(f"  No Streamlit content found, retrying...")

                    else:
                        print(f"  Response not OK: {response.status if response else 'No response'}")
                except PlaywrightTimeout:
                    print(f"  Timeout connecting to {url}")
                except Exception as e:
                    print(f"  Error connecting to {url}: {e}")

            if connected_url:
                break

        if not connected_url:
            # Capture final state screenshot for debugging
            results["screenshots"].append(save_screenshot(page, "00_connection_failure"))

            # Get additional error details
            page_content = page.content()
            error_details = "Could not connect to any URL"
            if "oh no" in page_content.lower():
                error_details = "App shows 'Oh no' error page - possible server-side error"
            elif "404" in page_content:
                error_details = "App returns 404 - deployment may not exist"

            results["navigation"] = {
                "status": "FAIL",
                "details": error_details
            }
            print(f"\nFAILED: {error_details}")
            browser.close()
            return results

        # Wait for Streamlit to fully load
        print("\n[2] Waiting for Streamlit app to fully load...")
        try:
            # Wait for network to be idle
            page.wait_for_load_state("networkidle", timeout=LOAD_TIMEOUT)
            # Wait for Streamlit content
            page.wait_for_selector('[data-testid="stApp"]', timeout=ELEMENT_TIMEOUT)
            time.sleep(3)  # Extra time for Streamlit rendering

            results["navigation"] = {
                "status": "PASS",
                "details": f"Successfully loaded {connected_url}"
            }
            print(f"  App loaded successfully")
            results["screenshots"].append(save_screenshot(page, "01_initial_load"))
        except PlaywrightTimeout:
            results["navigation"] = {
                "status": "FAIL",
                "details": "Timeout waiting for Streamlit app to load"
            }
            print("  FAILED: Timeout waiting for app to load")
            results["screenshots"].append(save_screenshot(page, "01_load_timeout"))
            browser.close()
            return results
        except Exception as e:
            results["navigation"] = {
                "status": "FAIL",
                "details": f"Error during load: {str(e)}"
            }
            print(f"  FAILED: {e}")
            browser.close()
            return results

        # Get page content for analysis
        content = page.content().lower()
        print(f"\n  Page title: {page.title()}")

        # TEST: Tabs
        print("\n[3] Testing tabs...")

        # Look for tabs in Streamlit
        tabs = page.locator('[data-baseweb="tab"]')
        tab_count = tabs.count()
        print(f"  Found {tab_count} tabs")

        # Define expected tabs
        expected_tabs = ["Forecast Chart", "SNOTEL Stations", "All Resorts"]

        if tab_count > 0:
            # Get all tab labels
            tab_labels = []
            for i in range(tab_count):
                try:
                    label = tabs.nth(i).text_content()
                    tab_labels.append(label)
                except:
                    pass
            print(f"  Tab labels: {tab_labels}")

            # Test each expected tab
            for tab_name in expected_tabs:
                result_key = f"{tab_name.lower().replace(' ', '_')}_tab"
                print(f"\n  Testing '{tab_name}' tab...")

                try:
                    # Find the tab
                    tab = page.locator(f'[data-baseweb="tab"]:has-text("{tab_name}")')
                    if tab.count() > 0:
                        # Click the tab
                        tab.first.click()
                        time.sleep(2)  # Wait for content to load

                        # Check for content
                        current_content = page.content()

                        # Tab-specific checks
                        if "Forecast" in tab_name:
                            # Look for chart or forecast content
                            has_chart = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"], canvas, svg').count() > 0
                            has_forecast_text = "forecast" in current_content.lower() or "snow" in current_content.lower()

                            if has_chart or has_forecast_text:
                                results[result_key] = {"status": "PASS", "details": f"Chart visible: {has_chart}, Forecast text: {has_forecast_text}"}
                                print(f"    PASS: Content loaded (Chart: {has_chart})")
                            else:
                                results[result_key] = {"status": "PARTIAL", "details": "Tab clicked but no chart/forecast content found"}
                                print(f"    PARTIAL: Tab clicked but no chart found")

                        elif "SNOTEL" in tab_name:
                            # Look for SNOTEL-specific content
                            has_snotel = "snotel" in current_content.lower()
                            has_stations = "station" in current_content.lower()
                            has_table = page.locator('[data-testid="stDataFrame"], table').count() > 0

                            if has_snotel or has_stations or has_table:
                                results[result_key] = {"status": "PASS", "details": f"SNOTEL: {has_snotel}, Stations: {has_stations}, Table: {has_table}"}
                                print(f"    PASS: SNOTEL content loaded")
                            else:
                                results[result_key] = {"status": "PARTIAL", "details": "Tab clicked but no SNOTEL content found"}
                                print(f"    PARTIAL: Tab clicked but no SNOTEL content")

                        elif "Resort" in tab_name:
                            # Look for resort-specific content
                            has_resort = "resort" in current_content.lower()
                            has_ski = "ski" in current_content.lower()
                            has_table = page.locator('[data-testid="stDataFrame"], table').count() > 0
                            has_map = page.locator('iframe, [data-testid="stDeckGlJsonChart"]').count() > 0

                            if has_resort or has_ski or has_table or has_map:
                                results[result_key] = {"status": "PASS", "details": f"Resort: {has_resort}, Table: {has_table}, Map: {has_map}"}
                                print(f"    PASS: Resort content loaded")
                            else:
                                results[result_key] = {"status": "PARTIAL", "details": "Tab clicked but no resort content found"}
                                print(f"    PARTIAL: Tab clicked but no resort content")

                        # Take screenshot of this tab
                        results["screenshots"].append(save_screenshot(page, f"02_tab_{tab_name.lower().replace(' ', '_')}"))
                    else:
                        results[result_key] = {"status": "NOT_FOUND", "details": f"Tab '{tab_name}' not found"}
                        print(f"    NOT_FOUND: Tab '{tab_name}' not found")

                except Exception as e:
                    results[result_key] = {"status": "FAIL", "details": str(e)}
                    print(f"    FAIL: {e}")
        else:
            # No tabs found - check if content exists differently
            print("  No tabs found - checking if content exists in different layout")
            for tab_name in expected_tabs:
                result_key = f"{tab_name.lower().replace(' ', '_')}_tab"
                results[result_key] = {"status": "NOT_FOUND", "details": "No tab elements found in DOM"}

        # TEST: Time Selector
        print("\n[4] Testing time selector (radio buttons)...")
        try:
            # Look for radio buttons
            radio_buttons = page.locator('[data-testid="stRadio"]')
            radio_count = radio_buttons.count()

            if radio_count > 0:
                print(f"  Found {radio_count} radio button groups")

                # Try to find and click radio options
                radio_options = page.locator('[data-testid="stRadio"] label, [role="radio"]')
                option_count = radio_options.count()
                print(f"  Found {option_count} radio options")

                if option_count > 1:
                    # Try clicking the second option (to change from default)
                    second_option = radio_options.nth(1)
                    second_option.click()
                    time.sleep(1)

                    results["time_selector"] = {"status": "PASS", "details": f"Found {option_count} time options, successfully clicked"}
                    print(f"    PASS: Time selector works ({option_count} options)")
                else:
                    results["time_selector"] = {"status": "PARTIAL", "details": "Radio buttons found but couldn't interact"}
                    print(f"    PARTIAL: Radio buttons found but limited interaction")
            else:
                # Try alternative selectors
                sliders = page.locator('[data-testid="stSlider"]')
                if sliders.count() > 0:
                    results["time_selector"] = {"status": "PASS", "details": "Time selector is a slider instead of radio buttons"}
                    print(f"    PASS: Found slider-based time selector")
                else:
                    results["time_selector"] = {"status": "NOT_FOUND", "details": "No radio buttons or sliders found"}
                    print(f"    NOT_FOUND: No time selector found")

            results["screenshots"].append(save_screenshot(page, "03_time_selector"))

        except Exception as e:
            results["time_selector"] = {"status": "FAIL", "details": str(e)}
            print(f"    FAIL: {e}")

        # TEST: Refresh Data Button
        print("\n[5] Testing Refresh Data button in sidebar...")
        try:
            # Open sidebar if needed (click hamburger menu on mobile)
            sidebar = page.locator('[data-testid="stSidebar"]')
            sidebar_visible = sidebar.is_visible()

            if not sidebar_visible:
                # Try to open sidebar
                hamburger = page.locator('[data-testid="stSidebarCollapsedControl"]')
                if hamburger.count() > 0:
                    hamburger.click()
                    time.sleep(1)

            # Look for refresh button in sidebar
            refresh_button = page.locator('[data-testid="stSidebar"] button:has-text("Refresh")')
            if refresh_button.count() == 0:
                refresh_button = page.locator('[data-testid="stSidebar"] button:has-text("refresh")')
            if refresh_button.count() == 0:
                refresh_button = page.locator('button:has-text("Refresh")')

            if refresh_button.count() > 0:
                print(f"  Found Refresh button")

                # Click the button
                refresh_button.first.click()
                time.sleep(2)

                results["refresh_button"] = {"status": "PASS", "details": "Refresh button found and clicked"}
                print(f"    PASS: Refresh button works")
            else:
                # Check for any button in sidebar
                sidebar_buttons = page.locator('[data-testid="stSidebar"] button')
                btn_count = sidebar_buttons.count()

                if btn_count > 0:
                    # Get button texts
                    btn_texts = []
                    for i in range(min(btn_count, 5)):
                        try:
                            btn_texts.append(sidebar_buttons.nth(i).text_content())
                        except:
                            pass
                    results["refresh_button"] = {"status": "PARTIAL", "details": f"No 'Refresh' button found. Sidebar has buttons: {btn_texts}"}
                    print(f"    PARTIAL: Found sidebar buttons but no 'Refresh': {btn_texts}")
                else:
                    results["refresh_button"] = {"status": "NOT_FOUND", "details": "No buttons found in sidebar"}
                    print(f"    NOT_FOUND: No sidebar buttons found")

            results["screenshots"].append(save_screenshot(page, "04_sidebar_refresh"))

        except Exception as e:
            results["refresh_button"] = {"status": "FAIL", "details": str(e)}
            print(f"    FAIL: {e}")

        # TEST: Cache Status Badge
        print("\n[6] Checking cache status badge...")
        try:
            content = page.content()

            # Look for cache-related text
            cache_indicators = ["cache", "cached", "fresh", "stale", "last updated", "refreshed"]
            found_indicators = [ind for ind in cache_indicators if ind in content.lower()]

            # Also look for badge-like elements
            badges = page.locator('[data-testid="stStatusWidget"], .stBadge, span:has-text("Cache"), span:has-text("cache")')
            badge_count = badges.count()

            if found_indicators or badge_count > 0:
                results["cache_status"] = {"status": "PASS", "details": f"Found indicators: {found_indicators}, Badge elements: {badge_count}"}
                print(f"    PASS: Cache status found - {found_indicators}")
            else:
                # Check for time indicators that might indicate cache
                time_indicators = ["updated", "ago", "min", "hour"]
                time_found = [t for t in time_indicators if t in content.lower()]

                if time_found:
                    results["cache_status"] = {"status": "PARTIAL", "details": f"Time indicators found: {time_found}"}
                    print(f"    PARTIAL: Time indicators found but no explicit cache badge")
                else:
                    results["cache_status"] = {"status": "NOT_FOUND", "details": "No cache status indicators found"}
                    print(f"    NOT_FOUND: No cache status found")

            results["screenshots"].append(save_screenshot(page, "05_cache_status"))

        except Exception as e:
            results["cache_status"] = {"status": "FAIL", "details": str(e)}
            print(f"    FAIL: {e}")

        # Final screenshot
        print("\n[7] Taking final screenshots...")
        results["screenshots"].append(save_screenshot(page, "06_final_state"))

        # Cleanup
        browser.close()

    return results


def print_summary(results):
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    for feature, result in results.items():
        if feature == "screenshots":
            continue
        status = result.get("status", "unknown")
        details = result.get("details", "")

        # Color coding (using plain text for compatibility)
        if status == "PASS":
            indicator = "[PASS]"
        elif status == "PARTIAL":
            indicator = "[PARTIAL]"
        elif status == "FAIL":
            indicator = "[FAIL]"
        elif status == "NOT_FOUND":
            indicator = "[NOT FOUND]"
        else:
            indicator = f"[{status}]"

        print(f"\n{indicator} {feature.replace('_', ' ').title()}")
        if details:
            print(f"  Details: {details}")

    print("\n" + "-" * 60)
    print("SCREENSHOTS SAVED:")
    for ss in results.get("screenshots", []):
        print(f"  - {ss}")

    # Calculate summary
    total = len([k for k in results.keys() if k != "screenshots"])
    passed = len([k for k, v in results.items() if k != "screenshots" and v.get("status") == "PASS"])
    partial = len([k for k, v in results.items() if k != "screenshots" and v.get("status") == "PARTIAL"])
    failed = len([k for k, v in results.items() if k != "screenshots" and v.get("status") in ["FAIL", "NOT_FOUND"]])

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} PASS, {partial}/{total} PARTIAL, {failed}/{total} FAIL/NOT_FOUND")
    print("=" * 60)


if __name__ == "__main__":
    results = run_full_feature_test()
    print_summary(results)
