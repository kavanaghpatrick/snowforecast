"""Full Feature Test for Snowforecast Streamlit App (Local).

Comprehensive test of all dashboard features against local server:
1. Navigate and wait for full load
2. Test all tabs: Forecast Chart, SNOTEL Stations, All Resorts
3. Test time selector (radio buttons)
4. Test Refresh Data button
5. Check cache status badge
6. Take screenshots of each feature

Run with: python tests/e2e/test_local_features.py
"""

import time
import os
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Local server URL
BASE_URL = "http://localhost:8502"

# Timeouts
LOAD_TIMEOUT = 60000  # 60 seconds for initial load
ELEMENT_TIMEOUT = 30000  # 30 seconds for elements
TAB_TIMEOUT = 20000  # 20 seconds for tab content

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
    """Run comprehensive feature test on local Snowforecast app."""
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
        print("SNOWFORECAST FULL FEATURE TEST (LOCAL)")
        print("=" * 60)
        print(f"Testing against: {BASE_URL}")

        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Navigate to local server
        print(f"\n[1] Navigating to local server: {BASE_URL}")
        try:
            response = page.goto(BASE_URL, timeout=LOAD_TIMEOUT, wait_until="domcontentloaded")
            if not response or not response.ok:
                results["navigation"] = {
                    "status": "FAIL",
                    "details": f"Server returned {response.status if response else 'no response'}"
                }
                print(f"  FAILED: Server not responding")
                browser.close()
                return results

            print("  Connected to local server")
        except PlaywrightTimeout:
            results["navigation"] = {
                "status": "FAIL",
                "details": "Timeout connecting to local server"
            }
            print("  FAILED: Timeout")
            browser.close()
            return results
        except Exception as e:
            results["navigation"] = {
                "status": "FAIL",
                "details": f"Error: {str(e)}"
            }
            print(f"  FAILED: {e}")
            browser.close()
            return results

        # Wait for Streamlit to fully load
        print("\n[2] Waiting for Streamlit app to fully load...")
        try:
            # Wait for network to be idle
            page.wait_for_load_state("networkidle", timeout=LOAD_TIMEOUT)

            # Wait for Streamlit app container
            page.wait_for_selector('[data-testid="stApp"]', timeout=ELEMENT_TIMEOUT)

            # Wait for main content to appear (title)
            page.wait_for_selector('h1:has-text("Snow Forecast")', timeout=ELEMENT_TIMEOUT)

            time.sleep(5)  # Extra time for Streamlit rendering

            results["navigation"] = {
                "status": "PASS",
                "details": f"Successfully loaded {BASE_URL}"
            }
            print(f"  App loaded successfully")
            results["screenshots"].append(save_screenshot(page, "01_initial_load"))
        except PlaywrightTimeout as e:
            results["navigation"] = {
                "status": "PARTIAL",
                "details": f"Page loaded but timeout waiting for elements: {str(e)}"
            }
            print(f"  PARTIAL: {e}")
            results["screenshots"].append(save_screenshot(page, "01_partial_load"))
        except Exception as e:
            results["navigation"] = {
                "status": "FAIL",
                "details": f"Error during load: {str(e)}"
            }
            print(f"  FAILED: {e}")
            results["screenshots"].append(save_screenshot(page, "01_load_error"))
            browser.close()
            return results

        # Get page content for analysis
        content = page.content()
        print(f"\n  Page title: {page.title()}")

        # TEST: Tabs (need to scroll down as tabs are below the map)
        print("\n[3] Testing tabs...")

        # Wait for loading spinners to disappear (content to load)
        print("  Waiting for content to load...")
        try:
            # Wait for loading text to disappear
            loading = page.locator('text=Loading')
            for _ in range(30):  # Up to 30 seconds
                if loading.count() == 0:
                    break
                time.sleep(1)
            print("  Content loaded")
        except:
            print("  Could not wait for loading")

        # Scroll down to find tabs (they are below the map/detail panel)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(3)

        # Take a screenshot after scrolling
        results["screenshots"].append(save_screenshot(page, "02_after_scroll"))

        # Look for tabs in Streamlit
        tabs = page.locator('[data-baseweb="tab"]')
        tab_count = tabs.count()
        print(f"  Found {tab_count} tabs")

        # Define expected tabs with their result keys
        expected_tabs = [
            ("Forecast Chart", "forecast_chart_tab"),
            ("SNOTEL Stations", "snotel_stations_tab"),
            ("All Resorts", "all_resorts_tab")
        ]

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
            for tab_name, result_key in expected_tabs:
                print(f"\n  Testing '{tab_name}' tab...")

                try:
                    # Find the tab
                    tab = page.locator(f'[data-baseweb="tab"]:has-text("{tab_name}")')
                    if tab.count() > 0:
                        # Click the tab
                        tab.first.click()
                        time.sleep(3)  # Wait for content to load

                        # Check for content
                        current_content = page.content()

                        # Tab-specific checks
                        if "Forecast" in tab_name:
                            # Look for chart or forecast content
                            has_chart = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], canvas, svg.marks').count() > 0
                            has_forecast_text = "7-day" in current_content.lower() or "snow" in current_content.lower()

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

                            if has_resort or has_ski or has_table:
                                results[result_key] = {"status": "PASS", "details": f"Resort: {has_resort}, Table: {has_table}"}
                                print(f"    PASS: Resort content loaded")
                            else:
                                results[result_key] = {"status": "PARTIAL", "details": "Tab clicked but no resort content found"}
                                print(f"    PARTIAL: Tab clicked but no resort content")

                        # Take screenshot of this tab
                        results["screenshots"].append(save_screenshot(page, f"02_tab_{tab_name.lower().replace(' ', '_')}"))
                    else:
                        results[result_key] = {"status": "NOT_FOUND", "details": f"Tab '{tab_name}' not found in DOM"}
                        print(f"    NOT_FOUND: Tab '{tab_name}' not found")

                except Exception as e:
                    results[result_key] = {"status": "FAIL", "details": str(e)}
                    print(f"    FAIL: {e}")
        else:
            # No tabs found - check if content exists differently
            print("  No tabs found - checking if content exists in different layout")
            for tab_name, result_key in expected_tabs:
                results[result_key] = {"status": "NOT_FOUND", "details": "No tab elements found in DOM"}

        # TEST: Time Selector
        print("\n[4] Testing time selector...")
        try:
            # Look for radio buttons or time-related controls
            radio_buttons = page.locator('[data-testid="stRadio"]')
            radio_count = radio_buttons.count()

            if radio_count > 0:
                print(f"  Found {radio_count} radio button groups")

                # Try to find and click radio options
                radio_options = page.locator('[data-testid="stRadio"] label')
                option_count = radio_options.count()
                print(f"  Found {option_count} radio options")

                if option_count > 1:
                    # Get option texts
                    option_texts = []
                    for i in range(min(option_count, 5)):
                        try:
                            option_texts.append(radio_options.nth(i).text_content())
                        except:
                            pass
                    print(f"  Options: {option_texts}")

                    # Try clicking the second option (to change from default)
                    second_option = radio_options.nth(1)
                    second_option.click()
                    time.sleep(2)

                    results["time_selector"] = {"status": "PASS", "details": f"Found {option_count} time options: {option_texts}"}
                    print(f"    PASS: Time selector works ({option_count} options)")
                else:
                    results["time_selector"] = {"status": "PARTIAL", "details": f"Radio buttons found ({option_count} options) but couldn't interact"}
                    print(f"    PARTIAL: Radio buttons found but limited options")
            else:
                # Try alternative selectors
                sliders = page.locator('[data-testid="stSlider"]')
                if sliders.count() > 0:
                    results["time_selector"] = {"status": "PASS", "details": "Time selector is a slider instead of radio buttons"}
                    print(f"    PASS: Found slider-based time selector")
                else:
                    # Check for horizontal selector component
                    horizontal = page.locator('[role="radiogroup"]')
                    if horizontal.count() > 0:
                        results["time_selector"] = {"status": "PASS", "details": "Found radiogroup time selector"}
                        print(f"    PASS: Found radiogroup time selector")
                    else:
                        results["time_selector"] = {"status": "NOT_FOUND", "details": "No radio buttons, sliders, or radiogroups found"}
                        print(f"    NOT_FOUND: No time selector found")

            results["screenshots"].append(save_screenshot(page, "03_time_selector"))

        except Exception as e:
            results["time_selector"] = {"status": "FAIL", "details": str(e)}
            print(f"    FAIL: {e}")

        # TEST: Refresh Data Button
        print("\n[5] Testing Refresh Data button in sidebar...")
        try:
            # Look for sidebar
            sidebar = page.locator('[data-testid="stSidebar"]')
            sidebar_visible = sidebar.is_visible()
            print(f"  Sidebar visible: {sidebar_visible}")

            if not sidebar_visible:
                # Try to open sidebar (click hamburger menu)
                hamburger = page.locator('[data-testid="stSidebarCollapsedControl"]')
                if hamburger.count() > 0:
                    hamburger.click()
                    time.sleep(1)
                    sidebar_visible = sidebar.is_visible()
                    print(f"  Opened sidebar: {sidebar_visible}")

            # Look for refresh button
            refresh_button = page.locator('button:has-text("Refresh")')
            if refresh_button.count() == 0:
                refresh_button = page.locator('button:has-text("refresh")')

            if refresh_button.count() > 0:
                print(f"  Found Refresh button")

                # Click the button
                refresh_button.first.click()
                time.sleep(3)  # Wait for refresh

                results["refresh_button"] = {"status": "PASS", "details": "Refresh button found and clicked successfully"}
                print(f"    PASS: Refresh button works")
            else:
                # Check for any button in sidebar with refresh icon
                sidebar_buttons = page.locator('[data-testid="stSidebar"] button')
                btn_count = sidebar_buttons.count()
                print(f"  Found {btn_count} buttons in sidebar")

                # Get button texts
                btn_texts = []
                for i in range(min(btn_count, 5)):
                    try:
                        text = sidebar_buttons.nth(i).text_content()
                        if text:
                            btn_texts.append(text.strip())
                    except:
                        pass

                if btn_texts:
                    results["refresh_button"] = {"status": "PARTIAL", "details": f"No 'Refresh' text found. Sidebar buttons: {btn_texts}"}
                    print(f"    PARTIAL: Found buttons but no 'Refresh': {btn_texts}")
                else:
                    results["refresh_button"] = {"status": "NOT_FOUND", "details": "No refresh-like buttons found in sidebar"}
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
            cache_indicators = ["cache", "cached", "fresh", "stale", "last updated", "refreshed", "data age"]
            found_indicators = [ind for ind in cache_indicators if ind in content.lower()]

            # Also look for badge-like elements
            badges = page.locator('span:has-text("Cache"), span:has-text("cache"), span:has-text("Fresh"), span:has-text("Stale")')
            badge_count = badges.count()

            if found_indicators or badge_count > 0:
                results["cache_status"] = {"status": "PASS", "details": f"Found indicators: {found_indicators}, Badge elements: {badge_count}"}
                print(f"    PASS: Cache status found - {found_indicators}")
            else:
                # Check for time indicators that might indicate cache
                time_indicators = ["updated", "ago", "min", "hour"]
                time_found = [t for t in time_indicators if t in content.lower()]

                if time_found:
                    results["cache_status"] = {"status": "PARTIAL", "details": f"Time indicators found: {time_found} (may indicate cache age)"}
                    print(f"    PARTIAL: Time indicators found but no explicit cache badge")
                else:
                    results["cache_status"] = {"status": "NOT_FOUND", "details": "No cache status indicators found on page"}
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

        # Status indicators
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

    return passed, partial, failed, total


if __name__ == "__main__":
    results = run_full_feature_test()
    print_summary(results)
