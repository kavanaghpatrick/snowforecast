#!/usr/bin/env python3
"""
Final QA Test Suite for Dashboard V2 (http://localhost:8502)
Uses Playwright for browser automation testing.
"""

import time
import re
from playwright.sync_api import sync_playwright


def run_qa_tests():
    results = {}
    issues = []
    sample_resorts = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # =================================================================
        # TEST 1: Page Load
        # =================================================================
        print("\n[TEST 1] Page Load...")
        try:
            page.goto("http://localhost:8502", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)
            time.sleep(4)  # Wait for Streamlit to fully render

            title = page.title()
            main_title = ""
            h1_elements = page.locator("h1").all()
            if h1_elements:
                main_title = h1_elements[0].inner_text()

            if "Snow Forecast" in title or "Snow Forecast" in main_title:
                results["page_load"] = {"status": "PASS", "details": f"Title: '{main_title}'"}
            else:
                results["page_load"] = {"status": "FAIL", "details": f"Unexpected title: '{title}'"}
                issues.append("Page title mismatch")
        except Exception as e:
            results["page_load"] = {"status": "FAIL", "details": str(e)[:60]}
            issues.append(f"Page load failed: {e}")
            browser.close()
            return results, issues, sample_resorts

        # =================================================================
        # TEST 2: Data Freshness
        # =================================================================
        print("[TEST 2] Data Freshness...")
        try:
            page_text = page.inner_text("body")
            freshness_match = re.search(r'Updated\s+([\d.]+)\s+hours?\s+ago', page_text)
            last_update_match = re.search(r'Last update:\s*([\d\-T:]+)', page_text)

            if freshness_match:
                hours_ago = float(freshness_match.group(1))
                detail = f"Updated {hours_ago} hours ago"
                if last_update_match:
                    detail += f" | {last_update_match.group(1)}"

                if hours_ago < 24:
                    results["data_freshness"] = {"status": "PASS", "details": detail}
                else:
                    results["data_freshness"] = {"status": "WARN", "details": f"Stale: {hours_ago}h old"}
                    issues.append(f"Data stale: {hours_ago} hours old")
            else:
                results["data_freshness"] = {"status": "WARN", "details": "Freshness format not found"}
        except Exception as e:
            results["data_freshness"] = {"status": "FAIL", "details": str(e)[:60]}

        # =================================================================
        # TEST 3: Resort Table
        # =================================================================
        print("[TEST 3] Resort Table...")
        try:
            time.sleep(2)
            page_text = page.inner_text("body")

            # Check columns
            col_checks = {
                "Resort": "Resort" in page_text,
                "State": "State" in page_text,
                "Elevation": any(x in page_text for x in ["Elevation", "Elev"]),
                "Base Depth": any(x in page_text for x in ["Base Depth", "Base", "Depth"]),
                "7-Day Snow": any(x in page_text for x in ["7-Day", "7 Day", "Snow"]),
                "Best Day": any(x in page_text for x in ["Best Day", "Best"])
            }
            cols_found = sum(col_checks.values())

            # Check resorts
            known_resorts = [
                "Stevens Pass", "Mt. Baker", "Crystal Mountain",  # WA
                "Mt. Bachelor", "Mt. Hood",  # OR
                "Mammoth", "Squaw Valley", "Palisades", "Heavenly",  # CA
                "Vail", "Aspen", "Breckenridge", "Keystone",  # CO
                "Jackson Hole", "Grand Targhee",  # WY
                "Big Sky", "Whitefish",  # MT
                "Park City", "Snowbird", "Alta", "Brighton",  # UT
                "Sun Valley", "Schweitzer"  # ID
            ]
            resorts_found = [r for r in known_resorts if r in page_text]

            # Check elevations for bug (all same value)
            elevations = re.findall(r'\b([12345]\d{3})\b', page_text)
            unique_elevs = len(set(elevations))
            elevation_bug = False
            if elevations and elevations.count("2000") > len(elevations) * 0.4:
                elevation_bug = True
                issues.append("ELEVATION BUG: Many 2000m values")

            # Check base depths for variation
            base_pattern = re.findall(r'\b(\d{1,3})\b', page_text)

            detail = f"Cols: {cols_found}/6, Resorts: {len(resorts_found)}, Unique elevs: {unique_elevs}"

            if len(resorts_found) >= 15 and not elevation_bug:
                results["resort_table"] = {"status": "PASS", "details": detail}
            elif len(resorts_found) >= 10:
                results["resort_table"] = {"status": "WARN" if not elevation_bug else "FAIL", "details": detail}
            else:
                results["resort_table"] = {"status": "FAIL", "details": detail}
                issues.append(f"Only {len(resorts_found)} resorts found")

        except Exception as e:
            results["resort_table"] = {"status": "FAIL", "details": str(e)[:60]}
            issues.append(f"Table check failed: {e}")

        # =================================================================
        # TEST 4: State Filter
        # =================================================================
        print("[TEST 4] State Filter...")
        try:
            # Click somewhere neutral first to close any open dropdowns
            page.locator("h1").first.click()
            time.sleep(0.5)

            # Find the selectbox by label
            selectbox_container = page.locator("[data-testid='stSelectbox']").first
            if not selectbox_container.is_visible():
                # Try alternative selector
                selectbox_container = page.locator(".stSelectbox").first

            if selectbox_container.is_visible():
                # Click the selectbox to open dropdown
                selectbox_container.click()
                time.sleep(1)

                # Look for dropdown options in the listbox
                listbox = page.locator("[role='listbox'], [data-baseweb='popover'] ul").first
                time.sleep(0.5)

                if listbox.is_visible():
                    options_text = listbox.inner_text()
                    states_available = ["California", "Colorado", "Idaho", "Montana", "Oregon", "Utah", "Washington"]
                    states_found = [s for s in states_available if s in options_text]

                    # Click Washington option
                    wa_option = page.locator("[role='option']:has-text('Washington'), li:has-text('Washington')").first
                    if wa_option.is_visible():
                        wa_option.click()
                        time.sleep(2)

                        # Check filtered results
                        filtered_text = page.inner_text("body")
                        wa_resorts = ["Stevens Pass", "Mt. Baker", "Crystal Mountain"]
                        non_wa = ["Vail", "Mammoth", "Park City", "Jackson Hole"]

                        wa_visible = [r for r in wa_resorts if r in filtered_text]
                        non_wa_visible = [r for r in non_wa if r in filtered_text]

                        if wa_visible and not non_wa_visible:
                            results["state_filter"] = {"status": "PASS", "details": f"Filter works: {len(wa_visible)} WA resorts shown"}
                        else:
                            results["state_filter"] = {"status": "WARN", "details": f"WA:{len(wa_visible)}, Others:{len(non_wa_visible)}"}

                        # Reset to All
                        selectbox_container = page.locator("[data-testid='stSelectbox'], .stSelectbox").first
                        selectbox_container.click()
                        time.sleep(0.5)
                        all_option = page.locator("[role='option']:has-text('All States'), li:has-text('All States')").first
                        if all_option.is_visible():
                            all_option.click()
                            time.sleep(1)
                    else:
                        results["state_filter"] = {"status": "WARN", "details": f"States available: {states_found}, but WA option not clickable"}
                else:
                    results["state_filter"] = {"status": "WARN", "details": "Dropdown opened but listbox not found"}
            else:
                results["state_filter"] = {"status": "FAIL", "details": "Selectbox not found"}
                issues.append("State filter selectbox not found")

        except Exception as e:
            results["state_filter"] = {"status": "FAIL", "details": str(e)[:60]}
            issues.append(f"State filter test error: {str(e)[:40]}")

        # =================================================================
        # TEST 5: Bar Chart
        # =================================================================
        print("[TEST 5] Bar Chart...")
        try:
            # Scroll to bottom to ensure chart is visible
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)

            # Look for chart elements
            vega = page.locator("[data-testid='stVegaLiteChart'], .vega-embed").all()
            plotly = page.locator("[data-testid='stPlotlyChart']").all()

            # Also check for SVG elements that look like charts
            svgs = page.locator("svg").all()
            large_svgs = []
            for svg in svgs:
                try:
                    box = svg.bounding_box()
                    if box and box['height'] > 150 and box['width'] > 300:
                        large_svgs.append(svg)
                except:
                    pass

            if vega:
                results["bar_chart"] = {"status": "PASS", "details": f"Vega/Altair chart rendered ({len(vega)} charts)"}
            elif plotly:
                results["bar_chart"] = {"status": "PASS", "details": f"Plotly chart rendered ({len(plotly)} charts)"}
            elif large_svgs:
                results["bar_chart"] = {"status": "PASS", "details": f"SVG chart found ({len(large_svgs)} large SVGs)"}
            else:
                results["bar_chart"] = {"status": "WARN", "details": "No chart definitively identified"}

        except Exception as e:
            results["bar_chart"] = {"status": "FAIL", "details": str(e)[:60]}

        # =================================================================
        # TEST 6: Data Validation
        # =================================================================
        print("[TEST 6] Data Validation...")
        try:
            # Scroll back to top
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            page_text = page.inner_text("body")

            targets = ["Stevens Pass", "Mt. Baker", "Mammoth Mountain", "Vail", "Jackson Hole"]
            for resort in targets:
                if resort in page_text:
                    sample_resorts[resort] = "FOUND"
                elif resort.replace(" Mountain", "") in page_text:
                    sample_resorts[resort] = "FOUND (partial)"
                else:
                    sample_resorts[resort] = "NOT FOUND"

            found = sum(1 for v in sample_resorts.values() if "FOUND" in v)

            # Check for suspicious patterns
            numbers = re.findall(r'\b(\d{2,4})\b', page_text)
            if numbers:
                from collections import Counter
                common = Counter(numbers).most_common(3)
                for val, cnt in common:
                    if cnt > 25 and val not in ["2026", "2025", "0"]:
                        issues.append(f"Suspicious: '{val}' appears {cnt} times")

            if found >= 4:
                results["data_validation"] = {"status": "PASS", "details": f"Found {found}/5 target resorts"}
            else:
                results["data_validation"] = {"status": "FAIL", "details": f"Only {found}/5 found"}
                issues.append(f"Missing target resorts: {found}/5")

        except Exception as e:
            results["data_validation"] = {"status": "FAIL", "details": str(e)[:60]}

        # Save final screenshot
        try:
            page.screenshot(path="/Users/patrickkavanagh/snowforecast/tests/dashboard_v2_qa_final.png", full_page=True)
            print("Screenshot saved: tests/dashboard_v2_qa_final.png")
        except:
            pass

        browser.close()

    return results, issues, sample_resorts


def print_report(results, issues, sample_resorts):
    print("\n" + "=" * 80)
    print("DASHBOARD V2 QA TEST REPORT")
    print("URL: http://localhost:8502")
    print("=" * 80)

    print("\n## Test Results Summary\n")
    print(f"| {'Test':<20} | {'Status':<8} | {'Details':<45} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*47}|")

    for test, data in results.items():
        status = data.get('status', 'UNKNOWN')
        details = data.get('details', '')[:45]
        print(f"| {test:<20} | {status:<8} | {details:<45} |")

    print("\n## Sample Resorts Data\n")
    for resort, value in sample_resorts.items():
        status_icon = "[x]" if "FOUND" in value else "[ ]"
        print(f"  {status_icon} {resort}: {value}")

    if issues:
        print("\n## Issues Found\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n## No Critical Issues Found")

    # Summary
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    failed = sum(1 for r in results.values() if r.get("status") == "FAIL")
    warned = sum(1 for r in results.values() if r.get("status") == "WARN")
    total = len(results)

    print(f"\n## Summary")
    print(f"  - PASS: {passed}/{total}")
    print(f"  - FAIL: {failed}/{total}")
    print(f"  - WARN: {warned}/{total}")

    overall = "PASS" if failed == 0 else "FAIL"
    print(f"\n## Overall Result: {overall}")
    print("=" * 80)

    return overall


if __name__ == "__main__":
    results, issues, sample_resorts = run_qa_tests()
    overall = print_report(results, issues, sample_resorts)
    exit(0 if overall == "PASS" else 1)
