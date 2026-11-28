#!/usr/bin/env python3
"""
Detailed QA Test Suite for Dashboard V2 (http://localhost:8502)
Uses Playwright for browser automation testing.
"""

import time
import re
from playwright.sync_api import sync_playwright


def run_detailed_qa_tests():
    results = {}
    issues = []
    sample_resorts = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # =================================================================
        # TEST 1: Page Load
        # =================================================================
        print("\n[TEST 1] Page Load...")
        try:
            page.goto("http://localhost:8502", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)
            time.sleep(3)

            title = page.title()

            # Check for main title
            main_title = page.locator("h1").first.inner_text() if page.locator("h1").count() > 0 else ""

            if "Snow Forecast" in title or "Snow Forecast" in main_title:
                results["page_load"] = {"status": "PASS", "details": f"Title: '{main_title}'"}
            else:
                results["page_load"] = {"status": "FAIL", "details": f"Title mismatch: '{title}' / '{main_title}'"}
                issues.append("Page title does not contain 'Snow Forecast'")
        except Exception as e:
            results["page_load"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"Page load failed: {e}")
            browser.close()
            return results, issues, sample_resorts

        # =================================================================
        # TEST 2: Data Freshness
        # =================================================================
        print("[TEST 2] Data Freshness...")
        try:
            # Look for the green success box with freshness info
            page_text = page.inner_text("body")

            freshness_match = re.search(r'Updated\s+([\d.]+)\s+hours?\s+ago', page_text)
            last_update_match = re.search(r'Last update:\s*([\d\-T:]+)', page_text)

            if freshness_match:
                hours_ago = float(freshness_match.group(1))
                freshness_text = f"Updated {hours_ago} hours ago"

                if last_update_match:
                    freshness_text += f" | Last update: {last_update_match.group(1)}"

                if hours_ago < 24:
                    results["data_freshness"] = {"status": "PASS", "details": freshness_text}
                else:
                    results["data_freshness"] = {"status": "WARN", "details": f"Data is {hours_ago} hours old"}
                    issues.append(f"Data is stale: {hours_ago} hours old")
            else:
                results["data_freshness"] = {"status": "WARN", "details": "Freshness indicator not found in expected format"}
        except Exception as e:
            results["data_freshness"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"Freshness check failed: {e}")

        # =================================================================
        # TEST 3: Resort Table
        # =================================================================
        print("[TEST 3] Resort Table...")
        try:
            # Wait for table to be present
            time.sleep(2)

            # Get all table-like content
            table_elements = page.locator("[data-testid='stDataFrame'], .stDataFrame, table, [role='grid']").all()

            if not table_elements:
                # Try iframe approach (Streamlit sometimes uses iframes for dataframes)
                iframes = page.frames
                for frame in iframes:
                    table_elements = frame.locator("table").all()
                    if table_elements:
                        break

            page_text = page.inner_text("body")

            # Check for column headers
            expected_columns = ["Resort", "State", "Elevation", "Base Depth", "7-Day Snow", "Best Day"]
            columns_found = []
            for col in expected_columns:
                if col.lower() in page_text.lower():
                    columns_found.append(col)

            # Count resorts by looking for known resort names
            known_resorts = [
                "Stevens Pass", "Mt. Baker", "Crystal Mountain", "Snoqualmie",  # WA
                "Mt. Hood Meadows", "Mt. Bachelor",  # OR
                "Mammoth", "Squaw", "Palisades", "Heavenly", "Kirkwood",  # CA
                "Vail", "Aspen", "Breckenridge", "Keystone", "Winter Park",  # CO
                "Jackson Hole", "Grand Targhee",  # WY
                "Big Sky", "Whitefish",  # MT
                "Park City", "Snowbird", "Alta", "Brighton", "Deer Valley",  # UT
                "Sun Valley", "Schweitzer"  # ID
            ]

            resorts_found = [r for r in known_resorts if r in page_text]
            resort_count = len(resorts_found)

            # Check for elevation values - look for reasonable elevations (1500-4000m range)
            elevations = re.findall(r'\b([12345]\d{3})\b', page_text)
            unique_elevations = set(elevations)

            # Check for the old bug (all 2000m)
            elevation_bug = False
            if elevations:
                if elevations.count("2000") > len(elevations) * 0.5:
                    elevation_bug = True
                    issues.append("ELEVATION BUG: More than 50% of elevations are 2000m")

            # Check base depth values
            base_depths = re.findall(r'\b(\d{1,3})\b', page_text)

            details = f"Columns: {len(columns_found)}/6 ({', '.join(columns_found)}), "
            details += f"Resorts: {resort_count}, Unique elevations: {len(unique_elevations)}"

            if len(columns_found) >= 4 and resort_count >= 10:
                if elevation_bug:
                    results["resort_table"] = {"status": "FAIL", "details": details + " | ELEVATION BUG DETECTED"}
                else:
                    results["resort_table"] = {"status": "PASS", "details": details}
            elif resort_count >= 5:
                results["resort_table"] = {"status": "WARN", "details": details}
            else:
                results["resort_table"] = {"status": "FAIL", "details": details}
                issues.append(f"Only found {resort_count} resorts in table")

        except Exception as e:
            results["resort_table"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"Resort table check failed: {e}")

        # =================================================================
        # TEST 4: State Filter
        # =================================================================
        print("[TEST 4] State Filter...")
        try:
            # Find the state filter dropdown
            filter_label = page.locator("text=Filter by State")

            if filter_label.count() > 0:
                # Find the selectbox near it
                selectbox = page.locator("[data-baseweb='select'], .stSelectbox").first

                if selectbox.is_visible():
                    # Click to open
                    selectbox.click()
                    time.sleep(1)

                    # Take screenshot with dropdown open
                    page.screenshot(path="/Users/patrickkavanagh/snowforecast/tests/dashboard_v2_dropdown.png")

                    # Look for Washington in the dropdown
                    dropdown_text = page.inner_text("body")
                    states_in_dropdown = ["California", "Colorado", "Idaho", "Montana", "Oregon", "Utah", "Washington"]
                    states_found = [s for s in states_in_dropdown if s in dropdown_text]

                    if "Washington" in states_found:
                        # Click Washington
                        wa_option = page.locator("text=Washington").first
                        wa_option.click()
                        time.sleep(2)

                        # Check filtered results
                        filtered_text = page.inner_text("body")

                        wa_resorts = ["Stevens Pass", "Mt. Baker", "Crystal Mountain", "Snoqualmie"]
                        non_wa_resorts = ["Vail", "Mammoth", "Jackson Hole", "Park City", "Aspen"]

                        wa_visible = [r for r in wa_resorts if r in filtered_text]
                        non_wa_visible = [r for r in non_wa_resorts if r in filtered_text]

                        if len(wa_visible) > 0 and len(non_wa_visible) == 0:
                            results["state_filter"] = {
                                "status": "PASS",
                                "details": f"WA filter works: showing {len(wa_visible)} WA resorts, hiding non-WA"
                            }
                        elif len(non_wa_visible) > 0:
                            results["state_filter"] = {
                                "status": "FAIL",
                                "details": f"Filter not working: still showing {non_wa_visible}"
                            }
                            issues.append("State filter not working correctly")
                        else:
                            results["state_filter"] = {
                                "status": "WARN",
                                "details": f"WA visible: {wa_visible}, unexpected: {non_wa_visible}"
                            }

                        # Reset to All States
                        selectbox = page.locator("[data-baseweb='select'], .stSelectbox").first
                        selectbox.click()
                        time.sleep(0.5)
                        all_option = page.locator("text=All States").first
                        if all_option.is_visible():
                            all_option.click()
                            time.sleep(1)
                    else:
                        results["state_filter"] = {
                            "status": "FAIL",
                            "details": f"Washington not in dropdown. Found: {states_found}"
                        }
                else:
                    results["state_filter"] = {"status": "FAIL", "details": "Selectbox not visible"}
            else:
                results["state_filter"] = {"status": "FAIL", "details": "Filter label not found"}

        except Exception as e:
            results["state_filter"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"State filter test failed: {e}")

        # =================================================================
        # TEST 5: Bar Chart
        # =================================================================
        print("[TEST 5] Bar Chart...")
        try:
            # Scroll down to see chart
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)

            # Look for chart elements
            vega_chart = page.locator("[data-testid='stVegaLiteChart'], .vega-embed, .marks").all()
            plotly_chart = page.locator("[data-testid='stPlotlyChart'], .plotly").all()
            canvas_charts = page.locator("canvas").all()
            svg_charts = page.locator("svg").all()

            # Filter SVGs that are large enough to be charts
            chart_svgs = []
            for svg in svg_charts:
                try:
                    box = svg.bounding_box()
                    if box and box['height'] > 100 and box['width'] > 200:
                        chart_svgs.append(svg)
                except:
                    pass

            total_charts = len(vega_chart) + len(plotly_chart) + len(canvas_charts) + len(chart_svgs)

            if len(vega_chart) > 0:
                results["bar_chart"] = {"status": "PASS", "details": f"Vega/Altair chart found ({len(vega_chart)})"}
            elif len(plotly_chart) > 0:
                results["bar_chart"] = {"status": "PASS", "details": f"Plotly chart found ({len(plotly_chart)})"}
            elif len(chart_svgs) > 0:
                results["bar_chart"] = {"status": "PASS", "details": f"SVG chart found ({len(chart_svgs)} chart-like SVGs)"}
            else:
                results["bar_chart"] = {"status": "WARN", "details": "No chart elements definitively identified"}

            # Take full page screenshot
            page.screenshot(path="/Users/patrickkavanagh/snowforecast/tests/dashboard_v2_full.png", full_page=True)

        except Exception as e:
            results["bar_chart"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"Bar chart check failed: {e}")

        # =================================================================
        # TEST 6: Data Validation - Extract actual values
        # =================================================================
        print("[TEST 6] Data Validation...")
        try:
            # Scroll back to top
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            page_text = page.inner_text("body")

            # Target resorts to find
            target_resorts = {
                "Stevens Pass": None,
                "Mt. Baker": None,
                "Mammoth Mountain": None,
                "Vail": None,
                "Jackson Hole": None
            }

            # Try to extract data from the page
            # Look for rows that contain resort names
            lines = page_text.split('\n')

            for resort in target_resorts.keys():
                for line in lines:
                    if resort in line:
                        # Try to extract numbers from this line or nearby
                        numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
                        target_resorts[resort] = {"found": True, "line": line[:80], "numbers": numbers[:6]}
                        break
                else:
                    # Check if resort appears anywhere
                    if resort in page_text:
                        target_resorts[resort] = {"found": True, "line": "Found but could not extract row", "numbers": []}
                    else:
                        # Try alternate names
                        alt_name = resort.replace(" Mountain", "").replace("Mt. ", "Mount ")
                        if alt_name in page_text or resort.split()[0] in page_text:
                            target_resorts[resort] = {"found": True, "line": f"Found as partial match", "numbers": []}
                        else:
                            target_resorts[resort] = {"found": False, "line": "NOT FOUND", "numbers": []}

            # Build sample resorts output
            for resort, data in target_resorts.items():
                if data["found"]:
                    sample_resorts[resort] = f"Found | {data.get('line', 'N/A')[:50]}"
                else:
                    sample_resorts[resort] = "NOT FOUND"

            found_count = sum(1 for d in target_resorts.values() if d["found"])

            # Check for suspicious patterns
            all_numbers = re.findall(r'\b(\d{1,4})\b', page_text)
            if all_numbers:
                from collections import Counter
                num_counts = Counter(all_numbers)
                most_common = num_counts.most_common(5)

                # Check if any number appears way too often (suspicious)
                for num, count in most_common:
                    if count > 20 and num not in ["0", "1", "2", "3", "7"]:  # Ignore common small numbers
                        issues.append(f"SUSPICIOUS: Value '{num}' appears {count} times")

            # Check for all zeros in important columns
            zero_count = page_text.count(" 0 ") + page_text.count("\t0\t") + page_text.count(",0,")
            if zero_count > 15:
                issues.append(f"WARNING: Many zero values detected ({zero_count})")

            if found_count >= 4:
                results["data_validation"] = {"status": "PASS", "details": f"Found {found_count}/5 target resorts"}
            elif found_count >= 2:
                results["data_validation"] = {"status": "WARN", "details": f"Only {found_count}/5 target resorts found"}
            else:
                results["data_validation"] = {"status": "FAIL", "details": f"Only {found_count}/5 target resorts found"}
                issues.append("Many target resorts not found")

        except Exception as e:
            results["data_validation"] = {"status": "FAIL", "details": str(e)}
            issues.append(f"Data validation failed: {e}")

        browser.close()

    return results, issues, sample_resorts


def print_report(results, issues, sample_resorts):
    print("\n" + "=" * 80)
    print("DASHBOARD V2 QA TEST REPORT")
    print("=" * 80)

    print("\n### Test Results Summary\n")
    print(f"{'Test':<20} {'Status':<10} {'Details'}")
    print("-" * 80)

    for test, data in results.items():
        status = data.get('status', 'UNKNOWN')
        details = data.get('details', '')[:55]
        print(f"{test:<20} {status:<10} {details}")

    print("\n### Sample Resorts Data\n")
    for resort, value in sample_resorts.items():
        print(f"  - {resort}: {value}")

    if issues:
        print("\n### Issues Found\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n### No Critical Issues Found")

    # Summary
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    failed = sum(1 for r in results.values() if r.get("status") == "FAIL")
    warned = sum(1 for r in results.values() if r.get("status") == "WARN")

    print(f"\n### Summary: {passed} PASS, {failed} FAIL, {warned} WARN out of {len(results)} tests")

    overall = "PASS" if failed == 0 else "FAIL"
    print(f"\n### Overall Result: {overall}")
    print("=" * 80)

    return overall


if __name__ == "__main__":
    results, issues, sample_resorts = run_detailed_qa_tests()
    print_report(results, issues, sample_resorts)
