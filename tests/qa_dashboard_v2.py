#!/usr/bin/env python3
"""
QA Test Suite for Dashboard V2 (http://localhost:8502)
Uses Playwright for browser automation testing.
"""

import time
from playwright.sync_api import sync_playwright, expect


def run_qa_tests():
    results = {
        "page_load": {"status": "PENDING", "details": ""},
        "data_freshness": {"status": "PENDING", "details": ""},
        "resort_table": {"status": "PENDING", "details": ""},
        "state_filter": {"status": "PENDING", "details": ""},
        "bar_chart": {"status": "PENDING", "details": ""},
        "data_validation": {"status": "PENDING", "details": ""},
    }

    issues = []
    sample_resorts = {}

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # =================================================================
        # TEST 1: Page Load
        # =================================================================
        print("\n[TEST 1] Page Load...")
        try:
            page.goto("http://localhost:8502", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)

            # Wait for Streamlit to fully render
            time.sleep(3)

            title = page.title()
            if "Snow Forecast" in title or "Streamlit" in title:
                results["page_load"]["status"] = "PASS"
                results["page_load"]["details"] = f"Title: {title}"
            else:
                results["page_load"]["status"] = "FAIL"
                results["page_load"]["details"] = f"Unexpected title: {title}"
                issues.append(f"Page title mismatch: got '{title}'")
        except Exception as e:
            results["page_load"]["status"] = "FAIL"
            results["page_load"]["details"] = str(e)
            issues.append(f"Page load failed: {e}")
            browser.close()
            return results, issues, sample_resorts

        # =================================================================
        # TEST 2: Data Freshness
        # =================================================================
        print("[TEST 2] Data Freshness...")
        try:
            # Look for freshness indicators
            page_content = page.content()

            # Check for various freshness indicators
            freshness_found = False
            freshness_text = ""

            # Look for "Updated" text
            updated_elements = page.locator("text=/Updated|hours ago|minutes ago|Fresh|Stale/i").all()
            if updated_elements:
                for elem in updated_elements:
                    try:
                        text = elem.inner_text()
                        if text:
                            freshness_text = text
                            freshness_found = True
                            break
                    except:
                        pass

            # Also check for success/warning boxes
            if not freshness_found:
                success_boxes = page.locator(".stSuccess, [data-testid='stSuccess']").all()
                warning_boxes = page.locator(".stWarning, [data-testid='stWarning']").all()

                for box in success_boxes + warning_boxes:
                    try:
                        text = box.inner_text()
                        if "update" in text.lower() or "hour" in text.lower() or "fresh" in text.lower():
                            freshness_text = text
                            freshness_found = True
                            break
                    except:
                        pass

            if freshness_found:
                results["data_freshness"]["status"] = "PASS"
                results["data_freshness"]["details"] = freshness_text[:100]
            else:
                results["data_freshness"]["status"] = "WARN"
                results["data_freshness"]["details"] = "No freshness indicator found"
        except Exception as e:
            results["data_freshness"]["status"] = "FAIL"
            results["data_freshness"]["details"] = str(e)
            issues.append(f"Data freshness check failed: {e}")

        # =================================================================
        # TEST 3: Resort Table
        # =================================================================
        print("[TEST 3] Resort Table...")
        try:
            # Wait for table to load
            time.sleep(2)

            # Look for dataframe/table
            table = page.locator("[data-testid='stDataFrame'], .stDataFrame, table").first

            if table.is_visible():
                # Get table content
                table_text = table.inner_text()

                # Check for expected columns
                expected_cols = ["Resort", "State", "Elevation", "Base", "Snow", "Day"]
                cols_found = sum(1 for col in expected_cols if col.lower() in table_text.lower())

                # Count rows (look for resort names)
                known_resorts = ["Stevens Pass", "Mt. Baker", "Mammoth", "Vail", "Jackson Hole",
                                "Big Sky", "Park City", "Snowbird", "Alta", "Aspen"]
                resorts_found = sum(1 for r in known_resorts if r in table_text)

                # Check for elevation bug (all 2000m)
                elevation_bug = table_text.count("2000") > 10

                # Check for base depth variation
                # This is tricky without parsing, let's look for multiple different numbers
                import re
                numbers = re.findall(r'\b\d{2,4}\b', table_text)
                unique_numbers = len(set(numbers))

                details = f"Columns found: {cols_found}/6, Resorts visible: {resorts_found}, Unique numbers: {unique_numbers}"

                if cols_found >= 4 and resorts_found >= 5:
                    if elevation_bug:
                        results["resort_table"]["status"] = "WARN"
                        details += " | WARNING: Possible elevation bug (many 2000m values)"
                        issues.append("Possible elevation bug - too many 2000m values")
                    else:
                        results["resort_table"]["status"] = "PASS"
                else:
                    results["resort_table"]["status"] = "FAIL"
                    issues.append("Table structure issues")

                results["resort_table"]["details"] = details
            else:
                results["resort_table"]["status"] = "FAIL"
                results["resort_table"]["details"] = "Table not visible"
                issues.append("Resort table not visible")
        except Exception as e:
            results["resort_table"]["status"] = "FAIL"
            results["resort_table"]["details"] = str(e)
            issues.append(f"Resort table check failed: {e}")

        # =================================================================
        # TEST 4: State Filter
        # =================================================================
        print("[TEST 4] State Filter...")
        try:
            # Look for selectbox/dropdown
            selectbox = page.locator("[data-testid='stSelectbox'], .stSelectbox, select").first

            if selectbox.is_visible():
                # Click to open dropdown
                selectbox.click()
                time.sleep(1)

                # Look for Washington option
                wa_option = page.locator("text=Washington").first
                if wa_option.is_visible():
                    wa_option.click()
                    time.sleep(2)

                    # Check if filtering worked
                    table_text = page.locator("[data-testid='stDataFrame'], .stDataFrame, table").first.inner_text()

                    # Washington resorts
                    wa_resorts = ["Stevens Pass", "Mt. Baker", "Crystal Mountain", "Snoqualmie"]
                    wa_found = sum(1 for r in wa_resorts if r in table_text)

                    # Non-WA resorts that should NOT appear
                    other_resorts = ["Vail", "Mammoth", "Jackson Hole"]
                    others_found = sum(1 for r in other_resorts if r in table_text)

                    if wa_found > 0 and others_found == 0:
                        results["state_filter"]["status"] = "PASS"
                        results["state_filter"]["details"] = f"WA filter working: {wa_found} WA resorts, {others_found} other resorts"
                    else:
                        results["state_filter"]["status"] = "WARN"
                        results["state_filter"]["details"] = f"Filter may not be working: WA={wa_found}, Others={others_found}"

                    # Reset to All
                    selectbox.click()
                    time.sleep(0.5)
                    all_option = page.locator("text=/^All$/").first
                    if all_option.is_visible():
                        all_option.click()
                        time.sleep(1)
                else:
                    results["state_filter"]["status"] = "WARN"
                    results["state_filter"]["details"] = "Washington option not found in dropdown"
            else:
                results["state_filter"]["status"] = "WARN"
                results["state_filter"]["details"] = "State filter selectbox not found"
        except Exception as e:
            results["state_filter"]["status"] = "FAIL"
            results["state_filter"]["details"] = str(e)
            issues.append(f"State filter test failed: {e}")

        # =================================================================
        # TEST 5: Bar Chart
        # =================================================================
        print("[TEST 5] Bar Chart...")
        try:
            # Look for Altair/Vega chart or plotly
            chart = page.locator("[data-testid='stVegaLiteChart'], .stVegaLiteChart, [data-testid='stPlotlyChart'], canvas, svg.marks").first

            if chart.is_visible():
                results["bar_chart"]["status"] = "PASS"
                results["bar_chart"]["details"] = "Chart rendered"
            else:
                # Try looking for any SVG that might be a chart
                svg_elements = page.locator("svg").all()
                chart_like_svgs = [s for s in svg_elements if s.bounding_box() and s.bounding_box()['height'] > 100]

                if chart_like_svgs:
                    results["bar_chart"]["status"] = "PASS"
                    results["bar_chart"]["details"] = f"Found {len(chart_like_svgs)} chart-like SVG elements"
                else:
                    results["bar_chart"]["status"] = "WARN"
                    results["bar_chart"]["details"] = "No chart element found"
        except Exception as e:
            results["bar_chart"]["status"] = "FAIL"
            results["bar_chart"]["details"] = str(e)
            issues.append(f"Bar chart check failed: {e}")

        # =================================================================
        # TEST 6: Data Validation
        # =================================================================
        print("[TEST 6] Data Validation...")
        try:
            # Get full page text for data extraction
            page_text = page.inner_text("body")

            target_resorts = ["Stevens Pass", "Mt. Baker", "Mammoth Mountain", "Vail", "Jackson Hole"]

            for resort in target_resorts:
                if resort in page_text:
                    # Try to extract data for this resort
                    # This is approximate since we can't easily parse the table structure
                    sample_resorts[resort] = "Found on page"
                else:
                    sample_resorts[resort] = "NOT FOUND"

            found_count = sum(1 for v in sample_resorts.values() if v != "NOT FOUND")

            if found_count >= 4:
                results["data_validation"]["status"] = "PASS"
                results["data_validation"]["details"] = f"Found {found_count}/5 target resorts"
            elif found_count >= 2:
                results["data_validation"]["status"] = "WARN"
                results["data_validation"]["details"] = f"Only found {found_count}/5 target resorts"
            else:
                results["data_validation"]["status"] = "FAIL"
                results["data_validation"]["details"] = f"Only found {found_count}/5 target resorts"
                issues.append(f"Missing many target resorts")

            # Check for suspicious patterns
            # All zeros
            if page_text.count(" 0 ") > 15 or page_text.count(",0,") > 10:
                issues.append("WARNING: Many zero values detected")

            # All identical values
            import re
            numbers = re.findall(r'\b(\d{3,4})\b', page_text)
            if numbers:
                from collections import Counter
                most_common = Counter(numbers).most_common(1)
                if most_common and most_common[0][1] > 15:
                    issues.append(f"WARNING: Value '{most_common[0][0]}' appears {most_common[0][1]} times - possible bug")

        except Exception as e:
            results["data_validation"]["status"] = "FAIL"
            results["data_validation"]["details"] = str(e)
            issues.append(f"Data validation failed: {e}")

        # Take screenshot for reference
        try:
            page.screenshot(path="/Users/patrickkavanagh/snowforecast/tests/dashboard_v2_screenshot.png", full_page=True)
            print("\nScreenshot saved to tests/dashboard_v2_screenshot.png")
        except Exception as e:
            print(f"Could not save screenshot: {e}")

        browser.close()

    return results, issues, sample_resorts


def print_report(results, issues, sample_resorts):
    print("\n" + "=" * 70)
    print("DASHBOARD V2 QA TEST REPORT")
    print("=" * 70)

    print("\n### Test Results Summary\n")
    print(f"{'Test':<20} {'Status':<10} {'Details'}")
    print("-" * 70)

    for test, data in results.items():
        status_emoji = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "PENDING": "----"}
        print(f"{test:<20} {status_emoji.get(data['status'], data['status']):<10} {data['details'][:45]}")

    print("\n### Sample Resorts Data\n")
    for resort, value in sample_resorts.items():
        print(f"  - {resort}: {value}")

    if issues:
        print("\n### Issues Found\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n### No Issues Found")

    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    warned = sum(1 for r in results.values() if r["status"] == "WARN")

    print(f"\n### Summary: {passed} PASS, {failed} FAIL, {warned} WARN")
    print("=" * 70)


if __name__ == "__main__":
    results, issues, sample_resorts = run_qa_tests()
    print_report(results, issues, sample_resorts)
