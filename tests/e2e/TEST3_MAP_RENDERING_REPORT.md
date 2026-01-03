# TEST 3: Map Rendering Test Report

**Test Date**: 2026-01-03
**Test Type**: Playwright E2E Test
**Target**: PyDeck Map Component in Snowforecast Dashboard

---

## Executive Summary

**RESULT: FAIL - Application Not Running**

Both Streamlit Cloud URLs are currently returning errors, preventing map rendering verification.

---

## URLs Tested

| URL | Status | Error |
|-----|--------|-------|
| https://snowforecast.streamlit.app | ERROR | "Oh no. Error running app." |
| https://kavanaghpatrick-snowforecast.streamlit.app | ERROR | "You do not have access to this app or it does not exist" |

---

## Test Findings

### 1. Application Status

**Primary URL (snowforecast.streamlit.app)**:
- Page loads successfully (10.58s load time)
- Streamlit framework renders
- Application crashes with generic error: "Oh no. Error running app. If this keeps happening, please contact support."
- No console errors related to PyDeck or WebGL observed
- No JavaScript exceptions caught

**Alternate URL (kavanaghpatrick-snowforecast.streamlit.app)**:
- Returns 401/404 errors
- Message: "You do not have access to this app or it does not exist"
- This URL appears to be invalid or the app is set to private

### 2. Console Errors Captured

```
Failed to load resource: the server responded with a status of 404 ()
Failed to load resource: the server responded with a status of 401 () (x3)
```

These errors are related to the alternate URL access denial, not PyDeck/map issues.

### 3. Map Component Analysis (Code Review)

Since the app is not running, I performed a code review of the map implementation:

**Map Technology**: PyDeck (deck.gl)
- Library: `pydeck>=0.9.0`
- Rendered via: `st.pydeck_chart(deck, use_container_width=True)`
- Location: `/Users/patrickkavanagh/snowforecast/src/snowforecast/dashboard/app.py` (lines 340-346)

**Map Features**:
- ScatterplotLayer for resort markers
- Color encoding based on snow depth
- Size encoding based on new snow amount
- Tooltip showing resort name, snow depth, new snow, probability
- CartoDB basemap (free, no API key required)

**Fallback Handling**:
```python
try:
    deck = render_resort_map(conditions_df)
    st.pydeck_chart(deck, use_container_width=True)
except Exception as e:
    st.error(f"Map error: {e}")
    st.map(conditions_df[["latitude", "longitude"]])  # Fallback
```

### 4. Screenshots Captured

| File | Description |
|------|-------------|
| `map_test_full_20260103_155835.png` | Full page showing "Oh no." error |
| `map_full_20260103_160006.png` | Alternate URL access denied page |
| `app_error_20260103_160028.png` | Primary URL error state |

---

## Technical Details

### Page Load Metrics

| Metric | Value |
|--------|-------|
| Total Load Time (Primary) | 10.58s |
| Total Load Time (Alternate) | 21.88s |
| Network State | idle achieved |
| JavaScript Errors | None |
| WebGL/Canvas Errors | None (app didn't reach map rendering) |

### Expected Map Elements (from code analysis)

When the app is working, these DOM elements should be present:
- `[data-testid="stDeckGlJsonChart"]` - Streamlit's PyDeck wrapper
- `canvas` - deck.gl rendering canvas
- Tooltip with resort data on hover

### Memory Considerations

The app.py shows memory optimization code:
```python
@st.cache_data(ttl=1800, max_entries=2, show_spinner="Loading resort conditions...")
def fetch_all_conditions() -> pd.DataFrame:
    """Memory optimization: max_entries=2, shorter TTL for Streamlit Cloud 1GB limit."""
```

The 1GB Streamlit Cloud memory limit may be related to the crash.

---

## Root Cause Analysis

The "Oh no. Error running app." message is a generic Streamlit Cloud error indicating the Python app crashed during execution. Possible causes:

1. **Memory Limit (1GB)** - The app loads HRRR weather data and terrain features which may exceed memory
2. **Import Error** - Missing dependency or import failure
3. **Data Source Timeout** - HRRR or DEM data source unavailable
4. **CachedPredictor Initialization** - The predictor requires DuckDB cache which may not initialize correctly

---

## Recommendations

1. **Check Streamlit Cloud Logs**
   - Access https://share.streamlit.io/ dashboard
   - Review app crash logs for specific error message

2. **Local Testing**
   ```bash
   cd /Users/patrickkavanagh/snowforecast
   streamlit run src/snowforecast/dashboard/app.py
   ```

3. **Memory Profiling**
   - Add memory monitoring to dashboard
   - Reduce cache size further if needed

4. **Verify Data Sources**
   - Ensure HRRR data is accessible
   - Check CachedPredictor initialization

---

## Test Artifacts

**Test Scripts**:
- `/Users/patrickkavanagh/snowforecast/tests/e2e/test_map_rendering.py`
- `/Users/patrickkavanagh/snowforecast/tests/e2e/test_map_rendering_v2.py`

**Screenshots**:
- `/Users/patrickkavanagh/snowforecast/tests/e2e/screenshots/`

**Existing E2E Tests**:
- `/Users/patrickkavanagh/snowforecast/tests/e2e/test_dashboard.py`
- `/Users/patrickkavanagh/snowforecast/tests/dashboard/test_map_view.py`

---

## Conclusion

**TEST 3: FAIL**

The PyDeck map rendering could not be verified because the Streamlit Cloud deployment is currently crashing. The application error prevents any UI components from loading, including the map.

**Action Required**: Investigate and fix the Streamlit Cloud deployment before re-running this test.

**Next Steps**:
1. Check Streamlit Cloud deployment logs
2. Run app locally to verify map works
3. Fix deployment issue
4. Re-run TEST 3
