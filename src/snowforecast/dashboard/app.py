"""Minimal Streamlit test for debugging deployment."""
import sys
from pathlib import Path

# Add src directory to path for imports
_app_file = Path(__file__).resolve()
_src_path = _app_file.parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import streamlit as st

st.set_page_config(page_title="Debug Test", page_icon="❄️", layout="wide")
st.title("Snow Forecast - Debug Mode")
st.write("Testing imports one by one...")

# Test 1: Basic imports
try:
    from datetime import datetime
    import pandas as pd
    import numpy as np
    st.success("1. Basic imports (datetime, pandas, numpy): OK")
except Exception as e:
    st.error(f"1. Basic imports FAILED: {e}")

# Test 2: DuckDB
try:
    import duckdb
    st.success(f"2. DuckDB import: OK (version {duckdb.__version__})")
except Exception as e:
    st.error(f"2. DuckDB FAILED: {e}")

# Test 3: Pydeck
try:
    import pydeck
    st.success(f"3. PyDeck import: OK")
except Exception as e:
    st.error(f"3. PyDeck FAILED: {e}")

# Test 4: streamlit-folium
try:
    import streamlit_folium
    st.success(f"4. streamlit-folium import: OK")
except Exception as e:
    st.error(f"4. streamlit-folium FAILED: {e}")

# Test 5: streamlit-local-storage
try:
    from streamlit_local_storage import LocalStorage
    st.success(f"5. streamlit-local-storage import: OK")
except Exception as e:
    st.error(f"5. streamlit-local-storage FAILED: {e}")

# Test 6: snowforecast package
try:
    import snowforecast
    st.success(f"6. snowforecast package: OK ({snowforecast.__version__})")
except Exception as e:
    st.error(f"6. snowforecast package FAILED: {e}")

# Test 7: cache module
try:
    from snowforecast.cache import CacheDatabase
    st.success("7. snowforecast.cache.CacheDatabase: OK")
except Exception as e:
    st.error(f"7. CacheDatabase FAILED: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test 8: CachedPredictor
try:
    from snowforecast.cache import CachedPredictor
    st.success("8. snowforecast.cache.CachedPredictor: OK")
except Exception as e:
    st.error(f"8. CachedPredictor FAILED: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test 9: dashboard components
try:
    from snowforecast.dashboard.components import render_resort_map
    st.success("9. dashboard.components: OK")
except Exception as e:
    st.error(f"9. dashboard.components FAILED: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("---")
st.write("If you see this, all tests completed!")
st.write(f"Python path: {sys.path[:3]}")
