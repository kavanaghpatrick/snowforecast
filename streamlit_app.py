"""Minimal Streamlit test for debugging deployment."""
import streamlit as st

st.set_page_config(page_title="Test", page_icon="❄️")
st.title("Snow Forecast Dashboard - Test Mode")
st.write("If you see this, basic Streamlit is working!")

# Now try importing the package
import sys
from pathlib import Path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

st.write(f"Python path includes: {src_path}")

try:
    import snowforecast
    st.success(f"snowforecast package imported: {snowforecast.__version__}")
except Exception as e:
    st.error(f"Failed to import snowforecast: {e}")

try:
    from snowforecast.cache import CachedPredictor
    st.success("CachedPredictor imported successfully")
except Exception as e:
    st.error(f"Failed to import CachedPredictor: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    from snowforecast.dashboard.components import render_resort_map
    st.success("Dashboard components imported successfully")
except Exception as e:
    st.error(f"Failed to import dashboard components: {e}")
    import traceback
    st.code(traceback.format_exc())
