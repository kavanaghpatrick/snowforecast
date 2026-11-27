"""Streamlit Cloud entrypoint: runs the v2 dashboard.

Streamlit Cloud executes this file from the repo root; the app's own modules
live in dashboard_v2/ and import each other by bare name, so that folder goes
on sys.path and app.py runs as the page.
"""
import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent / "dashboard_v2"
sys.path.insert(0, str(APP_DIR))
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
