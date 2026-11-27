"""Streamlit Cloud entrypoint: runs the v2 dashboard.

Streamlit Cloud executes this file from the repo root; the app's own modules
live in dashboard_v2/ and import each other by bare name, so that folder goes
on sys.path and app.py runs as the page.
"""
import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent / "dashboard_v2"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Streamlit re-runs this script from source but keeps imported modules cached
# in sys.modules. After the platform syncs new code, that mixes a fresh app.py
# with stale render/art/data modules and crashes on changed signatures. Purge
# our modules every run so each rerun imports the source that is on disk now.
_prefix = str(APP_DIR)
for _name, _mod in list(sys.modules.items()):
    _file = getattr(_mod, "__file__", None)
    if _file and _file.startswith(_prefix):
        del sys.modules[_name]

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
