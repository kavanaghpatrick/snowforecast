"""Shared fixtures for the dashboard_v2 suite.

Everything here is real: the real data/forecasts_v2.json, the real dashboard
modules imported from dashboard_v2/, real streamlit, and real JSON files under
fixtures/ derived from recorded payloads (see fixtures/make_fixtures.py).
No mocks and no monkeypatching anywhere in this suite. Network is never used:
tests that exercise the GitHub read run the real code in a subprocess whose
HTTPS proxy is a real local socket server (dead, or stalling).
"""

from __future__ import annotations

import copy
import html as _html
import json
import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH_DIR = ROOT / "dashboard_v2"
DATA_FILE = ROOT / "data" / "forecasts_v2.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(DASH_DIR) not in sys.path:
    sys.path.insert(0, str(DASH_DIR))

# streamlit's cache decorator logs "No runtime found" when imported outside a
# server; it is harmless here.
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="streamlit")

import data as D  # noqa: E402,F401
import everywhere as E  # noqa: E402,F401
import powder as P  # noqa: E402,F401
import render as R  # noqa: E402,F401

UNITS = ("cm", "in")
_REAL = json.loads(DATA_FILE.read_text(encoding="utf-8"))


def real_data():
    """A private deep copy of the real file, so no test can leak edits into another."""
    return copy.deepcopy(_REAL)


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def usable_resorts(data):
    """The same filter app.main() applies."""
    return [r for r in data["resorts"] if isinstance(r, dict) and r.get("id")]


@pytest.fixture(scope="session")
def real():
    return _REAL


@pytest.fixture(scope="session")
def resorts():
    return usable_resorts(_REAL)


@pytest.fixture(scope="session")
def updated():
    return D.updated_utc(_REAL)


@pytest.fixture(scope="session")
def fresh():
    """'Updated just now', for wording tests that must not turn into the
    staleness warning when the suite runs days after the file was written.
    Staleness itself is tested with the file's own timestamp."""
    return datetime.now(timezone.utc)


def resort_ids():
    return [r["id"] for r in usable_resorts(_REAL)]


# ----------------------------------------------------------------------------
# Text helpers: visible text of rendered HTML
# ----------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def visible(h):
    """Text a reader sees: no tags, no svg/style bodies, no title attributes."""
    h = re.sub(r"<svg.*?</svg>|<style.*?</style>", " ", h, flags=re.S)
    h = re.sub(r'\stitle="[^"]*"', "", h)
    t = _html.unescape(_TAG.sub(" ", h))
    t = t.replace("\u202f", " ").replace("\u200b", "").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", t).strip()


def inline_text(h):
    """Visible text of a short inline fragment, without adding spaces at tags."""
    t = _html.unescape(_TAG.sub("", h))
    return t.replace("\u202f", " ").replace("\u200b", "").replace("\xa0", " ").strip()


LEAK = re.compile(r"\b(None|nan|NaN|inf|Infinity|null|undefined)\b")
