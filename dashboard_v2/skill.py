"""What the backtest says about how far to trust a number (docs/model_improvements.md).

Nothing here changes a model number. It only answers two display questions:

- likely_range(): the disclosed "likely" range beside the 3-day hero, from the
  backtest's measured / forecast spread (skill_ranges.json, built by
  scripts/validation/hero_ranges.py). The hero itself stays the model's figure.
- rough_from_index(): the first day whose amount the backtest found to be a
  rough guide only (correlation 0.31 at 5 days ahead, about 0.12 at 6 and 7).
"""

from __future__ import annotations

import json
from pathlib import Path

RANGES_PATH = Path(__file__).resolve().parent / "skill_ranges.json"

# Days ahead from which day-by-day amounts carry little skill (report section 3:
# correlation 0.48 at day 4, 0.31 at day 5, 0.13 at day 6).
ROUGH_FROM_INDEX = 5


def _load():
    try:
        t = json.loads(RANGES_PATH.read_text())
    except (OSError, ValueError):
        return None
    return t if isinstance(t, dict) and isinstance(t.get("countries"), dict) else None


_TABLE = _load()


def likely_range(country, value_cm):
    """(low_cm, high_cm) for a 3-day forecast of value_cm in this country, or None.

    None when the country was never backtested (Switzerland, Austria, Japan),
    when the forecast is under the table's floor (5 cm, where a quarter of real
    3-day blocks measured nothing), or when the table is missing. Never borrows
    another country's misses.
    """
    if _TABLE is None or not isinstance(value_cm, (int, float)):
        return None
    if value_cm < _TABLE.get("min_forecast_cm", 5.0):
        return None
    for row in _TABLE["countries"].get(country) or []:
        lo, hi = row.get("min_cm"), row.get("max_cm")
        if value_cm >= lo and (hi is None or value_cm < hi):
            return value_cm * row["lo_ratio"], value_cm * row["hi_ratio"]
    return None
