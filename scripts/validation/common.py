"""Shared helpers for the model-validation backtest.

Everything fetched from the network is cached under data/validation/raw/ keyed by
URL, so every statistic in docs/model_validation_report.md can be recomputed
offline from the same bytes. Delete the cache to refetch.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "forecasts_v2.json"
VAL = ROOT / "data" / "validation"
RAW = VAL / "raw"
OUT = VAL / "out"
for p in (RAW, OUT):
    p.mkdir(parents=True, exist_ok=True)

UA = "snowforecast-validation/1 (+https://github.com/kavanaghpatrick/snowforecast)"

# Backtest windows (local calendar days, inclusive)
WINDOWS = {
    "W1_2025_26_winter": (date(2025, 12, 1), date(2026, 3, 31)),
    "W2_2026_early": (date(2026, 10, 15), date(2026, 11, 26)),
}

SNOW_CM_PER_MM = 0.7   # Open-Meteo's fixed frozen-water factor (7 cm per 10 mm)
RECOUNT_MIN_MM = 1.0   # dashboard_v2/data.py RECOUNT_MIN_MM


class Refused(Exception):
    """The server refused the request (quota / 4xx). Callers must stop, not retry."""


def _cache_path(url: str, suffix: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return RAW / "http" / f"{h}{suffix}"


def get(url: str, *, want="json", timeout=120, pause=0.0, attempts=2):
    """GET with an on-disk cache. Raises Refused on quota/4xx (no retry)."""
    suffix = ".json" if want == "json" else ".bin"
    cp = _cache_path(url, suffix)
    if cp.exists():
        raw = cp.read_bytes()
        return json.loads(raw) if want == "json" else raw
    cp.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            if want == "json":
                body = json.loads(raw)
                if isinstance(body, dict) and body.get("error"):
                    raise Refused(f"{body.get('reason')} :: {url[:160]}")
            cp.write_bytes(raw)
            (cp.with_suffix(cp.suffix + ".url")).write_text(url)
            if pause:
                time.sleep(pause)
            return json.loads(raw) if want == "json" else raw
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            if e.code in (400, 401, 403, 404, 429):
                raise Refused(f"HTTP {e.code} {body!r} :: {url[:160]}")
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed after {attempts} attempts: {last} :: {url[:160]}")


def load_product():
    return json.loads(DATA_FILE.read_text())


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
