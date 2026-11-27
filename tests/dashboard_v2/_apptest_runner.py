"""Run the real dashboard app.py through Streamlit's own AppTest runner.

    python3 _apptest_runner.py <app.py> '<json list of cases>'

Each case is {"params": {...query params...}}. Prints one JSON line: a list of
results with the exceptions raised, whether the honest error screen showed,
the markdown size, and any leaked None/nan in visible text. Called by
test_dv2_app.py in a subprocess whose HTTPS proxy is a local test server, so
the app's GitHub read never leaves the machine.
"""

import json
import logging
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from streamlit.testing.v1 import AppTest  # noqa: E402

LEAK = re.compile(r">[^<]*\b(None|nan|NaN|inf|Infinity)\b[^<]*<")


def run(app_path, case):
    at = AppTest.from_file(app_path, default_timeout=60)
    for k, v in case.get("params", {}).items():
        at.query_params[k] = v
    t = time.time()
    at.run()
    md = "\n".join(m.value for m in at.markdown)
    m = LEAK.search(md)
    return {
        "params": case.get("params", {}),
        "secs": round(time.time() - t, 2),
        "exceptions": [getattr(e, "message", str(e))[:400] for e in at.exception],
        "error_screen": "Forecast data" in md,
        "error_text": (re.search(r"Forecast data[^<]*", md) or [None])[0],
        "md_len": len(md),
        "leak": m.group(0)[:200] if m else None,
        "has_verdict": 'class="verdict' in md,
    }


if __name__ == "__main__":
    app, cases = sys.argv[1], json.loads(sys.argv[2])
    print(json.dumps([run(app, c) for c in cases]))
