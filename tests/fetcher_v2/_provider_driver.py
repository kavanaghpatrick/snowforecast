"""Run one provider function of the UNMODIFIED fetcher and dump what it produced.

Invoked in a subprocess by the tests, with HTTPS_PROXY / SSL_CERT_FILE (or
OPEN_METEO_URL) pointing at a local replay server, so the fetcher's own
network code runs against recorded payloads.

    python3 _provider_driver.py FUNC "Resort A,Resort B" RUN_UTC OUT.json

FUNC is a station batch (snotel_batch, meteocat_batch, ...) or "openmeteo_<kind>".
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_data_v2.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fetch_data_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_data_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    func, names, run_utc, out = sys.argv[1:5]
    f = load()
    run = datetime.fromisoformat(run_utc.replace("Z", "+00:00"))
    wanted = [n for n in names.split(",") if n]
    resorts = [r for r in f.build_resorts() if r["name"] in wanted]
    warnings, status = [], {}
    result = {}
    if func.startswith("openmeteo_"):
        kind = func.split("_", 1)[1]
        failed = []
        res = f.openmeteo_fetch(kind, resorts, failed, f._om_points)
        result = {"om": res, "failed_chunks": failed,
                  "om_counter": {"requests": f.OM_COUNTER.requests, "retries": f.OM_COUNTER.retries}}
    elif func in ("onthesnow_all", "tenki_all"):
        getattr(f, func)(resorts, warnings, status)
    else:
        getattr(f, func)(resorts, run, warnings, status)
    result.update({
        "warnings": warnings, "status": status,
        "resorts": [{"name": r["name"], "notes": r["notes"], "stations": r.get("stations", []),
                     "base_m": r["base_m"], "summit_m": r["summit_m"], "lat": r["lat"], "lon": r["lon"]}
                    for r in resorts],
    })
    Path(out).write_text(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
