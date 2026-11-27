"""Lead-time check on forecasts the product actually issued (git history).

The repo's git history holds data/forecasts_v2.json as committed by the refresh job
from 4 to 24 Jan 2026 (old pipeline: Open-Meteo /v1/gfs = HRRR 0-48 h + GFS,
daily snowfall_sum at the model cell, base resort coordinates, 7 days). For US
resorts with a SNOTEL station in the current file, this compares day-k forecasts
with SNOTEL depth gain on the valid day, and compares lead-to-lead totals over the
same valid days. It is an independent check on the previous-runs API result
(lead 5+ much drier than lead 1-4 in the US).
Output: data/validation/out/archived_forecasts.json
Usage: python3 scripts/validation/archived_forecasts.py  (needs obs_windows.csv)
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_product  # noqa: E402


def main():
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h %cI", "--", "data/forecasts_v2.json"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    snaps = {}
    for line in log:
        if not line.strip():
            continue
        h, ts = line.split()
        if not ("2026-01-04" <= ts[:10] <= "2026-01-24"):
            continue
        day = ts[:10]
        # keep the earliest commit of each day (log is newest first, so overwrite)
        snaps[day] = h
    prod = load_product()
    sn_by_name = {}
    for r in prod["resorts"]:
        s = [x for x in r.get("stations") or [] if x["provider"] == "SNOTEL"]
        if r["country"] == "USA" and s:
            sn_by_name[r["name"]] = f"SNOTEL:{s[0]['id']}"
    obs = defaultdict(dict)
    for o in csv.DictReader((OUT / "obs_windows.csv").open()):
        if o["provider"] == "SNOTEL" and o["dhs_cm"] not in ("", "None"):
            # local_day rows: t0 is local midnight of the day in UTC; the date is t0 + offset
            from datetime import datetime, timedelta
            t0 = datetime.fromisoformat(o["t0_utc"])
            d = (t0 + timedelta(hours=12)).date().isoformat()
            obs[f"SNOTEL:{o['station_id']}"][d] = max(float(o["dhs_cm"]), 0.0)
    fc = defaultdict(dict)  # (resort, valid) -> {lead: cm}
    for issue, h in sorted(snaps.items()):
        raw = subprocess.run(["git", "-C", str(ROOT), "show", f"{h}:data/forecasts_v2.json"],
                             capture_output=True, text=True, check=True).stdout
        d = json.loads(raw)
        for r in d.get("resorts", []):
            if r.get("name") not in sn_by_name:
                continue
            for f in r.get("forecast") or []:
                if f.get("new_snow_cm") is not None:
                    fc[(r["name"], f["date"])][f["day"]] = f["new_snow_cm"]
    out = {"issues": sorted(snaps), "resorts": sorted({k[0] for k in fc})}
    # same valid days at every lead 0..6
    full = {k: v for k, v in fc.items() if all(n in v for n in range(7))}
    out["n_resort_days_all_leads"] = len(full)
    out["sum_by_lead"] = {n: round(sum(v[n] for v in full.values()), 1) for n in range(7)}
    rows = []
    for (name, vd), v in full.items():
        m = obs.get(sn_by_name[name], {}).get(vd)
        if m is not None:
            rows.append((v, m))
    out["n_with_snotel"] = len(rows)
    out["snotel_sum"] = round(sum(m for _, m in rows), 1)
    out["by_lead"] = {}
    for n in range(7):
        x = np.array([v[n] for v, _ in rows])
        y = np.array([m for _, m in rows])
        out["by_lead"][n] = {"sum_fc": round(float(x.sum()), 1),
                             "ratio_meas_over_fc": round(float(y.sum() / x.sum()), 2) if x.sum() else None,
                             "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 2) if x.std() > 0 and y.std() > 0 else None,
                             "mae": round(float(np.mean(np.abs(x - y))), 2)}
    (OUT / "archived_forecasts.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
