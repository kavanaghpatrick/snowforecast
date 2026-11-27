"""Is the US lead-5+ dryness in the previous-runs archive real or an archive artifact?

Two SNOTEL sites (Alta 1308:UT, Stevens Pass 791:WA), Jan-Mar 2026, precipitation
forecast 1, 3 and 5 days ahead, for best_match and the US models behind it.
If the same single model (gfs_global) loses most of its precipitation between
day 3 and day 5, the drop is in the archive, not in forecasting skill.
Output: data/validation/out/model_mix_check.json. One cached request.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, get  # noqa: E402

URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=40.5905,47.7461"
       "&longitude=-111.6375,-121.0929&elevation=2667,1201&start_date=2026-01-01&end_date=2026-03-31"
       "&hourly=precipitation_previous_day1,precipitation_previous_day3,precipitation_previous_day5"
       "&models=best_match,gfs_seamless,gfs_global,ncep_nbm_conus&timezone=GMT")


def main():
    body = get(URL, timeout=120)
    out = {}
    for name, p in zip(("Alta (2,667 m)", "Stevens Pass (1,201 m)"), body):
        h = p["hourly"]
        out[name] = {k: round(sum(x for x in v if x is not None), 1) for k, v in h.items() if k != "time"}
    (OUT / "model_mix_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
