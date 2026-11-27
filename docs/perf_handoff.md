# Performance handoff (fetcher speed-up, 27 Nov 2026)

The fetcher (`scripts/fetch_data_v2.py`) now runs its three fetch chains at the same time and fetches report pages four at a time. A full 86-resort replay with production-like latency went from 308 s to 108 s, and the written file is byte-identical to the old fetcher's once the run clock fields are blanked. Nothing below was done to files outside `scripts/`, `tests/fetcher_v2/` and `dashboard_v2/data.py`; the items in this note are for the owners of those other files.

## What changed in the fetcher (for the queue-plan owners)

- Three chains run in parallel and are merged in a fixed order: measured stations, then resort reports (OnTheSnow, then tenki.jp), then Open-Meteo. The merge order is the order a serial run used, so notes, warnings and `provider_status` come out in the same order as before.
- A new station provider (depth coverage plan) goes into `measured_chain` inside `_main()`, in the same `g("Name", fn, rs, started, warnings, provider_status)` form the others use. It must only write its own resort keys (`stations`, its notes); two chains must never set the same resort key.
- OnTheSnow and tenki.jp: the per-page loop body moved to `_onthesnow_pages()` and `_tenki_pages()`; the fetching is in `_fetch_pages()`. Parse, staleness and closed-resort logic is unchanged. The line numbers quoted in `docs/depth_coverage_plan.md` (around line 424, "onthesnow_all() (line 1797) ... lines 1825-1829") now point at `_onthesnow_pages()` instead. The plan's idea of copying `openFlag` into `rep` still works there unchanged.
- Report-page politeness: request starts are at least 1.1 s apart (setting `FETCH_REPORT_SPACING_S`, never below 1 s), with at most 4 in flight (`FETCH_REPORT_WORKERS`). Before, pages went one at a time with a 1 s sleep, so about one every 2.4 s. When four pages in a row time out, the next pages are not requested, as before.
- Open-Meteo is still one call at a time. The pause between its calls is now `FETCH_OM_PAUSE_S` (default 1 s, never lower). Its quota is per minute, so parallel calls would only add 429s.
- Meteocat's 1 s robots.txt crawl delay is untouched.
- `FETCH_PARALLEL_CHAINS=0` runs the chains one after another again, as a fallback.

The snowline and powder plans touch derive logic only and are not affected. The derive path got one change: hour keys are cached (`_hfmt`), which cut assemble time from 0.48 s to 0.18 s for 86 resorts with byte-identical output.

## For the expansion-plan owners: runtime tables need correcting

`docs/resort_expansion_plan.md` (line 13 and the Runtime paragraph near line 736) and `docs/resort_expansion_asia.md` (section 4, line 345 onward) project runtime from the 685 s run. That run used the fallback Open-Meteo host, whose daily phase took 401 s against about 13 s on the main host, so it is not a fair baseline. Normal runs on the main host took 257 to 278 s before this change.

Measured on a real local replay of all 86 resorts (recorded payloads, latency set to match production logs, one Open-Meteo 429 like every real fast run):

| Fetcher | Wall time |
|---|---|
| Before | 308 s |
| Chains serial, report pages concurrent | 224 s |
| After (default) | 108 s |

The wall time is now the Open-Meteo chain. Its floor at 212 resorts is set by the per-minute quota (about 2,600 to 3,100 weighted calls, so about 4.5 to 5 minutes), not by the other providers. Raising `timeout-minutes` to 35 is still sensible headroom; 45 is not needed. Use the fallback host only when the quota forces it, never as a timing reference.

## For the dashboard owner (`dashboard_v2/`, not applied by me)

1. Done in `data.py`: `_read_github()` now asks for gzip and decompresses it with a size cap. The 3.0 MB file is 279 KB gzipped. Checked against a real local server (plain, gzip, truncated gzip, oversized gzip and an unknown encoding) and against the real raw.githubusercontent.com, which does answer with gzip.
2. Not applied: `@st.cache_resource(ttl=300)` in place of `@st.cache_data` on `load()` would save about 7 ms per rerun (16 to 8 ms on a resort page). The object is then shared by every viewer, so it is only safe while no render function mutates the data. The profile found no mutation today, but render code is still changing. If you take it, add a test that renders every view and asserts the data is unchanged.
3. Not applied: `orjson` parses the file in 24 ms against 100 ms for `json.loads`, once per 5-minute cache miss. It needs a line in `dashboard_v2/requirements.txt` (prebuilt wheels, no build step on Streamlit Cloud). Optional; the saving is small. It is deliberately not used in the fetcher, which stays stdlib-only.

## Rust or other native code

Not justified. The fetcher spends under 1 s of a 260 s run on CPU, the dashboard reruns in 14 to 22 ms, and a native build would add a toolchain to both GitHub Actions and Streamlit Cloud for no measurable gain.
