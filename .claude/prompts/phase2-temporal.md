# Phase 2 Agent Prompt: Temporal Alignment

## Issue #10: Implement temporal alignment and resampling

**Branch**: phase2/10-temporal
**Worktree**: ~/snowforecast-worktrees/phase2-temporal

## Objective

Align all data sources to common temporal resolution and ensure proper time zone handling.

## Context

Different data sources have different temporal resolutions:
- SNOTEL: Hourly observations
- GHCN: Daily observations
- ERA5-Land: Hourly (reanalysis)
- HRRR: Hourly forecasts

Target resolution: **Daily** (matching GHCN and practical prediction horizon)

## Tasks

1. Create `src/snowforecast/features/temporal.py`:
   ```python
   class TemporalAligner:
       """Aligns data to common temporal resolution."""

       def __init__(self, target_freq: str = "1D", timezone: str = "UTC"):
           """Initialize with target frequency and timezone."""

       def resample_hourly_to_daily(self, df: pd.DataFrame,
                                     agg_methods: dict = None) -> pd.DataFrame:
           """Resample hourly data to daily.

           Default agg_methods:
           - temperature: mean
           - precipitation: sum
           - snow_depth: mean (or max)
           - wind: mean
           """

       def align_to_date_range(self, df: pd.DataFrame,
                                start_date: str, end_date: str) -> pd.DataFrame:
           """Ensure DataFrame covers full date range (fill missing dates with NaN)."""

       def localize_to_utc(self, df: pd.DataFrame,
                            source_tz: str = None) -> pd.DataFrame:
           """Convert timestamps to UTC."""

       def create_date_index(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
           """Create complete daily date index."""
   ```

2. Create `tests/features/test_temporal.py` with 10+ tests

## Implementation Notes

1. All timestamps should be in UTC
2. Daily aggregation:
   - Temperature: daily mean
   - Precipitation/snowfall: daily sum
   - Snow depth: daily mean or end-of-day value
   - Wind: daily mean magnitude
3. NEVER interpolate missing data - use NaN
4. Handle DST transitions properly

## File Ownership

You own:
- `src/snowforecast/features/temporal.py`
- `tests/features/test_temporal.py`

## When Complete

1. Update `.claude/handoff.md`
2. Commit referencing Issue #10
3. Push to origin/phase2/10-temporal
