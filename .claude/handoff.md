# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Issue #15: Engineer lagged and rolling features

### Files Created/Modified

1. **LaggedFeatures Class** (Issue #15)
   - `src/snowforecast/features/__init__.py` - Package exports
   - `src/snowforecast/features/lagged.py` - LaggedFeatures class
   - `tests/features/__init__.py` - Test package
   - `tests/features/test_lagged.py` - 32 tests

### Implementation Summary

The `LaggedFeatures` class provides temporal feature engineering with the following capabilities:

**Features Created:**
- **Lagged values**: `{var}_lag_{n}d` - Values from N days ago
- **Rolling means**: `{var}_roll_mean_{n}d` - N-day moving averages
- **Rolling std**: `{var}_roll_std_{n}d` - N-day variability measures
- **Rolling min/max**: `{var}_roll_min_{n}d`, `{var}_roll_max_{n}d` - N-day extremes
- **Trend features**: `{var}_change_{n}d`, `{var}_pct_change_{n}d` - Changes over N days
- **Cumulative sums**: `{var}_cumsum_{n}d` - N-day cumulative sums (for precipitation)

**Key Design Decisions:**
- All operations are grouped by `station_id` to prevent data leakage between stations
- Default lags: [1, 2, 3, 7] days
- Default windows: [3, 7, 14] days
- NaN values at the start of each group are preserved (not filled)
- `compute_all()` method applies all feature types in one call
- `get_feature_names()` method returns all feature names without computing

### Tests

```bash
pytest tests/features/test_lagged.py -v
# Result: 32 passed
```

Test coverage includes:
- Initialization with default and custom parameters
- Basic feature creation for all 6 feature types
- Group-based computation (no data leakage between stations)
- Edge cases: empty DataFrames, NaN values, single rows, missing group column
- Original DataFrame preservation

### Dependencies

No additional dependencies required - uses only `pandas` and `numpy` from core dependencies.

## Outstanding Work
- None for Issue #15

## Notes for Integration
- The `LaggedFeatures` class is now exported from `snowforecast.features`
- Can be used with any DataFrame that has a `station_id` column (or custom group column)
- Designed to work with output from Phase 1 pipelines (SNOTEL, GHCN, ERA5, etc.)

## Example Usage

```python
from snowforecast.features import LaggedFeatures

lf = LaggedFeatures(default_lags=[1, 7], default_windows=[7, 14])

# Create all features for temperature and precipitation
df = lf.compute_all(df, ["temperature", "precip"], group_col="station_id")

# Or create specific features
df = lf.create_lags(df, ["temperature"], lags=[1, 3, 7])
df = lf.create_rolling_mean(df, ["precip"], windows=[7, 14, 30])
```
