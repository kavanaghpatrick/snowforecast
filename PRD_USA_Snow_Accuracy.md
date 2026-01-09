# PRD: USA Snow Data Accuracy Improvements

## Problem

Our SNOTEL-based snow depths differ 30-90% from resort-reported values:
- Mammoth: We show 231cm, resorts report 157cm base
- Stevens Pass: We show 91cm, resorts report 173cm mid-mountain
- Vail: We show 51cm, resorts report 76cm

**Root causes:**
1. SNOTEL measures snow settling/compaction, not fresh accumulation
2. Single station may not represent resort conditions
3. Station elevation/location differs from ski terrain

## Solution

### ~~Phase A: SWE-to-Snow Conversion~~ (ABANDONED)

**Finding:** SWE × SLR method is for estimating **fresh snowfall**, not base depth.
- SWE × SLR at Mammoth = 1040cm (uncompacted equivalent)
- Actual SNOWDEPTH = 231cm (settled snowpack) ← This is correct for base depth

**Why abandoned:** Variance goes BOTH directions:
- Mammoth: Our 231cm vs OnTheSnow 157cm (+47% HIGH)
- Stevens Pass: Our 91cm vs OnTheSnow 173cm (-47% LOW)
- Vail: Our 51cm vs OnTheSnow 76cm (-33% LOW)

The problem is **station location/elevation**, not measurement method.

---

### Phase B: Multi-Station Averaging (PROCEED)

**What:** Average 2-4 nearby SNOTEL stations instead of using single station.

**Why:** Reduces single-point measurement error. OpenSnow averages 4 stations within 6-14 miles.

**Data structure change:**

```python
# Before: single station
("Mammoth Mountain", 37.63, -119.03, "California", 3369, "574:CA:SNTL", "Mammoth Pass"),

# After: multiple stations with weights
("Mammoth Mountain", 37.63, -119.03, "California", 3369, [
    ("574:CA:SNTL", "Mammoth Pass", 1.0),      # Primary (closest)
    ("778:CA:SNTL", "Rock Creek", 0.7),        # Secondary
    ("539:CA:SNTL", "Kaiser Point", 0.5),      # Tertiary
]),
```

**Changes to `fetch_data_v2.py`:**

```python
def get_snotel_multi_station(stations):
    """Get weighted average snow depth from multiple SNOTEL stations."""
    depths = []
    for station_id, name, weight in stations:
        depth = get_snotel_snow_depth(station_id, name)
        if depth is not None:
            depths.append((depth, weight))

    if not depths:
        return None

    # Weighted average
    total = sum(d * w for d, w in depths)
    weights = sum(w for _, w in depths)
    return round(total / weights, 1)
```

**Scope:**
- Update `US_SKI_AREAS` tuples with station lists (~50 lines of config)
- Add `get_snotel_multi_station()` function (~20 lines)
- Modify `process_region()` to handle new format (~10 lines)

**Success metric:** Reduce variance vs resort reports to <20%

---

## Implementation Order

```
Phase B (multi-station averaging) → Test → Deploy
```

**Rationale:**
- Phase A abandoned - SWE method doesn't apply to base depth
- Phase B directly addresses location/elevation variance
- Each resort gets 2-4 stations weighted by proximity

## Research Required

### Phase B
- Identify 2-3 additional SNOTEL stations near each of 22 US resorts
- Calculate distance weights based on proximity to resort coordinates
- Estimated effort: 2 hours of SNOTEL map research

## Out of Scope

- Resort-reported data feeds (requires business relationships)
- Synoptic Data API integration (evaluate after A+B)
- New weather stations or hardware
- Real-time updates (daily refresh is sufficient)

## Risks

| Risk | Mitigation |
|------|------------|
| WTEQ data missing for some stations | Fall back to SNOWDEPTH |
| SLR estimation inaccurate | Use conservative 12:1 default |
| Multi-station slows fetch | Parallel requests (already have 0.3s delay) |

## Success Criteria

| Metric | Current | Phase B Target |
|--------|---------|----------------|
| Avg variance vs OnTheSnow | 50% | <25% |
| Stations with >40% variance | 5/22 | <2/22 |
| Data fetch time | ~3 min | ~5 min |

## Timeline

- Station research: 1-2 hours
- Implementation: 1 hour
- Testing: 1 hour
- Total: Half day
