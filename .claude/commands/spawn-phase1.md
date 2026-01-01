# Spawn Phase 1 Agents

Spawn 6 parallel agents to work on the Phase 1 data pipelines.

## Prerequisites Check

First verify the worktrees are set up:
```bash
git worktree list
```

If not set up, run: `./scripts/setup_worktrees.sh`

## Spawn Agents

Launch these 6 agents in parallel using the Task tool:

### Agent A: SNOTEL Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-snotel
Branch: phase1/2-snotel-pipeline
Issue: #2

Task: Implement SNOTEL data ingestion using metloom library.
- Read .claude/prompts/pipeline-snotel.md for detailed instructions
- Create src/snowforecast/pipelines/snotel.py
- Create tests/pipelines/test_snotel.py
- Add metloom to [project.optional-dependencies.snotel]
- Update .claude/handoff.md when complete
```

### Agent B: GHCN Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-ghcn
Branch: phase1/3-ghcn-pipeline
Issue: #3

Task: Implement GHCN-Daily data ingestion.
- Read .claude/prompts/pipeline-ghcn.md for detailed instructions
- Download from NOAA NCEI
- Handle GHCN quality flags
- Update .claude/handoff.md when complete
```

### Agent C: ERA5-Land Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-era5
Branch: phase1/4-era5-pipeline
Issue: #4

Task: Implement ERA5-Land reanalysis data ingestion using cdsapi.
- Read .claude/prompts/pipeline-era5.md for detailed instructions
- Handle CDS API queue system
- Store as NetCDF with efficient chunking
- Update .claude/handoff.md when complete
```

### Agent D: HRRR Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-hrrr
Branch: phase1/5-hrrr-pipeline
Issue: #5

Task: Implement HRRR archive data ingestion using herbie library.
- Read .claude/prompts/pipeline-hrrr.md for detailed instructions
- Access HRRR on AWS Open Data
- Handle GRIB2 format
- Update .claude/handoff.md when complete
```

### Agent E: DEM Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-dem
Branch: phase1/6-dem-pipeline
Issue: #6

Task: Implement Copernicus DEM terrain data pipeline.
- Read .claude/prompts/pipeline-dem.md for detailed instructions
- Download GLO-30 DEM tiles
- Calculate slope, aspect, TRI, TPI
- Update .claude/handoff.md when complete
```

### Agent F: OpenSkiMap Pipeline
```
Working directory: ~/snowforecast-worktrees/pipeline-openskimap
Branch: phase1/7-openskimap-pipeline
Issue: #7

Task: Extract ski resort locations from OpenSkiMap.
- Read .claude/prompts/pipeline-openskimap.md for detailed instructions
- Parse GeoJSON/OSM data
- Extract base/summit elevations
- Update .claude/handoff.md when complete
```

## After Spawning

Monitor progress with:
```bash
./scripts/check_status.sh
```

When all complete, run `/project:merge-phase1`
