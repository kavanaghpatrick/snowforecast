# Git Worktree Strategy for Parallel Development

## Overview

This document defines the git worktree strategy for parallel development of the snowforecast ML project with multiple Claude Code agents.

**Project Structure**: 28 issues across 4 phases
**Phase 1 Focus**: 6 independent data pipelines that can run in parallel after project setup

---

## 1. Directory Structure

### Worktree Layout

```
/Users/patrickkavanagh/
├── snowforecast/                    # Main repo (coordinator workspace)
│   ├── .git/                        # Shared git directory
│   ├── src/snowforecast/            # Package source
│   ├── data/                        # Shared data directory (gitignored)
│   ├── tests/                       # Test files
│   └── docs/                        # Documentation
│
├── snowforecast-worktrees/          # Parent directory for all worktrees
│   ├── setup/                       # Issue #1: Project setup
│   ├── pipeline-snotel/             # Issue #2: SNOTEL pipeline
│   ├── pipeline-ghcn/               # Issue #3: GHCN pipeline
│   ├── pipeline-era5/               # Issue #4: ERA5-Land pipeline
│   ├── pipeline-hrrr/               # Issue #5: HRRR pipeline
│   ├── pipeline-dem/                # Issue #6: DEM pipeline
│   ├── pipeline-openskimap/         # Issue #7: OpenSkiMap pipeline
│   ├── feature-engineering/         # Phase 2 work
│   └── integration/                 # Integration testing worktree
```

### Data Directory Structure (Shared via Symlink)

```
/Users/patrickkavanagh/snowforecast/data/
├── raw/
│   ├── snotel/                      # SNOTEL raw downloads
│   ├── ghcn/                        # GHCN raw downloads
│   ├── era5/                        # ERA5-Land raw downloads
│   ├── hrrr/                        # HRRR raw downloads
│   ├── dem/                         # DEM tiles
│   └── openskimap/                  # OpenSkiMap exports
├── processed/
│   ├── snotel/
│   ├── ghcn/
│   ├── era5/
│   ├── hrrr/
│   ├── dem/
│   └── openskimap/
└── cache/                           # Temporary cache files
```

---

## 2. Branch Naming Strategy

### Format
```
{phase}/{issue-number}-{short-description}
```

### Phase 1 Branches (Data Pipelines)

| Branch Name | Issue | Owner Agent |
|------------|-------|-------------|
| `phase1/1-project-setup` | #1 | Coordinator |
| `phase1/2-snotel-pipeline` | #2 | Agent-A |
| `phase1/3-ghcn-pipeline` | #3 | Agent-B |
| `phase1/4-era5-pipeline` | #4 | Agent-C |
| `phase1/5-hrrr-pipeline` | #5 | Agent-D |
| `phase1/6-dem-pipeline` | #6 | Agent-E |
| `phase1/7-openskimap-pipeline` | #7 | Agent-F |

### Integration Branches

| Branch Name | Purpose |
|-------------|---------|
| `develop` | Integration branch for completed features |
| `phase1/integration` | Phase 1 integration testing |
| `main` | Stable releases only |

---

## 3. Setup Commands

### Initial Setup (Run Once by Coordinator)

```bash
# Navigate to project root
cd /Users/patrickkavanagh/snowforecast

# Create worktrees parent directory
mkdir -p /Users/patrickkavanagh/snowforecast-worktrees

# Create develop branch for integration
git checkout -b develop
git push -u origin develop
git checkout main

# Create shared data directory structure
mkdir -p data/{raw,processed,cache}/{snotel,ghcn,era5,hrrr,dem,openskimap}

# Add data to gitignore
echo "data/" >> .gitignore
git add .gitignore
git commit -m "Add data directory to gitignore"

# Create base project structure
mkdir -p src/snowforecast/{pipelines,utils,models,features}
mkdir -p tests/{pipelines,utils,models,features}
touch src/snowforecast/__init__.py
touch src/snowforecast/pipelines/__init__.py
touch src/snowforecast/utils/__init__.py

# Commit base structure
git add .
git commit -m "Add base project structure for parallel development"
git push origin main
```

### Creating Worktrees for Phase 1 Pipelines

```bash
# Create all branches first (from main)
cd /Users/patrickkavanagh/snowforecast

for issue in "1-project-setup" "2-snotel-pipeline" "3-ghcn-pipeline" \
             "4-era5-pipeline" "5-hrrr-pipeline" "6-dem-pipeline" \
             "7-openskimap-pipeline"; do
    git branch "phase1/${issue}" main
done

# Create worktrees
WORKTREE_BASE="/Users/patrickkavanagh/snowforecast-worktrees"

git worktree add "${WORKTREE_BASE}/setup" phase1/1-project-setup
git worktree add "${WORKTREE_BASE}/pipeline-snotel" phase1/2-snotel-pipeline
git worktree add "${WORKTREE_BASE}/pipeline-ghcn" phase1/3-ghcn-pipeline
git worktree add "${WORKTREE_BASE}/pipeline-era5" phase1/4-era5-pipeline
git worktree add "${WORKTREE_BASE}/pipeline-hrrr" phase1/5-hrrr-pipeline
git worktree add "${WORKTREE_BASE}/pipeline-dem" phase1/6-dem-pipeline
git worktree add "${WORKTREE_BASE}/pipeline-openskimap" phase1/7-openskimap-pipeline

# Create symlinks to shared data directory in each worktree
for worktree in setup pipeline-snotel pipeline-ghcn pipeline-era5 \
                pipeline-hrrr pipeline-dem pipeline-openskimap; do
    ln -s /Users/patrickkavanagh/snowforecast/data "${WORKTREE_BASE}/${worktree}/data"
done
```

### Verify Worktree Setup

```bash
git worktree list
# Should show:
# /Users/patrickkavanagh/snowforecast                              main
# /Users/patrickkavanagh/snowforecast-worktrees/setup              phase1/1-project-setup
# /Users/patrickkavanagh/snowforecast-worktrees/pipeline-snotel    phase1/2-snotel-pipeline
# ... etc
```

---

## 4. Code Organization to Minimize Conflicts

### File Ownership Matrix

| File/Directory | Owner | Conflict Risk |
|----------------|-------|---------------|
| `src/snowforecast/pipelines/snotel.py` | Agent-A | LOW |
| `src/snowforecast/pipelines/ghcn.py` | Agent-B | LOW |
| `src/snowforecast/pipelines/era5.py` | Agent-C | LOW |
| `src/snowforecast/pipelines/hrrr.py` | Agent-D | LOW |
| `src/snowforecast/pipelines/dem.py` | Agent-E | LOW |
| `src/snowforecast/pipelines/openskimap.py` | Agent-F | LOW |
| `src/snowforecast/pipelines/__init__.py` | Coordinator | MEDIUM |
| `src/snowforecast/utils/io.py` | Coordinator | HIGH |
| `src/snowforecast/utils/geo.py` | Coordinator | HIGH |
| `pyproject.toml` | Coordinator | HIGH |
| `tests/conftest.py` | Coordinator | HIGH |

### High-Conflict Files: Resolution Strategy

#### pyproject.toml

**Problem**: Multiple agents need different dependencies.

**Solution**: Use a modular dependency structure:

```toml
[project]
name = "snowforecast"
version = "0.1.0"
dependencies = [
    # Core dependencies only - managed by coordinator
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "xarray>=2023.1.0",
]

[project.optional-dependencies]
# Each pipeline has its own optional deps
snotel = ["metloom>=0.3.0"]
ghcn = ["noaa-coops>=0.2.0"]
era5 = ["cdsapi>=0.6.0", "ecmwf-api-client>=1.6.0"]
hrrr = ["herbie-data>=2024.1.0"]
dem = ["rasterio>=1.3.0", "rioxarray>=0.14.0"]
openskimap = ["geojson>=3.0.0", "shapely>=2.0.0"]

# Development dependencies
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0"]

# All pipelines
all = [
    "snowforecast[snotel,ghcn,era5,hrrr,dem,openskimap,dev]"
]
```

**Coordination Protocol**:
1. Each agent adds dependencies to their section ONLY
2. Coordinator merges `pyproject.toml` changes from all branches before final integration
3. Use `uv pip compile` to generate lockfile after merge

#### Shared Utilities (utils/)

**Problem**: Multiple pipelines need common utilities.

**Solution**: Pre-define utility interfaces during setup phase:

```python
# src/snowforecast/utils/io.py - Defined by coordinator BEFORE parallel work

from pathlib import Path
from typing import Protocol

class DataWriter(Protocol):
    """Protocol for data writers - implement in pipeline modules."""
    def write(self, data, path: Path) -> None: ...

def get_data_path(pipeline: str, stage: str = "raw") -> Path:
    """Get standardized data path for a pipeline.

    Args:
        pipeline: One of 'snotel', 'ghcn', 'era5', 'hrrr', 'dem', 'openskimap'
        stage: One of 'raw', 'processed', 'cache'

    Returns:
        Path to the data directory
    """
    base = Path(__file__).parent.parent.parent.parent / "data"
    return base / stage / pipeline
```

```python
# src/snowforecast/utils/geo.py - Defined by coordinator BEFORE parallel work

from dataclasses import dataclass

@dataclass
class BoundingBox:
    """Geographic bounding box."""
    west: float
    south: float
    east: float
    north: float

@dataclass
class Point:
    """Geographic point."""
    lat: float
    lon: float
    elevation: float | None = None

# Western US bounding box (shared reference)
WESTERN_US_BBOX = BoundingBox(west=-125.0, south=31.0, east=-102.0, north=49.0)
```

### Test File Organization

**Structure**: Each pipeline has isolated tests.

```
tests/
├── conftest.py                      # Shared fixtures (coordinator-owned)
├── pipelines/
│   ├── test_snotel.py               # Agent-A only
│   ├── test_ghcn.py                 # Agent-B only
│   ├── test_era5.py                 # Agent-C only
│   ├── test_hrrr.py                 # Agent-D only
│   ├── test_dem.py                  # Agent-E only
│   └── test_openskimap.py           # Agent-F only
└── integration/
    └── test_pipeline_integration.py # Coordinator only (after merge)
```

**Shared Fixtures** (`conftest.py` - coordinator defines before parallel work):

```python
import pytest
from pathlib import Path

@pytest.fixture
def data_dir() -> Path:
    """Root data directory."""
    return Path(__file__).parent.parent / "data"

@pytest.fixture
def sample_stations() -> list[dict]:
    """Sample SNOTEL/GHCN station data for testing."""
    return [
        {"id": "1050", "name": "Snotel Station 1", "lat": 40.5, "lon": -111.5, "elev": 2500},
        {"id": "1051", "name": "Snotel Station 2", "lat": 41.0, "lon": -112.0, "elev": 2800},
    ]

@pytest.fixture
def sample_bbox() -> dict:
    """Sample bounding box for testing."""
    return {"west": -112.0, "south": 40.0, "east": -111.0, "north": 41.0}
```

---

## 5. Merge/Integration Strategy

### Merge Order (Phase 1)

```
main
  └── phase1/1-project-setup (FIRST - creates structure)
        └── develop (merge setup first)
              ├── phase1/2-snotel-pipeline
              ├── phase1/3-ghcn-pipeline
              ├── phase1/4-era5-pipeline
              ├── phase1/5-hrrr-pipeline
              ├── phase1/6-dem-pipeline
              └── phase1/7-openskimap-pipeline
                    └── phase1/integration (test all together)
                          └── main (release)
```

### Integration Protocol

#### Step 1: Merge Project Setup First

```bash
cd /Users/patrickkavanagh/snowforecast
git checkout develop
git merge phase1/1-project-setup --no-ff -m "Merge #1: Project setup complete"
git push origin develop
```

#### Step 2: Rebase All Pipeline Branches on develop

```bash
# Each agent runs this in their worktree BEFORE final work
cd /Users/patrickkavanagh/snowforecast-worktrees/pipeline-snotel
git fetch origin
git rebase origin/develop
```

#### Step 3: Merge Pipelines (Coordinator)

```bash
cd /Users/patrickkavanagh/snowforecast
git checkout develop

# Merge each pipeline (order doesn't matter - they're independent)
for branch in phase1/2-snotel-pipeline phase1/3-ghcn-pipeline \
              phase1/4-era5-pipeline phase1/5-hrrr-pipeline \
              phase1/6-dem-pipeline phase1/7-openskimap-pipeline; do
    git merge ${branch} --no-ff -m "Merge ${branch##*/}"
done

# Resolve any pyproject.toml conflicts by combining sections
git push origin develop
```

#### Step 4: Integration Testing

```bash
# Create integration worktree
git worktree add /Users/patrickkavanagh/snowforecast-worktrees/integration develop

cd /Users/patrickkavanagh/snowforecast-worktrees/integration

# Install all dependencies
uv pip install -e ".[all]"

# Run all tests
pytest tests/ -v

# If tests pass, merge to main
git checkout main
git merge develop --no-ff -m "Phase 1 complete: All data pipelines"
git push origin main
```

---

## 6. Agent Handoff Protocol

### Handoff Document Template

Each agent must maintain a handoff document at:
```
/Users/patrickkavanagh/snowforecast-worktrees/{worktree}/.claude/handoff.md
```

```markdown
# Agent Handoff: [Pipeline Name]

## Current Status
- [ ] In Progress / [x] Complete / [ ] Blocked

## Task Definition
Issue #X: [Title]
Branch: phase1/X-pipeline-name

## Files Created/Modified
- `src/snowforecast/pipelines/name.py` - Main pipeline module
- `tests/pipelines/test_name.py` - Unit tests
- `docs/pipelines/name.md` - Usage documentation

## Dependencies Added
- `packagename>=1.0.0` (in [project.optional-dependencies.name])

## Tests Status
- [x] Unit tests pass: `pytest tests/pipelines/test_name.py`
- [ ] Integration tests: N/A (pending merge)

## Outstanding Work
- None / List items

## Blocking Items
- None / List dependencies on other pipelines

## Notes for Next Agent
- Any gotchas, API rate limits, auth requirements, etc.
```

### Handoff Checklist

Before completing work on a worktree:

1. **Code Complete**
   - [ ] All pipeline code in `src/snowforecast/pipelines/{name}.py`
   - [ ] Exports added to `src/snowforecast/pipelines/__init__.py`
   - [ ] Type hints on all public functions

2. **Tests Pass**
   - [ ] Run `pytest tests/pipelines/test_{name}.py -v`
   - [ ] Coverage > 80%

3. **Documentation**
   - [ ] Docstrings on all public functions
   - [ ] Usage example in module docstring

4. **Dependencies**
   - [ ] Added to correct section in `pyproject.toml`
   - [ ] Version pinned with minimum version

5. **Git State**
   - [ ] All changes committed
   - [ ] Branch pushed to origin
   - [ ] Handoff document updated

6. **Data Artifacts**
   - [ ] Sample data (if any) in correct `data/` subdirectory
   - [ ] No large files committed to git

---

## 7. Conflict Resolution Procedures

### pyproject.toml Conflicts

When merging branches with conflicting `pyproject.toml`:

```bash
# During merge conflict
git checkout --ours pyproject.toml   # Keep current version
git show MERGE_HEAD:pyproject.toml > theirs.toml  # Get their version

# Manually combine the [project.optional-dependencies] sections
# Each pipeline's section should be independent

# After combining
git add pyproject.toml
git commit
```

### __init__.py Conflicts

```python
# Resolve by combining all imports
# Before (conflict):
<<<<<<< HEAD
from .snotel import SnotelPipeline
=======
from .ghcn import GHCNPipeline
>>>>>>> phase1/3-ghcn-pipeline

# After (resolved):
from .snotel import SnotelPipeline
from .ghcn import GHCNPipeline
```

### Shared Utility Conflicts

**Prevention**: Coordinator defines all shared utilities BEFORE parallel work begins.

**If conflict occurs**:
1. Keep the more general implementation
2. Add pipeline-specific logic to the pipeline module, not utils
3. Use composition over modification

---

## 8. Terminal/Session Management

### Terminal Title Convention

Each agent should set terminal title for visibility:

```bash
# In each worktree
echo -ne "\033]0;Snow: pipeline-snotel (Agent-A)\007"
```

### tmux Session Layout (Recommended)

```bash
# Create session with all worktrees
tmux new-session -s snowforecast -n main -d
tmux send-keys -t snowforecast:main "cd /Users/patrickkavanagh/snowforecast" Enter

# Create window for each pipeline
for wt in setup pipeline-snotel pipeline-ghcn pipeline-era5 \
          pipeline-hrrr pipeline-dem pipeline-openskimap; do
    tmux new-window -t snowforecast -n ${wt}
    tmux send-keys -t snowforecast:${wt} "cd /Users/patrickkavanagh/snowforecast-worktrees/${wt}" Enter
done

tmux attach -t snowforecast
```

---

## 9. Quick Reference Commands

### List All Worktrees
```bash
git worktree list
```

### Switch Agent to Different Worktree
```bash
cd /Users/patrickkavanagh/snowforecast-worktrees/pipeline-{name}
```

### Sync with Latest develop
```bash
git fetch origin
git rebase origin/develop
```

### Check Status Across All Worktrees
```bash
for wt in /Users/patrickkavanagh/snowforecast-worktrees/*/; do
    echo "=== $(basename $wt) ==="
    git -C "$wt" status -s
done
```

### Remove Completed Worktree
```bash
git worktree remove /Users/patrickkavanagh/snowforecast-worktrees/pipeline-snotel
git branch -d phase1/2-snotel-pipeline  # After merge
```

---

## 10. Phase 1 Execution Sequence

### Week 1: Setup (Coordinator Only)
1. Execute initial setup commands
2. Create all worktrees
3. Define shared utilities in `phase1/1-project-setup`
4. Merge setup to develop
5. Notify agents: "Ready for parallel work"

### Weeks 2-3: Parallel Pipeline Development
- **Agent-A**: SNOTEL pipeline in `pipeline-snotel` worktree
- **Agent-B**: GHCN pipeline in `pipeline-ghcn` worktree
- **Agent-C**: ERA5-Land pipeline in `pipeline-era5` worktree
- **Agent-D**: HRRR pipeline in `pipeline-hrrr` worktree
- **Agent-E**: DEM pipeline in `pipeline-dem` worktree
- **Agent-F**: OpenSkiMap pipeline in `pipeline-openskimap` worktree

### Week 4: Integration (Coordinator)
1. Collect handoff documents from all agents
2. Merge all branches to develop
3. Resolve conflicts
4. Run integration tests
5. Merge to main
6. Clean up worktrees

---

## Appendix: Complete Setup Script

Save as `scripts/setup_worktrees.sh`:

```bash
#!/bin/bash
set -e

PROJECT_ROOT="/Users/patrickkavanagh/snowforecast"
WORKTREE_BASE="/Users/patrickkavanagh/snowforecast-worktrees"

echo "Setting up snowforecast parallel development environment..."

cd "$PROJECT_ROOT"

# Create worktrees directory
mkdir -p "$WORKTREE_BASE"

# Create develop branch
git checkout -b develop 2>/dev/null || git checkout develop
git push -u origin develop 2>/dev/null || true
git checkout main

# Create data directory structure
mkdir -p data/{raw,processed,cache}/{snotel,ghcn,era5,hrrr,dem,openskimap}
echo "data/" >> .gitignore 2>/dev/null || true

# Create base project structure
mkdir -p src/snowforecast/{pipelines,utils,models,features}
mkdir -p tests/{pipelines,utils,models,features,integration}
touch src/snowforecast/__init__.py
touch src/snowforecast/{pipelines,utils,models,features}/__init__.py

# Create branches
BRANCHES=(
    "1-project-setup"
    "2-snotel-pipeline"
    "3-ghcn-pipeline"
    "4-era5-pipeline"
    "5-hrrr-pipeline"
    "6-dem-pipeline"
    "7-openskimap-pipeline"
)

WORKTREE_NAMES=(
    "setup"
    "pipeline-snotel"
    "pipeline-ghcn"
    "pipeline-era5"
    "pipeline-hrrr"
    "pipeline-dem"
    "pipeline-openskimap"
)

for i in "${!BRANCHES[@]}"; do
    branch="phase1/${BRANCHES[$i]}"
    worktree="${WORKTREE_NAMES[$i]}"

    # Create branch if it doesn't exist
    git branch "$branch" main 2>/dev/null || true

    # Create worktree if it doesn't exist
    if [ ! -d "$WORKTREE_BASE/$worktree" ]; then
        git worktree add "$WORKTREE_BASE/$worktree" "$branch"
    fi

    # Create data symlink
    if [ ! -L "$WORKTREE_BASE/$worktree/data" ]; then
        ln -s "$PROJECT_ROOT/data" "$WORKTREE_BASE/$worktree/data"
    fi

    # Create .claude directory for handoff docs
    mkdir -p "$WORKTREE_BASE/$worktree/.claude"
done

echo ""
echo "Setup complete! Worktrees created:"
git worktree list
echo ""
echo "Next steps:"
echo "1. Work on phase1/1-project-setup first (define shared utilities)"
echo "2. Merge setup to develop"
echo "3. Agents can then work in parallel on pipelines 2-7"
```

Make executable:
```bash
chmod +x scripts/setup_worktrees.sh
```
