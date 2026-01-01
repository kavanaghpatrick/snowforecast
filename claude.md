# CLAUDE.md - Snowforecast Project

> **This file extends `~/.claude/CLAUDE.md`**. All global rules (Grok reviews, simplicity protocol, AI tools) apply here.

---

## Project Overview

**Goal**: ML model predicting 24-hour snowfall and snow depth for mountain locations
**Success Metrics**: RMSE <15cm, F1 >85%, >20% improvement over baseline
**PRD**: `Product Requirements Document_ Mountain Snowfall Prediction Model.md`

---

## Quick Commands

```bash
# Setup parallel development environment
./scripts/setup_worktrees.sh

# Check all worktree status
./scripts/check_status.sh

# Spawn Phase 1 agents
/project:spawn-phase1

# Merge Phase 1 work
/project:merge-phase1

# Run tests for a specific pipeline
pytest tests/pipelines/test_{pipeline}.py -v

# Install with all dependencies
uv pip install -e ".[all]"
```

---

## AI Review Protocol (Project-Specific)

### After Implementing Any Pipeline

1. **Grok Review** (CRITICAL issues):
```bash
python3 << 'EOF'
import json
code = open('src/snowforecast/pipelines/{name}.py').read()
req = {"messages": [{"role": "user", "content": f"Review this data pipeline for CRITICAL issues (crashes, data corruption, security):\n\n{code}"}], "model": "grok-4", "temperature": 0}
json.dump(req, open('/tmp/req.json', 'w'))
EOF
curl -X POST https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $GROK_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json --max-time 300
```

2. **Gemini Review** (Architecture):
```bash
gemini -p "Review this data pipeline architecture for correctness and edge cases: $(cat src/snowforecast/pipelines/{name}.py)"
```

3. **Apply fixes** from reviews
4. **Re-test**: `pytest tests/pipelines/test_{name}.py -v`
5. **Then commit**

### Before Merging Any Phase

Run comprehensive review:
```bash
# Codex for autonomous review
codex exec --sandbox read-only "Review all Python files in src/snowforecast/ for bugs, security issues, and data integrity problems"

# Or parallel Grok + Gemini
gemini -p "Analyze architecture of $(ls src/snowforecast/pipelines/*.py)" &
python3 << 'EOF'
# ... Grok review of all files
EOF
```

---

## Parallel Development Workflow

### Architecture

```
COORDINATOR (main repo)           WORKERS (worktrees)
~/snowforecast/                   ~/snowforecast-worktrees/
├── Owns: setup, merges          ├── pipeline-snotel/  (Agent A)
├── Branch: main, develop        ├── pipeline-ghcn/    (Agent B)
└── Integration tests            ├── pipeline-era5/    (Agent C)
                                 ├── pipeline-hrrr/    (Agent D)
                                 ├── pipeline-dem/     (Agent E)
                                 └── pipeline-openskimap/ (Agent F)
```

### Phase 1 Parallelization (6 workers)

| Worktree | Branch | Issue | Agent Prompt |
|----------|--------|-------|--------------|
| `pipeline-snotel` | `phase1/2-snotel-pipeline` | #2 | `.claude/prompts/pipeline-snotel.md` |
| `pipeline-ghcn` | `phase1/3-ghcn-pipeline` | #3 | `.claude/prompts/pipeline-ghcn.md` |
| `pipeline-era5` | `phase1/4-era5-pipeline` | #4 | `.claude/prompts/pipeline-era5.md` |
| `pipeline-hrrr` | `phase1/5-hrrr-pipeline` | #5 | `.claude/prompts/pipeline-hrrr.md` |
| `pipeline-dem` | `phase1/6-dem-pipeline` | #6 | `.claude/prompts/pipeline-dem.md` |
| `pipeline-openskimap` | `phase1/7-openskimap-pipeline` | #7 | `.claude/prompts/pipeline-openskimap.md` |

### Worker Protocol

1. **Before Starting**:
   ```bash
   cd ~/snowforecast-worktrees/{your-worktree}
   git fetch origin && git rebase origin/develop
   ```

2. **File Ownership** (no conflicts):
   - Your pipeline: `src/snowforecast/pipelines/{name}.py`
   - Your tests: `tests/pipelines/test_{name}.py`
   - Your deps: `[project.optional-dependencies.{name}]` section only

3. **DO NOT MODIFY** (coordinator-owned):
   - `src/snowforecast/utils/*`
   - `pyproject.toml` core dependencies
   - `tests/conftest.py`
   - Any other pipeline's files

4. **Before Committing** (MANDATORY):
   - Run Grok review on your pipeline
   - Run tests: `pytest tests/pipelines/test_{name}.py -v`
   - Update `.claude/handoff.md`

5. **When Done**:
   - Push: `git push origin {branch}`
   - Mark GitHub issue as ready for review

---

## Handoff Document

Every worktree has `.claude/handoff.md` - UPDATE THIS:

```markdown
## Status: [In Progress | Complete | Blocked]
## Files: src/snowforecast/pipelines/{name}.py
## Tests: pytest tests/pipelines/test_{name}.py ✅
## Deps: {package}>=x.x.x added to optional-dependencies
## Grok Review: ✅ No critical issues
## Blocking: None
```

---

## Code Standards

### Pipeline Interface

All pipelines MUST implement:

```python
from snowforecast.utils.base import BasePipeline

class SnotelPipeline(BasePipeline):
    """SNOTEL data ingestion pipeline."""

    def download(self, start_date: str, end_date: str) -> Path:
        """Download raw data to data/raw/{name}/"""
        ...

    def process(self, raw_path: Path) -> pd.DataFrame:
        """Process raw data, return DataFrame"""
        ...

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate processed data"""
        ...
```

### Data Paths

```python
from snowforecast.utils.io import get_data_path

raw_path = get_data_path("snotel", "raw")      # data/raw/snotel/
processed_path = get_data_path("snotel", "processed")
```

---

## Merge Protocol

### After All Agents Complete

```bash
# Coordinator runs:
cd ~/snowforecast
git checkout develop

# Merge each pipeline
for branch in phase1/2-snotel-pipeline phase1/3-ghcn-pipeline \
              phase1/4-era5-pipeline phase1/5-hrrr-pipeline \
              phase1/6-dem-pipeline phase1/7-openskimap-pipeline; do
    git merge origin/$branch --no-ff -m "Merge ${branch##*/}"
done

# Run Grok review on merged code
python3 << 'EOF'
import json, glob
files = glob.glob('src/snowforecast/pipelines/*.py')
code = '\n\n'.join(open(f).read() for f in files)
req = {"messages": [{"role": "user", "content": f"Review all pipelines for integration issues:\n\n{code}"}], "model": "grok-4", "temperature": 0}
json.dump(req, open('/tmp/req.json', 'w'))
EOF
curl -X POST https://api.x.ai/v1/chat/completions -H "Authorization: Bearer $GROK_API_KEY" -H "Content-Type: application/json" -d @/tmp/req.json

# Run integration tests
pytest tests/ -v

# If passing, merge to main
git checkout main && git merge develop --no-ff
git push origin main develop
```

---

## Project-Specific Rules

### Data Integrity (from parent ~/CLAUDE.md)
- **NEVER** estimate, simulate, or extrapolate missing data values
- **NEVER** fill gaps with assumed or calculated values
- If data is missing, state clearly - do not substitute estimates

### Simplicity (from parent)
- Each pipeline is **one file, one class**
- No abstract base classes until 3+ implementations
- Hardcode Western US bounding box (no config files)

---

## Issue-to-Worktree Mapping

| Issue | Worktree | Milestone |
|-------|----------|-----------|
| #1 | setup | Phase 1 |
| #2-7 | pipeline-* | Phase 1 |
| #8 | (main) | Phase 1 |
| #9-18 | feature-engineering | Phase 2 |
| #19-23 | deep-learning | Phase 3 |
| #24-28 | evaluation | Phase 4 |

---

## Environment

```bash
# Already set in ~/.zshrc
export GROK_API_KEY="..."    # For Grok reviews
export CDS_API_KEY="..."     # For ERA5 Copernicus

# Python
python3 --version  # 3.11+
uv pip install -e ".[all]"
```

---

## Stuck Protocol

If stuck >5min on any pipeline:

1. **Launch parallel analysis**:
```bash
# Gemini: Why is this failing?
gemini -p "Why might this code be failing: $(cat problematic_code.py)"

# Grok: What's the fix?
# (use standard Grok API call)
```

2. Compare responses
3. Apply most conservative fix
4. If still stuck, ask human coordinator
