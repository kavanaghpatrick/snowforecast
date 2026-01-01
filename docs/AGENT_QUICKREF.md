# Agent Quick Reference Card

## Your Worktree

| Agent | Worktree Path | Branch | Issue |
|-------|---------------|--------|-------|
| Coordinator | `/Users/patrickkavanagh/snowforecast` | `main`/`develop` | All |
| Setup | `/Users/patrickkavanagh/snowforecast-worktrees/setup` | `phase1/1-project-setup` | #1 |
| Agent-A | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-snotel` | `phase1/2-snotel-pipeline` | #2 |
| Agent-B | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-ghcn` | `phase1/3-ghcn-pipeline` | #3 |
| Agent-C | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-era5` | `phase1/4-era5-pipeline` | #4 |
| Agent-D | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-hrrr` | `phase1/5-hrrr-pipeline` | #5 |
| Agent-E | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-dem` | `phase1/6-dem-pipeline` | #6 |
| Agent-F | `/Users/patrickkavanagh/snowforecast-worktrees/pipeline-openskimap` | `phase1/7-openskimap-pipeline` | #7 |

---

## Commands You Need

### Before Starting Work

```bash
# Go to your worktree
cd /Users/patrickkavanagh/snowforecast-worktrees/pipeline-{yourname}

# Sync with latest develop (do this first!)
git fetch origin
git rebase origin/develop
```

### Your Files (ONLY modify these)

```
src/snowforecast/pipelines/{yourpipeline}.py    # Main code
tests/pipelines/test_{yourpipeline}.py          # Tests
```

### Shared Files (DO NOT modify)

```
src/snowforecast/utils/io.py          # Use get_data_path()
src/snowforecast/utils/geo.py         # Use Point, BoundingBox
tests/conftest.py                     # Use fixtures
pyproject.toml                        # Only add to YOUR section
```

### Running Tests

```bash
# Your tests only
pytest tests/pipelines/test_{yourpipeline}.py -v

# With coverage
pytest tests/pipelines/test_{yourpipeline}.py --cov=src/snowforecast/pipelines/{yourpipeline}
```

### When Done

```bash
# Commit your changes
git add .
git commit -m "feat(pipeline): Implement {yourpipeline} data pipeline

- Add data fetching from {source}
- Add processing and validation
- Add unit tests

Closes #{issue_number}"

# Push to origin
git push -u origin phase1/{issue}-{yourpipeline}-pipeline

# Update handoff document
nano .claude/handoff.md
```

---

## Pipeline Implementation Checklist

```
[ ] 1. Create pipeline module: src/snowforecast/pipelines/{name}.py
[ ] 2. Add class/functions for:
    [ ] Fetching data from source
    [ ] Validating downloaded data
    [ ] Processing/cleaning data
    [ ] Saving to data/{stage}/{pipeline}/
[ ] 3. Add docstrings with usage examples
[ ] 4. Create test file: tests/pipelines/test_{name}.py
[ ] 5. Add dependencies to pyproject.toml [project.optional-dependencies.{name}]
[ ] 6. Run tests: pytest tests/pipelines/test_{name}.py -v
[ ] 7. Update .claude/handoff.md
[ ] 8. Commit and push
```

---

## Data Path Convention

Use the shared utility:

```python
from snowforecast.utils.io import get_data_path

# Get your pipeline's data directory
raw_dir = get_data_path("snotel", "raw")        # data/raw/snotel/
processed_dir = get_data_path("snotel", "processed")  # data/processed/snotel/
cache_dir = get_data_path("snotel", "cache")    # data/cache/snotel/
```

**NEVER** hardcode paths. Always use `get_data_path()`.

---

## Adding Dependencies

Edit `pyproject.toml` and add to YOUR section only:

```toml
[project.optional-dependencies]
# Only edit this section if you're Agent-A working on SNOTEL
snotel = [
    "metloom>=0.3.0",
    "your-new-package>=1.0.0",  # Add your packages here
]
```

**DO NOT** edit other agents' sections or core dependencies.

---

## Common Issues

### "Module not found" when importing snowforecast

```bash
# Install in development mode
uv pip install -e ".[{yourpipeline}]"
# Example: uv pip install -e ".[snotel]"
```

### Tests fail due to missing shared utilities

Wait for coordinator to merge `phase1/1-project-setup` to develop, then:

```bash
git fetch origin
git rebase origin/develop
```

### Merge conflicts in pyproject.toml

Tell coordinator. They will merge your branch after resolving conflicts.

### Need to modify shared utilities

**DON'T.** Instead:
1. Add your logic to your pipeline module
2. Document the need in your handoff
3. Coordinator will evaluate for future shared utilities

---

## Terminal Title (for visibility)

```bash
echo -ne "\033]0;Snow: {your-worktree}\007"
```

---

## Quick Status Check

```bash
# In your worktree
git status
git log --oneline -5
cat .claude/handoff.md
```
