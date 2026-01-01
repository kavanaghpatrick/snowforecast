# Merge Phase 1 Branches

Merge all Phase 1 data pipeline branches into develop.

## Pre-Merge Checklist

1. Check all handoff documents are marked complete:
```bash
for wt in pipeline-snotel pipeline-ghcn pipeline-era5 pipeline-hrrr pipeline-dem pipeline-openskimap; do
    echo "=== $wt ==="
    head -10 ~/snowforecast-worktrees/$wt/.claude/handoff.md
done
```

2. Verify all branches are pushed:
```bash
git fetch origin
git branch -r | grep phase1
```

## Merge Process

```bash
cd ~/snowforecast
git checkout develop

# Merge each pipeline branch
for branch in phase1/2-snotel-pipeline phase1/3-ghcn-pipeline \
              phase1/4-era5-pipeline phase1/5-hrrr-pipeline \
              phase1/6-dem-pipeline phase1/7-openskimap-pipeline; do
    echo "Merging $branch..."
    git merge origin/$branch --no-ff -m "Merge ${branch##*/}"
done
```

## Handle Conflicts

If `pyproject.toml` conflicts:
1. Each pipeline's optional-dependencies section should be independent
2. Combine all sections, don't overwrite
3. Verify with: `uv pip install -e ".[all]"`

If `__init__.py` conflicts:
1. Combine all imports
2. Each pipeline exports its own class

## Post-Merge Validation

```bash
# Install all dependencies
uv pip install -e ".[all]"

# Run all pipeline tests
pytest tests/pipelines/ -v

# Run integration tests
pytest tests/integration/ -v
```

## Merge to Main

If all tests pass:
```bash
git checkout main
git merge develop --no-ff -m "Phase 1 complete: All data pipelines"
git push origin main develop
```

## Cleanup

Remove worktrees and branches:
```bash
./scripts/cleanup_worktrees.sh
```
