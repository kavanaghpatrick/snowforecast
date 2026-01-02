# Spawn Phase 6 Agents (Dashboard V2)

## Execution Order

### Step 1: Complete Blocking Dependency
First, complete issue #48 (Color Scale Foundation) in the main repo:
```bash
cd /Users/patrickkavanagh/snowforecast
# Work on phase6/48-color-scale branch
# See prompt: .claude/prompts/48-color-scale.md
```

### Step 2: Launch Parallel Agents
After #48 is merged to develop, spawn parallel agents for:

| Worktree | Branch | Issue | Prompt |
|----------|--------|-------|--------|
| resort-map | phase6/35-resort-map | #35 | 35-resort-map.md |
| detail-panel | phase6/36-detail-panel | #36 | 36-detail-panel.md |
| favorites | phase6/37-favorites | #37 | 37-favorites.md |
| terrain-3d | phase6/38-terrain-3d | #38 | 38-terrain-3d.md |
| elevation-bands | phase6/39-elevation-bands | #39 | 39-elevation-bands.md |

### Agent Spawn Commands

```bash
# Terminal 1: Resort Map
cd ~/snowforecast-worktrees/resort-map
claude "$(cat ~/.claude/prompts/35-resort-map.md)"

# Terminal 2: Detail Panel
cd ~/snowforecast-worktrees/detail-panel
claude "$(cat ~/.claude/prompts/36-detail-panel.md)"

# Terminal 3: Favorites
cd ~/snowforecast-worktrees/favorites
claude "$(cat ~/.claude/prompts/37-favorites.md)"

# Terminal 4: 3D Terrain
cd ~/snowforecast-worktrees/terrain-3d
claude "$(cat ~/.claude/prompts/38-terrain-3d.md)"

# Terminal 5: Elevation Bands
cd ~/snowforecast-worktrees/elevation-bands
claude "$(cat ~/.claude/prompts/39-elevation-bands.md)"
```

### After Phase 1 Complete, Launch Phase 2:
Issues #40-44 (Time Selector, Overlay, Confidence, SNOTEL, Snow Quality)

### After Phase 2 Complete, Launch Phase 3:
Issues #45-47 (Responsive, Performance, Error Handling)

## Merge Protocol

After all agents complete:
```bash
cd ~/snowforecast
git checkout develop

# Merge each branch
for branch in phase6/35-resort-map phase6/36-detail-panel phase6/37-favorites \
              phase6/38-terrain-3d phase6/39-elevation-bands; do
    git merge origin/$branch --no-ff -m "Merge ${branch##*/}"
done

# Run tests
pytest tests/ -v

# Merge to main
git checkout main && git merge develop --no-ff
git push origin main develop
```
