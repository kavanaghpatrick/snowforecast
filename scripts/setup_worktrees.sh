#!/bin/zsh
# Git Worktree Setup for Snowforecast Parallel Development
# Run this script from anywhere - it uses absolute paths

set -e

PROJECT_ROOT="/Users/patrickkavanagh/snowforecast"
WORKTREE_BASE="/Users/patrickkavanagh/snowforecast-worktrees"

echo "=========================================="
echo "Snowforecast Parallel Development Setup"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Step 1: Create worktrees directory
echo "[1/7] Creating worktrees directory..."
mkdir -p "$WORKTREE_BASE"

# Step 2: Create develop branch
echo "[2/7] Setting up develop branch..."
git checkout -b develop 2>/dev/null || git checkout develop
git push -u origin develop 2>/dev/null || echo "  (develop already exists on origin)"
git checkout main

# Step 3: Create data directory structure
echo "[3/7] Creating shared data directory structure..."
mkdir -p data/{raw,processed,cache}/{snotel,ghcn,era5,hrrr,dem,openskimap}

# Add to gitignore if not already present
if ! grep -q "^data/$" .gitignore 2>/dev/null; then
    echo "data/" >> .gitignore
    echo "  Added data/ to .gitignore"
fi

# Step 4: Create base project structure
echo "[4/7] Creating base project structure..."
mkdir -p src/snowforecast/{pipelines,utils,models,features}
mkdir -p tests/{pipelines,utils,models,features,integration}

# Create __init__.py files if they don't exist
for dir in src/snowforecast src/snowforecast/{pipelines,utils,models,features}; do
    touch "$dir/__init__.py"
done

# Step 5: Define branches and worktrees
echo "[5/7] Creating branches and worktrees..."

# Arrays for branches and worktrees
worktrees=(setup pipeline-snotel pipeline-ghcn pipeline-era5 pipeline-hrrr pipeline-dem pipeline-openskimap)
branches=(phase1/1-project-setup phase1/2-snotel-pipeline phase1/3-ghcn-pipeline phase1/4-era5-pipeline phase1/5-hrrr-pipeline phase1/6-dem-pipeline phase1/7-openskimap-pipeline)

for i in {1..${#worktrees[@]}}; do
    worktree="${worktrees[$i]}"
    branch="${branches[$i]}"
    worktree_path="$WORKTREE_BASE/$worktree"

    echo "  Creating $worktree -> $branch"

    # Create branch if it doesn't exist
    git branch "$branch" main 2>/dev/null || true

    # Create worktree if it doesn't exist
    if [ ! -d "$worktree_path" ]; then
        git worktree add "$worktree_path" "$branch"
    else
        echo "    (worktree already exists)"
    fi
done

# Step 6: Create symlinks and .claude directories
echo "[6/7] Creating data symlinks and handoff directories..."
for worktree in $worktrees; do
    worktree_path="$WORKTREE_BASE/$worktree"

    # Create data symlink
    if [ ! -L "$worktree_path/data" ] && [ ! -d "$worktree_path/data" ]; then
        ln -s "$PROJECT_ROOT/data" "$worktree_path/data"
        echo "  Created data symlink in $worktree"
    fi

    # Create .claude directory for handoff docs
    mkdir -p "$worktree_path/.claude"

    # Create handoff template
    if [ ! -f "$worktree_path/.claude/handoff.md" ]; then
        cat > "$worktree_path/.claude/handoff.md" << 'HANDOFF'
# Agent Handoff Document

## Current Status
- [ ] In Progress
- [ ] Complete
- [ ] Blocked

## Task Definition
Issue #:
Branch:
Description:

## Files Created/Modified
-

## Dependencies Added
-

## Tests Status
- [ ] Unit tests pass
- [ ] Coverage > 80%

## Grok Review
- [ ] Complete - no critical issues

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Next Agent
-
HANDOFF
        echo "  Created handoff template in $worktree"
    fi
done

# Step 7: Summary
echo "[7/7] Setup complete!"
echo ""
echo "=========================================="
echo "Worktree Summary"
echo "=========================================="
git worktree list
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo "1. COORDINATOR: Work on setup worktree first"
echo "   cd $WORKTREE_BASE/setup"
echo "   - Define shared utilities in src/snowforecast/utils/"
echo "   - Create pyproject.toml with modular dependencies"
echo "   - Create conftest.py with shared fixtures"
echo ""
echo "2. COORDINATOR: Merge setup to develop"
echo "   cd $PROJECT_ROOT"
echo "   git checkout develop"
echo "   git merge phase1/1-project-setup --no-ff"
echo ""
echo "3. AGENTS: Rebase on develop and start parallel work"
echo "   cd $WORKTREE_BASE/pipeline-{name}"
echo "   git rebase origin/develop"
echo ""
echo "4. AGENTS: Update handoff document when done"
echo "   Edit .claude/handoff.md in your worktree"
echo ""
echo "=========================================="
