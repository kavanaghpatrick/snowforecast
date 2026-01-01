#!/bin/bash
# Cleanup worktrees after Phase 1 completion
# Only run this after all branches are merged to develop

set -e

PROJECT_ROOT="/Users/patrickkavanagh/snowforecast"
WORKTREE_BASE="/Users/patrickkavanagh/snowforecast-worktrees"

echo "=========================================="
echo "Snowforecast Worktree Cleanup"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Check current worktrees
echo "Current worktrees:"
git worktree list
echo ""

read -p "Are all branches merged to develop? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted. Merge all branches first."
    exit 1
fi

# Remove worktrees
WORKTREES=(
    "setup"
    "pipeline-snotel"
    "pipeline-ghcn"
    "pipeline-era5"
    "pipeline-hrrr"
    "pipeline-dem"
    "pipeline-openskimap"
)

echo ""
echo "Removing worktrees..."
for worktree in "${WORKTREES[@]}"; do
    worktree_path="$WORKTREE_BASE/$worktree"
    if [ -d "$worktree_path" ]; then
        echo "  Removing $worktree..."
        git worktree remove "$worktree_path" --force
    fi
done

# Remove merged branches
echo ""
echo "Removing merged branches..."
BRANCHES=(
    "phase1/1-project-setup"
    "phase1/2-snotel-pipeline"
    "phase1/3-ghcn-pipeline"
    "phase1/4-era5-pipeline"
    "phase1/5-hrrr-pipeline"
    "phase1/6-dem-pipeline"
    "phase1/7-openskimap-pipeline"
)

for branch in "${BRANCHES[@]}"; do
    echo "  Deleting $branch..."
    git branch -d "$branch" 2>/dev/null || echo "    (branch not found or not merged)"
done

# Prune worktree metadata
echo ""
echo "Pruning worktree metadata..."
git worktree prune

# Remove empty worktrees directory
if [ -d "$WORKTREE_BASE" ] && [ -z "$(ls -A $WORKTREE_BASE)" ]; then
    rmdir "$WORKTREE_BASE"
    echo "Removed empty worktrees directory"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
git worktree list
