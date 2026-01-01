#!/bin/bash
# Check status across all worktrees
# Useful for coordinator to monitor agent progress

PROJECT_ROOT="/Users/patrickkavanagh/snowforecast"
WORKTREE_BASE="/Users/patrickkavanagh/snowforecast-worktrees"

echo "=========================================="
echo "Snowforecast Development Status"
echo "$(date)"
echo "=========================================="
echo ""

# Main repo status
echo "=== MAIN REPO ==="
echo "Path: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
echo "Branch: $(git branch --show-current)"
echo "Status:"
git status -s
echo ""

# Worktree statuses
WORKTREES=(
    "setup:phase1/1-project-setup:Coordinator"
    "pipeline-snotel:phase1/2-snotel-pipeline:Agent-A"
    "pipeline-ghcn:phase1/3-ghcn-pipeline:Agent-B"
    "pipeline-era5:phase1/4-era5-pipeline:Agent-C"
    "pipeline-hrrr:phase1/5-hrrr-pipeline:Agent-D"
    "pipeline-dem:phase1/6-dem-pipeline:Agent-E"
    "pipeline-openskimap:phase1/7-openskimap-pipeline:Agent-F"
)

for entry in "${WORKTREES[@]}"; do
    IFS=':' read -r worktree branch agent <<< "$entry"
    worktree_path="$WORKTREE_BASE/$worktree"

    echo "=== $worktree ($agent) ==="

    if [ ! -d "$worktree_path" ]; then
        echo "  NOT CREATED"
        echo ""
        continue
    fi

    cd "$worktree_path"

    # Git status
    echo "Branch: $branch"
    changes=$(git status -s | wc -l | tr -d ' ')
    commits_ahead=$(git rev-list origin/main..HEAD --count 2>/dev/null || echo "?")

    if [ "$changes" -eq 0 ]; then
        echo "Working tree: clean"
    else
        echo "Working tree: $changes uncommitted changes"
        git status -s | head -5
        [ "$changes" -gt 5 ] && echo "  ... and $((changes-5)) more"
    fi

    echo "Commits ahead of main: $commits_ahead"

    # Check handoff status
    if [ -f ".claude/handoff.md" ]; then
        status=$(grep -E "^\- \[x\]" .claude/handoff.md | head -1 | sed 's/- \[x\] //')
        if [ -n "$status" ]; then
            echo "Handoff status: $status"
        else
            echo "Handoff status: In Progress"
        fi
    else
        echo "Handoff: No handoff document"
    fi

    # Check for tests
    test_file="tests/pipelines/test_${worktree#pipeline-}.py"
    if [ -f "$test_file" ]; then
        echo "Tests: $test_file exists"
    else
        echo "Tests: Not yet created"
    fi

    echo ""
done

# Summary
echo "=========================================="
echo "Quick Commands"
echo "=========================================="
echo ""
echo "View all worktrees:"
echo "  git worktree list"
echo ""
echo "Fetch latest from origin:"
echo "  git fetch --all"
echo ""
echo "Check if branches are mergeable to develop:"
echo "  git checkout develop && git merge --no-commit --no-ff phase1/X-pipeline && git merge --abort"
