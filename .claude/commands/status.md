# Check Project Status

Show the current status of all worktrees and parallel work.

## Worktree Status

```bash
./scripts/check_status.sh
```

## GitHub Issues Status

```bash
gh issue list --milestone "Phase 1: Data Acquisition & Pipeline" --json number,title,state --jq '.[] | "#\(.number) [\(.state)] \(.title)"'
```

## Branch Status

```bash
git fetch origin
git branch -vv
```

## Handoff Documents

```bash
for wt in ~/snowforecast-worktrees/*/; do
    if [ -f "$wt/.claude/handoff.md" ]; then
        echo "=== $(basename $wt) ==="
        grep -E "^##|Status:|Complete|Blocked" "$wt/.claude/handoff.md" | head -5
        echo ""
    fi
done
```
