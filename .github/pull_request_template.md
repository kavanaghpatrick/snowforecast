## Summary

<!-- Brief description of changes -->

## Related Issue

Closes #

## Checklist

### Code Quality
- [ ] Code follows project style (see CLAUDE.md)
- [ ] No files modified outside my ownership (parallel work)
- [ ] Type hints on public functions
- [ ] Docstrings on public classes/functions

### Testing
- [ ] Unit tests added/updated
- [ ] Tests pass locally: `pytest tests/pipelines/test_{name}.py -v`
- [ ] Coverage > 80%

### AI Review (MANDATORY for pipelines)
- [ ] Grok review completed - no CRITICAL issues
- [ ] Gemini architecture review (optional)
- [ ] Fixes from reviews applied

### Documentation
- [ ] `.claude/handoff.md` updated in worktree
- [ ] Any new dependencies documented in pyproject.toml

### Data Integrity
- [ ] No mock data in production code
- [ ] No estimated/interpolated values
- [ ] Missing data handled explicitly (NaN, not dropped)

## Handoff Summary

```
Status: Complete
Files:
Tests:
Deps:
Grok Review:
```
