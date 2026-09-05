## Coding Style

### 1. Simplicity First

Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite to prioritize maintainability.

### 2. Surgical Changes

Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (YoYo-XYZ/pinn-deepflow). See `docs/agents/issue-tracker.md`. Use github cli if available.

### Triage labels

Using default five canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (root `CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.
