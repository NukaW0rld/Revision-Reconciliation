# T06: 04-exports-history-and-amendments 06

**Slice:** S04 — **Milestone:** M001

## Description

Phase 4 gate: fix any remaining test failures, run the full test suite to green, build Docker with WeasyPrint, and do a human-verified end-to-end flow covering all four feature areas.

Purpose: Confirms the entire phase is production-ready before the milestone closes.
Output: Green test suite, passing Docker build, human-verified e2e flow.

## Must-Haves

- [ ] "pytest -q shows 0 failed tests and 0 xfail in test_exports, test_history, test_amendments"
- [ ] "Full test suite (pytest -q) is green"
- [ ] "Docker image builds successfully with WeasyPrint and Pango deps"
- [ ] "In the running Docker container, the complete sign-off + download + amendment flow works end-to-end"

## Files

- `tests/test_exports.py`
- `tests/test_history.py`
- `tests/test_amendments.py`
