# T08: 02-pipeline-bridge 08

**Slice:** S02 — **Milestone:** M001

## Description

Implement remaining test stubs, run the full test suite, and perform human Docker verification of the Phase 2 pipeline bridge.

Purpose: This is the Phase 2 gate plan — automated suite must be green and a human must verify the end-to-end submission flow in Docker before Phase 3 begins.
Output: All test_runs.py stubs implemented and passing, full test suite green, Docker e2e verification approved by engineer.

## Must-Haves

- [ ] "All tests in tests/test_runs.py pass (no xfail, no error)"
- [ ] "Full test suite is green"
- [ ] "Docker container starts successfully with docker compose up"
- [ ] "Engineer can submit a run end-to-end in Docker and see stage progress"
- [ ] "Failed run shows correct error state; warning run shows correct warning UI"
- [ ] "Alert bell badge reflects unread alert count after run failure"
