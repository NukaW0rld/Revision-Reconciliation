---
status: resolved
trigger: "Investigate issue: debug-report-download-fails-part8-9"
created: 2026-04-13T00:00:00-05:00
updated: 2026-04-13T00:24:00-05:00
---

## Current Focus

hypothesis: Confirmed root cause is a readiness mismatch between the debug footer and the export endpoint when `missing_added_truth_indexes` exists.
test: Completed. Footer readiness, debug review surfacing, and status-page blockers are now covered by tests and user verification.
expecting: Export is only offered when both visible exception rows and hidden missing-added-truth blockers are fully resolved.
next_action: Archive the resolved session and record the pattern in the knowledge base.

## Symptoms

expected: Clicking "Export debug json" at the bottom of the completed debug review page should download the debug report JSON file.
actual: The browser fails the download with "Site wasn't available". Server logs show repeated `400 Bad Request` responses for `GET /review/18/debug-report.json` and `GET /review/19/debug-report.json`.
errors: Browser reports site unavailable during download. Docker/uvicorn logs show `400 Bad Request` for the debug-report endpoint. Earlier static assets and snippet requests return 200.
reproduction: Start the app with `docker compose up --build`, log in, open a completed debug review page (part 8 or 9 / runs 18 or 19), click `Export debug json` at the bottom of the page.
started: This worked for parts 1 through 7, then started failing for parts 8 and 9. The user describes it as sudden.

## Eliminated

## Evidence

- timestamp: 2026-04-13T00:04:00-05:00
  checked: Existing debug-session and knowledge-base state
  found: `.planning/debug/knowledge-base.md` does not exist yet, so there is no known-pattern shortcut for this symptom.
  implication: Investigation must proceed from code and runtime evidence rather than prior resolved cases.

- timestamp: 2026-04-13T00:04:00-05:00
  checked: Route and symbol search for `debug-report.json`
  found: The download endpoint is defined in `shop/routers/review.py`, with payload assembly in `shop/services/review.py`.
  implication: The failure surface is narrow and likely sits in route guards or payload assembly rather than the template link itself.

- timestamp: 2026-04-13T00:09:00-05:00
  checked: `shop/routers/review.py`, `shop/services/review.py`, and `shop/templates/review/_signoff_footer.html`
  found: The endpoint gates readiness with `build_run_debug_summary()` which includes `missing_added_truth_indexes`, but the footer enables export from `debug_progress["total"]`, which only counts `exception_items`.
  implication: Runs with unresolved missing-ground-truth-added rows can show an enabled export button in the debug queue while the endpoint still returns `400 Bad Request`.

- timestamp: 2026-04-13T00:11:00-05:00
  checked: Local SQLite state for runs 18 and 19
  found: `shop.db` in this workspace does not contain those run ids, so the exact production-like artifacts are not available locally.
  implication: Reproduction and verification should use a targeted test that exercises the code path directly rather than local run-data inspection.

- timestamp: 2026-04-13T00:15:00-05:00
  checked: Focused regression test for one visible exception row plus one hidden `missing_added_truth_indexes` row
  found: Before the patch, `GET /review/{run_id}?debug=1` rendered export as enabled even though the missing-truth row remained unresolved.
  implication: The bug is in debug-page readiness rendering, not in payload serialization.

- timestamp: 2026-04-13T00:17:00-05:00
  checked: Patched router plus targeted and adjacent test suites
  found: `tests/test_debug_verdicts.py -k missing_truth_added_rows_exist` passed after changing debug footer totals to `debug_queue_state["debug_total"]`; `tests/test_debug_verdicts.py`, `tests/test_focused_debug_queue.py`, and `tests/test_debug_row_identity.py` all passed together.
  implication: The UI readiness logic now aligns with the endpoint guard without regressing adjacent debug queue/export behavior.

- timestamp: 2026-04-13T00:23:00-05:00
  checked: Follow-up local verification and UI surfacing patch
  found: The review queue now renders `missing_added_truth_indexes`, `/runs/{id}` surfaces missing added-truth blockers and their indexes, and `uv run pytest tests/test_debug_verdicts.py tests/test_run_status_debug_summary.py tests/test_focused_debug_queue.py tests/test_debug_row_identity.py -q` passed locally.
  implication: The original failing download path is fixed, and hidden blockers are now visible to the reviewer instead of only affecting export readiness indirectly.

## Resolution

root_cause: The debug review page computed export readiness from visible exception `ReviewItem`s only, while the download endpoint also counted unresolved `missing_added_truth_indexes`. For later runs with a resolved visible exception row plus a hidden missing-ground-truth-added blocker, the page enabled export but the endpoint correctly returned `400 Bad Request`.
fix: Aligned debug-page readiness counters with `debug_queue_state["debug_total"]`, passed `missing_added_truth_indexes` into the review queue template, exposed missing added-truth blockers in the debug queue and run status page, and added regression coverage for both the hidden-blocker export mismatch and status-page visibility.
verification: User confirmed the fix locally on the affected Docker runs. Automated verification passed with `uv run pytest tests/test_debug_verdicts.py tests/test_run_status_debug_summary.py tests/test_focused_debug_queue.py tests/test_debug_row_identity.py -q`.
files_changed: ["shop/routers/review.py", "shop/services/review.py", "shop/templates/review/queue.html", "shop/templates/runs/status.html", "tests/test_debug_verdicts.py", "tests/test_run_status_debug_summary.py"]
