# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## debug-report-download-fails-part8-9 — debug export enabled while hidden missing-truth blockers still made the endpoint return 400
- **Date:** 2026-04-13
- **Error patterns:** export debug json, debug-report.json, 400 bad request, site unavailable, missing added truth, missing_added_truth_indexes
- **Root cause:** The debug review page computed export readiness from visible exception `ReviewItem`s only, while the download endpoint also counted unresolved `missing_added_truth_indexes`. For later runs with a resolved visible exception row plus a hidden missing-ground-truth-added blocker, the page enabled export but the endpoint correctly returned `400 Bad Request`.
- **Fix:** Aligned debug-page readiness counters with `debug_queue_state["debug_total"]`, passed `missing_added_truth_indexes` into the review queue template, exposed missing added-truth blockers in the debug queue and run status page, and added regression coverage for both the hidden-blocker export mismatch and status-page visibility.
- **Files changed:** shop/routers/review.py, shop/services/review.py, shop/templates/review/queue.html, shop/templates/runs/status.html, tests/test_debug_verdicts.py, tests/test_run_status_debug_summary.py
---

