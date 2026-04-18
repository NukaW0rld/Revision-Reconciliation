# Phase 10: debug-exception-gating-and-advisory-surfacing - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-04-18
**Phase:** 10-debug-exception-gating-and-advisory-surfacing
**Mode:** assumptions
**Areas analyzed:** Debug blocker definition, service-level gating, advisory surfacing contract, acknowledgement durability

## Assumptions Presented

### Debug Blocker Definition
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 10 gating should treat unresolved debug exceptions as the existing review-service set: packet rows whose evaluation still needs review plus synthetic missing-added-truth rows that do not yet have a saved debug verdict. | Confident | `.planning/ROADMAP.md`, `shop/services/review.py`, `tests/test_run_status_debug_summary.py`, `tests/test_debug_verdicts.py` |

### Service-Level Gating
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Sign-off and audit/work-order export enforcement should reuse the same service-level debug summary state, not separate template counters or a simple `signed_off` status check. | Confident | `shop/routers/review.py`, `shop/services/review.py`, `shop/routers/exports.py`, `shop/templates/runs/status.html` |

### Advisory Surfacing Contract
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `confidence_flags` should be surfaced directly from packet `DeltaItem` data on maintainer-facing review/status/export surfaces, not re-derived from reasons, mismatches, or semantic summaries. | Confident | `delta_preservation/types.py`, `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py`, `shop/templates/review/_item_card.html`, `shop/templates/review/_item_card_debug.html`, `shop/templates/runs/status.html`, `shop/services/exports.py` |

### Acknowledgement Must Be Durable and Exportable
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| If Phase 10 introduces an explicit acknowledgement path to bypass unresolved debug blockers, that acknowledgement must be persisted and carried into generated audit/work-order artifacts or version metadata, not left as transient UI state. | Confident | `.planning/ROADMAP.md`, `shop/services/review.py`, `shop/services/exports.py`, `shop/routers/exports.py`, `shop/models.py` |

## Corrections Made

No corrections — all assumptions confirmed.
