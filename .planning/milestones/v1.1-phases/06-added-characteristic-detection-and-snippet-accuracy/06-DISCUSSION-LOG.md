# Phase 6: Added Characteristic Detection and Snippet Accuracy - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `06-CONTEXT.md`; this log preserves the analysis that led there.

**Date:** 2026-04-16
**Phase:** 06-added-characteristic-detection-and-snippet-accuracy
**Mode:** assumptions
**Areas analyzed:** Phase Boundary, Added False-Positive Suppression, Title-Block Exclusion, Added Truth Claiming

## Assumptions Presented

### Phase Boundary
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 6 should be solved in the shared reconcile and packet-generation path, not in the review/debug UI layer. | Confident | `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py`, `shop/services/review.py` |

### Added False-Positive Suppression
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The main false-positive added rows come from fragment-level unmatched spans that are already explained by an existing matched/grouped characteristic, so suppression must reason about grouped annotation ownership rather than exact raw-span ownership alone. | Likely | `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py`, `assets/debug_report_part8.json`, `assets/debug_report_part9.json` |

### Title-Block Exclusion
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Title-block and revision-table filtering needs one shared exclusion contract across anchor building, candidate generation, rescue scans, and added detection; the current duplicated heuristics are too inconsistent for SNP-01. | Likely | `delta_preservation/reconcile/anchors.py`, `delta_preservation/reconcile/classify.py`, `delta_preservation/reconcile/match.py` |

### Added Truth Claiming
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| For Part 9, finding more added rows is not sufficient by itself; duplicate added truth rows require deterministic disambiguation beyond normalized requirement text so every detected added packet row can actually claim a canonical truth index. | Likely | `delta_preservation/evaluation/conformance.py`, `shop/services/review.py`, `assets/part9/ground_truth.json`, `assets/debug_report_part9.json` |

## Corrections Made

No corrections — all assumptions confirmed.
