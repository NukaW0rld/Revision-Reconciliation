# Phase 9: full-corpus-added-characteristic-closure - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `09-CONTEXT.md`; this log preserves the analysis path.

**Date:** 2026-04-18
**Phase:** 09-full-corpus-added-characteristic-closure
**Mode:** assumptions
**Areas analyzed:** Authoritative baseline and evidence refresh, closure scope across the corpus, shared fix ownership, added-detection safety rails, corpus normalization and alias scope

## Assumptions Presented

### Authoritative Baseline and Evidence Refresh
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 9 should treat fresh `run.py` reruns and the Phase 7 algorithm-only fixture set as the authoritative baseline, while keeping `assets/debug_report_part*.json` as frozen Phase 6 corpus evidence. | Confident | `.planning/ROADMAP.md`; `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md`; `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md`; `tests/test_phase7_benchmark.py`; `run.py` |

### Shared Fix Ownership
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 9 should fix the shared packet/evaluation contract and any supporting evidence accounting, not solve the misses via `ground_truth.json` edits or maintainer-UI masking. | Confident | `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-CONTEXT.md`; `delta_preservation/reconcile/classify.py`; `delta_preservation/evaluation/conformance.py`; `delta_preservation/cli.py`; `shop/services/review.py` |

### Closure Scope Across the Whole Corpus
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 9 should close the full current miss bucket rather than only the Part 9 flatness issue: Part 1 accounting, Part 2 normalization/claiming, older Parts 3-5 detector/grouping misses, and Part 9 suppressor loss. | Likely | `.planning/ROADMAP.md`; `.planning/debug/phase06-aggregate-added-gaps.md`; `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-05-SUMMARY.md`; `tests/fixtures/phase7_algorithm_only/part*-debug-report.json` |

### Added-Detection Safety Rails
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The plan should preserve the Phase 6 grouped-evidence and conservative suppressor/claiming safety rails, and narrow owner-signature construction instead of relaxing suppression broadly. | Likely | `.planning/debug/phase06-part9-flatness-miss.md`; `delta_preservation/reconcile/classify.py`; `tests/test_added_detection_phase6.py`; `tests/test_added_truth_selection.py` |

### Corpus Normalization and Alias Scope
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Requirement-format normalization belongs in deterministic shared normalization/claiming logic, and `↗` / `⌰` alias handling should only be pulled in if the remaining closure or evidence refresh still depends on it. | Likely | `.planning/phases/04-gd-t-parser-fixes/04-VALIDATION.md`; `.planning/phases/08-gd-t-verification-recovery/08-RESEARCH.md`; `delta_preservation/reconcile/normalize.py`; `delta_preservation/reconcile/match.py` |

## Corrections Made

No corrections — all assumptions confirmed.

