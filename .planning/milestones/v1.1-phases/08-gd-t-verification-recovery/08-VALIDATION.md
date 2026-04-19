---
phase: 08
slug: gd-t-verification-recovery
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
audited: 2026-04-18
---

# Phase 08 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `08-01` | 1 | — | Close the stale GD&T type-gate debt so downstream classification recognizes current Phase 04 parser-produced symbols and word forms as GD&T. |
| `08-02` | 2 | `08-01` | Reconcile stale validation notes, write substantive `04-VERIFICATION.md`, and restore `GDT-01/02/03` traceability in `.planning/REQUIREMENTS.md`. |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py -x` |
| **Full suite command** | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py tests/test_phase7_regression.py -x` |
| **Estimated quick runtime** | ~15-30 seconds |

---

## Sampling Rate

- After every task commit: run the narrowest task-local command from the map below.
- After every plan wave:
  - Wave 1: `uv run pytest -q tests/test_semantic_extraction.py -k "classify_requirement_type or gdt_parsed" -x`
  - Wave 2: `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py tests/test_phase7_regression.py -x`
- Before `$gsd-verify-work`: the full phase command above must be green and the document grep checks must pass.
- Max feedback latency: target under 30 seconds for task-local runs.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 08-01 | 1 | GDT-01, GDT-02, GDT-03 | T-08-01, T-08-02 | `classify_requirement_type()` recognizes current parser-produced GD&T symbols and normalized word forms before non-GDT routing runs. | unit | `uv run pytest -q tests/test_semantic_extraction.py -k "classify_requirement_type_recognizes_phase4_gdt_symbols_and_word_forms or classify_requirement_type_keeps_non_gdt_families_stable" -x` | ✅ existing file extended | ✅ green |
| 08-01-02 | 08-01 | 1 | GDT-01, GDT-02, GDT-03 | T-08-01, T-08-03 | Existing Phase 04 regression suites still pass after the type-gate change; compact, word-form, and composite parser coverage remains green. | unit + integration | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py -x` | ✅ existing files reused | ✅ green |
| 08-02-01 | 08-02 | 2 | GDT-01, GDT-02, GDT-03 | T-08-02 | Stale manual-only validation checks are either replaced with current fixture-backed evidence or explicitly deferred to the roadmap phase that already owns the remaining gap. | document | `rg -n "^## Phase 8 Reconciliation|tests/test_phase7_regression.py|↗|⌰|Phase 09|ADD-01" .planning/phases/04-gd-t-parser-fixes/04-VALIDATION.md && rg -n "↗|⌰" assets/part8/ground_truth.json` | ✅ existing docs + fixtures | ✅ green |
| 08-02-02 | 08-02 | 2 | GDT-01, GDT-02, GDT-03 | T-08-02, T-08-03 | `04-VERIFICATION.md` cites current code, current tests, and current evidence rather than only old summaries. | document | `test -f .planning/phases/04-gd-t-parser-fixes/04-VERIFICATION.md && rg -n "## Goal Achievement|## Required Artifacts|## Key Link Verification|## Requirements Coverage|## Gaps Summary|04-REVIEW-FIX|tests/test_phase7_regression.py|GDT-01|GDT-02|GDT-03" .planning/phases/04-gd-t-parser-fixes/04-VERIFICATION.md` | ✅ new verification artifact | ✅ green |
| 08-02-03 | 08-02 | 2 | GDT-01, GDT-02, GDT-03 | T-08-02 | `.planning/REQUIREMENTS.md` no longer leaves the GDT rows orphaned; traceability points at Phase 8 plans after `04-VERIFICATION.md` is written. | document | `rg -n "^- \\[x\\] \\*\\*GDT-01\\*\\*|^- \\[x\\] \\*\\*GDT-02\\*\\*|^- \\[x\\] \\*\\*GDT-03\\*\\*|^\\| GDT-01 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$|^\\| GDT-02 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$|^\\| GDT-03 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$" .planning/REQUIREMENTS.md` | ✅ existing file updated | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Existing pytest infrastructure already covers the needed parser/comparison/doc-validation surfaces.
- [x] No new framework install is required.
- [x] Existing fixture evidence (`tests/fixtures/phase7_algorithm_only/`, `assets/part8/ground_truth.json`) is available for documentary reconciliation.

---

## Manual-Only Verifications

All Phase 8 behaviors can be closed with automated tests plus documentary grep checks, provided the execution work explicitly re-scopes any unresolved `↗` / `⌰` alias handling to Phase 9 instead of leaving it as an open-ended manual note.

---

## Threat References

| Threat ID | Category | Concern |
|-----------|----------|---------|
| T-08-01 | Integrity | Type-gate changes could over-broaden GD&T routing and misclassify weld, fit, or substring cases as GDT. |
| T-08-02 | Integrity | Verification recovery could certify historical summaries instead of current code/evidence, recreating the audit gap in a different form. |
| T-08-03 | Traceability | Requirements could be marked complete before a substantive `04-VERIFICATION.md` exists, leaving false-positive closure in the milestone docs. |

---

## Validation Sign-Off

- [x] All planned work areas have automated verification commands
- [x] Sampling continuity is preserved across waves
- [x] Wave 0 gaps are explicit and already closed by existing infrastructure
- [x] No watch-mode flags are used
- [x] Task-level feedback latency targets under 30 seconds
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** 2026-04-17 — validation strategy ready for execution

---

## Validation Audit 2026-04-18

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

**Execution evidence**

- `uv run pytest -q tests/test_semantic_extraction.py -k "classify_requirement_type_recognizes_phase4_gdt_symbols_and_word_forms or classify_requirement_type_keeps_non_gdt_families_stable" -x` -> pass
- `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py -x` -> pass
- `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py tests/test_phase7_regression.py -x` -> pass
- `rg -n "^## Phase 8 Reconciliation|tests/test_phase7_regression.py|↗|⌰|Phase 09|ADD-01" .planning/phases/04-gd-t-parser-fixes/04-VALIDATION.md && rg -n "↗|⌰" assets/part8/ground_truth.json` -> pass
- `test -f .planning/phases/04-gd-t-parser-fixes/04-VERIFICATION.md && rg -n "## Goal Achievement|## Required Artifacts|## Key Link Verification|## Requirements Coverage|## Gaps Summary|04-REVIEW-FIX|tests/test_phase7_regression.py|GDT-01|GDT-02|GDT-03" .planning/phases/04-gd-t-parser-fixes/04-VERIFICATION.md` -> pass
- `rg -n "^- \\[x\\] \\*\\*GDT-01\\*\\*|^- \\[x\\] \\*\\*GDT-02\\*\\*|^- \\[x\\] \\*\\*GDT-03\\*\\*|^\\| GDT-01 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$|^\\| GDT-02 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$|^\\| GDT-03 \\| Phase 8 \\| 08-01-PLAN.md, 08-02-PLAN.md \\|$" .planning/REQUIREMENTS.md` -> pass

**Audit note:** No missing Nyquist coverage remained. The only update required was syncing the validation map with the already-green Phase 8 execution state and tightening the documentary commands to match the current artifacts exactly.
