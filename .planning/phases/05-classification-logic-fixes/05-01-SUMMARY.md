---
phase: 05-classification-logic-fixes
plan: "01"
subsystem: classification-scaffold
tags: [confidence_flags, backward-compat, scaffold, tdd]
dependency_graph:
  requires: []
  provides: [confidence_flags-field, cli-getattr-guard]
  affects: [delta_preservation/reconcile/classify.py, delta_preservation/types.py, delta_preservation/cli.py]
tech_stack:
  added: []
  patterns: [dataclass-field-default, pydantic-field-default, getattr-defensive-access]
key_files:
  created:
    - tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility
  modified:
    - delta_preservation/reconcile/classify.py
    - delta_preservation/types.py
    - delta_preservation/cli.py
    - tests/test_pipeline_semantic_packet.py
    - tests/test_output_formatting.py
decisions:
  - "Use field(default_factory=list) on internal dataclass to guarantee independent mutable defaults per instance"
  - "Use getattr(delta_internal, 'confidence_flags', []) in CLI to tolerate legacy/fake internal objects without the attribute"
  - "Use Field(default_factory=list) on Pydantic model so legacy JSON payloads that omit the key still deserialize cleanly"
metrics:
  duration_minutes: 2
  completed_date: "2026-04-16"
  tasks_completed: 3
  files_modified: 5
---

# Phase 05 Plan 01: Confidence Flags Scaffold Summary

**One-liner:** Additive `confidence_flags: List[str]` field on both DeltaItem models with `getattr` CLI guard and deterministic backward-compat tests.

## What Was Built

Added the `confidence_flags` advisory field as an additive, optional attribute to both the internal (dataclass) and persisted (Pydantic) `DeltaItem` models. Hardened the CLI packet conversion with a `getattr` fallback so legacy or monkeypatched internal objects without the attribute do not crash packet generation. Replaced the previous empty Phase 5 stubs with three concrete compatibility tests.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Replace empty Phase 5 stubs with concrete compatibility tests (RED) | 941132f | tests/test_classify_bugfixes.py |
| 2 | Add confidence_flags to both DeltaItem models and harden CLI conversion (GREEN) | 11596f4 | classify.py, types.py, cli.py |
| 3 | Update fake internal DeltaItem helpers used by CLI-facing tests | fcf2cd2 | test_pipeline_semantic_packet.py, test_output_formatting.py |

## TDD Gate Compliance

- RED gate: `941132f` — `test(05-01): add failing TestConfidenceFlagsCompatibility scaffold tests` (3 tests failed before production code)
- GREEN gate: `11596f4` — `feat(05-01): add confidence_flags to both DeltaItem models and harden CLI conversion` (all 3 tests passed)

## Decisions Made

- `field(default_factory=list)` on the internal dataclass guarantees each instance gets an independent mutable list (avoids shared-mutable-default bug).
- `getattr(delta_internal, "confidence_flags", [])` in CLI is intentionally kept even after fake objects define the attribute — defensive access protects against future ad-hoc debugging helpers.
- Pydantic `Field(default_factory=list)` allows legacy JSON payloads (pre-field) to deserialize without error.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — `confidence_flags` is a real persisted field with a default. No placeholder or hardcoded-empty flows to UI rendering.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary crossings introduced.

## Self-Check

- [x] `delta_preservation/reconcile/classify.py` — `confidence_flags: List[str] = field(default_factory=list)` at line 33
- [x] `delta_preservation/types.py` — `confidence_flags: List[str] = Field(default_factory=list, ...)` at line 278
- [x] `delta_preservation/cli.py` — `confidence_flags=getattr(delta_internal, "confidence_flags", [])` at line 847
- [x] `tests/test_classify_bugfixes.py` — `TestConfidenceFlagsCompatibility` class with 3 passing tests
- [x] Commit 941132f exists (RED gate)
- [x] Commit 11596f4 exists (GREEN gate)
- [x] Commit fcf2cd2 exists (Task 3)
- [x] Full verification suite passes: 3 compatibility tests + 17 CLI/formatting tests

## Self-Check: PASSED
