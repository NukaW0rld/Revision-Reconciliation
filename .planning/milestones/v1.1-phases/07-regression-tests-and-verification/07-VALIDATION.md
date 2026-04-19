---
phase: 07
slug: regression-tests-and-verification
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-19
reconciled: 2026-04-19
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `07-01` | 1 | — | Per-cluster audit and gap identification — verify each Phase 4-6 fix cluster has at least one parametrized pytest case |
| `07-02` | 2 | `07-01` | TST-02 benchmark baseline derivation — cross-part benchmark test anchored to algorithm-only fixtures |
| `07-03` | 3 | `07-01`, `07-02` | VER-01 full 9-part ground-truth re-run and parity artifact resolution |
| `07-04` | 4 | `07-02`, `07-03` | Algorithm-only baseline fixtures and parity correction — anchor benchmark to standalone pipeline output |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_phase7_benchmark.py tests/test_phase7_regression.py -x` |
| **Full suite command** | `uv run pytest -q -x` |
| **Estimated runtime** | ~45 seconds full suite |

---

## Sampling Rate

- After every task commit: run the narrowest task-local command targeting the relevant test cluster.
- Before wave rollover: run the Phase 7 quick command (`uv run pytest -q tests/test_phase7_benchmark.py tests/test_phase7_regression.py -x`).
- Before phase close: run the full suite (`uv run pytest -q -x`).

---

## Nyquist Compliance (Phase 12 reconciliation)

This `07-VALIDATION.md` was authored during Phase 12 process-artifact closure to backfill the missing Nyquist validation artifact for Phase 7. Nyquist compliance is backed by `07-VERIFICATION.md`, which records Phase 7 as `status: passed` with `score: 3/3 must-haves verified` — covering TST-01 (per-cluster regression coverage), TST-02 (cross-part benchmark, 20 tests pass), and VER-01 (9-part parity-corrected ground-truth run, zero true regressions). See `07-VERIFICATION.md` and `v1.1-MILESTONE-AUDIT.md (nyquist: missing_phases: [7])` for the full evidence record.
