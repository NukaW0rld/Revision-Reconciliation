---
phase: "05"
slug: classification-logic-fixes
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-16
---

# Phase 05 - Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Audit basis: reconstructed from Phase 05 execution artifacts because the plan
> set does not contain a dedicated `<threat_model>` block.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Drawing/Form 3 requirement text -> `classify.py` helpers | Rev A anchor text and Rev B annotation text cross into helper logic that can suppress or promote classifications. | Untrusted drawing text -> in-memory classification state |
| Internal `DeltaItem` -> packet `DeltaItem` | Internal classifier metadata crosses into persisted `delta_packet.json` rows consumed by the debug workflow. | In-memory advisory flags and scores -> durable packet JSON |
| Removed/added pairing metadata -> reconciliation post-pass | Spatial and type metadata controls whether separate removed and added rows collapse into a single changed row. | BBox/page/type metadata -> durable classification rewrite |
| Snapshot fixtures -> regression harness | Checked-in debug reports are read by tests and must not mutate canonical fixtures or `ground_truth.json`. | Canonical JSON fixtures -> read-only assertions |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-05-01 | I (Integrity) | `delta_preservation/reconcile/classify.py`, `delta_preservation/types.py`, `delta_preservation/cli.py` | mitigate | Additive `confidence_flags` defaults at `classify.py:79` and `types.py:278`, plus packet-conversion fallback `getattr(delta_internal, "confidence_flags", [])` at `cli.py:850`, preserve legacy-object compatibility instead of crashing or silently dropping the field. Verified by `tests/test_classify_bugfixes.py:240` and current targeted regression run. | closed |
| T-05-02 | I (Integrity) | `delta_preservation/reconcile/classify.py` | mitigate | CLS-01 bleed suppression is gated by `_BLEED_FLAG`, whitespace-bounded `_BLEED_SPLIT_RE`, and `_looks_like_adjacency_bleed(...)` at `classify.py:15`, `:20`, and `:95`, so slash shape alone cannot suppress real changes. Verified by `tests/test_classify_bugfixes.py:311` and snapshot guards at `tests/test_classify_phase5_regression.py:99` and `:198`. | closed |
| T-05-03 | I (Integrity) | `delta_preservation/reconcile/classify.py` | mitigate | CLS-03 detects symmetric-to-asymmetric tolerance kind changes before `tolerances_match` can preserve `unchanged`, using `_ASYMMETRIC_SHAPE_RE` and `_is_symmetric_to_asymmetric_kind_change(...)` at `classify.py:33` and `:40`. Verified by `tests/test_classify_bugfixes.py:718` and snapshot sweep coverage at `tests/test_classify_phase5_regression.py:184` and `:250`. | closed |
| T-05-04 | I (Integrity) | `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py` | mitigate | `reconcile_removed_added_pairs(...)` at `classify.py:1717` enforces same-page, distance-bounded (`CLS02_MAX_DISTANCE_PT`), requirement-type-compatible, closest-wins pairing before `cli.py:511` rewrites results into the durable packet path. Verified by `tests/test_classify_bugfixes.py:417` and `tests/test_classify_phase5_regression.py:275`. | closed |
| T-05-05 | T (Tampering) | `tests/test_classify_phase5_regression.py` | mitigate | The Phase 5 regression harness loads checked-in `assets/debug_report_part*.json` through `Path.open()` at `tests/test_classify_phase5_regression.py:36` and `:49` and performs assertions only; it does not write to `assets/` or `ground_truth.json`, preserving canonical fixture stability. Verified by `tests/test_classify_phase5_regression.py:99` and `:198`. | closed |

*Status: open / closed*
*Disposition: mitigate (implementation required) / accept (documented risk) / transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-16 | 5 | 5 | 0 | Codex |

---

## Verification Evidence

- `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` -> `29 passed in 0.05s`
- Phase summaries `05-01-SUMMARY.md` through `05-05-SUMMARY.md` all report `Threat Flags: None` and limit scope to classifier logic, packet shaping, and tests rather than new web/API/auth surfaces.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-16
