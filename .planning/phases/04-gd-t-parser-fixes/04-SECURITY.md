---
phase: "04"
slug: gd-t-parser-fixes
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-14
---

# Phase 04 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Form 3 / PDF semantic text → `normalize.py` parser | Free-form text crosses into bounded regex parsing and must remain inside the GD&T-only path. | Untrusted drawing text (user-supplied) → structured GD&T payload |
| Composite parser output → `GdtSemanticPayload` schema | Multi-compartment parse state crosses from parser internals into the durable semantic model used by downstream comparison and packet generation. | Parsed compartment list → typed Pydantic model |
| `GdtSemanticPayload` → `semantic_compare.py` | New structured fields influence equality and change classification across revisions. | Structured GD&T data → semantic comparison result |
| Slash-containing semantic text → parser family dispatch | Composite GD&T handling must not consume weld or fit syntaxes that also contain `/`. | Raw token string → family dispatch branch |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-04-01 | T (Tampering) | `delta_preservation/reconcile/normalize.py` | mitigate | Composite slash split guarded by `all(_GDT_CONTROL_MAP.get(s[0]) is not None ...)` at line 766–769; compact helper only invoked after `control_idx` confirmed; word-control normalization wired into `_extract_semantic_payload` (not I/O layer). Weld fractions (`1/8 FILLET`) and fit classes (`H7/p6`) excluded by symbol-check. | closed |
| T-04-02 | I (Integrity) | `normalize.py`, `types.py`, `semantic_compare.py` | mitigate | Malformed-frame error path intact at lines 831 and 857 of `normalize.py` for inputs without tolerance token after compact parsing; routes to `reason_code="gdt_malformed_frame"`. `GdtSemanticPayload.compartments` uses `default_factory=list` (backward-compatible). Single-compartment equality regression at `test_semantic_comparison.py:468`; count-mismatch test at line 491. | closed |
| T-04-03 | D (Denial of Service) | `delta_preservation/reconcile/normalize.py` | mitigate | `_GDT_COMPACT_REMAINDER_RE` is fully anchored (`^...$`) with bounded character classes. `_split_compact_gdt_remainder` performs a single `fullmatch` call. Composite detection is one linear `split("/")` pass; recursion blocked via `_allow_composite=False` parameter — no exponential path. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-14 | 3 | 3 | 0 | gsd-security-auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-14
