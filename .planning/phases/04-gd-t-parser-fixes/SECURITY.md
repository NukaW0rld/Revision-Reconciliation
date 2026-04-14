# Security Audit — Phase 04: GD&T Parser Fixes

**Audited:** 2026-04-14
**ASVS Level:** 1
**block_on:** critical
**Plans audited:** 04-01, 04-02

---

## Threat Verification

| Threat ID | Category | Disposition | Status | Evidence |
|-----------|----------|-------------|--------|----------|
| T-04-01 | T (Tampering) | mitigate | CLOSED | `normalize.py:766–769` — composite slash split guarded by `all(_GDT_CONTROL_MAP.get(s[0]) is not None for s in stripped)`; compact helper invoked only at `normalize.py:828` inside the single-token branch where `control_idx` is already confirmed; word-control normalization called only inside `_extract_semantic_payload` at `normalize.py:603` before parser dispatch. |
| T-04-02 | I (Integrity) | mitigate | CLOSED | Malformed-frame error path preserved at `normalize.py:831` (compact path) and `normalize.py:857` (whitespace-tokenized path) — both return the string sentinel that routes to `reason_code="gdt_malformed_frame"` at `normalize.py:629`. `GdtSemanticPayload.compartments` defaults to `[]` via `default_factory=list` in `types.py:97–100`. Single-compartment equality regression covered in `tests/test_semantic_comparison.py:468` and compartment count mismatch in `tests/test_semantic_comparison.py:491`. |
| T-04-03 | D (Denial of Service) | mitigate | CLOSED | `_GDT_COMPACT_REMAINDER_RE` is anchored (`^...$`) with bounded character classes at `normalize.py:401–403`. `_split_compact_gdt_remainder` uses a single `fullmatch` call at `normalize.py:417`. Composite split is linear: one `split("/")` pass over slash-separated segments with no recursion allowed (`_allow_composite=False` at `normalize.py:774`). |

---

## Unregistered Threat Flags

None — neither `04-01-SUMMARY.md` nor `04-02-SUMMARY.md` contain a `## Threat Flags` section.

---

## Accepted Risks Log

*(empty — no threats were dispositioned as `accept` or `transfer` in this phase)*
