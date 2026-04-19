---
phase: 10
slug: debug-exception-gating-and-advisory-surfacing
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-18
---

# Phase 10 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Packet row -> review row rendering | Advisory flags must attach to the exact persisted review row, not just a matching `char_no`. | `DeltaItem.confidence_flags` (low sensitivity — classifier metadata) |
| Packet advisory text -> maintainer UI | The UI must surface packet-native flags, not warnings reconstructed from weaker proxy fields. | Packet string literals rendered in HTML templates |
| Review queue UI -> sign-off transaction | The visible CTA state and the service preflight must agree, or sign-off can be bypassed. | Gate predicate (`can_sign_off` bool + `unresolved_exceptions` count) |
| Sign-off transaction -> signed packet metadata | The packet version must capture the cleared debug state or signed exports can drift from the sign-off evidence. | Debug snapshot JSON payload written to `output_dir/packets/` |
| Signed packet version metadata -> export rendering | Exports must read the sign-off-time snapshot, not current mutable debug files. | `debug_snapshot_path` file read at export time |
| Signed debug snapshot rows -> work-order task list | Synthetic debug-only rows must not become actionable work-order rows. | `review_item_id` merge key (None = excluded) |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-10-01 | Integrity | review/advisory mapping | mitigate | `advisory_flags_by_item_id()` joins by `ReviewItem.id` via `build_debug_queue_state()` packet row order | closed |
| T-10-02 | Integrity | review/status templates | mitigate | Templates render `confidence_flags` directly from packet-derived context; legacy packets default to `[]` via `getattr` fallback | closed |
| T-10-03 | Integrity | review/sign-off gate | mitigate | `build_signoff_gate_state()` centralized preflight reused in router (line 113, 613) and `attempt_sign_off()` service path | closed |
| T-10-04 | Traceability | signed packet metadata | mitigate | `write_signed_debug_snapshot()` captures versioned JSON and records `debug_snapshot_path`, `debug_total`, `unresolved_exception_count` on `packet_versions` entry | closed |
| T-10-05 | Traceability | signed export rendering | mitigate | `_load_signed_debug_snapshot()` raises `ValueError` on missing contract; `_get_signed_run()` returns HTTP 409 instead of silent fallback | closed |
| T-10-06 | Integrity | work-order export | mitigate | `_snapshot_advisory_by_item_id()` excludes `review_item_id=None` rows; synthetic missing-added truth rows appear only in audit/debug summary layer | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-18 | 6 | 6 | 0 | gsd-secure-phase |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-18
