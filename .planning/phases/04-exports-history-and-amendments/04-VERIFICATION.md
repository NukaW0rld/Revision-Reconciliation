---
phase: 04-exports-history-and-amendments
verified: 2026-03-07T00:00:00Z
status: passed
score: 14/14 must-haves verified
re_verification: false
human_verification:
  - test: "Download Audit Packet PDF and inspect visual layout"
    expected: "Cover page (part number, revision, customer, reviewer, timestamp), summary table, and per-item detail cards with Rev A and Rev B snippet images render correctly at print quality"
    why_human: "WeasyPrint rendering fidelity and image quality cannot be verified programmatically; requires opening the actual PDF"
  - test: "Print work order PDF on monochrome printer"
    expected: "RE-MEASURE and NEW labels are legible in black-and-white; table layout does not break across rows"
    why_human: "Print output requires a physical device; visual readability on monochrome printers cannot be verified in code"
---

# Phase 4: Exports, History, and Amendments — Verification Report

**Phase Goal:** Engineers can download formal deliverables, browse past runs, and correct finalized reviews without destroying the original signed record
**Verified:** 2026-03-07
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | After sign-off, the audit packet is downloadable as PDF (cover page, summary table, per-item detail cards) and as CSV; re-downloadable from run history at any time | VERIFIED | `shop/services/exports.py` has `render_audit_packet_pdf()`, `generate_audit_packet_csv()`; `attempt_sign_off()` calls `generate_and_store_audit_packet()` before status=signed_off; routes at `/exports/{id}/audit-packet.pdf` and `.csv` with Content-Disposition attachment headers; `status.html` shows both buttons for signed_off runs; re-download served from `packet_versions[0].path` |
| 2 | After finalization, engineer can generate a partial FAI work order on demand as PDF and CSV listing only changed/added characteristics with RE-MEASURE/NEW labels | VERIFIED | `generate_work_order_pdf()` and `generate_work_order_csv()` in `exports.py`; `_effective_classification()` handles override decisions; `priority = "RE-MEASURE" if eff == "changed" else "NEW"` at line 136; routes at `/exports/{id}/work-order.pdf` and `.csv`; work_order.html renders `{{ row.priority }}` with four columns (Char #, Priority, Requirement (Rev B), Drawing Reference) |
| 3 | Engineer can view all shop runs filterable by part number and date; can reopen any finalized run to view decisions in read-only mode; finalized runs retained indefinitely | VERIFIED | `date_from` parameter at line 45 in `runs.py`; filter form in `list.html`; `read_only = run.status == "signed_off"` at line 54 in `review.py`; read-only banner in `queue.html` lines 89-92; approve/override guards via `{% if not read_only %}` in `_item_card.html`; signed_off excluded from `DELETABLE_STATUSES` in `tasks.py` |
| 4 | Admin can configure auto-cleanup period for unfinished and failed runs | VERIFIED | `POST /admin/settings/retention` route in `admin.py`; retention_days input in `settings.html`; `cleanup_old_runs` periodic Huey task in `tasks.py` at line 251 reads `retention_days` from ShopConfig; DELETABLE_STATUSES = {"queued", "running", "failed", "completed"} — never includes signed_off |
| 5 | Engineer can reopen a finalized run to create an amendment; original signed packet preserved; amendment has its own sign-off producing a new versioned packet; both packets accessible; input files cannot be changed | VERIFIED | `shop/services/amendments.py::create_amendment()` creates Run with `parent_run_id`, `status="reviewing"`, copies input file paths from parent, inherits `packet_versions`; `POST /review/{id}/amend` route; Amend button + modal in `status.html`; `generate_and_store_audit_packet()` computes version = len(existing)+1 so amendment produces v2; version-aware download at `/exports/{id}/audit-packet.pdf?version=N`; versioned packet list in status.html |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | weasyprint in dependencies | VERIFIED | Line 25: `"weasyprint"` present; `uv run python -c "from weasyprint import HTML"` exits 0 |
| `docker/Dockerfile` | Pango runtime apt deps | VERIFIED | Line 23: `libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0` in apt-get install |
| `shop/models.py` | parent_run_id + packet_versions on Run; retention_days on ShopConfig | VERIFIED | Lines 42, 77-80: all three columns present with correct types |
| `shop/database.py` | run_schema_migrations() | VERIFIED | Line 18: function defined; handles all three new columns |
| `shop/app.py` | run_schema_migrations called after create_all | VERIFIED | Lines 43-44: imported and called |
| `shop/services/exports.py` | generate_audit_packet_csv, render_audit_packet_pdf, generate_and_store_audit_packet, generate_work_order_pdf, generate_work_order_csv | VERIFIED | All five functions present at lines 10, 48, 79, 147, 159 |
| `shop/routers/exports.py` | GET routes for audit-packet.pdf, audit-packet.csv, work-order.pdf, work-order.csv | VERIFIED | All four routes at lines 25, 58, 74, 90; Content-Disposition attachment headers on all |
| `shop/templates/exports/audit_packet.html` | Cover page + summary table + per-item cards | VERIFIED | `.cover`, `.summary`, `.item-card` CSS classes; cover page has part_number, rev_a_label, customer, signed_at, Reviewer; summary table with Char #, Pipeline Classification, Reviewer Decision; per-item cards with snippet images |
| `shop/templates/exports/work_order.html` | RE-MEASURE/NEW priority labels; four columns | VERIFIED | `{{ row.priority }}` renders "RE-MEASURE"/"NEW" from service layer; four columns: Char #, Priority, Requirement (Rev B), Drawing Reference; empty-message fallback present |
| `shop/services/amendments.py` | create_amendment(db, parent_run, initiator_id) -> Run | VERIFIED | Function at line 14; sets parent_run_id, copies input paths, inherits packet_versions, clones ReviewItems with pre-filled decisions |
| `shop/routers/review.py` | POST /{run_id}/amend route | VERIFIED | Lines 243-257: registered, 403 guard for non-signed_off, 303 redirect to amendment queue |
| `shop/templates/runs/status.html` | Download buttons + Amend button + versioned packet list | VERIFIED | Lines 127-136 for download/work-order buttons; lines 142-165 for Amend button and modal; lines 175-188 for versioned packet list |
| `shop/templates/review/queue.html` | Read-only banner + amendment banner | VERIFIED | Line 89: read-only banner; line 76: amendment banner checking `run.parent_run_id` |
| `shop/tasks.py` | cleanup_old_runs periodic task | VERIFIED | Line 251: `@huey.periodic_task(crontab(hour="0", minute="0"))`; reads retention_days from ShopConfig; DELETABLE_STATUSES excludes signed_off |
| `shop/routers/admin.py` | POST /admin/settings/retention | VERIFIED | Line 145: route present; clamps to 1-3650 days |
| `shop/templates/admin/settings.html` | Run Retention section with retention_days input | VERIFIED | Lines 52-53: input present; line 45: states signed-off runs retained indefinitely |
| `shop/routers/runs.py` | date_from filter parameter | VERIFIED | Line 45: parameter; lines 53-59: filter logic with graceful ValueError handling; passed to template context |
| `tests/test_exports.py` | 7 tests passing | VERIFIED | 7 tests pass: test_audit_packet_pdf_bytes, test_audit_packet_csv_rows, test_audit_packet_redownload, test_work_order_button_visible, test_work_order_filters_status, test_work_order_priority_labels, test_work_order_pdf_csv |
| `tests/test_history.py` | 4 tests passing | VERIFIED | 4 tests pass: test_history_filters, test_signed_off_readonly_view, test_cleanup_exempt_signed_off, test_cleanup_deletes_old_runs |
| `tests/test_amendments.py` | 3 tests passing | VERIFIED | 3 tests pass: test_create_amendment, test_amendment_files_locked, test_amendment_versioned_packet |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `shop/services/review.py attempt_sign_off()` | `shop/services/exports.generate_and_store_audit_packet()` | Called inside try block before status=signed_off | WIRED | Lines 90-91 in review.py: imported and called; rollback on exception returns run to "reviewing" |
| `shop/routers/exports.py` | `shop/services/exports.render_audit_packet_pdf()` | GET route returning StreamingResponse | WIRED | Fallback re-render path uses `render_audit_packet_pdf`; primary path serves from stored packet_versions path |
| `shop/models.py Run.packet_versions` | `output_dir/packets/v{N}.pdf` | `generate_and_store_audit_packet` stores path in packet_versions JSON | WIRED | Lines 104-116 in exports.py: `packet_path = packets_dir / f"v{version}.pdf"`; path stored in packet_versions entry |
| `shop/templates/runs/status.html Amend modal` | `POST /review/{run_id}/amend` | Standard HTML form POST | WIRED | Line 165 in status.html: `action="/review/{{ run.id }}/amend"` with `method="POST"` |
| `shop/services/amendments.py create_amendment()` | `shop/models.py Run.parent_run_id` | New Run with parent_run_id=parent_run.id | WIRED | Line 40 in amendments.py: `parent_run_id=parent_run.id` |
| `shop/tasks.py cleanup_old_runs` | `ShopConfig.retention_days` | Reads retention_days from DB; defaults to 30 if None | WIRED | Lines 267-268: `config = db.query(ShopConfig).filter(...)` then `retention_days = getattr(config, "retention_days", None) or 30` |
| `shop/routers/review.py review_queue()` | `shop/templates/review/queue.html` | Passes read_only=True when run.status == 'signed_off' | WIRED | Lines 54, 70 in review.py: `read_only = run.status == "signed_off"` passed to template context |
| `shop/templates/runs/list.html` | `/runs?date_from=YYYY-MM-DD` | date_from input in filter form | WIRED | Lines 25-26 in list.html: `name="date_from"` input with current value; runs.py line 53 applies filter |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PACKET-01 | 04-02 | Audit packet PDF with cover page, summary table, per-item detail cards with snippets | SATISFIED | `audit_packet.html` has cover/summary/item-card sections; `render_audit_packet_pdf()` calls WeasyPrint; sign-off generates and stores; test passes |
| PACKET-02 | 04-02 | Audit packet CSV with one row per characteristic | SATISFIED | `generate_audit_packet_csv()` with all 8 fields; Content-Disposition attachment; test_audit_packet_csv_rows passes |
| PACKET-03 | 04-02 | Signed audit packet re-downloadable from run history at any time | SATISFIED | Route serves from stored packet_versions path; "View" link in list.html goes to status page which always shows download buttons for signed_off runs; test_audit_packet_redownload passes |
| WORK-01 | 04-03 | Engineer can generate partial FAI work order on demand after finalization | SATISFIED | Generate Work Order buttons on signed_off status page; test_work_order_button_visible passes |
| WORK-02 | 04-03 | Work order lists only changed and added characteristics | SATISFIED | `_work_order_rows()` filters `effective_classification in ("changed", "added")`; test_work_order_filters_status passes |
| WORK-03 | 04-03 | Each work order row includes char number, requirement text, drawing reference, priority flag | SATISFIED | Four columns in work_order.html; priority = "RE-MEASURE"/"NEW"; drawing_reference = "Balloon {char_no}"; test_work_order_priority_labels passes |
| WORK-04 | 04-03 | Work order exported as PDF and CSV | SATISFIED | Routes at `/work-order.pdf` and `/work-order.csv`; test_work_order_pdf_csv passes (CSV tested; PDF tested via Docker E2E per summary) |
| HISTORY-01 | 04-04 | Run list filterable by part number and date | SATISFIED | `date_from` + `part_number` query params in runs.py; filter form in list.html; test_history_filters passes |
| HISTORY-02 | 04-04 | Engineer can reopen finalized run to view review decisions in read-only mode | SATISFIED | `read_only = run.status == "signed_off"` passed to template; banner shown; controls hidden; test_signed_off_readonly_view passes |
| HISTORY-03 | 04-04 | Finalized runs retained indefinitely on server | SATISFIED | signed_off excluded from DELETABLE_STATUSES in cleanup task; settings.html states "Signed-off runs are always retained indefinitely"; test_cleanup_exempt_signed_off passes |
| HISTORY-04 | 04-04 | Admin can configure auto-cleanup period for unfinished and failed runs | SATISFIED | POST /admin/settings/retention route; retention_days input in settings.html; cleanup_old_runs Huey periodic task; test_cleanup_deletes_old_runs passes |
| AMEND-01 | 04-05 | Engineer can reopen finalized run to create amendment; original packet preserved | SATISFIED | create_amendment() creates new Run with pre-filled ReviewItems; original run.packet_versions unchanged; POST /review/{id}/amend route; test_create_amendment passes |
| AMEND-02 | 04-05 | Amendment scope limited to review decisions; input files cannot be changed | SATISFIED | create_amendment() copies revA_path/revB_path/form3_path from parent; no file upload in POST /amend route; amendment banner in queue.html states files are locked; test_amendment_files_locked passes |
| AMEND-03 | 04-05 | Amendment requires own sign-off producing new versioned packet; both packets accessible | SATISFIED | amendment inherits parent packet_versions so next sign-off produces v2; versioned packet list in status.html links to `?version=N`; version-aware download route; test_amendment_versioned_packet passes |

**All 14 requirements: SATISFIED**

---

### Anti-Patterns Found

No anti-patterns found. Scanned:
- `shop/services/exports.py` — no TODO/FIXME/placeholder comments; all functions substantive
- `shop/services/amendments.py` — no stubs; create_amendment fully implemented
- `shop/routers/exports.py` — no empty handlers; all routes return real responses
- `shop/tasks.py::cleanup_old_runs` — full implementation with shutil.rmtree and DB delete
- `shop/templates/exports/audit_packet.html` — no placeholder sections
- `shop/templates/exports/work_order.html` — no placeholder sections

The 2 xfailed tests in the full suite are from Phase 1 infrastructure stubs (`test_admin_creates_engineer`, `test_reviewer_fields`) — pre-existing, not Phase 4 regressions.

---

### Human Verification Required

#### 1. Audit Packet PDF Visual Quality (PACKET-01)

**Test:** Build Docker (`cd docker && docker compose up --build`), log in, open a signed-off run, click "Download Audit Packet (PDF)"
**Expected:** PDF renders with a formal cover page (part number, revision, customer, run ID, reviewer name, sign-off timestamp), a summary table of all characteristics, and per-item detail cards with Rev A and Rev B snippet images
**Why human:** WeasyPrint rendering quality — image clarity, pagination, layout correctness — requires visual inspection of the output PDF

#### 2. Work Order Monochrome Print Readability (WORK-03)

**Test:** Download work order PDF, print on monochrome (black-and-white) printer or view in PDF reader in grayscale mode
**Expected:** RE-MEASURE and NEW labels are legible; table does not break awkwardly across pages; header row is readable without color
**Why human:** Shop-floor use requires physical printer verification; automated tests only check CSV values and HTTP response codes

---

### Full Test Suite

```
87 passed, 2 xfailed (pre-existing from Phase 1), 11 warnings
```

Phase 4 test counts:
- `tests/test_exports.py`: 7 passed (PACKET-01..03, WORK-01..04)
- `tests/test_history.py`: 4 passed (HISTORY-01..04)
- `tests/test_amendments.py`: 3 passed (AMEND-01..03)

Docker human E2E verification: approved per 04-06-SUMMARY.md — all five flows verified by human engineer (audit packet PDF/CSV, work order PDF/CSV, history date filter, retention settings, amendment cycle with versioned packets).

---

*Verified: 2026-03-07*
*Verifier: Claude (gsd-verifier)*
