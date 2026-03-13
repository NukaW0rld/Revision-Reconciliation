# T02: 04-exports-history-and-amendments 02

**Slice:** S04 — **Milestone:** M001

## Description

Build the audit packet export service (PDF + CSV), integrate PDF generation into the two-phase sign-off atomicity, add download routes, and surface download links on the run status page.

Purpose: PACKET-01..03 — engineers can download a formal PDF/CSV audit record after sign-off; re-download is always available from run history.
Output: shop/services/exports.py, shop/routers/exports.py, WeasyPrint template, updated review.py sign-off, updated status.html.

## Must-Haves

- [ ] "Sign-off generates an audit packet PDF stored at output_dir/packets/v1.pdf"
- [ ] "Audit packet PDF has cover page, summary table, and per-item detail cards (one item per page)"
- [ ] "Signed-off run status page shows Download Audit Packet PDF and CSV links"
- [ ] "Both PDF and CSV respond with Content-Disposition attachment headers"
- [ ] "Re-downloading uses the stored path in packet_versions — does not re-render"
- [ ] "If PDF generation fails during sign-off, run rolls back to reviewing state"

## Files

- `shop/services/exports.py`
- `shop/services/review.py`
- `shop/routers/exports.py`
- `shop/app.py`
- `shop/templates/exports/audit_packet.html`
- `shop/templates/runs/status.html`
- `tests/test_exports.py`
