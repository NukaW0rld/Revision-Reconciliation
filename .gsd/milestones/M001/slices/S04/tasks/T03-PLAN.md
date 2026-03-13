# T03: 04-exports-history-and-amendments 03

**Slice:** S04 — **Milestone:** M001

## Description

Add the partial FAI work order export (PDF and CSV) and expose it via download routes and a status page button.

Purpose: WORK-01..04 — after sign-off, engineers generate a work order listing only characteristics that need re-measurement or new measurement, formatted for shop-floor use on monochrome printers.
Output: work_order.html WeasyPrint template, generate_work_order_pdf/csv service functions, new routes in exports router, updated status page.

## Must-Haves

- [ ] "Signed-off run status page has a Generate Work Order button (PDF and CSV)"
- [ ] "Work order PDF contains only changed and added characteristics"
- [ ] "Each work order row shows RE-MEASURE (changed) or NEW (added) in the Priority column"
- [ ] "Work order table columns are: Char #, Priority, Requirement (Rev B), Drawing Reference"
- [ ] "Work order PDF and CSV are both downloadable as immediate attachments"
- [ ] "Work order generation is synchronous (no Huey task)"

## Files

- `shop/services/exports.py`
- `shop/routers/exports.py`
- `shop/templates/exports/work_order.html`
- `shop/templates/runs/status.html`
- `tests/test_exports.py`
