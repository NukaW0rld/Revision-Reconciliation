---
id: T01
parent: S04
milestone: M003
provides:
  - Redesigned queue.html with full-screen 3-zone layout, keyboard shortcuts IIFE, position counter, keyboard legend
  - Redesigned _item_card.html with industrial classes, div-hidden override panel, data-pending attribute, OOB block intact
key_files:
  - shop/templates/review/queue.html
  - shop/templates/review/_item_card.html
key_decisions:
  - D013 (JS-managed item visibility — all cards rendered, display toggled client-side)
  - D014 (override panel: plain div hidden replaces DaisyUI collapse)
patterns_established:
  - 3-zone full-screen layout: `-m-6` negative margin to escape base.html `p-6`, then flex-col h-full with overflow zones
  - IIFE keyboard handler with input/textarea/dialog guards; `data-char-no` on cards for O-key target lookup
  - `htmx:afterSwap` auto-advance: scan cards for `data-pending="true"` forward from currentIndex, wrap to front if none found
  - Override toggle via both O-key (keyboard) and `data-override-toggle` button (mouse); both paths share same panel toggle logic
observability_surfaces:
  - console.log('[queue] navigate currentIndex=N') on J/K keypress
  - console.log('[queue] A approve currentIndex=N') on A keypress
  - console.log('[queue] O override-panel currentIndex=N hidden=bool') on O keypress
  - console.log('[queue] afterSwap: auto-advancing to index=N') on HTMX afterSwap auto-advance
  - data-pending="true|false" on each #review-item-{char_no} — readable in DevTools, drives auto-advance logic
  - #item-counter text "N / M" shows currentIndex + 1 and total — visible on screen
duration: ~35 minutes
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Redesign queue.html and _item_card.html with full-screen focused layout and keyboard shortcuts

**Converted review queue from scrollable card list to full-screen focused mode with J/K/A/O keyboard navigation and HTMX auto-advance.**

## What Happened

Both templates were fully redesigned. `queue.html` gets a 3-zone layout using `-m-6` negative margin to escape base.html's `p-6` padding, making the content fill the viewport cleanly:
- Top zone: run metadata strip with `font-mono` values, compact filter dropdowns, progress bar, and `border-l-4` banners (D011/D012 pattern)
- Middle zone: `#item-list` div with all item cards rendered but only one visible at a time via JS display toggling
- Bottom zone: `{% include "review/_signoff_footer.html" %}` sticky bar (unchanged)
- Fixed bottom-right keyboard legend (font-mono, low-opacity, pointer-events-none)

The keyboard shortcut IIFE tracks `currentIndex` and calls `showCard(n)` which sets `display:block/none` on each `.item-card`. Guards skip shortcuts when typing in form fields or when a dialog is open. The `htmx:afterSwap` listener scans `data-pending` attributes to auto-advance after approve/override.

`_item_card.html` was redesigned with:
- Root div using `border border-base-300 bg-base-200 border-l-4 p-5` (no card/shadow — D010)
- `data-pending` and `data-char-no` attributes for JS consumption
- Snippet images without `w-1/2` constraint — full flex-1 width in focused mode
- Zoom modals styled as `bg-base-200 border border-base-300` (no DaisyUI modal-box)
- Full-width `btn btn-success w-full` approve button
- `<div hidden>` override panel with `data-override-toggle` button (D014)
- Industrial selects/textareas with `bg-base-300 border border-base-300 font-mono` styling
- OOB block (lines 152-161) preserved verbatim with all IDs and includes intact

`read_only` and `is_amendment` variable guards moved to top of template (before first use), preserving all control flow.

## Verification

- `uv run pytest tests/` → **93 passed, 2 xfailed** (no regressions)
- `grep -c 'card bg-base-100' queue.html _item_card.html` → **0** each
- `grep -c 'collapse collapse-' _item_card.html` → **0**
- `grep -c 'alert alert-' queue.html` → **0**
- All HTMX swap IDs confirmed present: `#review-item-{char_no}`, `#progress-bar`, `#signoff-footer`, `#signoff-modal`, `#item-list`
- `data-pending`, `data-char-no`, `id="item-counter"`, keyboard legend, `htmx:afterSwap` handler all present
- Jinja2 template parse: both templates load without syntax errors
- Server starts and returns HTTP 302 (redirect to login) — app runs cleanly

## Diagnostics

- Keyboard handler: open DevTools console; press J/K/A/O on queue page → `[queue] navigate currentIndex=N` logs confirm handler bound
- Auto-advance: approve an item → `[queue] afterSwap: auto-advancing to index=N` confirms next-pending scan
- Override toggle: press O → `#override-panel-{char_no}` `hidden` attribute toggles; DevTools Elements panel shows state
- OOB swap failure: `#progress-bar` badge counts stale while card shows "Approved" — visible mismatch is the diagnostic
- If handler not binding: no `[queue]` console messages; A key has no effect on approve button

## Deviations

- `read_only` and `is_amendment` guards moved to top of `_item_card.html` (lines 1–2) for clarity; previously they were mid-template (line 57–58). Logic is identical — Jinja2 `{% set %}` at template start is functionally equivalent.

## Known Issues

None.

## Files Created/Modified

- `shop/templates/review/queue.html` — full rewrite: 3-zone full-screen layout, keyboard IIFE, position counter, legend, industrial banners and modal
- `shop/templates/review/_item_card.html` — full rewrite: industrial classes, div-hidden override panel, data-pending/data-char-no, OOB block intact
- `.gsd/milestones/M003/slices/S04/tasks/T01-PLAN.md` — added Observability Impact section (pre-flight fix)
