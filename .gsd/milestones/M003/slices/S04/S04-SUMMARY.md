---
id: S04
parent: M003
milestone: M003
provides:
  - Redesigned queue.html with full-screen 3-zone layout, keyboard shortcuts IIFE (J/K/A/O), item counter, keyboard legend, sign-off modal
  - Redesigned _item_card.html with industrial classes, div-hidden override panel, data-pending/data-char-no attributes, OOB block intact
  - Redesigned _progress_bar.html with font-mono labels and semantic text colors (no badge classes)
  - Redesigned _signoff_footer.html and _signoff_footer_oob.html (structurally identical; only hx-swap-oob differs)
  - Redesigned generating.html with pulsing amber dot and font-mono typography
  - Fixed HTMX afterSwap auto-advance bug (outerHTML detached-node issue — D015)
requires:
  - slice: S01
    provides: base.html sidebar+topbar layout, output.css industrial theme tokens
affects:
  - S05
key_files:
  - shop/templates/review/queue.html
  - shop/templates/review/_item_card.html
  - shop/templates/review/_progress_bar.html
  - shop/templates/review/_signoff_footer.html
  - shop/templates/review/_signoff_footer_oob.html
  - shop/templates/review/generating.html
key_decisions:
  - D013: Full-screen focused review — JS-managed item visibility (all cards rendered, display toggled client-side)
  - D014: Override panel — plain div hidden replaces DaisyUI collapse collapse-arrow
  - D015: HTMX afterSwap + outerHTML — check target.id string, not itemList.contains() (detached node)
patterns_established:
  - 3-zone full-screen layout: -m-6 negative margin to escape base.html p-6, flex-col h-full with overflow zones
  - IIFE keyboard handler with input/textarea/dialog guards; data-char-no on cards for O-key override panel lookup
  - htmx:afterSwap auto-advance: scan data-pending="true" forward from currentIndex; wrap to front if none found; use targetId.startsWith('review-item-') not itemList.contains() (detached node on outerHTML swap)
  - Override toggle shared path: O-key keyboard handler and data-override-toggle button both call same panel.hidden = !panel.hidden
  - signoff footer pair: _signoff_footer.html + _signoff_footer_oob.html byte-identical except hx-swap-oob="outerHTML" on root div
observability_surfaces:
  - console.log('[queue] navigate currentIndex=N') on J/K keypress
  - console.log('[queue] A approve currentIndex=N') on A keypress
  - console.log('[queue] O override-panel currentIndex=N hidden=bool') on O keypress
  - console.log('[queue] afterSwap: auto-advancing to index=N') on HTMX afterSwap auto-advance
  - console.log('[queue] afterSwap: no more pending items') when all resolved
  - data-pending="true|false" on each #review-item-{char_no} — readable in DevTools, drives auto-advance
  - #item-counter text "N / M" — visible on screen, confirms current position
  - #progress-bar font-mono labels — stale after approve = OOB swap failure diagnostic
  - #signoff-footer button[disabled] / opacity-40 class — disabled state; amber + enabled = all items resolved
drill_down_paths:
  - .gsd/milestones/M003/slices/S04/tasks/T01-SUMMARY.md
  - .gsd/milestones/M003/slices/S04/tasks/T02-SUMMARY.md
duration: ~75 minutes (T01: ~35m, T02: ~40m)
verification_result: passed
completed_at: 2026-03-14
---

# S04: Review Queue — Full-Screen Focused Mode

**Converted the review queue from a scrollable card list into a full-screen focused mode with J/K/A/O keyboard navigation, HTMX OOB swaps for progress and footer, and industrial dark-mode aesthetic across all 6 review templates.**

## What Happened

**T01** redesigned the two core templates. `queue.html` was rebuilt with a 3-zone layout using `-m-6` negative margin to escape `base.html`'s `p-6` padding, filling the viewport cleanly: a top metadata strip (run info, filter dropdowns, progress bar, alert banners), a scrollable middle `#item-list` zone, and a sticky bottom sign-off footer. All item cards are rendered into `#item-list` but only one is visible at a time — JS sets `display:block/none` on each `.item-card` element, leaving HTMX's OOB swap chain fully intact (D013). A keyboard shortcut IIFE guards against input/textarea/dialog focus and handles J/K (navigate), A (approve), O (override panel toggle). A fixed bottom-right keyboard legend (font-mono, low-opacity, pointer-events-none) is always visible.

`_item_card.html` was redesigned with `border border-base-300 bg-base-200 border-l-4 p-5` (no card/shadow), `data-pending` and `data-char-no` attributes for JS, full-width snippet images in focused mode, a `<div hidden>` override panel (D014, replacing DaisyUI collapse), industrial selects/textareas with `bg-base-300 border border-base-300 font-mono` styling, and the OOB block preserved verbatim at lines 152–161.

**T02** redesigned the 4 remaining partials. `_progress_bar.html` replaced `badge badge-ghost/success/warning` with font-mono inline labels using semantic text colors. `_signoff_footer.html` and `_signoff_footer_oob.html` received font-mono labels and `opacity-40 cursor-not-allowed` + native `disabled` for the disabled state — the two files remain structurally identical with exactly 1 diff line (the `hx-swap-oob="outerHTML"` attribute). `generating.html` replaced `loading loading-spinner loading-lg` with a pulsing amber dot (`w-3 h-3 bg-warning animate-pulse`) and font-mono heading; SSE JS is byte-identical to the original.

T02 also uncovered and fixed a latent bug in the `htmx:afterSwap` handler (D015): for `hx-swap="outerHTML"`, HTMX detaches the original target element before firing `afterSwap`, making `itemList.contains(e.detail.target)` always return false. The fix changes the guard to `e.detail.target.id.startsWith('review-item-')` — the `id` attribute is preserved on detached elements. Without this fix, auto-advance never fires after approve or override.

Browser verification confirmed the complete keyboard-driven flow end-to-end: J/K navigation, A approve (card → "✓ Approved", OOB progress/footer update), O override (panel opens, textarea auto-focused, submit → "⚠ Overridden", OOB update), all items resolved → Sign Off button enabled → modal opens → confirm → redirects.

## Verification

- `npm run build:css` → exit 0, no warnings ✓
- `uv run pytest tests/` → **93 passed, 2 xfailed** ✓
- `grep -c 'card bg-base-100' shop/templates/review/*.html` → **0 all files** ✓
- `grep -c 'alert alert-' shop/templates/review/*.html` → **0 all files** ✓
- `grep -c 'collapse collapse-' shop/templates/review/*.html` → **0 all files** ✓
- `diff _signoff_footer.html _signoff_footer_oob.html` → **exactly 1 line** (hx-swap-oob attribute) ✓
- HTMX swap IDs confirmed: `#review-item-{char_no}`, `#progress-bar`, `#signoff-footer`, `#signoff-modal`, `#item-list` all present ✓
- Browser: J/K → item changes, position counter updates ✓
- Browser: A approve → card updates to "✓ Approved", progress bar OOB swaps, auto-advances ✓
- Browser: O → override panel opens, textarea auto-focused; submit → card updates, progress OOB swaps, auto-advances ✓
- Browser: all items resolved → "All items resolved — ready to sign off", Sign Off button enabled ✓
- Browser: Sign Off modal opens → confirm → redirect (expected rollback on test run with no pipeline output) ✓
- Browser assertions: 6/6 pass (text visible, no console errors, no failed requests) ✓

## Requirements Advanced

- R009 — Full-screen focused review queue now live: one characteristic at a time, large snippet images, dominant action buttons, no scrolling required
- R010 — Keyboard shortcuts functional: A=approve, O=open override panel (auto-focuses textarea), J/K=navigate; visible keyboard legend on screen
- R011 — Sign-off footer and progress bar redesigned with industrial aesthetic; Sign Off CTA visually distinct when all items resolved

## Requirements Validated

- R009 — Verified in running browser: full-screen focused layout, one item visible at a time, counter shows position
- R010 — Verified in running browser: J/K navigates, A approves (HTMX fires), O opens panel + auto-focuses textarea; shortcuts suppressed in inputs and open dialogs
- R011 — Verified in running browser: OOB swaps update progress bar and footer without page reload; Sign Off button enables when all items resolved

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

**Auto-advance bug fix (T02, unplanned)** — The `htmx:afterSwap` handler in `queue.html` used `itemList.contains(e.detail.target)` which always returns false for `hx-swap="outerHTML"` (HTMX detaches original target before firing event). Fixed to `e.detail.target.id.startsWith('review-item-')`. Not in plan; discovered during browser verification. Decision D015 recorded.

**`read_only`/`is_amendment` guard position (T01, minor)** — In `_item_card.html`, these Jinja2 `{% set %}` guards were moved from line ~57 (mid-template) to lines 1–2 (top of template). Functionally identical — Jinja2 `{% set %}` at template top is equivalent.

## Known Limitations

- `_item_card.html` retains `htmx-indicator loading loading-spinner loading-sm` on the approve button's HTMX inflight indicator. This is a functional HTMX class, not a layout component — acceptable per plan scope (plan only prohibits `badge badge-*`, `alert alert-*`, `card bg-base-100`, `collapse collapse-*`).
- Sign-off confirm on a test run with no real pipeline output correctly fails and rolls back to `reviewing` state — expected behavior, not a bug.
- DOM weight: at ~200+ items, JS-managed visibility (D013) may add perceptible load. Not a concern for typical FAIR runs (50–100 items).

## Follow-ups

- S05 owns: admin screens, setup wizard sub-templates, visual consistency audit across all templates
- If `loading-spinner` aesthetics become an issue, a follow-on can replace the inflight indicator class

## Files Created/Modified

- `shop/templates/review/queue.html` — full rewrite: 3-zone full-screen layout, keyboard IIFE, position counter, legend, industrial banners and modal; bug fix: afterSwap guard
- `shop/templates/review/_item_card.html` — full rewrite: industrial classes, div-hidden override panel, data-pending/data-char-no, OOB block intact
- `shop/templates/review/_progress_bar.html` — redesigned: font-mono labels, semantic text colors, no badge classes
- `shop/templates/review/_signoff_footer.html` — redesigned: font-mono labels, opacity-40 disabled state
- `shop/templates/review/_signoff_footer_oob.html` — structurally identical to _signoff_footer.html + hx-swap-oob="outerHTML"
- `shop/templates/review/generating.html` — redesigned: pulsing amber dot, font-mono heading; SSE JS unchanged
- `.gsd/DECISIONS.md` — appended D014, D015
- `.gsd/milestones/M003/slices/S04/tasks/T01-PLAN.md` — added Observability Impact section
- `.gsd/milestones/M003/slices/S04/tasks/T02-PLAN.md` — added Observability Impact section

## Forward Intelligence

### What the next slice should know
- The `base.html` sidebar layout is fully stable. S05 extends it for admin and setup wizard screens the same way S02–S04 did — `{% extends "base.html" %}` with `{% block content %}`.
- All review HTMX swap IDs are battle-tested: `#progress-bar`, `#signoff-footer`, `#item-list`, `#review-item-{char_no}`, `#signoff-modal`. Do not rename these.
- The D013 pattern (JS-managed display toggling of pre-rendered cards) is now established. Do not change it without understanding that HTMX OOB swaps rely on all cards being in the DOM.

### What's fragile
- The `htmx:afterSwap` guard (`targetId.startsWith('review-item-')`) — any rename of item card IDs breaks auto-advance silently. The OOB swap will still work, but the queue won't advance.
- `_signoff_footer.html` and `_signoff_footer_oob.html` must stay structurally identical (1 diff line only). If either is modified, the other must be updated in tandem.

### Authoritative diagnostics
- `[queue]` console log prefix — any J/K/A/O keypress should emit a log; absence means handler not bound (check for JS parse errors)
- `#progress-bar` inner text — if counters are stale after approve, OOB swap response body missing `id="progress-bar"` wrapper
- `data-pending` attribute on item cards in DevTools — drives auto-advance; if wrong, check server-side `item.reviewer_decision` value

### What assumptions changed
- Assumed `itemList.contains(e.detail.target)` would work for outerHTML swaps — it doesn't (HTMX detaches before event). String ID check is the correct approach.
