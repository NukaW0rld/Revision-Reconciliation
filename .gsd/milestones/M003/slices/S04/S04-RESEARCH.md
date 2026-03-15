# S04: Review Queue — Full-Screen Focused Mode — Research

**Date:** 2026-03-14

## Summary

S04 redesigns 6 review templates: `queue.html`, `_item_card.html`, `_progress_bar.html`, `_signoff_footer.html`, `_signoff_footer_oob.html`, and `generating.html`. It adds keyboard shortcuts (A/O/J/K) inline in `queue.html`. The redesign converts from a scrollable list-of-cards to a full-screen focused mode: one item visible at a time, large snippet images, dominant action buttons, and a keyboard legend.

The core complexity is threefold: (1) the HTMX OOB swap chain in `_item_card.html` must survive the DOM restructure; (2) keyboard event handlers must survive HTMX swapping the item card out from under them (re-attach via `htmx:afterSwap`); (3) the full-screen layout must work within `base.html`'s `<main class="flex-1 overflow-y-auto p-6">` scroll container, not the viewport — meaning sticky footers use `sticky bottom-0` inside that container, not `fixed bottom-0`.

The current queue is a scrollable card list with DaisyUI `card bg-base-100 shadow-md`, `alert alert-*`, and `collapse collapse-arrow` components — all of which need elimination consistent with patterns from S01–S03. No keyboard shortcuts exist anywhere in the app — this is new code.

**Recommendation:** Restructure `queue.html` as a 3-zone full-screen layout within the content block — top metadata strip, middle split-pane (snippets left, controls right), bottom sticky sign-off footer. All items are rendered into a hidden array; JS controls which one is "current" and visible. HTMX approve/override still targets `#review-item-{char_no}` via outerHTML swap; keyboard handler re-attaches via `document.addEventListener("htmx:afterSwap", ...)`. Override panel uses a slide-down `<div>` (not DaisyUI collapse) toggled by the O key.

## Recommendation

Build a single-page focused layout with JS-managed item index. The rendered items list is preserved in the DOM (`#item-list` with all cards), but only the current item card is displayed (`display:block`/`display:none` via a `current-item` class). J/K changes the index and applies the class. A submits the approve form of the current card via `htmx.trigger()`. O toggles a visible override panel (`<div>` with hidden/block) and auto-focuses the textarea. After HTMX swaps a card, the `htmx:afterSwap` listener re-evaluates the current index and auto-advances to the next pending item.

This approach keeps the HTMX OOB swap chain 100% intact — the backend returns the same HTML structure, HTMX updates `#review-item-{char_no}`, `#progress-bar`, and `#signoff-footer` exactly as now. The only change is that display logic lives in JS rather than CSS scrolling.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| HTMX post-swap hook | `document.addEventListener("htmx:afterSwap", ...)` | HTMX fires this globally after any swap — use it to reattach keyboard context and auto-advance index |
| Sticky footer in overflow-y-auto pane | `sticky bottom-0` inside `<main>` | S01 decision: `<main>` is the scroll container; `fixed bottom-0` would escape the pane |
| Override panel toggle | Raw `<div hidden>` + `el.hidden = !el.hidden` | Simpler and more reliable than DaisyUI collapse for programmatic O-key toggle |
| Image zoom modal | Native `<dialog>` element (`showModal()`) | Already used in current `_item_card.html`; preserve the pattern |
| Classification badge colors | `badge-success` / `badge-warning` / `badge-ghost` | DaisyUI badge classes map to industrial theme tokens correctly |

## Existing Code and Patterns

- `shop/templates/review/queue.html` — current queue: scrollable card list, `card bg-base-100 shadow-sm` for run metadata, `alert alert-warning/info/error` for banners, filter bar with DaisyUI selects, `#item-list`, `#signoff-footer` include, `<dialog id="signoff-modal">` for confirmation. All must be redesigned; `#item-list`, `#signoff-modal` IDs preserved.
- `shop/templates/review/_item_card.html` — outer div `id="review-item-{{ item.char_no }}"` is the HTMX outerHTML swap target. Currently `card bg-base-100 shadow-md border-l-4`. OOB block (lines 110–118) must be preserved exactly: `<div id="progress-bar" hx-swap-oob="true">` + `{% include "_signoff_footer_oob.html" %}`. Redesign the visual shell; leave OOB structure untouched.
- `shop/templates/review/_signoff_footer.html` — `id="signoff-footer"` on root div; `sticky bottom-0 bg-base-200 border-t border-base-300`. ID must be preserved. Already uses the right sticky pattern. Update classes to industrial aesthetic.
- `shop/templates/review/_signoff_footer_oob.html` — identical to `_signoff_footer.html` except `hx-swap-oob="outerHTML"` on root div. Must stay structurally identical to the non-OOB version — if one changes, both change.
- `shop/templates/review/_progress_bar.html` — `id="progress-bar"` on root div. Currently `badge badge-ghost/success/warning`. Update to font-mono labels with industrial coloring.
- `shop/templates/review/generating.html` — extends `base.html`. SSE listener on `htmx:sseMessage` redirects on `signed_off`/`reviewing` status. Redesign the waiting state; JS logic is untouched.
- `shop/routers/review.py` — context vars injected: `run`, `run_id`, `items` (filtered), `user`, `pending`, `approved`, `overridden`, `total`, `status_filter`, `classification_filter`, `error`, `read_only`, `is_amendment`. Approve/override endpoints return `_item_card.html` with `oob_update=True`. No backend changes needed.
- `shop/templates/base.html` — `<main class="flex-1 overflow-y-auto p-6">` is the scroll container. Content block renders inside it. `sticky bottom-0` works relative to this container.
- `shop/templates/runs/status.html` — JS style reference: IIFE, `document.getElementById`, `addEventListener`, raw string HTML injection. S04 keyboard handler should follow the same IIFE + event listener pattern.
- `shop/templates/dashboard.html` — `border border-base-300 bg-base-200` section pattern; `font-mono` headings. Use for run metadata strip.

## Constraints

- **HTMX swap IDs must be preserved:** `#review-item-{char_no}` (outerHTML), `#progress-bar` (OOB), `#signoff-footer` (OOB via `_signoff_footer_oob.html`), `#item-list` (used by router filter logic), `#signoff-modal` (referenced by signoff-footer onclick).
- **OOB double-include:** `_item_card.html` OOB block includes `_signoff_footer_oob.html`. If `_signoff_footer.html` and `_signoff_footer_oob.html` differ structurally (class mismatches), the OOB swap will visually break the footer after approve/override. Both files must be kept in sync.
- **`sticky bottom-0` scope:** The sign-off footer must use `sticky bottom-0` inside the content block, not `fixed`. See S01: `<main>` is the scroll container. If the full-screen layout uses `h-full overflow-hidden` internally to prevent `<main>` scrolling, the sticky footer still works at the bottom of that inner layout.
- **Override note required:** Backend validates `override_note` is non-empty and returns 422 if missing. The redesigned override panel must preserve the `required` attribute on the textarea.
- **`read_only` and `is_amendment` guards:** `_item_card.html` has logic blocks for these — preserve them. When `read_only`, no approve/override controls render. When `is_amendment`, controls render even if `reviewer_decision` is set.
- **Keyboard shortcuts must not fire inside inputs/textareas:** A/O/J/K listeners should check `event.target.tagName` and skip if the user is typing in a form field (override note textarea, filter selects).
- **`htmx:afterSwap` for keyboard re-attach:** After approve/override HTMX swap, the JS must re-evaluate `currentIndex` (find first pending item, or stay at current position) and reattach any event state. Pure DOM listeners attached to `document` survive the swap; per-element listeners would not.
- **CSS build must stay clean:** No new `@layer` blocks or `@apply` without testing. The `border border-base-300 bg-base-200` pattern from D010 is confirmed safe.
- **Test baseline:** 93 passed, 2 xfailed. Template changes must not break any test. Tests check `id="review-item-{n}"`, `id="progress-bar"`, approve/override form POST responses.

## Common Pitfalls

- **Deleting OOB block from `_item_card.html`** — The lines 110–118 (`{% if oob_update %}` block with `id="progress-bar"` and `{% include "_signoff_footer_oob.html" %}`) must be preserved verbatim. This is the mechanism by which approve/override POSTs update the progress bar and signoff footer without a page reload.
- **Using `fixed bottom-0` for signoff footer** — Will escape the `<main>` scroll container and render at the bottom of the viewport, over the sidebar. Use `sticky bottom-0` only.
- **Keyboard handler on `event.target` check** — If `event.target` is inside the override textarea and user presses `A`, the approve form fires. Must guard with `if (event.target.matches('input, textarea, select')) return;`
- **Advance-to-next after approve** — After HTMX swaps the approved card (which now shows the green "Approved" badge instead of buttons), the keyboard focus should auto-advance to the next pending item. Implement in `htmx:afterSwap` by scanning items for next `reviewer_decision == null` element (check for presence of approve button as proxy).
- **`card bg-base-100 shadow-md`** — Must be eliminated from `_item_card.html` root div per D010. Use `border border-base-300 bg-base-200` + `border-l-4 border-{color}` for status accent.
- **`alert alert-warning/info/error`** — Must be replaced with `border-l-4 border-{warning/info/error} bg-base-200` per D011/D012 patterns from S02/S03.
- **`collapse collapse-arrow`** — DaisyUI collapse component used for override panel. Must be replaced with a plain div toggled by JS (for keyboard O-key control). DaisyUI collapse is controlled by checkbox/details state, not programmatically.
- **`details` element for override** — Current override uses `<details class="collapse ...">`. This can't be opened programmatically via JS in all browsers. Replace with `<div id="override-panel-{{ item.char_no }}" hidden>` toggled by `el.hidden = !el.hidden`.
- **Filter bar in full-screen mode** — The current filter bar (status + classification dropdowns) changes which items are shown. In full-screen mode, filters should probably be in the header strip rather than inline. Keep the `<form method="GET">` submit behavior; just move it to a less prominent position.
- **`signoff-modal` dialog reference** — `_signoff_footer.html` has `onclick="document.getElementById('signoff-modal').showModal()"`. The modal `<dialog id="signoff-modal">` must remain in `queue.html`. Don't remove it during restructuring.

## Open Risks

- **J/K navigation with filters active** — If status filter is "pending" only, the `items` list from the router is filtered. J/K can only navigate visible items. When an item is approved (card updates to show badge, no more pending class), the auto-advance needs to re-derive the visible pending list from the DOM state, not the original rendered order. **Mitigation:** Mark each item card with a `data-pending` attribute that JS reads; update it in `htmx:afterSwap`.
- **Focus management after O key** — The override textarea should auto-focus when O is pressed. If the textarea is inside a `hidden` div, `focus()` may not work in all browsers until the div is visible. **Mitigation:** Set `hidden = false` before calling `focus()`.
- **HTMX version compatibility** — `htmx:afterSwap` is a standard HTMX event that fires after any swap. Confirmed present in the HTMX version bundled at `static/js/htmx.min.js`. Low risk.
- **Sign-off modal firing after keyboard A in modal context** — If the sign-off confirmation modal is open and user presses A, the keyboard handler should be suppressed. **Mitigation:** Check `document.querySelector('dialog[open]')` before acting on keyboard events.
- **`hx-indicator` spinner** — Current `_item_card.html` uses `<span id="spinner-{{ item.char_no }}" class="htmx-indicator loading loading-spinner loading-sm">`. Must be preserved so HTMX shows loading state during approve POST. In full-screen mode, spinner placement may need adjustment.
- **Override classification select default** — `<option value="changed" selected>` is the default in the current override form. Preserve this — the backend doesn't validate a default, just coerces unknown values to "changed".

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| HTMX | n/a | none found — using HTMX docs directly |
| Tailwind CSS v4 / DaisyUI v5 | n/a | none found — patterns established in S01–S03 |

## Sources

- Existing codebase exploration (review router, templates, S01–S03 summaries)
- Decisions register: D004 (swap IDs preserved), D007 (standalone base), D008 (DaisyUI v5 theme), D010 (section wrapper), D011/D012 (alert banner pattern)
- S01 Forward Intelligence: `<main>` is scroll container, `sticky bottom-0` scopes to it; `| default()` guards on context vars
