---
estimated_steps: 7
estimated_files: 2
---

# T01: Redesign queue.html and _item_card.html with full-screen focused layout and keyboard shortcuts

**Slice:** S04 — Review Queue — Full-Screen Focused Mode
**Milestone:** M003

## Description

Convert the review queue from a scrollable card list to a full-screen focused mode showing one item at a time. Redesign both `queue.html` (layout shell, keyboard JS, filters, banners, modal) and `_item_card.html` (item display, approve/override controls, OOB block) with the industrial dark-mode aesthetic established in S01–S03.

The full-screen layout uses 3 zones inside the content block: a top metadata/filter strip, a middle focused item area (large snippets + controls), and a bottom sticky sign-off footer (handled by the `_signoff_footer.html` include). JS manages a `currentIndex` to control which item is visible; J/K changes the index; A triggers the approve form; O toggles the override panel.

## Steps

1. **Redesign `queue.html` layout structure:**
   - Replace the scrollable `space-y-6` div with a flex column layout that fills the content area (`flex flex-col h-full overflow-hidden`)
   - Top zone: run metadata strip (`border border-base-300 bg-base-200 p-4`) with part number, revisions, customer, job — use `font-mono` for values. Include compact filter dropdowns (status + classification) inline in this strip. Include `{% include "_progress_bar.html" %}`.
   - Replace DaisyUI breadcrumbs with plain `text-sm text-base-content/60` links
   - Replace `alert alert-warning/info/error` banners with `border-l-4 border-{color} bg-base-200 p-3` pattern (D011/D012)
   - Middle zone: `#item-list` div with `flex-1 overflow-y-auto` — all items rendered but only current one visible via `.item-card` class + JS toggle
   - Keep `{% include "_signoff_footer.html" %}` at the bottom (sticky positioning handled by the partial)
   - Redesign `<dialog id="signoff-modal">` with industrial aesthetic: `bg-base-200 border border-base-300`, `font-mono` heading, no `modal-box` DaisyUI class
   - Add keyboard legend: a small fixed or absolute legend showing A/O/J/K shortcuts, styled as `font-mono text-xs text-base-content/40`
   - Add item position counter: `font-mono` "3 / 12" display showing current index and total

2. **Redesign `_item_card.html` visual structure:**
   - Replace `card bg-base-100 shadow-md mb-4 border-l-4` with `border border-base-300 bg-base-200 border-l-4 p-5`
   - Remove `card-body` wrapper
   - Add `data-pending="{{ 'true' if not item.reviewer_decision else 'false' }}"` and `data-char-no="{{ item.char_no }}"` to root div
   - Keep `id="review-item-{{ item.char_no }}"` unchanged
   - Replace `badge badge-outline` / `badge badge-ghost` with `font-mono text-sm` + `bg-base-300 px-2 py-0.5 rounded` inline labels
   - Replace `badge badge-success badge-lg` / `badge badge-warning badge-lg` decision badges with `font-mono font-bold text-success` / `text-warning` styled spans with border-left accent
   - Make snippet images larger (remove `w-1/2` constraint — use more of the available space in focused mode)
   - Preserve `<dialog>` zoom modals for snippets — update modal styling to `bg-base-200 border border-base-300`
   - Replace `btn btn-success btn-sm` with full-width `btn btn-success` (larger in focused mode)
   - Keep `hx-indicator="#spinner-{{ item.char_no }}"` and the spinner element

3. **Convert override panel from `<details class="collapse">` to `<div hidden>`:**
   - Replace `<details class="collapse collapse-arrow border border-base-300 rounded">` with `<div id="override-panel-{{ item.char_no }}" hidden class="border border-base-300 bg-base-200 p-4 mt-2">`
   - Remove `<summary>` — the O key and a visible "Override (O)" button handle toggling
   - Add a visible toggle button: `<button type="button" class="btn btn-outline btn-warning w-full" data-override-toggle="{{ item.char_no }}">Override (O)</button>`
   - Keep override form contents: select, textarea (with `required`), submit button
   - Replace `select-bordered`, `textarea-bordered` with industrial-styled selects/textareas
   - Preserve all `hx-post`, `hx-target`, `hx-swap` attributes

4. **Preserve OOB block and guards:**
   - Lines 110–118 of current `_item_card.html` (`{% if oob_update %}` block) must be preserved verbatim — do not change IDs, includes, or structure
   - `{% set read_only = read_only if read_only is defined else false %}` and `{% set is_amendment = is_amendment if is_amendment is defined else false %}` must remain
   - `{% if not item.reviewer_decision or is_amendment %}` and `{% if not read_only %}` control blocks must remain

5. **Add keyboard shortcut handler (IIFE in `queue.html`):**
   - IIFE pattern consistent with `status.html` JS style
   - Track `currentIndex` (0-based into visible item cards)
   - `J` / `ArrowDown` → advance to next item (wrap or clamp at end)
   - `K` / `ArrowUp` → go to previous item (clamp at 0)
   - `A` → find the approve form button in current item, trigger click (which fires HTMX post)
   - `O` → toggle `hidden` on `#override-panel-{char_no}` of current item; if showing, `focus()` the textarea
   - Guard: `if (e.target.matches('input, textarea, select')) return;` — skip shortcuts when typing
   - Guard: `if (document.querySelector('dialog[open]')) return;` — skip when modal is open
   - Display logic: add/remove a class (e.g., `active-item`) to show/hide items; `display:block`/`display:none`
   - Update position counter on navigation

6. **Add `htmx:afterSwap` listener for auto-advance:**
   - Listen on `document` for `htmx:afterSwap`
   - Check if the swapped element is inside `#item-list` (a review item card)
   - Re-scan items for `data-pending="true"` — find next pending after current index
   - Auto-advance `currentIndex` to that item
   - Update position counter and visibility

7. **Verify templates pass tests:**
   - `uv run pytest tests/` → 93 passed, 2 xfailed
   - `grep -c 'card bg-base-100' shop/templates/review/queue.html shop/templates/review/_item_card.html` → 0
   - `grep -c 'collapse collapse-' shop/templates/review/_item_card.html` → 0
   - Spot-check that `id="review-item-"`, `id="progress-bar"`, `id="signoff-footer"`, `id="signoff-modal"` are present in rendered output

## Must-Haves

- [ ] `queue.html` renders a 3-zone full-screen layout (metadata strip, focused item, sticky footer)
- [ ] Only one item card visible at a time, controlled by JS index
- [ ] Keyboard shortcuts J/K/A/O functional with input/textarea/dialog guards
- [ ] `htmx:afterSwap` auto-advances to next pending item
- [ ] `_item_card.html` uses `border border-base-300 bg-base-200` (no card/shadow)
- [ ] Override panel is `<div hidden>` (no DaisyUI collapse)
- [ ] OOB block in `_item_card.html` preserved verbatim
- [ ] `read_only` / `is_amendment` guards preserved
- [ ] All HTMX swap IDs preserved: `#review-item-{char_no}`, `#progress-bar`, `#signoff-footer`, `#item-list`, `#signoff-modal`
- [ ] All `alert alert-*` replaced with `border-l-4` pattern
- [ ] Keyboard legend visible on queue page
- [ ] Item position counter (e.g., "3 / 12") visible
- [ ] No test regressions

## Verification

- `uv run pytest tests/` → 93 passed, 2 xfailed
- `grep -c 'card bg-base-100' shop/templates/review/queue.html shop/templates/review/_item_card.html` → 0
- `grep -c 'collapse collapse-' shop/templates/review/_item_card.html` → 0
- `grep -c 'alert alert-' shop/templates/review/queue.html` → 0
- All HTMX swap IDs present in template source

## Inputs

- `shop/templates/review/queue.html` — current scrollable card list layout with filter bar, banners, item list, signoff footer, modal
- `shop/templates/review/_item_card.html` — current item card with OOB block, read_only/is_amendment guards, collapse override
- `shop/templates/base.html` — layout shell with `<main class="flex-1 overflow-y-auto p-6">` scroll container
- S01 patterns: `border border-base-300 bg-base-200` section wrapper (D010), `border-l-4` alert pattern (D011/D012), `font-mono` headings
- S03 JS pattern: IIFE + `document.getElementById` + `addEventListener` in `status.html`
- S04 Research: full-screen layout recommendation, keyboard handler design, OOB swap chain analysis

## Expected Output

- `shop/templates/review/queue.html` — redesigned with full-screen 3-zone layout, keyboard shortcut IIFE, position counter, keyboard legend, industrial banners and modal
- `shop/templates/review/_item_card.html` — redesigned with industrial classes, `<div hidden>` override panel, `data-pending` attribute, OOB block intact

## Observability Impact

**Runtime signals:**
- Keyboard shortcut IIFE logs `[queue] J/K/A/O currentIndex=N` to `console.log` on each keypress — visible in DevTools console during reviewer sessions
- `htmx:afterSwap` handler logs `[queue] afterSwap: auto-advancing to index=N` when it finds next pending item

**Inspection surfaces:**
- Each `#review-item-{char_no}` card has `data-pending="true|false"` — readable in DevTools; JS uses this to find next pending on auto-advance
- `#item-counter` element shows "N / M" — visible on screen; confirms currentIndex state
- `#override-panel-{char_no}` has `hidden` attribute when closed — inspectable in DevTools to verify toggle state

**Failure visibility:**
- If OOB swap fails: `#progress-bar` badge counts stay stale while card shows "Approved" — visible mismatch is the diagnostic signal
- If keyboard handler fails to bind: console logs won't appear on J/K/A/O; pressing A will have no effect
- If auto-advance fails: item won't change after approve — reviewer must press J manually; console shows no `[queue] afterSwap` log
