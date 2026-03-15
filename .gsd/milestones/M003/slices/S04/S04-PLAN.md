# S04: Review Queue — Full-Screen Focused Mode

**Goal:** The review queue is a full-screen focused mode showing one item at a time, navigable by keyboard shortcuts (A/O/J/K), with all 6 review templates redesigned in the industrial dark-mode aesthetic.
**Demo:** A reviewer opens the queue, presses J/K to move between items, A to approve, O to open the override panel (auto-focuses textarea), and processes multiple items without touching the mouse. OOB swaps update progress bar and sign-off footer live. When all items are resolved, the sign-off button enables and the modal fires.

## Must-Haves

- Full-screen focused layout: one item visible at a time with large snippet images, dominant action buttons, and a keyboard legend
- J/K (or ↓/↑) navigate between items; A approves current; O opens override panel and auto-focuses textarea
- Keyboard shortcuts suppressed inside inputs/textareas and when a `<dialog>` is open
- After HTMX approve/override swap, auto-advance to next pending item via `htmx:afterSwap`
- All 6 review templates use industrial aesthetic — no `card bg-base-100 shadow`, no `alert alert-*`, no `collapse collapse-arrow`
- HTMX swap IDs preserved: `#review-item-{char_no}`, `#progress-bar`, `#signoff-footer`, `#item-list`, `#signoff-modal`
- OOB block in `_item_card.html` (lines 110–118) preserved verbatim
- `_signoff_footer.html` and `_signoff_footer_oob.html` kept structurally identical (only `hx-swap-oob` differs)
- `read_only` and `is_amendment` guards preserved in `_item_card.html`
- Override note `required` attribute preserved
- Sign-off modal retains `<dialog id="signoff-modal">` pattern
- `npm run build:css` exits clean
- Test suite: 93 passed, 2 xfailed (no regressions)

## Proof Level

- This slice proves: integration (keyboard JS + HTMX OOB swaps + full-screen layout working together in a running browser)
- Real runtime required: yes
- Human/UAT required: yes (keyboard shortcut flow must be exercised manually in browser)

## Verification

- `npm run build:css` → exit 0, no warnings
- `uv run pytest tests/` → 93 passed, 2 xfailed
- Browser: open review queue for a run with multiple items → one item visible at a time
- Browser: press J/K → item changes, position counter updates
- Browser: press A → approve fires, card updates to show "Approved" badge, auto-advances to next pending
- Browser: press O → override panel slides open, textarea auto-focused; submit override → card updates, auto-advances
- Browser: verify OOB swap — progress bar counters update after approve/override without page reload
- Browser: all items resolved → sign-off button enabled → click → modal opens → confirm → redirects to status page
- `grep -c 'card bg-base-100' shop/templates/review/*.html` → 0
- `grep -c 'alert alert-' shop/templates/review/*.html` → 0
- `grep -c 'collapse collapse-' shop/templates/review/*.html` → 0

## Observability / Diagnostics

- Runtime signals: keyboard shortcut handler logs current index to console on J/K/A/O press (dev aid, can be removed later)
- Inspection surfaces: `data-pending` attribute on each item card (true/false) — JS reads this to find next pending item; inspectable in DevTools
- Failure visibility: if OOB swap fails, `#progress-bar` counters won't update — visible mismatch between approved card state and stale counter
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `base.html` (sidebar + top bar layout), `output.css` (industrial theme tokens), review router context vars (`run`, `items`, `pending`, `approved`, `overridden`, `total`, `status_filter`, `classification_filter`, `error`, `read_only`, `is_amendment`)
- New wiring introduced in this slice: keyboard event handler JS (inline in queue.html), `htmx:afterSwap` listener for auto-advance, `data-pending` attribute on item cards
- What remains before the milestone is truly usable end-to-end: S05 (admin screens, setup wizard sub-templates, fragment cleanup, visual consistency audit)

## Tasks

- [x] **T01: Redesign queue.html and _item_card.html with full-screen focused layout and keyboard shortcuts** `est:1h30m`
  - Why: These two templates are tightly coupled — queue.html defines the full-screen layout structure and keyboard JS; _item_card.html is the card that JS navigates and HTMX swaps. Both must be redesigned together so the DOM structure matches the JS assumptions.
  - Files: `shop/templates/review/queue.html`, `shop/templates/review/_item_card.html`
  - Do: Convert queue.html to 3-zone layout (top metadata strip, middle focused item area, bottom sticky footer). Render all items into `#item-list` but only show current item via JS class toggling. Add keyboard handler IIFE: J/K changes index, A triggers approve, O toggles override panel. Add `htmx:afterSwap` listener for auto-advance. Convert _item_card.html: eliminate card/shadow/collapse, use `border border-base-300 bg-base-200` + `border-l-4` status accent, replace `<details class="collapse">` override with `<div id="override-panel-{char_no}" hidden>`. Add `data-pending` attribute. Preserve OOB block verbatim. Replace all `alert alert-*` banners with `border-l-4 bg-base-200` pattern. Relocate filter bar to compact header strip. Redesign sign-off modal with industrial aesthetic. Preserve all HTMX swap IDs, read_only/is_amendment guards, required attribute on override_note.
  - Verify: `uv run pytest tests/` passes (93 passed, 2 xfailed); `grep -c 'card bg-base-100' shop/templates/review/queue.html shop/templates/review/_item_card.html` → 0; `grep -c 'collapse collapse-' shop/templates/review/_item_card.html` → 0
  - Done when: queue.html renders a full-screen focused layout with one item visible and keyboard JS wired; _item_card.html uses industrial classes with OOB block intact; no test regressions

- [x] **T02: Redesign remaining partials, rebuild CSS, and verify full review flow in browser** `est:45m`
  - Why: The 4 remaining partials (_progress_bar, _signoff_footer, _signoff_footer_oob, generating) need industrial aesthetic. Then the complete review flow — keyboard navigation, approve, override, OOB swaps, sign-off — must be verified end-to-end in a running browser to retire the HTMX + keyboard integration risks.
  - Files: `shop/templates/review/_progress_bar.html`, `shop/templates/review/_signoff_footer.html`, `shop/templates/review/_signoff_footer_oob.html`, `shop/templates/review/generating.html`
  - Do: Update _progress_bar.html: replace `badge badge-ghost/success/warning` with `font-mono` labels and industrial coloring. Update _signoff_footer.html: keep `sticky bottom-0`, update classes to industrial aesthetic (bg-base-200 border-t border-base-300 already close — refine typography and button styling). Mirror changes to _signoff_footer_oob.html (must be structurally identical except for `hx-swap-oob`). Update generating.html: replace `loading loading-spinner` with industrial waiting state. Run `npm run build:css`. Start dev server, create a test run, navigate to review queue, and exercise: J/K navigation, A approve, O override flow, OOB swap verification, sign-off completion.
  - Verify: `npm run build:css` → exit 0; `uv run pytest tests/` → 93 passed, 2 xfailed; browser verification of full keyboard-driven review flow; `grep -c 'alert alert-' shop/templates/review/*.html` → 0
  - Done when: all 6 review templates use industrial aesthetic, CSS builds clean, full review flow works in browser with keyboard shortcuts, OOB swaps update progress/footer live, sign-off completes

## Files Likely Touched

- `shop/templates/review/queue.html`
- `shop/templates/review/_item_card.html`
- `shop/templates/review/_progress_bar.html`
- `shop/templates/review/_signoff_footer.html`
- `shop/templates/review/_signoff_footer_oob.html`
- `shop/templates/review/generating.html`
