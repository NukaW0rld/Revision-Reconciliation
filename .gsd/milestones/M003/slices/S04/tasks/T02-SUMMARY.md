---
id: T02
parent: S04
milestone: M003
provides:
  - Redesigned _progress_bar.html with font-mono labels and semantic text colors (no badge classes)
  - Redesigned _signoff_footer.html and _signoff_footer_oob.html (structurally identical, only hx-swap-oob differs)
  - Redesigned generating.html with pulsing amber dot and font-mono typography; SSE JS unchanged
  - Fixed HTMX afterSwap auto-advance bug: check target.id.startsWith() instead of itemList.contains() for outerHTML swaps
key_files:
  - shop/templates/review/_progress_bar.html
  - shop/templates/review/_signoff_footer.html
  - shop/templates/review/_signoff_footer_oob.html
  - shop/templates/review/generating.html
  - shop/templates/review/queue.html
key_decisions:
  - D015: htmx:afterSwap + outerHTML — check e.detail.target.id string, not itemList.contains() (detached node)
patterns_established:
  - signoff footer pair: _signoff_footer.html + _signoff_footer_oob.html are byte-identical except for hx-swap-oob="outerHTML" on root div; verify with diff (should show exactly 1 line)
  - htmx afterSwap guard for outerHTML: use targetId.startsWith('review-item-') to detect item card swaps; do NOT use contains() on detached node
observability_surfaces:
  - "[queue] navigate currentIndex=N" — J/K/A/O key presses; confirms handler is bound
  - "[queue] A approve currentIndex=N" — approve action triggered
  - "[queue] afterSwap: auto-advancing to index=N" — confirms auto-advance fired after outerHTML swap
  - "[queue] afterSwap: no more pending items" — all items resolved; stay on current card
  - "#progress-bar" inner text — font-mono labels; stale counters = OOB swap failure diagnostic
  - "#signoff-footer button[disabled]" with opacity-40 class — disabled state; enabled = Sign Off button amber
  - generating.html: pulsing amber dot; if dot pulses indefinitely = SSE endpoint/Huey worker failure signal
duration: ~40min
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Redesign remaining partials, rebuild CSS, and verify full review flow in browser

**Redesigned 4 remaining review partials, fixed HTMX afterSwap auto-advance bug, and verified full keyboard-driven review flow end-to-end in browser.**

## What Happened

Redesigned `_progress_bar.html` replacing `badge badge-ghost/success/warning` with font-mono inline labels using semantic text colors (`text-base-content/60`, `text-success`, `text-warning`). 

Redesigned `_signoff_footer.html` and `_signoff_footer_oob.html` — added `font-mono` labels, changed disabled state from DaisyUI `btn-disabled` to `opacity-40 cursor-not-allowed` + native `disabled`. The two files are structurally identical; only the `hx-swap-oob="outerHTML"` attribute differs on the root div (diff confirms exactly 1 line difference).

Redesigned `generating.html` — replaced `loading loading-spinner loading-lg` with a pulsing amber dot (`w-3 h-3 bg-warning animate-pulse`), added `font-mono font-bold tracking-wide` heading. SSE JS logic is byte-identical to original.

During browser verification, discovered that the HTMX `htmx:afterSwap` auto-advance handler was broken: for `hx-swap="outerHTML"`, HTMX detaches the original target element before firing `afterSwap`, so `itemList.contains(e.detail.target)` always returns false. The `data-pending` updates were landing in the DOM correctly but auto-advance never triggered. Fixed in `queue.html` by checking `e.detail.target.id.startsWith('review-item-')` instead (the `id` attribute is preserved on detached elements). Added D015 to DECISIONS.md.

## Verification

- `npm run build:css` → exit 0 ✓
- `uv run pytest tests/` → 93 passed, 2 xfailed ✓
- `grep -c 'card bg-base-100\|alert alert-\|collapse collapse-' shop/templates/review/*.html` → all zeros ✓
- `diff _signoff_footer.html _signoff_footer_oob.html` → exactly 1 line (hx-swap-oob attribute) ✓
- Browser: J/K navigation → counter updates, correct item shown ✓
- Browser: A approve → HTMX POST fires, card updates to `✓ Approved`, progress bar OOB swaps (3 pending → 2 pending), signoff footer OOB swaps (4 pending → 3 pending) ✓
- Browser: auto-advance → after approve/override, next pending item shown automatically (index advances) ✓
- Browser: O → override panel opens, textarea auto-focused; fill + Save Override → card shows `⚠ Overridden → changed`, progress counter updates ✓
- Browser: all 4 items resolved → progress bar `0 pending 3 approved 1 overridden`, sign-off footer shows "All items resolved — ready to sign off", Sign Off button enabled ✓
- Browser: Sign Off modal opens with part details → Confirm fires → redirects (to `?error=sign_off_failed` on test run due to missing pipeline output — expected behavior; rollback to reviewing state correct) ✓
- Browser assertions: 6/6 pass (text visible, no console errors, no failed requests) ✓

## Diagnostics

Console logs emit `[queue]` prefixed messages on J/K/A/O keystrokes and afterSwap events. Use DevTools console to verify:
- Handler bound: any J/K/A/O press should log `[queue] navigate currentIndex=N`
- Auto-advance after approve: `[queue] afterSwap: auto-advancing to index=N` (or `no more pending items`)
- Override panel state: `[queue] O override-panel currentIndex=N hidden=true/false`

OOB swap health:
- `#progress-bar` — if stale (counters don't update after approve), OOB swap failed; check HTMX response body contains `id="progress-bar"` wrapper
- `#signoff-footer` — if Sign Off button stays disabled after all items resolved, signoff footer OOB swap failed

generating.html:
- Pulsing amber dot never stops = SSE endpoint not firing; check Huey worker running and `/review/{run_id}/sign-off/sse` endpoint reachable

## Deviations

**Auto-advance bug fix** — unplanned fix to `queue.html` keyboard IIFE. The `htmx:afterSwap` handler used `itemList.contains(e.detail.target)` which always returned false for `outerHTML` swaps (detached node). Fixed to use `e.detail.target.id.startsWith('review-item-')`. This was a latent bug in T01's implementation that only manifested during T02 browser verification. Decision D015 recorded.

## Known Issues

- `_item_card.html` retains `htmx-indicator loading loading-spinner loading-sm` for the approve button HTMX inflight indicator. This is a functional indicator class, not a layout component — acceptable per plan which only prohibits `badge badge-*`, `alert alert-*`, `card bg-base-100`, `collapse collapse-*`.
- Sign-off confirm fails on test DB run (no real pipeline output) — expected; error banner renders correctly and run rolls back to reviewing state.

## Files Created/Modified

- `shop/templates/review/_progress_bar.html` — redesigned: font-mono labels, semantic text colors, no badge classes
- `shop/templates/review/_signoff_footer.html` — redesigned: font-mono labels, `opacity-40` disabled state
- `shop/templates/review/_signoff_footer_oob.html` — structurally identical to _signoff_footer.html + hx-swap-oob="outerHTML"
- `shop/templates/review/generating.html` — redesigned: pulsing amber dot, font-mono heading; SSE JS unchanged
- `shop/templates/review/queue.html` — bug fix: htmx:afterSwap guard changed from `itemList.contains()` to `targetId.startsWith('review-item-')`
- `.gsd/DECISIONS.md` — appended D015
- `.gsd/milestones/M003/slices/S04/tasks/T02-PLAN.md` — added Observability Impact section
