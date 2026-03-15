---
estimated_steps: 5
estimated_files: 4
---

# T02: Redesign remaining partials, rebuild CSS, and verify full review flow in browser

**Slice:** S04 — Review Queue — Full-Screen Focused Mode
**Milestone:** M003

## Description

Update the 4 remaining review partials (`_progress_bar.html`, `_signoff_footer.html`, `_signoff_footer_oob.html`, `generating.html`) with the industrial aesthetic. Rebuild CSS. Then run the complete review flow end-to-end in a browser to verify keyboard shortcuts, HTMX OOB swaps, auto-advance, and sign-off all work together. This task retires the two high-risk items from the roadmap: HTMX OOB swap compatibility and keyboard nav + HTMX interaction.

## Steps

1. **Redesign `_progress_bar.html`:**
   - Replace `badge badge-ghost/success/warning` with `font-mono text-sm` inline labels
   - Use semantic text colors: `text-base-content/60` for pending count, `text-success` for approved, `text-warning` for overridden
   - Keep `id="progress-bar"` on root div unchanged

2. **Redesign `_signoff_footer.html` and `_signoff_footer_oob.html`:**
   - Both must remain structurally identical — only `hx-swap-oob="outerHTML"` differs on root div
   - Keep `id="signoff-footer"`, keep `sticky bottom-0`
   - Update to industrial aesthetic: `bg-base-200 border-t border-base-300` (already close — refine with `font-mono` labels)
   - Update button: `btn btn-primary` styling is correct via theme; ensure disabled state uses `opacity-40` not DaisyUI `btn-disabled`
   - Verify `onclick="document.getElementById('signoff-modal').showModal()"` references the modal in `queue.html`

3. **Redesign `generating.html`:**
   - Replace `loading loading-spinner loading-lg` with a simpler industrial waiting state (pulsing amber dot or `animate-pulse` text)
   - Use `font-mono` for "Generating Audit Packet" heading
   - Keep SSE JS logic completely unchanged — it's functional and correct
   - Use industrial aesthetic: dark background, monospaced typography, `text-base-content/60` for subtitle

4. **Rebuild CSS and run tests:**
   - `npm run build:css` → exit 0
   - `uv run pytest tests/` → 93 passed, 2 xfailed
   - `grep -c 'card bg-base-100' shop/templates/review/*.html` → 0
   - `grep -c 'alert alert-' shop/templates/review/*.html` → 0
   - `grep -c 'collapse collapse-' shop/templates/review/*.html` → 0

5. **Browser integration verification:**
   - Start dev server (`uv run python run_web.py`)
   - Navigate to a run in reviewing state (or create one if needed)
   - Open review queue → verify single-item focused layout renders
   - Press J → verify next item appears, position counter updates
   - Press K → verify previous item appears
   - Press A → verify approve fires (HTMX POST), card updates to show "Approved", progress bar updates via OOB swap, auto-advances to next pending
   - Press O → verify override panel opens, textarea auto-focused; type note, select classification, submit → card updates, progress bar updates, auto-advances
   - Continue until all items resolved → verify sign-off button enables
   - Click Sign Off → modal opens → Confirm → redirects to status page (or generating page)
   - Verify no console errors during the entire flow

## Must-Haves

- [ ] `_progress_bar.html` uses `font-mono` labels, no DaisyUI badge classes
- [ ] `_signoff_footer.html` and `_signoff_footer_oob.html` are structurally identical (diff should show only the `hx-swap-oob` attribute)
- [ ] `generating.html` uses industrial aesthetic, SSE JS unchanged
- [ ] `npm run build:css` exits clean
- [ ] 93 tests pass, 2 xfailed
- [ ] Full keyboard-driven review flow verified in browser (J/K navigate, A approve, O override, sign-off completes)
- [ ] OOB swaps update progress bar and signoff footer live after approve/override
- [ ] No DaisyUI card/alert/collapse artifacts in any review template

## Verification

- `npm run build:css` → exit 0
- `uv run pytest tests/` → 93 passed, 2 xfailed
- `grep -c 'card bg-base-100\|alert alert-\|collapse collapse-' shop/templates/review/*.html` → 0
- Browser: full review flow completes with keyboard shortcuts only (documented above)
- Browser: no JS console errors

## Inputs

- T01 output: redesigned `queue.html` and `_item_card.html` with full-screen layout and keyboard JS
- `shop/templates/review/_progress_bar.html` — current partial with DaisyUI badge classes
- `shop/templates/review/_signoff_footer.html` — current partial (already close to industrial)
- `shop/templates/review/_signoff_footer_oob.html` — must mirror _signoff_footer.html
- `shop/templates/review/generating.html` — current SSE waiting page
- S01–S03 aesthetic patterns: `border border-base-300 bg-base-200`, `font-mono`, `border-l-4` banners

## Observability Impact

**Signals added/changed by this task:**
- `_progress_bar.html` — `id="progress-bar"` preserved; OOB swap target still resolves. Inspectable by checking `#progress-bar` inner text (font-mono labels, no badge classes)
- `_signoff_footer.html` / `_signoff_footer_oob.html` — `id="signoff-footer"` preserved; OOB swap target resolves. Disabled state uses `opacity-40 cursor-not-allowed` + native `disabled`; inspectable via DevTools attribute panel
- `generating.html` — SSE JS logic unchanged; `id="sse-status"` still present for HTMX SSE swap. Pulsing amber dot replaces spinner — visible regression signal if SSE never fires (dot keeps pulsing indefinitely)
- `[queue]` console logs from T01 keyboard IIFE are unchanged; remain the primary runtime diagnostic for nav/approve/override flows

**How to inspect after deploy:**
1. DevTools → Elements: `#progress-bar` should contain `font-mono` spans, no `.badge` class
2. DevTools → Elements: `#signoff-footer button[disabled]` should have `opacity-40` class
3. DevTools → Console: approve/override fires `[queue] afterSwap: auto-advancing to index=N`
4. DevTools → Network: HTMX POST to `/approve` or `/override` returns 200 with OOB body; look for `hx-swap-oob` in response

**Failure visibility:**
- If `#progress-bar` OOB swap fails: card shows "Approved" but counter text is stale (counters don't update without page reload)
- If `#signoff-footer` OOB swap fails: Sign Off button stays disabled even after all items resolved
- If `generating.html` SSE never fires: amber dot pulses indefinitely — operator signal to check Huey worker / SSE endpoint

## Expected Output

- `shop/templates/review/_progress_bar.html` — redesigned with font-mono labels and semantic text colors
- `shop/templates/review/_signoff_footer.html` — refined with industrial typography and button styling
- `shop/templates/review/_signoff_footer_oob.html` — structurally identical to _signoff_footer.html with `hx-swap-oob`
- `shop/templates/review/generating.html` — redesigned waiting state with industrial aesthetic
- Verified: complete keyboard-driven review flow works in a running browser
