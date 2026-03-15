# S04: Review Queue — Full-Screen Focused Mode — UAT

**Milestone:** M003
**Written:** 2026-03-14

## UAT Type

- UAT mode: live-runtime + human-experience
- Why this mode is sufficient: S04's core risks are keyboard+HTMX integration and OOB swap behavior — neither can be proven by unit tests alone. The full keyboard-driven flow must be exercised in a running browser against a real (or test) run with multiple review items.

## Preconditions

1. Dev server running: `uv run python run_web.py` (default port 8000) and Huey worker: `uv run python -m huey.bin.huey_consumer shop.tasks.huey -w 1`
2. App is set up (setup wizard completed) — if not, navigate to `http://localhost:8000/setup/` first
3. A run exists in the system with status `reviewing` and at least 4 review items with `reviewer_decision = null` (pending). Create one by uploading PDFs and Form 3, running the pipeline, then navigating to Review.
4. Log in as admin or engineer user
5. Browser DevTools console open to watch `[queue]` log messages

## Smoke Test

Navigate to `http://localhost:8000/review/<run_id>/queue`. The page should show a full-screen dark layout with one item card visible, a position counter (e.g., "1 / 4"), and a keyboard legend in the bottom-right corner. No scrolling should be required to see the full item.

---

## Test Cases

### 1. Full-screen layout — single item visible

1. Navigate to the review queue for a run with ≥2 pending items
2. Observe the page layout

**Expected:**
- Page fills the viewport — no white space or light-colored areas
- Exactly one item card is visible in the center zone
- Position counter shows "1 / N" where N is total items
- Top strip shows run metadata (part number, revision) in `font-mono`
- Keyboard legend visible bottom-right: `J` ↓ Next · `K` ↑ Prev · `A` Approve · `O` Override
- Sticky sign-off footer visible at bottom with progress counters

---

### 2. J/K navigation between items

1. On the review queue page with ≥2 items, open DevTools console
2. Press `K` (previous)
3. Press `J` (next)
4. Press `J` again

**Expected:**
- `K` on item 1: counter stays at "1 / N" (can't go before first), console logs `[queue] navigate currentIndex=0`
- First `J`: counter changes to "2 / N", second item card becomes visible, first card disappears
- Second `J`: counter changes to "3 / N" (or wraps if only 2 items)
- Console logs `[queue] navigate currentIndex=N` for each keypress

---

### 3. A key — approve current item

1. Navigate to item 1 (position counter "1 / N")
2. Open DevTools console
3. Press `A`

**Expected:**
- Console logs `[queue] A approve currentIndex=0`
- HTMX POST fires to the approve endpoint
- Item card updates in-place to show "✓ Approved" badge (or equivalent) without a page reload
- Progress bar in top strip updates: pending count decrements by 1, approved count increments by 1 — OOB swap fired
- Sign-off footer updates: pending count decrements by 1 — OOB swap fired
- Auto-advance: next pending item becomes visible; position counter updates; console logs `[queue] afterSwap: auto-advancing to index=N`

---

### 4. O key — open override panel and submit

1. Navigate to any pending item
2. Press `O`
3. Verify panel state
4. Type an override note in the auto-focused textarea
5. Click "Save Override" (or press Enter on the submit button)

**Expected:**
- Console logs `[queue] O override-panel currentIndex=N hidden=true` (panel was hidden, now shown)
- Override panel `<div>` becomes visible below the item details (no animation — plain div unhide)
- Textarea is auto-focused (cursor ready for typing)
- After submitting: card updates to show "⚠ Overridden" badge + override note text, no page reload
- Progress bar OOB swap fires: approved or overridden counter increments
- Auto-advance: next pending item becomes visible

---

### 5. OOB swap — progress bar and footer update live

1. Start with 4 pending items
2. Approve item 1 (press A)
3. Approve item 2 (press A on auto-advanced item)

**Expected after each approve:**
- `#progress-bar` counters (visible in top strip) update without page reload
  - After item 1: "3 pending · 1 approved · 0 overridden" (font-mono labels)
  - After item 2: "2 pending · 2 approved · 0 overridden"
- `#signoff-footer` pending counter also decrements after each approve
- Sign Off button remains disabled (grayed out / `opacity-40`) while pending items remain

---

### 6. All items resolved — Sign Off flow

1. Resolve all pending items (approve or override each)
2. Observe the sign-off footer after the last item is resolved
3. Click "Sign Off"
4. Confirm in the modal

**Expected:**
- After last item resolved: sign-off footer shows "All items resolved — ready to sign off" (or similar affirmative message)
- Sign Off button changes from disabled/gray to enabled amber styling (no longer `opacity-40`)
- Click Sign Off: `<dialog id="signoff-modal">` opens with part details
- Confirm in modal: HTMX POST fires, redirects to run status page (or shows error banner if test run has no pipeline output — expected)

---

### 7. Keyboard shortcut suppression in inputs

1. Navigate to any pending item
2. Press `O` to open the override panel — textarea is auto-focused
3. Press `J`, `K`, `A` while focused in the textarea

**Expected:**
- `J`, `K`, `A` characters are typed into the textarea — they do NOT trigger navigation or approve
- Console does NOT log `[queue] navigate` or `[queue] A approve` while typing in the textarea
- Press `Tab` to move focus out of the textarea; now press `J` again — navigation resumes

---

### 8. Override re-toggle (O key idempotent)

1. Navigate to a pending item
2. Press `O` once — override panel opens
3. Press `O` again — override panel closes
4. Press `O` a third time — override panel opens again

**Expected:**
- Console logs `hidden=true` → `hidden=false` → `hidden=true` alternating
- Panel opens and closes cleanly each time
- No JS errors in console

---

### 9. generating.html — intermediate state

1. Trigger a sign-off that requires async processing (or observe immediately after clicking "Start Review" on a completed run)
2. Navigate to the generating page if it appears

**Expected:**
- Dark page with pulsing amber dot (not a spinner ring)
- `font-mono` heading text
- SSE connection established — page auto-redirects when ready

---

### 10. CSS build and test regression check (automated, run before manual UAT)

```bash
npm run build:css
uv run pytest tests/ -q
grep -c 'card bg-base-100' shop/templates/review/*.html
grep -c 'alert alert-' shop/templates/review/*.html
grep -c 'collapse collapse-' shop/templates/review/*.html
diff shop/templates/review/_signoff_footer.html shop/templates/review/_signoff_footer_oob.html
```

**Expected:**
- `npm run build:css` → exit 0, no warnings
- pytest → `93 passed, 2 xfailed`
- All grep counts → 0
- diff → exactly 1 line changed (hx-swap-oob attribute on root div)

---

## Edge Cases

### Keyboard nav when only 1 item exists

1. Create a run with exactly 1 review item
2. Navigate to queue
3. Press J and K repeatedly

**Expected:** Counter stays at "1 / 1" — no navigation occurs, no JS errors logged

---

### Shortcut suppressed when sign-off modal is open

1. Resolve all items so Sign Off button enables
2. Click Sign Off to open `<dialog id="signoff-modal">`
3. Press J, K, A, O while modal is open

**Expected:** No navigation or approve fires — shortcuts suppressed when a `<dialog>` element is open. Console logs should NOT appear for these keypresses.

---

### Auto-advance wraps to front when all remaining pending items are before current index

1. Start at item 3 of 4, items 1 and 2 already approved
2. Approve item 3 (press A)

**Expected:** Auto-advance finds item 4 (next forward from index 2), shows it. If item 4 is the last pending, approve it → `[queue] afterSwap: no more pending items` in console; stays on current card.

---

### Override note required validation

1. Open override panel on any pending item
2. Clear the textarea (leave it empty)
3. Submit the form

**Expected:** HTML5 native `required` validation fires — browser shows validation tooltip, form does NOT submit, HTMX POST does NOT fire.

---

## Failure Signals

- **No `[queue]` console logs on J/K/A/O** — keyboard handler not bound; check for JS parse errors in console (likely syntax error in IIFE)
- **Progress bar counters stale after approve** — OOB swap response missing `id="progress-bar"` wrapper; inspect HTMX response body in Network tab
- **Auto-advance never fires** — `afterSwap` guard broken; confirm `e.detail.target.id.startsWith('review-item-')` present in queue.html; check D015
- **Override panel doesn't open** — `data-char-no` attribute missing on item card or `override-panel-{char_no}` ID mismatch; inspect DOM in DevTools
- **Sign Off button never enables** — OOB swap for `#signoff-footer` not firing, or pending counter logic wrong in `_signoff_footer_oob.html`; check `pending == 0` condition in template
- **DaisyUI class artifacts visible (card bg-base-100, alert alert-*)** — grep check failed or template not rebuilt; run `npm run build:css` and hard-reload browser

## Requirements Proved By This UAT

- R009 — Full-screen focused review queue: one item visible at a time, large snippets, no scrolling
- R010 — Keyboard shortcuts: J/K navigation, A approve, O override with auto-focus; shortcuts suppressed in inputs and dialogs
- R011 — Sign-off footer and progress bar: OOB swaps live, Sign Off CTA enables when all resolved, modal fires

## Not Proven By This UAT

- R001–R008 — Validated in prior slices (S01–S03); not retested here
- R012 — Admin screens redesign (S05 scope)
- R013 — Full fragment cleanup and visual consistency audit (S05 scope)
- Pipeline correctness — S04 is UI-only; backend review logic is unchanged and covered by existing pytest suite

## Notes for Tester

- The sign-off confirm step will fail on a test run with no real pipeline output (`delta_packet.json` missing) — this is expected. The app correctly shows an error banner and rolls the run back to `reviewing` state. Test the modal open/confirm flow anyway — the redirect behavior is the proof, not the final pipeline state.
- If J/K navigation feels sluggish, check whether items are rendering snippet images. Large PNGs from a real run may cause layout reflow — this is acceptable; focused mode still works.
- The keyboard legend is `pointer-events-none` and low-opacity — it should not block any UI interaction. Verify it's visible but doesn't interfere with clicking approve/override buttons.
- `generating.html` is only shown during the async sign-off generation phase. To test it, look for it after clicking "Start Review" on a run that just completed the pipeline.
