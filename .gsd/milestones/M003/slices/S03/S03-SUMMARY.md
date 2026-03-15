---
id: S03
parent: M003
milestone: M003
provides:
  - Redesigned status.html with industrial bordered-section aesthetic and border-l-4 banner pattern
  - SSE JS injectTerminalBanner() strings updated to match Jinja2 border-l-4 pattern
  - page_title block added for top-bar context injection
requires:
  - slice: S01
    provides: base.html sidebar+top-bar layout, output.css with dark industrial tokens
affects:
  - S05
key_files:
  - shop/templates/runs/status.html
  - shop/templates/runs/_stage_checklist.html
key_decisions:
  - JS injectTerminalBanner() strings updated to match Jinja2 border-l-4 pattern exactly, maintaining SSE/hard-reload visual parity
  - _stage_checklist.html left structurally unchanged; only its card wrapper in status.html was replaced
patterns_established:
  - border border-base-300 bg-base-200 p-4 section wrapper with text-base font-semibold font-mono mb-3 heading
  - border-l-4 border-{color} bg-base-200 p-4 flex items-start gap-3 status banner (replaces alert alert-{type})
observability_surfaces:
  - grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html → 0
  - grep -c 'alert alert-' shop/templates/runs/status.html → 0
  - grep -c 'border-l-4' shop/templates/runs/status.html → 7
  - "#terminal-banner-slot shows border-l-4 HTML after SSE close event (devtools Elements panel)"
drill_down_paths:
  - .gsd/milestones/M003/slices/S03/tasks/T01-SUMMARY.md
duration: 15m
verification_result: passed
completed_at: 2026-03-14
---

# S03: Pipeline Status & Run Detail

**Replaced all card/shadow wrappers and DaisyUI alert banners in status.html with bordered sections and border-l-4 status banners; JS SSE injection strings updated to match; SSE, DOM IDs, and steps-vertical checklist preserved intact.**

## What Happened

T01 was the single task in this slice. Read the 287-line `status.html` and identified three card wrappers (Pipeline Stages, Run Details, Packet Versions), five Jinja2 `alert alert-*` instances (signed-off success, failed, two warning variants, and a meta-info alert), and three JS banner strings inside `injectTerminalBanner()`.

The full redesign:
- `{% block page_title %}Run #{{ run.id }}{% endblock %}` added for top-bar context injection
- Three card wrappers → `border border-base-300 bg-base-200 p-4` with `text-base font-semibold font-mono mb-3` headings
- Five Jinja2 `alert alert-*` → `border-l-4 border-{success/error/warning} bg-base-200 p-4 flex items-start gap-3` with inline SVG icons aligned via `text-{color} mt-0.5`
- Three `injectTerminalBanner()` JS strings updated to match the same `border-l-4` class structure — SSE-injected and hard-reload banners are now visually identical
- Back-link and main heading styled `font-mono`
- `_stage_checklist.html` left entirely unchanged — `steps steps-vertical`, `id="stage-checklist"`, DaisyUI step classes, and `updateChecklist()` all preserved
- DaisyUI modal (`modal-box`, `modal-backdrop`) untouched

## Verification

```
grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html  → 0
grep -c 'alert alert-' shop/templates/runs/status.html                                    → 0
grep -c 'border-l-4' shop/templates/runs/status.html                                      → 7
grep 'id="stage-checklist"' shop/templates/runs/_stage_checklist.html                     → match
grep 'id="terminal-banner-slot"' shop/templates/runs/status.html                          → match
grep 'block page_title' shop/templates/runs/status.html                                   → match
npm run build:css                                                                          → exit 0, Done in 193ms
uv run pytest tests/                                                                       → 93 passed, 2 xfailed, 8 warnings
```

## Requirements Advanced

- R007 — Pipeline status page redesigned: all status states (running/failed/warning/completed/signed-off) use border-l-4 industrial banners; SSE JS updated to match

## Requirements Validated

- R007 — `status.html` fully redesigned: zero card/shadow artifacts, zero legacy DaisyUI alert classes, all DOM IDs preserved, SSE JS updated, CSS builds clean, 93 tests pass

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

none — plan was executed as written; no deviations required

## Known Limitations

- Browser-level visual verification (SSE live update during an active pipeline run) is left to human UAT — the test suite cannot exercise SSE in pytest without a live Huey worker
- R008 (run list page redesign) was validated in S02, not S03 as the requirements file originally stated as its "primary owning slice"

## Follow-ups

- none

## Files Created/Modified

- `shop/templates/runs/status.html` — full redesign: card → bordered sections, alert → border-l-4, page_title block added, JS banners updated
- `shop/templates/runs/_stage_checklist.html` — unchanged (already correct)
- `.gsd/milestones/M003/slices/S03/S03-PLAN.md` — Observability / Diagnostics section added pre-execution

## Forward Intelligence

### What the next slice should know
- The `border-l-4 border-{color} bg-base-200 p-4 flex items-start gap-3` banner pattern is now established in two places: Jinja2 template and inline JS strings — any future status-state additions must update both
- `_stage_checklist.html` DOM structure and all `id="stage-checklist"` references must be preserved verbatim; S04 and S05 have no reason to touch this file
- `id="terminal-banner-slot"` in `status.html` must survive any S05 cleanup pass — SSE JS writes to it by ID

### What's fragile
- JS `injectTerminalBanner()` strings are literal HTML inside a `<script>` block — not Jinja2-aware; any CSS class change to the Jinja2 banners must be manually mirrored in the JS strings or the SSE-injected banner will diverge visually
- DaisyUI `steps steps-vertical` step classes (`step-success`, `step-primary`, `step-error`) must remain in scope of the Tailwind JIT scan; `_stage_checklist.html` is included from `status.html` so the scan path is correct as long as the include survives

### Authoritative diagnostics
- `grep -c 'alert alert-' shop/templates/runs/status.html` → 0 is the fastest regression check; any non-zero count means a legacy class survived
- `grep -c 'border-l-4' shop/templates/runs/status.html` → 7 confirms all five Jinja2 banners and three JS strings have the accent class

### What assumptions changed
- Assumed `_stage_checklist.html` would need wrapping changes — it did not; the card wrapper was entirely in `status.html`, so the partial was left untouched
