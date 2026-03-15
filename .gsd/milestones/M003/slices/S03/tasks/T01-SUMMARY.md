---
id: T01
parent: S03
milestone: M003
provides:
  - Redesigned status.html with industrial bordered-section aesthetic and border-l-4 banner pattern
  - Observability section added to S03-PLAN.md and T01-PLAN.md
key_files:
  - shop/templates/runs/status.html
  - shop/templates/runs/_stage_checklist.html (unchanged — already correct)
key_decisions:
  - JS injectTerminalBanner() strings updated to match Jinja2 border-l-4 pattern exactly, maintaining SSE/hard-reload visual parity
  - _stage_checklist.html left structurally unchanged; only its card wrapper in status.html was replaced
patterns_established:
  - border border-base-300 bg-base-200 p-4 section wrapper with text-base font-semibold font-mono mb-3 heading
  - border-l-4 border-{color} bg-base-200 p-4 flex items-start gap-3 status banner (replaces alert alert-{type})
observability_surfaces:
  - grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html → 0
  - grep -c 'alert alert-' shop/templates/runs/status.html → 0
  - grep -c 'border-l-4' shop/templates/runs/status.html → 7 (one per status-state banner)
  - devtools Elements panel: #terminal-banner-slot shows border-l-4 HTML after SSE close event
duration: 15m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Redesign status.html and _stage_checklist.html with industrial aesthetic

**Replaced all card/shadow wrappers with bordered sections and all alert alert-* banners with border-l-4 pattern in both Jinja2 and JS.**

## What Happened

Read the 287-line `status.html` and identified: 3 card wrappers (Pipeline Stages, Run Details, Packet Versions), 5 Jinja2 `alert alert-*` instances (signed-off success, failed, two warning variants), and 3 JS banner strings inside `injectTerminalBanner()`. Preflight observability gaps were patched in S03-PLAN.md and T01-PLAN.md before coding.

Rewrote `status.html` with:
- `{% block page_title %}Run #{{ run.id }}{% endblock %}` added after `{% block title %}`
- Three card wrappers → `border border-base-300 bg-base-200 p-4` + `text-base font-semibold font-mono mb-3` headings
- Five Jinja2 `alert alert-*` → `border-l-4 border-{success/error/warning} bg-base-200 p-4 flex items-start gap-3` with inline SVG icons carrying `text-{color} mt-0.5` for vertical alignment
- Three JS banner strings in `injectTerminalBanner()` updated to match the same `border-l-4` class structure
- Back-link styled `font-mono`, heading `font-mono`
- `_stage_checklist.html` left entirely unchanged — `steps steps-vertical`, `id="stage-checklist"`, and `updateChecklist()` all preserved

## Verification

```
grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html → 0
grep -c 'alert alert-' shop/templates/runs/status.html → 0
grep 'id="stage-checklist"' shop/templates/runs/_stage_checklist.html → match
grep 'id="terminal-banner-slot"' shop/templates/runs/status.html → match
grep 'block page_title' shop/templates/runs/status.html → match
grep -c 'border-l-4' shop/templates/runs/status.html → 7
npm run build:css → exit 0, Done in 244ms
uv run pytest tests/ → 93 passed, 2 xfailed, 8 warnings
```

All slice-level verification checks pass. This is the only task in S03, so the slice is complete.

## Diagnostics

- `grep -c 'border-l-4' shop/templates/runs/status.html` should return 7 (signed-off top banner + failed + two warning + three JS strings)
- `grep -c 'alert alert-' shop/templates/runs/status.html` should return 0; any non-zero count signals reversion
- SSE banner parity: load a running run, watch `#terminal-banner-slot` in devtools Elements after SSE close — class structure should be `border-l-4 border-{color} bg-base-200 p-4 flex items-start gap-3`

## Deviations

none

## Known Issues

none

## Files Created/Modified

- `shop/templates/runs/status.html` — full redesign: card → bordered sections, alert → border-l-4, page_title block added, JS banners updated
- `.gsd/milestones/M003/slices/S03/S03-PLAN.md` — added Observability / Diagnostics section (preflight fix)
- `.gsd/milestones/M003/slices/S03/tasks/T01-PLAN.md` — added Observability Impact section (preflight fix)
