# S03: Pipeline Status & Run Detail — Research

**Date:** 2026-03-14

## Summary

S03 owns two templates: `status.html` (the live pipeline progress page) and `_stage_checklist.html` (the SSE-driven stage list partial). Both currently use generic DaisyUI card wrappers, `shadow-sm`, `card-body`, and `alert alert-{type}` components. The page is the most dynamic in the app — an inline `<script>` listens for SSE events and rewrites `#stage-checklist` innerHTML and `#terminal-banner-slot` using hardcoded DaisyUI class names. Any redesign must leave those DOM IDs and class names untouched unless the JS is also rewritten in sync.

The canonical S02 patterns (`border border-base-300 bg-base-200 p-4/p-5` section wrappers, `font-mono` headings) apply cleanly to the two info cards (Pipeline Stages and Run Details). The DaisyUI `steps steps-vertical` component compiles correctly with the industrial theme — step-success, step-primary, and step-error all resolve to the correct OKLCH colors — so the stage checklist can stay on DaisyUI steps without the D009 concern that killed wizard steps (wizard steps used pseudo-element counters against a near-black bg; this is a vertical progress indicator with a different rendering model). The `alert-{type}` status banners (failed/warning/completed) are theme-aware and resolve to the industrial palette. A decision is needed on whether to keep the full colored alert boxes or unify with the S02 `border-l-4` pattern — both are defensible; the key risk is that the SSE JS also hardcodes `alert alert-error`, `alert alert-warning`, and `alert alert-success` HTML strings inside `injectTerminalBanner()`, so if the Jinja2 static states switch to `border-l-4` the JS-injected banner will still render as a full colored box until the page reloads.

The modal (Amend confirmation dialog) uses DaisyUI `<dialog>` with `modal-box`, which renders with `--color-base-100` (our near-black) and `--radius-box: 0` (hard edges from theme) — already correct for the industrial aesthetic with no special handling needed. The one awkward element is the Packet Versions card (`card bg-base-100 shadow-sm`) which contains a simple link list — straight conversion to bordered section.

## Recommendation

Perform a CSS-class-only redesign (no structural HTML changes, no backend changes):

1. Replace all three `card bg-base-100 shadow-sm` + `card-body` + `card-title` patterns with the S02-canonical `border border-base-300 bg-base-200 p-4/p-5` section wrapper + `text-base font-semibold font-mono mb-3` heading.
2. Keep DaisyUI `steps steps-vertical` in `_stage_checklist.html` and the SSE JS — the component compiles correctly, and rewriting to custom elements would also require rewriting the JS string concatenation in `injectTerminalBanner()` and `updateChecklist()`.
3. **Status banners (alert-error / alert-warning / alert-success):** Unify with the S02 `border-l-4` pattern in Jinja2 markup AND in the SSE JS `injectTerminalBanner()` string. This keeps the pre-reload injected banner visually consistent with the post-reload rendered banner. Both must change together or not at all.
4. DaisyUI modal stays as-is — `modal-box` already uses our theme vars; no intervention needed.
5. Add `{% block page_title %}Run #{{ run.id }} — {{ run.part_number }}{% endblock %}` to the top bar breadcrumb slot (required pattern from S01/S02).

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Stage progress indicator | `ul.steps.steps-vertical` + `step-success/primary/error` | Already compiles with industrial theme; SSE JS already writes these class names; rewriting both is extra risk with no aesthetic gain |
| Amend confirmation dialog | DaisyUI `<dialog>` with `modal-box` | Renders with `--color-base-100` (near-black) and `--radius-box: 0` (hard edges) — already correct |
| DaisyUI loading spinner | `loading loading-spinner loading-xs` | Used in SSE JS string for the current stage indicator; keep for same reason as steps |
| Status badge | `badge badge-{{ run.status \| status_badge_class }}` | Filter is defined in `app.py`; maps each status to a DaisyUI semantic color class that resolves through theme vars |

## Existing Code and Patterns

- `shop/templates/runs/status.html` — 287-line template; 10 occurrences of card/shadow to replace. Contains inline `<script>` with SSE listener, `injectTerminalBanner()`, and `updateChecklist()`. DOM IDs `stage-checklist` and `terminal-banner-slot` are load-bearing.
- `shop/templates/runs/_stage_checklist.html` — 22-line server-rendered partial; included via `{% include %}` inside the Pipeline Stages section. Its `id="stage-checklist"` is targeted by the SSE JS. Uses `steps steps-vertical` which compiles correctly.
- `shop/templates/runs/_alert_banner.html` — Already redesigned in S02 with `border-l-4 border-error bg-base-200` pattern. Not in S03 scope.
- `shop/templates/runs/list.html` — Completed in S02; provides the table/mono-header pattern reference.
- `static/src/input.css` — Industrial theme vars: `--color-success`, `--color-warning`, `--color-error` all defined. `--radius-box: 0` means all DaisyUI components have hard edges.
- `shop/app.py::_status_badge_class()` — Maps run status strings to DaisyUI semantic color suffixes. Must not change.
- `tests/test_runs.py:463` — Asserts `"stage-checklist" in resp.text`. This DOM ID must survive the redesign.

## Constraints

- **`id="stage-checklist"` is hard-required.** The test at `tests/test_runs.py:463` asserts the string is in the page response. The SSE JS `updateChecklist()` calls `document.getElementById("stage-checklist")`. Rename = test failure + silent SSE breakage.
- **`id="terminal-banner-slot"` must be preserved.** The SSE `close` event handler calls `injectTerminalBanner()` which targets this ID to inject pre-reload status feedback.
- **SSE JS class names must match rendered Jinja2 class names.** If `alert alert-error` is replaced with `border-l-4 border-error bg-base-200` in the Jinja2 static-state blocks, the same replacement must happen in the `injectTerminalBanner()` JS strings — otherwise the pre-reload injected banner and the post-reload rendered banner look different.
- **`{% include "runs/_stage_checklist.html" %}` must remain.** The include is the mechanism for server-side rendering the initial checklist state. No restructuring of this relationship.
- **No backend changes.** `status.html` router passes: `user`, `run`, `stages`, `engineers`, `signed`, plus `**nav` (shop_name, unread_alert_count). Template must continue to consume these same vars.
- **DaisyUI semantic classes through `status_badge_class` filter must not change.** The filter maps to DaisyUI suffixes (success/error/warning/info/ghost); these resolve through `--color-*` vars in the industrial theme.
- **`npm run build:css` must stay clean** — any new utility class added must be in Tailwind's scan path (templates are already scanned).

## Common Pitfalls

- **Replacing steps without updating JS.** If `_stage_checklist.html` switches from `<li class="step step-success">` to custom divs, `updateChecklist()` in the SSE script still writes `<li class="step ...">` HTML — the DOM on the first SSE update will snap back to DaisyUI steps. They must change together.
- **Mismatched pre-reload vs post-reload alert appearance.** `injectTerminalBanner()` injects HTML directly into `#terminal-banner-slot`. If the Jinja2 state blocks (for failed/warning/completed) are redesigned but the JS strings are not updated, the banner flickers from new design → old design when the page reloads 500ms later.
- **`alert alert-success` signed banner at top.** This is a `{% if signed %}` block for post-sign-off confirmation. It must get the same treatment as the other status alerts (whichever direction is chosen).
- **Modal backdrop rendering.** The `<form method="dialog" class="modal-backdrop">` close button is DaisyUI-specific. Do not remove it — it's what allows clicking outside the modal to close it.
- **Missing `{% block page_title %}`.** Both S01 and S02 set this block. If status.html omits it, the top bar shows nothing in the breadcrumb slot. Include: `{% block page_title %}Run #{{ run.id }}{% endblock %}`.
- **`card-title` inside `card-body` produces implicit spacing.** When `card-body` is removed, ensure heading margins are set explicitly with `mb-3` or `mb-2` on the heading element — otherwise content runs flush.

## Open Risks

- **`alert alert-{type}` vs `border-l-4` consistency decision.** If the full colored alert boxes are kept for status-state alerts (which is the path of least resistance), the injectTerminalBanner JS strings are unchanged. If border-l-4 is chosen for consistency, the JS strings must also be updated — and the change must be verified by simulating a running → terminal transition in a browser. This is the only genuinely risky divergence in this slice.
- **DaisyUI steps-vertical visual validation needed.** The `steps-vertical` component was not visually verified in prior slices (only `ul.steps` horizontal and the abandoned wizard `ul.steps`). Worth a quick browser check after applying the new layout to confirm step-success/primary/error colors render correctly against the dark `bg-base-200` section background.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Tailwind CSS v4 / DaisyUI v5 | — | no additional skill needed (patterns established in S01/S02) |
| HTMX | — | no skill needed (this slice has no HTMX swap targets; SSE is already working) |

## Sources

- Examined directly: `shop/templates/runs/status.html`, `_stage_checklist.html`, `_alert_banner.html`, `shop/app.py`, `shop/routers/runs.py`, `static/src/input.css`, `static/dist/output.css`, `tests/test_runs.py`
- S01 and S02 summaries confirm: section wrapper pattern, font-mono heading pattern, page_title block requirement, DaisyUI semantic token resolution through industrial theme vars
