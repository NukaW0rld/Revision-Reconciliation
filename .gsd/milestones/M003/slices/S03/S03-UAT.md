# S03: Pipeline Status & Run Detail — UAT

**Milestone:** M003
**Written:** 2026-03-14

## UAT Type

- UAT mode: live-runtime
- Why this mode is sufficient: The key deliverable is visual (CSS classes on status banners and section wrappers) and behavioral (SSE JS updating #terminal-banner-slot). Automated tests cover backend behavior; visual parity between Jinja2-rendered and SSE-injected banners requires a running browser. The pytest suite (93 passed) guards backend and route correctness.

## Preconditions

- App running: `uv run python run_web.py` (default port 8000) with Huey worker active (`uv run huey_consumer shop.tasks.huey`)
- At least one completed run exists in the database, or tester can submit a new run via the New Run form
- Tester is logged in as an engineer or admin user
- Browser dev tools open (Elements + Network + Console tabs)
- A run in each status state is available for testing (completed, failed, warning, signed-off) — or tester accepts viewing the states reachable from available data

## Smoke Test

Navigate to `/runs/<id>` for any run. The page must load with a dark background, bordered section wrappers (not white cards), and a `font-mono` heading. If you see a white card or the old `alert alert-*` colored banners (blue/green/red DaisyUI blocks), the slice has regressed.

## Test Cases

### 1. Bordered section wrappers replace card/shadow wrappers

1. Navigate to `/runs/<id>` for any run.
2. Inspect the Pipeline Stages section, Run Details section, and Packet Versions section.
3. **Expected:** Each section is wrapped in a dark bordered box (`border border-base-300 bg-base-200`) with a `font-mono` heading. No white `card` backgrounds or box shadows visible. No `card-body` padding artifacts.

### 2. Completed run status banner

1. Navigate to `/runs/<id>` for a run with status `completed`.
2. **Expected:** A left-accent banner appears at the top of the page with a green left border (`border-l-4 border-success`), dark background, an SVG checkmark icon, and message text. The banner does NOT use a DaisyUI `alert alert-success` pill/box style.

### 3. Failed run status banner

1. Navigate to `/runs/<id>` for a run with status `failed`.
2. **Expected:** A left-accent banner with red left border (`border-l-4 border-error`), dark background, SVG X icon, and the failure message. No `alert alert-error` styling.

### 4. Warning run status banner

1. Navigate to `/runs/<id>` for a run with status `warning`.
2. **Expected:** A left-accent banner with amber left border (`border-l-4 border-warning`), dark background, SVG warning icon, and the warning message. If a run alert exists, a second warning banner lists the alert detail below. No `alert alert-warning` styling.

### 5. Signed-off run status banner

1. Navigate to `/runs/<id>` for a run with status `signed_off`.
2. **Expected:** A left-accent success banner at top (signed-off confirmation) plus a bottom modal trigger for the Amend button. The modal (`modal-box`, `modal-backdrop`) opens when Amend is clicked and functions normally.

### 6. SSE live stage checklist updates

1. Submit a new run via New Run form and immediately navigate to its `/runs/<id>` status page while the pipeline is running.
2. Watch the Pipeline Stages section (the `steps steps-vertical` checklist).
3. **Expected:** Stage steps update in real-time (step class changes to `step-success`, `step-error`, or `step-primary`) as the Huey worker progresses. The `#stage-checklist` element is patched live via `updateChecklist()` without a page reload.

### 7. SSE terminal banner injection matches Jinja2 banner style

1. Let the active run from Test Case 6 complete (or fail).
2. When the SSE `close` event fires, the `#terminal-banner-slot` div is populated by `injectTerminalBanner()`.
3. **Expected:** The injected banner uses the same `border-l-4 border-{success/error/warning} bg-base-200 p-4 flex items-start gap-3` class structure as the Jinja2-rendered banners. Hard-reload the page and compare — the banner appearance must be visually identical.

### 8. Back-link and page title in top bar

1. Navigate to `/runs/<id>`.
2. **Expected:** The browser tab title includes "Run #\<id\>". The page-level heading or top-bar context area shows the run ID in `font-mono`. A back-link to the runs list is present and styled monospacedly.

### 9. Running status — no terminal banner shown

1. Navigate to `/runs/<id>` for a run with status `running` (or `queued`).
2. **Expected:** No terminal status banner appears at the top. The SSE connection is established (visible in Network tab as an EventStream). The stage checklist shows the current progress state.

### 10. Page title block injection

1. Navigate to `/runs/<id>`.
2. Inspect the top bar (sidebar layout header area).
3. **Expected:** The `{% block page_title %}` value ("Run #\<id\>") is rendered in the top-bar context slot provided by `base.html`. This confirms the page_title block is present and wired to the base layout.

## Edge Cases

### Run with zero alerts (warning status)

1. Navigate to a `warning` status run that has no `RunAlert` records.
2. **Expected:** The primary warning banner renders correctly without the secondary alert-detail banner. No template error or empty div artifacts.

### Run with multiple alerts (warning status)

1. Navigate to a `warning` status run that has multiple `RunAlert` records.
2. **Expected:** Each alert renders as a separate `border-l-4 border-warning` banner below the primary warning banner. All banners maintain visual consistency.

### Hard reload after SSE terminal banner injection

1. Let a run complete and observe the SSE-injected terminal banner in `#terminal-banner-slot`.
2. Hard-reload the page (Ctrl+Shift+R / Cmd+Shift+R).
3. **Expected:** The Jinja2-rendered terminal banner (not JS-injected) appears in the same visual style. The two render paths are indistinguishable to the eye.

## Failure Signals

- White card backgrounds or box shadows visible on any section → card/shadow classes survived cleanup
- Colored pill-shaped alert blocks (blue/green/red DaisyUI `alert` style) → `alert alert-*` classes survived
- `#terminal-banner-slot` remains empty after SSE close event → JS `injectTerminalBanner()` not wiring correctly or SSE not firing
- SSE-injected banner looks different from Jinja2 banner (different padding, color, or border) → JS strings were not updated to match Jinja2 changes
- `#stage-checklist` does not update during an active run → `updateChecklist()` SSE handler broken or DOM ID changed
- Modal Amend button does not open on signed-off run → modal markup regressed

## Requirements Proved By This UAT

- R007 — Pipeline status page redesigned with industrial bordered-section aesthetic, all status states using border-l-4 banners, SSE live updates functional

## Not Proven By This UAT

- SSE behavior under network interruption or reconnection (not in scope for M003)
- Performance of SSE with very long-running pipelines (not in scope)
- Accessibility/keyboard navigation of the status page (not in scope for S03)

## Notes for Tester

- The most important check is Test Case 7 (SSE banner vs Jinja2 banner parity) — this is the only place where two separate code paths must produce visually identical output, and it cannot be verified by the pytest suite alone.
- If you don't have a Huey worker running, you can still verify all terminal status banners (Test Cases 2–5, 8) by navigating to runs with the appropriate statuses from your existing database.
- Test Case 9 (running status) requires an active Huey worker. If unavailable, skip and note it.
- The DaisyUI `steps steps-vertical` checklist appearance depends on `output.css` being current — run `npm run build:css` if steps look unstyled.
