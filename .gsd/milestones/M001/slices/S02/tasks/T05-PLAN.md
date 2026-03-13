# T05: 02-pipeline-bridge 05

**Slice:** S02 — **Milestone:** M001

## Description

Build the run status page with real-time stage progress via SSE and all three terminal/warning UI states.

Purpose: After submission, engineers need to see live stage progress and clear outcomes for failed, warning, and completed runs. The SSE endpoint bridges the Huey worker's DB updates to the browser in real time.
Output: GET /runs/{id} status page, GET /runs/{id}/sse streaming endpoint, POST /runs/{id}/abort, status.html with three distinct warning states, _stage_checklist.html HTMX partial.

## Must-Haves

- [ ] "Run status page shows numbered stage checklist with checkmark (completed), spinner (current), dim (pending)"
- [ ] "SSE endpoint streams stage updates to status page without page reload"
- [ ] "Failed run shows Failed badge + failing stage highlighted in red + plain-language error message"
- [ ] "RevB balloon warning shows yellow warning banner with Acknowledge and Open Review Queue button"
- [ ] "Low-confidence alignment shows inline warning section with confidence summary + Proceed/Abort buttons"
- [ ] "SSE connection closes when run reaches terminal state (no reconnect hammering)"

## Files

- `shop/routers/runs.py`
- `shop/templates/runs/status.html`
- `shop/templates/runs/_stage_checklist.html`
