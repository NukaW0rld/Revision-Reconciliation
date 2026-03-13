# T06: 02-pipeline-bridge 06

**Slice:** S02 — **Milestone:** M001

## Description

Wire up in-app failure alerts so the assigned reviewer sees unread failure notifications on the dashboard and in the nav bar badge.

Purpose: PIPE-11 requires the assigned reviewer to receive an in-app alert when their run fails. The RunAlert model and alert creation are in place (Plans 01+03). This plan surfaces those alerts in the UI.
Output: Dashboard alert banners for unread failures, bell badge count in nav bar, HTMX dismiss endpoint, POST /runs/{id}/acknowledge-warning for the revB_balloon warning gate.

## Must-Haves

- [ ] "Bell icon in nav bar shows red badge with unread alert count when alerts exist"
- [ ] "Unread failure alerts appear as a dismissible banner at the top of the dashboard"
- [ ] "Clicking the dashboard banner dismisses it and clears the badge for that alert"
- [ ] "Only the assigned reviewer sees failure alerts for their runs"
- [ ] "Alert shows part number, run ID, failure stage, and a View run link"

## Files

- `shop/routers/runs.py`
- `shop/templates/dashboard.html`
- `shop/templates/runs/_alert_banner.html`
