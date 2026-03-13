# T04: 02-pipeline-bridge 04

**Slice:** S02 — **Milestone:** M001

## Description

Add the persistent nav bar and update the dashboard with run visibility and submission CTA.

Purpose: Engineers need consistent navigation and run visibility from every page. The nav bar (locked decision) and dashboard update are foundational UI changes that all other Phase 2 templates extend.
Output: Updated base.html with nav bar, updated dashboard.html with run cards and Submit CTA, new runs/list.html, GET /runs route in runs router.

## Must-Haves

- [ ] "Every page has a persistent nav bar with: shop name, New Run link, My Runs link, alert bell icon, user email, logout"
- [ ] "Dashboard shows 5-10 most recent runs as compact cards with part number, revision letters, status badge, date, reviewer name"
- [ ] "Dashboard has a prominent Submit New Run CTA button linking to /runs/new"
- [ ] "Run list at /runs shows all runs filterable by part number and date"

## Files

- `shop/templates/base.html`
- `shop/templates/dashboard.html`
- `shop/routers/runs.py`
- `shop/templates/runs/list.html`
