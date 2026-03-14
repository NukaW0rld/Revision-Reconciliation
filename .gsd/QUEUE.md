# GSD Milestone Queue

## Queued Milestones

| ID | Title | Queued At | Dependencies |
|----|-------|-----------|--------------|
| M002 | Setup Simplification & Per-Run Column Mapping | 2026-03-13 | M001 complete |

---

### M002: Setup Simplification & Per-Run Column Mapping

**Queued:** 2026-03-13

**Summary:** Reduce first-run setup wizard from 4 steps to 2 (shop name + admin password with `admin@shop.local` displayed). Remove engineer-creation and column-mapping wizard steps — engineer accounts are managed via the existing admin dashboard. Move Form 3 column mapping from a one-time global setup step to an inline per-run confirmation on the new run form (HTMX trigger on xlsx upload, auto-detect + correction UI, confirmed mapping submitted with run POST). Remove column mapping from admin settings page and router.

**Context file:** `.gsd/milestones/M002/M002-CONTEXT.md`
