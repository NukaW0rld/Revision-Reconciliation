# M001: Migration

**Vision:** Delta Preservation is a web application for aerospace machine shop quality control teams that automates the most painful step in drawing revision management: determining exactly what changed between two engineering drawing revisions before the shop can ship again.

## Success Criteria


## Slices

- [x] **S01: Foundation** `risk:medium` `depends:[]`
  > After this: Create the test infrastructure scaffold for Phase 1 before any production code is written.
- [x] **S02: Pipeline Bridge** `risk:medium` `depends:[S01]`
  > After this: Lay the database and task queue foundation that every other Phase 2 plan depends on.
- [x] **S03: Review And Sign Off** `risk:medium` `depends:[S02]`
  > After this: Wave 1 prerequisite scaffold: test stubs, ReviewItem DB model, and pipeline patch for removed-item Rev B bboxes.
- [x] **S04: Exports History And Amendments** `risk:medium` `depends:[S03]`
  > After this: Install WeasyPrint, add Dockerfile system dependencies, add new DB columns for amendment and packet versioning, add startup schema migration, and create xfail test stubs for all Phase 4 requirements.
