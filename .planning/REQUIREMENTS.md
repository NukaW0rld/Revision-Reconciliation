# Requirements: Delta Preservation

**Defined:** 2026-04-09
**Core Value:** Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.

## v1 Requirements

### Ground Truth

- [ ] **GTRU-01**: Admin debug workflow can load the correct `ground_truth.json` for a completed part run
- [ ] **GTRU-02**: Admin debug workflow shows actionable validation errors when ground truth is missing, malformed, or incomplete for a run
- [ ] **GTRU-03**: Ground truth remains read-only during debug evaluation and manual review

### Evaluation

- [ ] **EVAL-01**: System automatically compares each completed review item against the expected ground-truth classification
- [ ] **EVAL-02**: System automatically compares each completed review item against the expected ground-truth Rev B requirement
- [ ] **EVAL-03**: System evaluates Rev A and Rev B snippet evidence with tolerance rules that accept visually correct context rather than exact center-coordinate matches
- [ ] **EVAL-04**: System auto-marks a characteristic as correct when its classification, requirement, and snippet evidence conform to accepted truth rules
- [ ] **EVAL-05**: System marks a characteristic as needing review when classification, requirement, snippet evidence, or truth data does not conform, and records explicit mismatch reasons

### Debug Review

- [ ] **DREV-01**: Admin debug reviewer can open a queue focused on only nonconforming or ambiguous characteristics for a run
- [ ] **DREV-02**: Admin debug reviewer can still inspect auto-passed characteristics in the run details or exported debug report
- [ ] **DREV-03**: Admin debug reviewer can record whether a nonconforming characteristic is an algorithm error or an acceptable alternate outcome
- [ ] **DREV-04**: Admin debug reviewer can attach rationale for any nonconforming characteristic that is not simply marked correct

### Exceptions History

- [ ] **HIST-01**: System stores acceptable alternate outcomes in a separate exceptions/history layer instead of editing `ground_truth.json`
- [ ] **HIST-02**: Each exceptions/history record stores run identity, part identity, characteristic identity, reviewed outcome, and rationale
- [ ] **HIST-03**: System can treat a previously approved acceptable alternate outcome for the same part and characteristic as conforming in a later run

### Reporting

- [ ] **RPT-01**: System generates a `debug_report.json` for each evaluated run without requiring manual verdict entry for auto-passed characteristics
- [ ] **RPT-02**: `debug_report.json` distinguishes canonical ground-truth matches, acceptable alternate matches, and unresolved review-needed rows
- [ ] **RPT-03**: `debug_report.json` includes mismatch reasons and any linked exceptions/history references for rows that need review

## v2 Requirements

### Consistency

- **CONS-01**: System detects contradictions between accepted outcomes across different parts and runs
- **CONS-02**: System highlights potential cross-part overfitting patterns before a reviewer accepts a new alternate outcome
- **CONS-03**: System summarizes benchmark performance trends across many runs and parts

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auto-edit `ground_truth.json` during evaluation or review | Canonical truth must remain stable and manually curated |
| Automatic algorithm self-tuning from debug history | This milestone stops at debugging infrastructure, not a learning loop |
| Multi-user debug collaboration features | The workflow is for a solo developer, not general product users |
| Cross-part contradiction analysis in the initial milestone | Important, but intentionally deferred until the history layer exists |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GTRU-01 | Phase 1 | Pending |
| GTRU-02 | Phase 1 | Pending |
| GTRU-03 | Phase 1 | Pending |
| EVAL-01 | Phase 1 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 1 | Pending |
| EVAL-04 | Phase 1 | Pending |
| EVAL-05 | Phase 1 | Pending |
| DREV-01 | Phase 2 | Pending |
| DREV-02 | Phase 2 | Pending |
| DREV-03 | Phase 2 | Pending |
| DREV-04 | Phase 2 | Pending |
| HIST-01 | Phase 3 | Pending |
| HIST-02 | Phase 3 | Pending |
| HIST-03 | Phase 3 | Pending |
| RPT-01 | Phase 2 | Pending |
| RPT-02 | Phase 2 | Pending |
| RPT-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after initialization*
