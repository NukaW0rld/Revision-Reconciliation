# Feature Research

**Domain:** Document review and sign-off web application for aerospace QC (AS9102 FAIR compliance)
**Researched:** 2026-03-01
**Confidence:** MEDIUM — core audit/sign-off patterns verified against industry tools (GroundControl, Net-Inspect, InspectionXpert) and regulatory references (21 CFR Part 11, AS9102C); specific UX anti-patterns drawn from adjacent domains (content moderation, pharma QMS, eQMS) with lower direct citation

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or legally insufficient.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Per-item review card with visual evidence | QC engineers cannot classify without seeing what they're approving; image-free review is audit-worthless | MEDIUM | Side-by-side Rev A / Rev B snippets are the core interaction; layout must be unambiguous about which is which |
| Mandatory reason on override | Every FAI-aware QC tool requires documented justification for any deviation from the system's finding; silent overrides fail audit | LOW | Single required text field per override; must block submit if empty |
| All items must reach terminal state before sign-off | Partial sign-off creates ambiguous coverage claims; industry standard for FAIR is 100% characteristic accountability | MEDIUM | Hard gate, not a soft warning; show remaining count prominently |
| Immutable signed packet | Post-sign-off modification of a FAIR is a compliance violation in aerospace QMS; records must be tamper-evident | HIGH | Append-only audit log + amendment model; original record must survive amendments |
| Reviewer identity in every audit record | AS9102 Form 3 requires accountable reviewer identity; anonymous approvals are non-compliant | LOW | Name + timestamp on every approve/override action; sourced from authenticated session |
| Run history with reopen and download | QC teams regularly reference prior FAIRs during audits and corrective actions | MEDIUM | Filterable by part number and date; download must re-export the exact signed packet, not regenerate |
| Role-based access (admin vs engineer) | Shop config (column mapping, retention) must be protected; engineer-level changes to quality decisions must be traceable | LOW | Two roles sufficient for v1; admin manages accounts and shop settings, engineer reviews and signs |
| File upload with format validation | Users upload wrong file types regularly; a raster PDF or wrong Excel layout must be caught before compute is wasted | LOW | Validate file type at upload; detect raster vs vector PDF before job submission |
| Async pipeline with stage-by-stage progress | An 8-stage pipeline running computer vision on a large PDF will take 30–120 seconds; a spinner with no feedback erodes trust | MEDIUM | Show stage name + completion count per stage; distinguish queued / running / failed states |
| In-app alert on run failure | Engineers need to know when a run fails without polling the history page | LOW | In-app notification when logged in; no email needed for v1 |
| Session persistence for partial review | A 50-item queue cannot be completed in one sitting; losing progress on browser close is unacceptable | MEDIUM | Persist approve/override state server-side as each decision is made; allow resume from any device |
| Setup wizard for first admin login | If the shop config (Excel column mapping, admin account) can be skipped, the system produces garbage output silently | MEDIUM | Wizard must be undismissable until complete; cover shop name, admin password, first engineer account, Form 3 column mapping |
| PDF + CSV export of audit packet | QC engineers deliver FAIR packages to customers; PDF is the expected format; CSV enables QMS import | HIGH | PDF must include cover page with part/revision/reviewer/timestamp, summary table, per-item cards with inline snippets; CSV for programmatic use |
| Partial FAI work order output | The primary cost-saving claim of the product — telling the shop which characteristics need re-inspection — must be exportable | MEDIUM | PDF + CSV; contains char number, requirement, drawing reference, priority flag |
| Atomic sign-off + packet generation | If packet generation fails after sign-off is recorded, the run is in an ambiguous state that cannot be resolved without manual intervention | MEDIUM | Roll back sign-off if PDF generation fails; surface clear error; run stays in reviewable state |

---

### Differentiators (Competitive Advantage)

Features that set this product apart. Not expected by default, but high value for the domain.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pipeline confidence score per item with reasons | Existing FAI tools show results, not reasoning; surfacing "matched via location score 0.91, text overlap 0.87" gives the engineer the right frame for disagreeing or agreeing | LOW | Already produced by the pipeline; surfacing in UI is a display decision, not new computation |
| Auto-classification with confidence-gated manual escalation | System auto-classifies high-confidence items; only uncertain items require active review; reduces queue from N to uncertain-N | MEDIUM | Threshold configurable by admin; transparent about what was auto-classified and why; distinguishes "auto-classified" from "reviewed by engineer" in audit record |
| Rev A vs Rev B annotation diff inline in review card | Engineers can see exactly which text tokens changed (requirement text A vs B) rather than free-reading two snippet images | MEDIUM | Text diff on requirement string; not a pixel diff; highlights added/removed/changed substrings |
| Confidence distribution warning before review | Low alignment confidence across a run is more useful to surface before review starts than per-item; prevents engineer from reviewing 50 items then discovering the homography was garbage | LOW | Pre-review summary screen showing confidence histogram; explicit "proceed" / "abort" decision |
| Amendment model with version trail | Most FAI tools either lock records completely or allow silent overwrites; an explicit amendment model with preserved original is audit-correct and user-visible | MEDIUM | Show version history on any finalized run; amendment creates new versioned packet, original is locked |
| Air-gapped / on-premises deployment | Net-Inspect and GroundControl require cloud access; ITAR-sensitive shops cannot use cloud-hosted tools; Docker on shop server removes the dependency | HIGH | All dependencies bundled in Docker image; no external API calls at runtime; documented pull-based update path |
| Shop-owned data with configurable retention | Cloud tools hold customer data; a shop-hosted tool means the shop controls what survives and when it is deleted | LOW | Admin sets retention period for unfinished/failed runs; finalized records retained indefinitely by default |
| Explicit partial FAI scope output | Existing FAI software completes FAIRs end-to-end; the work order that says "only re-inspect characteristics 4, 7, 23" does not exist as a distinct output in competitor tools | MEDIUM | Separate action from sign-off; customer-facing format with drawing reference and priority |
| Multi-page PDF page selector at upload | Most tools assume single-page drawings; aerospace drawings are commonly multi-page; page selection at upload time prevents a wrong-page run | LOW | Detect multi-page at upload; show page thumbnails with selection before job submission |
| Duplicate balloon / count token tracking | "4X Ø 0.5" is one balloon but four Form 3 rows; treating count changes as a classification signal catches a class of changes competitors miss | MEDIUM | Already in pipeline; surfaced in UI as count token diff on changed items |

---

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem helpful but undermine audit integrity or user trust.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Bulk approve / select-all approve | Speeds up review when engineer "knows" items are all fine | Destroys individual accountability; audit trail shows a single approval action for N items; removes the forced pause per item that catches errors; 21 CFR Part 11 analog tools explicitly prohibit this pattern | Surface items grouped by system classification; let engineer move quickly through a sorted queue; keyboard shortcuts for approve/next acceptable if each item is individually confirmed |
| Auto-sign-off when all items are approved | One fewer click; feels like a natural workflow completion | Conflates "review complete" with "sign-off authorized"; sign-off is a formal attestation requiring deliberate intent; unintended sign-off creates immutable records that require amendments to correct | Explicit sign-off button that is enabled only when queue is fully resolved; clear label "Sign and finalize — this creates an immutable audit record" |
| Editable requirement text in the review UI | Engineer sees typo or OCR error in requirement text and wants to fix it | Editing requirements in the review interface creates ambiguity about what was actually on the drawing vs what the system read; requirement text should reflect what the pipeline extracted from the actual document | Display raw extracted text alongside the Form 3 requirement; if the pipeline extracted wrong text, that is a pipeline issue to fix in a new run, not an in-review edit |
| Email notifications for every action | Looks like a "complete" feature; QMS tools often include this | Adds external dependency (SMTP server, email deliverability) disproportionate to v1 scale (O(10) engineers, O(100s) runs/year); in-app alerts are sufficient for same-team use | In-app notification panel; email deferred to v1.x when pilot validation confirms need |
| Concurrent multi-user review of same run | Looks like a collaboration feature; teams may want parallel review | State conflicts when two engineers approve the same item simultaneously require locking or merge logic; one reviewer per run keeps the audit record unambiguous about who reviewed what | Run assignment model; submitter defaults as reviewer; can explicitly reassign to another engineer |
| Real-time collaborative cursor / annotation on snippets | Borrowed from design review tools (Figma, Drawboard); looks premium | Adds WebSocket complexity for no audit value; aerospace QC review is an individual attestation, not a collaborative annotation session | Override note field captures the engineer's reasoning; comments belong in the audit record, not on the image |
| OAuth / SSO login | Looks enterprise-ready | SSO requires external identity provider which may not exist at a small machine shop; adds dependency, config complexity, and security surface area for marginal benefit at O(10) user scale | Simple email + password accounts; admin manages accounts; no external dependencies |
| Auto-update via internet pull | Looks like good DevOps hygiene | Air-gapped shops cannot reach the internet; auto-update breaks the no-external-dependency constraint; unexpected updates in a production quality environment are a compliance risk (what changed in the tool?) | Documented manual `docker compose pull` process; change log published per release; update is a shop IT decision, not automatic |
| GD&T semantic parser in v1 | Engineers want symbol/datum/modifier interpretation, not string matching | Semantic GD&T parsing is a multi-month engineering investment; string matching on tolerance values catches the most consequential changes (value changes); semantic failures surface as uncertain items routed to manual review | Opaque string matching in v1; flag uncertain GD&T items clearly; defer semantic parsing to v2 after pilot validates scope |

---

## Feature Dependencies

```
User accounts (admin + engineer)
    └──requires──> Role-based access control
                       └──requires──> Session management / auth

File upload with validation
    └──requires──> User accounts
    └──enables──> Async pipeline execution

Async pipeline execution
    └──requires──> File upload
    └──enables──> Review queue (pipeline output is queue input)
    └──requires──> Stage-by-stage progress display

Review queue
    └──requires──> Async pipeline execution (delta_packet.json + snippets)
    └──requires──> Session persistence
    └──requires──> Mandatory override notes
    └──enables──> Sign-off

Sign-off
    └──requires──> Review queue fully resolved (hard gate)
    └──requires──> Atomic packet generation
    └──enables──> Run history / download
    └──enables──> Amendment model

Audit packet (PDF + CSV)
    └──requires──> Sign-off trigger
    └──requires──> Snippet images from pipeline run

Partial FAI work order
    └──requires──> Finalized run (signed)
    └──independent from──> Audit packet (separate action)

Amendment model
    └──requires──> Immutable original signed packet
    └──requires──> Finalized run

Run history
    └──requires──> Completed runs (any state)
    └──requires──> User accounts (filter by user / assignment)

Setup wizard
    └──requires──> Admin account creation (bootstraps system)
    └──enables──> Excel column mapping
    └──enables──> Engineer account creation

Air-gapped deployment
    └──conflicts──> OAuth / SSO
    └──conflicts──> Auto-update
    └──conflicts──> Email notifications (SMTP dependency)
```

### Dependency Notes

- **Review queue requires pipeline output:** The review queue cannot be populated until the 8-stage pipeline completes and writes `delta_packet.json` + snippet images. Stage-by-stage progress is required to make the wait tolerable.
- **Sign-off requires atomic packet generation:** Packet generation failure must roll back the sign-off state; these two operations must execute in a single transaction boundary.
- **Amendment model requires immutable originals:** Amendment is meaningless without a preserved, locked original. The immutability constraint must be designed into the data model before sign-off is built, not added after.
- **Setup wizard enables column mapping:** Without Excel column mapping, the pipeline will fail to parse Form 3 correctly on every run. The wizard is not optional UX polish — it's a prerequisite for any run succeeding.
- **Air-gapped deployment conflicts with external dependencies:** OAuth/SSO, email notifications, and auto-update all introduce outbound network calls. These are incompatible with the air-gapped shop network constraint and must be explicitly excluded.

---

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed for a QC team to use this in their actual workflow and produce a legally recognizable audit artifact.

- [ ] User accounts (admin + engineer roles) with session management — without identity, audit records have no value
- [ ] Setup wizard (shop config, Form 3 column mapping, engineer accounts) — without this, no run will parse correctly
- [ ] File upload with validation (raster PDF detection, multi-page selector, Excel format check) — garbage in = garbage out
- [ ] Async pipeline execution with stage-by-stage progress display — computer vision pipeline takes time; progress required for trust
- [ ] Review queue: per-item card with Rev A / Rev B snippets, system classification, confidence score, requirement text, Approve / Override controls, required override note — this is the core UX
- [ ] Session persistence for partial review — engineers cannot complete a 50-item queue in one sitting
- [ ] Hard gate: all items terminal before sign-off available — partial sign-off creates invalid audit records
- [ ] Atomic sign-off + PDF/CSV audit packet generation (with rollback on failure) — the product output; what gets delivered to the customer
- [ ] Immutable signed packet; amendment model for corrections post-sign-off — audit records must survive
- [ ] Partial FAI work order (PDF + CSV) generated on demand after finalization — the primary cost-saving output
- [ ] Run history with filter, reopen, and re-download — teams reference past FAIRs during audits
- [ ] In-app alert on run failure — engineers need failure notification without polling
- [ ] Docker deployment via `docker-compose.yml` — the only viable delivery mechanism for air-gapped shops

### Add After Validation (v1.x)

Features to add once the pilot validates the core workflow.

- [ ] Confidence distribution warning screen before review starts — useful but not blocking for v1; engineers can read per-item confidence scores
- [ ] Rev A vs Rev B requirement text diff inline in review card — reduces review time but engineers can read two text strings without highlighting
- [ ] Run assignment to non-submitter engineer — submitter-as-reviewer is sufficient for pilot; assignment needed when shop has dedicated QC reviewers vs job submitters
- [ ] Admin-configurable confidence threshold for auto-classification — 0.9 hardcoded is reasonable for pilot; configurability deferred until pilot shows the threshold needs tuning

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] GD&T semantic parsing (feature control frame symbol/datum/modifier interpretation) — significant engineering investment; string matching is sufficient for pilot
- [ ] Cross-revision timeline view (Rev A→B, B→C history) — analytics feature; individual run comparisons sufficient for v1
- [ ] QMS integration (ETQ, IQS, direct API connectors) — CSV export is the v1 integration surface; direct connectors require per-customer work
- [ ] Email notifications — in-app alerts sufficient for small shop scale; add when shop grows or pilot asks for it
- [ ] OAuth / SSO — incompatible with air-gapped constraint; only relevant if shops move to cloud-hosted deployment

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Per-item review card with snippets | HIGH | MEDIUM | P1 |
| Mandatory override note | HIGH | LOW | P1 |
| All-items gate before sign-off | HIGH | LOW | P1 |
| Immutable signed packet | HIGH | HIGH | P1 |
| Reviewer identity in audit record | HIGH | LOW | P1 |
| Atomic sign-off + packet generation | HIGH | MEDIUM | P1 |
| Async pipeline with stage progress | HIGH | MEDIUM | P1 |
| File upload with validation | HIGH | LOW | P1 |
| User accounts + roles | HIGH | MEDIUM | P1 |
| Setup wizard | HIGH | MEDIUM | P1 |
| Session persistence for partial review | HIGH | MEDIUM | P1 |
| PDF + CSV audit packet | HIGH | HIGH | P1 |
| Partial FAI work order | HIGH | MEDIUM | P1 |
| Run history + reopen + download | MEDIUM | MEDIUM | P1 |
| In-app run failure alert | MEDIUM | LOW | P1 |
| Docker deployment | HIGH | LOW | P1 |
| Confidence score + reasons per item | HIGH | LOW | P2 |
| Confidence distribution pre-review warning | MEDIUM | LOW | P2 |
| Amendment model with version trail | HIGH | MEDIUM | P2 |
| Rev A vs Rev B text diff inline | MEDIUM | MEDIUM | P2 |
| Multi-page PDF page selector | MEDIUM | LOW | P2 |
| Run assignment to other engineer | MEDIUM | LOW | P2 |
| Admin-configurable confidence threshold | LOW | LOW | P2 |
| GD&T semantic parsing | HIGH | HIGH | P3 |
| Cross-revision timeline view | LOW | HIGH | P3 |
| QMS direct API integration | MEDIUM | HIGH | P3 |
| Email notifications | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

| Feature | Net-Inspect | GroundControl | InspectionXpert | Delta Preservation |
|---------|-------------|---------------|-----------------|-------------------|
| Electronic sign-off on Form 3 | Yes | Yes | Yes | Yes — with atomic packet generation |
| Per-characteristic color-coding | Yes (out-of-tolerance) | Yes | Yes | Yes — plus confidence score and reasons |
| Auto-balloon extraction | Partial (requires 2D/3D source) | Yes (AI-assisted) | Yes | Yes — existing pipeline |
| Audit trail | Basic | Basic | Basic | Append-only, tamper-evident |
| Amendment / version history | Unknown | Unknown | Unknown | Explicit version trail |
| Air-gapped / on-premises | No (cloud) | No (AWS GovCloud) | No (cloud) | Yes — Docker on shop server |
| Partial FAI work order output | No | No (full FAI focus) | No | Yes — explicit scope output |
| Visual evidence snippets per item | No | No | No | Yes — Rev A + Rev B PNG crops |
| Session persistence for partial review | Basic | Basic | Basic | Server-side per-decision persistence |
| Setup wizard for first-time config | Unknown | Unknown | Unknown | Yes — undismissable until complete |
| Required override justification | Unknown | Unknown | Unknown | Yes — mandatory field, no silent overrides |

---

## What QC/Audit-Focused Tools Get Right That General-Purpose Apps Miss

Based on research across FAI software (GroundControl, Net-Inspect, InspectionXpert), pharma QMS tools (SimplerQMS, QT9), and audit trail standards (21 CFR Part 11, AS9102C):

**1. Identity is permanent, not optional.** Every action in an audit-worthy record must carry who did it and when. General-purpose apps treat identity as a filter; QC tools treat it as the primary key of every record.

**2. Reversibility is controlled, not free.** General-purpose apps allow edit/delete liberally. QC tools allow amendments but never overwrites. The original record must survive any correction.

**3. Required fields enforce compliance, not just UX.** In general apps, required fields are UX guardrails. In QC tools, an empty override justification is a compliance failure, not a missing input. The system must treat them differently — blocking, not warning.

**4. Progress must be conservative.** General-purpose apps show optimistic progress ("Almost done!"). QC tools must show conservative, factual progress ("3 of 50 items confirmed; 47 pending"). Overconfident UI leads engineers to sign off prematurely.

**5. Slow actions need explanation.** Computer vision pipelines take time. Without stage-by-stage progress, engineers assume the system crashed and submit again. Specific microcopy ("Running cross-revision alignment — step 5 of 8") prevents double-submission.

**6. Export format is not a bonus feature.** In general apps, PDF export is nice to have. In aerospace QC, the PDF is the deliverable — what goes to the customer and what survives an audit. It must carry formal structure (cover page, reviewer identity, immutable timestamp), not just a print of the screen.

---

## Sources

- [Net-Inspect FAI Software](https://www.net-inspect.com/solutions/first-article-inspection-software/) — feature set, industry market position
- [GroundControl AS9102 Software](https://www.gndctl.com/) — AI-supporting-humans workflow model, ITAR/compliance positioning
- [InspectionXpert AS9102](https://www.inspectionxpert.com/fai/as9102) — competitor feature comparison
- [Ideagen AS9102](https://www.ideagen.com/standards/as9102) — audit trail documentation patterns
- [21 CFR Part 11 compliance requirements](https://www.cognidox.com/blog/what-is-fda-21-cfr-part-11) — electronic record / e-signature requirements informing audit trail design
- [Immutable Audit Log patterns](https://www.hubifi.com/blog/immutable-audit-log-basics) — append-only storage, hash chain verification
- [UI patterns for async workflows](https://blog.logrocket.com/ui-patterns-for-async-workflows-background-jobs-and-data-pipelines/) — stage-by-stage progress display, partial failure handling
- [Electronic signature audit trail](https://www.gonitro.com/resources/esignature-audit-trials) — immutability requirements, capture what/who/when/why
- [AS9102 partial FAI standards](https://as9100store.com/aerospace-standards-explained/what-is-as9102-first-article-inspection/) — characteristic accountability, partial FAI re-accomplishment scope
- [Quality audit best practice](https://www.compliancequest.com/blog/quality-audit-best-practice/) — documented justification requirements for corrective actions

---
*Feature research for: Document review and sign-off web application (aerospace QC, AS9102 FAIR compliance)*
*Researched: 2026-03-01*
