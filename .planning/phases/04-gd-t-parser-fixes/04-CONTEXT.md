# Phase 4: GD&T Parser Fixes - Context

**Gathered:** 2026-04-13 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Correct the GD&T parser in `normalize.py` so that compact concatenated frames (GDT-01), spelled-out word names (GDT-02), and composite multi-compartment FCFs (GDT-03) all produce structured parse results without triggering `gdt_malformed_frame` fallback. No changes to the web tier, review workflow, or ground truth evaluation. No per-part hacks — fixes must be general and exercise the shared parser path.

</domain>

<decisions>
## Implementation Decisions

### GDT-01: Compact Token Splitting
- **D-01:** Extend `_parse_gdt_frame` in `delta_preservation/reconcile/normalize.py` to regex-split the `remainder` string (everything after the leading control symbol character) into: optional diameter prefix (`∅`), numeric tolerance value, and trailing datum-ref characters — rather than relying on whitespace-tokenized adjacent tokens.
- **D-02:** The existing `_GDT_TOLERANCE_RE` (line 353) must be relaxed or a new splitting regex introduced to handle the datum-ref suffix (`ABC` in `⌖∅0.35ABC`). The datum-ref pattern (`_GDT_DATUM_RE`, line 364) must apply to individual extracted letters from the suffix, not to the concatenated string.

### GDT-02: Word-Name Normalization
- **D-03:** Add a new word-to-symbol mapping constant in `normalize.py` (covering at minimum: "circularity", "runout", "total runout", "position", "flatness", "perpendicularity") and apply normalization *before* `_parse_gdt_frame` is invoked — so the existing `_GDT_CONTROL_MAP` lookup at line 710 succeeds on the substituted symbol. Word normalization must not be placed in `xlsx.py` or any I/O layer.
- **D-04:** Word matching should be case-insensitive and handle leading/trailing whitespace to be robust against Form 3 text variation.

### GDT-03: Composite Frame Capture
- **D-05:** Extend `_parse_gdt_frame` to detect and split on the forward-slash separator (`/`) *before* its main token loop, producing two separate parse passes (one per compartment). Both compartments must be returned and captured — not silently discarded.
- **D-06:** `GdtSemanticPayload` in `delta_preservation/types.py` gains a `compartments` field (list of per-compartment data) to hold multi-compartment FCFs. `_compare_gdt` in `delta_preservation/reconcile/semantic_compare.py` is updated to compare primary compartments and treat secondary compartment presence as a potential differentiator.

### Entry Point and Test Strategy
- **D-07:** All three fixes are implemented exclusively in `delta_preservation/reconcile/normalize.py` (and the `GdtSemanticPayload` type if needed). The call chain `_parse_gdt_frame` → `_extract_semantic_payload` → `extract_semantic_callout` remains the single bounded parser entry point.
- **D-08:** Regression tests are written to the existing `extract_semantic_callout(pdf_spans=[...], form3_requirement=...)` interface in `tests/test_semantic_extraction.py`. Each fix cluster gets at least one parametrized case that fails on unfixed code and passes after the fix.

### Claude's Discretion
- Whether to introduce a new private helper (e.g., `_split_compact_gdt_token`) or fold the compact-splitting logic inline into `_parse_gdt_frame` — Claude decides based on readability and test isolation.
- Whether `compartments` in `GdtSemanticPayload` holds full `ParsedGdtFrame`-like dicts or only the essential fields (tolerance, datum_refs) — Claude decides based on what `_compare_gdt` actually needs to compare them.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Parser
- `delta_preservation/reconcile/normalize.py` — GD&T token parsing (`_parse_gdt_frame`, `_GDT_CONTROL_MAP`, `_GDT_TOLERANCE_RE`, `_GDT_DATUM_RE`, `_extract_semantic_payload`, `extract_semantic_callout`)

### Type Definitions
- `delta_preservation/types.py` — `GdtSemanticPayload`, `SemanticCallout`, `DeltaItem` — may need compartments field added

### Semantic Comparison
- `delta_preservation/reconcile/semantic_compare.py` — `_compare_gdt` (line ~112) — must handle multi-compartment payloads after GDT-03 fix

### Existing Tests
- `tests/test_semantic_extraction.py` — established test interface and existing passing cases (lines 18-101); new tests must not regress these

### Requirements
- `.planning/REQUIREMENTS.md` — GDT-01, GDT-02, GDT-03 acceptance criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_GDT_CONTROL_MAP` (normalize.py line 331-340): symbol → control type — will be reused; word-name table added alongside it
- `_GDT_TOLERANCE_RE` (line 353): must be relaxed or supplemented for compact token splitting
- `_GDT_DATUM_RE` (line 364): matches individual uppercase letter datum refs — valid for extracted suffix characters
- `extract_semantic_callout` public function: test entry point for all three fixes
- `tests/test_semantic_extraction.py`: parametrized test pattern with `pytest.mark.parametrize` — follow this pattern for new regression cases

### Established Patterns
- All GD&T parsing is concentrated in `normalize.py`; no other module duplicates this logic
- `_parse_gdt_frame` is the single function that produces either a structured result or `gdt_malformed_frame`
- Pipe (`|`) separator is already handled at line 702 by filtering — forward-slash will follow the same strip-then-re-parse approach, extended to produce multiple outputs
- `GdtSemanticPayload` is a Pydantic model with `Field()` validation — new `compartments` field follows the same pattern

### Integration Points
- `_extract_semantic_payload` (normalize.py line 519) calls `_parse_gdt_frame` and wraps result in `SemanticCallout` — this is where multi-compartment return value is consumed
- `semantic_compare.py:_compare_gdt` receives `SemanticCallout` payloads — must be updated if `GdtSemanticPayload` gains a new field
- `classify.py` calls `semantic_compare.py` — no direct GD&T parsing changes needed there

</code_context>

<specifics>
## Specific Ideas

- The analyzer confirmed that `⌖∅0.35ABC` as a single whitespace-free token has no `next_token`, so the existing `combined = f"{remainder}{next_token}"` path at line 735 is never triggered — the fix must handle the single-token case internally.
- The `/` slash in composite FCFs hits an explicit `return f"...unsupported segment: /"` at line 769 — this is the exact line to replace with compartment-splitting logic.
- Word-name normalization must happen before `_parse_gdt_frame` because by the time the parser runs, a word-form entry has no leading GD&T symbol character and fails the symbol lookup at line 710 immediately.

</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within phase scope.

</deferred>

---

*Phase: 04-gd-t-parser-fixes*
*Context gathered: 2026-04-13 (assumptions mode)*
