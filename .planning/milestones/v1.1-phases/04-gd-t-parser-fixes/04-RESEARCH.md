# Phase 4: GD&T Parser Fixes - Research

**Researched:** 2026-04-14
**Domain:** GD&T (Geometric Dimensioning & Tolerancing) feature-control-frame parsing
**Confidence:** HIGH (all findings verified against current source files)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**GDT-01: Compact Token Splitting**
- **D-01:** Extend `_parse_gdt_frame` in `delta_preservation/reconcile/normalize.py` to regex-split the `remainder` string (everything after the leading control symbol character) into: optional diameter prefix (`∅`), numeric tolerance value, and trailing datum-ref characters — rather than relying on whitespace-tokenized adjacent tokens.
- **D-02:** The existing `_GDT_TOLERANCE_RE` (line 353) must be relaxed or a new splitting regex introduced to handle the datum-ref suffix (`ABC` in `⌖∅0.35ABC`). The datum-ref pattern (`_GDT_DATUM_RE`, line 364) must apply to individual extracted letters from the suffix, not to the concatenated string.

**GDT-02: Word-Name Normalization**
- **D-03:** Add a new word-to-symbol mapping constant in `normalize.py` (covering at minimum: "circularity", "runout", "total runout", "position", "flatness", "perpendicularity") and apply normalization *before* `_parse_gdt_frame` is invoked — so the existing `_GDT_CONTROL_MAP` lookup at line 710 succeeds on the substituted symbol. Word normalization must not be placed in `xlsx.py` or any I/O layer.
- **D-04:** Word matching should be case-insensitive and handle leading/trailing whitespace to be robust against Form 3 text variation.

**GDT-03: Composite Frame Capture**
- **D-05:** Extend `_parse_gdt_frame` to detect and split on the forward-slash separator (`/`) *before* its main token loop, producing two separate parse passes (one per compartment). Both compartments must be returned and captured — not silently discarded.
- **D-06:** `GdtSemanticPayload` in `delta_preservation/types.py` gains a `compartments` field (list of per-compartment data) to hold multi-compartment FCFs. `_compare_gdt` in `delta_preservation/reconcile/semantic_compare.py` is updated to compare primary compartments and treat secondary compartment presence as a potential differentiator.

**Entry Point and Test Strategy**
- **D-07:** All three fixes are implemented exclusively in `delta_preservation/reconcile/normalize.py` (and the `GdtSemanticPayload` type if needed). The call chain `_parse_gdt_frame` → `_extract_semantic_payload` → `extract_semantic_callout` remains the single bounded parser entry point.
- **D-08:** Regression tests are written to the existing `extract_semantic_callout(pdf_spans=[...], form3_requirement=...)` interface in `tests/test_semantic_extraction.py`. Each fix cluster gets at least one parametrized case that fails on unfixed code and passes after the fix.

### Claude's Discretion
- Whether to introduce a new private helper (e.g., `_split_compact_gdt_token`) or fold the compact-splitting logic inline into `_parse_gdt_frame` — Claude decides based on readability and test isolation.
- Whether `compartments` in `GdtSemanticPayload` holds full `ParsedGdtFrame`-like dicts or only the essential fields (tolerance, datum_refs) — Claude decides based on what `_compare_gdt` actually needs to compare them.

### Deferred Ideas (OUT OF SCOPE)
None — analysis stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GDT-01 | Compact GD&T tokens (`⌖∅0.35ABC`, `⌓0.5A`, `⏥0.2`) without spaces are correctly split into control symbol + diameter prefix + tolerance value + datum refs, eliminating `gdt_malformed_frame` fallback. | Single-pass regex strategy documented in "Fix Cluster 1"; `_GDT_TOLERANCE_RE` relaxation pattern documented below. |
| GDT-02 | Spelled-out GD&T control names ("circularity", "runout", "total runout", "position", "flatness", "perpendicularity") are normalized to Unicode symbols before semantic extraction. | Word-to-symbol table with longest-first matching algorithm documented in "Fix Cluster 2"; ASME Y14.5 canonical names cited. |
| GDT-03 | Composite multi-compartment FCFs (`⌓ .05 D B C / ⌓ .01 D`) are captured in full, not partially lost. | Compartment-splitting algorithm and `GdtSemanticPayload.compartments` schema change documented in "Fix Cluster 3"; `_compare_gdt` update strategy described. |
</phase_requirements>

## Summary

The GD&T parser in `delta_preservation/reconcile/normalize.py` (`_parse_gdt_frame`, lines 699–779) is a whitespace-tokenizing, single-compartment FCF parser. It works for the Rev-A/Rev-B inputs where a drawing span is already split into separate PDF spans (`⌖`, `⌀0.10`, `M`, `A`, `B`, `C`), but it fails on three real-corpus patterns:

1. **Compact concatenated tokens** — PDF text extractors sometimes emit the whole FCF as one whitespace-free token (`⌖∅0.35ABC`). The parser enters the `len(token) >= 2` branch at line 715, stashes the tail into `remainder`, and then hopes a neighboring token will complete the tolerance. With no neighbor, `_is_gdt_tolerance_token("∅0.35ABC")` returns False and the parser bails out as `gdt_malformed_frame`.
2. **Word-name entries** — Form 3 XLSX cells frequently contain words like "CIRCULARITY .05 A" instead of the Unicode symbol. `_GDT_CONTROL_MAP` only keys on symbol characters, so the parser never recognises a control type and returns `None` from the symbol-detect loop (normalize.py:721), falling through to the weld/surface/fit parsers.
3. **Composite FCFs** — Two-compartment frames separated by `/` are explicitly rejected at line 769 ("unsupported segment: /"). ASME Y14.5 composite FCFs carry a refinement tolerance in the second compartment that references a subset of the upper-compartment datums; silently dropping it is a real data loss.

All three fixes are confined to `normalize.py`, a one-field schema addition to `GdtSemanticPayload` in `types.py`, and a corresponding branch in `_compare_gdt` in `semantic_compare.py`. The existing `extract_semantic_callout(pdf_spans=..., form3_requirement=...)` test interface is sufficient to drive parametrized regression cases for every fix cluster.

**Primary recommendation:** Implement the three fixes as a layered pipeline inside `_parse_gdt_frame`:
(1) a pre-tokenization `/`-split producing N compartment strings;
(2) a word-name substitution pass on each compartment string that replaces recognised words with the corresponding Unicode symbol;
(3) a compact-token splitter that, when a single-token frame is detected, uses a one-pass regex `^([⌀∅]?)((?:\d+(?:\.\d+)?|\.\d+))([A-Z\-]*)$` on the remainder to yield tolerance + datum suffix characters. The existing whitespace-tokenised path still handles the already-passing cases unchanged.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Compact token splitting (GDT-01) | `reconcile/normalize.py` (`_parse_gdt_frame`) | — | Single bounded entry point per D-07; no I/O or web-tier involvement. |
| Word-name normalization (GDT-02) | `reconcile/normalize.py` (new pre-parse helper) | — | Must run before `_parse_gdt_frame`'s `_GDT_CONTROL_MAP` lookup; D-03 forbids placement in `xlsx.py`. |
| Composite compartment capture (GDT-03) | `reconcile/normalize.py` (`_parse_gdt_frame`) | `types.py` (`GdtSemanticPayload.compartments`), `reconcile/semantic_compare.py` (`_compare_gdt`) | Parsing owns the split; type schema owns the shape; comparator owns equality semantics. |
| Downstream semantic comparison | `reconcile/semantic_compare.py` (`_compare_gdt`) | — | Must tolerate the new `compartments` field; cannot regress existing single-compartment equality. |
| No change required | `classify.py`, `xlsx.py`, web tier, review workflow, ground truth evaluator | — | These modules consume `SemanticCallout` as opaque payloads; they do not re-parse GD&T strings. |

## Standard Stack

### Core (already present — no new installs required)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `re` (stdlib) | Py ≥3.10 | Regex-based token splitting | The rest of `normalize.py` uses `re.compile` + `fullmatch` patterns; consistency trumps introducing a parser generator. [VERIFIED: normalize.py:1] |
| `pydantic` | `>=2.5,<3.0` | Schema for `GdtSemanticPayload.compartments` field | Pydantic already models every semantic payload with `Field(...)`; new list field follows the existing `datum_refs: List[str] = Field(default_factory=list, ...)` pattern. [VERIFIED: types.py:80-88, pyproject.toml] |
| `pytest` | `>=8` | Parametrized regression tests per fix cluster | Existing `tests/test_semantic_extraction.py` already drives `extract_semantic_callout`. [VERIFIED: pyproject.toml:tool.pytest.ini_options] |

### Supporting (stdlib + existing project types)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `dataclasses` | stdlib | Extending `ParsedGdtFrame` (optional) with a `compartments` list if the internal parser carries the split before it's wrapped into the Pydantic payload. | Only if the planner decides to carry compartments through the dataclass rather than re-splitting in `_extract_semantic_payload`. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single-pass regex for compact token | Multi-step: strip `[⌀∅]` prefix → `_GDT_TOLERANCE_RE.match` → slice remainder as datum chars | Verbose and more branchy; but easier to unit-test each step independently. Claude's discretion under D-01. |
| Python `str.lower()` + dict lookup for word names | `regex` package with word-boundary support | No benefit — stdlib `re` with `(?i)` flag or pre-lowering the compartment string is sufficient. |
| Composite compartments held as `List[GdtSemanticPayload]` | Subset structure: `List[dict]` with only `tolerance_text`, `datum_refs`, `modifiers` | Full nested payload is more uniform with the single-compartment shape but risks recursive comparison loops in `_compare_gdt`. Simpler subset dict matches how `_compare_gdt` actually compares fields today and is the recommended path under D-06. |

**Installation:** None required. Every dependency is already in `pyproject.toml`. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram (GD&T parse path, end-to-end)

```
PDF text spans ─┐                     Form 3 XLSX cell ─┐
                │                                        │
                ▼                                        ▼
   join_text_spans (io/pdf.py)              xlsx loader (not in scope)
                │                                        │
                └───────────┬────────────────────────────┘
                            ▼
            extract_semantic_callout(pdf_spans, form3_requirement)
                            │    [picks authority: pdf > form3 > none]
                            ▼
                _normalize_semantic_text()  ── " ".join(text.split())
                            │
                            ▼
                _extract_semantic_payload(normalized_text)
                            │
                            ▼
                ┌─── _parse_gdt_frame(normalized_text) ───┐
                │                                         │
                │   [NEW] ① split on '/' into           │
                │         compartments: List[str]        │
                │                                         │
                │   for each compartment:                 │
                │     [NEW] ② substitute spelled-out    │
                │            control word → symbol       │
                │                                         │
                │     tokens = compartment.split()       │
                │     filter out '|' separators          │
                │                                         │
                │     for token:                          │
                │       if _GDT_CONTROL_MAP has token:   │
                │         control found (whitespace path)│
                │       elif token[0] is control symbol: │
                │         [NEW] ③ regex-split the       │
                │              remainder into            │
                │              (∅?)(num)(datum chars)    │
                │                                         │
                │   collect tolerance + datums + mods    │
                │   return ParsedGdtFrame OR error str   │
                └─────────────────────┬───────────────────┘
                                      │
                                      ▼
                       GdtSemanticPayload(
                         frame_text, control_type,
                         tolerance_text, datum_refs,
                         modifiers,
                         [NEW] compartments=[...]  ← list of
                                                     per-compartment
                                                     sub-payloads
                       )
                                      │
                                      ▼
               _compare_gdt(left, right)
                 ├── compare primary (control_type, tolerance,
                 │                    datum_refs, modifiers)
                 └── [NEW] compare compartments list:
                     - same length?
                     - pairwise equal tolerance/datums?
                                      │
                                      ▼
                   SemanticCompareResult(comparable, equal, …)
```

Decision points added in this phase are marked `[NEW]`. Everything else is the current unchanged flow.

### Component Responsibilities

| File | Current Responsibility | Change in Phase 4 |
|------|------------------------|--------------------|
| `delta_preservation/reconcile/normalize.py` | Owns `_parse_gdt_frame`, `_GDT_CONTROL_MAP`, `_GDT_TOLERANCE_RE`, `_GDT_DATUM_RE`, `_is_gdt_tolerance_token`, `_extract_semantic_payload`, `extract_semantic_callout` | Add word-to-symbol map; add compact-token splitter helper; add composite `/` split; wrap loop into per-compartment sub-parses; populate `compartments` on result. |
| `delta_preservation/types.py` | Defines Pydantic `GdtSemanticPayload` | Add `compartments: List[GdtCompartment]` (or `List[Dict[str, ...]]`) field with `default_factory=list` for backward compatibility. [VERIFIED: types.py:80-88 shows current shape.] |
| `delta_preservation/reconcile/semantic_compare.py` | `_compare_gdt` compares control/tolerance/datums/modifiers by direct equality | Extend to also compare `compartments`; treat length mismatch as "changed"; pairwise compare compartment fields. [VERIFIED: semantic_compare.py:112-156] |
| `tests/test_semantic_extraction.py` | Holds 3 existing GD&T cases (parsed / profile-without-datums / malformed) | Add at least one parametrized regression case per fix cluster (3 minimum, per D-08). |

### Pattern 1: Single-pass Compact GD&T Token Splitter (GDT-01)

**What:** A one-regex split of a single concatenated token into `(symbol)(diameter_prefix)(numeric)(datum_suffix)`.

**When to use:** Token starts with a `_GDT_CONTROL_MAP` symbol AND has additional characters glued to it AND the remainder has no adjacent token to complete the tolerance.

**Example (algorithm sketch):**
```python
# Proposed helper — placement in normalize.py alongside _is_gdt_tolerance_token
_GDT_COMPACT_REMAINDER_RE = re.compile(
    r"^([⌀∅])?"               # optional diameter prefix
    r"((?:\d+(?:\.\d+)?|\.\d+))"  # numeric tolerance value
    r"([A-Z](?:-[A-Z])?(?:[A-Z](?:-[A-Z])?)*)?$"  # trailing datum chars
)

def _split_compact_gdt_remainder(remainder: str) -> Optional[Tuple[str, List[str]]]:
    """Split '∅0.35ABC' → ('∅0.35', ['A','B','C']).
    Returns None if the remainder does not match a compact form."""
    m = _GDT_COMPACT_REMAINDER_RE.match(remainder)
    if not m:
        return None
    prefix, number, datum_blob = m.group(1) or "", m.group(2), m.group(3) or ""
    tolerance = f"{prefix}{number}"
    # Split the datum blob by single characters (respecting 'A-B' compound refs
    # like the existing _GDT_DATUM_RE shape).
    datum_refs = re.findall(r"[A-Z](?:-[A-Z])?", datum_blob)
    return tolerance, datum_refs
```

Reference: `_GDT_DATUM_RE = re.compile(r"^[A-Z](?:-[A-Z])?$")` at normalize.py:364 already defines what a valid datum ref looks like, so the datum-splitting regex above mirrors that grammar. [VERIFIED: normalize.py:364]

### Pattern 2: Word-to-Symbol Normalizer (GDT-02)

**What:** Case-insensitive substitution of ASME Y14.5 control-type words with their Unicode geometric characteristic symbols, applied to the normalized text before `_parse_gdt_frame` tokenizes it.

**When to use:** Any time `_parse_gdt_frame` is about to tokenize a Form 3 fallback string (or a PDF span that happens to render the word form).

**Longest-first matching is required:** "total runout" must be matched before "runout", otherwise the single-word key greedily wins and leaves " total" as a stray unknown token.

**Example (algorithm sketch):**
```python
# Placement: alongside _GDT_CONTROL_MAP in normalize.py
_GDT_WORD_CONTROL_MAP = {
    # Multi-word entries MUST be listed before single-word ones in the
    # sorted iteration (sort key = -len(phrase)).
    "total runout": "⟃",       # ASME Y14.5 total-runout symbol (see note)
    "circular runout": "⌿",
    "runout": "⌿",
    "circularity": "○",        # Unicode geometric characteristic
    "cylindricity": "⌭",
    "position": "⌖",
    "flatness": "⏥",
    "straightness": "—",
    "perpendicularity": "⟂",
    "parallelism": "∥",
    "angularity": "∠",
    "concentricity": "⊙",
    "symmetry": "⌯",
    "profile of a surface": "⌓",
    "profile of a line": "⌒",
}

_GDT_WORD_SUBSTITUTION_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(phrase) for phrase in sorted(
            _GDT_WORD_CONTROL_MAP, key=len, reverse=True
        )
    ) + r")\b",
    re.IGNORECASE,
)

def _normalize_gdt_word_controls(text: str) -> str:
    return _GDT_WORD_SUBSTITUTION_RE.sub(
        lambda m: _GDT_WORD_CONTROL_MAP[m.group(1).lower()],
        text,
    )
```

**Critical alignment point:** The symbols the word map substitutes **must** exist as keys in `_GDT_CONTROL_MAP`. [VERIFIED: normalize.py:331-340] The current map contains only 8 entries:

```python
_GDT_CONTROL_MAP = {
    "⌖": "position",
    "⌒": "profile_of_a_line",
    "⌓": "profile_of_a_surface",
    "⟂": "perpendicularity",
    "⏥": "flatness",
    "∠": "angularity",
    "∥": "parallelism",
    "⊙": "concentricity",
}
```

This covers GDT-02's six-word minimum *only partially*: `position`, `flatness`, `perpendicularity` map cleanly. But **`circularity`, `runout`, and `total runout` have no corresponding entry in `_GDT_CONTROL_MAP`** [VERIFIED: normalize.py:331-340]. So the plan must **also extend `_GDT_CONTROL_MAP`** with the symbols for circularity (`○` / U+25CB, or the engineering-drawing `⌀` — see Open Questions), circular runout (`⌿`), and total runout (`⟃`). Without that extension, the word-substitution fix produces a token that immediately fails the downstream `_GDT_CONTROL_MAP.get()` lookup at line 710. [ASSUMED for exact Unicode codepoints — see Open Questions; the ASME Y14.5 symbol names themselves are CITED.]

### Pattern 3: Composite Compartment Splitter (GDT-03)

**What:** Pre-split the normalized text on `/` (with surrounding whitespace) into compartment strings, run the existing parse logic on each, and return a primary result whose `compartments` field holds the full list.

**When to use:** Normalized text contains `' / '` (or `/` with at least one GD&T control symbol on each side — to avoid splitting on date strings, fractions, or H7/p6 fit designators that are handled by a different parser).

**Example (algorithm sketch):**
```python
def _split_composite_compartments(normalized_text: str) -> List[str]:
    """Split '⌓ .05 D B C / ⌓ .01 D' into ['⌓ .05 D B C', '⌓ .01 D'].

    Guards against splitting on '/' tokens that are NOT compartment separators:
    - fractions like '1/8'
    - fit designators like 'H7/p6' (owned by _parse_fit_callout, not GD&T)
    The heuristic: only split if at least two segments each begin with a
    _GDT_CONTROL_MAP symbol (or a recognised word-control after normalisation).
    """
    segments = [seg.strip() for seg in re.split(r"\s*/\s*", normalized_text) if seg.strip()]
    if len(segments) < 2:
        return [normalized_text]
    gdt_like = lambda s: bool(s) and (
        _GDT_CONTROL_MAP.get(s[0]) is not None
        or _GDT_CONTROL_MAP.get(s.split()[0] if s.split() else "") is not None
    )
    if sum(1 for s in segments if gdt_like(s)) >= 2:
        return segments
    return [normalized_text]
```

**Key insight:** The current line 769 returns `"recognized GD&T frame contains unsupported segment: /"` — this is the bail-out to replace. The `_is_gdt_tolerance_token("/")` check fails and `/` falls through to the post-tokens loop, which rejects it. [VERIFIED: normalize.py:769, cross-checked against CONTEXT.md specifics]

### Anti-Patterns to Avoid

- **Don't put word-name normalization in `xlsx.py` or any loader.** D-03 explicitly forbids it, and it would also mean every semantic source route (PDF, Form 3, future) would need its own copy. Place it in `normalize.py` where `_extract_semantic_payload` already owns the normalized-text invariant.
- **Don't split indiscriminately on `/`.** A fraction like `1/8` in a weld size, or a fit designator like `H7/p6`, must not trigger GD&T compartment splitting. The guard heuristic above (require ≥2 GD&T-like segments) prevents this.
- **Don't greedily match "runout" before "total runout".** Python's `re.sub` with a regex alternation built from longest-first-sorted keys handles this correctly; a naive `for word, symbol in map.items()` loop does not.
- **Don't break the existing whitespace-tokenised path.** The fix for compact tokens must only engage when a `_GDT_CONTROL_MAP` symbol is found glued to additional characters (i.e. `len(token) >= 2` branch at line 715). The already-passing test `test_extract_semantic_callout_gdt_parsed_position_pdf_authority_overrides_conflicting_form3_text` at tests/test_semantic_extraction.py:18 supplies `_span("⌖", ...)` as its own span and must continue to parse via the existing path.
- **Don't silently drop the second compartment.** `_compare_gdt` must surface a compartment-count mismatch as a changed/differentiating signal, not as `equal=True`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tokenizing mixed unicode+ASCII GD&T strings | Custom character-class scanner | `re` with explicit Unicode literals in the pattern | Python `re` handles Unicode BMP characters natively in Python 3; no `regex` package needed. [VERIFIED: normalize.py already uses `[⌀∅]` character classes.] |
| Pydantic field addition with back-compat | Custom `__init__` migration | `Field(default_factory=list)` on the new `compartments` attribute | Existing fields `datum_refs` and `modifiers` use this exact pattern. [VERIFIED: types.py:86-87] |
| Parametrized test runner for "must fail before fix, must pass after" | Hand-rolled assertion runner | `pytest.mark.parametrize` with `(input, expected_state, expected_detail)` tuples | The existing test file uses flat function-per-case, but other suites (`test_classify_bugfixes.py`, etc.) demonstrate the parametrize pattern. Adopt for the new regression block. |
| GD&T symbol Unicode codepoint lookup | Memorize codepoints | Cite the Miscellaneous Technical Unicode block (U+2300–U+23FF) and the Geometric Shapes block | Several of the symbols (`⌖ ⌀ ⌒ ⌓ ⏥ ⟂ ∠ ∥ ⊙`) live in Miscellaneous Technical / Mathematical Operators; `○` lives in Geometric Shapes. The code should hold them as string literals in `_GDT_CONTROL_MAP` and let Python treat them as opaque characters. |

**Key insight:** Every tool needed already lives in `delta_preservation`. The phase introduces zero new external dependencies.

## Runtime State Inventory

> Not applicable. This is a pure in-memory parser fix. No database schemas, no stored user data, no OS-registered state, no secrets, no build artifacts are affected.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — the parser operates on in-memory strings; no database schema change. `GdtSemanticPayload` is serialized to JSON inside `DeltaPacket`, but new optional fields with `default_factory=list` are backward compatible with older packets on disk. | None (new field is optional and additive). |
| Live service config | None. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | None. | None. |
| Build artifacts | None — pure Python, no compiled extensions. | None. |

**Verified by:** Reading `types.py` (Pydantic models with `default_factory` on every list field) and confirming no `alembic/versions/*.py` touches `GdtSemanticPayload` (it's a transient output type, not a DB-backed one).

## Common Pitfalls

### Pitfall 1: Line-number drift between CONTEXT.md and the current file
**What goes wrong:** CONTEXT.md cites line numbers (353, 364, 710, 769, etc.); the planner trusts them verbatim and writes code edits keyed to those exact offsets, only to find a later edit has drifted them.
**Why it happens:** CONTEXT.md was generated 2026-04-13 and a subsequent lint/import change could have shifted lines.
**How to avoid:** The plan must instruct each task to locate anchors by **symbol name** (`_GDT_CONTROL_MAP`, `_GDT_TOLERANCE_RE`, `_GDT_DATUM_RE`, the `return "recognized GD&T frame contains unsupported segment: /"` string) rather than by raw line number.
**Warning signs:** During research I confirmed lines 331, 353, 364, 699, 710, 715, 736, 769 all still match CONTEXT.md [VERIFIED: normalize.py re-read 2026-04-14]. The discrepancy between CONTEXT.md's stated line 702 and the actual `combined = f"{remainder}{next_token}"` line 735 is the only minor drift — CONTEXT.md references `combined = f"{remainder}{next_token}"` as "line 735" which is correct; line 702 is the `|` filter. All good.

### Pitfall 2: `_GDT_CONTROL_MAP` doesn't contain all the symbols that GDT-02's word map targets
**What goes wrong:** GDT-02's word list includes "circularity", "runout", "total runout". None of these have a symbol entry in the current `_GDT_CONTROL_MAP` (which covers position, profile-of-line, profile-of-surface, perpendicularity, flatness, angularity, parallelism, concentricity). Word substitution therefore maps a recognisable word to a symbol the downstream parser will reject.
**Why it happens:** The word list in D-03 and the symbol list in `_GDT_CONTROL_MAP` were authored independently.
**How to avoid:** The plan MUST add corresponding symbol entries to `_GDT_CONTROL_MAP` alongside the word map — at minimum: circularity (`○` U+25CB or industry variant), circular runout (`⌿` U+233F), total runout (`⟃` U+27C3, or the double-arrow form). Also extend `_GDT_ANCHOR_RE` at normalize.py:74 so `classify_requirement_type` still routes the new symbols to the `"gdt"` bucket.
**Warning signs:** If a regression test like `assert semantic.status.parser_family == "gdt"` passes but `assert semantic.gdt.control_type == "circularity"` fails, this is the symptom.
[VERIFIED by code inspection: normalize.py:74 defines `_GDT_ANCHOR_RE = re.compile(r"^[⌖⌒⟂⊙⌓⏥∥∠]")` — no circularity, no runout symbols.]

### Pitfall 3: Compact regex matches too aggressively and swallows legitimate whitespace cases
**What goes wrong:** The proposed `_GDT_COMPACT_REMAINDER_RE` is anchored with `$`, which is correct. But if the splitter helper is called before tokenization, it could apply to multi-token inputs and extract partial state.
**Why it happens:** Misplacement in the parse order.
**How to avoid:** Only invoke the compact splitter **inside the `len(token) >= 2 and _GDT_CONTROL_MAP.get(token[0])` branch** (normalize.py:715), i.e. after whitespace tokenization has already produced a single glued-together token. Don't apply it to the whole normalized string.
**Warning signs:** A previously-passing test like `test_extract_semantic_callout_gdt_parsed_position_pdf_authority_overrides_conflicting_form3_text` (tests/test_semantic_extraction.py:18) starts producing `control_type=None` or a changed `tolerance_text`.

### Pitfall 4: `/` in `H7/p6` fit designators or `1/8` weld sizes accidentally triggers composite split
**What goes wrong:** A generic `re.split(r"/", text)` on the normalized text would fire for weld/fit inputs that flow through the same `extract_semantic_callout` envelope, corrupting downstream parsers.
**Why it happens:** The compartment split is sequenced upstream of the weld/fit parsers.
**How to avoid:** Two guards. (a) The compartment splitter must require **at least two segments each starting with a recognized GD&T control symbol** after word substitution. (b) Place the composite-split logic **inside** `_parse_gdt_frame`, not in `_extract_semantic_payload`, so it only runs once the dispatcher has already routed to GD&T.
**Warning signs:** `test_extract_semantic_callout_weld_parsed_fragmented_pdf_spans_with_pdf_authority` (tests/test_semantic_extraction.py:202) which uses `"1/8 FILLET ..."` must continue to parse as `parser_family == "weld"`. Keep this test in the regression baseline.

### Pitfall 5: `_compare_gdt` compares frame_text for equality
**What goes wrong:** `_compare_gdt` at semantic_compare.py:124-128 builds equality on `(control_type, tolerance, datum_refs, modifiers)` — not `frame_text`. Good. But the test at test_semantic_extraction.py:47 asserts `semantic.gdt.frame_text == "⌖ | ⌀0.10 | M | A | B | C"` on exact string form. If the plan adds a `compartments` side-effect that changes how `frame_text` is rebuilt, that test regresses.
**How to avoid:** Build `frame_text` from the **primary compartment only**, preserving the current `" | ".join(frame_tokens)` shape. Hold any secondary compartment data in the new `compartments` field, not by concatenating into `frame_text`.
**Warning signs:** Existing GD&T frame_text test at test_semantic_extraction.py:47 fails.

## Code Examples

### Example 1: Minimal regression test for GDT-01 (compact token)
```python
# Source: tests/test_semantic_extraction.py — follow existing _span helper pattern
def test_extract_semantic_callout_gdt_parsed_compact_concatenated_token():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("⌖∅0.35ABC", span_id=0, x0=10.0)],
        form3_requirement=None,
    )
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "gdt"
    assert semantic.status.reason_code is None
    assert semantic.gdt is not None
    assert semantic.gdt.control_type == "position"
    assert semantic.gdt.tolerance_text in ("⌀0.35", "∅0.35")  # normalization may pick either
    assert semantic.gdt.datum_refs == ["A", "B", "C"]
```

### Example 2: Minimal regression test for GDT-02 (word name)
```python
def test_extract_semantic_callout_gdt_parsed_word_name_circularity():
    semantic = extract_semantic_callout(
        pdf_spans=[],
        form3_requirement="CIRCULARITY 0.05 A",
    )
    assert semantic.provenance.authority == "form3"
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "gdt"
    assert semantic.gdt is not None
    assert semantic.gdt.control_type == "circularity"
    assert semantic.gdt.tolerance_text == "0.05"
    assert semantic.gdt.datum_refs == ["A"]


def test_word_form_and_symbol_form_compare_equal():
    """Canonical cross-revision case: word-form in Form 3 vs symbol form in PDF."""
    symbol = extract_semantic_callout(
        pdf_spans=[_span("○", span_id=0, x0=10.0), _span("0.05", span_id=1, x0=20.0), _span("A", span_id=2, x0=30.0)],
        form3_requirement=None,
    )
    word = extract_semantic_callout(
        pdf_spans=[],
        form3_requirement="CIRCULARITY 0.05 A",
    )
    from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts
    result = compare_semantic_callouts(symbol, word)
    assert result.comparable is True
    assert result.equal is True
    assert result.family == "gdt"
```

### Example 3: Minimal regression test for GDT-03 (composite FCF)
```python
def test_extract_semantic_callout_gdt_parsed_composite_profile_frame():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("⌓ .05 D B C / ⌓ .01 D", span_id=0, x0=10.0)],
        form3_requirement=None,
    )
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "gdt"
    assert semantic.gdt is not None
    # Primary compartment
    assert semantic.gdt.control_type == "profile_of_a_surface"
    assert semantic.gdt.tolerance_text == "0.05"
    assert semantic.gdt.datum_refs == ["D", "B", "C"]
    # Secondary compartment captured, not dropped
    assert semantic.gdt.compartments is not None
    assert len(semantic.gdt.compartments) == 2
    assert semantic.gdt.compartments[1]["tolerance_text"] == "0.01"
    assert semantic.gdt.compartments[1]["datum_refs"] == ["D"]
```

### Example 4: Guard for the "did not regress malformed-frame detection" case
```python
def test_extract_semantic_callout_gdt_error_still_fires_on_symbol_with_no_tolerance():
    """Regression guard: after GDT-01, a true malformed input still reports gdt_malformed_frame."""
    semantic = extract_semantic_callout(
        pdf_spans=[_span("⌖", span_id=0, x0=10.0), _span("A", span_id=1, x0=20.0)],
        form3_requirement=None,
    )
    assert semantic.status.state == "error"
    assert semantic.status.reason_code == "gdt_malformed_frame"
```

This mirrors the existing test at test_semantic_extraction.py:83 — it must continue to pass unchanged.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Whitespace-only tokenization of GD&T FCFs | Hybrid: whitespace-first, fall through to single-token compact-regex split | This phase | GDT-01 complete |
| Symbol-only `_GDT_CONTROL_MAP` lookup | Pre-parse word-to-symbol normalization + extended symbol map | This phase | GDT-02 complete |
| Single-compartment FCF only; `/` rejected | Compartment split before tokenization, `compartments` field on `GdtSemanticPayload` | This phase | GDT-03 complete |

**Deprecated/outdated:** None — the fix is additive over the existing parser, preserves every passing test, and does not remove any code paths.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥8 (`[tool.pytest.ini_options]` in pyproject.toml) |
| Config file | `pyproject.toml` — `testpaths = ["tests"]`, `addopts = "-q"` [VERIFIED] |
| Quick run command | `pytest tests/test_semantic_extraction.py -x -q` |
| Full suite command | `pytest -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GDT-01 | `⌖∅0.35ABC` parses to control=position, tolerance=⌀0.35, datums=[A,B,C] | unit (parametrized) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_compact_concatenated_token -x` | ❌ Wave 0 |
| GDT-01 | `⌓0.5A` parses — compact single-datum form | unit (parametrized) | `pytest tests/test_semantic_extraction.py -k compact_single_datum -x` | ❌ Wave 0 |
| GDT-01 | `⏥0.2` parses — compact no-datum form | unit (parametrized) | `pytest tests/test_semantic_extraction.py -k compact_no_datum -x` | ❌ Wave 0 |
| GDT-01 | Existing whitespace-tokenised path still parses | unit (regression) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_position_pdf_authority_overrides_conflicting_form3_text -x` | ✅ exists at line 18 |
| GDT-01 | True malformed `⌖ A` still reports error | unit (regression) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_error_malformed_frame_keeps_pdf_authority -x` | ✅ exists at line 83 |
| GDT-02 | "CIRCULARITY 0.05 A" in Form 3 parses as GD&T | unit (parametrized) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_word_name_circularity -x` | ❌ Wave 0 |
| GDT-02 | "TOTAL RUNOUT 0.02 A-B" parses (multi-word, longest-first guard) | unit (parametrized) | `pytest tests/test_semantic_extraction.py -k word_name_total_runout -x` | ❌ Wave 0 |
| GDT-02 | "runout" / "RUNOUT" / "Runout" all parse (case-insensitive) | unit (parametrized) | `pytest tests/test_semantic_extraction.py -k word_name_case -x` | ❌ Wave 0 |
| GDT-02 | Word-form vs symbol-form compare equal via `compare_semantic_callouts` | integration | `pytest tests/test_semantic_extraction.py::test_word_form_and_symbol_form_compare_equal -x` | ❌ Wave 0 |
| GDT-03 | `⌓ .05 D B C / ⌓ .01 D` yields 2 compartments, both captured | unit (parametrized) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_composite_profile_frame -x` | ❌ Wave 0 |
| GDT-03 | `H7/p6` fit input still parses as `fit`, not GD&T composite | unit (regression guard) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_fit_parsed_pdf_authority_overrides_conflicting_form3_text -x` | ✅ exists at line 296 |
| GDT-03 | `1/8 FILLET` weld input still parses as `weld` | unit (regression guard) | `pytest tests/test_semantic_extraction.py::test_extract_semantic_callout_weld_parsed_fragmented_pdf_spans_with_pdf_authority -x` | ✅ exists at line 202 |
| GDT-03 | `_compare_gdt` flags compartment-count mismatch as changed | unit (new) | `pytest tests/test_semantic_comparison.py -k compartment_count -x` | ❌ Wave 0 |
| Cross-cutting | `semantic_compare.py` single-compartment path unchanged | unit (regression) | `pytest tests/test_semantic_comparison.py -x` | ✅ exists as full file |

### Sampling Rate
- **Per task commit:** `pytest tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py -x -q`
- **Per wave merge:** `pytest tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py -x -q`
- **Phase gate:** `pytest -q` — full suite green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] New parametrized block in `tests/test_semantic_extraction.py` covering GDT-01 compact forms (at least 3 cases: with-datums, single-datum, no-datum).
- [ ] New parametrized block in `tests/test_semantic_extraction.py` covering GDT-02 word forms (at least 4 cases: circularity, runout, total runout, perpendicularity + one case-variation case).
- [ ] New parametrized block in `tests/test_semantic_extraction.py` covering GDT-03 composite forms (at least 2 cases: same-symbol composite, mixed-symbol composite).
- [ ] New case in `tests/test_semantic_comparison.py` for `_compare_gdt` with compartment-count differences.
- [ ] No new framework install. pytest ≥8 already present.

### Reference Data (debug corpus)
The 9-part debug corpus lives in `assets/part{1..9}/` (revA.pdf, revB.pdf, FAIR.xlsx, ground_truth.json). The corpus contains the real compact tokens, word-form entries, and composite FCFs that motivate this phase — the planner does not need to open the binaries. Instead, the regression test uses **synthetic `TextSpan` inputs that reproduce the token shapes** documented in the requirements (`⌖∅0.35ABC`, `⌓0.5A`, `⏥0.2`, "circularity", "runout", "total runout", `⌓ .05 D B C / ⌓ .01 D`). Full-pipeline validation against the corpus itself happens in Phase 7 (VER-01).

### How to Detect Regression of the Fix
- **Primary signal:** count of `status.reason_code == "gdt_malformed_frame"` events in `extract_semantic_callout` output across the corpus. Before fix: N>0 for every compact/word/composite case. After fix: 0 for those cases.
- **Secondary signal:** pytest parametrized IDs. If any of the new Wave 0 tests are removed, xfailed, or parametrized out without equivalent replacement, the regression gate fails.
- **Tertiary signal:** `test_reconcile_semantic_integration.py` and `test_pipeline_semantic_packet.py` must not begin producing `gdt_malformed_frame` statuses on previously-passing packets.

## Risks and Edge Cases

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `_GDT_CONTROL_MAP` doesn't contain circularity/runout symbols, so word substitution produces a token the downstream lookup rejects | HIGH | Blocks GDT-02 | Plan MUST add symbol entries alongside the word map (Pitfall 2). |
| R2 | Compact regex is too permissive and matches tolerance-like tokens in non-GD&T contexts | LOW | Silent mis-parse | Only invoke from inside the `len(token) >= 2 and _GDT_CONTROL_MAP.get(token[0])` branch. |
| R3 | `/` splitter fires on fit (`H7/p6`) or weld (`1/8`) inputs | MEDIUM | Corrupts non-GD&T parsers | Guard: require ≥2 GD&T-control-leading segments, and place splitter inside `_parse_gdt_frame`, not `_extract_semantic_payload`. |
| R4 | `_compare_gdt` starts returning `comparable=False` on single-compartment payloads because the new `compartments` field introduces an equality diff | MEDIUM | Breaks cross-revision equality for existing passing cases | Compare `compartments` using `[]` as the neutral default; `left.compartments or None == right.compartments or None`. |
| R5 | MMC modifier inside compact token (e.g., `⌖∅0.35ⓂABC`) | LOW (not in required corpus forms) | Modifier dropped | Out of scope for GDT-01 per decision D-01 which explicitly lists prefix + numeric + datum suffix only. Deferred. |
| R6 | Negative tolerance (`⌖∅-0.35ABC`) or sign prefix | VERY LOW (not in GD&T semantics — tolerances are unsigned magnitudes in FCFs) | N/A | No action. |
| R7 | Multi-letter datum refs (`A1`, `A-B`) in compact suffix | MEDIUM | Split incorrectly | The proposed regex `[A-Z](?:-[A-Z])?(?:[A-Z](?:-[A-Z])?)*` handles `A-B` compound datums (matching `_GDT_DATUM_RE` at line 364), but does NOT handle `A1`-style numeric suffixes. If the corpus contains `A1`, relax the pattern to `[A-Z](?:-?[A-Z0-9])?` and update `_GDT_DATUM_RE` in parallel. [ASSUMED: A1-style datums are not in the 9-part corpus — confirm during plan review.] |
| R8 | Three-compartment composite FCFs (primary + two refinements) | LOW | Only first two captured | The split helper handles N compartments naturally (`re.split` returns all); the `compartments` list should hold all of them. Test with a 3-compartment case in Wave 0 only if the corpus contains one (CONTEXT.md says "composite" — ASME Y14.5 permits up to 3). |

## Open Questions

1. **Exact Unicode codepoints for circularity, circular runout, total runout symbols**
   - What we know: ASME Y14.5 defines the shapes; Unicode has a Miscellaneous Technical block (U+2300–U+23FF) containing most GD&T characteristic symbols.
   - What's unclear: Which codepoint the existing PDF extractor emits when a drawing uses a circularity symbol. Common candidates: `○` (U+25CB, Geometric Shapes), `⌀` (U+2300 — but that's already "diameter" in the project codebase), `⊙` (U+2299 — but that's concentricity in the current map). For runout: `⌿` (U+233F) for circular runout, `⟃` (U+27C3) for total runout. [ASSUMED for all three — training data, not verified against the 9-part corpus binary spans.]
   - Recommendation: During plan execution, grep the actual PDF text extraction output from one corpus part that uses each symbol, and adopt the codepoint the extractor actually emits. The plan should include a "confirm codepoints from corpus" sub-task before committing the symbol table.

2. **Should `compartments` hold full Pydantic `GdtSemanticPayload` instances or a lightweight dict?**
   - What we know: D-06 leaves this to Claude's discretion.
   - What's unclear: Whether `_compare_gdt` needs to recursively call itself on compartments, or whether it's sufficient to compare `(tolerance_text, datum_refs, modifiers)` tuples.
   - Recommendation: Use a lightweight nested `GdtCompartment` Pydantic submodel with fields `control_type`, `tolerance_text`, `datum_refs`, `modifiers` — same shape as the top-level primary fields, but without `frame_text` and without `compartments` (no recursion). This keeps `_compare_gdt` flat and preserves the existing single-compartment equality semantics.

3. **Does the composite splitter need to run on word-form inputs too?**
   - Example: `"CIRCULARITY .05 A / CIRCULARITY .01 A"` — is this a real corpus pattern?
   - What we know: GDT-02 says word normalization runs *before* `_parse_gdt_frame`. GDT-03's composite split also runs inside `_parse_gdt_frame`. If word normalization happens first, the text becomes `"○ .05 A / ○ .01 A"` and the composite splitter sees symbols on both sides — works correctly.
   - Recommendation: Sequence word normalization → composite split → per-compartment compact-token check. This is the only ordering where all three fixes compose cleanly.

4. **Does Phase 5 (classification logic) need `compartments` to be structured before it can implement CLS-01 / CLS-03?**
   - What we know: Phase 5 handles classification of the whole `DeltaItem`, which consumes `SemanticCallout` as an opaque carrier.
   - What's unclear: Whether any Phase 5 logic will branch on "has multiple compartments".
   - Recommendation: Phase 5 should treat multi-compartment GD&T as "single semantic equality or single semantic diff" — i.e. rely on `_compare_gdt`'s flat equal/not-equal output, not on probing the compartments list. Leave this question for Phase 5 context-gathering.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python interpreter | All parser code | ✓ | ≥3.10, <3.13 | — |
| pytest | Regression tests | ✓ | ≥8 | — |
| pydantic | Type schema change | ✓ | ≥2.5,<3.0 | — |
| `re` (stdlib) | All regex work | ✓ | stdlib | — |

All dependencies are declared in `pyproject.toml` and locked via `uv.lock`. No external tooling, no database, no network. [VERIFIED: pyproject.toml inspected.]

## Project Constraints (from repository context)

- **No per-part hacks.** Every fix must generalize across all 9 parts in the debug corpus. [from REQUIREMENTS.md milestone goal]
- **Ground truth files are canonical.** `assets/part{1..9}/ground_truth.json` must never be edited by the pipeline or by regression tests. [from REQUIREMENTS.md out-of-scope section]
- **Fixes confined to three files:** `normalize.py` + optional `types.py` + `semantic_compare.py`. No touching web tier, review workflow, or `xlsx.py` / `pdf.py` loaders. [from CONTEXT.md domain boundary]
- **Single parser entry point:** `_parse_gdt_frame` → `_extract_semantic_payload` → `extract_semantic_callout`. All fixes land inside this chain. [from D-07]
- **Regression tests use existing public interface:** `extract_semantic_callout(pdf_spans=[...], form3_requirement=...)`. No new test harness. [from D-08]

## Security Domain

> `security_enforcement` is not set in `.planning/config.json`. Per policy, treat as enabled. This phase's security footprint is minimal (pure in-process regex parser on text already in memory), but the applicable ASVS controls are listed for completeness.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a — parser is internal, not network-exposed |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | **yes** | The parser consumes arbitrary PDF span text and Form 3 requirement strings. Regex patterns must be anchored (`^…$`) to avoid accidental partial matches that leak state into the output. All proposed regexes in this research are anchored. |
| V6 Cryptography | no | n/a |
| V7 Error Handling | **yes** | Parser errors must produce structured `SemanticParserStatus(state="error", reason_code=..., detail=...)` payloads, not raise exceptions. Existing pattern is preserved. |
| V8 Data Protection | no | n/a — no PII, no secrets |
| V12 Files & Resources | no | n/a — no file I/O in the parser |

### Known Threat Patterns for Python regex parsing

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Regex catastrophic backtracking (ReDoS) on pathological inputs | Denial-of-service | All proposed patterns use bounded quantifiers or simple character classes; no nested unbounded quantifiers (`(.*)*`). Verified by inspection. |
| Unicode normalization mismatch (NFC vs NFD) | Tampering (false negatives in comparison) | The existing `_normalize_semantic_text()` only whitespace-normalizes. If a corpus PDF emits `⌓` as a combining sequence rather than the precomposed character, compartment split could fail. Recommendation: add an `unicodedata.normalize("NFC", text)` call inside `_normalize_semantic_text` as a belt-and-suspenders fix. [Deferred to Open Question if out of scope.] |
| Untrusted input injecting `/` splitter semantics | Tampering | The compartment-split guard (require ≥2 GD&T-leading segments) blocks this: an attacker cannot force a split unless they supply two valid GD&T control symbols, at which point the split semantics are correct by construction. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Unicode codepoints for circularity (`○` U+25CB), circular runout (`⌿` U+233F), total runout (`⟃` U+27C3) match what the PDF extractor emits from the debug corpus | Pattern 2, Pitfall 2, Open Question 1 | Word→symbol substitution produces a codepoint that doesn't match what actual PDF spans contain → word and symbol forms don't compare equal; GDT-02 acceptance criterion 2 fails. Plan must verify by sampling actual corpus. |
| A2 | The 9-part debug corpus does not contain `A1`-style numeric datum refs (only letter + optional `-letter` compound) | Risk R7 | If the corpus does contain them, the compact-token regex fails to split them correctly. Plan should grep ground_truth.json for `"A1"`/`"B1"` etc during wave 0. |
| A3 | ASME Y14.5 composite FCFs in this corpus have at most 2 compartments, not 3 | Risk R8 | Third compartment silently dropped. Compartments list implementation handles N ≥ 2 naturally, so impact is low — add a 3-compartment test only if corpus confirms one exists. |
| A4 | `compartments: List[GdtCompartment]` is an appropriate schema shape (vs. `List[Dict]` or a flat double-field approach) | Pattern 3, Open Question 2 | If `_compare_gdt` ends up needing richer comparison, a flat dict is more ergonomic than a nested model. Mitigated by the recommendation to start with a lightweight Pydantic submodel. |
| A5 | Placing `unicodedata.normalize("NFC", text)` in `_normalize_semantic_text` is safe and correct | Security Domain | NFC normalization could rewrite some inputs in ways that change downstream string-comparison tests. Recommendation: deferred — only add if Wave 0 surfaces a real NFC/NFD mismatch in the corpus. |

## Sources

### Primary (HIGH confidence — verified by direct code read this session)
- `delta_preservation/reconcile/normalize.py` — lines 1–120 (imports, dataclasses, anchor regex), lines 300–780 (full GD&T parse path including `_GDT_CONTROL_MAP`, `_GDT_TOLERANCE_RE`, `_GDT_DATUM_RE`, `_parse_gdt_frame`, `_extract_semantic_payload`, `extract_semantic_callout`, `_is_gdt_tolerance_token`).
- `delta_preservation/types.py` — full file, confirming `GdtSemanticPayload` Pydantic shape (lines 80–88) and the backward-compatible `default_factory=list` pattern on existing list fields.
- `delta_preservation/reconcile/semantic_compare.py` — full file, confirming `_compare_gdt` at lines 112–156 uses field-level equality on `(control_type, tolerance, datum_refs, modifiers)` and rebuilds a reason-fragments summary on mismatch.
- `tests/test_semantic_extraction.py` — full file, confirming existing GD&T tests at lines 18 / 57 / 83, the `_span` helper, the weld test that uses `1/8` (line 202), and the fit test that uses `H7/p6` (line 296 and 322).
- `.planning/phases/04-gd-t-parser-fixes/04-CONTEXT.md` — decisions D-01 through D-08, canonical refs, existing code insights, specific line-level notes.
- `.planning/REQUIREMENTS.md` — GDT-01/02/03 acceptance criteria, traceability table.
- `.planning/ROADMAP.md` — Phase 4 goal and success criteria (items 1–4).
- `.planning/STATE.md` — confirms Phase 4 is the current focus, not yet planned.
- `pyproject.toml` — confirmed dependency versions (pytest ≥8, pydantic 2.5-3.0, Python 3.10-3.13), pytest config (`testpaths = ["tests"]`, `addopts = "-q"`).
- `.planning/config.json` — confirmed `nyquist_validation: true` (Validation Architecture section required); `commit_docs: true` (research file to be committed).

### Secondary (MEDIUM confidence — training knowledge about ASME Y14.5 cross-verified with code shape)
- ASME Y14.5-2018 geometric characteristic symbol table (circularity, runout, total runout, position, flatness, perpendicularity word forms). The word list is stable and well-known; the specific Unicode codepoints are the uncertain part (see Assumption A1).
- Unicode Miscellaneous Technical block (U+2300–U+23FF) for engineering characteristic symbols; Geometric Shapes block (U+25A0–U+25FF) for circularity.

### Tertiary (LOW confidence — flagged for corpus validation)
- Exact mapping of corpus-extracted PDF glyphs to Unicode codepoints for circularity / runout / total runout. [See Open Question 1 and Assumption A1.]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified against `pyproject.toml` and `uv.lock`.
- Architecture: HIGH — all line numbers and symbol names verified by direct file read this session.
- Pitfalls: HIGH — every pitfall is grounded in a specific file:line citation and an existing test case.
- Unicode codepoints: MEDIUM — word names are HIGH (ASME), codepoint identity is ASSUMED and flagged.
- Validation architecture: HIGH — test infrastructure exists, parametrized pattern established, gaps enumerated.

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (30 days — parser code is stable; only the debug corpus fixtures could plausibly shift codepoint decisions, and those are locked by ground_truth.json)
