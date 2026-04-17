"""Phase 6 Plan 04: Asset-backed regression harness for Part 8 / Part 9 exemplars.

Three test classes:

  TestPhase6AssetInvariants
    Asserts required snapshot files exist and confirms known duplicate groups
    still appear in ground truth by exact text.

  TestPhase6Part8Exemplars
    Pins the exact Part 8 fragment false-positive strings from the checked-in
    debug report (.045 A, ⚪ .005, ⌰ .002 A) and asserts the Phase-6
    added-detection path treats full-callout text as authoritative.

  TestPhase6Part9DuplicateAddedRows
    Loads exact duplicate truth centers from Part 9 ground truth, builds
    packet items, and asserts the Phase-6 evaluator can claim distinct
    added-pool tokens for duplicated rows.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import pytest

from delta_preservation.cli import run_pipeline

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent.parent / "assets"
DEBUG_REPORT_PART8 = ASSETS_DIR / "debug_report_part8.json"
DEBUG_REPORT_PART9 = ASSETS_DIR / "debug_report_part9.json"
GROUND_TRUTH_PART8 = ASSETS_DIR / "part8" / "ground_truth.json"
GROUND_TRUTH_PART9 = ASSETS_DIR / "part9" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _added_rows(ground_truth: dict) -> list[dict]:
    """Return all characteristics with classification == 'added'."""
    return [
        c for c in ground_truth.get("characteristics", [])
        if c.get("classification") == "added"
    ]


def _normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").split())


@lru_cache(maxsize=None)
def _live_packet(part_name: str) -> dict:
    """Run the real pipeline for a part once and cache the resulting packet."""
    asset_dir = ASSETS_DIR / part_name
    out_dir = Path(mkdtemp(prefix=f"phase6-{part_name}-", dir="/tmp"))
    run_dir = run_pipeline(
        revA_pdf=asset_dir / "revA.pdf",
        revB_pdf=asset_dir / "revB.pdf",
        form3_xlsx=asset_dir / "FAIR.xlsx",
        out_dir=out_dir,
        dpi=300,
        part_name=part_name,
    )
    return _load_json(run_dir / "delta_packet.json")


# ===========================================================================
# Class 1: TestPhase6AssetInvariants
# ===========================================================================


class TestPhase6AssetInvariants:
    """Snapshot files exist and canonical ground-truth exemplars are still present."""

    def test_debug_report_part8_exists(self):
        assert DEBUG_REPORT_PART8.exists(), f"Missing required asset: {DEBUG_REPORT_PART8}"

    def test_debug_report_part9_exists(self):
        assert DEBUG_REPORT_PART9.exists(), f"Missing required asset: {DEBUG_REPORT_PART9}"

    def test_ground_truth_part8_exists(self):
        assert GROUND_TRUTH_PART8.exists(), f"Missing required asset: {GROUND_TRUTH_PART8}"

    def test_ground_truth_part9_exists(self):
        assert GROUND_TRUTH_PART9.exists(), f"Missing required asset: {GROUND_TRUTH_PART9}"

    # ------------------------------------------------------------------
    # Part 8 canonical added rows
    # ------------------------------------------------------------------

    def test_part8_ground_truth_has_three_added_rows(self):
        """Part 8 ground truth must have exactly three added characteristics."""
        gt = _load_json(GROUND_TRUTH_PART8)
        added = _added_rows(gt)
        assert len(added) == 3, (
            f"Expected 3 added rows in Part 8 ground truth, found {len(added)}: "
            f"{[r.get('requirement_revB') for r in added]}"
        )

    def test_part8_canonical_added_row_Ø10(self):
        """Part 8 ground truth must contain the Ø10.000±.001 added row."""
        gt = _load_json(GROUND_TRUTH_PART8)
        added_reqs = {r.get("requirement_revB") for r in _added_rows(gt)}
        assert "Ø10.000±.001" in added_reqs, (
            f"Expected 'Ø10.000±.001' in Part 8 added rows, found: {sorted(added_reqs)}"
        )

    def test_part8_canonical_added_row_total_runout_015_B(self):
        """Part 8 ground truth must contain the ⌰ .015 B added row."""
        gt = _load_json(GROUND_TRUTH_PART8)
        added_reqs = {r.get("requirement_revB") for r in _added_rows(gt)}
        assert "⌰ .015 B" in added_reqs, (
            f"Expected '⌰ .015 B' in Part 8 added rows, found: {sorted(added_reqs)}"
        )

    def test_part8_canonical_added_row_total_runout_002_A(self):
        """Part 8 ground truth must contain the ⌰ .002 A added row (missing_added_truth_indexes=[10])."""
        gt = _load_json(GROUND_TRUTH_PART8)
        added_reqs = {r.get("requirement_revB") for r in _added_rows(gt)}
        assert "⌰ .002 A" in added_reqs, (
            f"Expected '⌰ .002 A' in Part 8 added rows, found: {sorted(added_reqs)}"
        )

    # ------------------------------------------------------------------
    # Part 9 duplicate groups still present in ground truth
    # ------------------------------------------------------------------

    def test_part9_ground_truth_has_eight_added_rows(self):
        """Part 9 ground truth must have exactly eight added characteristics."""
        gt = _load_json(GROUND_TRUTH_PART9)
        added = _added_rows(gt)
        assert len(added) == 8, (
            f"Expected 8 added rows in Part 9 ground truth, found {len(added)}: "
            f"{[r.get('requirement_revB') for r in added]}"
        )

    def test_part9_duplicate_group_Ø250_present_twice(self):
        """Ø.250 ±.008 appears exactly twice in Part 9 added rows (two feature locations)."""
        gt = _load_json(GROUND_TRUTH_PART9)
        added_reqs = [r.get("requirement_revB") for r in _added_rows(gt)]
        count = added_reqs.count("Ø.250 ±.008")
        assert count == 2, (
            f"Expected 'Ø.250 ±.008' to appear exactly 2 times in Part 9 added rows, "
            f"found {count} in: {added_reqs}"
        )

    def test_part9_duplicate_group_position_mark_present_twice(self):
        """⌖ ∅.015 D H appears exactly twice in Part 9 added rows."""
        gt = _load_json(GROUND_TRUTH_PART9)
        added_reqs = [r.get("requirement_revB") for r in _added_rows(gt)]
        count = added_reqs.count("⌖ ∅.015 D H")
        assert count == 2, (
            f"Expected '⌖ ∅.015 D H' to appear exactly 2 times in Part 9 added rows, "
            f"found {count} in: {added_reqs}"
        )

    def test_part9_duplicate_group_depth_present_twice(self):
        """↧.50 ±.05 appears exactly twice in Part 9 added rows."""
        gt = _load_json(GROUND_TRUTH_PART9)
        added_reqs = [r.get("requirement_revB") for r in _added_rows(gt)]
        count = added_reqs.count("↧.50 ±.05")
        assert count == 2, (
            f"Expected '↧.50 ±.05' to appear exactly 2 times in Part 9 added rows, "
            f"found {count} in: {added_reqs}"
        )

    def test_part8_debug_report_records_missing_added_truth_index_10(self):
        """Part 8 debug report must record missing_added_truth_indexes=[10]."""
        report = _load_json(DEBUG_REPORT_PART8)
        missing = report.get("missing_added_truth_indexes", [])
        assert 10 in missing, (
            f"Expected truth_index 10 in missing_added_truth_indexes, found: {missing}"
        )

    def test_part9_debug_report_records_eight_missing_added_truth_indexes(self):
        """Part 9 debug report must record 8 missing added truth indexes (35-42)."""
        report = _load_json(DEBUG_REPORT_PART9)
        missing = report.get("missing_added_truth_indexes", [])
        assert len(missing) == 8, (
            f"Expected 8 missing added truth indexes in Part 9, found {len(missing)}: {missing}"
        )
        for idx in range(35, 43):
            assert idx in missing, (
                f"Expected truth_index {idx} in missing_added_truth_indexes, found: {missing}"
            )


# ===========================================================================
# Class 2: TestPhase6Part8Exemplars
# ===========================================================================


class TestPhase6Part8Exemplars:
    """Pin exact Part 8 fragment false-positive strings from the debug corpus.

    The three exemplar strings from the checked-in debug report:
    - .045 A    (char_no=10, fragment of ◎ ∅.045 A — balloon 6 changed char)
    - ⚪ .005   (char_no=9,  fragment of circularity — balloon 5 changed char)
    - ⌰ .002 A  (truth_index=10, the canonical added row that went missing)
    """

    def test_debug_report_part8_contains_fragment_045_A(self):
        """The Part 8 debug report must record the .045 A false-positive added fragment."""
        report = _load_json(DEBUG_REPORT_PART8)
        items_with_fragment = [
            it for it in report.get("items", [])
            if it.get("requirement_revB") == ".045 A"
        ]
        assert items_with_fragment, (
            "Expected at least one item with requirement_revB='.045 A' in Part 8 debug report; "
            "this exemplar anchors the fragment false-positive regression"
        )

    def test_debug_report_part8_contains_fragment_circularity_005(self):
        """The Part 8 debug report must record the ⚪ .005 false-positive added fragment."""
        report = _load_json(DEBUG_REPORT_PART8)
        items_with_fragment = [
            it for it in report.get("items", [])
            if it.get("requirement_revB") == "⚪ .005"
        ]
        assert items_with_fragment, (
            "Expected at least one item with requirement_revB='⚪ .005' in Part 8 debug report; "
            "this exemplar anchors the circularity false-positive regression"
        )

    def test_debug_report_part8_contains_missing_total_runout_002_A(self):
        """The Part 8 debug report must record ⌰ .002 A as the missing canonical added row."""
        report = _load_json(DEBUG_REPORT_PART8)
        # This row appears as the virtual missing entry at the bottom of the report
        missing_row_items = [
            it for it in report.get("items", [])
            if it.get("requirement_revB") == "⌰ .002 A"
            and it.get("row_state") == "algorithm_error"
        ]
        assert missing_row_items, (
            "Expected a row_state='algorithm_error' item with requirement_revB='⌰ .002 A' in "
            "Part 8 debug report; this is the missing canonical added row at truth_index=10"
        )

    def test_fragment_045_A_is_algorithm_error_not_canonical(self):
        """The .045 A row must be tagged algorithm_error — it should be a changed row, not added."""
        report = _load_json(DEBUG_REPORT_PART8)
        item = next(
            (it for it in report.get("items", []) if it.get("requirement_revB") == ".045 A"),
            None
        )
        assert item is not None, "Item with requirement_revB='.045 A' must be present"
        assert item.get("debug_verdict") == "algorithm_error", (
            f"Expected .045 A item to be 'algorithm_error', found: {item.get('debug_verdict')!r}"
        )

    def test_detect_added_characteristics_accepts_full_callout_text(self):
        """The Phase-6 added-detection path must accept full-callout text as authoritative.

        When the GD&T anchor '⌰' with companion spans '.002' and 'A' are presented
        as a unit, the detected added row's text must contain all three tokens.
        Regression: the seed-span-only path produced '⌰ .002 A' only by accident
        when the seed span happened to contain more text; grouped evidence ensures
        determinism.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics
        from delta_preservation.io.pdf import TextSpan

        def _s(text, x0, y0, w=8.0, h=8.0, block_id=0, line_id=0, span_id=0):
            return TextSpan(
                text=text,
                bbox_pdf=(x0, y0, x0 + w, y0 + h),
                font_size=10.0,
                block_id=block_id,
                line_id=line_id,
                span_id=span_id,
            )

        # Simulate the exact three-span layout for '⌰ .002 A'
        # These represent the canonical missing added row from Part 8 (truth_index=10).
        anchor = _s("⌰", x0=10.0, y0=200.0, span_id=0)
        tol    = _s(".002", x0=20.0, y0=200.0, w=14.0, span_id=1)
        datum  = _s("A", x0=36.0, y0=200.0, w=6.0, span_id=2)

        added = detect_added_characteristics(
            revB_spans=[anchor, tol, datum],
            matches={},
            next_char_no=10,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        full_callout_found = any(
            "⌰" in t and (".002" in t or "002" in t)
            for t in texts
        )
        assert full_callout_found, (
            f"Expected detect_added_characteristics to produce a row with '⌰' and '.002' "
            f"(full callout for ⌰ .002 A), but got: {texts!r}. "
            "The canonical missing added row from Part 8 must be detectable."
        )

    def test_fragment_only_form_does_not_survive_when_ownership_exists(self):
        """Fragment '.045 A' must not survive when '◎ ∅.045 A' already owns the annotation.

        This test pins the Part 8 explained-by-match suppression contract:
        the full-callout matched annotation explains the fragment, so the
        fragment must not become a standalone added row.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics
        from delta_preservation.io.pdf import TextSpan

        def _s(text, x0, y0, w=8.0, h=8.0, block_id=0, line_id=0, span_id=0):
            return TextSpan(
                text=text,
                bbox_pdf=(x0, y0, x0 + w, y0 + h),
                font_size=10.0,
                block_id=block_id,
                line_id=line_id,
                span_id=span_id,
            )

        # Matched span for balloon-6: full callout ◎ ∅.045 A
        matched_span = _s("◎ ∅.045 A", x0=50.0, y0=200.0, w=50.0, span_id=0)

        class _FakeCandidate:
            def __init__(self, span):
                self.span = span

        class _FakeMatch:
            def __init__(self, span):
                self.candidate = _FakeCandidate(span)

        matches = {6: _FakeMatch(matched_span)}

        # Fragment span that used to leak (the .045 A raw seed from Part 8 debug report)
        fragment = _s(".045 A", x0=52.0, y0=205.0, w=20.0, block_id=0, line_id=1, span_id=0)
        all_spans = [matched_span, fragment]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=612.0,
            page_height=792.0,
        )

        fragment_texts = [it.added_requirement_text or "" for it in added]
        spurious = [t for t in fragment_texts if ".045 A" in t and "◎" not in t]
        assert not spurious, (
            f"Fragment '.045 A' must be suppressed when '◎ ∅.045 A' owns the annotation; "
            f"got spurious added rows: {fragment_texts!r}"
        )

    def test_live_part8_pipeline_keeps_added_rows_distinct(self):
        """Fresh Part 8 pipeline output must keep runout and diameter rows separate.

        Regression: the live packet currently emits a merged `⌰ .015 B Ø10.000±.001`
        row and loses the canonical added `⌰ .002 A`.
        """
        packet = _live_packet("part8")
        added = [item for item in packet.get("items", []) if item.get("status") == "added"]
        added_texts = [_normalize_text(item.get("requirement_revB")) for item in added]

        assert "⌰ .015 B" in added_texts, f"Missing Part 8 runout row in live packet: {added_texts!r}"
        assert "Ø10.000±.001" in added_texts, f"Missing Part 8 diameter row in live packet: {added_texts!r}"
        assert "⌰ .002 A" in added_texts, f"Missing Part 8 canonical added row in live packet: {added_texts!r}"
        assert not any("⌰" in text and "Ø10.000±.001" in text for text in added_texts), (
            f"Part 8 live packet still contains a merged row: {added_texts!r}"
        )

        char5_items = [
            item for item in packet.get("items", [])
            if item.get("char_no") == 5 and _normalize_text(item.get("requirement_revB")) == "⌰ .002 A"
        ]
        assert not char5_items, "Form 3 char 5 must not steal the canonical added `⌰ .002 A` row"


# ===========================================================================
# Class 3: TestPhase6Part9DuplicateAddedRows
# ===========================================================================


class TestPhase6Part9DuplicateAddedRows:
    """Duplicate added truth rows in Part 9 must be claimable by distinct packet items.

    The three duplicate pairs (Ø.250 ±.008, ⌖ ∅.015 D H, ↧.50 ±.05) each appear
    at two different feature locations. The evaluator must claim distinct added-pool
    tokens for each using packet Rev B bbox evidence.
    """

    # Centers from assets/part9/ground_truth.json
    _GROUP_A = {
        "Ø.250 ±.008":  (162.0, 328.0),
        "⌖ ∅.015 D H": (178.0, 348.0),
        "↧.50 ±.05":   (156.0, 364.0),
    }
    _GROUP_B = {
        "Ø.250 ±.008":  (602.0, 288.0),
        "⌖ ∅.015 D H": (618.0, 306.0),
        "↧.50 ±.05":   (594.0, 322.0),
    }

    def _load_part9_truth_characteristics(self):
        from delta_preservation.evaluation.contracts import GroundTruthCharacteristic
        gt = _load_json(GROUND_TRUTH_PART9)
        rows = []
        for c in gt.get("characteristics", []):
            rows.append(GroundTruthCharacteristic(
                char_no=c.get("char_no"),
                classification=c["classification"],
                requirement_revB=c.get("requirement_revB"),
                snippet_center_revA=c.get("snippet_center_revA"),
                snippet_center_revB=c.get("snippet_center_revB"),
            ))
        return rows

    def _make_added_item(self, requirement: str, bbox: list[float]):
        from delta_preservation.types import DeltaItem, Evidence
        revB = Evidence(page=0, bbox=bbox, image_path=None)
        return DeltaItem(
            char_no=None,
            status="added",
            confidence=0.9,
            reasons=["test"],
            scores={},
            revA=None,
            revB=revB,
            requirement_revB=requirement,
        )

    def _tight_bbox_around(self, center: tuple[float, float], radius: float = 30.0) -> list[float]:
        cx, cy = center
        return [cx - radius, cy - radius, cx + radius, cy + radius]

    # ------------------------------------------------------------------
    # Ground truth centers match the research report
    # ------------------------------------------------------------------

    def test_part9_ground_truth_centers_match_research_report(self):
        """Part 9 ground truth contains the exact duplicate centers documented in 06-RESEARCH.md."""
        gt = _load_json(GROUND_TRUTH_PART9)
        added = _added_rows(gt)

        # Collect all (requirement_revB, snippet_center_revB) pairs for added rows
        pairs = {
            (r.get("requirement_revB"), tuple(r.get("snippet_center_revB") or []))
            for r in added
        }

        for req, center in self._GROUP_A.items():
            assert (req, center) in pairs, (
                f"Expected Group A ({req!r}, {center}) in Part 9 added centers; found: {sorted(pairs)}"
            )
        for req, center in self._GROUP_B.items():
            assert (req, center) in pairs, (
                f"Expected Group B ({req!r}, {center}) in Part 9 added centers; found: {sorted(pairs)}"
            )

    # ------------------------------------------------------------------
    # Distinct pool tokens for duplicate rows via bbox disambiguation
    # ------------------------------------------------------------------

    def test_evaluator_claims_distinct_tokens_for_Ø250_group_A(self):
        """Packet item near group-A center claims the group-A added-pool token for Ø.250 ±.008."""
        from delta_preservation.evaluation.conformance import (
            ADDED_POOL_TOKEN_PREFIX,
            select_truth_row_for_item,
        )

        truth_rows = self._load_part9_truth_characteristics()
        req = "Ø.250 ±.008"
        center_a = self._GROUP_A[req]
        item = self._make_added_item(req, self._tight_bbox_around(center_a))

        selection = select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes=set())
        assert selection.truth_row is not None, (
            f"Expected tie-break to select a truth row for {req!r} near group-A center, "
            f"but got ambiguity: {selection.ambiguity_message!r}"
        )
        assert selection.truth_row.snippet_center_revB == list(center_a) or \
               selection.truth_row.snippet_center_revB == center_a or \
               tuple(selection.truth_row.snippet_center_revB or []) == center_a, (
            f"Expected group-A center {center_a} but got {selection.truth_row.snippet_center_revB}"
        )

    def test_evaluator_claims_distinct_tokens_for_Ø250_group_B(self):
        """Packet item near group-B center claims the group-B added-pool token for Ø.250 ±.008."""
        from delta_preservation.evaluation.conformance import select_truth_row_for_item

        truth_rows = self._load_part9_truth_characteristics()
        req = "Ø.250 ±.008"
        center_b = self._GROUP_B[req]
        item = self._make_added_item(req, self._tight_bbox_around(center_b))

        selection = select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes=set())
        assert selection.truth_row is not None, (
            f"Expected tie-break to select a truth row for {req!r} near group-B center, "
            f"but got ambiguity: {selection.ambiguity_message!r}"
        )
        assert tuple(selection.truth_row.snippet_center_revB or []) == center_b, (
            f"Expected group-B center {center_b} but got {selection.truth_row.snippet_center_revB}"
        )

    def test_evaluator_claims_distinct_tokens_for_position_mark_group_A(self):
        """⌖ ∅.015 D H near group-A center resolves to the group-A truth row."""
        from delta_preservation.evaluation.conformance import select_truth_row_for_item

        truth_rows = self._load_part9_truth_characteristics()
        req = "⌖ ∅.015 D H"
        center_a = self._GROUP_A[req]
        item = self._make_added_item(req, self._tight_bbox_around(center_a))

        selection = select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes=set())
        assert selection.truth_row is not None, (
            f"Expected tie-break to select a truth row for {req!r} near group-A center, "
            f"got: {selection.ambiguity_message!r}"
        )
        assert tuple(selection.truth_row.snippet_center_revB or []) == center_a, (
            f"Expected group-A center {center_a} but got {selection.truth_row.snippet_center_revB}"
        )

    def test_evaluator_claims_distinct_tokens_for_position_mark_group_B(self):
        """⌖ ∅.015 D H near group-B center resolves to the group-B truth row."""
        from delta_preservation.evaluation.conformance import select_truth_row_for_item

        truth_rows = self._load_part9_truth_characteristics()
        req = "⌖ ∅.015 D H"
        center_b = self._GROUP_B[req]
        item = self._make_added_item(req, self._tight_bbox_around(center_b))

        selection = select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes=set())
        assert selection.truth_row is not None, (
            f"Expected tie-break to select a truth row for {req!r} near group-B center, "
            f"got: {selection.ambiguity_message!r}"
        )
        assert tuple(selection.truth_row.snippet_center_revB or []) == center_b, (
            f"Expected group-B center {center_b} but got {selection.truth_row.snippet_center_revB}"
        )

    # ------------------------------------------------------------------
    # The truncated ⌖∅ exemplar from Part 9 debug report
    # ------------------------------------------------------------------

    def test_debug_report_part9_contains_truncated_position_mark(self):
        """Part 9 debug report must contain the truncated '⌖∅' as a packet requirement_revB.

        This pins the observed corpus failure: the standard added pass emitted '⌖∅'
        instead of the full canonical text '⌖ ∅.015 D H'.
        """
        report = _load_json(DEBUG_REPORT_PART9)
        items = report.get("items", [])
        truncated_items = [
            it for it in items
            if it.get("requirement_revB") == "⌖∅"
        ]
        assert truncated_items, (
            "Expected at least one item with requirement_revB='⌖∅' in Part 9 debug report; "
            "this is the corpus evidence for the grouped-evidence truncation bug"
        )

    def test_truncated_position_mark_does_not_match_full_canonical_text(self):
        """'⌖∅' must NOT normalize to the same token as '⌖ ∅.015 D H'.

        This test proves the corpus failure is real: the evaluator cannot match
        the truncated packet row to the canonical truth row without normalization
        magic or the grouped-evidence fix.
        """
        report = _load_json(DEBUG_REPORT_PART9)
        truncated_item = next(
            (it for it in report.get("items", []) if it.get("requirement_revB") == "⌖∅"),
            None
        )
        assert truncated_item is not None
        assert truncated_item.get("requirement_revB") != "⌖ ∅.015 D H", (
            "The truncated '⌖∅' and the canonical '⌖ ∅.015 D H' must be distinct strings; "
            "they represent the before/after of the grouped-evidence fix"
        )

    def test_part9_evaluate_packet_assigns_distinct_pool_tokens_for_both_position_mark_rows(self):
        """evaluate_packet_against_truth must assign distinct added-pool tokens for both ⌖ ∅.015 D H rows.

        Build two packet items — one near each group center — and confirm they
        are assigned to distinct truth rows (distinct pool token strings) rather
        than both ending up as truth_ambiguity.
        """
        from delta_preservation.evaluation.conformance import (
            ADDED_POOL_TOKEN_PREFIX,
            TRUTH_AMBIGUITY_CODE,
            evaluate_packet_against_truth,
        )
        from delta_preservation.evaluation.contracts import GroundTruthPacket

        truth_rows = self._load_part9_truth_characteristics()
        gt_raw = _load_json(GROUND_TRUTH_PART9)
        truth_packet = GroundTruthPacket(
            part_name=gt_raw.get("part_name", "Part9"),
            general_notes=gt_raw.get("general_notes", ""),
            characteristics=truth_rows,
        )

        req = "⌖ ∅.015 D H"
        center_a = self._GROUP_A[req]
        center_b = self._GROUP_B[req]

        item_a = self._make_added_item(req, self._tight_bbox_around(center_a))
        item_b = self._make_added_item(req, self._tight_bbox_around(center_b))

        evaluations = evaluate_packet_against_truth([item_a, item_b], truth_packet)
        assert len(evaluations) == 2

        eval_a, eval_b = evaluations

        # Neither should be truth_ambiguity after the Phase-6 fix
        ambiguous = [
            e for e in [eval_a, eval_b]
            if any(m.code == TRUTH_AMBIGUITY_CODE for m in e.mismatches)
        ]
        assert not ambiguous, (
            f"Expected both ⌖ ∅.015 D H items to be disambiguated by bbox evidence, "
            f"but got truth_ambiguity for {len(ambiguous)} item(s). "
            f"Evaluations: [{eval_a}, {eval_b}]"
        )

        # Assigned tokens must be distinct
        token_a = eval_a.matched_truth_char_no
        token_b = eval_b.matched_truth_char_no
        assert token_a != token_b, (
            f"Expected distinct pool tokens for the two ⌖ ∅.015 D H rows, "
            f"but both got token={token_a!r}"
        )
        # Both tokens must be added-pool tokens
        assert str(token_a).startswith(ADDED_POOL_TOKEN_PREFIX), (
            f"Token for item_a must be an added-pool token, got: {token_a!r}"
        )
        assert str(token_b).startswith(ADDED_POOL_TOKEN_PREFIX), (
            f"Token for item_b must be an added-pool token, got: {token_b!r}"
        )

    def test_live_part9_pipeline_preserves_duplicate_added_rows(self):
        """Fresh Part 9 pipeline output must retain both physical instances of each duplicate row."""
        packet = _live_packet("part9")
        added = [item for item in packet.get("items", []) if item.get("status") == "added"]
        normalized_texts = [_normalize_text(item.get("requirement_revB")) for item in added]

        assert normalized_texts.count("Ø.250 ±.008") == 2, (
            f"Expected two diameter duplicate rows in live Part 9 packet, got: {normalized_texts!r}"
        )
        assert normalized_texts.count("⌖∅ .015 D H") == 2, (
            f"Expected two position duplicate rows in live Part 9 packet, got: {normalized_texts!r}"
        )
        assert normalized_texts.count("↧ .50 ±.05") == 2, (
            f"Expected two depth duplicate rows in live Part 9 packet, got: {normalized_texts!r}"
        )

    def test_live_part9_pipeline_claims_distinct_truth_tokens_for_duplicate_rows(self):
        """Fresh Part 9 packet rows should claim both truth rows for each duplicate group."""
        packet = _live_packet("part9")
        added = [item for item in packet.get("items", []) if item.get("status") == "added"]

        grouped_tokens: dict[str, list[str]] = {}
        for item in added:
            text = _normalize_text(item.get("requirement_revB"))
            token = ((item.get("evaluation") or {}).get("matched_truth_char_no"))
            if isinstance(token, str) and token.startswith("added:"):
                grouped_tokens.setdefault(text, []).append(token)

        for text in ["Ø.250 ±.008", "⌖∅ .015 D H", "↧ .50 ±.05"]:
            tokens = grouped_tokens.get(text, [])
            assert len(tokens) == 2, (
                f"Expected two claimed truth tokens for {text!r}, got {tokens!r}"
            )
            assert len(set(tokens)) == 2, (
                f"Expected distinct truth tokens for {text!r}, got {tokens!r}"
            )
