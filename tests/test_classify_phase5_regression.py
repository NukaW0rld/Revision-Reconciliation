"""Phase 5 regression harness — snapshot exemplars and synthetic CLS-02 coverage.

Three test classes:

  TestPhase5SnapshotExemplars
    Asserts positive and negative bleed/asymmetric exemplars using exact
    requirement_revB strings sourced from the checked-in debug_report_part*.json
    snapshots.  Does not guess counts — uses explicit assertions.

  TestPhase5SnapshotSweep
    Iterates every snapshot item where evaluation.status == "conforming" and
    pipeline_classification == "unchanged" and asserts that neither bleed
    detection nor the asymmetric-shape regex fires unless the exact string is in
    an explicit allowlist.

  TestPhase5SyntheticReconciliation
    Packet-level synthetic regression for CLS-02 (reconcile_removed_added_pairs).
    Uses inline DeltaItem / Anchor construction following test_classify_bugfixes.py
    conventions.  Covers two cases:
      1. Grouped compatible added item near a removed anchor → changed
      2. Same geometry but added_page differs → items stay separate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent.parent / "assets"

SNAPSHOT_FILES = {
    "part1": ASSETS_DIR / "debug_report_part1.json",
    "part4": ASSETS_DIR / "debug_report_part4.json",
    "part7": ASSETS_DIR / "debug_report_part7.json",
}


# ---------------------------------------------------------------------------
# Helpers shared by multiple classes
# ---------------------------------------------------------------------------

def _load_snapshot(name: str) -> dict:
    path = SNAPSHOT_FILES[name]
    assert path.exists(), f"Snapshot file missing: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _items_by_revB(snapshot: dict) -> dict[str, dict]:
    """Return a dict keyed by requirement_revB string for all top-level items."""
    result: dict[str, dict] = {}
    for item in snapshot.get("items", []):
        revB = item.get("requirement_revB")
        if revB is not None:
            # Last writer wins when duplicates exist (same revB may appear for multiple chars)
            result[revB] = item
    return result


# ---------------------------------------------------------------------------
# TextSpan helper (mirrors test_classify_bugfixes._span)
# ---------------------------------------------------------------------------

from delta_preservation.io.pdf import TextSpan


def _span(
    text: str,
    *,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
    x0: float = 10.0,
    y0: float = 10.0,
    width: float = 20.0,
    height: float = 8.0,
) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


# ===========================================================================
# Class 1: Explicit exemplar assertions
# ===========================================================================

class TestPhase5SnapshotExemplars:
    """Positive/negative exemplar guards using exact strings from checked-in snapshots."""

    # ------------------------------------------------------------------
    # Existence gate: snapshot files must be present before any other test
    # runs (Rule 3 guard — tests blocked without fixtures).
    # ------------------------------------------------------------------

    def test_snapshot_files_exist(self):
        for name, path in SNAPSHOT_FILES.items():
            assert path.exists(), f"Required snapshot file missing: {path} ({name})"

    # ------------------------------------------------------------------
    # Load exemplar rows from their canonical snapshot files
    # ------------------------------------------------------------------

    def _part1_items(self):
        return _items_by_revB(_load_snapshot("part1"))

    def _part4_items(self):
        return _items_by_revB(_load_snapshot("part4"))

    def _part7_items(self):
        return _items_by_revB(_load_snapshot("part7"))

    # ------------------------------------------------------------------
    # Bleed exemplar — positive: "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5" (part1)
    # ------------------------------------------------------------------

    def test_bleed_positive_four_x_hole(self):
        """4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5 must trigger bleed detection."""
        from delta_preservation.reconcile.classify import _looks_like_adjacency_bleed

        revB = "4 x \u00d88 THRU ALL / \u2334 \u00d813.5 \u21a7 8.5"
        items = self._part1_items()
        assert revB in items, (
            f"Expected snapshot row with requirement_revB={revB!r} in part1 snapshot"
        )
        revA = items[revB]["requirement_revA"]
        assert _looks_like_adjacency_bleed(revB, revA) is True, (
            f"Expected bleed=True for revB={revB!r}, revA={revA!r}"
        )

    # ------------------------------------------------------------------
    # Bleed exemplar — positive: "2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B" (part4)
    # ------------------------------------------------------------------

    def test_bleed_positive_twoX_drilled_thread(self):
        """2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B must trigger bleed detection."""
        from delta_preservation.reconcile.classify import _looks_like_adjacency_bleed

        revB = "2X \u00d8.201 \u21a7 0.50 / 1/4-20 UNC - 2B"
        items = self._part4_items()
        assert revB in items, (
            f"Expected snapshot row with requirement_revB={revB!r} in part4 snapshot"
        )
        revA = items[revB]["requirement_revA"]
        assert _looks_like_adjacency_bleed(revB, revA) is True, (
            f"Expected bleed=True for revB={revB!r}, revA={revA!r}"
        )

    # ------------------------------------------------------------------
    # Bleed exemplar — negative: "70 / 30" (part7)
    # Slash without surrounding count prefix — must NOT trigger bleed.
    # ------------------------------------------------------------------

    def test_bleed_negative_seventy_slash_thirty(self):
        """70 / 30 must NOT trigger bleed detection (no count prefix)."""
        from delta_preservation.reconcile.classify import _looks_like_adjacency_bleed

        revB = "70 / 30"
        items = self._part7_items()
        assert revB in items, (
            f"Expected snapshot row with requirement_revB={revB!r} in part7 snapshot"
        )
        revA = items[revB]["requirement_revA"]
        assert _looks_like_adjacency_bleed(revB, revA) is False, (
            f"Expected bleed=False for revB={revB!r}, revA={revA!r}"
        )

    # ------------------------------------------------------------------
    # Asymmetric exemplar: "2X 22.0° +0.3° / −0.1°" (part7)
    # Must match _ASYMMETRIC_SHAPE_RE.
    # ------------------------------------------------------------------

    def test_asymmetric_shape_re_matches_part7_exemplar(self):
        """_ASYMMETRIC_SHAPE_RE must match the asymmetric tolerance exemplar from part7."""
        from delta_preservation.reconcile.classify import _ASYMMETRIC_SHAPE_RE

        revB = "2X 22.0\u00b0 +0.3\u00b0 / \u22120.1\u00b0"
        assert _ASYMMETRIC_SHAPE_RE.search(revB) is not None, (
            f"Expected _ASYMMETRIC_SHAPE_RE to match: {revB!r}"
        )


# ===========================================================================
# Class 2: Snapshot sweep — all conforming+unchanged items must be clean
# ===========================================================================

class TestPhase5SnapshotSweep:
    """Sweep conforming+unchanged items across all 3 snapshots.

    Bleed and asymmetric-shape detection must not fire unless the exact
    requirement_revB string is in an explicit allowlist.
    """

    # Explicit allowlist for bleed: strings where _looks_like_adjacency_bleed
    # is expected to return True for a conforming+unchanged item.
    # (Currently empty — no conforming+unchanged item should have bleed.)
    BLEED_ALLOWLIST: frozenset[str] = frozenset()

    # Explicit allowlist for asymmetric shape: strings where _ASYMMETRIC_SHAPE_RE
    # is expected to match for a conforming+unchanged item.
    # (Currently empty — conforming+unchanged asymmetric items should be
    # classified as 'changed' by the fixed classifier.)
    ASYMMETRIC_ALLOWLIST: frozenset[str] = frozenset()

    def _conforming_unchanged_items(self):
        """Yield (snapshot_name, item) tuples for all conforming+unchanged rows."""
        for name in SNAPSHOT_FILES:
            snapshot = _load_snapshot(name)
            for item in snapshot.get("items", []):
                evaluation = item.get("evaluation") or {}
                if (
                    evaluation.get("status") == "conforming"
                    and item.get("pipeline_classification") == "unchanged"
                ):
                    yield name, item

    def test_no_unexpected_bleed_in_conforming_unchanged(self):
        """Bleed detection must not fire on any conforming+unchanged item
        unless the exact revB string is in BLEED_ALLOWLIST."""
        from delta_preservation.reconcile.classify import _looks_like_adjacency_bleed

        violations: list[str] = []
        for snapshot_name, item in self._conforming_unchanged_items():
            revB = item.get("requirement_revB") or ""
            revA = item.get("requirement_revA") or ""
            if revB in self.BLEED_ALLOWLIST:
                continue
            result = _looks_like_adjacency_bleed(revB, revA)
            if result:
                violations.append(
                    f"[{snapshot_name}] char_no={item.get('char_no')} "
                    f"revB={revB!r} unexpectedly triggered bleed detection"
                )
        assert not violations, (
            "Unexpected bleed detections in conforming+unchanged items:\n"
            + "\n".join(violations)
        )

    def test_no_unexpected_asymmetric_in_conforming_unchanged(self):
        """Asymmetric-shape regex must not match any conforming+unchanged item
        unless the exact revB string is in ASYMMETRIC_ALLOWLIST."""
        from delta_preservation.reconcile.classify import _ASYMMETRIC_SHAPE_RE

        violations: list[str] = []
        for snapshot_name, item in self._conforming_unchanged_items():
            revB = item.get("requirement_revB") or ""
            if revB in self.ASYMMETRIC_ALLOWLIST:
                continue
            if _ASYMMETRIC_SHAPE_RE.search(revB):
                violations.append(
                    f"[{snapshot_name}] char_no={item.get('char_no')} "
                    f"revB={revB!r} matched asymmetric-shape regex unexpectedly"
                )
        assert not violations, (
            "Unexpected asymmetric-shape matches in conforming+unchanged items:\n"
            + "\n".join(violations)
        )


# ===========================================================================
# Class 3: Synthetic packet-level regression for CLS-02
# ===========================================================================

class TestPhase5SyntheticReconciliation:
    """Packet-level synthetic regression for reconcile_removed_added_pairs.

    Tests two scenarios using inline DeltaItem / Anchor construction:
      1. Grouped compatible added item close to a removed anchor → 'changed'
      2. Same geometry but added_page differs → items stay separate
    """

    # ------------------------------------------------------------------
    # Helpers (mirror test_classify_bugfixes.py pattern)
    # ------------------------------------------------------------------

    def _removed_item(
        self,
        char_no: int,
        *,
        req_bbox: Optional[tuple] = (50.0, 50.0, 70.0, 58.0),
        requirement_raw: str = "Ø 8.0",
    ):
        from delta_preservation.reconcile.classify import DeltaItem as _DI
        item = _DI(
            char_no=char_no,
            status="removed",
            confidence=0.8,
            reasons=["No candidate found within search window"],
            component_scores={"location": 0.0, "text": 0.0, "context": 0.0},
        )
        # req_bbox stored on the anchor, not the item, so just return item
        return item

    def _anchor_for(
        self,
        char_no: int,
        *,
        req_bbox: Optional[tuple] = (50.0, 50.0, 70.0, 58.0),
        balloon_bbox: tuple = (10.0, 10.0, 20.0, 20.0),
        requirement_raw: str = "Ø 8.0",
        page: int = 0,
    ):
        from delta_preservation.reconcile.anchors import Anchor
        return Anchor(
            char_no=char_no,
            page=page,
            balloon_bbox=balloon_bbox,
            req_bbox=req_bbox,
            requirement_raw=requirement_raw,
            requirement_norm=requirement_raw,
            local_context=[],
        )

    def _added_item(
        self,
        char_no: int,
        *,
        added_requirement_text: str = "Ø 8.5",
        added_bbox: tuple = (55.0, 51.0, 70.0, 59.0),
        added_page: int = 0,
        requirement_raw: Optional[str] = None,
    ):
        from delta_preservation.reconcile.classify import DeltaItem as _DI
        req_raw = requirement_raw or added_requirement_text
        span = _span(req_raw, x0=added_bbox[0], y0=added_bbox[1],
                     width=added_bbox[2] - added_bbox[0],
                     height=added_bbox[3] - added_bbox[1])
        item = _DI(
            char_no=char_no,
            status="added",
            confidence=0.75,
            reasons=[f"New requirement detected in Rev B: \"{req_raw}\""],
            component_scores={"location": 0.0, "text": 1.0, "context": 0.0},
            added_span=span,
        )
        item.added_requirement_text = added_requirement_text
        item.added_bbox = added_bbox
        item.added_page = added_page
        return item

    # ------------------------------------------------------------------
    # Test 1: Grouped compatible added item near removed anchor → changed
    # ------------------------------------------------------------------

    def test_grouped_compatible_added_near_removed_becomes_changed(self):
        """A grouped compatible added item spatially close to a removed anchor
        must cause reconcile_removed_added_pairs to return a 'changed' row."""
        from delta_preservation.reconcile.classify import reconcile_removed_added_pairs

        # Removed char_no=1 with req_bbox centroid at (60, 54)
        removed = self._removed_item(1, req_bbox=(50.0, 50.0, 70.0, 58.0), requirement_raw="Ø 8.0")
        anchor = self._anchor_for(1, req_bbox=(50.0, 50.0, 70.0, 58.0), requirement_raw="Ø 8.0", page=0)

        # Added item using full grouped text; bbox centroid at (62.5, 55) — ~2 pt away
        # added_requirement_text uses the same type (dimensional) as the removed item
        added = self._added_item(
            200,
            added_requirement_text="Ø 8.0 / ⌖ Ø0.35 A B C",
            added_bbox=(60.0, 51.0, 65.0, 59.0),
            added_page=0,
        )

        result = reconcile_removed_added_pairs([removed, added], [anchor])

        changed = [r for r in result if r.char_no == 1]
        assert changed, "Removed item must survive in result (as changed)"
        assert changed[0].status == "changed", (
            f"Expected 'changed' after reconciliation, got '{changed[0].status}'"
        )
        # Consumed added item must not appear separately
        still_added = [r for r in result if r.char_no == 200 and r.status == "added"]
        assert not still_added, "Consumed added item must be removed from output"

        # Reconciliation reason must reference added_requirement_text / metadata
        reasons_text = " ".join(changed[0].reasons)
        assert "reconciled removed+added pair" in reasons_text, (
            f"Expected reconciliation reason, got: {changed[0].reasons!r}"
        )

    # ------------------------------------------------------------------
    # Test 2: Same geometry but added_page differs → items stay separate
    # ------------------------------------------------------------------

    def test_cross_page_added_stays_separate(self):
        """When added_page differs from the removed anchor's page, the pair
        must NOT be reconciled — the page gate must prevent merging."""
        from delta_preservation.reconcile.classify import reconcile_removed_added_pairs

        # Removed char_no=2 with req_bbox centroid at (60, 54) on page 0
        removed = self._removed_item(2, req_bbox=(50.0, 50.0, 70.0, 58.0), requirement_raw="Ø 8.0")
        anchor = self._anchor_for(2, req_bbox=(50.0, 50.0, 70.0, 58.0), requirement_raw="Ø 8.0", page=0)

        # Geometrically identical added item — but on page 1
        added = self._added_item(
            201,
            added_requirement_text="Ø 8.0",
            added_bbox=(60.0, 51.0, 65.0, 59.0),
            added_page=1,   # different page → must not merge
        )

        result = reconcile_removed_added_pairs([removed, added], [anchor])

        removed_items = [r for r in result if r.char_no == 2]
        added_items = [r for r in result if r.char_no == 201]

        assert removed_items and removed_items[0].status == "removed", (
            "Cross-page removed item must stay 'removed'"
        )
        assert added_items and added_items[0].status == "added", (
            "Cross-page added item must stay 'added'"
        )
