import json
from unittest.mock import patch

from sqlalchemy.orm import sessionmaker

from delta_preservation.evaluation.contracts import GroundTruthCharacteristic, GroundTruthPacket
from shop.models import Run
from shop.services.review import build_debug_queue_state


def _seed_run(db_engine, tmp_path, *, items, part_number="PN-ROW", packet_inputs=None):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / "debug-row-identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    packet = {"run_id": "row-identity", "items": items}
    if packet_inputs is not None:
        packet["inputs"] = packet_inputs
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))
    run = Run(
        part_number=part_number,
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-ROW",
        status="completed",
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return db, run


def _make_truth_packet(*classifications):
    """Build a GroundTruthPacket with characteristics of the given classifications.

    Supplies the minimum required fields per classification:
    - "removed": only char_no + snippet_center_revA (no requirement_revB, no snippet_center_revB)
    - "added":   only requirement_revB + snippet_center_revB (no char_no, no snippet_center_revA)
    - others:    all four required fields
    """
    chars = []
    for i, cls in enumerate(classifications):
        char_no = None if cls == "added" else (i + 1)
        req_revB = None if cls == "removed" else f"req-{i}"
        center_revA = None if cls == "added" else (0.0, 0.0)
        center_revB = None if cls == "removed" else (0.0, 0.0)
        chars.append(
            GroundTruthCharacteristic(
                char_no=char_no,
                classification=cls,
                requirement_revB=req_revB,
                snippet_center_revA=center_revA,
                snippet_center_revB=center_revB,
            )
        )
    return GroundTruthPacket(part_name="test", general_notes="", characteristics=chars)


def test_duplicate_and_none_char_rows_keep_distinct_review_item_ids(db_engine, tmp_path):
    db, run = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 5,
                "status": "changed",
                "confidence": 0.91,
                "requirement_revB": "Packet duplicate first",
                "scores": {"location": 0.3},
                "reasons": ["first duplicate"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 5,
                    "snippet_conforms": False,
                    "mismatches": [{"code": "first_duplicate", "message": "first duplicate"}],
                },
            },
            {
                "char_no": 5,
                "status": "removed",
                "confidence": 0.42,
                "requirement_revB": "Packet duplicate second",
                "scores": {"location": 0.2},
                "reasons": ["second duplicate"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 5,
                    "snippet_conforms": False,
                    "mismatches": [{"code": "second_duplicate", "message": "second duplicate"}],
                },
            },
            {
                "char_no": None,
                "status": "added",
                "confidence": 0.57,
                "requirement_revB": "Packet null char",
                "scores": {"location": 0.1},
                "reasons": ["null char row"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": "added-0",
                    "snippet_conforms": False,
                    "mismatches": [{"code": "null_char", "message": "null char row"}],
                },
            },
        ],
    )

    try:
        queue_state = build_debug_queue_state(db, run)
        all_items = queue_state["all_items"]
        exception_items = queue_state["exception_items"]
        packet_items_by_item_id = queue_state["packet_items_by_item_id"]

        assert [item.char_no for item in all_items] == [5, 5, None]
        assert [item.id for item in all_items] == sorted({item.id for item in all_items})
        assert [item.id for item in exception_items] == [item.id for item in all_items]

        paired_requirements = [
            packet_items_by_item_id[item.id].requirement_revB
            for item in all_items
        ]
        assert paired_requirements == [
            "Packet duplicate first",
            "Packet duplicate second",
            "Packet null char",
        ]
        assert queue_state["debug_total"] == 3
    finally:
        db.close()


def test_missing_added_truth_indexes_detected_when_packet_misses_a_truth_added_row(db_engine, tmp_path):
    """Ground truth has 2 added rows; packet only claimed index 0 — index 1 is missing."""
    # Packet has one "added" item matching truth index 0
    items = [
        {
            "char_no": None,
            "status": "added",
            "confidence": 0.80,
            "requirement_revB": "Added item 0",
            "scores": {},
            "reasons": [],
            "revA": None,
            "revB": None,
            "evaluation": {
                "status": "conforming",
                "matched_truth_char_no": "added:0",
                "snippet_conforms": True,
                "mismatches": [],
            },
        }
    ]
    db, run = _seed_run(db_engine, tmp_path, items=items)

    # Ground truth has 2 added rows (index 0 and 1)
    truth_packet = _make_truth_packet("added", "added")

    try:
        with patch("shop.services.review.load_ground_truth_packet", return_value=truth_packet):
            queue_state = build_debug_queue_state(db, run)

        assert queue_state["missing_added_truth_indexes"] == [1]
        exception_count = len(queue_state["exception_items"])
        assert queue_state["debug_total"] == exception_count + 1
    finally:
        db.close()


def test_no_missing_added_when_all_truth_added_rows_are_claimed(db_engine, tmp_path):
    """Ground truth has 1 added row; packet claimed index 0 — nothing missing."""
    items = [
        {
            "char_no": None,
            "status": "added",
            "confidence": 0.80,
            "requirement_revB": "Added item 0",
            "scores": {},
            "reasons": [],
            "revA": None,
            "revB": None,
            "evaluation": {
                "status": "conforming",
                "matched_truth_char_no": "added:0",
                "snippet_conforms": True,
                "mismatches": [],
            },
        }
    ]
    db, run = _seed_run(db_engine, tmp_path, items=items)

    truth_packet = _make_truth_packet("added")

    try:
        with patch("shop.services.review.load_ground_truth_packet", return_value=truth_packet):
            queue_state = build_debug_queue_state(db, run)

        assert queue_state["missing_added_truth_indexes"] == []
    finally:
        db.close()


def test_missing_added_silently_skipped_when_no_fixture(db_engine, tmp_path):
    """No truth_fixture_key in inputs and part_number doesn't match any fixture — silent []."""
    items = [
        {
            "char_no": 1,
            "status": "unchanged",
            "confidence": 0.95,
            "requirement_revB": "Some req",
            "scores": {},
            "reasons": [],
            "revA": None,
            "revB": None,
            "evaluation": {
                "status": "conforming",
                "matched_truth_char_no": 1,
                "snippet_conforms": True,
                "mismatches": [],
            },
        }
    ]
    # Use a part_number that doesn't have a fixture and no inputs key
    db, run = _seed_run(db_engine, tmp_path, items=items, part_number="NO-FIXTURE-PART")

    try:
        with patch(
            "shop.services.review.load_ground_truth_packet",
            side_effect=Exception("fixture not found"),
        ):
            queue_state = build_debug_queue_state(db, run)

        assert queue_state["missing_added_truth_indexes"] == []
        # No exception was raised
    finally:
        db.close()
