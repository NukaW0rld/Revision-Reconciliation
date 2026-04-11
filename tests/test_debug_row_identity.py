import json

from sqlalchemy.orm import sessionmaker

from shop.models import Run
from shop.services.review import build_debug_queue_state


def _seed_run(db_engine, tmp_path, *, items):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / "debug-row-identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_packet.json").write_text(json.dumps({"run_id": "row-identity", "items": items}))
    run = Run(
        part_number="PN-ROW",
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
