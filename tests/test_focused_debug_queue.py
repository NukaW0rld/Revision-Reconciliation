import json

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import ReviewItem, Run
from shop.services.auth import create_session


def _login(client, db_engine, user):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, user)
    db.close()
    client.cookies.set("session_token", token)


def _seed_run(db_engine, tmp_path, *, items, status="completed"):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / f"focused-debug-{status}-{len(items)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_packet.json").write_text(json.dumps({"run_id": "focused-debug", "items": items}))
    run = Run(
        part_number="PN-FDQ",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-FDQ",
        status=status,
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()
    return run_id


def _review_items(db_engine, run_id):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        return (
            db.query(ReviewItem)
            .filter(ReviewItem.run_id == run_id)
            .order_by(ReviewItem.id)
            .all()
        )
    finally:
        db.close()


def test_admin_debug_queue_only_shows_review_needed_rows(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 1,
                "status": "unchanged",
                "confidence": 0.99,
                "requirement_revB": "Conforming row",
                "scores": {},
                "reasons": ["matched truth"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "conforming",
                    "matched_truth_char_no": 1,
                    "snippet_conforms": True,
                    "mismatches": [],
                },
            },
            {
                "char_no": 2,
                "status": "changed",
                "confidence": 0.42,
                "requirement_revB": "Exception row one",
                "scores": {},
                "reasons": ["classification differs"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 2,
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "classification_mismatch", "message": "classification differs"}
                    ],
                },
            },
            {
                "char_no": 3,
                "status": "removed",
                "confidence": 0.37,
                "requirement_revB": "Exception row two",
                "scores": {},
                "reasons": ["missing revB evidence"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 3,
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "snippet_missing_revB", "message": "missing revB evidence"}
                    ],
                },
            },
        ],
    )

    debug_resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert debug_resp.status_code == 200

    review_items = _review_items(db_engine, run_id)
    conforming_item, exception_item_one, exception_item_two = review_items

    debug_html = debug_resp.text
    assert "Debug export readiness: 0 of 2 verdicts saved" in debug_html
    assert f'id="review-item-{conforming_item.id}"' not in debug_html
    assert f'id="review-item-{exception_item_one.id}"' in debug_html
    assert f'id="review-item-{exception_item_two.id}"' in debug_html

    normal_resp = client.get(f"/review/{run_id}", follow_redirects=False)
    assert normal_resp.status_code == 200
    normal_html = normal_resp.text
    assert "Debug export readiness" not in normal_html
    assert f'id="review-item-{conforming_item.id}"' in normal_html
    assert f'id="review-item-{exception_item_one.id}"' in normal_html
    assert f'id="review-item-{exception_item_two.id}"' in normal_html


def test_admin_debug_queue_redirects_all_conforming_run_to_status_page(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 7,
                "status": "unchanged",
                "confidence": 0.97,
                "requirement_revB": "Conforming row one",
                "scores": {},
                "reasons": ["matched truth"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "conforming",
                    "matched_truth_char_no": 7,
                    "snippet_conforms": True,
                    "mismatches": [],
                },
            },
            {
                "char_no": 8,
                "status": "unchanged",
                "confidence": 0.95,
                "requirement_revB": "Conforming row two",
                "scores": {},
                "reasons": ["matched truth"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "conforming",
                    "matched_truth_char_no": 8,
                    "snippet_conforms": True,
                    "mismatches": [],
                },
            },
        ],
    )

    debug_resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert debug_resp.status_code == 302
    assert debug_resp.headers["location"] == f"/runs/{run_id}"

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        assert run is not None
        assert run.status == "reviewing"
    finally:
        db.close()
