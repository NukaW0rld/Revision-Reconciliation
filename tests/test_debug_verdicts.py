import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import ReviewItem, Run
from shop.services.auth import create_session
from shop.services.review import (
    load_debug_verdicts,
    open_review_queue,
    save_debug_verdict,
    validate_debug_verdict_payload,
)


def _login(client, db_engine, user):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, user)
    db.close()
    client.cookies.set("session_token", token)


def _seed_run(db_engine, tmp_path, *, items, status="completed"):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / f"run-{status}-{len(items)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_packet.json").write_text(json.dumps({"run_id": "dbg", "items": items}))
    run = Run(
        part_number="PN-DBG",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-DBG",
        status=status,
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.pdf",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()
    return run_id, out_dir


def _open_queue(db_engine, run_id):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        items = open_review_queue(db, run)
        item_ids = [item.id for item in items]
    finally:
        db.close()
    return item_ids


def _get_item(db_engine, item_id):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
        return {
            "id": item.id,
            "run_id": item.run_id,
            "char_no": item.char_no,
            "reviewer_decision": item.reviewer_decision,
            "override_classification": item.override_classification,
            "override_note": item.override_note,
        }
    finally:
        db.close()


def test_save_debug_verdict_creates_file_keyed_by_item_id(db_engine, tmp_path):
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 7,
                "status": "changed",
                "confidence": 0.61,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
        payload = validate_debug_verdict_payload(verdict="correct")
        verdicts = save_debug_verdict(run, item, payload)
    finally:
        db.close()

    raw = json.loads((out_dir / "debug_verdicts.json").read_text())
    assert list(raw.keys()) == [str(item_id)]
    assert raw[str(item_id)]["item_id"] == item_id
    assert raw[str(item_id)]["char_no"] == 7
    assert raw[str(item_id)]["verdict"] == "correct"
    assert item_id in verdicts


def test_save_debug_verdict_overwrites_existing_item_entry(db_engine, tmp_path):
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 3,
                "status": "uncertain",
                "confidence": 0.44,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
        save_debug_verdict(run, item, validate_debug_verdict_payload(verdict="correct"))
        save_debug_verdict(
            run,
            item,
            validate_debug_verdict_payload(
                verdict="incorrect",
                corrected_classification="changed",
                corrected_requirement_revA="A revised",
                corrected_requirement_revB="B revised",
                explanation="Pipeline classified the item incorrectly.",
            ),
        )
    finally:
        db.close()

    raw = json.loads((out_dir / "debug_verdicts.json").read_text())
    assert list(raw.keys()) == [str(item_id)]
    assert raw[str(item_id)]["verdict"] == "incorrect"
    assert raw[str(item_id)]["corrected_classification"] == "changed"
    assert raw[str(item_id)]["explanation"] == "Pipeline classified the item incorrectly."


def test_char_no_none_persists_inside_payload(db_engine, tmp_path):
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": None,
                "status": "added",
                "confidence": 0.9,
                "scores": {},
                "reasons": ["new item"],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
        save_debug_verdict(run, item, validate_debug_verdict_payload(verdict="correct"))
        loaded = load_debug_verdicts(run)
    finally:
        db.close()

    raw = json.loads((out_dir / "debug_verdicts.json").read_text())
    assert raw[str(item_id)]["char_no"] is None
    assert loaded[item_id]["char_no"] is None


def test_signed_off_debug_post_succeeds_without_mutating_review_state(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        status="signed_off",
        items=[
            {
                "char_no": 11,
                "status": "changed",
                "confidence": 0.77,
                "scores": {"location": 0.2},
                "reasons": ["text changed"],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    approve_resp = client.post(f"/review/{run_id}/items/11/approve")
    assert approve_resp.status_code == 409

    before = _get_item(db_engine, item_id)
    resp = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "partially_correct",
            "corrected_classification": "changed",
            "corrected_requirement_revA": "Original note",
            "corrected_requirement_revB": "Updated note",
            "explanation": "The model detected movement but missed the requirement delta.",
        },
    )
    assert resp.status_code == 200, resp.text

    after = _get_item(db_engine, item_id)
    assert after["reviewer_decision"] == before["reviewer_decision"] is None
    assert after["override_classification"] == before["override_classification"] is None
    assert after["override_note"] == before["override_note"] is None

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        assert run.status == "signed_off"
    finally:
        db.close()

    raw = json.loads((out_dir / "debug_verdicts.json").read_text())
    assert raw[str(item_id)]["verdict"] == "partially_correct"
    assert raw[str(item_id)]["char_no"] == 11


def test_non_admin_debug_post_is_rejected(
    client: TestClient, engineer_user, db_engine, tmp_path
):
    _login(client, db_engine, engineer_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 1,
                "status": "changed",
                "confidence": 0.5,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    resp = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={"verdict": "correct"},
    )
    assert resp.status_code == 403


def test_invalid_debug_post_does_not_overwrite_prior_saved_data(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 9,
                "status": "uncertain",
                "confidence": 0.33,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    ok = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={"verdict": "correct"},
    )
    assert ok.status_code == 200
    saved_before = json.loads((out_dir / "debug_verdicts.json").read_text())

    invalid = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "incorrect",
            "corrected_classification": "changed",
            "corrected_requirement_revA": "",
            "corrected_requirement_revB": "Updated req",
            "explanation": "Missing one corrected field should fail.",
        },
    )
    assert invalid.status_code == 422
    assert "required for non-correct verdicts" in invalid.text
    assert json.loads((out_dir / "debug_verdicts.json").read_text()) == saved_before


def test_debug_post_rejects_missing_and_unsupported_verdicts(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 5,
                "status": "changed",
                "confidence": 0.5,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    missing = client.post(f"/review/{run_id}/debug/items/{item_id}/verdict", data={})
    assert missing.status_code == 422
    assert "Verdict is required" in missing.text

    unsupported = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={"verdict": "maybe"},
    )
    assert unsupported.status_code == 422
    assert "Unsupported debug verdict" in unsupported.text


def test_debug_post_rejects_missing_run_item_and_cross_run_mismatch(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run1_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 1,
                "status": "changed",
                "confidence": 0.5,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    run2_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 2,
                "status": "removed",
                "confidence": 0.4,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item1_id = _open_queue(db_engine, run1_id)[0]
    item2_id = _open_queue(db_engine, run2_id)[0]

    missing_run = client.post("/review/999/debug/items/999/verdict", data={"verdict": "correct"})
    assert missing_run.status_code == 404

    missing_item = client.post(f"/review/{run1_id}/debug/items/999/verdict", data={"verdict": "correct"})
    assert missing_item.status_code == 404

    mismatch = client.post(f"/review/{run1_id}/debug/items/{item2_id}/verdict", data={"verdict": "correct"})
    assert mismatch.status_code == 404
    assert item1_id != item2_id
