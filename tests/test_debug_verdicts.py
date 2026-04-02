import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import ReviewItem, Run
from shop.services.auth import create_session
from shop.services.review import (
    assemble_debug_report_payload,
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


def test_debug_queue_rehydrates_saved_verdict_and_progress_after_post_and_fresh_get(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 4,
                "status": "changed",
                "confidence": 0.73,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            },
            {
                "char_no": 8,
                "status": "removed",
                "confidence": 0.81,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            },
        ],
    )
    item_id, _ = _open_queue(db_engine, run_id)

    initial = client.get(f"/review/{run_id}?debug=1")
    assert initial.status_code == 200
    assert "Debug verdicts: 0 of 2 submitted" in initial.text
    assert f'/review/{run_id}/debug/items/{item_id}/verdict' in initial.text

    saved = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "incorrect",
            "corrected_classification": "changed",
            "corrected_requirement_revA": "A legacy note",
            "corrected_requirement_revB": "B corrected note",
            "explanation": "The model missed the textual requirement change.",
        },
    )
    assert saved.status_code == 200, saved.text
    assert "Debug verdicts: 1 of 2 submitted" in saved.text
    assert "Saved: incorrect" in saved.text

    fresh = client.get(f"/review/{run_id}?debug=1")
    assert fresh.status_code == 200
    assert "Debug verdicts: 1 of 2 submitted" in fresh.text
    assert 'option value="incorrect" selected' in fresh.text
    assert 'option value="changed" selected' in fresh.text
    assert "A legacy note" in fresh.text
    assert "B corrected note" in fresh.text
    assert "The model missed the textual requirement change." in fresh.text


def test_invalid_debug_submit_renders_inline_error_and_leaves_progress_unchanged(
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
    assert "Debug verdicts: 0 of 1 submitted" in invalid.text
    assert 'option value="incorrect" selected' in invalid.text
    assert "Updated req" in invalid.text
    assert not (out_dir / "debug_verdicts.json").exists()



def test_debug_queue_ignores_malformed_saved_entries_on_fresh_get(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, out_dir = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 2,
                "status": "changed",
                "confidence": 0.51,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    (out_dir / "debug_verdicts.json").write_text(
        json.dumps(
            {
                "bad-key": {"verdict": "correct"},
                str(item_id): {
                    "item_id": item_id,
                    "char_no": 2,
                    "verdict": "incorrect",
                    "corrected_classification": "changed",
                },
            }
        )
    )

    resp = client.get(f"/review/{run_id}?debug=1")
    assert resp.status_code == 200
    assert "Debug verdicts: 0 of 1 submitted" in resp.text
    assert "Saved: incorrect" not in resp.text
    assert f'/review/{run_id}/debug/items/{item_id}/verdict' in resp.text



def test_signed_off_debug_queue_keeps_verdict_form_editable_while_review_controls_stay_read_only(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        status="signed_off",
        items=[
            {
                "char_no": 12,
                "status": "changed",
                "confidence": 0.88,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    resp = client.get(f"/review/{run_id}?debug=1")
    assert resp.status_code == 200
    assert f'/review/{run_id}/debug/items/{item_id}/verdict' in resp.text
    assert "Save Debug Verdict" in resp.text
def test_debug_export_payload_preserves_queue_order_for_duplicate_and_none_char_numbers(
    db_engine, tmp_path
):
    run_id, _ = _seed_run(
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
                "revA": {"bbox": [0, 0, 10, 10], "image_path": None, "page": 0},
                "revB": {"bbox": [20, 20, 30, 30], "image_path": None, "page": 0},
            },
            {
                "char_no": 5,
                "status": "removed",
                "confidence": 0.42,
                "requirement_revB": "Packet duplicate second",
                "scores": {"location": 0.1},
                "reasons": ["second duplicate"],
                "revA": {"bbox": [100, 100, 120, 120], "image_path": None, "page": 0},
                "revB": None,
            },
            {
                "char_no": None,
                "status": "added",
                "confidence": 0.88,
                "requirement_revB": "Packet added row",
                "scores": {"text": 0.8},
                "reasons": ["none char"],
                "revA": None,
                "revB": {"bbox": [200, 200, 240, 260], "image_path": None, "page": 0},
            },
        ],
    )
    item_ids = _open_queue(db_engine, run_id)

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        items = [db.query(ReviewItem).filter(ReviewItem.id == item_id).first() for item_id in item_ids]
        save_debug_verdict(run, items[0], validate_debug_verdict_payload(verdict="correct"))
        save_debug_verdict(
            run,
            items[1],
            validate_debug_verdict_payload(
                verdict="incorrect",
                corrected_classification="removed",
                corrected_requirement_revA="Old duplicate",
                corrected_requirement_revB="New duplicate",
                explanation="Second duplicate stayed distinct.",
            ),
        )
        save_debug_verdict(
            run,
            items[2],
            validate_debug_verdict_payload(
                verdict="partially_correct",
                corrected_classification="added",
                corrected_requirement_revA="No Rev A requirement for added row",
                corrected_requirement_revB="Added requirement",
                explanation="Added row has no Rev A requirement.",
            ),
        )
        payload = assemble_debug_report_payload(db, run)
    finally:
        db.close()

    assert [row["review_item_id"] for row in payload["items"]] == item_ids
    assert [row["packet_item"].get("requirement_revB") for row in payload["items"]] == [
        "Packet duplicate first",
        "Packet duplicate second",
        "Packet added row",
    ]
    assert payload["items"][0]["requirement_revB"] == "Packet duplicate first"
    assert payload["items"][1]["requirement_revB"] == "Packet duplicate second"
    assert payload["items"][2]["char_no"] is None
    assert payload["items"][2]["revA_center"] is None
    assert payload["items"][2]["revB_center"] == [220.0, 230.0] or payload["items"][2]["revB_center"] == (220.0, 230.0)



def test_debug_export_route_rejects_incomplete_coverage(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 1,
                "status": "changed",
                "confidence": 0.71,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            },
            {
                "char_no": 2,
                "status": "removed",
                "confidence": 0.33,
                "scores": {},
                "reasons": [],
                "revA": None,
                "revB": None,
            },
        ],
    )
    item_id, _ = _open_queue(db_engine, run_id)

    ok = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={"verdict": "correct"},
    )
    assert ok.status_code == 200

    export = client.get(f"/review/{run_id}/debug-report.json")
    assert export.status_code == 400
    assert export.json()["detail"].startswith("Debug export is incomplete")



def test_debug_export_route_returns_attachment_with_full_payload(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 7,
                "status": "changed",
                "confidence": 0.77,
                "requirement_revB": "Rev B packet req",
                "scores": {"location": 0.2, "text": 0.8},
                "reasons": ["text changed"],
                "semantic_callout": {
                    "raw_text": "PROFILE 0.2",
                    "status": {"state": "parsed", "parser_family": "gdt", "reason_code": None, "detail": None},
                    "provenance": {"authority": "pdf", "source_type": "drawing_pdf", "source_ref": "page:1/span:7", "notes": []},
                    "gdt": {"frame_text": "⌖ ⌀0.2", "control_type": "position", "tolerance_text": "⌀0.2", "datum_refs": ["A"], "modifiers": []},
                },
                "revA": {"bbox": [10, 20, 30, 40], "image_path": None, "page": 0},
                "revB": {"bbox": [50, 60, 70, 80], "image_path": None, "page": 0},
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    saved = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "incorrect",
            "corrected_classification": "changed",
            "corrected_requirement_revA": "Old requirement",
            "corrected_requirement_revB": "Corrected requirement",
            "explanation": "Pipeline missed the semantic delta.",
        },
    )
    assert saved.status_code == 200

    export = client.get(f"/review/{run_id}/debug-report.json")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/json")
    assert export.headers["content-disposition"] == 'attachment; filename="debug_report.json"'

    payload = export.json()
    assert payload["run_id"] == run_id
    assert payload["run_status"] == "reviewing"
    assert payload["debug_total"] == 1
    assert payload["debug_submitted"] == 1
    row = payload["items"][0]
    assert row["review_item_id"] == item_id
    assert row["char_no"] == 7
    assert row["debug_verdict"] == "incorrect"
    assert row["corrected_classification"] == "changed"
    assert row["corrected_requirement_revA"] == "Old requirement"
    assert row["corrected_requirement_revB"] == "Corrected requirement"
    assert row["explanation"] == "Pipeline missed the semantic delta."
    assert row["scores"] == {"location": 0.2, "text": 0.8}
    assert row["reasons"] == ["text changed"]
    assert row["semantic_callout"]["raw_text"] == "PROFILE 0.2"
    assert row["packet_item"]["requirement_revB"] == "Rev B packet req"
    assert row["revA_center"] == [20.0, 30.0] or row["revA_center"] == (20.0, 30.0)
    assert row["revB_center"] == [60.0, 70.0] or row["revB_center"] == (60.0, 70.0)



def test_signed_off_debug_export_route_still_allows_completed_attachment(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(
        db_engine,
        tmp_path,
        status="signed_off",
        items=[
            {
                "char_no": 11,
                "status": "changed",
                "confidence": 0.66,
                "scores": {"location": 0.4},
                "reasons": ["signed off row"],
                "revA": None,
                "revB": None,
            }
        ],
    )
    item_id = _open_queue(db_engine, run_id)[0]

    saved = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={"verdict": "correct"},
    )
    assert saved.status_code == 200

    export = client.get(f"/review/{run_id}/debug-report.json")
    assert export.status_code == 200
    payload = export.json()
    assert payload["run_status"] == "signed_off"
    assert payload["items"][0]["debug_verdict"] == "correct"



def test_non_admin_debug_export_route_is_rejected(
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
    _open_queue(db_engine, run_id)

    resp = client.get(f"/review/{run_id}/debug-report.json")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Admin only"
