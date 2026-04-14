import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from delta_preservation.evaluation.contracts import GroundTruthCharacteristic, GroundTruthPacket
from shop.models import ReviewItem, Run
from shop.services.auth import create_session
from shop.services.review import open_review_queue, save_debug_verdict, validate_debug_verdict_payload


def _login(client, db_engine, user):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, user)
    db.close()
    client.cookies.set("session_token", token)


def _seed_run(db_engine, tmp_path, *, items, status="completed"):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / f"run-status-debug-{status}-{len(items)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_packet.json").write_text(json.dumps({"run_id": "status-debug", "items": items}))
    run = Run(
        part_number="PN-STATUS",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="JOB-STATUS",
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


def _resolve_all_exception_rows(db_engine, run_id):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        items = open_review_queue(db, run)
        for item in items:
            save_debug_verdict(
                run,
                item,
                validate_debug_verdict_payload(
                    verdict="acceptable_alternate",
                    explanation=f"Accepted alternate for review item {item.id}.",
                ),
            )
    finally:
        db.close()


def _make_truth_packet(*classifications):
    chars = []
    for i, classification in enumerate(classifications):
        chars.append(
            GroundTruthCharacteristic(
                char_no=None if classification == "added" else i + 1,
                classification=classification,
                requirement_revB=None if classification == "removed" else f"req-{i}",
                snippet_center_revA=None if classification == "added" else (0.0, 0.0),
                snippet_center_revB=None if classification == "removed" else (0.0, 0.0),
            )
        )
    return GroundTruthPacket(part_name="status-debug", general_notes="", characteristics=chars)


def test_all_conforming_admin_run_stays_on_status_page(
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
                "requirement_revB": "Canonical row one",
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
                "status": "unchanged",
                "confidence": 0.98,
                "requirement_revB": "Canonical row two",
                "scores": {},
                "reasons": ["matched truth"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "conforming",
                    "matched_truth_char_no": 2,
                    "snippet_conforms": True,
                    "mismatches": [],
                },
            },
        ],
    )

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "Debug evaluation summary" in resp.text
    assert "All characteristics auto-passed" in resp.text
    assert "Conforming rows:" in resp.text
    assert "Download debug_report.json" in resp.text
    assert "Open Exception Queue" not in resp.text
    assert "Canonical matches" in resp.text


def test_mixed_admin_run_shows_exception_counts_and_queue_cta(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 3,
                "status": "unchanged",
                "confidence": 0.97,
                "requirement_revB": "Canonical row",
                "scores": {},
                "reasons": ["matched truth"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "conforming",
                    "matched_truth_char_no": 3,
                    "snippet_conforms": True,
                    "mismatches": [],
                },
            },
            {
                "char_no": 4,
                "status": "changed",
                "confidence": 0.44,
                "requirement_revB": "Exception row",
                "scores": {},
                "reasons": ["classification differs"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 4,
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "classification_mismatch", "message": "classification differs"}
                    ],
                },
            },
        ],
    )

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "Debug evaluation summary" in resp.text
    assert "Conforming rows:" in resp.text
    assert "Exception rows:" in resp.text
    assert "Resolved exceptions:" in resp.text
    assert "Unresolved exceptions:" in resp.text
    assert "Open Exception Queue" in resp.text
    assert 'href="/review/' in resp.text
    assert "Download debug_report.json" not in resp.text
    assert "Canonical matches" in resp.text


def test_resolved_exception_run_shows_download_link_on_status_page(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 5,
                "status": "removed",
                "confidence": 0.51,
                "requirement_revB": "Exception row",
                "scores": {},
                "reasons": ["missing revB evidence"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 5,
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "snippet_missing_revB", "message": "missing revB evidence"}
                    ],
                },
            }
        ],
    )
    _resolve_all_exception_rows(db_engine, run_id)

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert "Debug evaluation summary" in resp.text
    assert "Resolved exceptions:" in resp.text
    assert "Unresolved exceptions:" in resp.text
    assert "Open Exception Queue" in resp.text
    assert 'href="/review/' in resp.text
    assert "Download debug_report.json" in resp.text


def test_status_page_surfaces_missing_added_truth_blockers(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(
        db_engine,
        tmp_path,
        items=[
            {
                "char_no": 5,
                "status": "changed",
                "confidence": 0.44,
                "requirement_revB": "Exception row",
                "scores": {},
                "reasons": ["classification differs"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": 5,
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "classification_mismatch", "message": "classification differs"}
                    ],
                },
            }
        ],
    )
    _resolve_all_exception_rows(db_engine, run_id)

    with patch("shop.services.review.load_ground_truth_packet", return_value=_make_truth_packet("added")):
        resp = client.get(f"/runs/{run_id}")

    assert resp.status_code == 200
    assert "Ground truth includes 1 added characteristic" in resp.text
    assert "this run did not capture." in resp.text
    assert "Missing truth row index" in resp.text
    assert "0." in resp.text
    assert "Download debug_report.json" not in resp.text
