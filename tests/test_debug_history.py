import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import AcceptedAlternateHistory, Run
from shop.services.auth import create_session
from shop.services.review import open_review_queue


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_FIXTURE = ROOT / "assets" / "part1" / "ground_truth.json"


def _login(client: TestClient, db_engine, user) -> None:
    session = sessionmaker(bind=db_engine)()
    try:
        token = create_session(session, user)
    finally:
        session.close()
    client.cookies.set("session_token", token)


def _seed_run(db_engine, tmp_path, *, part_number: str = "PN-HIST", status: str = "completed") -> tuple[int, Path]:
    out_dir = tmp_path / "history-run"
    out_dir.mkdir(parents=True, exist_ok=True)
    packet = {
        "run_id": "history-test",
        "items": [
            {
                "char_no": 42,
                "status": "changed",
                "confidence": 0.81,
                "requirement_revB": "REV B REQUIREMENT",
                "scores": {"location": 0.4, "text": 0.9},
                "reasons": ["requirement changed"],
                "revA": None,
                "revB": None,
                "evaluation": {
                    "status": "review_needed",
                    "matched_truth_char_no": "truth-42",
                    "snippet_conforms": False,
                    "mismatches": [
                        {"code": "classification_mismatch", "message": "classification differs"},
                        {"code": "requirement_mismatch", "message": "requirement differs"},
                    ],
                },
            }
        ],
    }
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    session = sessionmaker(bind=db_engine)()
    try:
        run = Run(
            part_number=part_number,
            rev_a_label="A",
            rev_b_label="B",
            customer="Test",
            job_number="J-HIST",
            status=status,
            output_dir=str(out_dir),
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/f.pdf",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id, out_dir
    finally:
        session.close()


def _open_queue(db_engine, run_id: int) -> int:
    session = sessionmaker(bind=db_engine)()
    try:
        run = session.query(Run).filter(Run.id == run_id).first()
        [item] = open_review_queue(session, run)
        return item.id
    finally:
        session.close()


def _fetch_history_rows(db_engine) -> list[AcceptedAlternateHistory]:
    session = sessionmaker(bind=db_engine)()
    try:
        return session.query(AcceptedAlternateHistory).order_by(AcceptedAlternateHistory.id).all()
    finally:
        session.close()


def test_acceptable_alternate_creates_active_history_record(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(db_engine, tmp_path)
    item_id = _open_queue(db_engine, run_id)

    response = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "acceptable_alternate",
            "explanation": "Reviewed alternate is acceptable for this part.",
        },
    )

    assert response.status_code == 200, response.text

    [history_row] = _fetch_history_rows(db_engine)
    assert history_row.run_id == run_id
    assert history_row.review_item_id == item_id
    assert history_row.part_number == "PN-HIST"
    assert history_row.char_no == 42
    assert history_row.matched_truth_char_no == "truth-42"
    assert history_row.reviewed_classification == "changed"
    assert history_row.reviewed_requirement_revB == "REV B REQUIREMENT"
    assert history_row.mismatch_codes == ["classification_mismatch", "requirement_mismatch"]
    assert history_row.rationale == "Reviewed alternate is acceptable for this part."
    assert history_row.is_active is True
    assert history_row.superseded_at is None


def test_history_record_is_deactivated_when_verdict_changes_away_from_acceptable_alternate(
    client: TestClient, admin_user, db_engine, tmp_path
):
    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(db_engine, tmp_path)
    item_id = _open_queue(db_engine, run_id)

    first = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "acceptable_alternate",
            "explanation": "Initial alternate approval.",
        },
    )
    assert first.status_code == 200, first.text

    second = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "algorithm_error",
            "corrected_classification": "changed",
            "explanation": "Later review decided the algorithm was wrong.",
        },
    )
    assert second.status_code == 200, second.text

    [history_row] = _fetch_history_rows(db_engine)
    assert history_row.run_id == run_id
    assert history_row.review_item_id == item_id
    assert history_row.is_active is False
    assert history_row.superseded_at is not None
    assert history_row.matched_truth_char_no == "truth-42"
    assert history_row.mismatch_codes == ["classification_mismatch", "requirement_mismatch"]


def test_saving_history_does_not_mutate_ground_truth_fixture(
    client: TestClient, admin_user, db_engine, tmp_path
):
    before_bytes = GROUND_TRUTH_FIXTURE.read_bytes()

    _login(client, db_engine, admin_user)
    run_id, _ = _seed_run(db_engine, tmp_path, part_number="PN-IMMUTABLE")
    item_id = _open_queue(db_engine, run_id)

    response = client.post(
        f"/review/{run_id}/debug/items/{item_id}/verdict",
        data={
            "verdict": "acceptable_alternate",
            "explanation": "History persistence must stay separate from truth fixtures.",
        },
    )

    assert response.status_code == 200, response.text
    assert GROUND_TRUTH_FIXTURE.read_bytes() == before_bytes

    [history_row] = _fetch_history_rows(db_engine)
    assert history_row.part_number == "PN-IMMUTABLE"
