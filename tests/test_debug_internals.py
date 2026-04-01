"""Tests for debug_internals_by_char service function and router wiring."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import Run
from shop.services.auth import create_session
from shop.services.review import debug_internals_by_char


# ── Helpers ────────────────────────────────────────────────────────────

def _make_run(db_engine, tmp_path, items):
    """Create a Run with a delta_packet containing the given items list."""
    out_dir = tmp_path / "dbg-run"
    out_dir.mkdir(exist_ok=True)
    packet = {"run_id": "dbg-test", "items": items}
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    Session = sessionmaker(bind=db_engine)
    db = Session()
    run = Run(
        part_number="PN-DBG",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-DBG",
        status="completed",
        output_dir=str(out_dir),
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.pdf",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    result = run  # keep attached
    return db, result


def _login(client, db_engine, user):
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, user)
    db.close()
    client.cookies.set("session_token", token)


# ── Unit: correct keys and values ──────────────────────────────────────

def test_debug_internals_returns_scores_and_centers(db_engine, tmp_path):
    items = [
        {
            "char_no": 1,
            "status": "changed",
            "confidence": 0.8,
            "scores": {"location": 0.9, "text": 0.7, "context": 0.5},
            "reasons": ["text differs", "location shifted"],
            "revA": {"bbox": [10, 20, 30, 40], "image_path": None},
            "revB": {"bbox": [100, 200, 300, 400], "image_path": None},
        }
    ]
    db, run = _make_run(db_engine, tmp_path, items)
    result = debug_internals_by_char(run)
    db.close()

    assert 1 in result
    entry = result[1]
    assert entry["scores"] == {"location": 0.9, "text": 0.7, "context": 0.5}
    assert entry["reasons"] == ["text differs", "location shifted"]
    assert entry["revA_center"] == (20.0, 30.0)
    assert entry["revB_center"] == (200.0, 300.0)


# ── Unit: None bbox → None center ──────────────────────────────────────

def test_debug_internals_none_bbox(db_engine, tmp_path):
    items = [
        {
            "char_no": None,
            "status": "added",
            "confidence": 0.5,
            "scores": {},
            "reasons": ["new item"],
            "revA": None,
            "revB": {"bbox": None, "image_path": None},
        }
    ]
    db, run = _make_run(db_engine, tmp_path, items)
    result = debug_internals_by_char(run)
    db.close()

    assert None in result
    entry = result[None]
    assert entry["revA_center"] is None
    assert entry["revB_center"] is None
    assert entry["reasons"] == ["new item"]


# ── Integration: admin GET /review/{id}?debug=1 returns 200 ───────────

def test_admin_debug_review_returns_200(client: TestClient, admin_user, db_engine, tmp_path):
    items = [
        {
            "char_no": 5,
            "status": "unchanged",
            "confidence": 0.95,
            "scores": {"location": 1.0},
            "reasons": [],
            "revA": {"bbox": [0, 0, 10, 10], "image_path": None, "page": 1},
            "revB": {"bbox": [0, 0, 10, 10], "image_path": None, "page": 1},
        }
    ]
    # Seed directly via helper
    out_dir = tmp_path / "int-run"
    out_dir.mkdir(exist_ok=True)
    packet = {"run_id": "int-test", "items": items}
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    Session = sessionmaker(bind=db_engine)
    db = Session()
    run = Run(
        part_number="PN-INT",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-INT",
        status="completed",
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

    _login(client, db_engine, admin_user)
    resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert resp.status_code == 200
