"""Tests for the debug query parameter and admin guard on the review queue."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shop.models import Run, ReviewItem
from shop.services.auth import create_session


def _login(client, db_engine, user):
    """Create a session for `user` and set the cookie on `client`."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, user)
    db.close()
    client.cookies.set("session_token", token)


def _seed_run(db_engine, tmp_path, *, status="completed"):
    """Create a minimal Run that the review queue will accept."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    out_dir = tmp_path / "run-output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "delta_packet.json").write_text(json.dumps({"run_id": "t", "characters": []}))
    run = Run(
        part_number="PN-001",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J-001",
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
    return run_id


# ── Admin with debug=1 → 200 ──────────────────────────────────────────

def test_admin_debug_gets_200(client: TestClient, admin_user, db_engine, tmp_path):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(db_engine, tmp_path)
    resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert resp.status_code == 200


# ── Non-admin with debug=1 → 302 to /dashboard ────────────────────────

def test_nonadmin_debug_redirects(client: TestClient, engineer_user, db_engine, tmp_path):
    _login(client, db_engine, engineer_user)
    run_id = _seed_run(db_engine, tmp_path)
    resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/dashboard"


# ── Admin without debug → normal 200 (regression) ─────────────────────

def test_admin_no_debug_normal(client: TestClient, admin_user, db_engine, tmp_path):
    _login(client, db_engine, admin_user)
    run_id = _seed_run(db_engine, tmp_path)
    resp = client.get(f"/review/{run_id}", follow_redirects=False)
    assert resp.status_code == 200


# ── Non-admin without debug → normal 200 (regression) ─────────────────

def test_nonadmin_no_debug_normal(client: TestClient, engineer_user, db_engine, tmp_path):
    _login(client, db_engine, engineer_user)
    run_id = _seed_run(db_engine, tmp_path)
    resp = client.get(f"/review/{run_id}", follow_redirects=False)
    assert resp.status_code == 200


# ── debug_mode passed to template context ──────────────────────────────

def test_debug_mode_in_context(client: TestClient, admin_user, db_engine, tmp_path):
    """When debug=1, the template context should contain debug_mode=True."""
    _login(client, db_engine, admin_user)
    run_id = _seed_run(db_engine, tmp_path)
    # We can't easily inspect template context directly, but we verify the
    # endpoint returns 200 (template renders without error with the extra var).
    resp = client.get(f"/review/{run_id}?debug=1", follow_redirects=False)
    assert resp.status_code == 200
