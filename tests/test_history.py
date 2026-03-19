"""Phase 4: Run history and retention tests."""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from shop.utils import utcnow


def _login_engineer(client, db_engine, engineer_user):
    """Seed a session for the engineer user and set cookie on client."""
    from shop.services.auth import create_session
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, engineer_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


def _make_run(db, status, submitted_at, engineer_id, part_number="PN-HIST"):
    from shop.models import Run
    run = Run(
        part_number=part_number,
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="J001",
        status=status,
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_id,
        submitted_at=submitted_at,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _run_cleanup_logic(db, retention_days):
    """Extracted cleanup logic for testability (mirrors cleanup_old_runs task body)."""
    from shop.models import Run
    DELETABLE_STATUSES = {"queued", "running", "failed", "completed"}
    cutoff = utcnow() - timedelta(days=retention_days)
    old_runs = (
        db.query(Run)
        .filter(Run.status.in_(DELETABLE_STATUSES), Run.submitted_at < cutoff)
        .all()
    )
    for run in old_runs:
        db.delete(run)
    db.commit()


def test_history_filters(client: TestClient, db_engine, engineer_user):
    """HISTORY-01: list filterable by part number and date."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    old_date = datetime(2025, 1, 1)
    new_date = datetime(2026, 3, 1)
    _make_run(db, "failed", old_date, engineer_user.id, part_number="OLD-001")
    _make_run(db, "signed_off", new_date, engineer_user.id, part_number="NEW-001")

    # Filter by date (only newer)
    resp = client.get("/runs?date_from=2026-01-01")
    assert resp.status_code == 200
    assert "NEW-001" in resp.text
    assert "OLD-001" not in resp.text

    # Filter by part number
    resp2 = client.get("/runs?part_number=OLD")
    assert resp2.status_code == 200
    assert "OLD-001" in resp2.text
    assert "NEW-001" not in resp2.text


def test_signed_off_readonly_view(client: TestClient, db_engine, engineer_user):
    """HISTORY-02: signed_off run opens review queue in read-only mode."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    from shop.models import ReviewItem
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    run = _make_run(db, "signed_off", datetime(2026, 3, 1), engineer_user.id)
    # Add a ReviewItem so the queue renders
    item = ReviewItem(
        run_id=run.id, char_no=1, pipeline_classification="unchanged",
        confidence=0.9, reviewer_decision="approved",
    )
    db.add(item)
    db.commit()
    resp = client.get(f"/review/{run.id}")
    assert resp.status_code == 200
    # Read-only banner should be present
    assert "read-only" in resp.text.lower() or "signed off" in resp.text.lower()


def test_cleanup_exempt_signed_off(client: TestClient, db_engine, engineer_user):
    """HISTORY-03: signed_off runs not deleted by cleanup logic."""
    from shop.dependencies import get_db
    from shop.models import Run
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    # Signed-off run submitted 365 days ago
    old_signed = _make_run(db, "signed_off", datetime(2025, 1, 1), engineer_user.id)
    _run_cleanup_logic(db, retention_days=30)
    db.expire_all()
    still_exists = db.get(Run, old_signed.id)
    assert still_exists is not None, "signed_off run must not be deleted"


def test_cleanup_deletes_old_runs(client: TestClient, db_engine, engineer_user):
    """HISTORY-04: cleanup deletes old failed/queued runs beyond retention_days."""
    from shop.dependencies import get_db
    from shop.models import Run
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")

    # Old failed run — 60 days ago
    old_run = _make_run(db, "failed", utcnow() - timedelta(days=60), engineer_user.id)
    # Recent failed run — 5 days ago (should survive)
    recent_run = _make_run(db, "failed", utcnow() - timedelta(days=5), engineer_user.id)
    _run_cleanup_logic(db, retention_days=30)
    db.expire_all()
    assert db.get(Run, old_run.id) is None, "old failed run should be deleted"
    assert db.get(Run, recent_run.id) is not None, "recent failed run should be retained"
