"""
Test stubs for Phase 3 Review and Sign-Off requirements.

All 10 tests are marked xfail(strict=False) so they appear in collection output
and the xfail count is visible throughout the phase while individual tests are
implemented in Plans 02-05.  Each stub raises NotImplementedError so it is
genuinely red until the corresponding plan implements the feature.

Requirements covered:
  REVIEW-01..07  — Review queue and item management
  SIGNOFF-01..03 — Sign-off gate, rollback, and immutability
"""

import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# REVIEW-01: Review queue loads
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-01: not yet implemented — Plan 02")
def test_review_queue_loads(client: TestClient, engineer_user):
    """GET /review/{run_id} returns 200 with all ReviewItems listed."""
    raise NotImplementedError("REVIEW-01")


# ---------------------------------------------------------------------------
# REVIEW-02: Review item card HTML
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-02: not yet implemented — Plan 02")
def test_review_item_card_html(client: TestClient, engineer_user):
    """Card HTML contains snippet img, char_no, classification, approve/override controls."""
    raise NotImplementedError("REVIEW-02")


# ---------------------------------------------------------------------------
# REVIEW-03: Approve item
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-03: not yet implemented — Plan 03")
def test_approve_item(client: TestClient, engineer_user, db_engine):
    """POST approve saves reviewer_decision='approved' on the ReviewItem."""
    raise NotImplementedError("REVIEW-03")


# ---------------------------------------------------------------------------
# REVIEW-04: Override requires note
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-04: not yet implemented — Plan 03")
def test_override_requires_note(client: TestClient, engineer_user, db_engine):
    """POST override with empty note returns 422; non-empty note saves successfully."""
    raise NotImplementedError("REVIEW-04")


# ---------------------------------------------------------------------------
# REVIEW-05: Review state persisted
# ---------------------------------------------------------------------------

def test_review_state_persisted(tmp_path, db_engine):
    """open_review_queue is idempotent: second call returns same rows without duplicating."""
    from shop.services.review import open_review_queue
    from shop.models import Run, ReviewItem

    # Build a minimal delta_packet.json with 3 DeltaItems
    packet = {
        "run_id": "test-run-001",
        "inputs": {},
        "items": [
            {"char_no": 1, "status": "unchanged", "confidence": 0.95, "reasons": [], "scores": {}, "revA": None, "revB": None},
            {"char_no": 2, "status": "changed",   "confidence": 0.80, "reasons": [], "scores": {}, "revA": None, "revB": None},
            {"char_no": 3, "status": "removed",   "confidence": 0.70, "reasons": [], "scores": {}, "revA": None, "revB": None},
        ],
    }
    out_dir = tmp_path / "out" / "test-run-001"
    out_dir.mkdir(parents=True)
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    # Set up a DB session using the test engine
    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()

    try:
        # Create a Run pointing at our tmp output dir
        run = Run(
            part_number="P-001",
            rev_a_label="A",
            rev_b_label="B",
            customer="ACME",
            job_number="JOB-001",
            status="completed",
            output_dir=str(out_dir),
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        # First call — should create 3 ReviewItems
        items1 = open_review_queue(db, run)
        assert len(items1) == 3

        # Second call — idempotent: must not duplicate
        items2 = open_review_queue(db, run)
        count = db.query(ReviewItem).filter(ReviewItem.run_id == run.id).count()
        assert count == 3, f"Expected 3 ReviewItems, got {count}"

        # Run.status must be "reviewing" after first open
        db.refresh(run)
        assert run.status == "reviewing"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# REVIEW-06: Admin can reassign reviewer
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-06: not yet implemented — Plan 04")
def test_admin_can_reassign(client: TestClient, admin_user, engineer_user, db_engine):
    """Admin can reassign reviewer on a run; engineer role cannot reassign."""
    raise NotImplementedError("REVIEW-06")


# ---------------------------------------------------------------------------
# REVIEW-07: Review counts
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="REVIEW-07: not yet implemented — Plan 02")
def test_review_counts(client: TestClient, engineer_user, db_engine):
    """Queue response contains pending/approved/overridden counts."""
    raise NotImplementedError("REVIEW-07")


# ---------------------------------------------------------------------------
# SIGNOFF-01: Sign-off gate
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="SIGNOFF-01: not yet implemented — Plan 05")
def test_sign_off_gate(client: TestClient, engineer_user, db_engine):
    """Sign-off button disabled when pending > 0; enabled only when all items resolved."""
    raise NotImplementedError("SIGNOFF-01")


# ---------------------------------------------------------------------------
# SIGNOFF-02: Sign-off rollback
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="SIGNOFF-02: not yet implemented — Plan 05")
def test_sign_off_rollback(client: TestClient, admin_user, db_engine):
    """If sign-off packet write fails, Run.status rolls back to 'reviewing'."""
    raise NotImplementedError("SIGNOFF-02")


# ---------------------------------------------------------------------------
# SIGNOFF-03: Signed-off immutable
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="SIGNOFF-03: not yet implemented — Plan 05")
def test_signed_off_immutable(client: TestClient, admin_user, db_engine):
    """Run.signed_at is immutable after sign-off; second attempt is blocked."""
    raise NotImplementedError("SIGNOFF-03")
