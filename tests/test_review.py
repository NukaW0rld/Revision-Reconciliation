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

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


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

@pytest.mark.xfail(strict=False, reason="REVIEW-05: not yet implemented — Plan 03")
def test_review_state_persisted(client: TestClient, engineer_user, db_engine):
    """ReviewItem rows survive across requests; second GET returns same decision state."""
    raise NotImplementedError("REVIEW-05")


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
