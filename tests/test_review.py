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

from shop.services.review import semantic_contracts_by_char


def _login_engineer(client, db_engine, engineer_user):
    """Seed a session for the engineer user and set cookie on client."""
    from shop.services.auth import create_session
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, engineer_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


def _make_run_with_packet(tmp_path, db_engine, n_items=3, *, status="completed"):
    """Helper: create a Run in the DB with a delta_packet.json in tmp_path."""
    from shop.models import Run
    out_dir = tmp_path / "out" / "test-run"
    out_dir.mkdir(parents=True)
    items = [
        {
            "char_no": i + 1,
            "status": ["unchanged", "changed", "uncertain"][i % 3],
            "confidence": 0.9 - i * 0.1,
            "reasons": [],
            "scores": {},
            "revA": None,
            "revB": None,
        }
        for i in range(n_items)
    ]
    packet = {"run_id": "test-run-001", "inputs": {}, "items": items}
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        run = Run(
            part_number="P-001",
            rev_a_label="A",
            rev_b_label="B",
            customer="ACME",
            job_number="JOB-001",
            status=status,
            output_dir=str(out_dir),
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
    finally:
        db.close()
    return run_id


def test_review_semantic_contract_shapes_parsed_and_fallback_states(tmp_path, db_engine):
    """semantic_contract: review service exposes parsed and fallback semantic summaries."""
    from shop.models import Run

    packet = {
        "run_id": "semantic-review-001",
        "inputs": {},
        "items": [
            {
                "char_no": 10,
                "status": "changed",
                "confidence": 0.84,
                "reasons": [
                    "semantic GD&T changed: tolerance ⌀0.10 → ⌀0.20",
                    "meaningful drawing requirement change",
                ],
                "scores": {"location": 0.88, "text": 0.70, "context": 0.81},
                "revA": None,
                "revB": None,
                "semantic_callout": {
                    "provenance": {
                        "authority": "pdf",
                        "source_type": "drawing_pdf",
                        "source_ref": "page:1/span:42",
                        "notes": ["pdf span selected"],
                    },
                    "status": {
                        "state": "parsed",
                        "parser_family": "gdt",
                        "reason_code": None,
                        "detail": "parsed feature control frame from PDF spans",
                    },
                    "raw_text": "⌖ ⌀0.20 M A B C",
                    "normalized_text": "⌖ ⌀0.20 M A B C",
                    "gdt": {
                        "frame_text": "⌖ | ⌀0.20 | M | A | B | C",
                        "control_type": "position",
                        "tolerance_text": "⌀0.20",
                        "datum_refs": ["A", "B", "C"],
                        "modifiers": ["MMC"],
                    },
                    "metadata": {"authority_source": "pdf"},
                },
            },
            {
                "char_no": 11,
                "status": "unchanged",
                "confidence": 0.61,
                "reasons": [
                    "semantic comparison fallback: left semantic state error/weld_malformed",
                    "Primary dimension matches",
                ],
                "scores": {"location": 0.65, "text": 0.60, "context": 0.58},
                "revA": None,
                "revB": None,
                "semantic_callout": {
                    "provenance": {
                        "authority": "pdf",
                        "source_type": "drawing_pdf",
                        "source_ref": "page:2/span:8",
                        "notes": ["pdf span selected"],
                    },
                    "status": {
                        "state": "error",
                        "parser_family": "weld",
                        "reason_code": "weld_malformed",
                        "detail": "recognized weld callout is missing a parseable size token before the weld type",
                    },
                    "raw_text": "FILLET BOTH SIDES",
                    "normalized_text": "FILLET BOTH SIDES",
                    "metadata": {"authority_source": "pdf"},
                },
            },
            {
                "char_no": 12,
                "status": "unchanged",
                "confidence": 0.97,
                "reasons": ["Location and text matched"],
                "scores": {"location": 0.98, "text": 0.96, "context": 0.94},
                "revA": None,
                "revB": None,
            },
        ],
    }
    out_dir = tmp_path / "out" / "semantic-review-001"
    out_dir.mkdir(parents=True)
    (out_dir / "delta_packet.json").write_text(json.dumps(packet))

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        run = Run(
            part_number="P-SEM",
            rev_a_label="A",
            rev_b_label="B",
            customer="ACME",
            job_number="JOB-SEM",
            status="completed",
            output_dir=str(out_dir),
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        contracts = semantic_contracts_by_char(run)
        parsed = contracts[10]
        fallback = contracts[11]

        assert parsed["family"] == "gdt"
        assert parsed["family_label"] == "GD&T"
        assert parsed["status"] == "parsed"
        assert parsed["summary"] == "position ⌀0.20 datums A, B, C modifiers MMC"
        assert parsed["block_label"] == "GD&T parsed"
        assert parsed["reason_fragments"] == ["semantic GD&T changed: tolerance ⌀0.10 → ⌀0.20"]
        assert parsed["is_parsed"] is True
        assert parsed["is_fallback"] is False

        assert fallback["family"] == "weld"
        assert fallback["status"] == "error"
        assert fallback["reason_code"] == "weld_malformed"
        assert fallback["summary"] == "FILLET BOTH SIDES"
        assert fallback["block_label"] == "Weld error"
        assert fallback["reason_fragments"] == [
            "semantic comparison fallback: left semantic state error/weld_malformed"
        ]
        assert fallback["is_parsed"] is False
        assert fallback["is_fallback"] is True

        assert contracts[12] is None
    finally:
        db.close()


def test_review_queue_loads(client: TestClient, engineer_user, db_engine, tmp_path):
    """GET /review/{run_id} returns 200 with all ReviewItems listed."""
    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=3, status="completed")

    resp = client.get(f"/review/{run_id}", follow_redirects=False)
    assert resp.status_code == 200, resp.text

    # Each char_no should appear in the response
    for char_no in [1, 2, 3]:
        assert str(char_no) in resp.text, f"char_no {char_no} not found in response"

    # Item cards must NOT have display:none — they should be visible in the flat list
    assert 'style="display:none"' not in resp.text, (
        "Item cards have display:none — review queue items are invisible to the user"
    )

    # Run.status should now be "reviewing"
    from shop.models import Run
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        assert run.status == "reviewing"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# REVIEW-02: Review item card HTML
# ---------------------------------------------------------------------------

def test_review_item_card_html(client: TestClient, engineer_user, db_engine, tmp_path):
    """Card HTML contains char_no, classification, approve/override controls; resolved state shows badge not buttons."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    # Open the review queue to seed ReviewItems
    client.get(f"/review/{run_id}", follow_redirects=False)

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        char_no = item.char_no
    finally:
        db.close()

    # Approve item via POST — should return resolved card HTML
    resp = client.post(f"/review/{run_id}/items/{char_no}/approve")
    assert resp.status_code == 200, resp.text

    # Resolved state: contains "Approved" badge
    assert "Approved" in resp.text
    # Resolved state: does NOT contain "Approve" button (action removed)
    assert 'hx-post' not in resp.text or "approve" not in resp.text.split('hx-post')[1].split('"')[1] if 'hx-post' in resp.text else True

    # OOB progress bar included
    assert "progress-bar" in resp.text
    assert "hx-swap-oob" in resp.text

    # border-success class present for resolved card
    assert "border-success" in resp.text


# ---------------------------------------------------------------------------
# REVIEW-03: Approve item
# ---------------------------------------------------------------------------

def test_approve_item(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST approve saves reviewer_decision='approved' on the ReviewItem."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import Run, ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    # Open the review queue to seed ReviewItems
    client.get(f"/review/{run_id}", follow_redirects=False)

    # Find the char_no for the created item
    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        assert item is not None, "ReviewItem not seeded"
        char_no = item.char_no
    finally:
        db.close()

    resp = client.post(f"/review/{run_id}/items/{char_no}/approve")
    assert resp.status_code == 200, resp.text

    # Verify DB state
    db = TestingSession()
    try:
        db_item = db.query(ReviewItem).filter(
            ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
        ).first()
        assert db_item.reviewer_decision == "approved"
        assert db_item.reviewed_by_id == engineer_user.id
        assert db_item.reviewed_at is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# REVIEW-04: Override requires note
# ---------------------------------------------------------------------------

def test_override_requires_note(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST override with empty note returns 422; non-empty note saves successfully."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import Run, ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    # Open the review queue to seed ReviewItems
    client.get(f"/review/{run_id}", follow_redirects=False)

    # Find the char_no for the created item
    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        assert item is not None, "ReviewItem not seeded"
        char_no = item.char_no
    finally:
        db.close()

    # Empty note must return 422
    resp = client.post(f"/review/{run_id}/items/{char_no}/override",
                       data={"override_note": "", "override_classification": "changed"})
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    # Non-empty note must succeed
    resp = client.post(f"/review/{run_id}/items/{char_no}/override",
                       data={"override_note": "Tolerance band is acceptable", "override_classification": "changed"})
    assert resp.status_code == 200, resp.text

    # Verify DB state
    db = TestingSession()
    try:
        db_item = db.query(ReviewItem).filter(
            ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
        ).first()
        assert db_item.reviewer_decision == "overridden"
        assert db_item.override_note == "Tolerance band is acceptable"
        assert db_item.override_classification == "changed"
        assert db_item.reviewed_by_id == engineer_user.id
    finally:
        db.close()


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

def _login_admin(client, db_engine, admin_user):
    """Seed a session for the admin user and set cookie on client."""
    from shop.services.auth import create_session
    AdminSession = sessionmaker(bind=db_engine)
    db = AdminSession()
    token = create_session(db, admin_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


def test_admin_can_reassign(client: TestClient, admin_user, engineer_user, db_engine, tmp_path):
    """Admin can reassign reviewer on a run; engineer role cannot reassign."""
    from shop.models import Run

    # Create a run assigned to engineer
    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        run = Run(
            part_number="P-ASSIGN",
            rev_a_label="A",
            rev_b_label="B",
            customer="ACME",
            job_number="JOB-ASSIGN",
            status="reviewing",
            output_dir=str(tmp_path / "out" / "assign-run"),
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
            reviewer_id=engineer_user.id,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
    finally:
        db.close()

    # Admin can reassign — log in as admin
    _login_admin(client, db_engine, admin_user)
    resp = client.post(
        f"/review/{run_id}/reassign",
        data={"reviewer_id": admin_user.id},
        follow_redirects=False,
    )
    assert resp.status_code == 302, f"Expected 302, got {resp.status_code}: {resp.text}"
    assert f"/runs/{run_id}" in resp.headers["location"]

    # Verify DB updated
    db = TestingSession()
    try:
        updated_run = db.query(Run).filter(Run.id == run_id).first()
        assert updated_run.reviewer_id == admin_user.id
    finally:
        db.close()

    # Engineer cannot reassign — log in as engineer
    _login_engineer(client, db_engine, engineer_user)
    resp = client.post(
        f"/review/{run_id}/reassign",
        data={"reviewer_id": engineer_user.id},
        follow_redirects=False,
    )
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}"


# ---------------------------------------------------------------------------
# REVIEW-07: Review counts
# ---------------------------------------------------------------------------

def test_review_counts(client: TestClient, engineer_user, db_engine, tmp_path):
    """Queue response contains pending/approved/overridden counts."""
    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=3, status="completed")

    resp = client.get(f"/review/{run_id}", follow_redirects=False)
    assert resp.status_code == 200, resp.text

    # The progress bar should show counts. All 3 are pending (no decisions yet).
    assert "3 pending" in resp.text
    assert "0 approved" in resp.text
    assert "0 overridden" in resp.text
    assert "/ 3 total" in resp.text


# ---------------------------------------------------------------------------
# SIGNOFF-01: Sign-off gate
# ---------------------------------------------------------------------------

def test_sign_off_gate(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST /sign-off/confirm redirects to error when pending > 0; redirects to generating when all resolved."""
    from shop.models import ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    # Seed the review queue
    client.get(f"/review/{run_id}", follow_redirects=False)

    # POST sign-off/confirm with 1 pending item — must redirect to ?error=pending_items
    resp = client.post(f"/review/{run_id}/sign-off/confirm", follow_redirects=False)
    assert resp.status_code == 302, f"Expected 302, got {resp.status_code}"
    assert "error=pending_items" in resp.headers["location"], (
        f"Expected redirect to ?error=pending_items, got: {resp.headers['location']}"
    )

    # Approve the item
    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        char_no = item.char_no
    finally:
        db.close()

    client.post(f"/review/{run_id}/items/{char_no}/approve")

    # POST sign-off/confirm with 0 pending — must redirect to /review/{id}/generating
    resp = client.post(f"/review/{run_id}/sign-off/confirm", follow_redirects=False)
    assert resp.status_code == 302, f"Expected 302, got {resp.status_code}"
    location = resp.headers["location"]
    assert f"/review/{run_id}/generating" in location, (
        f"Expected redirect to /review/{run_id}/generating, got: {location}"
    )


# ---------------------------------------------------------------------------
# SIGNOFF-02: Sign-off rollback
# ---------------------------------------------------------------------------

def test_sign_off_rollback(db_engine):
    """SIGNOFF-02: if sign-off fails midway, Run.status rolls back to reviewing."""
    from unittest.mock import patch
    from shop.services.review import attempt_sign_off
    from shop.models import Run
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    db = Session()
    run = Run(part_number="P1", rev_a_label="A", rev_b_label="B",
              customer="C", job_number="J1", status="reviewing",
              revA_path="", revB_path="", form3_path="")
    db.add(run)
    db.commit()
    db.refresh(run)

    # Simulate PDF generation failure (Phase 4 integration point)
    with patch(
        "shop.services.exports.generate_and_store_audit_packet",
        side_effect=RuntimeError("Simulated packet write failure"),
    ):
        result = attempt_sign_off(db, run, reviewer_id=1)

    assert result is False
    # Re-query to confirm status
    db.expire_all()
    fresh = db.query(Run).filter(Run.id == run.id).first()
    assert fresh.status == "reviewing"
    assert fresh.signed_at is None
    db.close()


# ---------------------------------------------------------------------------
# SIGNOFF-03: Signed-off immutable
# ---------------------------------------------------------------------------

def test_signed_off_immutable(db_engine):
    """SIGNOFF-03: signed_off run cannot be re-signed; signed_at is immutable."""
    from unittest.mock import patch
    from shop.services.review import attempt_sign_off
    from shop.models import Run
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    db = Session()
    run = Run(part_number="P1", rev_a_label="A", rev_b_label="B",
              customer="C", job_number="J1", status="reviewing",
              revA_path="", revB_path="", form3_path="")
    db.add(run)
    db.commit()
    db.refresh(run)

    # First sign-off succeeds (mock PDF generation so no real output_dir needed)
    with patch("shop.services.exports.generate_and_store_audit_packet"):
        result1 = attempt_sign_off(db, run, reviewer_id=1)
    assert result1 is True
    db.refresh(run)
    first_signed_at = run.signed_at
    assert run.status == "signed_off"
    assert first_signed_at is not None

    # Second sign-off attempt returns False; signed_at unchanged
    result2 = attempt_sign_off(db, run, reviewer_id=2)
    assert result2 is False
    db.refresh(run)
    assert run.signed_at == first_signed_at  # immutable
    assert run.status == "signed_off"
    db.close()


# ---------------------------------------------------------------------------
# RESET-01: Reset approved item
# ---------------------------------------------------------------------------

def test_reset_approved_item(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST reset on an approved item clears reviewer_decision and restores pending state."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    # Seed ReviewItems
    client.get(f"/review/{run_id}", follow_redirects=False)

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        char_no = item.char_no
    finally:
        db.close()

    # Approve the item
    resp = client.post(f"/review/{run_id}/items/{char_no}/approve")
    assert resp.status_code == 200

    # Confirm approved in DB
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(
            ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
        ).first()
        assert item.reviewer_decision == "approved"
    finally:
        db.close()

    # Reset the item
    resp = client.post(f"/review/{run_id}/items/{char_no}/reset")
    assert resp.status_code == 200, resp.text

    # DB: reviewer_decision cleared
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(
            ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
        ).first()
        assert item.reviewer_decision is None
        assert item.reviewed_by_id is None
        assert item.reviewed_at is None
    finally:
        db.close()

    # HTML: pending card shows Approve button, not the approved badge
    assert "Approved" not in resp.text or "Change Decision" not in resp.text
    assert 'hx-post' in resp.text
    # The reset card is pending: data-pending="true"
    assert 'data-pending="true"' in resp.text
    # OOB swap present
    assert "progress-bar" in resp.text
    assert "hx-swap-oob" in resp.text


# ---------------------------------------------------------------------------
# RESET-02: Reset overridden item
# ---------------------------------------------------------------------------

def test_reset_overridden_item(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST reset on an overridden item clears all override fields."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ReviewItem

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    client.get(f"/review/{run_id}", follow_redirects=False)

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        char_no = item.char_no
    finally:
        db.close()

    # Override the item
    resp = client.post(
        f"/review/{run_id}/items/{char_no}/override",
        data={"override_classification": "changed", "override_note": "Dimension shifted"},
    )
    assert resp.status_code == 200

    # Reset
    resp = client.post(f"/review/{run_id}/items/{char_no}/reset")
    assert resp.status_code == 200, resp.text

    # DB: all override fields cleared
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(
            ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
        ).first()
        assert item.reviewer_decision is None
        assert item.override_classification is None
        assert item.override_note is None
        assert item.reviewed_by_id is None
        assert item.reviewed_at is None
    finally:
        db.close()

    # Card is pending again
    assert 'data-pending="true"' in resp.text


# ---------------------------------------------------------------------------
# RESET-03: Reset on signed-off run returns 409
# ---------------------------------------------------------------------------

def test_reset_signed_off_returns_409(client: TestClient, engineer_user, db_engine, tmp_path):
    """POST reset on a signed-off run is rejected with 409."""
    from unittest.mock import patch
    from sqlalchemy.orm import sessionmaker
    from shop.models import ReviewItem
    from shop.services.review import attempt_sign_off

    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=1, status="completed")

    client.get(f"/review/{run_id}", follow_redirects=False)

    TestingSession = sessionmaker(bind=db_engine)
    db = TestingSession()
    try:
        item = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).first()
        char_no = item.char_no
        # Approve so the queue has no pending items
        item.reviewer_decision = "approved"
        db.commit()

        # Force run to signed_off directly (bypass PDF generation)
        from shop.models import Run
        run = db.query(Run).filter(Run.id == run_id).first()
        run.status = "signed_off"
        db.commit()
    finally:
        db.close()

    resp = client.post(f"/review/{run_id}/items/{char_no}/reset")
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"
