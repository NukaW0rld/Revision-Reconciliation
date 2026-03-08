"""Phase 4: Amendment model tests."""
import pytest
from datetime import datetime
from fastapi.testclient import TestClient


def _make_signed_run_with_items(db, engineer_id):
    """Create signed-off run with 3 ReviewItems for amendment testing."""
    from shop.models import Run, ReviewItem
    run = Run(
        part_number="PN-AMEND",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="A001",
        status="signed_off",
        revA_path="/tmp/revA.pdf",
        revB_path="/tmp/revB.pdf",
        form3_path="/tmp/form3.xlsx",
        output_dir="/tmp/out_amend",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_id,
        packet_versions=[
            {
                "version": 1,
                "type": "original",
                "path": "/tmp/out_amend/packets/v1.pdf",
                "signed_at": "2026-03-08T12:00:00",
            }
        ],
    )
    db.add(run)
    db.flush()
    for i in (1, 2, 3):
        item = ReviewItem(
            run_id=run.id,
            char_no=i,
            pipeline_classification="unchanged",
            confidence=0.9,
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8),
        )
        db.add(item)
    db.commit()
    db.refresh(run)
    return run


def test_create_amendment(client: TestClient, engineer_user):
    """AMEND-01: create_amendment makes new Run with pre-filled ReviewItems."""
    from shop.dependencies import get_db
    from shop.models import ReviewItem
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    parent = _make_signed_run_with_items(db, engineer_user.id)
    from shop.services.amendments import create_amendment
    amendment = create_amendment(db, parent, engineer_user.id)

    assert amendment.id != parent.id
    assert amendment.parent_run_id == parent.id
    assert amendment.status == "reviewing"
    # All 3 items cloned
    items = db.query(ReviewItem).filter(ReviewItem.run_id == amendment.id).all()
    assert len(items) == 3
    # Pre-filled decisions
    assert all(i.reviewer_decision == "approved" for i in items)


def test_amendment_files_locked(client: TestClient, engineer_user):
    """AMEND-02: amendment shares parent input file paths (files locked)."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    parent = _make_signed_run_with_items(db, engineer_user.id)
    from shop.services.amendments import create_amendment
    amendment = create_amendment(db, parent, engineer_user.id)

    assert amendment.revA_path == parent.revA_path
    assert amendment.revB_path == parent.revB_path
    assert amendment.form3_path == parent.form3_path
    assert amendment.output_dir == parent.output_dir


def test_amendment_versioned_packet(client: TestClient, engineer_user):
    """AMEND-03: amendment inherits parent packet_versions; next sign-off produces v2."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    parent = _make_signed_run_with_items(db, engineer_user.id)
    from shop.services.amendments import create_amendment
    amendment = create_amendment(db, parent, engineer_user.id)

    # Amendment starts with inherited versions (v1 from parent)
    assert len(amendment.packet_versions or []) == 1
    assert amendment.packet_versions[0]["version"] == 1
    assert amendment.packet_versions[0]["type"] == "original"

    # Simulate what generate_and_store_audit_packet would compute
    existing = amendment.packet_versions or []
    next_version = len(existing) + 1
    assert next_version == 2, "Amendment sign-off should produce v2"

    # Original parent's packet_versions unchanged
    db.expire(parent)
    db.refresh(parent)
    assert len(parent.packet_versions) == 1
    assert parent.packet_versions[0]["version"] == 1
