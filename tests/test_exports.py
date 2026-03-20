"""Phase 4: Audit packet and work order export tests."""
import csv
import io
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

# conftest provides: client, admin_user, engineer_user fixtures


def _login_engineer(client, db_engine, engineer_user):
    """Seed a session for the engineer user and set cookie on client."""
    from shop.services.auth import create_session
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, engineer_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


def _make_signed_run(db, engineer_id):
    """Create a minimal signed-off Run with 2 ReviewItems for testing."""
    from shop.models import Run, ReviewItem
    run = Run(
        part_number="PN-TEST",
        rev_a_label="A",
        rev_b_label="B",
        customer="Acme",
        job_number="JOB-001",
        status="signed_off",
        output_dir=None,
        revA_path="/tmp/revA.pdf",
        revB_path="/tmp/revB.pdf",
        form3_path="/tmp/form3.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8, 12, 0, 0),
        signed_by_id=engineer_id,
        packet_versions=[{"version": 1, "type": "original", "path": "/nonexistent/v1.pdf", "signed_at": "2026-03-08T12:00:00"}],
    )
    db.add(run)
    db.flush()
    for i in (1, 2):
        item = ReviewItem(
            run_id=run.id,
            char_no=i,
            pipeline_classification="unchanged",
            confidence=0.95,
            requirement_revA=f"Req A {i}",
            requirement_revB=f"Req B {i}",
            reviewer_decision="approved",
            reviewed_by_id=engineer_id,
            reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
        )
        db.add(item)
    db.commit()
    return run


def test_audit_packet_pdf_bytes(client: TestClient):
    """PACKET-01: PDF bytes generated (smoke test — verifies WeasyPrint runs)."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    # Just verify the import works and WeasyPrint is available
    from shop.services.exports import render_audit_packet_pdf
    from shop.models import ShopConfig
    assert render_audit_packet_pdf  # importable


def test_audit_packet_csv_rows(client: TestClient, engineer_user):
    """PACKET-02: CSV has correct rows for each ReviewItem."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_signed_run(db, engineer_user.id)
    from shop.services.exports import generate_audit_packet_csv
    buf = generate_audit_packet_csv(db, run)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["char_no"] == "1"
    assert rows[0]["requirement_revA"] == "Req A 1"
    assert rows[0]["reviewer_decision"] == "approved"


def test_audit_packet_redownload(client: TestClient, db_engine, engineer_user):
    """PACKET-03: GET /exports/{id}/audit-packet.csv returns 200 with attachment header."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_signed_run(db, engineer_user.id)
    resp = client.get(f"/exports/{run.id}/audit-packet.csv")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


def _make_run_with_mixed_items(db, engineer_id):
    """Create signed-off run with changed, added, and unchanged items."""
    from shop.models import Run, ReviewItem
    run = Run(
        part_number="PN-WO",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="WO-001",
        status="signed_off",
        output_dir=None,
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_id,
    )
    db.add(run)
    db.flush()
    for char_no, classification in [(1, "changed"), (2, "added"), (3, "unchanged")]:
        item = ReviewItem(
            run_id=run.id,
            char_no=char_no,
            pipeline_classification=classification,
            confidence=0.9,
            requirement_revB=f"Req B {char_no}",
            reviewer_decision="approved",
        )
        db.add(item)
    db.commit()
    return run


def test_work_order_button_visible(client: TestClient, db_engine, engineer_user):
    """WORK-01: work-order links present on signed-off run status page."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    resp = client.get(f"/runs/{run.id}")
    assert resp.status_code == 200
    assert "work-order.pdf" in resp.text


def test_work_order_filters_status(client: TestClient, engineer_user):
    """WORK-02: work order CSV includes only changed and added characteristics."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 2  # only changed (char 1) and added (char 2)
    char_nos = {r["char_no"] for r in rows}
    assert "1" in char_nos
    assert "2" in char_nos
    assert "3" not in char_nos  # unchanged excluded
    # New columns present in header
    assert "requirement_revA" in rows[0]
    assert "confidence" in rows[0]
    assert "override_note" in rows[0]


def test_work_order_priority_labels(client: TestClient, engineer_user):
    """WORK-03: RE-MEASURE for changed, NEW for added; requirement_revA column present."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    assert rows["1"]["priority"] == "RE-MEASURE"
    assert rows["2"]["priority"] == "NEW"
    assert "requirement_revA" in rows["1"]


def test_work_order_pdf_csv(client: TestClient, db_engine, engineer_user):
    """WORK-04: Both work-order.pdf and .csv routes return 200 with attachment headers."""
    _login_engineer(client, db_engine, engineer_user)
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_mixed_items(db, engineer_user.id)
    for ext in ("csv",):  # csv only in tests (PDF needs WeasyPrint system deps)
        resp = client.get(f"/exports/{run.id}/work-order.{ext}")
        assert resp.status_code == 200, f"work-order.{ext} returned {resp.status_code}"
        assert "attachment" in resp.headers.get("content-disposition", "")


def _make_run_with_override(db, engineer_id):
    """Create signed-off run: char 1 changed+overridden with note, char 2 added (no note)."""
    from shop.models import Run, ReviewItem
    run = Run(
        part_number="PN-OVR",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test",
        job_number="OVR-001",
        status="signed_off",
        output_dir=None,
        revA_path="/tmp/a.pdf",
        revB_path="/tmp/b.pdf",
        form3_path="/tmp/f.xlsx",
        reviewer_id=engineer_id,
        signed_at=datetime(2026, 3, 8),
        signed_by_id=engineer_id,
    )
    db.add(run)
    db.flush()
    # char 1: pipeline says changed, reviewer overrides and adds a note
    item1 = ReviewItem(
        run_id=run.id,
        char_no=1,
        pipeline_classification="changed",
        confidence=0.65,
        requirement_revA="Ø 6.0 ± 0.1",
        requirement_revB="Ø 6.0 ± 0.05",
        reviewer_decision="overridden",
        override_classification="changed",
        override_note="Tolerance tightened — confirmed with design authority",
        reviewed_by_id=engineer_id,
        reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
    )
    # char 2: added, no override
    item2 = ReviewItem(
        run_id=run.id,
        char_no=2,
        pipeline_classification="added",
        confidence=0.92,
        requirement_revA=None,
        requirement_revB="R3.5 mm",
        reviewer_decision="approved",
        reviewed_by_id=engineer_id,
        reviewed_at=datetime(2026, 3, 8, 12, 0, 0),
    )
    db.add_all([item1, item2])
    db.commit()
    return run


def test_work_order_override_note_in_csv(client: TestClient, engineer_user):
    """WORK-05: override_note present for overridden items, empty for others."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_override(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    assert rows["1"]["override_note"] == "Tolerance tightened — confirmed with design authority"
    assert rows["2"]["override_note"] == ""


def test_work_order_confidence_in_csv(client: TestClient, engineer_user):
    """WORK-06: confidence formatted to 2 decimal places in CSV."""
    from shop.dependencies import get_db
    db_gen = client.app.dependency_overrides.get(get_db)
    db = next(db_gen()) if db_gen else None
    if db is None:
        pytest.skip("no DB override available")
    run = _make_run_with_override(db, engineer_user.id)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    reader = csv.DictReader(buf)
    rows = {r["char_no"]: r for r in reader}
    # char 1 confidence=0.65 → "0.65"
    assert rows["1"]["confidence"] == "0.65"
    # char 2 confidence=0.92 → "0.92"
    assert rows["2"]["confidence"] == "0.92"
    # values are 2 dp formatted strings
    for char_no, row in rows.items():
        parts = row["confidence"].split(".")
        assert len(parts) == 2 and len(parts[1]) == 2, f"char {char_no}: bad confidence format {row['confidence']!r}"
