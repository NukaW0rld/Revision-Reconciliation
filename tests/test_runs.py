"""
Test stubs for Phase 2 run/upload/pipeline requirements.

UPLOAD-01..05 tests are implemented here; PIPE-* remain as xfail stubs
until Plans 03-07 implement those features.

Requirement traceability:
  UPLOAD-01..05  Upload form and file validation
  PIPE-01..06    Pipeline task execution and status transitions
  PIPE-11        Alert creation on run failure
"""
import io
import pytest
from sqlalchemy.orm import sessionmaker


def _make_vector_pdf_bytes() -> bytes:
    """Return minimal single-page vector PDF with extractable text."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1.00 ± 0.01")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_multipage_pdf_bytes(n_pages: int = 3) -> bytes:
    """Return minimal vector PDF with n_pages pages."""
    import fitz
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page()
        page.insert_text((50, 50), "drawing text")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_excel_bytes() -> bytes:
    """Return valid minimal Excel bytes."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Char No", "Requirement", "Reference Location"])
    ws.append([1, "1.00 ± 0.01", "Zone A"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _login_engineer(client, db_engine, engineer_user):
    """Seed a session for the engineer user and set cookie on client."""
    from sqlalchemy.orm import sessionmaker
    from shop.services.auth import create_session
    Session = sessionmaker(bind=db_engine)
    db = Session()
    token = create_session(db, engineer_user)
    db.close()
    client.cookies.set("session_token", token)
    return token


# ---------------------------------------------------------------------------
# Upload requirements
# ---------------------------------------------------------------------------


def test_upload_creates_run(client, db_engine, engineer_user, huey_immediate):
    """UPLOAD-01: Engineer uploads Rev A PDF, Rev B PDF, Form 3 Excel, and part metadata;
    a Run record is created and the response redirects to /runs/{id}."""
    from shop.models import Run
    Session = sessionmaker(bind=db_engine)

    _login_engineer(client, db_engine, engineer_user)
    pdf_bytes = _make_vector_pdf_bytes()
    excel_bytes = _make_excel_bytes()

    resp = client.post(
        "/runs/new",
        data={
            "part_number": "PN-001",
            "rev_a_label": "A",
            "rev_b_label": "B",
            "customer": "Acme",
            "job_number": "JOB-1",
            "revA_page": "0",
            "revB_page": "0",
        },
        files={
            "revA_pdf": ("revA.pdf", pdf_bytes, "application/pdf"),
            "revB_pdf": ("revB.pdf", pdf_bytes, "application/pdf"),
            "form3_xlsx": ("form3.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302, f"Expected 302 redirect, got {resp.status_code}: {resp.text[:200]}"
    location = resp.headers.get("location", "")
    assert location.startswith("/runs/"), f"Expected redirect to /runs/{{id}}, got {location!r}"

    # Check Run was persisted
    db = Session()
    run_id = int(location.split("/")[-1])
    run = db.query(Run).filter(Run.id == run_id).first()
    db.close()
    assert run is not None
    assert run.status == "queued"


def test_raster_pdf_rejected(client, db_engine, engineer_user):
    """UPLOAD-02: POST /runs/validate-pdf with a raster PDF returns error partial."""
    _login_engineer(client, db_engine, engineer_user)

    # Create a raster-like PDF: no text, one large image covering the page
    import fitz
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    raster_pdf_bytes = buf.getvalue()  # No text, no images — will be treated as not-raster
    # Use actual raster PDF: insert a full-page image
    # For simplicity, test the validate-pdf endpoint with a genuinely raster PDF
    # We'll create one by embedding a PNG image that covers the page
    import struct
    import zlib

    def _make_png(w: int = 10, h: int = 10) -> bytes:
        """Minimal valid PNG."""
        def chunk(name, data):
            c = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)
        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
        idat_data = zlib.compress(raw)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr_data)
            + chunk(b"IDAT", idat_data)
            + chunk(b"IEND", b"")
        )

    png_bytes = _make_png(100, 100)
    doc2 = fitz.open()
    page = doc2.new_page(width=595, height=842)
    rect = fitz.Rect(0, 0, 595, 842)
    page.insert_image(rect, stream=png_bytes)
    buf2 = io.BytesIO()
    doc2.save(buf2)
    doc2.close()
    raster_pdf_bytes = buf2.getvalue()

    resp = client.post(
        "/runs/validate-pdf",
        data={"field": "revA"},
        files={"file": ("revA.pdf", raster_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    # Should return error partial — not the page selector
    assert "error" in resp.text.lower() or "scanned" in resp.text.lower() or "vector" in resp.text.lower()


def test_multipage_pdf_page_selector(client, db_engine, engineer_user):
    """UPLOAD-03: POST /runs/validate-pdf with multi-page PDF returns page selector partial."""
    _login_engineer(client, db_engine, engineer_user)
    pdf_bytes = _make_multipage_pdf_bytes(3)

    resp = client.post(
        "/runs/validate-pdf",
        data={"field": "revA"},
        files={"file": ("revA.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    # Should return page selector partial
    assert "page" in resp.text.lower() or "select" in resp.text.lower()


def test_invalid_excel_rejected(client, db_engine, engineer_user):
    """UPLOAD-04: POST /runs/new with an unreadable Excel returns 422 with error message."""
    _login_engineer(client, db_engine, engineer_user)
    pdf_bytes = _make_vector_pdf_bytes()

    resp = client.post(
        "/runs/new",
        data={
            "part_number": "PN-001",
            "rev_a_label": "A",
            "rev_b_label": "B",
            "customer": "Acme",
            "job_number": "JOB-1",
            "revA_page": "0",
            "revB_page": "0",
        },
        files={
            "revA_pdf": ("revA.pdf", pdf_bytes, "application/pdf"),
            "revB_pdf": ("revB.pdf", pdf_bytes, "application/pdf"),
            "form3_xlsx": ("form3.xlsx", b"not an excel file at all!!", "application/octet-stream"),
        },
        follow_redirects=False,
    )
    # Should get 422 with error shown in the form
    assert resp.status_code == 422
    assert "error" in resp.text.lower() or "form" in resp.text.lower()


def test_rev_labels_stored(client, db_engine, engineer_user, huey_immediate):
    """UPLOAD-05: Rev A and Rev B labels are persisted in Run.rev_a_label and Run.rev_b_label."""
    from shop.models import Run
    Session = sessionmaker(bind=db_engine)

    _login_engineer(client, db_engine, engineer_user)
    pdf_bytes = _make_vector_pdf_bytes()
    excel_bytes = _make_excel_bytes()

    resp = client.post(
        "/runs/new",
        data={
            "part_number": "PN-REV",
            "rev_a_label": "C",
            "rev_b_label": "D",
            "customer": "Corp",
            "job_number": "JOB-REV",
            "revA_page": "0",
            "revB_page": "0",
        },
        files={
            "revA_pdf": ("revA.pdf", pdf_bytes, "application/pdf"),
            "revB_pdf": ("revB.pdf", pdf_bytes, "application/pdf"),
            "form3_xlsx": ("form3.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    location = resp.headers.get("location", "")
    run_id = int(location.split("/")[-1])

    db = Session()
    run = db.query(Run).filter(Run.id == run_id).first()
    db.close()
    assert run.rev_a_label == "C"
    assert run.rev_b_label == "D"


# ---------------------------------------------------------------------------
# Pipeline task requirements (remain as xfail stubs until Plan 03)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-01")
def test_pipeline_task_enqueued(huey_immediate):
    """After a valid upload, a Huey pipeline task is enqueued and executed
    asynchronously (synchronously in tests via huey_immediate fixture)."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-02")
def test_stage_progress_updates():
    """During execution, Run.current_stage and Run.current_stage_index are
    updated at each of the 8 pipeline stages."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-03")
def test_run_status_lifecycle():
    """Run.status transitions through queued -> running -> completed for a
    successful pipeline execution."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-04")
def test_revA_balloon_failure():
    """When Rev A balloon detection fails, Run.status is set to 'failed',
    Run.failure_stage is populated, and a descriptive failure_message is stored."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-05")
def test_revB_balloon_warning():
    """When Rev B balloon detection fails (non-fatal), Run.status is set to
    'warning' and Run.warning_type is set to 'revB_balloon'."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-06")
def test_low_confidence_warning():
    """When alignment confidence is uniformly low, Run.status is set to 'warning',
    Run.warning_type is 'low_confidence', and Run.confidence_summary is populated."""
    pytest.fail("not implemented")


# ---------------------------------------------------------------------------
# Alert requirements
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="not implemented — PIPE-11")
def test_failure_alert_created():
    """When a run fails, a RunAlert is created for the assigned reviewer so
    they receive an in-app notification."""
    pytest.fail("not implemented")
