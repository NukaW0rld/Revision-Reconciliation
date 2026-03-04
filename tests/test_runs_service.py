"""
Unit tests for shop/services/runs.py — Task 1 (TDD RED phase).

Tests for:
  - validate_pdf_bytes: raster detection, page count
  - validate_excel_bytes: raises ValueError on bad Excel
  - create_run: creates Run record with correct fields
  - save_upload: saves bytes to correct path
"""
import io
import struct
import pytest
from pathlib import Path
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker

from shop.database import Base
from shop.models import Run, User
from shop.services.auth import hash_password


def _make_db():
    """Create a fresh in-memory SQLite DB with all tables."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def _make_minimal_vector_pdf_bytes() -> bytes:
    """Return minimal vector PDF bytes (has extractable text)."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1.00 ± 0.01")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_minimal_multipage_pdf_bytes(n_pages: int = 3) -> bytes:
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


def _make_minimal_excel_bytes() -> bytes:
    """Return valid minimal Excel bytes with some header data."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Char No", "Requirement", "Reference Location"])
    ws.append([1, "1.00 ± 0.01", "Zone A"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# validate_pdf_bytes
# ---------------------------------------------------------------------------

class TestValidatePdfBytes:

    def test_vector_single_page_not_raster(self):
        from shop.services.runs import validate_pdf_bytes
        pdf_bytes = _make_minimal_vector_pdf_bytes()
        result = validate_pdf_bytes(pdf_bytes)
        assert result["is_raster"] is False
        assert result["page_count"] == 1

    def test_vector_multipage_returns_correct_count(self):
        from shop.services.runs import validate_pdf_bytes
        pdf_bytes = _make_minimal_multipage_pdf_bytes(3)
        result = validate_pdf_bytes(pdf_bytes)
        assert result["is_raster"] is False
        assert result["page_count"] == 3

    def test_result_has_required_keys(self):
        from shop.services.runs import validate_pdf_bytes
        pdf_bytes = _make_minimal_vector_pdf_bytes()
        result = validate_pdf_bytes(pdf_bytes)
        assert "is_raster" in result
        assert "page_count" in result


# ---------------------------------------------------------------------------
# validate_excel_bytes
# ---------------------------------------------------------------------------

class TestValidateExcelBytes:

    def test_valid_excel_returns_none(self):
        from shop.services.runs import validate_excel_bytes
        excel_bytes = _make_minimal_excel_bytes()
        result = validate_excel_bytes(excel_bytes)
        assert result is None

    def test_empty_bytes_raises_value_error(self):
        from shop.services.runs import validate_excel_bytes
        with pytest.raises(ValueError):
            validate_excel_bytes(b"")

    def test_garbage_bytes_raises_value_error(self):
        from shop.services.runs import validate_excel_bytes
        with pytest.raises(ValueError):
            validate_excel_bytes(b"this is not an excel file at all!!")


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------

class TestCreateRun:

    def setup_method(self):
        self.engine = _make_db()
        self.Session = sessionmaker(bind=self.engine)
        # Seed a user
        db = self.Session()
        user = User(
            email="eng@test.local",
            hashed_password=hash_password("x"),
            role="engineer",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        self.user_id = user.id
        db.close()

    def teardown_method(self):
        Base.metadata.drop_all(self.engine)

    def test_creates_run_record(self):
        from shop.services.runs import create_run
        db = self.Session()
        run = create_run(
            db,
            part_number="PN-001",
            rev_a_label="A",
            rev_b_label="B",
            customer="Acme",
            job_number="JOB-1",
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
            revA_page=0,
            revB_page=0,
            reviewer_id=self.user_id,
        )
        db.close()
        assert run.id is not None
        assert run.status == "queued"

    def test_rev_labels_stored(self):
        from shop.services.runs import create_run
        db = self.Session()
        run = create_run(
            db,
            part_number="PN-002",
            rev_a_label="C",
            rev_b_label="D",
            customer="AcmeCo",
            job_number="JOB-2",
            revA_path="/tmp/a.pdf",
            revB_path="/tmp/b.pdf",
            form3_path="/tmp/form3.xlsx",
            reviewer_id=self.user_id,
        )
        db.close()
        assert run.rev_a_label == "C"
        assert run.rev_b_label == "D"

    def test_paths_stored_as_strings(self):
        from shop.services.runs import create_run
        db = self.Session()
        run = create_run(
            db,
            part_number="PN-003",
            rev_a_label="A",
            rev_b_label="B",
            customer="Corp",
            job_number="JOB-3",
            revA_path=Path("/tmp/a.pdf"),
            revB_path=Path("/tmp/b.pdf"),
            form3_path=Path("/tmp/form3.xlsx"),
            reviewer_id=self.user_id,
        )
        db.close()
        assert isinstance(run.revA_path, str)
        assert isinstance(run.revB_path, str)
        assert isinstance(run.form3_path, str)


# ---------------------------------------------------------------------------
# save_upload
# ---------------------------------------------------------------------------

class TestSaveUpload:

    def test_saves_bytes_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
        # Must reload the module to pick up new env var
        import importlib, shop.services.runs as svc
        importlib.reload(svc)
        result = svc.save_upload(b"hello pdf", ".pdf", "test-run-uuid")
        assert result.exists()
        assert result.read_bytes() == b"hello pdf"

    def test_returns_path_with_correct_suffix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
        import importlib, shop.services.runs as svc
        importlib.reload(svc)
        result = svc.save_upload(b"data", ".xlsx", "run-uuid-2")
        assert result.suffix == ".xlsx"
