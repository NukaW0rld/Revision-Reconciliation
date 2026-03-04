"""
Test stubs for Phase 2 run/upload/pipeline requirements.

All tests are marked xfail(strict=False) — they appear in collection output
and are implemented incrementally across Plans 02-07.

Requirement traceability:
  UPLOAD-01..05  Upload form and file validation
  PIPE-01..06    Pipeline task execution and status transitions
  PIPE-11        Alert creation on run failure
"""
import pytest


# ---------------------------------------------------------------------------
# Upload requirements
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="not implemented — UPLOAD-01")
def test_upload_creates_run():
    """Engineer uploads Rev A PDF, Rev B PDF, Form 3 Excel, and part metadata;
    a Run record is created and the response redirects to the status page."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — UPLOAD-02")
def test_raster_pdf_rejected():
    """Uploading a raster (scanned) PDF returns an error partial instead of
    a page selector or success response."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — UPLOAD-03")
def test_multipage_pdf_page_selector():
    """Uploading a multi-page PDF triggers the HTMX page-selector partial
    so the engineer can choose the correct drawing page."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — UPLOAD-04")
def test_invalid_excel_rejected():
    """Uploading an unreadable or unrecognized Excel file produces a clear
    validation error message before creating any Run record."""
    pytest.fail("not implemented")


@pytest.mark.xfail(strict=False, reason="not implemented — UPLOAD-05")
def test_rev_labels_stored():
    """Revision labels (Rev A letter, Rev B letter) captured from the upload
    form are persisted in Run.rev_a_label and Run.rev_b_label."""
    pytest.fail("not implemented")


# ---------------------------------------------------------------------------
# Pipeline task requirements
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
