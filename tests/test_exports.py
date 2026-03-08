"""Phase 4: Audit packet and work order export tests.

Stubs marked xfail(strict=False) — will be implemented in Plans 02 and 03.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="PACKET-01: audit packet PDF not yet implemented")
def test_audit_packet_pdf_bytes():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="PACKET-02: audit packet CSV not yet implemented")
def test_audit_packet_csv_rows():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="PACKET-03: re-download audit packet not yet implemented")
def test_audit_packet_redownload():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="WORK-01: work order button not yet implemented")
def test_work_order_button_visible():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="WORK-02: work order filter not yet implemented")
def test_work_order_filters_status():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="WORK-03: work order priority labels not yet implemented")
def test_work_order_priority_labels():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="WORK-04: work order PDF/CSV not yet implemented")
def test_work_order_pdf_csv():
    assert False, "not implemented"
