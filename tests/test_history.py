"""Phase 4: Run history and retention tests.

Stubs marked xfail(strict=False) — will be implemented in Plan 04.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="HISTORY-01: date filter not yet implemented")
def test_history_filters():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="HISTORY-02: signed-off read-only view not yet implemented")
def test_signed_off_readonly_view():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="HISTORY-03: signed-off exempt from cleanup not yet implemented")
def test_cleanup_exempt_signed_off():
    assert False, "not implemented"


@pytest.mark.xfail(strict=False, reason="HISTORY-04: cleanup deletes old runs not yet implemented")
def test_cleanup_deletes_old_runs():
    assert False, "not implemented"
