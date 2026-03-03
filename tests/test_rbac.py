"""
RBAC tests: AUTH-05 (role-based access control enforcement).
Stubs marked xfail until shop/ package (Plan 02+) exists.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_engineer_cannot_access_admin(client, db_engine):
    """AUTH-05: Engineer role cannot access admin-only routes.

    Steps:
    - Create an engineer user in the DB.
    - POST /login with engineer credentials.
    - GET /admin with the returned session cookie.
    - Assert response is a redirect to /dashboard (not 200).
    """
    raise NotImplementedError("shop/ not implemented yet")
