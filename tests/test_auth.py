"""
Auth tests: AUTH-01 (engineer creation), AUTH-02 (session persistence),
AUTH-03 (logout). Stubs marked xfail until shop/ package (Plan 02+) exists.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_admin_creates_engineer(client, admin_user):
    """AUTH-01: POST /admin/users with name+email creates an engineer account.

    Steps:
    - Log in as admin.
    - POST /admin/users with name and email payload.
    - Assert 200 response and user present in DB with role=engineer.
    """
    raise NotImplementedError("shop/ not implemented yet")


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_session_persists(client, admin_user):
    """AUTH-02: Session cookie persists across requests after login.

    Steps:
    - POST /login with valid admin credentials.
    - GET /dashboard using the returned session cookie.
    - Assert 200 response (not redirect to login).
    """
    raise NotImplementedError("shop/ not implemented yet")


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_logout(client, admin_user):
    """AUTH-03: POST /logout deletes the session and clears the cookie.

    Steps:
    - POST /login as admin.
    - POST /logout.
    - Assert session is deleted and session cookie is cleared (expired or absent).
    """
    raise NotImplementedError("shop/ not implemented yet")
