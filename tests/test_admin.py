"""
Admin user management tests: AUTH-04 (deactivate engineer).
Stubs marked xfail until shop/ package (Plan 02+) exists.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_deactivate_engineer(client, admin_user, db_engine):
    """AUTH-04: Admin can deactivate an engineer account.

    Steps:
    - Create an engineer user in the DB.
    - Log in as admin.
    - POST /admin/users/{id}/deactivate.
    - Assert user.is_active=False in the DB.
    """
    raise NotImplementedError("shop/ not implemented yet")
