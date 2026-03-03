"""
DB model tests: AUTH-06 (User schema — reviewer fields present).
Stubs marked xfail until shop/ package (Plan 02+) exists.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_reviewer_fields(db_engine):
    """AUTH-06: User model has required fields for reviewer identity in audit records.

    Steps:
    - Create a User instance via the DB engine.
    - Verify the User object exposes: email, hashed_password, role, is_active, created_at.
    - These fields are required for reviewer identity in signed audit packets.
    """
    raise NotImplementedError("shop/ not implemented yet")
