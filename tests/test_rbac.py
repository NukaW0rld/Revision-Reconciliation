"""
RBAC tests: AUTH-05 (role-based access control enforcement).
Tests written TDD-style — RED first, then implementation makes them GREEN.
Admin routes are implemented in Plan 05; tests are xfail until then.
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from shop.models import User, UserSession
from shop.services.auth import hash_password


def _seed_engineer(db_engine) -> User:
    """Seed an active engineer user and return the User object."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        user = User(
            email="engineer@shop.local",
            hashed_password=hash_password("password123"),
            role="engineer",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _make_session_cookie(db_engine, user: User) -> str:
    """Insert a valid UserSession and return the token (simulates login)."""
    import secrets
    token = secrets.token_urlsafe(32)
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        sess = UserSession(
            id=token,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(hours=8),
        )
        db.add(sess)
        db.commit()
    finally:
        db.close()
    return token


def test_engineer_cannot_access_admin(client, db_engine):
    """AUTH-05: Engineer role cannot access admin-only routes.

    GET /admin/users with engineer session: silently redirects to /dashboard.
    """
    engineer = _seed_engineer(db_engine)
    token = _make_session_cookie(db_engine, engineer)

    resp = client.get("/admin/users", cookies={"session_token": token}, follow_redirects=False)
    # Must be a redirect (302), not 200 or 403
    assert resp.status_code == 302
    assert "/dashboard" in resp.headers.get("location", "")


def test_no_session_redirects_to_login(client):
    """GET /admin/users without session cookie redirects to /login."""
    resp = client.get("/admin/users", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers.get("location", "")


def test_admin_can_access_admin(client, admin_user, db_engine):
    """AUTH-05: Admin role can access admin-only routes."""
    token = _make_session_cookie(db_engine, admin_user)
    resp = client.get("/admin/users", cookies={"session_token": token}, follow_redirects=False)
    assert resp.status_code == 200
