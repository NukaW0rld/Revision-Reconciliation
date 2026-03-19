"""
Admin user management tests: AUTH-01, AUTH-04 (create/deactivate engineer).
Tests written TDD-style for Plan 04 — admin router implementation.
"""
from datetime import timedelta
from shop.utils import utcnow
from sqlalchemy.orm import sessionmaker
from shop.models import User, UserSession
from shop.services.auth import hash_password


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
            expires_at=utcnow() + timedelta(hours=8),
        )
        db.add(sess)
        db.commit()
    finally:
        db.close()
    return token


def _seed_engineer(db_engine) -> User:
    """Seed an active engineer user and return the User object."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        user = User(
            username="engineer_test",
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


def test_admin_list_users(client, admin_user, db_engine):
    """AUTH-04: Admin can view the list of all users at GET /admin/users."""
    token = _make_session_cookie(db_engine, admin_user)
    client.cookies.set("session_token", token)
    try:
        resp = client.get("/admin/users", follow_redirects=False)
        assert resp.status_code == 200
        assert admin_user.username in resp.text
    finally:
        client.cookies.delete("session_token")


def test_create_engineer(client, admin_user, db_engine):
    """AUTH-01: Admin can create an engineer account via POST /admin/users."""
    token = _make_session_cookie(db_engine, admin_user)
    client.cookies.set("session_token", token)
    try:
        resp = client.post(
            "/admin/users",
            data={"username": "new_engineer", "password": "securepass", "role": "engineer"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        # Verify row HTML is returned
        assert "new_engineer" in resp.text
        # Verify user exists in DB
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            created = db.query(User).filter(User.username == "new_engineer").first()
            assert created is not None
            assert created.is_active is True
            assert created.role == "engineer"
            assert created.created_at is not None
        finally:
            db.close()
    finally:
        client.cookies.delete("session_token")


def test_create_engineer_duplicate_email(client, admin_user, db_engine):
    """POST /admin/users with duplicate email returns 200 with error (no 500)."""
    token = _make_session_cookie(db_engine, admin_user)
    client.cookies.set("session_token", token)
    try:
        # First creation
        client.post(
            "/admin/users",
            data={"username": "dupe_user", "password": "pass12345"},
            follow_redirects=False,
        )
        # Second creation — should return error partial, not 500
        resp = client.post(
            "/admin/users",
            data={"username": "dupe_user", "password": "pass12345"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "already exists" in resp.text.lower() or "dupe_user" in resp.text
    finally:
        client.cookies.delete("session_token")


def test_deactivate_engineer(client, admin_user, db_engine):
    """AUTH-04: Admin can deactivate an engineer account."""
    engineer = _seed_engineer(db_engine)
    token = _make_session_cookie(db_engine, admin_user)
    client.cookies.set("session_token", token)
    try:
        resp = client.post(
            f"/admin/users/{engineer.id}/deactivate",
            follow_redirects=False,
        )
        assert resp.status_code == 200

        # Verify is_active=False in DB
        Session = sessionmaker(bind=db_engine)
        db = Session()
        try:
            refreshed = db.query(User).filter(User.id == engineer.id).first()
            assert refreshed.is_active is False
        finally:
            db.close()
    finally:
        client.cookies.delete("session_token")


def test_deactivate_nonexistent_user(client, admin_user, db_engine):
    """POST /admin/users/{id}/deactivate on non-existent id returns 404."""
    token = _make_session_cookie(db_engine, admin_user)
    client.cookies.set("session_token", token)
    try:
        resp = client.post(
            "/admin/users/99999/deactivate",
            follow_redirects=False,
        )
        assert resp.status_code == 404
    finally:
        client.cookies.delete("session_token")
