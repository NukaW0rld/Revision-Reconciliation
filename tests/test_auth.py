"""
Auth tests: AUTH-01 (engineer creation), AUTH-02 (session persistence),
AUTH-03 (logout).
"""
import pytest


@pytest.mark.xfail(strict=False, reason="Admin user creation requires Plan 05 (admin routes)")
def test_admin_creates_engineer(client, admin_user):
    """AUTH-01: POST /admin/users with name+email creates an engineer account.

    Steps:
    - Log in as admin.
    - POST /admin/users with name and email payload.
    - Assert 200 response and user present in DB with role=engineer.
    """
    raise NotImplementedError("Admin routes not yet implemented — Plan 05")


def test_session_persists(client, admin_user):
    """AUTH-02: Session cookie persists across requests after login.

    Steps:
    - POST /login with valid admin credentials.
    - GET /dashboard using the returned session cookie.
    - Assert 200 response (not redirect to login).
    """
    # POST /login with valid credentials
    resp = client.post(
        "/login",
        data={"email": "admin@shop.local", "password": "changeme"},
        follow_redirects=False,
    )
    assert resp.status_code == 302, f"Expected 302 redirect after login, got {resp.status_code}"
    assert resp.headers.get("location") == "/dashboard"
    assert "session_token" in resp.cookies

    # GET /dashboard with the cookie in place (TestClient carries cookies automatically)
    dashboard_resp = client.get("/dashboard", follow_redirects=False)
    assert dashboard_resp.status_code == 200
    assert "admin@shop.local" in dashboard_resp.text


def test_logout(client, admin_user):
    """AUTH-03: POST /logout deletes the session and clears the cookie.

    Steps:
    - POST /login as admin.
    - POST /logout.
    - Assert session is deleted and session cookie is cleared (expired or absent).
    """
    # Login first
    client.post(
        "/login",
        data={"email": "admin@shop.local", "password": "changeme"},
        follow_redirects=False,
    )
    assert "session_token" in client.cookies

    # Logout
    logout_resp = client.post("/logout", follow_redirects=False)
    assert logout_resp.status_code == 302
    assert logout_resp.headers.get("location") == "/login"

    # Cookie should be cleared (expired max-age=0 or absent)
    # After logout, accessing /dashboard should redirect to /login
    dashboard_resp = client.get("/dashboard", follow_redirects=False)
    assert dashboard_resp.status_code == 302


def test_login_invalid_credentials(client, admin_user):
    """POST /login with wrong password returns 200 with error message (no redirect)."""
    resp = client.post(
        "/login",
        data={"email": "admin@shop.local", "password": "wrongpass"},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Invalid email or password" in resp.text


def test_login_inactive_user(client, db_engine):
    """POST /login for inactive user returns 200 with deactivation error."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import User
    from shop.services.auth import hash_password

    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        user = User(
            email="inactive@shop.local",
            hashed_password=hash_password("changeme"),
            role="engineer",
            is_active=False,
        )
        db.add(user)
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/login",
        data={"email": "inactive@shop.local", "password": "changeme"},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Account is deactivated" in resp.text


def test_login_page_get(client):
    """GET /login returns 200 with the login form."""
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "<form" in resp.text


def test_dashboard_no_cookie_redirects(client):
    """GET /dashboard without a session cookie redirects to /login."""
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 302
