"""
Setup wizard tests: SETUP-01 through SETUP-04.
"""
import pytest


def test_wizard_intercepts_all_routes(client_setup_incomplete, shop_config):
    """SETUP-01: Setup wizard middleware redirects all routes to /setup/step1
    when setup_complete=False.

    Steps:
    - Ensure shop_config has setup_complete=False (via client_setup_incomplete fixture).
    - GET / (home/dashboard).
    - Assert response redirects to /setup/step1.
    """
    response = client_setup_incomplete.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/setup/step1"


def test_setup_step1_get_returns_form(client_setup_incomplete, shop_config):
    """GET /setup/step1 returns 200 with shop name form."""
    response = client_setup_incomplete.get("/setup/step1", follow_redirects=False)
    assert response.status_code == 200
    assert "shop_name" in response.text


def test_setup_step1_post_advances_wizard(client_setup_incomplete, shop_config):
    """POST /setup/step1 with shop_name saves it and redirects to /setup/step2."""
    response = client_setup_incomplete.post(
        "/setup/step1",
        data={"shop_name": "Acme Quality Lab"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/setup/step2" in response.headers["location"]


def test_setup_step2_blocks_skip(client_setup_incomplete, shop_config):
    """GET /setup/step2 with wizard_step=0 redirects to /setup/step1 (no skipping)."""
    response = client_setup_incomplete.get("/setup/step2", follow_redirects=False)
    assert response.status_code == 302
    assert "/setup/step1" in response.headers["location"]


def test_setup_step2_rejects_default_password(client_setup_incomplete, db_engine):
    """POST /setup/step2 with 'changeme' returns 200 with an error message."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ShopConfig
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        db.add(ShopConfig(id=1, setup_complete=False, wizard_step=1))
        db.commit()
    finally:
        db.close()

    response = client_setup_incomplete.post(
        "/setup/step2",
        data={"new_password": "changeme", "confirm_password": "changeme"},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert "Cannot reuse the default password" in response.text


def test_setup_step3_blocks_skip(client_setup_incomplete, shop_config):
    """GET /setup/step3 with wizard_step=0 redirects back (no skipping)."""
    response = client_setup_incomplete.get("/setup/step3", follow_redirects=False)
    assert response.status_code == 302
    location = response.headers["location"]
    assert "/setup/step1" in location or "/setup/step2" in location


def test_setup_root_redirect(client_setup_incomplete, shop_config):
    """GET /setup/ redirects to /setup/step{wizard_step+1}."""
    response = client_setup_incomplete.get("/setup/", follow_redirects=False)
    assert response.status_code == 302
    assert "/setup/step1" in response.headers["location"]


def test_form3_upload_autodetect(client, admin_user, db_engine, shop_config):
    """SETUP-02: POST /setup/step4/upload with a valid xlsx returns column mapping UI.

    Steps:
    - Construct a minimal AS9102 Form 3 xlsx in memory (BytesIO).
    - POST it to /setup/step4/upload as a multipart file upload.
    - Assert 200 response body contains HTML with dropdown elements for column mapping.
    """
    from io import BytesIO
    from datetime import datetime, timedelta
    from sqlalchemy.orm import sessionmaker
    from shop.models import UserSession
    import openpyxl

    # Seed wizard_step=3 so step4 GET/POST is accessible
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        from shop.models import ShopConfig
        cfg = db.query(ShopConfig).first()
        cfg.wizard_step = 3
        db.commit()
    finally:
        db.close()

    # Create a session cookie for admin user
    import secrets
    token = secrets.token_urlsafe(32)
    db = Session()
    try:
        sess = UserSession(
            id=token,
            user_id=admin_user.id,
            expires_at=datetime.utcnow() + timedelta(hours=8),
        )
        db.add(sess)
        db.commit()
    finally:
        db.close()

    # Build minimal xlsx with Form 3 headers
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Char No", "Reference Location", "Requirement"])
    ws.append([1, "Zone A", "1.25 +/- 0.05"])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    resp = client.post(
        "/setup/step4/upload",
        files={"file": ("form3.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        cookies={"session_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    # Should return mapping partial — contains select dropdowns
    assert "<select" in resp.text


def test_empty_file_error(client, admin_user, db_engine, shop_config):
    """SETUP-03: POST /setup/step4/upload with empty bytes returns error before mapping UI.

    Steps:
    - POST an empty bytes object to /setup/step4/upload.
    - Assert an error response is returned (not the column mapping HTML).
    """
    from io import BytesIO
    from datetime import datetime, timedelta
    from sqlalchemy.orm import sessionmaker
    from shop.models import UserSession
    import secrets

    Session = sessionmaker(bind=db_engine)

    token = secrets.token_urlsafe(32)
    db = Session()
    try:
        sess = UserSession(
            id=token,
            user_id=admin_user.id,
            expires_at=datetime.utcnow() + timedelta(hours=8),
        )
        db.add(sess)
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/setup/step4/upload",
        files={"file": ("empty.xlsx", BytesIO(b""), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        cookies={"session_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    # Should return error partial — no select dropdowns
    assert "<select" not in resp.text
    # Should contain error messaging
    assert "error" in resp.text.lower() or "empty" in resp.text.lower()


def test_noncontiguous_char_no(client, admin_user, db_engine, shop_config):
    """SETUP-04: Non-contiguous char_no values are accepted without normalization.

    Steps:
    - Construct an xlsx with char_no values [1, 5, 99, 200] (non-contiguous).
    - POST it to /setup/step4/upload.
    - Assert all four char_no values appear in the response (accepted as-is).
    """
    from io import BytesIO
    from datetime import datetime, timedelta
    from sqlalchemy.orm import sessionmaker
    from shop.models import UserSession, ShopConfig
    import openpyxl
    import secrets

    Session = sessionmaker(bind=db_engine)

    # Seed wizard_step=3
    db = Session()
    try:
        cfg = db.query(ShopConfig).first()
        cfg.wizard_step = 3
        db.commit()
    finally:
        db.close()

    token = secrets.token_urlsafe(32)
    db = Session()
    try:
        sess = UserSession(
            id=token,
            user_id=admin_user.id,
            expires_at=datetime.utcnow() + timedelta(hours=8),
        )
        db.add(sess)
        db.commit()
    finally:
        db.close()

    # Build xlsx with non-contiguous char_no values
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Char No", "Requirement"])
    for val in [1, 5, 99, 200]:
        ws.append([val, f"Req {val}"])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    resp = client.post(
        "/setup/step4/upload",
        files={"file": ("form3.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        cookies={"session_token": token},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    # All four non-contiguous char_no values must appear in the preview
    for val in ["1", "5", "99", "200"]:
        assert val in resp.text
