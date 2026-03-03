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


@pytest.mark.xfail(strict=False, reason="Step 4 not implemented yet — Wave 3+")
def test_form3_upload_autodetect(client, admin_user, shop_config):
    """SETUP-02: POST /setup/step4/upload with a valid xlsx returns column mapping UI.

    Steps:
    - Construct a minimal AS9102 Form 3 xlsx in memory (BytesIO).
    - POST it to /setup/step4/upload as a multipart file upload.
    - Assert 200 response body contains HTML with dropdown elements for column mapping.
    """
    raise NotImplementedError("Step 4 not implemented yet")


@pytest.mark.xfail(strict=False, reason="Step 4 not implemented yet — Wave 3+")
def test_empty_file_error(client, admin_user, shop_config):
    """SETUP-03: POST /setup/step4/upload with empty bytes returns error before mapping UI.

    Steps:
    - POST an empty bytes object to /setup/step4/upload.
    - Assert an error response is returned (not the column mapping HTML).
    """
    raise NotImplementedError("Step 4 not implemented yet")


@pytest.mark.xfail(strict=False, reason="Step 4 not implemented yet — Wave 3+")
def test_noncontiguous_char_no(client, admin_user, shop_config):
    """SETUP-04: Non-contiguous char_no values are accepted without normalization.

    Steps:
    - Construct an xlsx with char_no values [1, 5, 99, 200] (non-contiguous).
    - POST it to /setup/step4/upload.
    - Assert all four char_no values appear in the response (accepted as-is).
    """
    raise NotImplementedError("Step 4 not implemented yet")
