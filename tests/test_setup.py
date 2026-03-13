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


def test_setup_step1_to_step2(client_setup_incomplete, shop_config):
    """Full step 1 → step 2 → completion flow redirects to /login."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ShopConfig, User

    # Step 1: post shop name
    r1 = client_setup_incomplete.post(
        "/setup/step1",
        data={"shop_name": "Test Shop"},
        follow_redirects=False,
    )
    assert r1.status_code == 302
    assert "/setup/step2" in r1.headers["location"]

    # Seed wizard_step=1 so step2 allows access
    # (the test client already committed wizard_step=1 via step1_post)

    # Step 2: set admin password — should complete setup and redirect to /login
    r2 = client_setup_incomplete.post(
        "/setup/step2",
        data={"new_password": "secure1234", "confirm_password": "secure1234"},
        follow_redirects=False,
    )
    assert r2.status_code == 302
    assert r2.headers["location"] == "/login"


def test_setup_step2_creates_admin(client_setup_incomplete, db_engine):
    """Step 2 with no existing admin creates an admin user and sets setup_complete=True."""
    from sqlalchemy.orm import sessionmaker
    from shop.models import ShopConfig, User

    Session = sessionmaker(bind=db_engine)

    # Seed wizard_step=1 (step1 already done)
    db = Session()
    try:
        db.add(ShopConfig(id=1, setup_complete=False, wizard_step=1))
        db.commit()
    finally:
        db.close()

    response = client_setup_incomplete.post(
        "/setup/step2",
        data={"new_password": "secure1234", "confirm_password": "secure1234"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/login"

    # Verify DB state
    db = Session()
    try:
        config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
        assert config.setup_complete is True
        admin = db.query(User).filter(User.role == "admin").first()
        assert admin is not None
        assert admin.email == "admin@shop.local"
    finally:
        db.close()


def test_setup_step3_redirects_to_login_when_complete(client, shop_config):
    """GET /setup/step3 redirects to /login when setup is already complete."""
    response = client.get("/setup/step3", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_setup_step3_redirects_to_setup_when_incomplete(client_setup_incomplete, shop_config):
    """GET /setup/step3 redirects to /setup/ when setup is not complete."""
    response = client_setup_incomplete.get("/setup/step3", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/setup/"


def test_removed_step4_redirects_when_complete(client, shop_config):
    """GET /setup/step4/upload redirects to /login when setup is already complete."""
    response = client.get("/setup/step4/upload", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_removed_step4_redirects_when_incomplete(client_setup_incomplete, shop_config):
    """GET /setup/step4/upload redirects to /setup/ when setup is not complete."""
    response = client_setup_incomplete.get("/setup/step4/upload", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/setup/"


def test_setup_root_redirect(client_setup_incomplete, shop_config):
    """GET /setup/ redirects to /setup/step{wizard_step+1}."""
    response = client_setup_incomplete.get("/setup/", follow_redirects=False)
    assert response.status_code == 302
    assert "/setup/step1" in response.headers["location"]
