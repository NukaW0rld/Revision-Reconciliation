import pytest  # noqa: E402
from sqlalchemy import create_engine, StaticPool  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# These imports will fail until shop/ package is created in Plan 02 (Wave 1).
# That is expected — this is a scaffold. conftest.py is not loaded at collection
# time if no test in the session needs it.
from shop.app import create_app  # noqa: E402
from shop.database import Base  # noqa: E402
from shop.dependencies import get_db  # noqa: E402
from shop.services.auth import hash_password  # noqa: E402
from shop.models import User, ShopConfig, Run, RunAlert  # noqa: E402


@pytest.fixture(scope="function")
def db_engine():
    """Create an in-memory SQLite engine with all tables; tear down after each test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def client(db_engine):
    """Return a TestClient with the FastAPI app wired to the in-memory DB.

    Seeds setup_complete=True so SetupGuardMiddleware does not block route tests.
    Passes TestingSessionLocal to create_app so the middleware uses the test DB.
    Route tests for /setup/... use the `client_setup_incomplete` fixture instead.
    """
    TestingSessionLocal = sessionmaker(bind=db_engine)

    # Seed setup_complete=True so middleware does not redirect during auth/admin tests
    db = TestingSessionLocal()
    try:
        db.add(ShopConfig(setup_complete=True, wizard_step=0))
        db.commit()
    finally:
        db.close()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app(session_factory=TestingSessionLocal)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="function")
def client_setup_incomplete(db_engine):
    """Return a TestClient where setup_complete=False (tests setup wizard middleware)."""
    TestingSessionLocal = sessionmaker(bind=db_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app(session_factory=TestingSessionLocal)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="function")
def admin_user(db_engine):
    """Seed an active admin user into the in-memory DB; return the User object."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        user = User(
            username="admin",
            hashed_password=hash_password("changeme"),
            role="admin",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


@pytest.fixture(scope="function")
def shop_config(db_engine):
    """Seed a ShopConfig row with setup_complete=False, wizard_step=0."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        config = ShopConfig(setup_complete=False, wizard_step=0)
        db.add(config)
        db.commit()
        db.refresh(config)
        return config
    finally:
        db.close()


@pytest.fixture(scope="function")
def engineer_user(db_engine):
    """Seed an active engineer user into the in-memory DB; return the User object."""
    Session = sessionmaker(bind=db_engine)
    db = Session()
    try:
        user = User(
            username="engineer",
            hashed_password=hash_password("changeme"),
            role="engineer",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


@pytest.fixture(scope="function")
def huey_immediate():
    """Put Huey in immediate (synchronous) mode for test isolation."""
    from shop.tasks import huey
    huey.immediate = True
    yield
    huey.immediate = False
    huey.storage.flush_all()
