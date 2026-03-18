import os
from sqlalchemy import create_engine, text, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker, DeclarativeBase

SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./shop.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def run_schema_migrations(eng):
    """Add new nullable columns to existing SQLite tables if not present.

    Called at app startup after Base.metadata.create_all(). Safe to re-run —
    no-ops if columns already exist. SQLite ALTER TABLE only supports adding
    nullable or DEFAULT-valued columns.
    """
    with eng.connect() as conn:
        inspector = sa_inspect(eng)
        run_cols = {c["name"] for c in inspector.get_columns("runs")}
        if "parent_run_id" not in run_cols:
            conn.execute(text(
                "ALTER TABLE runs ADD COLUMN parent_run_id INTEGER REFERENCES runs(id)"
            ))
        if "packet_versions" not in run_cols:
            conn.execute(text("ALTER TABLE runs ADD COLUMN packet_versions JSON"))
        cfg_cols = {c["name"] for c in inspector.get_columns("shop_config")}
        if "retention_days" not in cfg_cols:
            conn.execute(text(
                "ALTER TABLE shop_config ADD COLUMN retention_days INTEGER DEFAULT 30"
            ))
        # M004: rename users.email -> users.username for arbitrary username support.
        # SQLite does not support RENAME COLUMN until 3.25.0 (2018-09-15).
        # For existing deployments: if the 'email' column exists but 'username' does not,
        # copy data via a new column. Both paths handled here.
        user_cols = {c["name"] for c in inspector.get_columns("users")}
        if "username" not in user_cols and "email" in user_cols:
            # Add username column, copy email data, then we leave email in place
            # (SQLite cannot drop columns < 3.35). The ORM maps 'username' from now on.
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN username TEXT"
            ))
            conn.execute(text("UPDATE users SET username = email"))
            conn.commit()
            # Re-build unique index on username (can't add UNIQUE via ALTER TABLE in old SQLite;
            # ORM will enforce this in application logic via IntegrityError on insert)
        conn.commit()
