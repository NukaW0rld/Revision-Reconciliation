import shutil
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory


ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(db_url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def _alembic_head(cfg: AlembicConfig) -> str:
    return ScriptDirectory.from_config(cfg).get_current_head()


def test_baseline_upgrade_handles_repo_db_state(tmp_path, monkeypatch):
    src_db = ROOT / "data" / "shop.db"
    db_path = tmp_path / "legacy.db"
    shutil.copy(src_db, db_path)
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_config(db_url)
    monkeypatch.setenv("DATABASE_URL", db_url)

    con = sqlite3.connect(db_path)
    try:
        before_rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        con.close()

    assert before_rows  # shop.db has a known migration state; upgrade path is what's tested below

    command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        version_rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
        user_table_exists = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()[0]
    finally:
        con.close()

    assert version_rows == [(_alembic_head(cfg),)]
    assert user_table_exists == 1
