import os
from pathlib import Path
from huey import SqliteHuey

_default_huey_db = os.environ.get("HUEY_DB", "/app/data/huey.db")

# When running outside Docker (e.g., dev or tests), /app/data may not exist.
# Fall back to a local path that can always be created.
if not Path(_default_huey_db).parent.exists():
    _default_huey_db = str(Path(__file__).parent.parent / "huey.db")

HUEY_DB = _default_huey_db
huey = SqliteHuey(filename=HUEY_DB)


@huey.task()
def run_pipeline_task(
    run_id: int,
    revA_path: str,
    revB_path: str,
    form3_path: str,
    part_name: str,
    dpi: int = 300,
):
    """
    Huey task: wraps delta_preservation.cli.run_pipeline().
    Updates Run.current_stage and Run.status in the shop DB at each stage.
    Full implementation added in Plan 03.
    """
    from shop.database import SessionLocal
    from shop.models import Run

    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run is None:
            return
        run.status = "running"
        db.commit()
        # Plan 03 replaces this stub with real pipeline orchestration
        run.status = "completed"
        db.commit()
    finally:
        db.close()
