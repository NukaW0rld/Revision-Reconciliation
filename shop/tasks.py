import json
import logging
import os
from pathlib import Path

from huey import SqliteHuey

logger = logging.getLogger(__name__)

_default_huey_db = os.environ.get("HUEY_DB", "/app/data/huey.db")

# When running outside Docker (e.g., dev or tests), /app/data may not exist.
# Fall back to a local path that can always be created.
if not Path(_default_huey_db).parent.exists():
    _default_huey_db = str(Path(__file__).parent.parent / "huey.db")

HUEY_DB = _default_huey_db
OUT_DIR = os.environ.get("OUT_DIR", "/app/out")

huey = SqliteHuey(filename=HUEY_DB)

# Stage name registry (0-indexed, matches cli.py stage_callback ordering)
STAGE_NAMES = [
    "Form 3 parsing",       # 0
    "Balloon detection",    # 1
    "Text extraction",      # 2
    "Anchor building",      # 3
    "Alignment",            # 4
    "Candidate matching",   # 5
    "Classification",       # 6
    "Output",               # 7
]


# ---------------------------------------------------------------------------
# Module-level references patched in tests
# ---------------------------------------------------------------------------

# These are imported lazily inside the task body to avoid circular imports
# between shop.tasks and shop.database/models at module load time.
# However, we expose them at module level so tests can patch them.

def _get_session_local():
    """Return the SessionLocal factory, imported lazily to avoid circular imports."""
    from shop.database import SessionLocal
    return SessionLocal


def _get_run_pipeline():
    """Return run_pipeline, imported lazily to avoid circular imports."""
    from delta_preservation.cli import run_pipeline
    return run_pipeline


# Module-level symbols that tests can patch with patch("shop.tasks.SessionLocal", ...)
# These are set at module level so mock.patch can find and replace them.
try:
    from shop.database import SessionLocal
except ImportError:
    SessionLocal = None  # type: ignore[assignment,misc]

try:
    from delta_preservation.cli import run_pipeline
except ImportError:
    run_pipeline = None  # type: ignore[assignment]


def _update_stage(db, run, stage_idx: int, stage_name: str) -> None:
    """Persist the current stage index and name to the Run row."""
    run.current_stage_index = stage_idx
    run.current_stage = stage_name
    db.commit()


def _create_alert(db, run) -> None:
    """Create a RunAlert for the reviewer if one is assigned."""
    from shop.models import RunAlert
    if run.reviewer_id is None:
        return
    alert = RunAlert(
        run_id=run.id,
        user_id=run.reviewer_id,
        message=(
            f"Run {run.id} ({run.part_number}) failed at stage: "
            f"{run.failure_stage or 'Unknown'}"
        ),
        failure_stage=run.failure_stage,
        is_read=False,
    )
    db.add(alert)
    db.commit()


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
    Huey task: calls delta_preservation.cli.run_pipeline() with stage_callback.

    Updates Run row at each stage; handles failure and warning states.

    Stage progress (PIPE-02):
        stage_callback updates Run.current_stage and current_stage_index at
        each of the 8 pipeline stages.

    Failure path (PIPE-04):
        - If pipeline returns 0 items (no Rev A balloons), set status=failed,
          failure_stage="Balloon detection", failure_message descriptive,
          create RunAlert.
        - If exception occurs, set status=failed, failure_stage from
          current_stage, failure_message=str(exc), create RunAlert.

    Warning path (PIPE-05 / PIPE-06):
        - Rev B balloon failure is surfaced through low-confidence alignment
          (inlier_ratio < threshold). In v1, both cases use warning_type=low_confidence.
        - If >50% of items have location score < 0.5, set status=warning,
          warning_type=low_confidence, populate confidence_summary JSON.

    Success path (PIPE-03):
        set status=completed, output_dir=str(out_dir).

    PIPE-07, PIPE-08, PIPE-09, PIPE-10:
        These requirements (unchanged/changed/removed/uncertain classification)
        are handled entirely by the existing delta_preservation/reconcile/classify.py
        logic. No additional code is required in this task.
    """
    print("HUEY TASK STARTED run_id=", run_id)
    logger.info("run_pipeline_task: starting run_id=%d", run_id)

    from shop.models import Run

    # Use module-level symbols (patchable in tests).
    # Fall back to lazy import if the module-level symbol is None (import failed at
    # load time, e.g., because delta_preservation was not on sys.path yet).
    _SessionLocal = SessionLocal if SessionLocal is not None else _get_session_local()
    _run_pipeline = run_pipeline if run_pipeline is not None else _get_run_pipeline()

    # Initialize run to None and db to None so the except/finally blocks can
    # safely check whether they were successfully created before using them.
    run = None
    db = None
    try:
        db = _SessionLocal()
        run = db.query(Run).filter(Run.id == run_id).first()
        if run is None:
            logger.error("run_pipeline_task: Run %d not found", run_id)
            return

        # PIPE-03: status transitions queued -> running -> completed
        run.status = "running"
        run.current_stage_index = 0
        run.current_stage = STAGE_NAMES[0]
        db.commit()

        def stage_callback(stage_idx: int, stage_name: str) -> None:
            # PIPE-02: update stage in DB so SSE / polling can observe progress
            _update_stage(db, run, stage_idx, stage_name)

        out_dir = _run_pipeline(
            revA_pdf=revA_path,
            revB_pdf=revB_path,
            form3_xlsx=form3_path,
            out_dir=OUT_DIR,
            dpi=dpi,
            part_name=part_name,
            stage_callback=stage_callback,
        )

        # --- Post-run checks ---

        # Check for Rev A balloon failure (PIPE-04):
        # The pipeline does not raise for empty balloons — it produces an empty
        # delta packet. We detect this by checking items count in the output.
        packet_path = Path(out_dir) / "delta_packet.json"
        items = []
        if packet_path.exists():
            with open(packet_path) as f:
                packet = json.load(f)
            items = packet.get("items", [])

        # Hard-fail: no characteristics found = Rev A balloon detection failed
        if len(items) == 0:
            run.status = "failed"
            run.failure_stage = "Balloon detection"
            run.failure_message = (
                "No characteristics found in Rev A PDF. "
                "Ensure the drawing uses numbered balloon annotations."
            )
            db.commit()
            _create_alert(db, run)
            return

        # PIPE-06: Low-confidence warning check
        # Compute the ratio of items with low location scores (< 0.5).
        # If the majority (> 50%) have low confidence, set warning state.
        # This also covers PIPE-05 (RevB balloon failure): when RevB balloons
        # are absent the alignment stage produces near-zero inlier ratios,
        # which results in low location scores for all characteristics.
        scores = [item.get("scores", {}).get("location", 1.0) for item in items]
        low_conf = sum(1 for s in scores if s < 0.5)
        low_conf_ratio = low_conf / len(scores) if scores else 0.0

        if low_conf_ratio > 0.5:
            run.status = "warning"
            run.warning_type = "low_confidence"
            run.confidence_summary = {
                "low_confidence_count": low_conf,
                "total": len(scores),
                "low_confidence_ratio": round(low_conf_ratio, 2),
                "message": (
                    f"{int(low_conf_ratio * 100)}% of characteristics have "
                    "low location confidence in Rev B"
                ),
            }
            run.output_dir = str(out_dir)
            db.commit()
            return

        # PIPE-03: Success path
        run.status = "completed"
        run.output_dir = str(out_dir)
        db.commit()

    except Exception as exc:
        logger.exception("run_pipeline_task failed for run %d", run_id)
        print("HUEY TASK FAILED run_id=", run_id, "exc=", exc)
        if run is not None and db is not None:
            try:
                run.status = "failed"
                run.failure_stage = run.current_stage or "Unknown"
                run.failure_message = str(exc)
                db.commit()
                _create_alert(db, run)
            except Exception:
                logger.exception(
                    "run_pipeline_task: failed to persist failure state for run %d", run_id
                )
    finally:
        if db is not None:
            db.close()
