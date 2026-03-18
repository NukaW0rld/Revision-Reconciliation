"""
TDD tests for run_pipeline_task.call_local() in shop/tasks.py.

RED phase: tests verify the full task implementation including:
- Stage progress updates in DB
- Rev A balloon failure sets status=failed, failure_stage, failure_message
- Low confidence alignment sets status=warning, warning_type=low_confidence
- Exception handling sets status=failed, creates RunAlert
- Success path sets status=completed, output_dir

These are the implementations for PIPE-01 through PIPE-06, PIPE-11.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker

from shop.database import Base
from shop.models import Run, RunAlert, User


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def SessionLocal(db_engine):
    """Return a factory that creates sessions bound to the test engine."""
    return sessionmaker(bind=db_engine)


@pytest.fixture
def reviewer(db):
    user = User(
        username="reviewer",
        hashed_password="x",
        role="engineer",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def run_record(db, reviewer, tmp_path):
    """Seed a Run record in 'queued' status."""
    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    run = Run(
        part_number="PN-TEST",
        rev_a_label="A",
        rev_b_label="B",
        customer="Test Corp",
        job_number="J-001",
        status="queued",
        revA_path=str(revA),
        revB_path=str(revB),
        form3_path=str(form3),
        reviewer_id=reviewer.id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@pytest.fixture
def run_record_no_reviewer(db, tmp_path):
    """Seed a Run record with no reviewer (reviewer_id=None)."""
    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    run = Run(
        part_number="PN-NOREV",
        rev_a_label="A",
        rev_b_label="B",
        customer="Corp",
        job_number="J-002",
        status="queued",
        revA_path=str(revA),
        revB_path=str(revB),
        form3_path=str(form3),
        reviewer_id=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _make_delta_packet(items=None):
    """Return a minimal delta packet dict."""
    return {
        "run_id": "test-run",
        "inputs": {},
        "items": items or [],
    }


class TestRunPipelineTaskSuccessPath:
    """PIPE-01, PIPE-02, PIPE-03: Success path — status lifecycle and stage updates."""

    def test_task_sets_status_completed_on_success(self, db, SessionLocal, run_record, tmp_path):
        """PIPE-03: status transitions to completed after successful pipeline run."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        # Create a minimal delta packet
        import json
        packet = _make_delta_packet(items=[
            {"scores": {"location": 0.9}},
            {"scores": {"location": 0.8}},
        ])
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        stages_seen = []

        def fake_run_pipeline(**kwargs):
            cb = kwargs.get("stage_callback")
            if cb:
                for i, name in enumerate([
                    "Form 3 parsing", "Balloon detection", "Text extraction",
                    "Anchor building", "Alignment", "Candidate matching",
                    "Classification", "Output",
                ]):
                    cb(i, name)
                    stages_seen.append((i, name))
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        run = db.query(Run).filter(Run.id == run_record.id).first()
        assert run.status == "completed"
        assert run.output_dir is not None
        assert "test_run" in run.output_dir

    def test_task_updates_stage_in_db(self, db, SessionLocal, run_record, tmp_path):
        """PIPE-02: current_stage and current_stage_index are updated during execution."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        packet = _make_delta_packet()
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        stage_snapshots = []

        def fake_run_pipeline(**kwargs):
            cb = kwargs.get("stage_callback")
            if cb:
                for i, name in enumerate([
                    "Form 3 parsing", "Balloon detection", "Text extraction",
                    "Anchor building", "Alignment", "Candidate matching",
                    "Classification", "Output",
                ]):
                    cb(i, name)
                    # Read from DB to verify it was updated
                    fresh_db = SessionLocal()
                    try:
                        r = fresh_db.query(Run).filter(Run.id == run_record.id).first()
                        stage_snapshots.append((r.current_stage_index, r.current_stage))
                    finally:
                        fresh_db.close()
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        # Verify all 8 stages were recorded in DB
        assert len(stage_snapshots) == 8
        assert stage_snapshots[0] == (0, "Form 3 parsing")
        assert stage_snapshots[1] == (1, "Balloon detection")
        assert stage_snapshots[7] == (7, "Output")

    def test_task_sets_status_running_at_start(self, db, SessionLocal, run_record, tmp_path):
        """PIPE-03: status is set to running when task starts."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        packet = _make_delta_packet()
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        status_at_start = []

        def fake_run_pipeline(**kwargs):
            fresh_db = SessionLocal()
            try:
                r = fresh_db.query(Run).filter(Run.id == run_record.id).first()
                status_at_start.append(r.status)
            finally:
                fresh_db.close()
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        assert status_at_start == ["running"]


class TestRevABalloonFailure:
    """PIPE-04: Rev A balloon detection failure sets status=failed."""

    def test_revA_no_items_sets_failed(self, db, SessionLocal, run_record, tmp_path):
        """When pipeline returns 0 items (no balloons), status=failed."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        # Empty items = no balloons found
        packet = _make_delta_packet(items=[])
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        def fake_run_pipeline(**kwargs):
            cb = kwargs.get("stage_callback")
            if cb:
                cb(0, "Form 3 parsing")
                cb(1, "Balloon detection")
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        run = db.query(Run).filter(Run.id == run_record.id).first()
        assert run.status == "failed"
        assert run.failure_stage == "Balloon detection"
        assert run.failure_message is not None
        assert len(run.failure_message) > 0

    def test_revA_no_items_creates_alert(self, db, SessionLocal, run_record, tmp_path):
        """When Rev A balloon failure, RunAlert is created for reviewer."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        packet = _make_delta_packet(items=[])
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        def fake_run_pipeline(**kwargs):
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        alerts = db.query(RunAlert).filter(RunAlert.run_id == run_record.id).all()
        assert len(alerts) >= 1


class TestExceptionFailurePath:
    """PIPE-04 (exception path): Exception sets status=failed and creates alert."""

    def test_exception_sets_failed_status(self, db, SessionLocal, run_record, tmp_path):
        """When pipeline raises, status=failed with failure_message."""
        from shop.tasks import run_pipeline_task

        def fake_run_pipeline(**kwargs):
            raise RuntimeError("Pipeline exploded")

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", "/tmp/out"):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        run = db.query(Run).filter(Run.id == run_record.id).first()
        assert run.status == "failed"
        assert "Pipeline exploded" in run.failure_message

    def test_exception_creates_alert(self, db, SessionLocal, run_record, tmp_path):
        """When pipeline raises and reviewer_id is set, RunAlert is created."""
        from shop.tasks import run_pipeline_task

        def fake_run_pipeline(**kwargs):
            raise RuntimeError("stage failed")

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", "/tmp/out"):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        alerts = db.query(RunAlert).filter(RunAlert.run_id == run_record.id).all()
        assert len(alerts) == 1
        assert alerts[0].user_id == run_record.reviewer_id
        assert run_record.part_number in alerts[0].message

    def test_no_alert_when_no_reviewer(self, db, SessionLocal, run_record_no_reviewer, tmp_path):
        """When reviewer_id is None, no RunAlert is created on failure."""
        from shop.tasks import run_pipeline_task

        def fake_run_pipeline(**kwargs):
            raise RuntimeError("oops")

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", "/tmp/out"):
            run_pipeline_task.call_local(
                run_id=run_record_no_reviewer.id,
                revA_path=run_record_no_reviewer.revA_path,
                revB_path=run_record_no_reviewer.revB_path,
                form3_path=run_record_no_reviewer.form3_path,
                part_name=run_record_no_reviewer.part_number,
            )

        db.expire_all()
        alerts = db.query(RunAlert).filter(RunAlert.run_id == run_record_no_reviewer.id).all()
        assert len(alerts) == 0


class TestLowConfidenceWarning:
    """PIPE-06: Low alignment confidence sets status=warning."""

    def test_majority_low_location_scores_sets_warning(self, db, SessionLocal, run_record, tmp_path):
        """When >50% of items have low location scores, status=warning, warning_type=low_confidence."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        # 4 out of 5 items have low location scores (< 0.5)
        packet = _make_delta_packet(items=[
            {"scores": {"location": 0.1}},
            {"scores": {"location": 0.2}},
            {"scores": {"location": 0.15}},
            {"scores": {"location": 0.3}},
            {"scores": {"location": 0.9}},
        ])
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        def fake_run_pipeline(**kwargs):
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        run = db.query(Run).filter(Run.id == run_record.id).first()
        assert run.status == "warning"
        assert run.warning_type == "low_confidence"
        assert run.confidence_summary is not None
        assert "low_confidence_count" in run.confidence_summary
        assert run.output_dir is not None

    def test_minority_low_scores_does_not_set_warning(self, db, SessionLocal, run_record, tmp_path):
        """When <50% of items have low location scores, status stays completed."""
        from shop.tasks import run_pipeline_task

        out_dir = tmp_path / "out"
        fake_run_dir = out_dir / "test_run"
        fake_run_dir.mkdir(parents=True)
        import json
        # Only 1 out of 5 items low confidence
        packet = _make_delta_packet(items=[
            {"scores": {"location": 0.9}},
            {"scores": {"location": 0.8}},
            {"scores": {"location": 0.7}},
            {"scores": {"location": 0.9}},
            {"scores": {"location": 0.1}},  # only one low
        ])
        (fake_run_dir / "delta_packet.json").write_text(json.dumps(packet))

        def fake_run_pipeline(**kwargs):
            return fake_run_dir

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", str(out_dir)):
            run_pipeline_task.call_local(
                run_id=run_record.id,
                revA_path=run_record.revA_path,
                revB_path=run_record.revB_path,
                form3_path=run_record.form3_path,
                part_name=run_record.part_number,
            )

        db.expire_all()
        run = db.query(Run).filter(Run.id == run_record.id).first()
        assert run.status == "completed"
        assert run.warning_type is None


class TestRunNotFound:
    """Edge case: run_id does not exist."""

    def test_nonexistent_run_returns_early(self, db, SessionLocal, tmp_path):
        """When run_id doesn't exist, task returns early without error."""
        from shop.tasks import run_pipeline_task

        def fake_run_pipeline(**kwargs):
            raise AssertionError("Should not be called")

        with patch("shop.tasks.SessionLocal", SessionLocal), \
             patch("shop.tasks.run_pipeline", fake_run_pipeline), \
             patch("shop.tasks.OUT_DIR", "/tmp/out"):
            # Should not raise
            run_pipeline_task.call_local(
                run_id=99999,
                revA_path="/tmp/a.pdf",
                revB_path="/tmp/b.pdf",
                form3_path="/tmp/f.xlsx",
                part_name="test",
            )
