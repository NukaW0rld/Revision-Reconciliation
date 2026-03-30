"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_EXPECTED_TABLES = {
    "users",
    "sessions",
    "shop_config",
    "runs",
    "run_alerts",
    "review_items",
}


def _existing_tables() -> set[str]:
    bind = op.get_bind()
    return set(inspect(bind).get_table_names())


def _adopt_existing_schema_if_present() -> bool:
    """Return True when a legacy pre-Alembic DB already has the full schema.

    In that case, this baseline migration becomes a no-op so Alembic can record
    revision ``0001`` instead of trying to recreate tables that already exist.
    If only part of the schema exists, fail loudly rather than guessing.
    """
    existing = _existing_tables()
    app_tables = existing & _EXPECTED_TABLES

    if not app_tables:
        return False

    missing = _EXPECTED_TABLES - existing
    if missing:
        missing_list = ", ".join(sorted(missing))
        present_list = ", ".join(sorted(app_tables))
        raise RuntimeError(
            "Refusing to auto-adopt partially initialized legacy schema. "
            f"Present tables: {present_list}. Missing tables: {missing_list}."
        )

    return True


def upgrade() -> None:
    if _adopt_existing_schema_if_present():
        return

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    # --- sessions ---
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])

    # --- shop_config ---
    op.create_table(
        "shop_config",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("shop_name", sa.String(), nullable=False, server_default=""),
        sa.Column("setup_complete", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("column_mapping", sa.JSON(), nullable=True),
        sa.Column("wizard_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retention_days", sa.Integer(), nullable=True, server_default="30"),
    )

    # --- runs ---
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("part_number", sa.String(), nullable=False),
        sa.Column("rev_a_label", sa.String(), nullable=False),
        sa.Column("rev_b_label", sa.String(), nullable=False),
        sa.Column("customer", sa.String(), nullable=False),
        sa.Column("job_number", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("current_stage", sa.String(), nullable=True),
        sa.Column("current_stage_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_stage", sa.String(), nullable=True),
        sa.Column("failure_message", sa.String(), nullable=True),
        sa.Column("warning_type", sa.String(), nullable=True),
        sa.Column("confidence_summary", sa.JSON(), nullable=True),
        sa.Column("output_dir", sa.String(), nullable=True),
        sa.Column("revA_path", sa.String(), nullable=False),
        sa.Column("revB_path", sa.String(), nullable=False),
        sa.Column("form3_path", sa.String(), nullable=False),
        sa.Column("revA_page", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revB_page", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("submitted_at", sa.DateTime(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("signed_at", sa.DateTime(), nullable=True),
        sa.Column("signed_by_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("parent_run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=True),
        sa.Column("packet_versions", sa.JSON(), nullable=True),
    )

    # --- run_alerts ---
    op.create_table(
        "run_alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("failure_stage", sa.String(), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    # --- review_items ---
    op.create_table(
        "review_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("char_no", sa.Integer(), nullable=True),
        sa.Column("pipeline_classification", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("requirement_revA", sa.String(), nullable=True),
        sa.Column("requirement_revB", sa.String(), nullable=True),
        sa.Column("revA_snippet_path", sa.String(), nullable=True),
        sa.Column("revB_snippet_path", sa.String(), nullable=True),
        sa.Column("revA_bbox", sa.JSON(), nullable=True),
        sa.Column("revB_bbox", sa.JSON(), nullable=True),
        sa.Column("reviewer_decision", sa.String(), nullable=True),
        sa.Column("override_classification", sa.String(), nullable=True),
        sa.Column("override_note", sa.String(), nullable=True),
        sa.Column("reviewed_by_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_review_items_run_id", "review_items", ["run_id"])


def downgrade() -> None:
    op.drop_table("review_items")
    op.drop_table("run_alerts")
    op.drop_table("runs")
    op.drop_table("shop_config")
    op.drop_table("sessions")
    op.drop_table("users")
