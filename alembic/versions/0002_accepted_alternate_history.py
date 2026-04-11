"""accepted alternate history

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-11
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "accepted_alternate_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("review_item_id", sa.Integer(), sa.ForeignKey("review_items.id"), nullable=False),
        sa.Column("reviewed_by_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("part_number", sa.String(), nullable=False),
        sa.Column("char_no", sa.Integer(), nullable=True),
        sa.Column("matched_truth_char_no", sa.String(), nullable=True),
        sa.Column("reviewed_classification", sa.String(), nullable=False),
        sa.Column("reviewed_requirement_revB", sa.String(), nullable=True),
        sa.Column("mismatch_codes", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_accepted_alternate_history_part_truth_active",
        "accepted_alternate_history",
        ["part_number", "matched_truth_char_no", "is_active"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_accepted_alternate_history_part_truth_active",
        table_name="accepted_alternate_history",
    )
    op.drop_table("accepted_alternate_history")
