"""Add M4 job kind and multi-paper answer records.

Revision ID: bd7492a4c18e
Revises: 9f1a43cd621e
"""
from alembic import op
import sqlalchemy as sa

revision = "bd7492a4c18e"
down_revision = "9f1a43cd621e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_jobs", sa.Column("job_type", sa.String(24), nullable=False, server_default="research"))
    op.create_index("ix_research_jobs_job_type", "research_jobs", ["job_type"])
    op.create_table(
        "multi_paper_answers",
        sa.Column("answer_id", sa.String(32), primary_key=True),
        sa.Column("paper_ids", sa.JSON(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("multi_paper_answers")
    op.drop_index("ix_research_jobs_job_type", table_name="research_jobs")
    op.drop_column("research_jobs", "job_type")
