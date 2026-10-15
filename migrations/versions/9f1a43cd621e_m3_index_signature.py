"""Track which vector index actually contains each paper.

Revision ID: 9f1a43cd621e
Revises: 2c63dc4ac6b4
"""
from alembic import op
import sqlalchemy as sa

revision = "9f1a43cd621e"
down_revision = "2c63dc4ac6b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("index_signature", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "index_signature")
