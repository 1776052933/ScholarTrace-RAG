"""Add paper collections and membership."""
from alembic import op
import sqlalchemy as sa

revision = "4ab6c912f30d"
down_revision = "e713ab60d425"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_collections",
        sa.Column("collection_id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "paper_collection_items",
        sa.Column("collection_id", sa.String(32), sa.ForeignKey("paper_collections.collection_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("paper_id", sa.String(255), sa.ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("added_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_paper_collection_items_paper_id", "paper_collection_items", ["paper_id"])


def downgrade() -> None:
    op.drop_index("ix_paper_collection_items_paper_id", table_name="paper_collection_items")
    op.drop_table("paper_collection_items")
    op.drop_table("paper_collections")
