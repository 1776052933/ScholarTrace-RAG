"""Persist conversations and ordered, recoverable exchanges."""
from alembic import op
import sqlalchemy as sa

revision = "e713ab60d425"
down_revision = "bd7492a4c18e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("paper_ids", sa.JSON(), nullable=False),
        sa.Column("scope_key", sa.String(64), nullable=False),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("active_turn_id", sa.String(32), nullable=True),
        sa.Column("active_since", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])
    op.create_index("ix_conversations_scope_key", "conversations", ["scope_key"])
    op.create_table(
        "conversation_turns",
        sa.Column("turn_id", sa.String(32), primary_key=True),
        sa.Column("conversation_id", sa.String(32), sa.ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("retrieval_question", sa.Text(), nullable=True),
        sa.Column("answer", sa.JSON(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("conversation_id", "turn_number", name="uq_conversation_turn_number"),
    )


def downgrade() -> None:
    op.drop_table("conversation_turns")
    op.drop_index("ix_conversations_updated_at", table_name="conversations")
    op.drop_index("ix_conversations_scope_key", table_name="conversations")
    op.drop_table("conversations")
