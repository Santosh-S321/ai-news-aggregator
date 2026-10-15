"""track which embedding model produced each vector

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("embedding_model", sa.String(100)))
    # Any vectors created before this column existed came from the OpenAI default
    op.execute(
        "UPDATE articles SET embedding_model = 'text-embedding-3-small' "
        "WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("articles", "embedding_model")